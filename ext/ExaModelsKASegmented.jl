# Segmented reductions and the fused evaluation of augmented constraint rows.
# Included in ExaModelsKernelAbstractions.

# ---- Segmented reductions -------------------------------------------------------
#
# Several callbacks reduce a list of values sorted by output index: the objective
# gradient and the products J*v, J'*v and H*v. Reducing each output entry in one work
# item leaves a device mostly idle when there are few outputs with long segments, and
# serializes each long segment. Instead, every segment is cut into chunks of at most
# `SEGCHUNK` entries: one work item sums a chunk into a partial, and one work item per
# output entry adds that entry's partials. The chunk table fixes the order of the
# summation, so results are the same from run to run.

const SEGCHUNK = 256

# Chunk table for the segments given by `ptr` (segment s is ptr[s]:ptr[s+1]-1):
# chunk c is cstart[c]:cstart[c+1]-1, and segment s has chunks kptr[s]:kptr[s+1]-1.
# Built on the host, once, when the model is built.
function _seg_chunks(ptr::Vector{Int})
    cstart = Int[]
    kptr = Int[1]
    for s in 1:(length(ptr) - 1)
        for st in ptr[s]:SEGCHUNK:(ptr[s + 1] - 1)
            push!(cstart, st)
        end
        push!(kptr, length(cstart) + 1)
    end
    push!(cstart, isempty(ptr) ? 1 : ptr[end])
    return cstart, kptr
end

# A planned reduction over a sorted coordinate list; `partial` holds the chunk sums.
struct SegPlan{VI, VT}
    cstart::VI
    kptr::VI
    partial::VT
end
function _seg_plan(backend, x0, ptr)
    cstart, kptr = _seg_chunks(Array(ptr))
    return SegPlan(
        ExaModels.convert_array(cstart, backend),
        ExaModels.convert_array(kptr, backend),
        similar(x0, max(length(cstart) - 1, 0)),
    )
end
_nseg(p::SegPlan) = length(p.kptr) - 1

# The terms reduced. Coordinates are `(key, index)` for a gather and `((i, j), index)`
# for a sparse product.
struct _Gather end                          # y[key] += V[index]
@inline _segkey(::_Gather, c) = c[1]
@inline _segterm(::_Gather, c, V, x) = @inbounds V[c[2]]
struct _RowSpmv end                         # y[i] += V * x[j]
@inline _segkey(::_RowSpmv, c) = c[1][1]
@inline _segterm(::_RowSpmv, c, V, x) = @inbounds V[c[2]] * x[c[1][2]]
struct _ColSpmv end                         # y[j] += V * x[i]
@inline _segkey(::_ColSpmv, c) = c[1][2]
@inline _segterm(::_ColSpmv, c, V, x) = @inbounds V[c[2]] * x[c[1][1]]
struct _ColSySpmv end                       # y[j] += V * x[i], off-diagonal entries only
@inline _segkey(::_ColSySpmv, c) = c[1][2]
@inline _segterm(::_ColSySpmv, c, V, x) =
    @inbounds c[1][1] != c[1][2] ? V[c[2]] * x[c[1][1]] : zero(eltype(V))

@kernel function seg_partial!(partial, @Const(op), @Const(coord), @Const(V), @Const(x), @Const(cstart))
    I = @index(Global)
    @inbounds begin
        s = zero(eltype(partial))
        for l in cstart[I]:(cstart[I + 1] - 1)
            s += _segterm(op, coord[l], V, x)
        end
        partial[I] = s
    end
end
@kernel function seg_finish!(y, @Const(op), @Const(partial), @Const(coord), @Const(cstart), @Const(kptr))
    I = @index(Global)
    @inbounds begin
        s = zero(eltype(y))
        for c in kptr[I]:(kptr[I + 1] - 1)
            s += partial[c]
        end
        y[_segkey(op, coord[cstart[kptr[I]]])] += s
    end
end

# y[key] += the sum of the terms of each segment of the sorted list `coord`
function _seg_reduce!(backend, y, op, plan::SegPlan, coord, V, x)
    n = _nseg(plan)
    n == 0 && return y
    seg_partial!(backend)(plan.partial, op, coord, V, x, plan.cstart; ndrange = length(plan.partial))
    seg_finish!(backend)(y, op, plan.partial, coord, plan.cstart, plan.kptr; ndrange = n)
    return y
end

# The same reduction where output entry s is the s-th segment itself (V[s] += its
# terms), as in coalescing duplicate coordinates.
@kernel function seg_finish_index!(y, @Const(partial), @Const(kptr))
    I = @index(Global)
    @inbounds begin
        s = zero(eltype(y))
        for c in kptr[I]:(kptr[I + 1] - 1)
            s += partial[c]
        end
        y[I] += s
    end
end
function _seg_reduce_index!(backend, y, plan::SegPlan, coord, V)
    n = _nseg(plan)
    n == 0 && return y
    seg_partial!(backend)(plan.partial, _Gather(), coord, V, nothing, plan.cstart; ndrange = length(plan.partial))
    seg_finish_index!(backend)(y, plan.partial, plan.kptr; ndrange = n)
    return y
end

# ---- Fused evaluation of augmented constraint rows -------------------------------
#
# The terms that `add_con!` adds to constraint rows were evaluated into a buffer and
# then summed per row, one work item per row. With few rows and many terms per row
# (a quadratic constraint has one term per nonzero of its matrix) that loop dominated
# `cons!`. Now each augmentation block keeps the permutation that orders its terms by
# target row (its data stays where it is, so Jacobian and Hessian coordinates keep their
# order) and a chunk table whose chunks never cross a row: one work item evaluates the
# terms of a chunk and writes their sum, and the chunk sums of all blocks are added per
# row.

struct AugBlockPlan{VI}
    perm::VI        # the block's terms in row order (stable: a row keeps its terms in order)
    cstart::VI      # chunk c covers the terms perm[cstart[c]:cstart[c+1]-1]
    poff::Int       # offset of this block's chunks in the partial buffer
end
struct AugPlan{B, VI, VT}
    blocks::B       # per block of `cons`: an AugBlockPlan for an augmentation, else nothing
    partial::VT     # chunk sums of all blocks
    perm::VI        # chunk indices sorted by row
    ptr::VI         # the chunks of the r-th augmented row are perm[ptr[r]:ptr[r+1]-1]
    rows::VI        # the augmented rows
end

# `rows_host[oa + I]` is the target row of term I of the augmentation block at `oa`.
function _aug_plan(backend, x0, cons, rows_host::Vector{Int})
    chunk_rows = Int[]
    plan_of(con) = _block_plan(con, backend, rows_host, chunk_rows)
    blocks = cons isa _LongTuple ? Any[plan_of(con) for con in Any[cons...]] : map(plan_of, cons)
    perm = sortperm(chunk_rows)
    sorted_rows = chunk_rows[perm]
    ptr = isempty(sorted_rows) ? [1] : ExaModels.getptr(nothing, sorted_rows)
    rows = sorted_rows[ptr[1:(end - 1)]]
    return AugPlan(
        blocks,
        similar(x0, length(chunk_rows)),
        ExaModels.convert_array(perm, backend),
        ExaModels.convert_array(ptr, backend),
        ExaModels.convert_array(rows, backend),
    )
end
_block_plan(con, backend, rows_host, chunk_rows) = nothing
function _block_plan(con::ExaModels.ConstraintAugmentation, backend, rows_host, chunk_rows)
    n = length(con.itr)
    r = rows_host[(con.oa + 1):(con.oa + n)]
    perm = sortperm(r)
    r = r[perm]
    cstart = Int[]
    i = 1
    while i <= n
        j = i
        while j < n && r[j + 1] == r[i] && j - i + 1 < SEGCHUNK
            j += 1
        end
        push!(cstart, i)
        push!(chunk_rows, r[i])
        i = j + 1
    end
    poff = length(chunk_rows) - length(cstart)
    push!(cstart, n + 1)
    return AugBlockPlan(ExaModels.convert_array(perm, backend), ExaModels.convert_array(cstart, backend), poff)
end

@kernel function aug_chunk_sum!(partial, @Const(f), @Const(itr), @Const(perm), @Const(cstart), @Const(x), @Const(θ), @Const(poff))
    I = @index(Global)
    @inbounds begin
        s = zero(eltype(partial))
        for t in cstart[I]:(cstart[I + 1] - 1)
            s += f(itr[perm[t]], x, θ)
        end
        partial[poff + I] = s
    end
end
@kernel function aug_finish!(y, @Const(partial), @Const(perm), @Const(ptr), @Const(rows))
    I = @index(Global)
    @inbounds begin
        s = zero(eltype(y))
        for k in ptr[I]:(ptr[I + 1] - 1)
            s += partial[perm[k]]
        end
        y[rows[I]] += s
    end
end

_aug_chunks!(backend, partial, ::Nothing, con, x, θ) = nothing
function _aug_chunks!(backend, partial, b::AugBlockPlan, con, x, θ)
    isempty(b.perm) && return nothing
    aug_chunk_sum!(backend)(partial, con.f, con.itr, b.perm, b.cstart, x, θ, b.poff; ndrange = length(b.cstart) - 1)
    return nothing
end
_aug_cons!(backend, partial, ::Tuple{}, ::Tuple{}, x, θ) = nothing
function _aug_cons!(backend, partial, (b, bs...), (con, cons...), x, θ)
    _aug_chunks!(backend, partial, b, con, x, θ)
    return _aug_cons!(backend, partial, bs, cons, x, θ)
end
function _aug_cons!(backend, partial, bs::Vector{Any}, cons::_LongTuple, x, θ)
    blocks = Any[cons...]
    for i in length(blocks):-1:1
        _aug_chunks!(backend, partial, bs[i], blocks[i], x, θ)
    end
    return nothing
end

# y[row] += the sum of the augmentation terms of each row
function _conaugs_fused!(backend, y, plan::AugPlan, cons, x, θ)
    isempty(plan.rows) && return y
    _aug_cons!(backend, plan.partial, plan.blocks, cons, x, θ)
    aug_finish!(backend)(y, plan.partial, plan.perm, plan.ptr, plan.rows; ndrange = length(plan.rows))
    return y
end

# ---- Fused Jacobian-vector product -------------------------------------------------
#
# J*v without forming the Jacobian: a term's contribution to its row is its directional
# derivative ∇termᵀv, read off the reverse-mode graph (the same local partials `grpass`
# uses) without writing any coordinate. A constraint row writes its value directly;
# augmentation terms are reduced per row with the chunk plan of `cons!`. The first
# `jprod` compiles one kernel per constraint block (`kerjv` or `aug_chunk_jv!`) in
# addition to the `kerj` kernels that `jac_coord!` and `jtprod` use, so its first-call
# latency grows with the number of block types.

@inline _gdot(::Union{ExaModels.AdjointNull, ExaModels.DataIndexed, Real}, v, adj) = zero(adj)
@inline _gdot(d::ExaModels.AdjointNode1, v, adj) = _gdot(d.inner, v, ExaModels._amul(adj, d.y))
@inline _gdot(d::ExaModels.AdjointNode2, v, adj) =
    _gdot(d.inner1, v, ExaModels._amul(adj, d.y1)) + _gdot(d.inner2, v, ExaModels._amul(adj, d.y2))
@inline _gdot(d::ExaModels.AdjointNodeVar, v, adj) = @inbounds adj * v[d.i]

@kernel function kerjv(Jv, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(v))
    I = @index(Global)
    @inbounds Jv[ExaModels.offset0(f, itr, I)] =
        _gdot(f(itr[I], ExaModels.AdjointNodeSource(x), θ), v, one(eltype(Jv)))
end
@kernel function aug_chunk_jv!(partial, @Const(f), @Const(itr), @Const(perm), @Const(cstart), @Const(x), @Const(θ), @Const(v), @Const(poff))
    I = @index(Global)
    @inbounds begin
        s = zero(eltype(partial))
        for t in cstart[I]:(cstart[I + 1] - 1)
            s += _gdot(f(itr[perm[t]], ExaModels.AdjointNodeSource(x), θ), v, one(eltype(partial)))
        end
        partial[poff + I] = s
    end
end

_jv_block!(backend, Jv, con, x, θ, v) = nothing
function _jv_block!(backend, Jv, con::ExaModels.Constraint, x, θ, v)
    isempty(con.itr) && return nothing
    kerjv(backend)(Jv, con.f, con.itr, x, θ, v; ndrange = length(con.itr))
    return nothing
end
_jv_rows!(backend, Jv, ::Tuple{}, x, θ, v) = nothing
function _jv_rows!(backend, Jv, (con, cons...), x, θ, v)
    _jv_rows!(backend, Jv, cons, x, θ, v)
    return _jv_block!(backend, Jv, con, x, θ, v)
end
function _jv_rows!(backend, Jv, cons::_LongTuple, x, θ, v)
    blocks = Any[cons...]
    for i in length(blocks):-1:1
        _jv_block!(backend, Jv, blocks[i], x, θ, v)
    end
    return nothing
end

_aug_jv_chunks!(backend, partial, ::Nothing, con, x, θ, v) = nothing
function _aug_jv_chunks!(backend, partial, b::AugBlockPlan, con, x, θ, v)
    isempty(b.perm) && return nothing
    aug_chunk_jv!(backend)(partial, con.f, con.itr, b.perm, b.cstart, x, θ, v, b.poff; ndrange = length(b.cstart) - 1)
    return nothing
end
_aug_jv!(backend, partial, ::Tuple{}, ::Tuple{}, x, θ, v) = nothing
function _aug_jv!(backend, partial, (b, bs...), (con, cons...), x, θ, v)
    _aug_jv_chunks!(backend, partial, b, con, x, θ, v)
    return _aug_jv!(backend, partial, bs, cons, x, θ, v)
end
function _aug_jv!(backend, partial, bs::Vector{Any}, cons::_LongTuple, x, θ, v)
    blocks = Any[cons...]
    for i in length(blocks):-1:1
        _aug_jv_chunks!(backend, partial, bs[i], blocks[i], x, θ, v)
    end
    return nothing
end

# Jv = J(x) * v for the SIMD constraints (rows of oracles are not touched)
function _jprod_fused!(backend, Jv, plan::AugPlan, cons, x, θ, v)
    _jv_rows!(backend, Jv, cons, x, θ, v)
    if !isempty(plan.rows)
        _aug_jv!(backend, plan.partial, plan.blocks, cons, x, θ, v)
        aug_finish!(backend)(Jv, plan.partial, plan.perm, plan.ptr, plan.rows; ndrange = length(plan.rows))
    end
    return Jv
end

# ---- Fused objective ---------------------------------------------------------------
#
# The objective is the sum of all its terms. Each work item evaluates up to
# `SEGCHUNK` consecutive terms of a block and writes their sum, and the chunk sums are
# then added on the device, so no buffer with one entry per term is written.

struct ObjPlan{O, VT}
    poff::O         # per block: offset of its chunks in `partial`
    partial::VT
end
function _obj_plan(x0, objs)
    n = 0
    off(o) = (k = n; n += cld(length(o.itr), SEGCHUNK); k)
    poff = objs isa _LongTuple ? Int[off(o) for o in Any[objs...]] : map(off, objs)
    return ObjPlan(poff, similar(x0, n))
end

@kernel function obj_chunk_sum!(partial, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(n), @Const(poff))
    I = @index(Global)
    @inbounds begin
        s = zero(eltype(partial))
        for t in ((I - 1) * SEGCHUNK + 1):min(I * SEGCHUNK, n)
            s += f(itr[t], x, θ)
        end
        partial[poff + I] = s
    end
end
function _obj_block!(backend, partial, poff, o, x, θ)
    n = length(o.itr)
    n == 0 && return nothing
    obj_chunk_sum!(backend)(partial, o.f, o.itr, x, θ, n, poff; ndrange = cld(n, SEGCHUNK))
    return nothing
end
_obj_fused!(backend, partial, ::Tuple{}, ::Tuple{}, x, θ) = nothing
function _obj_fused!(backend, partial, (p, ps...), (o, os...), x, θ)
    _obj_block!(backend, partial, p, o, x, θ)
    return _obj_fused!(backend, partial, ps, os, x, θ)
end
function _obj_fused!(backend, partial, ps::Vector{Int}, os::_LongTuple, x, θ)
    blocks = Any[os...]
    for i in length(blocks):-1:1
        _obj_block!(backend, partial, ps[i], blocks[i], x, θ)
    end
    return nothing
end

# the SIMD part of the objective at `x`, by the fused plan
function _obj_sum(m, x)
    partial = m.ext.objplan.partial
    isempty(partial) && return zero(eltype(partial))
    _obj_fused!(m.ext.backend, partial, m.ext.objplan.poff, m.objs, x, m.θ)
    return ExaModels.sum(partial)
end
