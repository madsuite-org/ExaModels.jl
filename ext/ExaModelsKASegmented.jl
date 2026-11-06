# Segmented reductions and the fused evaluation of augmented constraint rows.
# Included in ExaModelsKernelAbstractions.

# ---- Segmented reductions -------------------------------------------------------
#
# Several callbacks reduce a list of values sorted by output index: the objective
# gradient, the products J*v, J'*v and H*v, and the programs of buffered subexpressions.
# Reducing each output entry in one work item leaves a device mostly idle when there
# are few outputs with long segments, and serializes each long segment. Instead, a short
# segment is summed by one work item, and a long one is cut into ranges that one
# workgroup each sums with `@groupreduce`, after which one work item per long segment
# adds its partial sums.
#
# `@groupreduce` leaves the reduction inside a workgroup to the backend, and the number
# of work items follows the number of terms, so nothing here is fitted to a device.

const GROUPSIZE = 256  # work items of a workgroup that sums terms
const SHORTROW = 128   # a segment or row up to this long does not get a workgroup

# The most workgroups of one segment: enough to fill the device, while bounding the serial
# sum of the segment's partials. `multiprocessor_count` is 0 when a backend does not know.
_maxgroups(backend) = max(1024, 16 * KernelAbstractions.KernelInterface.multiprocessor_count(backend))

# The ranges of a long segment a:b, one per workgroup.
function _group_ranges(a, b, backend)
    size = max(GROUPSIZE, cld(b - a + 1, _maxgroups(backend)))
    return ((st, min(st + size - 1, b)) for st in a:size:b)
end

# A planned reduction of the segments given by `ptr` (segment s is ptr[s]:ptr[s+1]-1).
struct SegPlan{VI, VT}
    ptr::VI
    short::VI       # the segments one work item sums
    long::VI        # the segments cut into ranges
    gstart::VI      # workgroup g sums the entries gstart[g]:gstop[g]
    gstop::VI
    kptr::VI        # the r-th long segment has the workgroups kptr[r]:kptr[r+1]-1
    partial::VT     # the sum of each workgroup
end
# `conv` moves a host vector to the device. Built once, when the model is built.
function _seg_table(conv, x0, ptr::Vector{Int})
    short, long, gstart, gstop, kptr = Int[], Int[], Int[], Int[], Int[1]
    for s in 1:(length(ptr) - 1)
        a, b = ptr[s], ptr[s + 1] - 1
        if b - a + 1 <= SHORTROW
            push!(short, s)
        else
            push!(long, s)
            for (st, stop) in _group_ranges(a, b, KernelAbstractions.get_backend(x0))
                push!(gstart, st)
                push!(gstop, stop)
            end
            push!(kptr, length(gstart) + 1)
        end
    end
    return SegPlan(conv(ptr), conv(short), conv(long), conv(gstart), conv(gstop), conv(kptr), similar(x0, length(gstart)))
end
_seg_plan(backend, x0, ptr) = _seg_table(v -> ExaModels.convert_array(v, backend), x0, Array(ptr))
_nseg(p::SegPlan) = length(p.ptr) - 1

# What is reduced: `_segterm` is the value of entry l, and `_segput!` stores the sum of
# segment s. `args` holds the arrays; here coordinates are `(key, index)` for a gather
# and `((i, j), index)` for a sparse product, with `args = (coord, V, x)`.
struct _Gather end                          # y[key] += V[index]
@inline _segterm(::_Gather, l, (coord, V, x)) = @inbounds V[coord[l][2]]
@inline _segput!(::_Gather, y, s, ptr, acc, (coord, V, x)) = @inbounds y[coord[ptr[s]][1]] += acc
struct _GatherIndex end                     # y[s] += V[index], as in coalescing duplicates
@inline _segterm(::_GatherIndex, l, (coord, V, x)) = @inbounds V[coord[l][2]]
@inline _segput!(::_GatherIndex, y, s, ptr, acc, args) = @inbounds y[s] += acc
struct _RowSpmv end                         # y[i] += V * x[j]
@inline _segterm(::_RowSpmv, l, (coord, V, x)) = @inbounds V[coord[l][2]] * x[coord[l][1][2]]
@inline _segput!(::_RowSpmv, y, s, ptr, acc, (coord, V, x)) = @inbounds y[coord[ptr[s]][1][1]] += acc
struct _ColSpmv end                         # y[j] += V * x[i]
@inline _segterm(::_ColSpmv, l, (coord, V, x)) = @inbounds V[coord[l][2]] * x[coord[l][1][1]]
@inline _segput!(::_ColSpmv, y, s, ptr, acc, (coord, V, x)) = @inbounds y[coord[ptr[s]][1][2]] += acc
struct _ColSySpmv end                       # y[j] += V * x[i], off-diagonal entries only
@inline function _segterm(::_ColSySpmv, l, (coord, V, x))
    @inbounds c = coord[l]
    return @inbounds c[1][1] != c[1][2] ? V[c[2]] * x[c[1][1]] : zero(eltype(V))
end
@inline _segput!(::_ColSySpmv, y, s, ptr, acc, (coord, V, x)) = @inbounds y[coord[ptr[s]][1][2]] += acc

@kernel function seg_short!(y, @Const(op), @Const(ptr), @Const(segs), args)
    I = @index(Global)
    @inbounds begin
        s = segs[I]
        acc = zero(eltype(y))
        for l in ptr[s]:(ptr[s + 1] - 1)
            acc += _segterm(op, l, args)
        end
        _segput!(op, y, s, ptr, acc, args)
    end
end
@kernel function seg_group!(partial, @Const(op), @Const(gstart), @Const(gstop), args)
    g = @index(Group, Linear)
    l = @index(Local, Linear)
    acc = zero(eltype(partial))
    @inbounds for t in (gstart[g] + l - 1):GROUPSIZE:gstop[g]
        acc += _segterm(op, t, args)
    end
    res = @groupreduce(+, acc, zero(eltype(partial)); groupsize = GROUPSIZE)
    if l == 1
        @inbounds partial[g] = res
    end
end
@kernel function seg_finish!(y, @Const(op), @Const(ptr), @Const(long), @Const(kptr), @Const(partial), args)
    I = @index(Global)
    @inbounds begin
        acc = zero(eltype(y))
        for c in kptr[I]:(kptr[I + 1] - 1)
            acc += partial[c]
        end
        _segput!(op, y, long[I], ptr, acc, args)
    end
end

function _seg_reduce!(backend, y, op, plan::SegPlan, args::Tuple)
    isempty(plan.short) || seg_short!(backend)(y, op, plan.ptr, plan.short, args; ndrange = length(plan.short))
    if !isempty(plan.long)
        seg_group!(backend, GROUPSIZE)(plan.partial, op, plan.gstart, plan.gstop, args; ndrange = length(plan.gstart) * GROUPSIZE)
        seg_finish!(backend)(y, op, plan.ptr, plan.long, plan.kptr, plan.partial, args; ndrange = length(plan.long))
    end
    return y
end
# y[key] += the sum of the terms of each segment of the sorted list `coord`
_seg_reduce!(backend, y, op, plan::SegPlan, coord, V, x) = _seg_reduce!(backend, y, op, plan, (coord, V, x))
# the same where output entry s is the s-th segment itself
_seg_reduce_index!(backend, y, plan::SegPlan, coord, V) = _seg_reduce!(backend, y, _GatherIndex(), plan, (coord, V, nothing))

# ---- Fused evaluation of augmented constraint rows -------------------------------
#
# The terms that `add_con!` adds to constraint rows were evaluated into a buffer and
# then summed per row, one work item per row. With few rows and many terms per row
# (a quadratic constraint has one term per nonzero of its matrix) that loop dominated
# `cons!`. Now each augmentation block keeps the permutation that orders its terms by
# target row (its data stays where it is, so Jacobian and Hessian coordinates keep their
# order). A short row's terms are evaluated one per work item, and a long row is cut
# into ranges that one workgroup evaluates and sums with `@groupreduce`, so the number
# of work items follows the number of terms and the backend does the rest. The partial
# sums of all blocks are then added per row. A constraint block whose pairs do not map
# one to one onto its rows (`add_con(c, 1 => ... for i in 1:4)`) sums its rows the same
# way, after those rows are cleared.

struct AugBlockPlan{VI, VC}
    perm::VI        # the block's terms in row order (stable: a row keeps its terms in order)
    nshort::Int     # perm[1:nshort] are the terms of short rows, one partial each
    groups::VC      # (start, stop): workgroup g sums the terms perm[start:stop], one partial
    poff::Int       # offset of this block's partials in the partial buffer
end
struct AugPlan{B, VI, VT}
    blocks::B       # per block of `cons`: an AugBlockPlan for an augmentation, else nothing
    partial::VT     # partial sums of all blocks
    perm::VI        # partial indices sorted by row
    ptr::VI         # the partials of the r-th augmented row are perm[ptr[r]:ptr[r+1]-1]
    rows::VI        # the augmented rows
    reset::VI       # the rows of constraint blocks that are summed here
end

# `rows_host[oa + I]` is the target row of term I of the augmentation block at `oa`.
function _aug_plan(backend, x0, cons, rows_host::Vector{Int})
    chunk_rows, reset = Int[], Int[]
    plan_of(con) = _block_plan(con, backend, x0, rows_host, chunk_rows, reset)
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
        ExaModels.convert_array(reset, backend),
    )
end
_block_plan(con, backend, x0, rows_host, chunk_rows, reset) = nothing
_block_plan(con::ExaModels.ConstraintAugmentation, backend, x0, rows_host, chunk_rows, reset) =
    _row_plan(rows_host[(con.oa + 1):(con.oa + length(con.itr))], backend, chunk_rows)
@kernel function ker_conrows!(r, @Const(f), @Const(itr))
    I = @index(Global)
    @inbounds r[I] = ExaModels.offset0(f, itr, I)
end
function _block_plan(
        con::ExaModels.Constraint{F}, backend, x0, rows_host, chunk_rows, reset,
    ) where {P <: Pair, F <: ExaModels.SIMDFunction{P}}
    n = length(con.itr)
    r = similar(x0, Int, n)
    n > 0 && ker_conrows!(backend)(r, con.f, con.itr; ndrange = n)
    r = Array(r)
    span = (con.offset + 1):(con.offset + ExaModels.total(con.size))
    sort(r) == span && return _row_plan(Int[], backend, chunk_rows)
    append!(reset, span)
    return _row_plan(r, backend, chunk_rows)
end
# `r[t]` is the row of term t.
function _row_plan(r, backend, chunk_rows)
    n = length(r)
    sorted = sortperm(r)
    poff = length(chunk_rows)
    short, long = Int[], Int[]
    ranges = Tuple{Int, Int}[]      # of long rows, into `long`
    i = 1
    while i <= n
        j = i
        while j < n && r[sorted[j + 1]] == r[sorted[i]]
            j += 1
        end
        if j - i + 1 <= SHORTROW
            append!(short, view(sorted, i:j))
        else
            push!(ranges, (length(long) + 1, length(long) + j - i + 1))
            append!(long, view(sorted, i:j))
        end
        i = j + 1
    end
    append!(chunk_rows, r[short])
    groups = Tuple{Int, Int}[]
    ns = length(short)
    for (a, b) in ranges, (st, stop) in _group_ranges(a, b, backend)
        push!(groups, (ns + st, ns + stop))
        push!(chunk_rows, r[long[a]])
    end
    perm = vcat(short, long)
    return AugBlockPlan(ExaModels.convert_array(perm, backend), ns, ExaModels.convert_array(groups, backend), poff)
end

@kernel function aug_term!(partial, @Const(f), @Const(itr), @Const(perm), @Const(x), @Const(θ), @Const(poff))
    I = @index(Global)
    @inbounds partial[poff + I] = f(itr[perm[I]], x, θ)
end
@kernel function aug_group!(partial, @Const(f), @Const(itr), @Const(perm), @Const(groups), @Const(x), @Const(θ), @Const(poff))
    g = @index(Group, Linear)
    l = @index(Local, Linear)
    start, stop = @inbounds groups[g]
    s = zero(eltype(partial))
    @inbounds for t in (start + l - 1):GROUPSIZE:stop
        s += f(itr[perm[t]], x, θ)
    end
    res = @groupreduce(+, s, zero(eltype(partial)); groupsize = GROUPSIZE)
    if l == 1
        @inbounds partial[poff + g] = res
    end
end
@kernel function aug_reset!(y, @Const(rows))
    I = @index(Global)
    @inbounds y[rows[I]] = zero(eltype(y))
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
    b.nshort > 0 && aug_term!(backend)(partial, con.f, con.itr, b.perm, x, θ, b.poff; ndrange = b.nshort)
    ng = length(b.groups)
    ng > 0 && aug_group!(backend, GROUPSIZE)(partial, con.f, con.itr, b.perm, b.groups, x, θ, b.poff + b.nshort; ndrange = ng * GROUPSIZE)
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
    isempty(plan.reset) || aug_reset!(backend)(y, plan.reset; ndrange = length(plan.reset))
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
@kernel function aug_term_jv!(partial, @Const(f), @Const(itr), @Const(perm), @Const(x), @Const(θ), @Const(v), @Const(poff))
    I = @index(Global)
    @inbounds partial[poff + I] = _gdot(f(itr[perm[I]], ExaModels.AdjointNodeSource(x), θ), v, one(eltype(partial)))
end
@kernel function aug_group_jv!(partial, @Const(f), @Const(itr), @Const(perm), @Const(groups), @Const(x), @Const(θ), @Const(v), @Const(poff))
    g = @index(Group, Linear)
    l = @index(Local, Linear)
    start, stop = @inbounds groups[g]
    s = zero(eltype(partial))
    @inbounds for t in (start + l - 1):GROUPSIZE:stop
        s += _gdot(f(itr[perm[t]], ExaModels.AdjointNodeSource(x), θ), v, one(eltype(partial)))
    end
    res = @groupreduce(+, s, zero(eltype(partial)); groupsize = GROUPSIZE)
    if l == 1
        @inbounds partial[poff + g] = res
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
    b.nshort > 0 && aug_term_jv!(backend)(partial, con.f, con.itr, b.perm, x, θ, v, b.poff; ndrange = b.nshort)
    ng = length(b.groups)
    ng > 0 && aug_group_jv!(backend, GROUPSIZE)(partial, con.f, con.itr, b.perm, b.groups, x, θ, v, b.poff + b.nshort; ndrange = ng * GROUPSIZE)
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
    isempty(plan.reset) || aug_reset!(backend)(Jv, plan.reset; ndrange = length(plan.reset))
    if !isempty(plan.rows)
        _aug_jv!(backend, plan.partial, plan.blocks, cons, x, θ, v)
        aug_finish!(backend)(Jv, plan.partial, plan.perm, plan.ptr, plan.rows; ndrange = length(plan.rows))
    end
    return Jv
end

# ---- Fused objective ---------------------------------------------------------------
#
# The objective is the sum of all its terms. Each work item evaluates one term, each
# workgroup sums its terms with `@groupreduce`, and these sums are then added on the
# device, so no buffer with one entry per term is written.

struct ObjPlan{O, VT}
    poff::O         # per block: offset of its chunks in `partial`
    partial::VT
end
function _obj_plan(x0, objs)
    n = 0
    off(o) = (k = n; n += cld(length(o.itr), GROUPSIZE); k)
    poff = objs isa _LongTuple ? Int[off(o) for o in Any[objs...]] : map(off, objs)
    return ObjPlan(poff, similar(x0, n))
end

@kernel function obj_group!(partial, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(n), @Const(poff))
    I = @index(Global)
    res = @groupreduce(+, I <= n ? (@inbounds f(itr[I], x, θ)) : zero(eltype(partial)), zero(eltype(partial)); groupsize = GROUPSIZE)
    if @index(Local, Linear) == 1
        @inbounds partial[poff + @index(Group, Linear)] = res
    end
end
function _obj_block!(backend, partial, poff, o, x, θ)
    n = length(o.itr)
    n == 0 && return nothing
    obj_group!(backend, GROUPSIZE)(partial, o.f, o.itr, x, θ, n, poff; ndrange = n)
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
