# merge.jl — static family merging at construction time
#
# Every constraint block whose expression tree is mergeable (a type-level
# property) is stored in MERGED form from its first add: a one-segment
# merged block whose scalar leaves are hoisted into the iterator's own data
# (`Integer` leaves per row, since they feed variable indices that
# sparsity-structure evaluation reads with θ = NaNSource; `AbstractFloat`
# leaves once per segment, as loop-invariant struct loads that cannot alias
# the output vectors the way θ reads would).  A later `add_con` /
# `add_con!` whose tree type matches an existing family appends one segment;
# the core's type does not change.  All decisions are static: same tree type
# means merge, always; there is no runtime merge heuristic and no refusal
# path.  Trees containing node kinds outside the hoisting walk (`SumNode`,
# `ProdNode`, recipe placeholders, registered-function internals) are
# statically non-mergeable and are stored plain, as before this feature.
#
# The slot layout is a pure function of the tree type (one slot per scalar
# leaf position; no value-dependent sharing), so the merged block's type, the
# sparsity footprint, and all compiled code are identical at any replication
# count.  The hoisting walk is a generated function: the generator walks the
# tree TYPE and emits flat code (one getfield chain per leaf, literal slot
# tuples), so `iv::NTuple{K, Int}` and `fv::NTuple{S, T}` are concrete at any
# tree depth.  A runtime-recursive walk concatenating tuples per node widens
# past inference's limits on deep trees, which leaves a dynamic splat in
# compiled model builders — exactly what `juliac --trim` refuses.  The
# generator is pure codegen from the type (no eval), so it is trim-safe.
# The cost of position-wise slots: a tree
# relying on ===-shared spliced subtrees (deep `add_expr` chains) carries a
# correspondingly larger per-row sparsity footprint; `add_expr(...; lift =
# true)` is the remedy for such models.
#
# Merging is unconditional: there is no off switch, so every core pays the
# same (statically decided) path and the whole suite exercises merged form.

"""
    MergedRow{K, D}

One row of a merged family: global row offset `o0`, Jacobian/Hessian nonzero
bases `o1`/`o2`, the `K` hoisted Integer leaves, the base of this row's
block's Float slots in `θ`, and the original iterator element `d`.
"""
struct MergedRow{K, S, T, D}
    o0::Int
    o1::Int
    o2::Int
    iv::NTuple{K, Int}
    fv::NTuple{S, T}
    d::D
end

const _MROW_IV = 4  # field position of `iv` in MergedRow
const _MROW_FV = 5  # field position of `fv` in MergedRow
const _MROW_D  = 6  # field position of `d` in MergedRow

# ── merged iterators ─────────────────────────────────────────────────────────

struct MergedSeg{K, S, T, I}
    itr::I
    o0::Int
    o1::Int
    o2::Int
    iv::NTuple{K, Int}
    fv::NTuple{S, T}
end

"""
    SegmentedItr{K, D, I, RF}

Lazy merged iterator for affine-offset families: one `MergedSeg` per source
block, the original iterators kept alive (a `UnitRange` stays a `UnitRange`),
rows materialized on demand.  `rep` is the first block's `SIMDFunction`, the
layout key for later arrivals.  The containers are mutable inside an
immutable struct: appending a segment changes no types, so the core's type
is stable as a family grows.
"""
struct SegmentedItr{K, S, T, D, I, RF} <: AbstractVector{MergedRow{K, S, T, D}}
    rep::RF
    segs::Vector{MergedSeg{K, S, T, I}}
    offs::Vector{Int}           # cumulative row counts
    seglen::Base.RefValue{Int}  # > 0 when all segments have equal length
    o1step::Int
    o2step::Int
end
Base.size(s::SegmentedItr) = (isempty(s.offs) ? 0 : (@inbounds s.offs[end]),)
@inline _segof(s::SegmentedItr, i::Int) =
    s.seglen[] > 0 ? div(i - 1, s.seglen[]) + 1 : searchsortedfirst(s.offs, i)
@inline function Base.getindex(s::SegmentedItr{K, S, T, D}, i::Int) where {K, S, T, D}
    j = _segof(s, i)
    seg = @inbounds s.segs[j]
    base = j == 1 ? 0 : @inbounds s.offs[j-1]
    r = i - base
    return MergedRow{K, S, T, D}(
        seg.o0 + r,
        seg.o1 + s.o1step * (r - 1),
        seg.o2 + s.o2step * (r - 1),
        seg.iv, seg.fv, (@inbounds seg.itr[r]),
    )
end

"""
    MergedRows{K, D, RF}

Materialized merged iterator, used for augmentation families (their row
targets are data-driven, not affine) and for device finalization.  Rows are
appended in place as the family grows.
"""
struct MergedRows{K, S, T, D, RF} <: AbstractVector{MergedRow{K, S, T, D}}
    rep::RF
    rows::Vector{MergedRow{K, S, T, D}}
end
Base.size(m::MergedRows) = Base.size(m.rows)
@inline Base.getindex(m::MergedRows, i::Int) = @inbounds m.rows[i]

const _MergedItr = Union{SegmentedItr, MergedRows}

_mrep(s::SegmentedItr) = s.rep
_mrep(m::MergedRows) = m.rep
_mrow_k(::Type{MergedRow{K, S, T, D}}) where {K, S, T, D} = K
_mrow_dtype(::Type{MergedRow{K, S, T, D}}) where {K, S, T, D} = D

# ── per-row offsets for merged blocks ────────────────────────────────────────
@inbounds @inline offset0(f::F, itr::AbstractVector{<:MergedRow}, i, dims) where {F <: SIMDFunction} =
    getfield(itr[i], :o0)
@inbounds @inline offset0(f::F, itr::AbstractVector{<:MergedRow}, i) where {F <: SIMDFunction} =
    getfield(itr[i], :o0)
@inbounds @inline offset1(a::Constraint{F, I}, i) where {F, I <: AbstractVector{<:MergedRow}} =
    getfield(a.itr[i], :o1)
@inbounds @inline offset2(a::Constraint{F, I}, i) where {F, I <: AbstractVector{<:MergedRow}} =
    getfield(a.itr[i], :o2)

# ── mergeability: a property of the tree TYPE ────────────────────────────────

_mergeable(::Type{Node1{F, I}}) where {F, I} = _mergeable(I)
_mergeable(::Type{Node2{F, I1, I2}}) where {F, I1, I2} = _mergeable(I1) && _mergeable(I2)
_mergeable(::Type{Var{I}}) where {I} = _mergeable(I)
_mergeable(::Type{ParameterNode{I}}) where {I} = _mergeable(I)
_mergeable(::Type{DataIndexed{I, J}}) where {I, J} = _mergeable(I)
_mergeable(::Type{DataSource}) = true
_mergeable(::Type{<:Constant}) = true
_mergeable(::Type{VarSource}) = true
_mergeable(::Type{ParameterSource}) = true
_mergeable(::Type{<:Val}) = true
_mergeable(::Type{Null{T}}) where {T} = T <: Real || T === Nothing
_mergeable(::Type{T}) where {T <: Real} = true
_mergeable(::Type{Pair{A, B}}) where {A, B} = _mergeable(B)   # aug: idx => expr
_mergeable(::Type) = false

# ── the static hoisting walk ─────────────────────────────────────────────────
# `_shoist(b)` returns (merged_node, ivals::NTuple{K, Int}, fvals::NTuple{S}).
# Slot layout = first-visit order over leaf POSITIONS (depth-first, inner1
# before inner2); slot indices live in FIELDS of the accessor nodes.  It is
# a generated function: the generator recurses over the tree TYPE (ordinary
# recursion at generation time, no inference involved) and emits straight-
# line code — one getfield chain per subnode and literal tuples for the
# slots — so the result types are concrete however deep the tree is.

"""
    IvRef <: AbstractNode

Reads the `k`-th hoisted Integer leaf out of a [`MergedRow`](@ref) element.
Available in structure passes (which carry θ = NaNSource), since the value
comes from the element, not from θ.
"""
struct IvRef <: AbstractNode
    k::Int
end
@inline (v::IvRef)(i, x, θ) = @inbounds getfield(i, _MROW_IV)[v.k]
@inline (v::IvRef)(i::Identity, x, θ) = eltype(θ)(NaN)

"""
    FvRef <: AbstractNode

Reads the `k`-th hoisted Float leaf out of a [`MergedRow`](@ref) element.
The values live in the segment (loop-invariant, one copy per source block),
not in θ, so no aliasing with the output vectors blocks LLVM from hoisting
the loads out of the row loop.
"""
struct FvRef <: AbstractNode
    k::Int
end
@inline (v::FvRef)(i, x, θ) = @inbounds getfield(i, _MROW_FV)[v.k]
@inline (v::FvRef)(i::Identity, x, θ) = eltype(θ)(NaN)

# Generation-time walk: one method per node kind, dispatched on the node's
# TYPE (`::Type{...}`), mirroring `_mergeable` above.  `path` is the symbol
# holding that subnode of `b`; each method emits SSA statements into `stmts`
# and slot expressions into `ivs`/`fvs` (slot number = push order, matching
# the accessor literals) and returns the merged node's symbol or expression.
function _gwalk(::Type{Node1{F, I}}, path, stmts, ivs, fvs) where {F, I}
    s = gensym(:c)
    push!(stmts, :($s = getfield($path, :inner)))
    n = _gwalk(I, s, stmts, ivs, fvs)
    out = gensym(:n)
    push!(stmts, :($out = _g_rebuild1($path, $n)))
    return out
end
function _gwalk(::Type{Node2{F, I1, I2}}, path, stmts, ivs, fvs) where {F, I1, I2}
    s1 = gensym(:c)
    push!(stmts, :($s1 = getfield($path, :inner1)))
    n1 = _gwalk(I1, s1, stmts, ivs, fvs)
    s2 = gensym(:c)
    push!(stmts, :($s2 = getfield($path, :inner2)))
    n2 = _gwalk(I2, s2, stmts, ivs, fvs)
    out = gensym(:n)
    push!(stmts, :($out = _g_rebuild2($path, $n1, $n2)))
    return out
end
function _gwalk(::Type{Var{I}}, path, stmts, ivs, fvs) where {I}
    s = gensym(:c)
    push!(stmts, :($s = getfield($path, :i)))
    n = _gwalk(I, s, stmts, ivs, fvs)
    out = gensym(:n)
    push!(stmts, :($out = Var{typeof($n)}($n)))
    return out
end
function _gwalk(::Type{ParameterNode{I}}, path, stmts, ivs, fvs) where {I}
    s = gensym(:c)
    push!(stmts, :($s = getfield($path, :i)))
    n = _gwalk(I, s, stmts, ivs, fvs)
    out = gensym(:n)
    push!(stmts, :($out = ParameterNode{typeof($n)}($n)))
    return out
end
function _gwalk(::Type{DataIndexed{I, J}}, path, stmts, ivs, fvs) where {I, J}
    s = gensym(:c)
    push!(stmts, :($s = getfield($path, :inner)))
    n = _gwalk(I, s, stmts, ivs, fvs)
    out = gensym(:n)
    # J may be a Symbol (a field key), so quote it rather than splice it
    push!(stmts, :($out = DataIndexed($n, $(QuoteNode(J)))))
    return out
end
function _gwalk(::Type{DataSource}, path, stmts, ivs, fvs)
    out = gensym(:n)
    push!(stmts, :($out = DataIndexed(DataSource(), $_MROW_D)))
    return out
end
# structure lives entirely in the type: reuse the arriving node
_gwalk(::Type{<:Constant}, path, stmts, ivs, fvs) = path
_gwalk(::Type{VarSource}, path, stmts, ivs, fvs) = path
_gwalk(::Type{ParameterSource}, path, stmts, ivs, fvs) = path
_gwalk(::Type{<:Val}, path, stmts, ivs, fvs) = path
_gwalk(::Type{Null{Nothing}}, path, stmts, ivs, fvs) = path
# a Null's value is a scalar field like any other: hoist it
function _gwalk(::Type{Null{T}}, path, stmts, ivs, fvs) where {T <: Real}
    push!(fvs, :(getfield($path, :value)))
    return :(FvRef($(length(fvs))))
end
function _gwalk(::Type{T}, path, stmts, ivs, fvs) where {T <: Integer}
    push!(ivs, :(Int($path)))
    return :(IvRef($(length(ivs))))
end
function _gwalk(::Type{T}, path, stmts, ivs, fvs) where {T <: Real}
    push!(fvs, path)
    return :(FvRef($(length(fvs))))
end
# unreachable behind the _mergeable gate; loud if the two ever drift
_gwalk(::Type{T}, path, stmts, ivs, fvs) where {T} =
    error("_shoist: node type not covered by the hoisting walk: ", T)

@inline _g_rebuild1(::Node1{F}, n) where {F} = Node1{F, typeof(n)}(n)
@inline _g_rebuild2(::Node2{F}, n1, n2) where {F} =
    Node2{F, typeof(n1), typeof(n2)}(n1, n2)

@generated function _shoist(b)
    stmts = Any[]
    ivs = Any[]
    fvs = Any[]
    node = _gwalk(b, :b, stmts, ivs, fvs)
    return quote
        $(Expr(:meta, :inline))
        $(stmts...)
        ($node, ($(ivs...),), ($(fvs...),))
    end
end

@inline _family_tree(f::SIMDFunction) = _pair_second(f.f)
@inline _pair_second(p::Pair) = p.second
@inline _pair_second(t) = t

# ── family membership: dispatch on the stored representative's type ─────────

@inline _same_family(a, f, ::Type{E}) where {E} = false
@inline _same_family(a::Constraint{F2, I}, f::F, ::Type{E}) where {F2, F, E, I <: _MergedItr} =
    typeof(_mrep(getfield(a, :itr))) == F && _mrow_dtype(eltype(getfield(a, :itr))) == E

# ── block creation and appends ───────────────────────────────────────────────

# per-row o0 values of an arriving block, in iteration order (folds in Pair
# row targets and dims for pair-headed plain blocks)
function _row_o0s(f, pars, dims)
    h = Constraint(f, pars, 0, dims, nothing)
    return Int[offset0(h, r) for r in 1:length(pars)]
end

# `map(T, t)` on tuples longer than 32 leaves Base's inlined path and goes
# through a Vector{Any} splat, which is a dynamic call `juliac --trim`
# refuses; `ntuple(.., Val(N))` emits a literal tuple at any length.
@inline _fvt(::Type{T}, fvs::Tuple) where {T} =
    ntuple(i -> T(@inbounds fvs[i]), Val(length(fvs)))

# First block of a family: a one-segment merged block.  The merged tree's
# sparsity (`mf.o1step`/`o2step`) is what the caller accounts nnz with, so
# the family's footprint is fixed by the type from the start and later
# arrivals can never mismatch.  Segment bases come from the arriving block's
# own counters (f.o0/f.o1/f.o2 were taken from the core at this add).
@inline function _merged_first(::Type{T}, f, pars, dims, tag, backend) where {T}
    rep = f
    tree, ivs, fvs = _shoist(_family_tree(rep))
    mf = _simdfunction(T, tree, 0, 0, 0)
    itr = _merged_first_itr(T, getfield(f, :f), backend, rep, mf, f, pars, dims,
                            ivs, _fvt(T, fvs))
    return Constraint(mf, itr, 0, (length(pars),), tag), mf
end

# a plain (non-pair) head on the host gets the lazy segmented iterator
@inline function _merged_first_itr(::Type{T}, head, ::Nothing, rep, mf, f, pars, dims,
                                   ivs, fvt) where {T}
    K = length(ivs)
    S = length(fvt)
    segs = MergedSeg{K, S, T, typeof(pars)}[MergedSeg{K, S, T, typeof(pars)}(pars, f.o0, f.o1, f.o2, ivs, fvt)]
    return SegmentedItr{K, S, T, eltype(pars), typeof(pars), typeof(rep)}(
        rep, segs, Int[length(pars)], Ref(length(pars)), mf.o1step, mf.o2step)
end
# pair heads (data-driven row targets) and device backends materialize rows
@inline _merged_first_itr(::Type{T}, head::Pair, ::Nothing, rep, mf, f, pars, dims,
                          ivs, fvt) where {T} =
    _merged_rows(T, rep, mf, f, pars, dims, ivs, fvt)
@inline _merged_first_itr(::Type{T}, head, backend, rep, mf, f, pars, dims,
                          ivs, fvt) where {T} =
    _merged_rows(T, rep, mf, f, pars, dims, ivs, fvt)

@inline function _merged_rows(::Type{T}, rep, mf, f, pars, dims, ivs, fvt) where {T}
    K = length(ivs)
    S = length(fvt)
    o0s = _row_o0s(f, pars, dims)
    itrc = collect(pars)
    rows = [MergedRow{K, S, T, eltype(itrc)}(o0s[r], f.o1 + mf.o1step * (r - 1),
                      f.o2 + mf.o2step * (r - 1), ivs, fvt, itrc[r]) for r in eachindex(itrc)]
    return MergedRows{K, S, T, eltype(itrc), typeof(rep)}(rep, rows)
end

# A later arrival: extract in the representative's layout, append one
# segment.  nnz bases again come from the arriving block's own counters.
@inline function _merged_append(::Type{T}, prev, f, pars, dims, tag) where {T}
    # same family means same tree type, so the arriving tree's own walk uses
    # the representative's slot layout by construction
    _, ivs, fvs = _shoist(_family_tree(f))
    return _merged_append_itr(getfield(prev, :itr), prev, f, pars, dims, tag,
                              ivs, _fvt(T, fvs))
end

@inline function _merged_append_itr(m::SegmentedItr, prev, f, pars, dims, tag, ivs, fvt)
    push!(m.segs, eltype(m.segs)(pars, f.o0, f.o1, f.o2, ivs, fvt))
    newlen = length(pars)
    push!(m.offs, m.offs[end] + newlen)
    m.seglen[] = (m.seglen[] == newlen) ? newlen : 0
    return Constraint(prev.f, m, 0, (m.offs[end],), tag)
end
@inline function _merged_append_itr(m::MergedRows, prev, f, pars, dims, tag, ivs, fvt)
    mf = prev.f
    o0s = _row_o0s(f, pars, dims)
    itrc = collect(pars)
    for r in eachindex(itrc)
        push!(m.rows, eltype(m.rows)(o0s[r], f.o1 + mf.o1step * (r - 1),
                                     f.o2 + mf.o2step * (r - 1), ivs, fvt, itrc[r]))
    end
    return Constraint(prev.f, m, 0, (length(m.rows),), tag)
end

# ── the merge entry point used by _add_con / _add_con! ───────────────────────
#
# Returns `nothing` when the block is statically non-mergeable (stored plain,
# as before this feature), or `(cons′, o1step, o2step)`: the updated block
# list and the per-row nnz steps to account nnzj/nnzh with.

@inline function _merge_block(c::ExaCore{T}, f, pars, dims, tag) where {T}
    _mergeable(typeof(f.f)) || return nothing
    _merge_allowed(getfield(c, :nargs), getfield(f, :f), c.backend) || return nothing
    _merge_itr_ok(pars) || return nothing
    return _merge_result(_smerge(T, c.cons, f, pars, dims, tag, eltype(pars)),
                         T, c, f, pars, dims, tag)
end

# recipes (nargs > 0) are stored plain; pair-headed plain blocks (data-driven
# row targets) stay plain on device backends, where merging would move
# cross-block row collisions from sequential launches into one kernel
@inline _merge_allowed(::Val{0}, head, backend) = true
@inline _merge_allowed(::Val{0}, head::Pair, ::Nothing) = true
@inline _merge_allowed(::Val{0}, head::Pair, backend) = false
@inline _merge_allowed(nargs, head, backend) = false

# an existing family absorbed the block, or (on `nothing`) it starts one
@inline _merge_result(r::Tuple, ::Type, c, f, pars, dims, tag) = r
@inline function _merge_result(::Nothing, ::Type{T}, c, f, pars, dims, tag) where {T}
    con, mf = _merged_first(T, f, pars, dims, tag, c.backend)
    return _prep(c.cons, con), mf.o1step, mf.o2step
end

@inline _merge_itr_ok(pars) = _merge_itr_ok(Base.IteratorSize(pars))
@inline _merge_itr_ok(::Union{Base.HasLength, Base.HasShape}) = true
@inline _merge_itr_ok(::Base.IteratorSize) = false

# scan + replace in one recursion; typed for tuple storage, dynamic for the
# Vector{Any} default — the same fold either way
@inline _smerge(::Type{T}, cons::Tuple{}, f, pars, dims, tag, ::Type{E}) where {T, E} = nothing
@inline function _smerge(::Type{T}, cons::Tuple, f, pars, dims, tag, ::Type{E}) where {T, E}
    b = first(cons)
    rest = Base.tail(cons)
    if _same_family(b, f, E) && getfield(b, :tag) === tag &&
       _appendable(getfield(b, :itr), pars)
        nb = _merged_append(T, b, f, pars, dims, tag)
        mf = getfield(nb, :f)
        return (nb, rest...), mf.o1step, mf.o2step
    end
    r = _smerge(T, rest, f, pars, dims, tag, E)
    r === nothing && return nothing
    ncons, s1, s2 = r
    return (b, ncons...), s1, s2
end
function _smerge(::Type{T}, cons::Vector{Any}, f, pars, dims, tag, ::Type{E}) where {T, E}
    for (i, b) in enumerate(cons)
        if _same_family(b, f, E) && getfield(b, :tag) === tag &&
           _appendable(getfield(b, :itr), pars)
            nb = _merged_append(T, b, f, pars, dims, tag)
            out = copy(cons)
            out[i] = nb
            mf = getfield(nb, :f)
            return out, mf.o1step, mf.o2step
        end
    end
    return nothing
end
@inline _appendable(m::SegmentedItr{K, S, T, D, I}, pars::I2) where {K, S, T, D, I, I2} = I === I2
@inline _appendable(m::MergedRows, pars) = _merge_itr_ok(pars)
@inline _appendable(m, pars) = false

# ── model-build finalization ─────────────────────────────────────────────────
# Device backends evaluate merged blocks through the extension's kernels over
# a materialized element array; non-concrete models keep var/par/refs
# type-erased so the model's type is a function of its families alone.

_is_merged(b) = false
_is_merged(::Constraint{F, I}) where {F, I <: _MergedItr} = true
_has_merged(cons) = any(_is_merged, cons)

_erase_refs(refs::NamedTuple) = Pair{Symbol, Any}[k => v for (k, v) in pairs(refs)]
_erase_refs(refs) = refs

# device finalization: merged iterators become device element arrays
_materialize_block(backend, b) = b
_materialize_block(backend, b::Constraint{F, I}) where {F, I <: _MergedItr} =
    Constraint(b.f, _materialize_mergeditr(backend, getfield(b, :itr)), 0, b.size, b.tag)

_materialize_mergeditr(backend, s::SegmentedItr) =
    convert_array([s[i] for i in 1:length(s)], backend)
_materialize_mergeditr(backend, m::MergedRows) = convert_array(m.rows, backend)

_finalize_merged(c::ExaCore) = _finalize_merged2(c, c.backend, getfield(c, :var))

# plain CPU, concrete storage: identity, fully static
@inline _finalize_merged2(c::ExaCore, ::Nothing, ::Tuple) = c
# plain CPU, erased storage: keep var/par/refs erased when anything merged
function _finalize_merged2(c::ExaCore, ::Nothing, ::Vector{Any})
    _has_merged(c.cons) || return c
    return ExaCore(c; cons = Tuple(Any[b for b in c.cons]),
                   var = Any[c.var...], par = Any[c.par...],
                   refs = _erase_refs(getfield(c, :refs)))
end
# device: additionally materialize merged iterators as device arrays
function _finalize_merged2(c::ExaCore, backend, var::Vector{Any})
    cons = Any[_materialize_block(backend, b) for b in c.cons]
    return ExaCore(c; cons = Tuple(cons), var = Any[c.var...], par = Any[c.par...],
                   refs = _erase_refs(getfield(c, :refs)))
end
function _finalize_merged2(c::ExaCore, backend, var)
    return ExaCore(c; cons = Tuple(Any[_materialize_block(backend, b) for b in c.cons]))
end

# ── specialized hot loops for the lazy iterator ──────────────────────────────
# One affine @simd inner loop per segment, materializing the isbits row
# element on the fly: the same loop shape the unmerged blocks had.

function sjacobian!(y1, y2, f::Constraint{F, I}, x, θ, adj) where {F, I <: SegmentedItr}
    s = getfield(f, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = eltype(s)(seg.o0 + r, seg.o1 + s.o1step * (r - 1),
                           seg.o2 + s.o2step * (r - 1), seg.iv, seg.fv,
                           (@inbounds seg.itr[r]))
            @inbounds sjacobian!(y1, y2, f.f.f, el, x, θ, f.f.comp1,
                                 seg.o0 + r, seg.o1 + s.o1step * (r - 1), adj)
        end
    end
end

function shessian!(y1, y2, f::Constraint{F, I}, x, θ, adj1, adj2) where {F, I <: SegmentedItr}
    s = getfield(f, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = eltype(s)(seg.o0 + r, seg.o1 + s.o1step * (r - 1),
                           seg.o2 + s.o2step * (r - 1), seg.iv, seg.fv,
                           (@inbounds seg.itr[r]))
            @inbounds shessian!(y1, y2, f.f.f, el, x, θ, f.f.comp2,
                                seg.o2 + s.o2step * (r - 1), adj1, adj2)
        end
    end
end

function shessian!(y1, y2, f::Constraint{F, I}, x, θ, adj1s::V, adj2) where {F, I <: SegmentedItr, V <: AbstractVector}
    s = getfield(f, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = eltype(s)(seg.o0 + r, seg.o1 + s.o1step * (r - 1),
                           seg.o2 + s.o2step * (r - 1), seg.iv, seg.fv,
                           (@inbounds seg.itr[r]))
            @inbounds shessian!(y1, y2, f.f.f, el, x, θ, f.f.comp2,
                                seg.o2 + s.o2step * (r - 1), adj1s[seg.o0 + r], adj2)
        end
    end
end

function _cons_rows!(g, con::Constraint{F, I}, x, θ) where {F, I <: SegmentedItr}
    s = getfield(con, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = eltype(s)(seg.o0 + r, seg.o1 + s.o1step * (r - 1),
                           seg.o2 + s.o2step * (r - 1), seg.iv, seg.fv,
                           (@inbounds seg.itr[r]))
            @inbounds g[seg.o0 + r] += con.f(el, x, θ)
        end
    end
    return nothing
end
