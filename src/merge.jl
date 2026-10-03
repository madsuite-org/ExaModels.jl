# merge.jl: static family merging at construction time
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
# count.  The hoisting walk is three single-purpose dispatch recursions
# (tree, Integer slots, Float slots); the walk section below says why the
# split is load-bearing under `juliac --trim`.
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

# ── the static hoisting walk ─────────────────────────────────────────────────────────────

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

# The hoisting walk: three single-purpose recursions, one dispatch method
# per node kind each: the merged TREE (threading running slot counts as
# plain Int arguments), the Integer-slot tuple, and the Float-slot tuple.
# Keeping each recursion's return type small (a node, or one tuple) is what
# lets deep trees infer concretely: a single walk returning all three at
# once widens past inference's limits and leaves a dynamic splat in
# compiled model builders, which `juliac --trim` refuses.  Slot numbering
# is first-visit order over leaf positions (depth-first, inner1 before
# inner2), identical in all three recursions.

# the merged tree: scalar leaves become slot accessors (indices in FIELDS,
# never in type parameters), everything else is rebuilt as-is.  Slot offsets
# travel DOWN as plain Int arguments and subtree slot COUNTS come back up in
# the same recursion, so no separate counting traversal is needed; the return
# type stays small (a node plus two Ints) at any depth.
@inline function _stree(a::Node1{F, I}, io::Int, fo::Int) where {F, I}
    n, k, s = _stree(getfield(a, :inner), io, fo)
    return Node1{F, typeof(n)}(n), k, s
end
@inline function _stree(a::Node2{F, I1, I2}, io::Int, fo::Int) where {F, I1, I2}
    n1, k1, s1 = _stree(getfield(a, :inner1), io, fo)
    n2, k2, s2 = _stree(getfield(a, :inner2), io + k1, fo + s1)
    return Node2{F, typeof(n1), typeof(n2)}(n1, n2), k1 + k2, s1 + s2
end
@inline function _stree(a::Var, io::Int, fo::Int)
    n, k, s = _stree(getfield(a, :i), io, fo)
    return Var{typeof(n)}(n), k, s
end
@inline function _stree(a::ParameterNode, io::Int, fo::Int)
    n, k, s = _stree(getfield(a, :i), io, fo)
    return ParameterNode{typeof(n)}(n), k, s
end
@inline function _stree(a::DataIndexed{I, J}, io::Int, fo::Int) where {I, J}
    n, k, s = _stree(getfield(a, :inner), io, fo)
    return DataIndexed(n, J), k, s
end
@inline _stree(a::DataSource, io::Int, fo::Int) = DataIndexed(DataSource(), _MROW_D), 0, 0
@inline _stree(a::Constant, io::Int, fo::Int) = a, 0, 0
@inline _stree(a::VarSource, io::Int, fo::Int) = a, 0, 0
@inline _stree(a::ParameterSource, io::Int, fo::Int) = a, 0, 0
@inline _stree(a::Val, io::Int, fo::Int) = a, 0, 0
@inline _stree(a::Null{Nothing}, io::Int, fo::Int) = a, 0, 0
# a Null's value is a scalar field like any other: hoist it
@inline _stree(a::Null{T}, io::Int, fo::Int) where {T <: Real} = FvRef(fo + 1), 0, 1
@inline _stree(a::T, io::Int, fo::Int) where {T <: Integer} = IvRef(io + 1), 1, 0
@inline _stree(a::T, io::Int, fo::Int) where {T <: Real} = FvRef(fo + 1), 0, 1

# the slots.  Concatenating a tuple per node would mint one tuple TYPE per
# slot (81 of them on a deep capture tree), and inference pays for each, so
# collection is mutating instead: the recursions return only the running
# slot count, and one flat `ntuple` materializes the tuple at the end.
Base.@assume_effects :foldable _niv(::Type{Node1{F, I}}) where {F, I} = _niv(I)
Base.@assume_effects :foldable _niv(::Type{Node2{F, I1, I2}}) where {F, I1, I2} = _niv(I1) + _niv(I2)
Base.@assume_effects :foldable _niv(::Type{Var{I}}) where {I} = _niv(I)
Base.@assume_effects :foldable _niv(::Type{ParameterNode{I}}) where {I} = _niv(I)
Base.@assume_effects :foldable _niv(::Type{DataIndexed{I, J}}) where {I, J} = _niv(I)
_niv(::Type{T}) where {T <: Integer} = 1
_niv(::Type{<:Real}) = 0
_niv(::Type) = 0

Base.@assume_effects :foldable _nfv(::Type{Node1{F, I}}) where {F, I} = _nfv(I)
Base.@assume_effects :foldable _nfv(::Type{Node2{F, I1, I2}}) where {F, I1, I2} = _nfv(I1) + _nfv(I2)
Base.@assume_effects :foldable _nfv(::Type{Var{I}}) where {I} = _nfv(I)
Base.@assume_effects :foldable _nfv(::Type{ParameterNode{I}}) where {I} = _nfv(I)
Base.@assume_effects :foldable _nfv(::Type{DataIndexed{I, J}}) where {I, J} = _nfv(I)
_nfv(::Type{Null{T}}) where {T <: Real} = 1
_nfv(::Type{<:Null}) = 0
_nfv(::Type{<:Integer}) = 0
_nfv(::Type{<:Real}) = 1
_nfv(::Type) = 0

@inline _civ!(out, b::Node1, k::Int) = _civ!(out, getfield(b, :inner), k)
@inline _civ!(out, b::Node2, k::Int) =
    _civ!(out, getfield(b, :inner2), _civ!(out, getfield(b, :inner1), k))
@inline _civ!(out, b::Var, k::Int) = _civ!(out, getfield(b, :i), k)
@inline _civ!(out, b::ParameterNode, k::Int) = _civ!(out, getfield(b, :i), k)
@inline _civ!(out, b::DataIndexed, k::Int) = _civ!(out, getfield(b, :inner), k)
@inline _civ!(out, b::T, k::Int) where {T <: Integer} = (@inbounds out[k] = Int(b); k + 1)
@inline _civ!(out, b, k::Int) = k

@inline _cfv!(out, b::Node1, k::Int) = _cfv!(out, getfield(b, :inner), k)
@inline _cfv!(out, b::Node2, k::Int) =
    _cfv!(out, getfield(b, :inner2), _cfv!(out, getfield(b, :inner1), k))
@inline _cfv!(out, b::Var, k::Int) = _cfv!(out, getfield(b, :i), k)
@inline _cfv!(out, b::ParameterNode, k::Int) = _cfv!(out, getfield(b, :i), k)
@inline _cfv!(out, b::DataIndexed, k::Int) = _cfv!(out, getfield(b, :inner), k)
@inline _cfv!(out::Vector{T}, b::Null{<:Real}, k::Int) where {T} =
    (@inbounds out[k] = getfield(b, :value); k + 1)
@inline _cfv!(out::Vector, b::Integer, k::Int) = k
@inline _cfv!(out::Vector{T}, b::Real, k::Int) where {T} = (@inbounds out[k] = T(b); k + 1)
@inline _cfv!(out, b, k::Int) = k

@inline function _siv(b)
    K = _niv(typeof(b))
    out = Vector{Int}(undef, K)
    _civ!(out, b, 1)
    return ntuple(i -> (@inbounds out[i]), Val(K))
end
@inline function _sfv(::Type{T}, b) where {T}
    S = _nfv(typeof(b))
    out = Vector{T}(undef, S)
    _cfv!(out, b, 1)
    return ntuple(i -> (@inbounds out[i]), Val(S))
end

@inline _shoist(::Type{T}, b) where {T} = (first(_stree(b, 0, 0)), _siv(b), _sfv(T, b))

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

# First block of a family: a one-segment merged block.  The merged tree's
# sparsity (`mf.o1step`/`o2step`) is what the caller accounts nnz with, so
# the family's footprint is fixed by the type from the start and later
# arrivals can never mismatch.  Segment bases come from the arriving block's
# own counters (f.o0/f.o1/f.o2 were taken from the core at this add).
@inline function _merged_first(::Type{T}, f, pars, dims, tag, backend) where {T}
    rep = f
    tree, ivs, fvt = _shoist(T, _family_tree(rep))
    mf = _simdfunction(T, tree, 0, 0, 0)
    itr = _merged_first_itr(T, getfield(f, :f), backend, rep, mf, f, pars, dims,
                            ivs, fvt)
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
    _, ivs, fvt = _shoist(T, _family_tree(f))
    return _merged_append_itr(getfield(prev, :itr), prev, f, pars, dims, tag,
                              ivs, fvt)
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
# Vector{Any} default; the same fold either way
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
