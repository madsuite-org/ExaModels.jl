# merge.jl: static family merging at construction time
#
# Every constraint or objective block whose expression tree is mergeable (a
# type-level property) is stored in MERGED form from its first add: a
# one-segment merged block whose scalar leaves are hoisted into the iterator's
# own data (`Integer` leaves per row, since they feed variable indices that
# sparsity-structure evaluation reads with θ = NaNSource; `AbstractFloat`
# leaves once per segment, as loop-invariant struct loads that cannot alias
# the output vectors the way θ reads would).  A later `add_con` / `add_obj`
# whose tree type matches an existing family appends one segment; the core's
# type does not change.  All decisions are static: there is no runtime merge
# heuristic.  Stored plain instead: trees containing
# node kinds outside the hoisting walk (`SumNode`, `ProdNode`, recipe
# placeholders, registered-function internals), augmentation (`add_con!`)
# blocks, recipe cores (`nargs > 0`), and pair-headed blocks on device
# backends (`_merge_allowed`).
#
# The slot layout is a function of the tree type (one slot per scalar leaf
# position), so the merged block's type and all compiled code are identical at
# any replication count.  Sparsity is the source block's own: each block's
# nonzero compressors (`comp1`/`comp2`, the leaf-position to nonzero-slot maps
# its plain form had, computed by the `===` probe over its original tree) and
# per-row steps travel with its segment and reach the derivative passes
# through the row element, so two sites that reference the same variable
# (`x[i+1]` twice) share one Jacobian slot in merged form exactly as they did
# unmerged, and a family's segments may differ in footprint.  The compressor
# types are fixed by the tree type (one entry per leaf position), so this
# costs the family decision nothing: same tree type means merge, in both
# storage modes.  The hoisting walk is three single-purpose dispatch
# recursions (tree, Integer slots, Float slots); the walk section below says
# why the split is load-bearing under `juliac --trim`.
#
# Merging is unconditional: there is no off switch, so every core pays the
# same (statically decided) path and the whole suite exercises merged form.

"""
    MergedRow{T, IV, FV, D, C1, C2}

One row of a merged family: global row offset `o0`, Jacobian/Hessian nonzero
bases `o1`/`o2`, the hoisted Integer leaves `iv`, the hoisted Float leaves
`fv` (of type `T`; both shared by every row of the segment), the original
iterator element `d`, and the source block's own nonzero compressors
`comp1`/`comp2` (shared by every row of the segment).  The derivative passes
read the compressors from the row (`_comp1`/`_comp2`), never from the
family's `SIMDFunction`, so each segment keeps the sparsity its plain block
had.  On the host `iv`/`fv` are tuples and the compressors `Compressor`s; on
a device they are views into the layout's arrays ([`SegView`](@ref),
[`CompView`](@ref)), since a dynamic index into a tuple held by value in a
kernel stages the tuple into per-thread local memory.
"""
struct MergedRow{T, IV, FV, D, C1, C2}
    o0::Int
    o1::Int
    o2::Int
    iv::IV
    fv::FV
    d::D
    comp1::C1
    comp2::C2
end
@inline _comp1(f, p::MergedRow) = getfield(p, :comp1)
@inline _comp2(f, p::MergedRow) = getfield(p, :comp2)

const _MROW_IV = 4  # field position of `iv` in MergedRow
const _MROW_FV = 5  # field position of `fv` in MergedRow
const _MROW_D = 6  # field position of `d` in MergedRow


# ── merged iterators ─────────────────────────────────────────────────────────

struct MergedSeg{K, S, T, I, C1, C2}
    itr::I
    o0::Int
    o1::Int
    o2::Int
    iv::NTuple{K, Int}
    fv::NTuple{S, T}
    comp1::C1
    comp2::C2
    o1step::Int
    o2step::Int
    dims::Tuple                 # the source block's size (a 2-D generator's product shape), for writers
end
# the segment a block contributes: its own offsets, leaves, compressors, steps and shape
@inline _mergedseg(::Type{T}, f, pars, dims, ivs::NTuple{K, Int}, fvt::NTuple{S, T}) where {K, S, T} =
    MergedSeg{K, S, T, typeof(pars), typeof(f.comp1), typeof(f.comp2)}(
        pars, f.o0, f.o1, f.o2, ivs, fvt, f.comp1, f.comp2, f.o1step, f.o2step, dims
    )

"""
    SegmentedItr{K, S, T, D, I, RF, C1, C2}

Lazy merged iterator for affine-offset families: one `MergedSeg` per source
block (holding the iterator `_adapt_gen` produced for it), rows materialized
on demand.  `rep` is the first block's `SIMDFunction`, the layout key for
later arrivals.  Appending copies the segment containers (`_merged_append_itr`)
so earlier cores and models keep the iterator they had; the copy changes no
types, so the core's type is stable as a family grows.
"""
struct SegmentedItr{K, S, T, D, I, RF, C1, C2} <: AbstractVector{MergedRow{T, NTuple{K, Int}, NTuple{S, T}, D, C1, C2}}
    rep::RF
    segs::Vector{MergedSeg{K, S, T, I, C1, C2}}
    offs::Vector{Int}           # cumulative row counts
    seglen::Int                 # > 0 when all segments have equal length (copy-on-append: never mutated)
end
Base.size(s::SegmentedItr) = (isempty(s.offs) ? 0 : (@inbounds s.offs[end]),)
Base.IndexStyle(::Type{<:SegmentedItr}) = IndexLinear()
@inline _segof(s::SegmentedItr, i::Int) =
    s.seglen > 0 ? div(i - 1, s.seglen) + 1 : searchsortedfirst(s.offs, i)
@inline function Base.getindex(s::SegmentedItr, i::Int)
    j = _segof(s, i)
    seg = @inbounds s.segs[j]
    base = j == 1 ? 0 : @inbounds s.offs[j - 1]
    return _segrow(s, seg, i - base)
end
# row `r` of segment `seg` (1-based within the segment)
@inline function _segrow(s::SegmentedItr, seg, r::Int)
    return eltype(s)(
        seg.o0 + r,
        seg.o1 + seg.o1step * (r - 1),
        seg.o2 + seg.o2step * (r - 1),
        seg.iv, seg.fv, (@inbounds seg.itr[r]), seg.comp1, seg.comp2,
    )
end

"""
    MergedRows{K, S, T, D, RF, C1, C2}

Materialized merged iterator, used on the host for pair-headed families
(their row targets are data-driven, not affine) and, on device backends, as
the element array every merged block is finalized into.  Growing a family
copies the row vector (`_merged_append_itr`), so earlier cores and models keep
the rows they had.
"""
struct MergedRows{K, S, T, D, RF, C1, C2} <: AbstractVector{MergedRow{T, NTuple{K, Int}, NTuple{S, T}, D, C1, C2}}
    rep::RF
    rows::Vector{MergedRow{T, NTuple{K, Int}, NTuple{S, T}, D, C1, C2}}
end
Base.size(m::MergedRows) = Base.size(m.rows)
Base.IndexStyle(::Type{<:MergedRows}) = IndexLinear()
@inline Base.getindex(m::MergedRows, i::Int) = @inbounds m.rows[i]

const _MergedItr = Union{SegmentedItr, MergedRows}

_mrep(s::SegmentedItr) = s.rep
_mrep(m::MergedRows) = m.rep
_mrow_dtype(::Type{MergedRow{T, IV, FV, D, C1, C2}}) where {T, IV, FV, D, C1, C2} = D

# ── per-row offsets for merged blocks ────────────────────────────────────────
@inbounds @inline offset0(f::F, itr::AbstractVector{<:MergedRow}, i, dims) where {F <: SIMDFunction} =
    getfield(itr[i], :o0)
@inbounds @inline offset0(f::F, itr::AbstractVector{<:MergedRow}, i) where {F <: SIMDFunction} =
    getfield(itr[i], :o0)
# The pair-headed methods in nlp.jl dispatch on the head alone, so the two
# above would be ambiguous with them for a pair head over merged rows. That
# combination never arises (`_merged_first` hoists the pair's second half),
# but the intersections are defined so `detect_ambiguities` stays clean.
@inbounds @inline offset0(f::F, itr::AbstractVector{<:MergedRow}, i, dims) where {P <: Pair, F <: SIMDFunction{P}} =
    getfield(itr[i], :o0)
@inbounds @inline offset0(f::F, itr::AbstractVector{<:MergedRow}, i, dims) where {I <: Integer, P <: Pair{I}, F <: SIMDFunction{P}} =
    getfield(itr[i], :o0)
@inbounds @inline offset0(f::F, itr::AbstractVector{<:MergedRow}, i, dims) where {T <: Tuple, P <: Pair{T}, F <: SIMDFunction{P}} =
    getfield(itr[i], :o0)
@inbounds @inline offset0(f::F, itr::AbstractVector{<:MergedRow}, i) where {P <: Pair, F <: SIMDFunction{P}} =
    getfield(itr[i], :o0)
@inbounds @inline offset0(f::F, itr::AbstractVector{<:MergedRow}, i) where {I <: Integer, P <: Pair{I}, F <: SIMDFunction{P}} =
    getfield(itr[i], :o0)
@inbounds @inline offset1(a::Constraint{F, I}, i) where {F, I <: AbstractVector{<:MergedRow}} =
    getfield(a.itr[i], :o1)
@inbounds @inline offset2(a::Constraint{F, I}, i) where {F, I <: AbstractVector{<:MergedRow}} =
    getfield(a.itr[i], :o2)
@inbounds @inline offset1(a::Objective{F, I}, i) where {F, I <: AbstractVector{<:MergedRow}} =
    getfield(a.itr[i], :o1)
@inbounds @inline offset2(a::Objective{F, I}, i) where {F, I <: AbstractVector{<:MergedRow}} =
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
# a dims-form constraint (`add_con(c, n1, n2)`, a `Null{Nothing}` tree) stays
# plain: it has no tree to share, and a family keeps only a flat row count
# where augmentations and writers need the block's own dims
_mergeable(::Type{Null{T}}) where {T} = T <: Real
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
# printed form inside expressions (the pretty-printer's fallback would show
# `ExaModels.IvRef(1)` at every hoisted leaf)
Base.show(io::IO, v::IvRef) = print(io, "⟨i", v.k, "⟩")
Base.show(io::IO, v::FvRef) = print(io, "⟨f", v.k, "⟩")
_short_type(::IvRef) = "IvRef"
_short_type(::FvRef) = "FvRef"
@inline (v::FvRef)(i, x, θ) = @inbounds getfield(i, _MROW_FV)[v.k]
@inline (v::FvRef)(i::Identity, x, θ) = eltype(θ)(NaN)

"""
    NullRef <: AbstractNode

Reads the `k`-th hoisted Float leaf the way a [`Null`](@ref) value node
would: a plain value under a value source, `AdjointNull`/`SecondAdjointNull`
under the adjoint sources. A `Null` leaf is hoisted into this node rather
than an [`FvRef`](@ref) because the two differ in adjoint shape under `*`:
`Null(c) * x[i]` builds `SecondAdjointNode2{*}(SecondAdjointNull, Var)`,
which `hrpass0` walks and writes one Hessian slot for, while a plain value
builds `SecondAdjointNode1{FirstFixed{*}}(Var)`, whose linear shortcut
writes none. The source block's compressors and steps come from its own
tree, so the hoisted tree must build the same adjoint nodes, or a slot is
accounted and never written.
"""
struct NullRef <: AbstractNode
    k::Int
end
Base.show(io::IO, v::NullRef) = print(io, "⟨n", v.k, "⟩")
_short_type(::NullRef) = "NullRef"
@inline (v::NullRef)(i, x, θ) = @inbounds getfield(i, _MROW_FV)[v.k]
@inline (v::NullRef)(i, x::AdjointNodeSource{T}, θ) where {T} =
    @inbounds AdjointNull(eltype(T)(getfield(i, _MROW_FV)[v.k]))
@inline (v::NullRef)(i, x::SecondAdjointNodeSource{T}, θ) where {T} =
    @inbounds SecondAdjointNull(eltype(T)(getfield(i, _MROW_FV)[v.k]))
@inline (v::NullRef)(i::Identity, x::AdjointNodeSource{T}, θ) where {T} = AdjointNull(eltype(T)(NaN))
@inline (v::NullRef)(i::Identity, x::SecondAdjointNodeSource{T}, θ) where {T} = SecondAdjointNull(eltype(T)(NaN))

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
# type stays small (a node plus two Ints) at any depth.  Every leaf position
# reads its own slot; which positions share a NONZERO is the segment's
# compressors' business, not the tree's.
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
# a Null's value is a scalar field like any other: hoist it, keeping Null's adjoint shape
@inline _stree(a::Null{T}, io::Int, fo::Int) where {T <: Real} = NullRef(fo + 1), 0, 1
@inline _stree(a::T, io::Int, fo::Int) where {T <: Integer} = IvRef(io + 1), 1, 0
@inline _stree(a::T, io::Int, fo::Int) where {T <: Real} = FvRef(fo + 1), 0, 1

# the slots.  Concatenating a tuple per node would mint one tuple TYPE per
# slot, and inference pays for each, so
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

@inline function _shoist(::Type{T}, b) where {T}
    return first(_stree(b, 0, 0)), _siv(b), _sfv(T, b)
end

# ── block kind ───────────────────────────────────────────────────────────────
# Constraints and objectives share the whole merge machinery and differ only
# in how a block is wrapped and whether it carries a tag.  Both are dispatch.

@inline _mkblock(::Type{Constraint}, mf, itr, n, tag) = Constraint(mf, itr, 0, (n,), tag)
@inline _mkblock(::Type{Objective}, mf, itr, n, tag) = Objective(mf, itr)
# rewrap an existing block around a grown iterator, keeping its own kind
@inline _rewrap(prev::Constraint, mf, itr, n, tag) = Constraint(mf, itr, 0, (n,), tag)
@inline _rewrap(prev::Objective, mf, itr, n, tag) = Objective(mf, itr)
# objectives have no tag, so every objective family is tag-compatible
@inline _blocktag(b::Constraint) = getfield(b, :tag)
@inline _blocktag(b::Objective) = nothing
@inline _blocktag(b) = nothing
# tags take part in the family decision by TYPE: a value comparison would make
# the core's type depend on tag values in concrete storage. Merging blocks
# that differ only in tag value is sound: the stored block's tag is read by
# nothing in src/, and each user handle keeps its own tag.
@inline _same_tag(a, b) = typeof(a) === typeof(b)

@inline _family_tree(f::SIMDFunction) = _pair_second(f.f)
@inline _pair_second(p::Pair) = p.second
@inline _pair_second(t) = t

# ── family membership: dispatch on the stored representative's type ─────────

# same representative type (tree type) and same data eltype: a type-level
# test, so it folds at inference and the core's type stays static
@inline _same_family(a, f, ::Type{E}) where {E} = false
@inline _same_family(a::Constraint{F2, I}, f::F, ::Type{E}) where {F2, F, E, I <: _MergedItr} =
    typeof(_mrep(getfield(a, :itr))) == F && _mrow_dtype(eltype(getfield(a, :itr))) == E
@inline _same_family(a::Objective{F2, I}, f::F, ::Type{E}) where {F2, F, E, I <: _MergedItr} =
    typeof(_mrep(getfield(a, :itr))) == F && _mrow_dtype(eltype(getfield(a, :itr))) == E

# ── block creation and appends ───────────────────────────────────────────────

# per-row o0 values of an arriving block, in iteration order (folds in Pair
# row targets and dims for pair-headed plain blocks)
function _row_o0s(f, pars, dims)
    h = Constraint(f, pars, 0, dims, nothing)
    return Int[offset0(h, r) for r in 1:length(pars)]
end

# First block of a family: a one-segment merged block.  The segment's bases,
# compressors and steps are the arriving block's own (f.o0/f.o1/f.o2 were
# taken from the core at this add; f.comp1/f.comp2/f.o1step/f.o2step came
# out of the sparsity probe over its original tree), so the caller accounts
# nnz with the block's own steps, as it would have unmerged.
@inline function _merged_first(::Type{T}, kind, f, pars, dims, tag, backend) where {T}
    rep = f
    tree, ivs, fvt = _shoist(T, _family_tree(rep))
    mf = _family_function(T, tree, rep)
    itr = _merged_first_itr(T, getfield(f, :f), backend, rep, f, pars, dims, ivs, fvt)
    return _mkblock(kind, mf, itr, length(pars), tag)
end

# The family's `SIMDFunction`: the merged tree, with every affine field zero.
# The derivative passes take a row's nonzero bases and compressors from the
# row element, never from here; the compressor fields hold the first block's
# (the right type, and the right values for a one-segment family), the steps
# are zero so that a generic affine reader would address slot 0 rather than
# a plausible wrong one.  No sparsity probe runs over the merged tree: a
# source block's compressors address the leaf positions of its own tree, and
# `_stree` keeps every position (it replaces leaf scalars with accessors and
# nothing else), so the layout agrees by construction.  A probe here was
# measured at 3.6 s of compilation per family on a 93-leaf tree, against
# 1.3 s for the source block's own probe; MergeTest's recipe-parity fixtures
# check the agreement instead.
@inline _family_function(::Type{T}, tree, rep) where {T} =
    SIMDFunction(replace_T(T, tree), rep.comp1, rep.comp2, 0, 0, 0, 0, 0)

@inline function _merged_first_itr(::Type{T}, head, backend, rep, f, pars, dims, ivs, fvt) where {T}
    seg = _mergedseg(T, f, pars, dims, ivs, fvt)
    K = length(ivs)
    S = length(fvt)
    return SegmentedItr{K, S, T, eltype(pars), typeof(pars), typeof(rep), typeof(f.comp1), typeof(f.comp2)}(
        rep, [seg], Int[length(pars)], length(pars)
    )
end
# pair heads (data-driven row targets) materialize rows; they merge on the
# host only (`_merge_allowed`)
@inline _merged_first_itr(::Type{T}, head::Pair, ::Nothing, rep, f, pars, dims, ivs, fvt) where {T} =
    _merged_rows(T, rep, f, pars, dims, ivs, fvt)

@inline function _merged_rows(::Type{T}, rep, f, pars, dims, ivs, fvt) where {T}
    itrc = collect(pars)
    rows = _block_rows(T, f, pars, dims, ivs, fvt, itrc)
    return MergedRows{length(ivs), length(fvt), T, eltype(itrc), typeof(rep), typeof(f.comp1), typeof(f.comp2)}(rep, rows)
end
# the materialized rows of one block, each with the block's own bases,
# compressors and steps
function _block_rows(::Type{T}, f, pars, dims, ivs::NTuple{K, Int}, fvt::NTuple{S, T}, itrc) where {K, S, T}
    o0s = _row_o0s(f, pars, dims)
    return [
        MergedRow{T, NTuple{K, Int}, NTuple{S, T}, eltype(itrc), typeof(f.comp1), typeof(f.comp2)}(
                o0s[r], f.o1 + f.o1step * (r - 1), f.o2 + f.o2step * (r - 1),
                ivs, fvt, itrc[r], f.comp1, f.comp2
            ) for r in eachindex(itrc)
    ]
end

# A later arrival: extract in the representative's layout, append one
# segment.  Bases, compressors and steps are again the arriving block's own.
@inline function _merged_append(::Type{T}, prev, f, pars, dims, tag) where {T}
    # same family means same tree type, so the arriving tree's own walk uses
    # the representative's slot layout by construction
    _, ivs, fvt = _shoist(T, _family_tree(f))
    return _merged_append_itr(getfield(prev, :itr), prev, f, pars, dims, tag, ivs, fvt)
end

# Copy-on-append: the core API is persistent, so the previous core — and any
# model already built from it — must keep the iterator it had. Appending in
# place would grow their block past the meta they were built with (out-of-
# bounds writes under @inbounds), and two cores branched off one base would
# see each other's segments. The copy is O(#segments) per add, so the cost
# still scales with the number of blocks, and the iterator's type does not
# change, so the core's type stays stable as a family grows.
@inline function _merged_append_itr(m::SegmentedItr{K, S, T}, prev, f, pars, dims, tag, ivs, fvt) where {K, S, T}
    segs = copy(m.segs)
    offs = copy(m.offs)
    push!(segs, _mergedseg(T, f, pars, dims, ivs, fvt))
    newlen = length(pars)
    push!(offs, offs[end] + newlen)
    seglen = (m.seglen == newlen) ? newlen : 0
    m2 = typeof(m)(m.rep, segs, offs, seglen)
    return _rewrap(prev, prev.f, m2, offs[end], _blocktag(prev))   # the family keeps its first block's tag
end
@inline function _merged_append_itr(m::MergedRows{K, S, T}, prev, f, pars, dims, tag, ivs, fvt) where {K, S, T}
    rows = copy(m.rows)
    Base.append!(rows, _block_rows(T, f, pars, dims, ivs, fvt, collect(pars)))   # ExaModels has its own `append!`
    m2 = typeof(m)(m.rep, rows)
    return _rewrap(prev, prev.f, m2, length(rows), _blocktag(prev))
end

# ── the merge entry point used by _add_con / _add_con! ───────────────────────
#
# Returns `nothing` when the block is statically non-mergeable (stored plain),
# or `(cons′, o1step, o2step)`: the updated block
# list and the per-row nnz steps to account nnzj/nnzh with.

# a block whose row/nonzero bases are deferred (an undeclared recipe after a
# deferred-length block makes `c.ncon`/`c.nnzj`/`c.nnzh` arg nodes) cannot be
# stored in a segment's Int fields; it stays plain and instantiates later
@inline _merge_offsets_ok(::SIMDFunction{F, C1, C2, Int, Int, Int}) where {F, C1, C2} = true
@inline _merge_offsets_ok(::SIMDFunction) = false

@inline function _merge_block(c::ExaCore{T}, f, pars, dims, tag) where {T}
    _mergeable(typeof(f.f)) || return nothing
    _merge_allowed(getfield(c, :nargs), getfield(f, :f), c.backend) || return nothing
    _merge_itr_ok(pars) || return nothing
    _merge_offsets_ok(f) || return nothing
    return _merge_result(
        _smerge(T, Constraint, c.cons, f, pars, dims, tag, eltype(pars)),
        T, Constraint, c, c.cons, f, pars, dims, tag
    )
end

# objectives: same machinery, `c.obj` instead of `c.cons`, no tag.  The nnz
# steps come back as (nnzg, nnzh) per row rather than (nnzj, nnzh).
@inline function _merge_obj(c::ExaCore{T}, f, pars) where {T}
    _mergeable(typeof(f.f)) || return nothing
    _merge_allowed(getfield(c, :nargs), getfield(f, :f), c.backend) || return nothing
    _merge_itr_ok(pars) || return nothing
    _merge_offsets_ok(f) || return nothing
    dims = (length(pars),)
    return _merge_result(
        _smerge(T, Objective, c.obj, f, pars, dims, nothing, eltype(pars)),
        T, Objective, c, c.obj, f, pars, dims, nothing
    )
end

# recipes (nargs > 0) are stored plain; pair-headed plain blocks (data-driven
# row targets) stay plain on device backends, where merging would move
# cross-block row collisions from sequential launches into one kernel
@inline _merge_allowed(::Val{0}, head, backend) = true
@inline _merge_allowed(::Val{0}, head::Pair, ::Nothing) = true
@inline _merge_allowed(::Val{0}, head::Pair, backend) = false
@inline _merge_allowed(nargs, head, backend) = false

# an existing family absorbed the block, or (on `nothing`) it starts one;
# either way the nnz steps are the block's own
@inline _merge_result(r, ::Type, kind, c, blocks, f, pars, dims, tag) = r, f.o1step, f.o2step
@inline function _merge_result(::Nothing, ::Type{T}, kind, c, blocks, f, pars, dims, tag) where {T}
    b = _merged_first(T, kind, f, pars, dims, tag, c.backend)
    return _prep(blocks, b), f.o1step, f.o2step
end

@inline _merge_itr_ok(pars) = _merge_itr_ok(Base.IteratorSize(pars))
# a deferred iterator (an undeclared recipe's `for d in arg.data`): its length
# and eltype are deferred nodes, so the block stays plain and is merged by
# nothing; `IteratorSize` alone would admit it
@inline _merge_itr_ok(::AbstractArgNode) = false
@inline _merge_itr_ok(::Union{Base.HasLength, Base.HasShape}) = true
@inline _merge_itr_ok(::Base.IteratorSize) = false

# scan + replace in one recursion; typed for tuple storage, dynamic for the
# Vector{Any} default; the same fold either way.  Returns the updated block
# list, or `nothing` when no family took the block.
@inline _smerge(::Type{T}, kind, cons::Tuple{}, f, pars, dims, tag, ::Type{E}) where {T, E} = nothing
@inline function _smerge(::Type{T}, kind, cons::Tuple, f, pars, dims, tag, ::Type{E}) where {T, E}
    b = first(cons)
    rest = Base.tail(cons)
    if _same_family(b, f, E) && _same_tag(_blocktag(b), tag) &&
            _appendable(getfield(b, :itr), pars)
        return (_merged_append(T, b, f, pars, dims, tag), rest...)
    end
    r = _smerge(T, kind, rest, f, pars, dims, tag, E)
    r === nothing && return nothing
    return (b, r...)
end
function _smerge(::Type{T}, kind, cons::Vector{Any}, f, pars, dims, tag, ::Type{E}) where {T, E}
    for (i, b) in enumerate(cons)
        if _same_family(b, f, E) && _same_tag(_blocktag(b), tag) &&
                _appendable(getfield(b, :itr), pars)
            out = copy(cons)
            out[i] = _merged_append(T, b, f, pars, dims, tag)
            return out
        end
    end
    return nothing
end
@inline _appendable(m::SegmentedItr{K, S, T, D, I}, pars::I2) where {K, S, T, D, I, I2} = I === I2
@inline _appendable(m::MergedRows, pars) = _merge_itr_ok(pars)
@inline _appendable(m, pars) = false

# ── model-build finalization ─────────────────────────────────────────────────
# `_finalize_merged` runs on a concretized core (tuple storage). On the host it
# is the identity. On a device backend each merged block's lazy iterator is
# materialized into a device element array, which the extension's kernels
# index directly; the map over the block tuples keeps the result inferable.

# a recipe's instantiated blocks are moved to the backend in `instantiate`
# (`_iters_to_backend`, nlp.jl); a merged family is not, since its layout is
# built from the host segments at finalization
_iter_to_backend(b::Constraint{F, I}, backend) where {F, I <: _MergedItr} = b
_iter_to_backend(b::Objective{F, I}, backend) where {F, I <: _MergedItr} = b

_materialize_block(backend, b) = b
_materialize_block(backend, b::Constraint{F, I}) where {F, I <: _MergedItr} =
    Constraint(b.f, _materialize_mergeditr(backend, getfield(b, :itr)), 0, b.size, b.tag)
_materialize_block(backend, b::Objective{F, I}) where {F, I <: _MergedItr} =
    Objective(b.f, _materialize_mergeditr(backend, getfield(b, :itr)))
# A one-segment family saves no launches on a device, while its merged layout
# reads every hoisted leaf from memory where the plain kernel has it as a
# constant (the CI benchmark read CUDA cons/jac/hess 1.1x to 1.25x of main
# that way). It goes back to the plain block `add_con`/`add_obj` would have
# stored: the source tree with its leaves restored, the source block's own
# compressors, steps and offsets, and its iterator converted as main does.
# No sparsity probe runs; the compressors address the restored tree's leaf
# positions, which `_unhoist` keeps. The host keeps the merged form.
function _materialize_block(backend, b::Constraint{F, I}) where {F, I <: SegmentedItr}
    s = getfield(b, :itr)
    length(s.segs) == 1 || return Constraint(b.f, _materialize_mergeditr(backend, s), 0, b.size, b.tag)
    seg = @inbounds s.segs[1]
    return Constraint(_plain_function(b, seg), convert_array(seg.itr, backend), seg.o0, seg.dims, b.tag)
end
function _materialize_block(backend, b::Objective{F, I}) where {F, I <: SegmentedItr}
    s = getfield(b, :itr)
    length(s.segs) == 1 || return Objective(b.f, _materialize_mergeditr(backend, s))
    seg = @inbounds s.segs[1]
    return Objective(_plain_function(b, seg), convert_array(seg.itr, backend))
end
@inline _plain_function(b, seg) = SIMDFunction(
    _unhoist(getfield(b, :f).f, seg.iv, seg.fv), seg.comp1, seg.comp2,
    seg.o0, seg.o1, seg.o2, seg.o1step, seg.o2step,
)

"""
    SegView{V}

A segment's hoisted leaves on a device: a slice (`base + 1 : base + K`) of
the layout's flat leaf vector `v`, read by [`IvRef`](@ref)/[`FvRef`](@ref)
with `view[k]`. A row holds this instead of a copy of the tuple: a dynamic
index into a tuple held by value in a kernel stages the whole tuple into
per-thread local memory (PTX `__local_depot` of 1016 bytes, 85 `ld.local`
per row, for the COPS dirichlet PDE block with `K = 94`, whose Jacobian
kernel then ran 4x slower than main's), while a read through the view is
one global load at an address uniform across the segment's threads.
"""
struct SegView{V}
    v::V
    base::Int
end
@inline Base.getindex(s::SegView, k::Int) = @inbounds s.v[s.base + k]

"""
    CompView{V}

A segment's nonzero compressor on a device: a slice of the layout's flat
compressor vector, callable like a `Compressor` (`comp(n)`), for the same
reason as [`SegView`](@ref).
"""
struct CompView{V}
    v::V
    base::Int
end
@inline (c::CompView)(n) = @inbounds c.v[c.base + n]

"""
    DeviceSegItr{T, D}

Device layout of a [`SegmentedItr`](@ref) with several segments: offsets and
steps once per segment (`seg_*`), the hoisted leaves and compressors of all
segments in flat vectors (`iv`, `fv`, `comp1`, `comp2`; `K`, `S`, `N1`, `N2`
entries per segment), the original iterator elements of all segments
concatenated (`d`), and the cumulative row counts (`offs`). `getindex`
assembles the `MergedRow` on the fly, with views into the flat vectors, so a
row costs its element and nothing else on the device.
"""
struct DeviceSegItr{
        T, D, VI <: AbstractVector{Int}, VT <: AbstractVector{T}, VD <: AbstractVector{D},
    } <: AbstractVector{MergedRow{T, SegView{VI}, SegView{VT}, D, CompView{VI}, CompView{VI}}}
    seg_o0::VI
    seg_o1::VI
    seg_o2::VI
    seg_o1step::VI
    seg_o2step::VI
    iv::VI                      # K Integer leaves per segment
    fv::VT                      # S Float leaves per segment
    comp1::VI                   # N1 compressor entries per segment
    comp2::VI                   # N2 compressor entries per segment
    offs::VI                    # cumulative row counts, one per segment
    d::VD                       # all segments' iterator elements, in order
    seg_dims::VI                # the segments' source shapes, concatenated (writers only)
    seg_dimptr::VI              # segment j's shape is seg_dims[seg_dimptr[j]:seg_dimptr[j+1]-1]
    K::Int
    S::Int
    N1::Int
    N2::Int
    seglen::Int                 # > 0 when all segments have equal length
    n::Int
end
Base.size(s::DeviceSegItr) = (s.n,)
Base.IndexStyle(::Type{<:DeviceSegItr}) = IndexLinear()
@inline function _segof(s::DeviceSegItr, i::Int)
    s.seglen > 0 && return div(i - 1, s.seglen) + 1
    # first segment whose cumulative count reaches i (binary search; no
    # allocation, runs inside device kernels)
    lo, hi = 1, length(s.offs)
    @inbounds while lo < hi
        mid = (lo + hi) >> 1
        s.offs[mid] < i ? (lo = mid + 1) : (hi = mid)
    end
    return lo
end
@inline function Base.getindex(s::DeviceSegItr, i::Int)
    j = _segof(s, i)
    base = j == 1 ? 0 : @inbounds s.offs[j - 1]
    r = i - base
    return @inbounds eltype(s)(
        s.seg_o0[j] + r,
        s.seg_o1[j] + s.seg_o1step[j] * (r - 1),
        s.seg_o2[j] + s.seg_o2step[j] * (r - 1),
        SegView(s.iv, (j - 1) * s.K), SegView(s.fv, (j - 1) * s.S), s.d[i],
        CompView(s.comp1, (j - 1) * s.N1), CompView(s.comp2, (j - 1) * s.N2),
    )
end
# kernels receive the device arrays through Adapt (KernelAbstractions adapts
# every kernel argument); the Int fields pass through. The parameters are
# written out so that an empty Float vector still names `T`.
@inline function _devsegitr(::Type{T}, ::Type{D}, o0, o1, o2, s1, s2, iv, fv, c1, c2, offs, d, dims, dimptr, K, S, N1, N2, seglen, n) where {T, D}
    return DeviceSegItr{T, D, typeof(o0), typeof(fv), typeof(d)}(
        o0, o1, o2, s1, s2, iv, fv, c1, c2, offs, d, dims, dimptr, K, S, N1, N2, seglen, n,
    )
end
function Adapt.adapt_structure(to, s::DeviceSegItr{T, D}) where {T, D}
    return _devsegitr(
        T, D, Adapt.adapt(to, s.seg_o0), Adapt.adapt(to, s.seg_o1), Adapt.adapt(to, s.seg_o2),
        Adapt.adapt(to, s.seg_o1step), Adapt.adapt(to, s.seg_o2step),
        Adapt.adapt(to, s.iv), Adapt.adapt(to, s.fv), Adapt.adapt(to, s.comp1), Adapt.adapt(to, s.comp2),
        Adapt.adapt(to, s.offs), Adapt.adapt(to, s.d), Adapt.adapt(to, s.seg_dims), Adapt.adapt(to, s.seg_dimptr),
        s.K, s.S, s.N1, s.N2, s.seglen, s.n,
    )
end

"""
    DeviceSeg1Itr{T, D}

Device layout of a single-segment [`SegmentedItr`](@ref): the segment's
offsets and steps are fields of the iterator itself (kernel parameter space),
its leaves and compressors are device vectors read through views, and the
iterator elements `d` are a device vector. The common case on a GPU: one
block per family, so no segment search per row.
"""
struct DeviceSeg1Itr{
        T, D, VI <: AbstractVector{Int}, VT <: AbstractVector{T}, VD <: AbstractVector{D},
    } <: AbstractVector{MergedRow{T, SegView{VI}, SegView{VT}, D, CompView{VI}, CompView{VI}}}
    o0::Int
    o1::Int
    o2::Int
    o1step::Int
    o2step::Int
    iv::VI
    fv::VT
    comp1::VI
    comp2::VI
    d::VD
    dims::VI                    # the source block's shape (writers only)
    n::Int
end
Base.size(s::DeviceSeg1Itr) = (s.n,)
Base.IndexStyle(::Type{<:DeviceSeg1Itr}) = IndexLinear()
@inline function Base.getindex(s::DeviceSeg1Itr, i::Int)
    return @inbounds eltype(s)(
        s.o0 + i, s.o1 + s.o1step * (i - 1), s.o2 + s.o2step * (i - 1),
        SegView(s.iv, 0), SegView(s.fv, 0), s.d[i], CompView(s.comp1, 0), CompView(s.comp2, 0),
    )
end
@inline _devseg1itr(::Type{T}, ::Type{D}, o0, o1, o2, s1, s2, iv, fv, c1, c2, d, dims, n) where {T, D} =
    DeviceSeg1Itr{T, D, typeof(iv), typeof(fv), typeof(d)}(o0, o1, o2, s1, s2, iv, fv, c1, c2, d, dims, n)
Adapt.adapt_structure(to, s::DeviceSeg1Itr{T, D}) where {T, D} =
    _devseg1itr(
    T, D, s.o0, s.o1, s.o2, s.o1step, s.o2step, Adapt.adapt(to, s.iv), Adapt.adapt(to, s.fv),
    Adapt.adapt(to, s.comp1), Adapt.adapt(to, s.comp2), Adapt.adapt(to, s.d), Adapt.adapt(to, s.dims), s.n,
)

const _DeviceMergedItr = Union{DeviceSegItr, DeviceSeg1Itr}
# host copy for the host-side passes that read every block's rows
# (`_build_subexpr_jac` and friends): adapt the arrays to `Array`, which is a
# bulk copy, where `Array(itr)` would index the device arrays element by element
_host_array(s::_DeviceMergedItr) = Adapt.adapt(Array, s)

# the layout's `T` is the Float leaves' eltype AFTER `convert_array`, so a
# backend without Float64 (Metal) rewrites it along with the arrays
function _materialize_mergeditr(backend, s::SegmentedItr{K, S, T, D}) where {K, S, T, D}
    segs = s.segs
    cv = v -> convert_array(v, backend)
    if length(segs) == 1
        seg = @inbounds segs[1]
        # a 2-D generator's iterator is a Matrix; the layout holds a vector, in the
        # same (column-major) order the host's linear getindex uses
        dd = cv(vec(collect(D, seg.itr)))   # may rewrite Float64 inside D (Metal)
        fv = cv(collect(T, seg.fv))
        return _devseg1itr(
            eltype(fv), eltype(dd), seg.o0, seg.o1, seg.o2, seg.o1step, seg.o2step,
            cv(collect(Int, seg.iv)), fv, cv(collect(Int, seg.comp1.inner)), cv(collect(Int, seg.comp2.inner)),
            dd, cv(_dims_code(seg.dims)), length(s),
        )
    end
    d = D[]
    for seg in segs
        Base.append!(d, seg.itr)   # ExaModels has its own `append!`
    end
    dd = cv(d)   # may rewrite Float64 inside D (Metal): the layout takes its D from the device array
    fv = cv(T[x for seg in segs for x in seg.fv])
    N1 = length(first(segs).comp1.inner)
    N2 = length(first(segs).comp2.inner)
    return _devsegitr(
        eltype(fv), eltype(dd),
        cv(Int[seg.o0 for seg in segs]), cv(Int[seg.o1 for seg in segs]), cv(Int[seg.o2 for seg in segs]),
        cv(Int[seg.o1step for seg in segs]), cv(Int[seg.o2step for seg in segs]),
        cv(Int[x for seg in segs for x in seg.iv]), fv,
        cv(Int[x for seg in segs for x in seg.comp1.inner]), cv(Int[x for seg in segs for x in seg.comp2.inner]),
        cv(copy(s.offs)), dd,
        cv(Int[x for seg in segs for x in _dims_code(seg.dims)]), cv(cumsum(Int[1; [3 * length(seg.dims) for seg in segs]])),
        K, S, N1, N2, s.seglen, length(s),
    )
end
_materialize_mergeditr(backend, m::MergedRows) = convert_array(m.rows, backend)

_finalize_merged(c::ExaCore) = _finalize_merged2(c, c.backend)
@inline _finalize_merged2(c::ExaCore, ::Nothing) = c
@inline _finalize_merged2(c::ExaCore, backend) = _with_blocks(
    c, c.var, c.par,
    map(b -> _materialize_block(backend, b), c.obj),
    map(b -> _materialize_block(backend, b), c.cons),
    c.refs,
)

# ── source-block view of a family (writers such as ExaModelsIO) ──────────────
# A family stores one merged tree plus per-segment leaf values. A writer that
# serializes expressions wants the source blocks back: for each segment, the
# merged tree with the segment's leaf values substituted for the slot
# references (the inverse of `_stree`), its own row offset and its own
# iterator. `expand_family` returns those plain blocks for a merged block and
# `[b]` for any other block.
@inline function _unhoist(a::Node1{F, I}, iv, fv) where {F, I}
    n = _unhoist(getfield(a, :inner), iv, fv)
    return Node1{F, typeof(n)}(n)
end
@inline function _unhoist(a::Node2{F, I1, I2}, iv, fv) where {F, I1, I2}
    n1 = _unhoist(getfield(a, :inner1), iv, fv)
    n2 = _unhoist(getfield(a, :inner2), iv, fv)
    return Node2{F, typeof(n1), typeof(n2)}(n1, n2)
end
@inline function _unhoist(a::Var, iv, fv)
    n = _unhoist(getfield(a, :i), iv, fv)
    return Var{typeof(n)}(n)
end
@inline function _unhoist(a::ParameterNode, iv, fv)
    n = _unhoist(getfield(a, :i), iv, fv)
    return ParameterNode{typeof(n)}(n)
end
@inline function _unhoist(a::DataIndexed{I, J}, iv, fv) where {I, J}
    inner = getfield(a, :inner)
    # `_stree` wrapped the element access as `DataIndexed(DataSource(), _MROW_D)`
    (inner isa DataSource && J === _MROW_D) && return DataSource()
    return DataIndexed(_unhoist(inner, iv, fv), J)
end
@inline _unhoist(a::IvRef, iv, fv) = @inbounds iv[a.k]
@inline _unhoist(a::FvRef, iv, fv) = @inbounds fv[a.k]
@inline _unhoist(a::NullRef, iv, fv) = @inbounds Null(fv[a.k])
@inline _unhoist(a, iv, fv) = a

# a block's size on a device layout: (start, length, is-a-range) per dimension,
# flat; a data axis (`for (i, t) in nodes`) travels as its length
_dim_code(d::Integer) = (1, Int(d), 0)
_dim_code(d::AbstractRange) = (Int(first(d)), length(d), 1)
_dim_code(d) = (1, length(d), 0)
_dims_code(dims) = Int[v for d in dims for v in _dim_code(d)]
_dims_decode(v::AbstractVector{Int}) = Tuple(
    v[3k] == 0 ? v[3k - 1] : (v[3k - 2]:(v[3k - 2] + v[3k - 1] - 1)) for k in 1:(length(v) ÷ 3)
)

# (o0, o1, o2, iv, fv, elements, dims) per segment, on the host
_family_segments(s::SegmentedItr) =
    [(seg.o0, seg.o1, seg.o2, seg.iv, seg.fv, collect(seg.itr), seg.dims) for seg in s.segs]
function _family_segments(s::DeviceSeg1Itr)
    h = _host_array(s)
    return [(h.o0, h.o1, h.o2, Tuple(h.iv), Tuple(h.fv), collect(h.d), _dims_decode(h.dims))]
end
function _family_segments(s::DeviceSegItr)
    h = _host_array(s)
    return [
        (
            h.seg_o0[j], h.seg_o1[j], h.seg_o2[j],
            Tuple(h.iv[((j - 1) * h.K + 1):(j * h.K)]), Tuple(h.fv[((j - 1) * h.S + 1):(j * h.S)]),
            collect(h.d[(j == 1 ? 0 : h.offs[j - 1]) + 1:h.offs[j]]),
            _dims_decode(h.seg_dims[h.seg_dimptr[j]:(h.seg_dimptr[j + 1] - 1)]),
        )
            for j in eachindex(h.offs)
    ]
end
_family_segments(::MergedRows) = throw(
    ArgumentError("a merged pair-headed block (`add_con(c, i => expr for ...)`) cannot be expanded into its source blocks"),
)

_mrow_T(::Type{MergedRow{T, IV, FV, D, C1, C2}}) where {T, IV, FV, D, C1, C2} = T
_is_family(b) = false
_is_family(::Constraint{F, I}) where {F, I <: Union{_MergedItr, _DeviceMergedItr}} = true
_is_family(::Objective{F, I}) where {F, I <: Union{_MergedItr, _DeviceMergedItr}} = true

"""
    expand_family(block)

The source blocks of a merged constraint or objective block, as plain blocks
(one per segment, each with its own tree, row offset, iterator and shape); `[block]`
for a block that is not a family. Used by writers that serialize expressions.
A pair-headed family (`MergedRows`, from `add_con(c, i => expr for ...)` on the
host) keeps no segments and throws an `ArgumentError`.
"""
function expand_family(b)
    _is_family(b) || return Any[b]
    T = _mrow_T(eltype(getfield(b, :itr)))
    out = Any[]
    for (o0, o1, o2, iv, fv, itr, dims) in _family_segments(getfield(b, :itr))
        f = _simdfunction(T, _unhoist(getfield(b, :f).f, iv, fv), o0, o1, o2)
        push!(out, _plain_block(b, f, itr, o0, dims))
    end
    return out
end
_plain_block(b::Constraint, f, itr, o0, dims) = Constraint(f, itr, o0, dims, b.tag)
_plain_block(b::Objective, f, itr, o0, dims) = Objective(f, itr)

# ── specialized hot loops for the lazy iterator ──────────────────────────────
# One affine @simd inner loop per segment, materializing the isbits row
# element on the fly: the same loop shape the unmerged blocks had.

function sjacobian!(y1, y2, f::Constraint{F, I}, x, θ, adj) where {F, I <: SegmentedItr}
    s = getfield(f, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = _segrow(s, seg, r)
            @inbounds @inline sjacobian!(y1, y2, f.f.f, el, x, θ, seg.comp1, el.o0, el.o1, adj)
        end
    end
    return
end

function shessian!(y1, y2, f::Union{Constraint{F, I}, Objective{F, I}}, x, θ, adj1, adj2) where {F, I <: SegmentedItr}
    s = getfield(f, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = _segrow(s, seg, r)
            @inbounds @inline shessian!(y1, y2, f.f.f, el, x, θ, seg.comp2, el.o2, adj1, adj2)
        end
    end
    return
end

function shessian!(y1, y2, f::Constraint{F, I}, x, θ, adj1s::V, adj2) where {F, I <: SegmentedItr, V <: AbstractVector}
    s = getfield(f, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = _segrow(s, seg, r)
            @inbounds @inline shessian!(y1, y2, f.f.f, el, x, θ, seg.comp2, el.o2, adj1s[el.o0], adj2)
        end
    end
    return
end

# merged objectives: the same per-segment loops for the value and the dense
# and sparse gradients (the generic drivers index the lazy iterator per row,
# which finds the segment and assembles the row element every time; the
# CI benchmark read the CPU objective 3x to 20x slower that way)
function _obj_block(obj::Objective{F, I}, x, θ) where {F, I <: SegmentedItr}
    s = getfield(obj, :itr)
    acc = zero(eltype(x))
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            acc += obj.f(_segrow(s, seg, r), x, θ)
        end
    end
    return acc
end
function gradient!(y, f::Objective{F, I}, x, θ, adj) where {F, I <: SegmentedItr}
    s = getfield(f, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            @inbounds @inline gradient!(y, f.f, x, θ, _segrow(s, seg, r), adj)
        end
    end
    return y
end
function sgradient!(y, f::Objective{F, I}, x, θ, adj) where {F, I <: SegmentedItr}
    s = getfield(f, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = _segrow(s, seg, r)
            @inbounds @inline sgradient!(y, f.f, el, x, θ, seg.comp1, el.o1, adj)
        end
    end
    return y
end

function _cons_rows!(g, con::Constraint{F, I}, x, θ) where {F, I <: SegmentedItr}
    s = getfield(con, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = _segrow(s, seg, r)
            @inbounds g[el.o0] += con.f(el, x, θ)
        end
    end
    return nothing
end
