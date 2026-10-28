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
    MergedRow{K, S, T, D, C1, C2}

One row of a merged family: global row offset `o0`, Jacobian/Hessian nonzero
bases `o1`/`o2`, the `K` hoisted Integer leaves `iv`, the `S` hoisted Float
leaves `fv` (of type `T`, shared by every row of the segment), the original
iterator element `d`, and the source block's own nonzero compressors
`comp1`/`comp2` (shared by every row of the segment).  The derivative passes
read the compressors from the row (`_comp1`/`_comp2`), never from the
family's `SIMDFunction`, so each segment keeps the sparsity its plain block
had.
"""
struct MergedRow{K, S, T, D, C1, C2}
    o0::Int
    o1::Int
    o2::Int
    iv::NTuple{K, Int}
    fv::NTuple{S, T}
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
end
# the segment a block contributes: its own offsets, leaves, compressors and steps
@inline _mergedseg(::Type{T}, f, pars, ivs::NTuple{K, Int}, fvt::NTuple{S, T}) where {K, S, T} =
    MergedSeg{K, S, T, typeof(pars), typeof(f.comp1), typeof(f.comp2)}(
        pars, f.o0, f.o1, f.o2, ivs, fvt, f.comp1, f.comp2, f.o1step, f.o2step
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
struct SegmentedItr{K, S, T, D, I, RF, C1, C2} <: AbstractVector{MergedRow{K, S, T, D, C1, C2}}
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
struct MergedRows{K, S, T, D, RF, C1, C2} <: AbstractVector{MergedRow{K, S, T, D, C1, C2}}
    rep::RF
    rows::Vector{MergedRow{K, S, T, D, C1, C2}}
end
Base.size(m::MergedRows) = Base.size(m.rows)
Base.IndexStyle(::Type{<:MergedRows}) = IndexLinear()
@inline Base.getindex(m::MergedRows, i::Int) = @inbounds m.rows[i]

const _MergedItr = Union{SegmentedItr, MergedRows}

_mrep(s::SegmentedItr) = s.rep
_mrep(m::MergedRows) = m.rep
_mrow_dtype(::Type{MergedRow{K, S, T, D, C1, C2}}) where {K, S, T, D, C1, C2} = D

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
# a Null's value is a scalar field like any other: hoist it
@inline _stree(a::Null{T}, io::Int, fo::Int) where {T <: Real} = FvRef(fo + 1), 0, 1
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
# same Integer-slot sharing pattern: a value-level test, consulted only by the

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
# a plausible wrong one.  The probe over the merged tree runs once per family
# and serves one purpose: a source block's compressors address leaf positions
# of ITS tree, and the check confirms the hoisted tree has the same layout.
@inline function _family_function(::Type{T}, tree, rep) where {T}
    probe = _simdfunction(T, tree, 0, 0, 0)
    _same_layout(probe.comp1, rep.comp1)
    _same_layout(probe.comp2, rep.comp2)
    return SIMDFunction(probe.f, rep.comp1, rep.comp2, 0, 0, 0, 0, 0)
end
@inline _same_layout(::C, ::C) where {C} = nothing
# one literal: a `*` over many string pieces is a vararg call that `juliac --trim`
# cannot resolve, and this error sits on the model-building path
_same_layout(a, b) = error("internal error: a merged tree's leaf layout differs from its source block's")

# a plain (non-pair) head gets the lazy segmented iterator on every backend;
# a device backend materializes it into a per-segment device layout at
# finalization (`DeviceSegItr`)
@inline function _merged_first_itr(::Type{T}, head, backend, rep, f, pars, dims, ivs, fvt) where {T}
    seg = _mergedseg(T, f, pars, ivs, fvt)
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
        MergedRow{K, S, T, eltype(itrc), typeof(f.comp1), typeof(f.comp2)}(
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
    push!(segs, _mergedseg(T, f, pars, ivs, fvt))
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

@inline function _merge_block(c::ExaCore{T}, f, pars, dims, tag) where {T}
    _mergeable(typeof(f.f)) || return nothing
    _merge_allowed(getfield(c, :nargs), getfield(f, :f), c.backend) || return nothing
    _merge_itr_ok(pars) || return nothing
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

_materialize_block(backend, b) = b
_materialize_block(backend, b::Constraint{F, I}) where {F, I <: _MergedItr} =
    Constraint(b.f, _materialize_mergeditr(backend, getfield(b, :itr)), 0, b.size, b.tag)
_materialize_block(backend, b::Objective{F, I}) where {F, I <: _MergedItr} =
    Objective(b.f, _materialize_mergeditr(backend, getfield(b, :itr)))

"""
    DeviceSegItr{K, S, T, D}

Device layout of a [`SegmentedItr`](@ref): the hoisted leaves and offsets once
per segment (`seg_*`), the original iterator elements of all segments
concatenated (`d`), and the cumulative row counts (`offs`). `getindex`
assembles the `MergedRow` on the fly, so a row costs its element and nothing
else on the device; a `MergedRow` per row would carry its own copy of the `K`
Integer and `S` Float leaves (880 bytes per row against 96 at `K = 94`).
"""
struct DeviceSegItr{
        K, S, T, D, C1, C2, VI <: AbstractVector{Int}, VIV <: AbstractVector{NTuple{K, Int}},
        VFV <: AbstractVector{NTuple{S, T}}, VC1 <: AbstractVector{C1}, VC2 <: AbstractVector{C2},
        VD <: AbstractVector{D},
    } <: AbstractVector{MergedRow{K, S, T, D, C1, C2}}
    seg_o0::VI
    seg_o1::VI
    seg_o2::VI
    seg_iv::VIV
    seg_fv::VFV
    seg_comp1::VC1
    seg_comp2::VC2
    seg_o1step::VI
    seg_o2step::VI
    offs::VI                    # cumulative row counts, one per segment
    d::VD                       # all segments' iterator elements, in order
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
        s.seg_iv[j], s.seg_fv[j], s.d[i], s.seg_comp1[j], s.seg_comp2[j],
    )
end
# kernels receive the device arrays through Adapt (KernelAbstractions adapts
# every kernel argument); the Int fields pass through. The parameters are
# written out: with no Float slots the empty `seg_fv` eltype leaves `T`
# undetermined for the implicit constructor.
@inline function _devsegitr(K::Int, S::Int, ::Type{T}, ::Type{D}, o0, o1, o2, iv, fv, c1, c2, s1, s2, offs, d, seglen, n) where {T, D}
    return DeviceSegItr{K, S, T, D, eltype(c1), eltype(c2), typeof(o0), typeof(iv), typeof(fv), typeof(c1), typeof(c2), typeof(d)}(
        o0, o1, o2, iv, fv, c1, c2, s1, s2, offs, d, seglen, n,
    )
end
function Adapt.adapt_structure(to, s::DeviceSegItr{K, S, T, D}) where {K, S, T, D}
    return _devsegitr(
        K, S, T, D, Adapt.adapt(to, s.seg_o0), Adapt.adapt(to, s.seg_o1), Adapt.adapt(to, s.seg_o2),
        Adapt.adapt(to, s.seg_iv), Adapt.adapt(to, s.seg_fv),
        Adapt.adapt(to, s.seg_comp1), Adapt.adapt(to, s.seg_comp2),
        Adapt.adapt(to, s.seg_o1step), Adapt.adapt(to, s.seg_o2step),
        Adapt.adapt(to, s.offs), Adapt.adapt(to, s.d), s.seglen, s.n,
    )
end

"""
    DeviceSeg1Itr{K, S, T, D}

Device layout of a single-segment [`SegmentedItr`](@ref): the segment's
offsets and hoisted leaves are fields of the iterator itself, so they reach a
kernel inside its argument struct (parameter space, uniform across threads),
the way the leaf values inside a plain block's tree do; only the iterator
elements `d` are a device array. A per-thread load of `K` Integer leaves from
a segment table (the [`DeviceSegItr`](@ref) layout) cost the COPS dirichlet
PDE block (`K = 94`) about 2x in `jac_coord!`/`hess_coord!` against main.
"""
struct DeviceSeg1Itr{K, S, T, D, C1, C2, VD <: AbstractVector{D}} <: AbstractVector{MergedRow{K, S, T, D, C1, C2}}
    o0::Int
    o1::Int
    o2::Int
    iv::NTuple{K, Int}
    fv::NTuple{S, T}
    d::VD
    comp1::C1
    comp2::C2
    o1step::Int
    o2step::Int
    n::Int
end
Base.size(s::DeviceSeg1Itr) = (s.n,)
Base.IndexStyle(::Type{<:DeviceSeg1Itr}) = IndexLinear()
@inline function Base.getindex(s::DeviceSeg1Itr, i::Int)
    return @inbounds eltype(s)(
        s.o0 + i, s.o1 + s.o1step * (i - 1), s.o2 + s.o2step * (i - 1), s.iv, s.fv, s.d[i],
        s.comp1, s.comp2,
    )
end
@inline _devseg1itr(::Type{T}, ::Type{D}, o0, o1, o2, iv::NTuple{K, Int}, fv::NTuple{S, T}, d, c1, c2, o1step, o2step, n) where {K, S, T, D} =
    DeviceSeg1Itr{K, S, T, D, typeof(c1), typeof(c2), typeof(d)}(o0, o1, o2, iv, fv, d, c1, c2, o1step, o2step, n)
Adapt.adapt_structure(to, s::DeviceSeg1Itr{K, S, T, D}) where {K, S, T, D} =
    _devseg1itr(T, D, s.o0, s.o1, s.o2, s.iv, s.fv, Adapt.adapt(to, s.d), s.comp1, s.comp2, s.o1step, s.o2step, s.n)

const _DeviceMergedItr = Union{DeviceSegItr, DeviceSeg1Itr}
# host copy for the host-side passes that read every block's rows
# (`_build_subexpr_jac` and friends): adapt the arrays to `Array`, which is a
# bulk copy, where `Array(itr)` would index the device arrays element by element
_host_array(s::_DeviceMergedItr) = Adapt.adapt(Array, s)
# the Float-slot tuple travels as a field, so a backend without Float64 (Metal)
# gets it rewritten here; `convert_array` only sees arrays
@inline _fvt(::Type{T}, fv::NTuple{S, T}, backend) where {S, T} = fv

function _materialize_mergeditr(backend, s::SegmentedItr{K, S, T, D}) where {K, S, T, D}
    segs = s.segs
    if length(segs) == 1
        seg = @inbounds segs[1]
        # a 2-D generator's iterator is a Matrix; the layout holds a vector, in the
        # same (column-major) order the host's linear getindex uses
        dd = convert_array(vec(collect(D, seg.itr)), backend)   # may rewrite Float64 inside D (Metal)
        return _devseg1itr(
            T, eltype(dd), seg.o0, seg.o1, seg.o2, seg.iv, _fvt(T, seg.fv, backend),
            dd, seg.comp1, seg.comp2, seg.o1step, seg.o2step, length(s),
        )
    end
    cv = v -> convert_array(v, backend)
    d = D[]
    for seg in segs
        Base.append!(d, seg.itr)   # ExaModels has its own `append!`
    end
    dd = cv(d)   # may rewrite Float64 inside D (Metal): the layout takes its D from the device array
    return _devsegitr(
        K, S, T, eltype(dd),
        cv(Int[seg.o0 for seg in segs]), cv(Int[seg.o1 for seg in segs]), cv(Int[seg.o2 for seg in segs]),
        cv(NTuple{K, Int}[seg.iv for seg in segs]), cv(NTuple{S, T}[seg.fv for seg in segs]),
        cv([seg.comp1 for seg in segs]), cv([seg.comp2 for seg in segs]),
        cv(Int[seg.o1step for seg in segs]), cv(Int[seg.o2step for seg in segs]),
        cv(copy(s.offs)), dd, s.seglen, length(s),
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
@inline _unhoist(a, iv, fv) = a

# (o0, o1, o2, iv, fv, elements) per segment, on the host
_family_segments(s::SegmentedItr) =
    [(seg.o0, seg.o1, seg.o2, seg.iv, seg.fv, collect(seg.itr)) for seg in s.segs]
function _family_segments(s::DeviceSeg1Itr)
    h = _host_array(s)
    return [(h.o0, h.o1, h.o2, h.iv, h.fv, collect(h.d))]
end
function _family_segments(s::DeviceSegItr)
    h = _host_array(s)
    return [
        (h.seg_o0[j], h.seg_o1[j], h.seg_o2[j], h.seg_iv[j], h.seg_fv[j],
            collect(h.d[(j == 1 ? 0 : h.offs[j - 1]) + 1:h.offs[j]]))
            for j in eachindex(h.offs)
    ]
end
_family_segments(::MergedRows) = throw(
    ArgumentError("a merged pair-headed block (`add_con(c, i => expr for ...)`) cannot be expanded into its source blocks"),
)

_mrow_T(::Type{MergedRow{K, S, T, D, C1, C2}}) where {K, S, T, D, C1, C2} = T
_is_family(b) = false
_is_family(::Constraint{F, I}) where {F, I <: Union{_MergedItr, _DeviceMergedItr}} = true
_is_family(::Objective{F, I}) where {F, I <: Union{_MergedItr, _DeviceMergedItr}} = true

"""
    expand_family(block)

The source blocks of a merged constraint or objective block, as plain blocks
(one per segment, each with its own tree, row offset and iterator); `[block]`
for a block that is not a family. Used by writers that serialize expressions.
"""
function expand_family(b)
    _is_family(b) || return Any[b]
    T = _mrow_T(eltype(getfield(b, :itr)))
    out = Any[]
    for (o0, o1, o2, iv, fv, itr) in _family_segments(getfield(b, :itr))
        f = _simdfunction(T, _unhoist(getfield(b, :f).f, iv, fv), o0, o1, o2)
        push!(out, _plain_block(b, f, itr, o0))
    end
    return out
end
_plain_block(b::Constraint, f, itr, o0) = Constraint(f, itr, o0, (length(itr),), b.tag)
_plain_block(b::Objective, f, itr, o0) = Objective(f, itr)

# ── specialized hot loops for the lazy iterator ──────────────────────────────
# One affine @simd inner loop per segment, materializing the isbits row
# element on the fly: the same loop shape the unmerged blocks had.

function sjacobian!(y1, y2, f::Constraint{F, I}, x, θ, adj) where {F, I <: SegmentedItr}
    s = getfield(f, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = _segrow(s, seg, r)
            @inbounds sjacobian!(y1, y2, f.f.f, el, x, θ, seg.comp1, el.o0, el.o1, adj)
        end
    end
    return
end

function shessian!(y1, y2, f::Constraint{F, I}, x, θ, adj1, adj2) where {F, I <: SegmentedItr}
    s = getfield(f, :itr)
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = _segrow(s, seg, r)
            @inbounds shessian!(y1, y2, f.f.f, el, x, θ, seg.comp2, el.o2, adj1, adj2)
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
            @inbounds shessian!(y1, y2, f.f.f, el, x, θ, seg.comp2, el.o2, adj1s[el.o0], adj2)
        end
    end
    return
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
