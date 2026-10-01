# merge.jl — family merge at ExaModel(c) time (PROTOTYPE)
#
# Collapse every group of constraint blocks sharing one tree TYPE into a
# single block, whatever variables or scalar coefficients each block bakes
# into its tree.  The tree type fixes the structure; the values that differ
# between blocks (variable-index offsets, coefficients) are HOISTED into the
# iterator data, and each row carries its own row/nonzero offsets, so no row,
# bound, or sparsity position moves.  One block per constraint family means
# the callback tuple stays short regardless of how many units/scenarios/
# repeated patterns a model instantiates, and on GPU one kernel launch covers
# the whole family.
#
# Soundness posture: every step refuses (throws `_MergeRefuse`, caught per
# group) rather than guessing — a refused group simply stays unmerged, so the
# transform can make a model faster, never wrong.  Runs only for cores built
# with non-concrete storage; `concrete = Val(true)` (the juliac path) never
# reaches it.

struct _MergeRefuse <: Exception
    why::String
end
_MergeRefuse() = _MergeRefuse("?")

"""
    MergedRow{V, D}

One row of a merged constraint family: its global row offset `o0`, its
Jacobian/Hessian nonzero bases `o1`/`o2` (what the original block's affine
`offsetX` returned for this row), the tuple `v` of hoisted per-block field
values, and the original iterator element `d`.

The merged tree reads `v` and `d` through `DataIndexed` accessors built by
`_hoist` (fields 4 and 5 by position); the evaluation kernels read `o0`/`o1`/
`o2` through the `offset0/1/2` methods below.
"""
struct MergedRow{K, D}
    o0::Int
    o1::Int
    o2::Int
    iv::NTuple{K, Int}   # hoisted Integer leaves (feed variable indices, so
                         # they must be readable in structure passes, where
                         # θ is a NaNSource)
    pbase::Int           # this row's block's base into θ for hoisted Floats
    d::D                 # the original iterator element
end

const _MROW_IV = 4  # field position of `iv` in MergedRow
const _MROW_PB = 5  # field position of `pbase` in MergedRow
const _MROW_D  = 6  # field position of `d` in MergedRow

# ── per-row offsets for merged blocks ────────────────────────────────────────
# `offset0` flows through the (f, itr, i, dims) form; `offset1`/`offset2` are
# normally f-only (affine in i), so merged blocks override at Constraint level.
@inbounds @inline offset0(f::F, itr::AbstractVector{<:MergedRow}, i, dims) where {F <: SIMDFunction} =
    getfield(itr[i], :o0)
@inbounds @inline offset1(a::Constraint{F, I}, i) where {F, I <: AbstractVector{<:MergedRow}} =
    getfield(a.itr[i], :o1)
@inbounds @inline offset2(a::Constraint{F, I}, i) where {F, I <: AbstractVector{<:MergedRow}} =
    getfield(a.itr[i], :o2)

# ── lockstep hoisting walk ───────────────────────────────────────────────────
# `ns` holds the same node position from every block's tree (all of one type,
# guaranteed by grouping on `typeof(f)`).  Where the values agree the node is
# rebuilt as-is; where Real leaves differ, the per-block values go to `acc`
# and the leaf becomes an accessor into `MergedRow.v`; `DataSource` is
# re-rooted to `MergedRow.d`.  Anything unexpected refuses the whole group.

# Sharing must survive the rebuild: `add_expr` splicing leaves ===-identical
# subtrees in a tree, and the sparsity compressor (`_ident_unique`) dedups by
# egality — rebuilding a shared subtree into two distinct objects would
# multiply the per-row sparsity footprint (caught by the o1step guard, but as
# a refusal).  The memo maps each input node to its rebuild; a hit is reused
# only when EVERY block shares the same way (otherwise rebuilt fresh, and any
# resulting sparsity change still trips the o1step guard).
function _hoist_nodes(ns::Vector{Any}, acc, memo)
    key = ns[1]
    hit = get(memo, key, nothing)
    if hit !== nothing
        stored, built = hit
        all(k -> ns[k] === stored[k], eachindex(ns)) && return built
        return _hoist(ns[1], ns, acc, memo)
    end
    built = _hoist(ns[1], ns, acc, memo)
    memo[key] = (copy(ns), built)
    return built
end

function _hoist(n1::Node1{F, I}, ns, acc, memo) where {F, I}
    inner = _hoist_nodes(Any[getfield(n, :inner) for n in ns], acc, memo)
    return Node1{F, typeof(inner)}(inner)
end
function _hoist(n1::Node2{F, I1, I2}, ns, acc, memo) where {F, I1, I2}
    a = _hoist_nodes(Any[getfield(n, :inner1) for n in ns], acc, memo)
    b = _hoist_nodes(Any[getfield(n, :inner2) for n in ns], acc, memo)
    return Node2{F, typeof(a), typeof(b)}(a, b)
end
function _hoist(n1::Var, ns, acc, memo)
    i = _hoist_nodes(Any[getfield(n, :i) for n in ns], acc, memo)
    return Var{typeof(i)}(i)
end
function _hoist(n1::ParameterNode, ns, acc, memo)
    i = _hoist_nodes(Any[getfield(n, :i) for n in ns], acc, memo)
    return ParameterNode{typeof(i)}(i)
end
function _hoist(n1::DataIndexed{I, J}, ns, acc, memo) where {I, J}
    inner = _hoist_nodes(Any[getfield(n, :inner) for n in ns], acc, memo)
    return DataIndexed(inner, J)
end
# the original element moves to `MergedRow.d`
_hoist(n1::DataSource, ns, acc, memo) = DataIndexed(DataSource(), _MROW_D)
_hoist(n1::Constant, ns, acc, memo) = n1
_hoist(n1::VarSource, ns, acc, memo) = n1
_hoist(n1::ParameterSource, ns, acc, memo) = n1
# type-level constants (e.g. Val sizes hoisted by model code) carry no block-
# varying state: the shared type guarantees they agree
_hoist(n1::Val, ns, acc, memo) = n1
function _hoist(n1::Null{T}, ns, acc, memo) where {T}
    all(n -> getfield(n, :value) === getfield(n1, :value), ns) || throw(_MergeRefuse("Null values differ"))
    return n1
end
# EVERY Real leaf is hoisted, equal-valued or not: which slots exist is then
# read off the tree structure alone, so the merged block's TYPE is a pure
# function of the family's type and the compiled merge output is reused for
# any number of blocks and any data (Sungho, msg 173240/173248).  Integers
# become element-tuple slots (structure passes need them with θ = NaNSource);
# Floats become θ reads, stored once per BLOCK, addressed as θ[pbase + s].
function _hoist(n1::T, ns, acc, memo) where {T <: Integer}
    push!(acc.i, Any[ns...])
    return DataIndexed(DataIndexed(DataSource(), _MROW_IV), length(acc.i))
end
function _hoist(n1::T, ns, acc, memo) where {T <: Real}
    push!(acc.f, Any[ns...])
    return ParameterNode(Node2(+, DataIndexed(DataSource(), _MROW_PB), length(acc.f)))
end
# SumNode/ProdNode/Pair/ArgLeaf/anything else: refuse, stay unmerged.
_hoist(n1, ns, acc, memo) = throw(_MergeRefuse("node kind $(typeof(n1))"))

# ── group merge ──────────────────────────────────────────────────────────────

# Dispatch decides family membership: one tree type (and one iterator element
# type, so the data columns concatenate) = one family.  The fallback method is
# the "create a new block" answer.  Augmentations form their own families
# (their SIMDFunction types differ from any plain block's).
_same_family(a::Constraint{F, <:AbstractArray{E}}, b::Constraint{F, <:AbstractArray{E}}) where {F, E} = true
_same_family(a::ConstraintAugmentation{F, <:AbstractArray{E}}, b::ConstraintAugmentation{F, <:AbstractArray{E}}) where {F, E} = true
_same_family(a, b) = false

# An augmentation's tree is `idx => expr`: the row target `idx` is evaluated
# per row AT MERGE TIME (through the block's own `offset0`, which also folds
# in the base constraint's offset and dims), so only the `expr` half survives
# into the merged tree and the Pair disappears.
_hoist_root(ts::Vector{Any}, acc) = _hoist_root(ts[1], ts, acc, IdDict{Any, Any}())
_hoist_root(t1::Pair, ts, acc, memo) = _hoist_nodes(Any[p.second for p in ts], acc, memo)
_hoist_root(t1, ts, acc, memo) = _hoist_nodes(ts, acc, memo)

function _merge_group(::Type{T}, blocks::Vector{Any}, backend, θext::Vector{T}, θlen0::Int) where {T}
    acc = (i = Vector{Any}[], f = Vector{Any}[])
    tree = _hoist_root(Any[b.f.f for b in blocks], acc)
    mf = _simdfunction(T, tree, 0, 0, 0)
    # per-row o1/o2 bases below assume the merged tree has the same per-row
    # sparsity footprint as the originals
    (mf.o1step == blocks[1].f.o1step && mf.o2step == blocks[1].f.o2step) ||
        throw(_MergeRefuse("o1step/o2step mismatch: merged $(mf.o1step)/$(mf.o2step) vs $(blocks[1].f.o1step)/$(blocks[1].f.o2step)"))
    all(b -> b.tag === blocks[1].tag, blocks) || throw(_MergeRefuse("tags differ"))

    K = length(acc.i)
    S = length(acc.f)
    nrows = sum(b -> length(b.itr), blocks)
    θadd = T[]

    # Plain families (no Pair row targets) get the LAZY segmented iterator:
    # O(#blocks) memory, original iterators kept alive (a UnitRange stays a
    # UnitRange), rows materialized on the fly by the specialized loops.
    # Pair/augmentation families need per-row data-driven row targets, so
    # they keep the materialized array.
    lazy = !(blocks[1].f.f isa Pair) &&
        all(b -> typeof(b.itr) == typeof(blocks[1].itr), blocks)
    if lazy
        segs = MergedSeg{K, typeof(blocks[1].itr)}[]
        offs = Int[]
        tot = 0
        for (j, b) in enumerate(blocks)
            iv = ntuple(k -> Int(acc.i[k][j]), K)
            pbase = θlen0 + length(θext) + length(θadd)
            for v in (acc.f[s][j] for s in 1:S)
                push!(θadd, T(v))
            end
            f = b.f
            # plain blocks: offset0(b, r) == f.o0 + r, so the o0 base is f.o0
            push!(segs, MergedSeg(b.itr, f.o0, f.o1, f.o2, iv, pbase))
            tot += length(b.itr)
            push!(offs, tot)
        end
        uniform = all(sg -> length(sg.itr) == length(segs[1].itr), segs)
        itr = SegmentedItr{K, eltype(blocks[1].itr), typeof(blocks[1].itr)}(
            segs, offs, uniform ? length(segs[1].itr) : 0,
            blocks[1].f.o1step, blocks[1].f.o2step)
        con = Constraint(mf, itr, 0, (nrows,), blocks[1].tag)
        Base.append!(θext, θadd)
        return con
    end

    rows = Vector{MergedRow}(undef, 0)
    for (j, b) in enumerate(blocks)
        iv = ntuple(k -> Int(acc.i[k][j]), K)
        pbase = θlen0 + length(θext) + length(θadd)
        for v in (acc.f[s][j] for s in 1:S)
            push!(θadd, T(v))
        end
        f = b.f
        itr = collect(b.itr)
        for r in eachindex(itr)
            push!(rows, MergedRow(
                offset0(b, r),   # block-level: folds in Pair row targets + dims
                f.o1 + f.o1step * (r - 1),
                f.o2 + f.o2step * (r - 1),
                iv,
                pbase,
                itr[r],
            ))
        end
    end
    # tighten to the concrete element type so kernels specialize once
    rows_c = [r for r in rows]
    rows_t = convert(Vector{typeof(rows_c[1])}, rows_c)
    length(rows_t) == nrows || throw(_MergeRefuse("row count"))
    con = Constraint(mf, convert_array(rows_t, backend), 0, (nrows,), blocks[1].tag)
    Base.append!(θext, θadd)   # committed only on success
    return con
end

"""
    _merge_families(c::ExaCore)

Group the core's `Constraint` blocks by `(typeof(f), eltype(itr))` and merge
each multi-block group into one block via [`_merge_group`](@ref).  Blocks of
any other kind (augmentations, oracles' evals) and refused groups pass
through untouched, in their original relative order (each merged family sits
where its first member sat).
"""
function _merge_families(c::ExaCore{T}) where {T}
    cons = Any[b for b in c.cons]
    reps = Int[]            # first-seen member of each family
    members = Vector{Int}[]
    for (i, b) in enumerate(cons)
        b isa Union{Constraint, ConstraintAugmentation} || continue
        j = findfirst(r -> _same_family(cons[r], b), reps)
        if j === nothing
            push!(reps, i); push!(members, [i])
        else
            push!(members[j], i)
        end
    end
    merged = Dict{Int, Any}()   # first-member index => merged block
    drop = Set{Int}()
    θlen0 = length(c.θ)
    θext = T[]
    for idx in members
        length(idx) > 1 || continue
        try
            # cons is pushfirst-ordered (most recent first); merge in ADD order
            blocks = Any[cons[i] for i in reverse(idx)]
            m = _merge_group(T, blocks, c.backend, θext, θlen0)
            merged[idx[end]] = m          # idx[end] = earliest-added member
            for i in idx[1:(end-1)]
                push!(drop, i)
            end
        catch e
            e isa _MergeRefuse || rethrow()
            haskey(ENV, "EXAMODELS_MERGE_DEBUG") &&
                println("merge refused (", length(idx), " blocks): ", e.why)
        end
    end
    isempty(merged) && return c
    out = Any[]
    for (i, b) in enumerate(cons)
        i in drop && continue
        push!(out, get(merged, i, b))
    end
    θnew = isempty(θext) ? c.θ : convert_array(vcat(Vector(c.θ), θext), c.backend)
    # var/par blocks are offset metadata, not kernels: keeping them
    # type-erased keeps the MODEL type independent of how many variable
    # blocks (units) the flowsheet has, so the whole merged model's type is a
    # function of its constraint families alone.
    # refs likewise: the materialized NamedTuple puts every named block's
    # type (one set per unit) into the model's type.  The Vector form is
    # already served by the _refs_* accessors on models, so keep it erased.
    refs = getfield(c, :refs)
    erased_refs = refs isa NamedTuple ?
        Pair{Symbol, Any}[k => v for (k, v) in pairs(refs)] : refs
    return ExaCore(c; cons = Tuple(out), θ = θnew, npar = length(θnew),
                   var = Any[c.var...], par = Any[c.par...], refs = erased_refs)
end

# ── lazy segmented iterator (no per-row arrays) ──────────────────────────────
#
# Everything MergedRow stores per ROW is constant or affine per SEGMENT (one
# segment = one original block): o0/o1/o2 are affine in the local row, iv and
# pbase are per-block constants, and d is the original iterator's element.
# So for plain (non-Pair) families the merged iterator stores one descriptor
# per segment, keeps the original iterators (UnitRange stays a UnitRange),
# and materializes MergedRow on the fly: O(#blocks) memory, and the
# specialized loops below run one affine @simd inner loop per segment,
# which is the same loop shape the unmerged blocks had.

struct MergedSeg{K, I}
    itr::I
    o0::Int
    o1::Int
    o2::Int
    iv::NTuple{K, Int}
    pbase::Int
end

struct SegmentedItr{K, D, I} <: AbstractVector{MergedRow{K, D}}
    segs::Vector{MergedSeg{K, I}}
    offs::Vector{Int}    # cumulative row counts; offs[end] == length
    seglen::Int          # > 0 when all segments have equal length (O(1) lookup)
    o1step::Int
    o2step::Int
end
Base.size(s::SegmentedItr) = (isempty(s.offs) ? 0 : s.offs[end],)
@inline _segof(s::SegmentedItr, i::Int) =
    s.seglen > 0 ? div(i - 1, s.seglen) + 1 : searchsortedfirst(s.offs, i)
@inline function Base.getindex(s::SegmentedItr{K, D}, i::Int) where {K, D}
    j = _segof(s, i)
    seg = @inbounds s.segs[j]
    base = j == 1 ? 0 : @inbounds s.offs[j-1]
    r = i - base
    return MergedRow{K, D}(
        seg.o0 + r,
        seg.o1 + s.o1step * (r - 1),
        seg.o2 + s.o2step * (r - 1),
        seg.iv, seg.pbase, (@inbounds seg.itr[r]),
    )
end

# offset1/offset2 for segmented blocks go through getindex (lazy), same as
# the array form: the MergedRow methods above already cover both, since both
# containers are AbstractVector{<:MergedRow}.

# ── specialized hot loops: one affine @simd inner loop per segment ───────────

function sjacobian!(y1, y2, f::Constraint{F, I}, x, θ, adj) where {F, I <: SegmentedItr}
    s = f.itr
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = MergedRow(seg.o0 + r, seg.o1 + s.o1step * (r - 1),
                           seg.o2 + s.o2step * (r - 1), seg.iv, seg.pbase,
                           @inbounds seg.itr[r])
            @inbounds sjacobian!(y1, y2, f.f.f, el, x, θ, f.f.comp1,
                                 seg.o0 + r, seg.o1 + s.o1step * (r - 1), adj)
        end
    end
end

function shessian!(y1, y2, f::Constraint{F, I}, x, θ, adj1, adj2) where {F, I <: SegmentedItr}
    s = f.itr
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = MergedRow(seg.o0 + r, seg.o1 + s.o1step * (r - 1),
                           seg.o2 + s.o2step * (r - 1), seg.iv, seg.pbase,
                           @inbounds seg.itr[r])
            @inbounds shessian!(y1, y2, f.f.f, el, x, θ, f.f.comp2,
                                seg.o2 + s.o2step * (r - 1), adj1, adj2)
        end
    end
end

function shessian!(y1, y2, f::Constraint{F, I}, x, θ, adj1s::V, adj2) where {F, I <: SegmentedItr, V <: AbstractVector}
    s = f.itr
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = MergedRow(seg.o0 + r, seg.o1 + s.o1step * (r - 1),
                           seg.o2 + s.o2step * (r - 1), seg.iv, seg.pbase,
                           @inbounds seg.itr[r])
            @inbounds shessian!(y1, y2, f.f.f, el, x, θ, f.f.comp2,
                                seg.o2 + s.o2step * (r - 1), adj1s[seg.o0 + r], adj2)
        end
    end
end

function _cons_rows!(g, con::Constraint{F, I}, x, θ) where {F, I <: SegmentedItr}
    s = con.itr
    for j in eachindex(s.segs)
        seg = @inbounds s.segs[j]
        @simd for r in 1:length(seg.itr)
            el = MergedRow(seg.o0 + r, seg.o1 + s.o1step * (r - 1),
                           seg.o2 + s.o2step * (r - 1), seg.iv, seg.pbase,
                           @inbounds seg.itr[r])
            @inbounds g[seg.o0 + r] += con.f(el, x, θ)
        end
    end
end
