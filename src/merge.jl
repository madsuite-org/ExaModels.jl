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
struct MergedRow{V, D}
    o0::Int
    o1::Int
    o2::Int
    v::V
    d::D
end

const _MROW_V = 4  # field position of `v` in MergedRow
const _MROW_D = 5  # field position of `d` in MergedRow

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
function _hoist(n1::T, ns, acc, memo) where {T <: Real}
    all(n -> n === n1, ns) && return n1
    push!(acc, Any[ns...])
    return DataIndexed(DataIndexed(DataSource(), _MROW_V), length(acc))
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

function _merge_group(::Type{T}, blocks::Vector{Any}, backend) where {T}
    acc = Vector{Any}[]
    tree = _hoist_root(Any[b.f.f for b in blocks], acc)
    mf = _simdfunction(T, tree, 0, 0, 0)
    # per-row o1/o2 bases below assume the merged tree has the same per-row
    # sparsity footprint as the originals
    (mf.o1step == blocks[1].f.o1step && mf.o2step == blocks[1].f.o2step) ||
        throw(_MergeRefuse("o1step/o2step mismatch: merged $(mf.o1step)/$(mf.o2step) vs $(blocks[1].f.o1step)/$(blocks[1].f.o2step)"))
    all(b -> b.tag === blocks[1].tag, blocks) || throw(_MergeRefuse("tags differ"))

    nrows = sum(b -> length(b.itr), blocks)
    rows = Vector{MergedRow}(undef, 0)
    for (j, b) in enumerate(blocks)
        v = Tuple(acc[s][j] for s in eachindex(acc))
        f = b.f
        itr = collect(b.itr)
        for r in eachindex(itr)
            push!(rows, MergedRow(
                offset0(b, r),   # block-level: folds in Pair row targets + dims
                f.o1 + f.o1step * (r - 1),
                f.o2 + f.o2step * (r - 1),
                v,
                itr[r],
            ))
        end
    end
    # tighten to the concrete element type so kernels specialize once
    rows_c = [r for r in rows]
    rows_t = convert(Vector{typeof(rows_c[1])}, rows_c)
    length(rows_t) == nrows || throw(_MergeRefuse("row count"))
    return Constraint(mf, convert_array(rows_t, backend), 0, (nrows,), blocks[1].tag)
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
    for idx in members
        length(idx) > 1 || continue
        try
            # cons is pushfirst-ordered (most recent first); merge in ADD order
            blocks = Any[cons[i] for i in reverse(idx)]
            m = _merge_group(T, blocks, c.backend)
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
    return ExaCore(c; cons = Tuple(out))
end
