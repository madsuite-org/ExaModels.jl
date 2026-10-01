# ── Buffered subexpressions ───────────────────────────────────────────────────
#
# A buffered subexpression is evaluated ONCE per element into a dedicated slot
# of the model's θ vector (a "computed parameter"), and consumers reference the
# slot through a `SubexprNode` leaf instead of splicing the subexpression tree
# into their own.  Consumer tree types stay shallow, so compilation cost is
# additive (one kernel per stage) rather than multiplicative in nesting depth.
#
# First-order only for now: objective gradients flow through an adjoint buffer
# (`GradTarget`); Jacobian/Hessian support is guarded with errors at the
# NLPModels entry points (see nlp.jl).

"""
    SubexprNode{I} <: AbstractNode

Leaf node referencing the buffered value of a subexpression element, stored at
`θ[i]`.  In primal mode it reads the buffer directly; in first-order adjoint
mode it produces an [`AdjointNodeSubexpr`](@ref) leaf so the reverse pass can
accumulate the adjoint into the subexpression adjoint buffer, from which the
subexpression's own reverse pass is later seeded.
"""
struct SubexprNode{I} <: AbstractNode
    i::I
end

"""
    AdjointNodeSubexpr{I, T} <: AbstractAdjointNode

Gradient-pass leaf for a buffered subexpression element.

# Fields
- `i::I`: θ-slot index of the subexpression element (also the slot in the
  adjoint buffer)
- `x::T`: buffered primal value `θ[i]`
"""
struct AdjointNodeSubexpr{I,T} <: AbstractAdjointNode
    i::I
    x::T
end

# Primal evaluation reads the buffer.  Index expressions contain only
# data/constant leaves, so they evaluate to integers in every mode.
@inline (v::SubexprNode{I})(i, x, θ) where {I<:AbstractNode} =
    @inbounds θ[v.i(i, x, θ)]
@inline (v::SubexprNode{I})(i, x, θ) where {I} = @inbounds θ[v.i]

# First-order adjoint mode: produce the leaf carrying the slot and the value.
@inline function (v::SubexprNode{I})(i, x::AdjointNodeSource, θ) where {I<:AbstractNode}
    j = v.i(i, x, θ)
    return @inbounds AdjointNodeSubexpr(j, θ[j])
end
@inline (v::SubexprNode{I})(i, x::AdjointNodeSource, θ) where {I} =
    @inbounds AdjointNodeSubexpr(v.i, θ[v.i])

# Identity probe (SIMDFunction construction): store the NODE itself as the
# index so identity-based slot deduplication distinguishes distinct leaf sites
# (mirroring Var, whose probe stores the index expression).  Evaluating the
# index here would yield NaN (or a plain Int colliding with Var indices) and
# conflate distinct slots.
@inline (v::SubexprNode{I})(i::Identity, x::AdjointNodeSource, θ) where {I<:AbstractNode} =
    AdjointNodeSubexpr(v, eltype(θ)(NaN))
@inline (v::SubexprNode{I})(i::Identity, x::AdjointNodeSource, θ) where {I} =
    AdjointNodeSubexpr(v, eltype(θ)(NaN))

"""
    SecondAdjointNodeSubexpr{I, T} <: AbstractSecondAdjointNode

Hessian-pass leaf for a buffered subexpression element, treated as an extra
coordinate of the extended (x, s) space — the second-order analogue of
[`AdjointNodeSubexpr`](@ref).  Structure passes mark its coordinate with a
negated θ-slot; build-time elimination composes the entries with the stage
Jacobians and stage Hessians.
"""
struct SecondAdjointNodeSubexpr{I,T} <: AbstractSecondAdjointNode
    i::I
    x::T
end

@inline function (v::SubexprNode{I})(i, x::SecondAdjointNodeSource, θ) where {I<:AbstractNode}
    j = v.i(i, x, θ)
    return @inbounds SecondAdjointNodeSubexpr(j, θ[j])
end
@inline (v::SubexprNode{I})(i, x::SecondAdjointNodeSource, θ) where {I} =
    @inbounds SecondAdjointNodeSubexpr(v.i, θ[v.i])
# Identity probe: keep the node for identity-based pair deduplication.
@inline (v::SubexprNode{I})(i::Identity, x::SecondAdjointNodeSource, θ) where {I<:AbstractNode} =
    SecondAdjointNodeSubexpr(v, eltype(θ)(NaN))
@inline (v::SubexprNode{I})(i::Identity, x::SecondAdjointNodeSource, θ) where {I} =
    SecondAdjointNodeSubexpr(v, eltype(θ)(NaN))

"""
    GradTarget{Y, A}

Carrier threaded through `drpass` in place of the plain gradient vector when
the model has buffered subexpressions.  Variable leaves accumulate into the
dense gradient `y`; subexpression leaves accumulate into the adjoint buffer
`abuf` (θ-length), which later seeds the stages' own reverse passes.

A dedicated struct (rather than a tuple) is used because `drpass`'s
`AdjointNodeVar` leaf method indexes its target with `y[d.i]`, and a tuple
would accept that indexing silently.
"""
struct GradTarget{Y,A}
    y::Y
    abuf::A
end
# The gradient driver seeds the reverse pass with one(eltype(target)).
Base.eltype(::Type{GradTarget{Y,A}}) where {Y,A} = eltype(Y)

@inline function drpass(d::D, t::GradTarget, adj) where {D<:AdjointNodeVar}
    @inbounds t.y[d.i] += adj
    nothing
end
# Seed-only sweeps (Hessian stage seeds) discard variable-leaf contributions.
@inline drpass(d::D, t::GradTarget{Nothing,A}, adj) where {D<:AdjointNodeVar,A} = nothing
@inline function drpass(d::D, t::GradTarget, adj) where {D<:AdjointNodeSubexpr}
    @inbounds t.abuf[d.i] += adj
    nothing
end
# A buffered leaf reached without a GradTarget means a driver was not routed
# through the buffered-gradient path — fail loudly rather than corrupt.
@inline drpass(d::D, y, adj) where {D<:AdjointNodeSubexpr} = throw(
    ArgumentError(
        "buffered subexpression leaf reached by a non-buffered reverse pass",
    ),
)

# First-order sparsity: a buffered leaf occupies one slot in the extended
# (variables ∪ subexpression elements) coordinate space, mirroring
# AdjointNodeVar.  Structure passes mark the subexpression coordinate with a
# NEGATED column (-θslot); the build-time elimination in _build_subexpr_jac
# rewrites those entries into x-columns using the stage Jacobians.
@inline function grpass(d::D, comp::Nothing, y, o1, cnt::Vector, adj) where {D<:AdjointNodeSubexpr}
    push!(cnt, d.i)
    return cnt
end
@inline function grpass(
    d::D,
    comp::Nothing,
    y,
    o1,
    cnt::Tuple{<:Tuple,<:Tuple},
    adj,
) where {D<:AdjointNodeSubexpr}
    mapping, uniques = cnt
    idx = _grpass_find_ident(d.i, uniques, 1)
    if idx === 0
        return ((mapping..., length(uniques) + 1), (uniques..., d.i))
    else
        return ((mapping..., idx), uniques)
    end
end

# Value mode: write the local adjoint into the extended slot (like a Var).
@inline function jrpass(d::D, comp, i, y1, y2, o1, cnt, adj) where {D<:AdjointNodeSubexpr}
    @inbounds y1[o1+comp(cnt+=1)] += adj
    return cnt
end
# Structure mode: record the row and the NEGATED θ-slot as the column marker.
@inline function jrpass(
    d::D,
    comp,
    i,
    y1::V,
    y2::V,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeSubexpr,I<:Integer,V<:AbstractVector{I}}
    ind = o1 + comp(cnt += 1)
    @inbounds y1[ind] = i
    @inbounds y2[ind] = -d.i
    return cnt
end

_gr_val(::Type{<:AdjointNodeSubexpr}) = Val(1)

"""
    SubexprStage{F, I}

One buffered-subexpression evaluation stage.  The forward pass writes
`θ[offset + k] = f(itr[k], x, θ)` for each element `k`; the reverse pass
re-evaluates the (shallow) stage tree in adjoint mode and propagates the
buffered adjoint seed down to variable leaves and deeper stages.

# Fields
- `f::F`: the stage `SIMDFunction`
- `itr::I`: the collected iterator
- `offset::Int`: θ offset; values live at `θ[offset+1 : offset+length(itr)]`
"""
struct SubexprStage{F,I}
    f::F
    itr::I
    offset::Int
end

# Stage tuples are built newest-first (prepended, like `cons`/`objs`), and a
# stage can only reference stages added before it.  Forward sync therefore
# recurses tail-first (oldest first); the reverse pass runs head-first.
_sync_subexprs!(stages::Tuple{}, x, θ) = nothing
@inline function _sync_subexprs!(stages::Tuple, x, θ)
    _sync_subexprs!(Base.tail(stages), x, θ)
    s = first(stages)
    @simd for k in eachindex(s.itr)
        @inbounds θ[s.offset+k] = s.f(s.itr[k], x, θ)
    end
    return nothing
end

# Adjoint buffer sized like θ (only the subexpression slots are used); empty
# when the model has no buffered subexpressions so no memory is wasted there.
_make_abuf(::Tuple{}, θ) = similar(θ, 0)
_make_abuf(::Tuple, θ) = fill!(similar(θ), zero(eltype(θ)))

# ── Second-order pass methods for SecondAdjointNodeSubexpr ───────────────────
# Mirrors every SecondAdjointNodeVar method in hessian.jl.  In structure modes
# the subexpression coordinate is written NEGATED (-θslot); mixed var/subexpr
# pairs are canonicalized as (var, -slot) so the build-time expansion can
# classify entries by sign alone.

# Recursion combos: subexpr leaf against interior nodes.
@inline function hdrpass(t1::T1, t2::T2, comp, y1, y2, o2, cnt, adj) where {T1<:SecondAdjointNodeSubexpr,T2<:SecondAdjointNode1}
    return hdrpass(t1, t2.inner, comp, y1, y2, o2, cnt, adj * t2.y)
end
@inline function hdrpass(t1::SecondAdjointNodeSubexpr, t2::SecondAdjointNode1, comp::Nothing, y1, y2, o2, cnt, adj)  # despecialized
    return hdrpass(t1, t2.inner, comp, y1, y2, o2, cnt, adj * t2.y)
end
@inline function hdrpass(t1::T1, t2::T2, comp, y1, y2, o2, cnt, adj) where {T1<:SecondAdjointNode1,T2<:SecondAdjointNodeSubexpr}
    return hdrpass(t1.inner, t2, comp, y1, y2, o2, cnt, adj * t1.y)
end
@inline function hdrpass(t1::SecondAdjointNode1, t2::SecondAdjointNodeSubexpr, comp::Nothing, y1, y2, o2, cnt, adj)  # despecialized
    return hdrpass(t1.inner, t2, comp, y1, y2, o2, cnt, adj * t1.y)
end
@inline function hdrpass(t1::T1, t2::T2, comp, y1, y2, o2, cnt, adj) where {T1<:SecondAdjointNodeSubexpr,T2<:SecondAdjointNode2}
    cnt = hdrpass(t1, t2.inner1, comp, y1, y2, o2, cnt, adj * t2.y1)
    cnt = hdrpass(t1, t2.inner2, comp, y1, y2, o2, cnt, adj * t2.y2)
    return cnt
end
@inline function hdrpass(t1::SecondAdjointNodeSubexpr, t2::SecondAdjointNode2, comp::Nothing, y1, y2, o2, cnt, adj)  # despecialized
    cnt = hdrpass(t1, t2.inner1, comp, y1, y2, o2, cnt, adj * t2.y1)
    cnt = hdrpass(t1, t2.inner2, comp, y1, y2, o2, cnt, adj * t2.y2)
    return cnt
end
@inline function hdrpass(t1::T1, t2::T2, comp, y1, y2, o2, cnt, adj) where {T1<:SecondAdjointNode2,T2<:SecondAdjointNodeSubexpr}
    cnt = hdrpass(t1.inner1, t2, comp, y1, y2, o2, cnt, adj * t1.y1)
    cnt = hdrpass(t1.inner2, t2, comp, y1, y2, o2, cnt, adj * t1.y2)
    return cnt
end
@inline function hdrpass(t1::SecondAdjointNode2, t2::SecondAdjointNodeSubexpr, comp::Nothing, y1, y2, o2, cnt, adj)  # despecialized
    cnt = hdrpass(t1.inner1, t2, comp, y1, y2, o2, cnt, adj * t1.y1)
    cnt = hdrpass(t1.inner2, t2, comp, y1, y2, o2, cnt, adj * t1.y2)
    return cnt
end

# Leaf pairs: value mode.
@inline function hdrpass(t1::T1, t2::T2, comp, y1, y2, o2, cnt, adj) where {T1<:SecondAdjointNodeSubexpr,T2<:SecondAdjointNodeSubexpr}
    i, j = t1.i, t2.i
    @inbounds if i == j
        y1[o2+comp(cnt+=1)] += 2 * adj
    else
        y1[o2+comp(cnt+=1)] += adj
    end
    return cnt
end
@inline function hdrpass(t1::T1, t2::T2, comp, y1, y2, o2, cnt, adj) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNodeSubexpr}
    @inbounds y1[o2+comp(cnt+=1)] += adj
    return cnt
end
@inline function hdrpass(t1::T1, t2::T2, comp, y1, y2, o2, cnt, adj) where {T1<:SecondAdjointNodeSubexpr,T2<:SecondAdjointNodeVar}
    @inbounds y1[o2+comp(cnt+=1)] += adj
    return cnt
end

# Leaf pairs: structure mode (negated subexpression coordinates; x-coordinate
# first for mixed pairs).
@inline function hdrpass(t1::T1, t2::T2, comp, y1::V, y2::V, o2, cnt, adj) where {T1<:SecondAdjointNodeSubexpr,T2<:SecondAdjointNodeSubexpr,I<:Integer,V<:AbstractVector{I}}
    ind = o2 + comp(cnt += 1)
    @inbounds y1[ind] = -t1.i
    @inbounds y2[ind] = -t2.i
    return cnt
end
@inline function hdrpass(t1::T1, t2::T2, comp, y1::V, y2::V, o2, cnt, adj) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNodeSubexpr,I<:Integer,V<:AbstractVector{I}}
    ind = o2 + comp(cnt += 1)
    @inbounds y1[ind] = t1.i
    @inbounds y2[ind] = -t2.i
    return cnt
end
@inline function hdrpass(t1::T1, t2::T2, comp, y1::V, y2::V, o2, cnt, adj) where {T1<:SecondAdjointNodeSubexpr,T2<:SecondAdjointNodeVar,I<:Integer,V<:AbstractVector{I}}
    ind = o2 + comp(cnt += 1)
    @inbounds y1[ind] = t2.i
    @inbounds y2[ind] = -t1.i
    return cnt
end

# Leaf pairs: raw sparsity probe (Vector push, identity-deduped later).
@inline function hdrpass(t1::SecondAdjointNodeSubexpr, t2::SecondAdjointNodeSubexpr, comp::Nothing, y1, y2, o2, cnt::Vector, adj)
    push!(cnt, (t1.i, t2.i))
    return cnt
end
@inline function hdrpass(t1::SecondAdjointNodeVar, t2::SecondAdjointNodeSubexpr, comp::Nothing, y1, y2, o2, cnt::Vector, adj)
    push!(cnt, (t1.i, t2.i))
    return cnt
end
@inline function hdrpass(t1::SecondAdjointNodeSubexpr, t2::SecondAdjointNodeVar, comp::Nothing, y1, y2, o2, cnt::Vector, adj)
    push!(cnt, (t1.i, t2.i))
    return cnt
end

# Leaf pairs: tuple-based sparsity probe (juliac path).
@inline function hdrpass(t1::T1, t2::T2, comp::Nothing, y1, y2, o2, cnt::Tuple{<:Tuple,<:Tuple}, adj) where {T1<:Union{SecondAdjointNodeVar,SecondAdjointNodeSubexpr},T2<:SecondAdjointNodeSubexpr}
    pair = (t1.i, t2.i)
    mapping, uniques = cnt
    idx = _hpass_find_pair(pair, uniques, 1)
    if idx === 0
        return ((mapping..., length(uniques) + 1), (uniques..., pair))
    else
        return ((mapping..., idx), uniques)
    end
end
@inline function hdrpass(t1::T1, t2::T2, comp::Nothing, y1, y2, o2, cnt::Tuple{<:Tuple,<:Tuple}, adj) where {T1<:SecondAdjointNodeSubexpr,T2<:SecondAdjointNodeVar}
    pair = (t1.i, t2.i)
    mapping, uniques = cnt
    idx = _hpass_find_pair(pair, uniques, 1)
    if idx === 0
        return ((mapping..., length(uniques) + 1), (uniques..., pair))
    else
        return ((mapping..., idx), uniques)
    end
end

# hrpass leaf: second-order diagonal of the subexpression coordinate.
@inline function hrpass(t::T, comp, y1, y2, o2, cnt, adj, adj2) where {T<:SecondAdjointNodeSubexpr}
    @inbounds y1[o2+comp(cnt+=1)] += adj2
    return cnt
end
@inline function hrpass(t::T, comp, y1::V, y2::V, o2, cnt, adj, adj2) where {T<:SecondAdjointNodeSubexpr,I<:Integer,V<:AbstractVector{I}}
    ind = o2 + comp(cnt += 1)
    @inbounds y1[ind] = -t.i
    @inbounds y2[ind] = -t.i
    return cnt
end
function hrpass(t::SecondAdjointNodeSubexpr, comp::Nothing, y1, y2, o2, cnt::Vector, adj, adj2)
    push!(cnt, (t.i, t.i))
    return cnt
end
@inline function hrpass(t::T, comp::Nothing, y1, y2, o2, cnt::Tuple{<:Tuple,<:Tuple}, adj, adj2) where {T<:SecondAdjointNodeSubexpr}
    pair = (t.i, t.i)
    mapping, uniques = cnt
    idx = _hpass_find_pair(pair, uniques, 1)
    if idx === 0
        return ((mapping..., length(uniques) + 1), (uniques..., pair))
    else
        return ((mapping..., idx), uniques)
    end
end

# hrpass0: a top-level linear occurrence of a subexpression contributes no
# second-order slot (its curvature enters through the stage seed instead).
@inline hrpass0(t::T, comp, y1, y2, o2, cnt, adj, adj2) where {T<:SecondAdjointNodeSubexpr} = cnt
@inline hrpass0(t::T, comp::Nothing, y1, y2, o2, cnt, adj, adj2) where {T<:SecondAdjointNodeSubexpr} = cnt

# Second-order slot counting (type-level, mirrors SecondAdjointNodeVar).
_hr0_val(::Type{<:SecondAdjointNodeSubexpr}) = Val(0)
_hrpass_val(::Type{<:SecondAdjointNodeSubexpr}) = Val(1)
_hdrpass_val(::Type{<:SecondAdjointNodeSubexpr}, ::Type{<:SecondAdjointNodeSubexpr}) = Val(1)
_hdrpass_val(::Type{<:SecondAdjointNodeVar}, ::Type{<:SecondAdjointNodeSubexpr}) = Val(1)
_hdrpass_val(::Type{<:SecondAdjointNodeSubexpr}, ::Type{<:SecondAdjointNodeVar}) = Val(1)
_hdrpass_val(::Type{<:SecondAdjointNodeSubexpr}, ::Type{SecondAdjointNode1{F,T,I}}) where {F,T,I} =
    _hdrpass_fixedvar_val(I)
_hdrpass_val(::Type{SecondAdjointNode1{F,T,I}}, ::Type{<:SecondAdjointNodeSubexpr}) where {F,T,I} =
    _hdrpass_fixedvar_val(I)
_hdrpass_val(::Type{<:SecondAdjointNodeSubexpr}, ::Type{SecondAdjointNode2{F,T,I1,I2}}) where {F,T,I1,I2} =
    _add_vals(_hdrpass_fixedvar_val(I1), _hdrpass_fixedvar_val(I2))
_hdrpass_val(::Type{SecondAdjointNode2{F,T,I1,I2}}, ::Type{<:SecondAdjointNodeSubexpr}) where {F,T,I1,I2} =
    _add_vals(_hdrpass_fixedvar_val(I1), _hdrpass_fixedvar_val(I2))
_hdrpass_fixedvar_val(::Type{<:SecondAdjointNodeSubexpr}) = Val(1)

# ── Jacobian via build-time elimination ──────────────────────────────────────
#
# Structure passes run in the extended coordinate space (x-columns positive,
# subexpression θ-slots negated).  At model build, _build_subexpr_jac resolves
# every stage element's Jacobian to x-columns in topological (add) order and
# emits flat index-mapped "programs"; evaluation then runs only shallow
# per-stage kernels plus type-generic product loops, so no composed tree types
# are ever compiled.

"""
    SubexprJac{T}

Build-time artifact holding the buffers, index maps, and final COO structure
for Jacobian evaluation with buffered subexpressions.

Value evaluation (see `_jac_coord_subexpr!`):
1. per-stage `jrpass` fills `stage_ext` (extended-space stage entries);
2. `res[dst] = stage_ext[src]` (direct) and `res[dst] = stage_ext[a] * res[b]`
   (chained) resolve every stage entry to an x-column, in emission order
   (`b < dst` always, so one sequential sweep is valid);
3. consumer `jrpass` fills `cons_ext`;
4. `jac[dst] = cons_ext[src]` (direct) and `jac[dst] = cons_ext[a] * res[b]`
   (composed) produce the final COO values.  Duplicate (row, col) pairs are
   summed by the COO convention.
"""
struct SubexprJac{T}
    stage_ext::Vector{T}
    stage_o1::Vector{Int}      # per-stage offsets into stage_ext, newest-first (tuple order)
    res::Vector{T}
    res_copy_dst::Vector{Int}
    res_copy_src::Vector{Int}
    res_prod_dst::Vector{Int}
    res_prod_a::Vector{Int}
    res_prod_b::Vector{Int}
    cons_ext::Vector{T}
    jac_copy_dst::Vector{Int}
    jac_copy_src::Vector{Int}
    jac_prod_dst::Vector{Int}
    jac_prod_a::Vector{Int}
    jac_prod_b::Vector{Int}
    rows::Vector{Int}
    cols::Vector{Int}
end

_build_subexpr_jac(::Type{T}, stages::Tuple{}, cons, nnzj_ext) where {T} = (nothing, nothing)

function _build_subexpr_jac(::Type{T}, stages::Tuple, cons, nnzj_ext) where {T}
    ordered = reverse(collect(Any, stages))   # oldest first
    # θ-slot → [(xcol, res index)]
    resolved = Dict{Int,Vector{Tuple{Int,Int}}}()
    res_copy_dst = Int[]; res_copy_src = Int[]
    res_prod_dst = Int[]; res_prod_a = Int[]; res_prod_b = Int[]
    res_n = 0
    stage_o1 = Int[]
    off = 0
    for s in ordered
        push!(stage_o1, off)
        n = length(s.itr)
        step = s.f.o1step
        erows = zeros(Int, n * step)
        ecols = zeros(Int, n * step)
        for k in eachindex(s.itr)
            sjacobian!(
                erows,
                ecols,
                s.f,
                s.itr[k],
                NaNSource{T}(),
                NaNSource{T}(),
                s.f.comp1,
                s.offset + k,
                (k - 1) * step,
                T(NaN),
            )
        end
        for ind in 1:(n*step)
            row = erows[ind]
            col = ecols[ind]
            lst = get!(() -> Tuple{Int,Int}[], resolved, row)
            if col > 0
                res_n += 1
                push!(res_copy_dst, res_n)
                push!(res_copy_src, off + ind)
                push!(lst, (col, res_n))
            elseif col < 0
                for (xc, b) in get(resolved, -col, Tuple{Int,Int}[])
                    res_n += 1
                    push!(res_prod_dst, res_n)
                    push!(res_prod_a, off + ind)
                    push!(res_prod_b, b)
                    push!(lst, (xc, res_n))
                end
            end
        end
        off += n * step
    end
    reverse!(stage_o1)   # match the newest-first stage tuple order

    jrows = zeros(Int, nnzj_ext)
    jcols = zeros(Int, nnzj_ext)
    _jac_structure!(T, cons, jrows, jcols)
    rows = Int[]; cols = Int[]
    jac_copy_dst = Int[]; jac_copy_src = Int[]
    jac_prod_dst = Int[]; jac_prod_a = Int[]; jac_prod_b = Int[]
    # Deduplicate (row, col) pairs: programs accumulate (+=) into shared slots,
    # so the final COO stays at the true-pattern size instead of one entry per
    # (consumer entry × resolved stage entry) product.
    slot = Dict{Tuple{Int,Int},Int}()
    emit(r, c) = get!(slot, (r, c)) do
        push!(rows, r); push!(cols, c)
        length(rows)
    end
    for ind in 1:nnzj_ext
        r = jrows[ind]
        c = jcols[ind]
        if c > 0
            push!(jac_copy_dst, emit(r, c))
            push!(jac_copy_src, ind)
        elseif c < 0
            for (xc, b) in get(resolved, -c, Tuple{Int,Int}[])
                push!(jac_prod_dst, emit(r, xc))
                push!(jac_prod_a, ind)
                push!(jac_prod_b, b)
            end
        end
    end
    return (
        SubexprJac{T}(
            zeros(T, off),
            stage_o1,
            zeros(T, res_n),
            res_copy_dst,
            res_copy_src,
            res_prod_dst,
            res_prod_a,
            res_prod_b,
            zeros(T, nnzj_ext),
            jac_copy_dst,
            jac_copy_src,
            jac_prod_dst,
            jac_prod_a,
            jac_prod_b,
            rows,
            cols,
        ),
        resolved,
    )
end

_stage_jac_fill!(stages::Tuple{}, o1s, i, sj, x, θ) = nothing
@inline function _stage_jac_fill!(stages::Tuple, o1s, i, sj, x, θ)
    s = first(stages)
    off = o1s[i]
    step = s.f.o1step
    for k in eachindex(s.itr)
        sjacobian!(
            sj.stage_ext,
            nothing,
            s.f,
            @inbounds(s.itr[k]),
            x,
            θ,
            s.f.comp1,
            0,
            off + (k - 1) * step,
            one(eltype(x)),
        )
    end
    _stage_jac_fill!(Base.tail(stages), o1s, i + 1, sj, x, θ)
    return nothing
end

# Refresh the resolved stage-Jacobian values at the current point (shared by
# Jacobian and Hessian evaluation).  Assumes θ is synced.
function _resolve_stage_jac!(sj::SubexprJac, stages::Tuple, x, θ)
    fill!(sj.stage_ext, zero(eltype(sj.stage_ext)))
    _stage_jac_fill!(stages, sj.stage_o1, 1, sj, x, θ)
    @inbounds for m in eachindex(sj.res_copy_dst)
        sj.res[sj.res_copy_dst[m]] = sj.stage_ext[sj.res_copy_src[m]]
    end
    @inbounds for m in eachindex(sj.res_prod_dst)
        sj.res[sj.res_prod_dst[m]] = sj.stage_ext[sj.res_prod_a[m]] * sj.res[sj.res_prod_b[m]]
    end
    return nothing
end

function _jac_coord_subexpr!(sj::SubexprJac, stages::Tuple, cons, x, θ, jac)
    _resolve_stage_jac!(sj, stages, x, θ)
    fill!(sj.cons_ext, zero(eltype(sj.cons_ext)))
    _jac_coord!(cons, x, θ, sj.cons_ext)
    fill!(jac, zero(eltype(jac)))
    @inbounds for m in eachindex(sj.jac_copy_dst)
        jac[sj.jac_copy_dst[m]] += sj.cons_ext[sj.jac_copy_src[m]]
    end
    @inbounds for m in eachindex(sj.jac_prod_dst)
        jac[sj.jac_prod_dst[m]] += sj.cons_ext[sj.jac_prod_a[m]] * sj.res[sj.jac_prod_b[m]]
    end
    return jac
end

# ── Hessian via build-time elimination ────────────────────────────────────────
#
# Extended-space Hessian entries (consumer blocks weighted by σ/y, stage blocks
# weighted by the accumulated adjoint seeds) are expanded to x-coordinates with
# the resolved stage Jacobians:
#   (x,x):   copy;
#   (x,s):   H[a,c]  += coef · h · J[j,c]          (coef 2 on the diagonal);
#   (s,s):   H[c1,c2] += coef · h · J[j1,c1] · J[j2,c2].
# All coordinates, products, and coefficients are enumerated at model build.

"""
    SubexprHess{T}

Build-time artifact for Hessian evaluation with buffered subexpressions.
`hess_ext` holds the extended-space entries: the consumer segment first
(offsets baked at `add_obj`/`add_con` time), then one segment per stage.
"""
struct SubexprHess{T}
    hess_ext::Vector{T}
    stage_o2::Vector{Int}      # per-stage offsets into hess_ext, newest-first (tuple order)
    hess_copy_dst::Vector{Int}
    hess_copy_src::Vector{Int}
    p1_dst::Vector{Int}
    p1_a::Vector{Int}
    p1_b::Vector{Int}
    p1_c::Vector{T}
    p2_dst::Vector{Int}
    p2_a::Vector{Int}
    p2_b1::Vector{Int}
    p2_b2::Vector{Int}
    p2_c::Vector{T}
    rows::Vector{Int}
    cols::Vector{Int}
end

_build_subexpr_hess(::Type{T}, stages::Tuple{}, objs, cons, nnzh_ext, resolved) where {T} =
    nothing

function _build_subexpr_hess(::Type{T}, stages::Tuple, objs, cons, nnzh_ext, resolved) where {T}
    ordered = reverse(collect(Any, stages))   # oldest first
    stage_o2 = Int[]
    total = nnzh_ext
    for s in ordered
        push!(stage_o2, total)
        total += length(s.itr) * s.f.o2step
    end
    erows = zeros(Int, total)
    ecols = zeros(Int, total)
    _obj_hess_structure!(T, objs, erows, ecols)
    _con_hess_structure!(T, cons, erows, ecols)
    for (si, s) in enumerate(ordered)
        step = s.f.o2step
        for k in eachindex(s.itr)
            shessian!(
                erows,
                ecols,
                s.f,
                s.itr[k],
                NaNSource{T}(),
                NaNSource{T}(),
                s.f.comp2,
                stage_o2[si] + (k - 1) * step,
                T(NaN),
                T(NaN),
            )
        end
    end
    reverse!(stage_o2)   # match the newest-first stage tuple order

    rows = Int[]; cols = Int[]
    hess_copy_dst = Int[]; hess_copy_src = Int[]
    p1_dst = Int[]; p1_a = Int[]; p1_b = Int[]; p1_c = T[]
    p2_dst = Int[]; p2_a = Int[]; p2_b1 = Int[]; p2_b2 = Int[]; p2_c = T[]
    nores = Tuple{Int,Int}[]
    # Deduplicate normalized (max, min) pairs; value programs accumulate (+=).
    slot = Dict{Tuple{Int,Int},Int}()
    emit(i, j) = get!(slot, (max(i, j), min(i, j))) do
        push!(rows, max(i, j)); push!(cols, min(i, j))
        length(rows)
    end
    for ind in 1:total
        r = erows[ind]
        c = ecols[ind]
        (r == 0 && c == 0) && continue
        if r > 0 && c > 0
            push!(hess_copy_dst, emit(r, c))
            push!(hess_copy_src, ind)
        elseif r > 0 && c < 0 || r < 0 && c > 0
            a = r > 0 ? r : c
            j = r > 0 ? -c : -r
            for (xc, b) in get(resolved, j, nores)
                push!(p1_dst, emit(a, xc))
                push!(p1_a, ind)
                push!(p1_b, b)
                push!(p1_c, a == xc ? T(2) : T(1))
            end
        else
            j1 = -r
            j2 = -c
            if j1 == j2
                R = get(resolved, j1, nores)
                for p in eachindex(R), q in p:length(R)
                    (c1, b1) = R[p]
                    (c2, b2) = R[q]
                    push!(p2_dst, emit(c1, c2))
                    push!(p2_a, ind)
                    push!(p2_b1, b1)
                    push!(p2_b2, b2)
                    push!(p2_c, (p != q && c1 == c2) ? T(2) : T(1))
                end
            else
                for (c1, b1) in get(resolved, j1, nores), (c2, b2) in get(resolved, j2, nores)
                    push!(p2_dst, emit(c1, c2))
                    push!(p2_a, ind)
                    push!(p2_b1, b1)
                    push!(p2_b2, b2)
                    push!(p2_c, c1 == c2 ? T(2) : T(1))
                end
            end
        end
    end
    return SubexprHess{T}(
        zeros(T, total),
        stage_o2,
        hess_copy_dst,
        hess_copy_src,
        p1_dst,
        p1_a,
        p1_b,
        p1_c,
        p2_dst,
        p2_a,
        p2_b1,
        p2_b2,
        p2_c,
        rows,
        cols,
    )
end

# Weighted first-order reverse sweep accumulating the stage Hessian seeds
# ∂L/∂s_j = σ ∂f/∂s_j + Σᵢ yᵢ ∂cᵢ/∂s_j into the adjoint buffer, chained
# through nested stages.  Variable-leaf contributions are discarded
# (GradTarget with y = nothing).
_seed_objs!(objs::Tuple{}, x, θ, t, w) = nothing
@inline function _seed_objs!(objs::Tuple, x, θ, t, w)
    _seed_objs!(Base.tail(objs), x, θ, t, w)
    gradient!(t, first(objs), x, θ, w)
    return nothing
end

_seed_cons!(cons::Tuple{}, x, θ, t, y) = nothing
@inline function _seed_cons!(cons::Tuple, x, θ, t, y)
    _seed_cons!(Base.tail(cons), x, θ, t, y)
    con = first(cons)
    for i in eachindex(con.itr)
        gradient!(t, con.f, x, θ, @inbounds(con.itr[i]), @inbounds(y[offset0(con, i)]))
    end
    return nothing
end

_stage_hess_fill!(stages::Tuple{}, o2s, i, sh, abuf, x, θ) = nothing
@inline function _stage_hess_fill!(stages::Tuple, o2s, i, sh, abuf, x, θ)
    s = first(stages)
    off = o2s[i]
    step = s.f.o2step
    for k in eachindex(s.itr)
        shessian!(
            sh.hess_ext,
            nothing,
            s.f,
            @inbounds(s.itr[k]),
            x,
            θ,
            s.f.comp2,
            off + (k - 1) * step,
            @inbounds(abuf[s.offset+k]),
            zero(eltype(x)),
        )
    end
    _stage_hess_fill!(Base.tail(stages), o2s, i + 1, sh, abuf, x, θ)
    return nothing
end

function _hess_coord_subexpr!(
    sh::SubexprHess,
    sj::SubexprJac,
    stages::Tuple,
    objs,
    cons,
    x,
    θ,
    abuf,
    hess,
    obj_weight,
    y,
)
    _sync_subexprs!(stages, x, θ)
    _resolve_stage_jac!(sj, stages, x, θ)
    # stage seeds
    fill!(abuf, zero(eltype(abuf)))
    t = GradTarget(nothing, abuf)
    _seed_objs!(objs, x, θ, t, obj_weight)
    y === nothing || _seed_cons!(cons, x, θ, t, y)
    _reverse_subexprs!(stages, x, θ, t)
    # extended-space entries
    fill!(sh.hess_ext, zero(eltype(sh.hess_ext)))
    _obj_hess_coord!(objs, x, θ, sh.hess_ext, obj_weight)
    y === nothing || _con_hess_coord!(cons, x, θ, y, sh.hess_ext, obj_weight)
    _stage_hess_fill!(stages, sh.stage_o2, 1, sh, abuf, x, θ)
    # compose
    fill!(hess, zero(eltype(hess)))
    @inbounds for m in eachindex(sh.hess_copy_dst)
        hess[sh.hess_copy_dst[m]] += sh.hess_ext[sh.hess_copy_src[m]]
    end
    @inbounds for m in eachindex(sh.p1_dst)
        hess[sh.p1_dst[m]] += sh.p1_c[m] * sh.hess_ext[sh.p1_a[m]] * sj.res[sh.p1_b[m]]
    end
    @inbounds for m in eachindex(sh.p2_dst)
        hess[sh.p2_dst[m]] +=
            sh.p2_c[m] * sh.hess_ext[sh.p2_a[m]] * sj.res[sh.p2_b1[m]] * sj.res[sh.p2_b2[m]]
    end
    return hess
end

_reverse_subexprs!(stages::Tuple{}, x, θ, t::GradTarget) = nothing
@inline function _reverse_subexprs!(stages::Tuple, x, θ, t::GradTarget)
    s = first(stages)
    for k in eachindex(s.itr)
        adj = @inbounds t.abuf[s.offset+k]
        graph = s.f.f(@inbounds(s.itr[k]), AdjointNodeSource(x), θ)
        drpass(graph, t, adj)
    end
    _reverse_subexprs!(Base.tail(stages), x, θ, t)
    return nothing
end
