# ── Buffered subexpressions ───────────────────────────────────────────────────
#
# A buffered subexpression is evaluated ONCE per element into a dedicated slot
# of the model's θ vector (a "computed parameter"), and consumers reference the
# slot through a `SubexprNode` leaf instead of splicing the subexpression tree
# into their own.  Consumer tree types stay shallow, so compilation cost is
# additive (one kernel per stage) rather than multiplicative in nesting depth.
#
# Derivatives: gradients flow through an adjoint buffer (`GradTarget`);
# Jacobians and Hessians run in the extended (x, s) coordinate space and are
# resolved to x-coordinates by build-time elimination (see the sections below);
# Hessian stage curvature is seeded through the (λ, μ) buffers (`HessTarget`).

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
Base.show(io::IO, node::SubexprNode{I}) where {I} = print(io, "subexpr(θ[", node.i, "])")

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

# Value mode (sparse gradient slots, used by the KA gradient path): write the
# local adjoint into the compressed slot, like a Var.
@inline function grpass(d::D, comp, y, o1, cnt, adj) where {D<:AdjointNodeSubexpr}
    @inbounds y[o1+comp(cnt+=1)] += adj
    return cnt
end
# Gradient-scatter structure collection: the dense target of a subexpression
# leaf is the adjoint buffer, not the gradient; mark it with the negated slot
# (the extension shifts it to nvar + slot when assembling the extended target).
@inline function grpass(
    d::D,
    comp,
    y::V,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeSubexpr,V<:AbstractVector{Tuple{Int,Int}}}
    ind = o1 + comp(cnt += 1)
    @inbounds y[ind] = (-d.i, ind)
    return cnt
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

# ── Hessian target: one second-order sweep carries everything ────────────────
#
# For buffered models the second-order passes are driven with a HessTarget in
# place of the plain Hessian-slot vector.  Ordinary slot writes pass through to
# `ext`; buffered-leaf methods additionally accumulate the stage seeds:
#   abuf[j]  += adj    (λⱼ = ∂L/∂sⱼ, first-order adjoint arriving at the leaf)
#   abuf2[j] += adj2   (μⱼ, coefficient of ∇sⱼ∇sⱼᵀ, the DIAGONAL coupling)
# Each stage is then replayed ONCE with seeds (λⱼ, μⱼ), which produces
# λⱼ·∇²sⱼ + μⱼ·∇sⱼ∇sⱼᵀ natively and chains through nested stages.  Only the
# CROSS couplings (pairs of different coordinates) remain for the build-time
# composition maps.
struct HessTarget{Y,A}
    ext::Y
    abuf::A
    abuf2::A
end
Base.@propagate_inbounds Base.getindex(t::HessTarget, i) = t.ext[i]
Base.@propagate_inbounds Base.setindex!(t::HessTarget, v, i) = (t.ext[i] = v; v)
Base.eltype(::Type{HessTarget{Y,A}}) where {Y,A} = eltype(Y)

# Buffered leaf under a HessTarget: accumulate seeds instead of writing slots.
# The slot counter still advances where the probe allocated one, keeping the
# compressor aligned; the corresponding ext slots stay zero and the build emits
# no programs for (-j,-j) entries.
@inline function hrpass(
    t::T,
    comp,
    y1::HessTarget,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeSubexpr}
    @inbounds y1.abuf[t.i] += adj
    @inbounds y1.abuf2[t.i] += adj2
    return cnt + 1
end
@inline function hrpass0(
    t::T,
    comp,
    y1::HessTarget,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeSubexpr}
    @inbounds y1.abuf[t.i] += adj
    return cnt
end
@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1::HessTarget,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNodeSubexpr,T2<:SecondAdjointNodeSubexpr}
    i, j = t1.i, t2.i
    @inbounds if i == j
        y1.abuf2[i] += 2 * adj
        cnt += 1
    else
        y1[o2+comp(cnt+=1)] += adj
    end
    return cnt
end

# Stage SIMDFunction: the second-order compressor must be built with FULL
# hrpass counting, not hrpass0.  hrpass0 skips leaves reached through purely
# linear paths, which is valid only when the top-level second-order seed is
# zero; a stage replay is seeded with μ ≠ 0, so a linear stage tree still
# produces μ·∇s∇sᵀ and needs those slots.
@inline function _stage_simdfunction(T, gen::Base.Generator)
    f = replace_T(T, gen.f(DataSource()))

    d = f(Identity(), AdjointNodeSource(NaNSource{T}()), NaNSource{T}())
    raw1 = Any[]
    grpass(d, nothing, nothing, nothing, raw1, T(NaN))

    t = f(Identity(), SecondAdjointNodeSource(NaNSource{T}()), NaNSource{T}())
    raw2 = Any[]
    hrpass(t, nothing, nothing, nothing, nothing, raw2, T(NaN), T(NaN))

    unique1 = _ident_unique(raw1)
    o1step = length(unique1)
    mapping1 = Int[findfirst(y -> y === x, unique1) for x in raw1]
    c1 = Compressor(ntuple(i -> mapping1[i], _gr_val(typeof(d))))

    unique2 = _ident_unique(raw2)
    o2step = length(unique2)
    mapping2 = Int[findfirst(y -> y === x, unique2) for x in raw2]
    c2 = Compressor(ntuple(i -> mapping2[i], _hrpass_val(typeof(t))))

    return SIMDFunction(f, c1, c2, 0, 0, 0, o1step, o2step)
end

# Stage second-order inner driver: hrpass directly (see _stage_simdfunction).
@inline function _stage_shessian!(y1, y2, f, p, x, θ, comp, o2, adj1, adj2)
    graph = f(p, SecondAdjointNodeSource(x), θ)
    hrpass(graph, comp, y1, y2, o2, 0, adj1, adj2)
    return nothing
end

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
    vals::Vector{T}            # scratch for COO-based jprod/jtprod
end

# Build-time structure passes iterate block iterators on the host; for device
# models, reconstruct the blocks with host copies of their iterators.
_host_array(x::Array) = x
_host_array(x) = Array(x)
_host_block(s::SubexprStage) = SubexprStage(s.f, _host_array(s.itr), s.offset)
_host_blocks(t::Tuple) = map(_host_block, t)
# _host_block methods for Objective/Constraint/ConstraintAugmentation are in
# nlp.jl (those types are defined after this file is included).

_build_subexpr_jac(::Type{T}, stages::Tuple{}, cons, nnzj_ext) where {T} =
    (nothing, nothing, nothing)

function _build_subexpr_jac(::Type{T}, stages::Tuple, cons, nnzj_ext) where {T}
    cons = _host_blocks(cons)
    ordered = reverse(collect(Any, _host_blocks(stages)))   # oldest first
    # θ-slot → [(xcol, res index)] : stage Jacobians fully resolved to x-columns
    resolved = Dict{Int,Vector{Tuple{Int,Int}}}()
    # θ-slot → [(signed col, stage_ext index)] : ONE-level (local) rows, used by
    # the sequential Hessian elimination
    localrows = Dict{Int,Vector{Tuple{Int,Int}}}()
    res_copy_dst = Int[]; res_copy_src = Int[]
    res_prod_dst = Int[]; res_prod_a = Int[]; res_prod_b = Int[]
    res_n = 0
    rowmaps = Dict{Int,Dict{Int,Int}}()
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
            col == 0 && continue
            push!(get!(() -> Tuple{Int,Int}[], localrows, row), (col, off + ind))
            # Merge resolved entries per (row, x-column): one res slot per
            # distinct column, contributions accumulated (+=) at evaluation.
            # Without merging the row width enumerates PATHS (2^depth on a
            # chain) instead of columns.
            lst = get!(() -> Tuple{Int,Int}[], resolved, row)
            rmap = get!(() -> Dict{Int,Int}(), rowmaps, row)
            getres!(xc) = get!(rmap, xc) do
                res_n += 1
                push!(lst, (xc, res_n))
                res_n
            end
            if col > 0
                push!(res_copy_dst, getres!(col))
                push!(res_copy_src, off + ind)
            elseif col < 0
                for (xc, b) in get(resolved, -col, Tuple{Int,Int}[])
                    push!(res_prod_dst, getres!(xc))
                    push!(res_prod_a, off + ind)
                    push!(res_prod_b, b)
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
            zeros(T, length(rows)),
        ),
        resolved,
        localrows,
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
# Fill only the local (one-level) stage Jacobian values — all the Hessian
# needs; the Jacobian additionally resolves them to x-columns below.
function _fill_stage_jac!(sj::SubexprJac, stages::Tuple, x, θ)
    fill!(sj.stage_ext, zero(eltype(sj.stage_ext)))
    _stage_jac_fill!(stages, sj.stage_o1, 1, sj, x, θ)
    return nothing
end

function _resolve_stage_jac!(sj::SubexprJac, stages::Tuple, x, θ)
    _fill_stage_jac!(sj, stages, x, θ)
    fill!(sj.res, zero(eltype(sj.res)))
    @inbounds for m in eachindex(sj.res_copy_dst)
        sj.res[sj.res_copy_dst[m]] += sj.stage_ext[sj.res_copy_src[m]]
    end
    @inbounds for m in eachindex(sj.res_prod_dst)
        sj.res[sj.res_prod_dst[m]] += sj.stage_ext[sj.res_prod_a[m]] * sj.res[sj.res_prod_b[m]]
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

# ── Hessian via sequential (stage-by-stage) elimination ──────────────────────
#
# Extended-space Hessian entries (consumer blocks weighted by σ/y, stage blocks
# weighted by the (λ, μ) seeds) are eliminated one stage at a time, newest
# first.  Cross pairs live in a merged intermediate buffer C, one slot per
# distinct coordinate pair per level; eliminating a stage multiplies each pair
# involving one of its elements by that element's ONE-LEVEL (local) Jacobian
# entries, producing pairs over strictly older coordinates:
#   (x, x)      → final COO slot          (coef 2 when the pair collapses);
#   (s, s) same → μ seed of that element  (handled by its later replay);
#   otherwise   → merged C slot one level down.
# Merging per level is what keeps the work at the fill size: the flat
# alternative (expanding every entry straight to x-coordinates) enumerates
# PATHS and scales as O(K³) on a depth-K chain, versus O(K²) here.

"""
    SubexprHess{T}

Build-time artifact for Hessian evaluation with buffered subexpressions.
`hess_ext` holds the extended-space entries (consumer segment first, then one
segment per stage); `C` holds the merged intermediate cross values.  All
program slices are per-stage, newest-first, aligned with the stage tuple.
"""
struct SubexprHess{T}
    hess_ext::Vector{T}
    abuf2::Vector{T}           # second-order (μ) seed buffer, θ-length
    C::Vector{T}               # merged intermediate cross values
    stage_o2::Vector{Int}      # per-stage offsets into hess_ext, newest-first
    lvl_ccopy::Vector{UnitRange{Int}}
    lvl_cmul::Vector{UnitRange{Int}}
    lvl_amul::Vector{UnitRange{Int}}
    lvl_hmul::Vector{UnitRange{Int}}
    ccopy_dst::Vector{Int}     # C[dst] += hess_ext[src]
    ccopy_src::Vector{Int}
    cmul_dst::Vector{Int}      # C[dst] += C[src] * stage_ext[a]
    cmul_src::Vector{Int}
    cmul_a::Vector{Int}
    amul_dst::Vector{Int}      # abuf2[dst] += 2 * C[src] * stage_ext[a]
    amul_src::Vector{Int}
    amul_a::Vector{Int}
    hmul_dst::Vector{Int}      # hess[dst] += coef * C[src] * stage_ext[a]
    hmul_src::Vector{Int}
    hmul_a::Vector{Int}
    hmul_c::Vector{T}
    hess_copy_dst::Vector{Int} # hess[dst] += hess_ext[src]  (direct xx)
    hess_copy_src::Vector{Int}
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{T}            # scratch for COO-based hprod
end

_build_subexpr_hess(::Type{T}, stages::Tuple{}, objs, cons, nnzh_ext, localrows, nθ) where {T} =
    nothing

function _build_subexpr_hess(::Type{T}, stages::Tuple, objs, cons, nnzh_ext, localrows, nθ) where {T}
    objs = _host_blocks(objs)
    cons = _host_blocks(cons)
    ordered = reverse(collect(Any, _host_blocks(stages)))   # oldest first: levels 1..nlv
    nlv = length(ordered)
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
            _stage_shessian!(
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

    slotlvl = zeros(Int, nθ)
    for (i, s) in enumerate(ordered)
        slotlvl[(s.offset+1):(s.offset+length(s.itr))] .= i
    end
    lvlof(ξ) = ξ > 0 ? 0 : slotlvl[-ξ]
    normp(a, b) = a >= b ? (a, b) : (b, a)

    rows = Int[]; cols = Int[]
    final = Dict{Tuple{Int,Int},Int}()
    emit(i, j) = get!(final, (max(i, j), min(i, j))) do
        push!(rows, max(i, j)); push!(cols, min(i, j))
        length(rows)
    end

    # pair → C index, per level; programs collected per level, flattened in
    # iteration (newest-first) order at the end.
    entries = [Dict{Tuple{Int,Int},Int}() for _ in 1:nlv]
    Cn = Ref(0)
    getC!(p, l) = get!(() -> (Cn[] += 1), entries[l], p)
    ccopyL = [Tuple{Int,Int}[] for _ in 1:nlv]
    cmulL = [NTuple{3,Int}[] for _ in 1:nlv]
    amulL = [NTuple{3,Int}[] for _ in 1:nlv]
    hmulL = [Tuple{Int,Int,Int,T}[] for _ in 1:nlv]
    hess_copy_dst = Int[]; hess_copy_src = Int[]

    # Seed: classify every extended entry.
    for ind in 1:total
        r = erows[ind]
        c = ecols[ind]
        (r == 0 && c == 0) && continue
        if r > 0 && c > 0
            push!(hess_copy_dst, emit(r, c))
            push!(hess_copy_src, ind)
        elseif r < 0 && r == c
            # slot diagonal: redirected to the μ buffer at runtime; dead slot
            continue
        else
            p = normp(r, c)
            l = max(lvlof(p[1]), lvlof(p[2]))
            cidx = getC!(p, l)
            push!(ccopyL[l], (cidx, ind))
        end
    end

    nores = Tuple{Int,Int}[]
    # Eliminate levels newest → oldest.  Within a level: rank-2 pairs (both
    # coordinates at this level) first — they produce rank-1 pairs at the same
    # level — then rank-1 pairs, which only produce strictly older pairs.
    for l in nlv:-1:1
        d = entries[l]
        expand(p, cp) = begin
            ξexp, oth = lvlof(p[1]) == l ? (p[1], p[2]) : (p[2], p[1])
            for (ccol, aidx) in get(localrows, -ξexp, nores)
                if ccol > 0 && oth > 0
                    push!(hmulL[l], (emit(ccol, oth), cp, aidx, ccol == oth ? T(2) : T(1)))
                elseif ccol < 0 && ccol == oth
                    push!(amulL[l], (-ccol, cp, aidx))
                else
                    np = normp(ccol, oth)
                    l2 = max(lvlof(np[1]), lvlof(np[2]))
                    push!(cmulL[l], (getC!(np, l2), cp, aidx))
                end
            end
        end
        r2 = [p for p in keys(d) if lvlof(p[1]) == l && lvlof(p[2]) == l]
        for p in r2
            expand(p, d[p])
        end
        r1 = [p for p in keys(d) if xor(lvlof(p[1]) == l, lvlof(p[2]) == l)]
        for p in r1
            expand(p, d[p])
        end
    end

    # Flatten per-level programs in iteration order (newest first).
    lvl_ccopy = UnitRange{Int}[]; ccopy_dst = Int[]; ccopy_src = Int[]
    lvl_cmul = UnitRange{Int}[]; cmul_dst = Int[]; cmul_src = Int[]; cmul_a = Int[]
    lvl_amul = UnitRange{Int}[]; amul_dst = Int[]; amul_src = Int[]; amul_a = Int[]
    lvl_hmul = UnitRange{Int}[]; hmul_dst = Int[]; hmul_src = Int[]; hmul_a = Int[]; hmul_c = T[]
    for l in nlv:-1:1
        n0 = length(ccopy_dst)
        for (dst, s) in ccopyL[l]
            push!(ccopy_dst, dst); push!(ccopy_src, s)
        end
        push!(lvl_ccopy, (n0+1):length(ccopy_dst))
        n0 = length(cmul_dst)
        for (dst, s, a) in cmulL[l]
            push!(cmul_dst, dst); push!(cmul_src, s); push!(cmul_a, a)
        end
        push!(lvl_cmul, (n0+1):length(cmul_dst))
        n0 = length(amul_dst)
        for (dst, s, a) in amulL[l]
            push!(amul_dst, dst); push!(amul_src, s); push!(amul_a, a)
        end
        push!(lvl_amul, (n0+1):length(amul_dst))
        n0 = length(hmul_dst)
        for (dst, s, a, cf) in hmulL[l]
            push!(hmul_dst, dst); push!(hmul_src, s); push!(hmul_a, a); push!(hmul_c, cf)
        end
        push!(lvl_hmul, (n0+1):length(hmul_dst))
    end
    reverse!(stage_o2)   # match the newest-first stage tuple order

    return SubexprHess{T}(
        zeros(T, total),
        zeros(T, nθ),
        zeros(T, Cn[]),
        stage_o2,
        lvl_ccopy,
        lvl_cmul,
        lvl_amul,
        lvl_hmul,
        ccopy_dst,
        ccopy_src,
        cmul_dst,
        cmul_src,
        cmul_a,
        amul_dst,
        amul_src,
        amul_a,
        hmul_dst,
        hmul_src,
        hmul_a,
        hmul_c,
        hess_copy_dst,
        hess_copy_src,
        rows,
        cols,
        zeros(T, length(rows)),
    )
end

# Interleaved sweep, newest-first: replay stage i (its (λ, μ) seeds are
# complete: consumers, newer replays, and newer levels' amul programs have all
# run), then execute level-i programs, which eliminate this stage's
# coordinates from the cross buffer.
_hess_seq!(stages::Tuple{}, i, sh, sj, t::HessTarget, hess, x, θ) = nothing
@inline function _hess_seq!(stages::Tuple, i, sh, sj, t::HessTarget, hess, x, θ)
    s = first(stages)
    off = sh.stage_o2[i]
    step = s.f.o2step
    for k in eachindex(s.itr)
        _stage_shessian!(
            t,
            nothing,
            s.f,
            @inbounds(s.itr[k]),
            x,
            θ,
            s.f.comp2,
            off + (k - 1) * step,
            @inbounds(t.abuf[s.offset+k]),
            @inbounds(t.abuf2[s.offset+k]),
        )
    end
    C = sh.C
    ext = sh.hess_ext
    sx = sj.stage_ext
    @inbounds for m in sh.lvl_ccopy[i]
        C[sh.ccopy_dst[m]] += ext[sh.ccopy_src[m]]
    end
    @inbounds for m in sh.lvl_cmul[i]
        C[sh.cmul_dst[m]] += C[sh.cmul_src[m]] * sx[sh.cmul_a[m]]
    end
    @inbounds for m in sh.lvl_amul[i]
        t.abuf2[sh.amul_dst[m]] += 2 * C[sh.amul_src[m]] * sx[sh.amul_a[m]]
    end
    @inbounds for m in sh.lvl_hmul[i]
        hess[sh.hmul_dst[m]] += sh.hmul_c[m] * C[sh.hmul_src[m]] * sx[sh.hmul_a[m]]
    end
    _hess_seq!(Base.tail(stages), i + 1, sh, sj, t, hess, x, θ)
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
    _fill_stage_jac!(sj, stages, x, θ)
    fill!(abuf, zero(eltype(abuf)))
    fill!(sh.abuf2, zero(eltype(sh.abuf2)))
    fill!(sh.hess_ext, zero(eltype(sh.hess_ext)))
    fill!(sh.C, zero(eltype(sh.C)))
    fill!(hess, zero(eltype(hess)))
    t = HessTarget(sh.hess_ext, abuf, sh.abuf2)
    _obj_hess_coord!(objs, x, θ, t, obj_weight)
    y === nothing || _con_hess_coord!(cons, x, θ, y, t, obj_weight)
    _hess_seq!(stages, 1, sh, sj, t, hess, x, θ)
    @inbounds for m in eachindex(sh.hess_copy_dst)
        hess[sh.hess_copy_dst[m]] += sh.hess_ext[sh.hess_copy_src[m]]
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

# Device (KernelAbstractions) support artifact: the extension provides a method
# for KA backends; the base fallback means CPU models carry nothing extra.
build_subexpr_ka(c, sjac, shess, nvar) = nothing
