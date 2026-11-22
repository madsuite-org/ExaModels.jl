module ExaModelsOptimalControl

using ExaModels, LinearAlgebra, OptimalControl

# ============================================================================
# Section A: Null node pass-throughs for LinearAlgebra functions
# ============================================================================

for f in [:inv, :abs, :sqrt, :cbrt, :abs2, :exp, :log, :sin, :cos, :tan]
    @eval @inline Base.$f(x::ExaModels.Null{T}) where {T <: Real} =
        ExaModels.Null(Base.$f(x.value))
end

# ============================================================================
# Section B: Scalar Null arithmetic with zero/one elimination
# ============================================================================

# Null op Null
for op in (:+, :-, :*)
    @eval @inline function Base.$op(
            a::ExaModels.Null{T},
            b::ExaModels.Null{S},
        ) where {T <: Real, S <: Real}
        return ExaModels.Null(Base.$op(a.value, b.value))
    end
end

# Null + AbstractNode / AbstractNode + Null (zero elimination)
@inline function Base.:+(a::ExaModels.Null{T}, b::ExaModels.AbstractNode) where {T <: Real}
    return ExaModels.Node2(+, a, b)
end
@inline function Base.:+(a::ExaModels.AbstractNode, b::ExaModels.Null{T}) where {T <: Real}
    return b.value == zero(T) ? a : ExaModels.Node2(+, a, b)
end

# Null * AbstractNode / AbstractNode * Null (zero/one elimination)
@inline function Base.:*(a::ExaModels.Null{T}, b::ExaModels.AbstractNode) where {T <: Real}
    return a.value == zero(T) ? ExaModels.Null(zero(T)) :
        a.value == one(T) ? b : ExaModels.Node2(*, a, b)
end
@inline function Base.:*(a::ExaModels.AbstractNode, b::ExaModels.Null{T}) where {T <: Real}
    return b.value == zero(T) ? ExaModels.Null(zero(T)) :
        b.value == one(T) ? a : ExaModels.Node2(*, a, b)
end

# Null - AbstractNode / AbstractNode - Null
@inline function Base.:-(a::ExaModels.Null{T}, b::ExaModels.AbstractNode) where {T <: Real}
    return a.value == zero(T) ? ExaModels.Node1(-, b) : ExaModels.Node2(-, a, b)
end
@inline function Base.:-(a::ExaModels.AbstractNode, b::ExaModels.Null{T}) where {T <: Real}
    return b.value == zero(T) ? a : ExaModels.Node2(-, a, b)
end

# Null op Real / Real op Null
for op in (:+, :-, :*)
    @eval @inline function Base.$op(a::ExaModels.Null{T}, b::Real) where {T <: Real}
        return ExaModels.Null(Base.$op(a.value, b))
    end
    @eval @inline function Base.$op(a::Real, b::ExaModels.Null{T}) where {T <: Real}
        return ExaModels.Null(Base.$op(a, b.value))
    end
end

# Null op Integer / Integer op Null — disambiguate Null{T} op Real vs AbstractNode op Integer
for op in (:+, :-, :*)
    @eval @inline function Base.$op(a::ExaModels.Null{T}, b::Integer) where {T <: Real}
        return ExaModels.Null(Base.$op(a.value, b))
    end
    @eval @inline function Base.$op(a::Integer, b::ExaModels.Null{T}) where {T <: Real}
        return ExaModels.Null(Base.$op(a, b.value))
    end
end

# Integer × AbstractNode zero/one elimination (more specific than core's Real × AbstractNode)
# Fixes: 0 * x → Null(0), 1 * x → x, 0 + x → x, etc.
@inline function Base.:*(a::Integer, b::ExaModels.AbstractNode)
    return iszero(a) ? ExaModels.Null(zero(a)) :
        isone(a) ? b : ExaModels.Node2(*, a, b)
end
@inline function Base.:*(a::ExaModels.AbstractNode, b::Integer)
    return iszero(b) ? ExaModels.Null(zero(b)) :
        isone(b) ? a : ExaModels.Node2(*, a, b)
end
@inline function Base.:+(a::Integer, b::ExaModels.AbstractNode)
    return iszero(a) ? b : ExaModels.Node2(+, a, b)
end
@inline function Base.:+(a::ExaModels.AbstractNode, b::Integer)
    return iszero(b) ? a : ExaModels.Node2(+, a, b)
end
@inline function Base.:-(a::Integer, b::ExaModels.AbstractNode)
    return iszero(a) ? ExaModels.Node1(-, b) : ExaModels.Node2(-, a, b)
end
@inline function Base.:-(a::ExaModels.AbstractNode, b::Integer)
    return iszero(b) ? a : ExaModels.Node2(-, a, b)
end

# ============================================================================
# Section C: Type aliases, promotion, and adjoint for nodes
# ============================================================================

const ExaNode = Union{
    ExaModels.AbstractNode,
    ExaModels.AbstractAdjointNode,
    ExaModels.AbstractSecondAdjointNode,
}
const VecExaNode = AbstractVector{<:ExaNode}
const MatExaNode = AbstractMatrix{<:ExaNode}

# Type promotion: [x, 0] should give Vector{AbstractNode} with Null(0), not Vector{Any}
Base.promote_rule(::Type{<:ExaModels.AbstractNode}, ::Type{<:Real}) = ExaModels.AbstractNode
Base.convert(::Type{ExaModels.AbstractNode}, x::Real) =
    iszero(x) ? zero(ExaModels.AbstractNode) : ExaModels.Null(x)

# zero/one for ExaNode types — needed by stdlib (e.g. tr) and general array ops
Base.zero(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(0)
Base.zero(::ExaNode) = ExaModels.Null(0)
Base.one(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(1)
Base.one(::ExaNode) = ExaModels.Null(1)

# adjoint/transpose for scalar ExaNode — nodes are real-valued, so both are identity
Base.adjoint(x::ExaModels.AbstractNode) = x
Base.adjoint(x::ExaModels.AbstractAdjointNode) = x
Base.adjoint(x::ExaModels.AbstractSecondAdjointNode) = x
Base.transpose(x::ExaModels.AbstractNode) = x
Base.transpose(x::ExaModels.AbstractAdjointNode) = x
Base.transpose(x::ExaModels.AbstractSecondAdjointNode) = x

# adjoint/transpose for matrices of ExaNode — materialize to plain Matrix
function Base.adjoint(A::MatExaNode)
    return [A[j, i] for i in axes(A, 2), j in axes(A, 1)]
end
function Base.transpose(A::MatExaNode)
    return [A[j, i] for i in axes(A, 2), j in axes(A, 1)]
end

# Dispatch pair constants for 3-way type combos
const _VEC_PAIRS = [
    (VecExaNode, VecExaNode),
    (AbstractVector{<:Real}, VecExaNode),
    (VecExaNode, AbstractVector{<:Real}),
]
const _MAT_PAIRS = [
    (MatExaNode, MatExaNode),
    (AbstractMatrix{<:Real}, MatExaNode),
    (MatExaNode, AbstractMatrix{<:Real}),
]

# ============================================================================
# Section D: Tier 2 operations — plain Julia scalar decompositions
# ============================================================================

# --- sum ---

function Base.sum(v::VecExaNode)
    s = v[1]
    for i in 2:length(v)
        s = s + v[i]
    end
    return s
end

# --- dot ---

for (T1, T2) in _VEC_PAIRS
    @eval function LinearAlgebra.dot(a::$T1, b::$T2)
        @assert length(a) == length(b)
        s = a[1] * b[1]
        for i in 2:length(a)
            s = s + a[i] * b[i]
        end
        return s
    end
end

# --- scalar * vector ---

for (T1, T2) in [(Real, VecExaNode), (ExaNode, AbstractVector{<:Real}), (ExaNode, VecExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a * b[i] for i in eachindex(b)]
end
for (T1, T2) in [(VecExaNode, Real), (AbstractVector{<:Real}, ExaNode), (VecExaNode, ExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a[i] * b for i in eachindex(a)]
end

# --- scalar * matrix ---

for (T1, T2) in [(Real, MatExaNode), (ExaNode, AbstractMatrix{<:Real}), (ExaNode, MatExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a * b[i, j] for i in axes(b, 1), j in axes(b, 2)]
end
for (T1, T2) in [(MatExaNode, Real), (AbstractMatrix{<:Real}, ExaNode), (MatExaNode, ExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a[i, j] * b for i in axes(a, 1), j in axes(a, 2)]
end

# --- matrix * vector (inline dot to avoid dispatch issues with view types) ---

function _dot_row(A, i, x)
    n = size(A, 2)
    s = A[i, 1] * x[1]
    for j in 2:n
        s = s + A[i, j] * x[j]
    end
    return s
end

for (T1, T2) in [
        (MatExaNode, VecExaNode), (AbstractMatrix{<:Real}, VecExaNode),
        (MatExaNode, AbstractVector{<:Real}),
    ]
    @eval function Base.:*(A::$T1, x::$T2)
        m = size(A, 1)
        @assert size(A, 2) == length(x)
        return [_dot_row(A, i, x) for i in 1:m]
    end
end

# --- matrix * matrix (inline dot to avoid dispatch issues) ---

function _dot_col(A, i, B, j)
    n = size(A, 2)
    s = A[i, 1] * B[1, j]
    for k in 2:n
        s = s + A[i, k] * B[k, j]
    end
    return s
end

for (T1, T2) in _MAT_PAIRS
    @eval function Base.:*(A::$T1, B::$T2)
        @assert size(A, 2) == size(B, 1)
        m, n = size(A, 1), size(B, 2)
        return [_dot_col(A, i, B, j) for i in 1:m, j in 1:n]
    end
end

# --- vector +/- ---

for op in (:+, :-)
    for (T1, T2) in _VEC_PAIRS
        @eval function Base.$op(a::$T1, b::$T2)
            @assert length(a) == length(b)
            return [$op(a[i], b[i]) for i in eachindex(a)]
        end
    end
end

# Win dispatch over Base's +(::Array, ::Array...) from arraymath.jl
for (T1, T2) in [(ExaNode, ExaNode), (Real, ExaNode), (ExaNode, Real)]
    @eval function Base.:+(a::Array{<:$T1, 1}, b::Array{<:$T2, 1})
        @assert length(a) == length(b)
        return [a[i] + b[i] for i in eachindex(a)]
    end
    @eval function Base.:+(A::Array{<:$T1, 2}, B::Array{<:$T2, 2})
        @assert size(A) == size(B)
        return [A[i, j] + B[i, j] for i in axes(A, 1), j in axes(A, 2)]
    end
end

# Unary minus for vector/matrix of nodes
function Base.:-(a::VecExaNode)
    return [-a[i] for i in eachindex(a)]
end
function Base.:-(A::MatExaNode)
    return [-A[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

# --- matrix +/- ---

for op in (:+, :-)
    for (T1, T2) in _MAT_PAIRS
        @eval function Base.$op(A::$T1, B::$T2)
            @assert size(A) == size(B)
            return [$op(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
        end
    end
end

# --- tr ---

function _tr_impl(A)
    @assert size(A, 1) == size(A, 2) "Matrix must be square for tr"
    n = size(A, 1)
    s = A[1, 1]
    for i in 2:n
        s = s + A[i, i]
    end
    return s
end

LinearAlgebra.tr(A::MatExaNode) = _tr_impl(A)
# More specific methods to win dispatch over stdlib's tr(::Matrix{T}) (Julia 1.10)
# and tr(::StridedMatrix{T}) (Julia 1.12+)
LinearAlgebra.tr(A::Matrix{<:ExaNode}) = _tr_impl(A)
LinearAlgebra.tr(A::StridedMatrix{<:ExaNode}) = _tr_impl(A)

# --- diag ---

function LinearAlgebra.diag(A::MatExaNode)
    n = minimum(size(A))
    return [A[i, i] for i in 1:n]
end

# --- diagm ---

function LinearAlgebra.diagm(v::VecExaNode)
    n = length(v)
    T = typeof(v[1])
    M = Matrix{Union{T, ExaModels.Null{Int}}}(undef, n, n)
    for i in 1:n, j in 1:n
        if i == j
            M[i, j] = v[i]
        else
            M[i, j] = ExaModels.Null(0)
        end
    end
    return M
end

# --- adjoint/transpose operations ---

# v' * w = dot(v, w)
for (TA, TV, TB) in [
        (ExaNode, VecExaNode, VecExaNode),
        (ExaNode, VecExaNode, AbstractVector{<:Real}),
        (Real, AbstractVector{<:Real}, VecExaNode),
    ]
    @eval function Base.:*(a::LinearAlgebra.Adjoint{<:$TA, <:$TV}, b::$TB)
        return LinearAlgebra.dot(parent(a), b)
    end
end

# v' * A
for (TA, TV, TB) in [
        (ExaNode, VecExaNode, MatExaNode),
        (Real, AbstractVector{<:Real}, MatExaNode),
    ]
    @eval function Base.:*(a::LinearAlgebra.Adjoint{<:$TA, <:$TV}, B::$TB)
        v = parent(a)
        @assert length(v) == size(B, 1)
        n = size(B, 2)
        return adjoint([LinearAlgebra.dot(v, [B[k, j] for k in 1:size(B, 1)]) for j in 1:n])
    end
end
function Base.:*(
        a::LinearAlgebra.Adjoint{<:ExaNode, <:VecExaNode},
        B::AbstractMatrix{<:Real},
    )
    v = parent(a)
    @assert length(v) == size(B, 1)
    n = size(B, 2)
    return adjoint([LinearAlgebra.dot(v, view(B, :, j)) for j in 1:n])
end

# ============================================================================
# Section E: Tier 1 operations — optimized scalar expansions
# ============================================================================

# --- det (specialized for small sizes) ---

# 1x1
function _det_1x1(A)
    return A[1, 1]
end

# 2x2: a11*a22 - a12*a21
function _det_2x2(A)
    a11 = A[1, 1]
    a12 = A[1, 2]
    a21 = A[2, 1]
    a22 = A[2, 2]
    return a11 * a22 - a12 * a21
end

# 3x3: Sarrus' rule (optimized expansion)
function _det_3x3(A)
    a11 = A[1, 1]
    a12 = A[1, 2]
    a13 = A[1, 3]
    a21 = A[2, 1]
    a22 = A[2, 2]
    a23 = A[2, 3]
    a31 = A[3, 1]
    a32 = A[3, 2]
    a33 = A[3, 3]
    return a11 * (a22 * a33 - a23 * a32) -
        a12 * (a21 * a33 - a23 * a31) +
        a13 * (a21 * a32 - a22 * a31)
end

# 4x4: cofactor expansion along first row
function _det_4x4(A)
    a11 = A[1, 1]
    a12 = A[1, 2]
    a13 = A[1, 3]
    a14 = A[1, 4]
    m11 = A[2, 2] * (A[3, 3] * A[4, 4] - A[3, 4] * A[4, 3]) -
        A[2, 3] * (A[3, 2] * A[4, 4] - A[3, 4] * A[4, 2]) +
        A[2, 4] * (A[3, 2] * A[4, 3] - A[3, 3] * A[4, 2])
    m12 = A[2, 1] * (A[3, 3] * A[4, 4] - A[3, 4] * A[4, 3]) -
        A[2, 3] * (A[3, 1] * A[4, 4] - A[3, 4] * A[4, 1]) +
        A[2, 4] * (A[3, 1] * A[4, 3] - A[3, 3] * A[4, 1])
    m13 = A[2, 1] * (A[3, 2] * A[4, 4] - A[3, 4] * A[4, 2]) -
        A[2, 2] * (A[3, 1] * A[4, 4] - A[3, 4] * A[4, 1]) +
        A[2, 4] * (A[3, 1] * A[4, 2] - A[3, 2] * A[4, 1])
    m14 = A[2, 1] * (A[3, 2] * A[4, 3] - A[3, 3] * A[4, 2]) -
        A[2, 2] * (A[3, 1] * A[4, 3] - A[3, 3] * A[4, 1]) +
        A[2, 3] * (A[3, 1] * A[4, 2] - A[3, 2] * A[4, 1])
    return a11 * m11 - a12 * m12 + a13 * m13 - a14 * m14
end

# General determinant via cofactor expansion (recursive, for N > 4)
function _det_recursive(A)
    n = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"
    if n == 1
        return _det_1x1(A)
    elseif n == 2
        return _det_2x2(A)
    elseif n == 3
        return _det_3x3(A)
    elseif n == 4
        return _det_4x4(A)
    end
    s = A[1, 1] * _det_recursive(A[2:end, 2:end])
    for j in 2:n
        cols = vcat(1:(j - 1), (j + 1):n)
        minor = _det_recursive(A[2:end, cols])
        if iseven(j)
            s = s - A[1, j] * minor
        else
            s = s + A[1, j] * minor
        end
    end
    return s
end

# Dispatch det for matrices containing ExaNode elements
function LinearAlgebra.det(A::MatExaNode)
    @assert size(A, 1) == size(A, 2) "Matrix must be square for det"
    return _det_recursive(A)
end

# --- norm ---

# 2-norm for vectors of nodes: sqrt(sum(xi^2))
function LinearAlgebra.norm(v::VecExaNode)
    s = v[1]^2
    for i in 2:length(v)
        s = s + v[i]^2
    end
    return sqrt(s)
end

# p-norm for vectors of nodes: (sum(abs(xi)^p))^(1/p)
function LinearAlgebra.norm(v::VecExaNode, p::Real)
    if p == 2
        return LinearAlgebra.norm(v)
    elseif p == 1
        s = abs(v[1])
        for i in 2:length(v)
            s = s + abs(v[i])
        end
        return s
    elseif p == Inf
        error("Inf-norm is not differentiable and not supported for ExaNode vectors")
    else
        s = abs(v[1])^p
        for i in 2:length(v)
            s = s + abs(v[i])^p
        end
        return s^(1 / p)
    end
end

# Frobenius norm for matrices of nodes: sqrt(sum(aij^2))
function LinearAlgebra.norm(A::MatExaNode)
    s = A[1, 1]^2
    for j in axes(A, 2), i in axes(A, 1)
        (i == 1 && j == 1) && continue
        s = s + A[i, j]^2
    end
    return sqrt(s)
end

# --- cross product (3D only) ---

for (T1, T2) in _VEC_PAIRS
    @eval function LinearAlgebra.cross(a::$T1, b::$T2)
        @assert length(a) == 3 && length(b) == 3 "Cross product requires 3D vectors"
        return [
            a[2] * b[3] - a[3] * b[2],
            a[3] * b[1] - a[1] * b[3],
            a[1] * b[2] - a[2] * b[1],
        ]
    end
end

# ============================================================================
# Section F: Vector constraint method
# ============================================================================

function ExaModels.add_con(
        c::ExaModels.ExaCore{T},
        v::VecExaNode;
        start = zero(T), lcon = zero(T), ucon = zero(T), kwargs...,
    ) where {T}
    n = length(v)
    _get(x::AbstractVector, i) = x[i]
    _get(x, _) = x
    c1 = nothing
    for i in 1:n
        c1, _ = ExaModels.constraint(
            c1, v[i]; start = _get(start, i), lcon = _get(lcon, i), ucon = _get(ucon, i),
            kwargs...,
        )
    end
    return c1
end

end # module ExaModelsOptimalControl
