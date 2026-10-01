@inline (a::Pair{P,S} where {P,S<:AbstractNode})(i, x, θ) = a.second(i, x, θ)

"""
    Compressor{I}

Data structure for the sparse index

# Fields:
- `inner::I`: stores the sparse index as a tuple form
"""
struct Compressor{I}
    inner::I
end
@inline (i::Compressor{I})(n) where {I} = @inbounds i.inner[n]

struct SIMDFunction{F,C1,C2}
    f::F
    comp1::C1
    comp2::C2
    o0::Int
    o1::Int
    o2::Int
    o1step::Int
    o2step::Int
end

@inline (sf::SIMDFunction{F,C1,C2})(i, x, θ) where {F,C1,C2} = sf.f(i, x, θ)
@inline (sf::SIMDFunction{F,C1,C2})(i, x, θ) where {F <: Real,C1,C2} = sf.f

"""
    SIMDFunction(gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0)

Returns a `SIMDFunction` using the `gen`.

# Arguments:
- `gen`: an iterable function specified in `Base.Generator` format
- `o0`: offset for the function evaluation
- `o1`: offset for the derivative evalution
- `o2`: offset for the second-order derivative evalution
"""
@inline function SIMDFunction(T, gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0)
    _simdfunction(T, gen.f(DataSource()), o0, o1, o2)
end

@inline function _simdfunction(T, f::F, o0, o1, o2) where {F<:Real}
    f = replace_T(T, f)
    SIMDFunction(
        f,
        Compressor{Tuple{}}(()),
        Compressor{Tuple{}}(()),
        o0,
        o1,
        o2,
        0,
        0,
    )
end

# Identity-based deduplication — avoids Set{Any} (which juliac can't resolve for Any elements).
# Sparsity patterns are tiny, so O(n^2) is fine.
function _ident_unique(v::Vector)
    result = similar(v, 0)
    for x in v
        new_x = true
        for y in result
            y === x && (new_x = false; break)
        end
        new_x && push!(result, x)
    end
    return result
end

@inline function _simdfunction(T, f, o0, o1, o2)
    f = replace_T(T, f)

    d = f(Identity(), AdjointNodeSource(NaNSource{T}()), NaNSource{T}())
    raw1 = Any[]
    ExaModels.grpass(d, nothing, nothing, nothing, raw1, T(NaN))

    t = f(Identity(), SecondAdjointNodeSource(NaNSource{T}()), NaNSource{T}())
    raw2 = Any[]
    ExaModels.hrpass0(t, nothing, nothing, nothing, nothing, raw2, T(NaN), T(NaN))

    unique1 = _ident_unique(raw1)
    o1step = length(unique1)
    mapping1 = Int[findfirst(y -> y === x, unique1) for x in raw1]
    c1 = Compressor(ntuple(i -> mapping1[i], _gr_val(typeof(d))))

    unique2 = _ident_unique(raw2)
    o2step = length(unique2)
    mapping2 = Int[findfirst(y -> y === x, unique2) for x in raw2]
    c2 = Compressor(ntuple(i -> mapping2[i], _hr0_val(typeof(t))))

    SIMDFunction(f, c1, c2, o0, o1, o2, o1step, o2step)
end

# === Val-based compile-time NTuple size computation (juliac-compatible, no @generated) ===
# Each function returns Val{N}() where N is encoded in the return type via dispatch.
# This lets ntuple(f, Val{N}()) produce a concrete NTuple{N,...}.

@inline _add_vals(::Val{A}, ::Val{B}) where {A,B} = Val(A + B)

# --- Gradient count: mirrors grpass leaf dispatch ---
_gr_val(::Type{<:AdjointNodeVar}) = Val(1)
_gr_val(::Type{<:AdjointNull}) = Val(0)
_gr_val(::Type{<:Real}) = Val(0)
_gr_val(::Type{<:DataIndexed}) = Val(0)
_gr_val(::Type{AdjointNode1{F,T,I}}) where {F,T,I} = _gr_val(I)
_gr_val(::Type{AdjointNode2{F,T,I1,I2}}) where {F,T,I1,I2} =
    _add_vals(_gr_val(I1), _gr_val(I2))

# --- Hessian count: mirrors hrpass0 → hrpass → hdrpass dispatch ---
const _LinearHr1F = Union{
    FirstFixed{typeof(*)}, SecondFixed{typeof(*)},
    FirstFixed{typeof(+)}, SecondFixed{typeof(+)},
    FirstFixed{typeof(-)}, SecondFixed{typeof(-)},
    typeof(+), typeof(-),
}

_hr0_val(::Type{<:SecondAdjointNodeVar}) = Val(0)
_hr0_val(::Type{<:SecondAdjointNull}) = Val(0)
_hr0_val(::Type{<:Real}) = Val(0)
_hr0_val(::Type{SecondAdjointNode1{F,T,I}}) where {F<:_LinearHr1F,T,I} = _hr0_val(I)
_hr0_val(::Type{SecondAdjointNode2{typeof(+),T,I1,I2}}) where {T,I1,I2} =
    _add_vals(_hr0_val(I1), _hr0_val(I2))
_hr0_val(::Type{SecondAdjointNode2{typeof(-),T,I1,I2}}) where {T,I1,I2} =
    _add_vals(_hr0_val(I1), _hr0_val(I2))
_hr0_val(T::Type) = _hrpass_val(T)  # fallthrough to hrpass

_hrpass_val(::Type{<:SecondAdjointNull}) = Val(0)
_hrpass_val(::Type{<:Real}) = Val(0)
_hrpass_val(::Type{<:SecondAdjointNodeVar}) = Val(1)
_hrpass_val(::Type{SecondAdjointNode1{F,T,I}}) where {F,T,I} = _hrpass_val(I)
_hrpass_val(::Type{SecondAdjointNode2{F,T,I1,I2}}) where {F,T,I1,I2} =
    _add_vals(_add_vals(_hrpass_val(I1), _hrpass_val(I2)), _hdrpass_val(I1, I2))

_hdrpass_val(::Type{<:SecondAdjointNull}, ::Type) = Val(0)
_hdrpass_val(::Type, ::Type{<:SecondAdjointNull}) = Val(0)
_hdrpass_val(::Type{<:SecondAdjointNull}, ::Type{<:SecondAdjointNull}) = Val(0)
_hdrpass_val(::Type{<:SecondAdjointNodeVar}, ::Type{<:SecondAdjointNodeVar}) = Val(1)
_hdrpass_val(::Type{<:SecondAdjointNodeVar}, ::Type{SecondAdjointNode1{F,T,I}}) where {F,T,I} =
    _hdrpass_fixedvar_val(I)
_hdrpass_val(::Type{SecondAdjointNode1{F,T,I}}, ::Type{<:SecondAdjointNodeVar}) where {F,T,I} =
    _hdrpass_fixedvar_val(I)
_hdrpass_val(::Type{SecondAdjointNode1{F1,T1,I1}}, ::Type{SecondAdjointNode1{F2,T2,I2}}) where {F1,T1,I1,F2,T2,I2} =
    _hdrpass_val(I1, I2)
_hdrpass_val(::Type{SecondAdjointNode2{F,T,I1,I2}}, ::Type{SecondAdjointNode2{G,U,J1,J2}}) where {F,T,I1,I2,G,U,J1,J2} =
    _add_vals(_add_vals(_hdrpass_val(I1,J1), _hdrpass_val(I1,J2)),
              _add_vals(_hdrpass_val(I2,J1), _hdrpass_val(I2,J2)))
_hdrpass_val(::Type{SecondAdjointNode2{F,T,I1,I2}}, ::Type{SecondAdjointNode1{G,U,J}}) where {F,T,I1,I2,G,U,J} =
    _add_vals(_hdrpass_val(I1, J), _hdrpass_val(I2, J))
_hdrpass_val(::Type{SecondAdjointNode1{F,T,I}}, ::Type{SecondAdjointNode2{G,U,J1,J2}}) where {F,T,I,G,U,J1,J2} =
    _add_vals(_hdrpass_val(I, J1), _hdrpass_val(I, J2))
_hdrpass_val(::Type{<:SecondAdjointNodeVar}, ::Type{SecondAdjointNode2{F,T,I1,I2}}) where {F,T,I1,I2} =
    _add_vals(_hdrpass_fixedvar_val(I1), _hdrpass_fixedvar_val(I2))
_hdrpass_val(::Type{SecondAdjointNode2{F,T,I1,I2}}, ::Type{<:SecondAdjointNodeVar}) where {F,T,I1,I2} =
    _add_vals(_hdrpass_fixedvar_val(I1), _hdrpass_fixedvar_val(I2))

_hdrpass_fixedvar_val(::Type{<:SecondAdjointNodeVar}) = Val(1)
_hdrpass_fixedvar_val(::Type{<:Union{SecondAdjointNull,Real}}) = Val(0)
_hdrpass_fixedvar_val(::Type{SecondAdjointNode1{F,T,I}}) where {F,T,I} = _hdrpass_fixedvar_val(I)
_hdrpass_fixedvar_val(::Type{SecondAdjointNode2{F,T,I1,I2}}) where {F,T,I1,I2} =
    _add_vals(_hdrpass_fixedvar_val(I1), _hdrpass_fixedvar_val(I2))


@inline replace_T(t, n::AbstractNode) = n
@inline replace_T(t, n::Real) = n
@inline replace_T(t, n::AbstractFloat) = t(n)
# For Constant{T}, the value lives in the type parameter, so we re-wrap the
# type-converted value in a new Constant rather than touching a field.
@inline replace_T(t, ::Constant{n}) where {n} = Constant(replace_T(t, n))
@inline replace_T(t, ::Val{n}) where {n} = Val(replace_T(t, n))
@inline replace_T(t, (a,b)::Pair) = a => replace_T(t, b)
@inline function replace_T(t, n::Node1{F,I}) where {F, I}
    i = replace_T(t, n.inner)
    return Node1{F,typeof(i)}(i)
end
@inline function replace_T(t, n::Node2{F,I1,I2}) where {F, I1, I2}
    i1 = replace_T(t, n.inner1)
    i2 = replace_T(t, n.inner2)
    return Node2{F,typeof(i1),typeof(i2)}(i1, i2)
end
@inline replace_T(t, n::Null{T}) where T <: Real = Null{t}(t(n.value))
@inline function replace_T(t, n::SumNode{I}) where {I}
    inners = map(x -> replace_T(t, x), n.inners)
    SumNode(inners)
end
@inline function replace_T(t, n::ProdNode{I}) where {I}
    inners = map(x -> replace_T(t, x), n.inners)
    ProdNode(inners)
end
