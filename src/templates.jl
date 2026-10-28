# Convert array to the appropriate device using Adapt.jl.
convert_array(v, ::Nothing) = v
convert_array(v, backend) = adapt(backend, v)

# Recursively replace Float64 with Float32 in scalars, containers, and the
# fields of arbitrary structs. Used by backend extensions (Metal, oneAPI) that
# either reject Float64 outright or perform poorly on it. Type-stable via
# multiple dispatch on leaf types and Val(fieldcount(T)) on the generic struct
# path.
replace_float_64_by_32(x::Float64) = Float32(x)
replace_float_64_by_32(x::Tuple) = map(replace_float_64_by_32, x)
replace_float_64_by_32(x::NamedTuple) = map(replace_float_64_by_32, x)
replace_float_64_by_32(x::AbstractArray{Float64}) = Float32.(x)
replace_float_64_by_32(x::AbstractArray{T}) where {T} =
    _contains_float64(T) ? replace_float_64_by_32.(x) : x
@inline replace_float_64_by_32(x::T) where {T} =
    _contains_float64(T) ? _rebuild_float_32(x, Val(fieldcount(T))) : x

# Whether a Float64 can occur anywhere inside a value of type `T`. Values of
# Float64-free types are returned as they are: there is nothing to rewrite, and
# the rebuild is not a safe no-op for them. On Julia 1.13.2 the broadcast of
# the struct rebuild over an array of nested isbits structs returns corrupt
# elements (the scalar call is right, the broadcast is not; 1.12 is right on
# both), measured 2026-10-19 on an M1 Air with the test suite's I1/I2 index
# structs: it was the augmentation iterators of luksan_struct arriving on the
# device as garbage that took the metal CI leg red on 1.13.
_contains_float64(::Type{Float64}) = true
_contains_float64(::Type{<:AbstractArray{T}}) where {T} = _contains_float64(T)
_contains_float64(::Type{T}) where {T <: Tuple} = any(_contains_float64, fieldtypes(T))
_contains_float64(::Type{<:NamedTuple{names, T}}) where {names, T} = _contains_float64(T)
function _contains_float64(::Type{T}) where {T}
    isconcretetype(T) || return true      # abstract or unknown: rewrite to be safe
    isprimitivetype(T) && return false
    return any(_contains_float64, fieldtypes(T))
end
@inline _rebuild_float_32(x, ::Val{0}) = x
@inline function _rebuild_float_32(x::T, ::Val{N}) where {T, N}
    # Rebuild via the type's unparameterized constructor (T.name.wrapper), so a
    # parametric struct with Float64 fields reconstructs with Float32 ones.
    # Try the explicit Float32 parameterization first, then constructor
    # inference; structs whose constructors accept neither (e.g. concretely
    # Float64-typed fields) are returned unchanged.
    W = T.name.wrapper
    vals = ntuple(i -> replace_float_64_by_32(getfield(x, i)), Val(N))
    try
        return W{Float32}(vals...)
    catch
        try
            return W(vals...)
        catch
            return x
        end
    end
end

# to avoid type privacy
sort!(array; kwargs...) = Base.sort!(array; kwargs...)

# Placeholder for ExaModels.Optimizer
global Optimizer

# Placeholder for ExaModels.SIMDMode
global SIMDMode
