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
replace_float_64_by_32(x::AbstractArray) = replace_float_64_by_32.(x)
@inline replace_float_64_by_32(x::T) where {T} = _rebuild_float_32(x, Val(fieldcount(T)))
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
