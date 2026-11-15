module ExaModelsOneAPI

import ExaModels, oneAPI

if pkgversion(oneAPI) < v"2.6"

    ExaModels.convert_array(v, backend::oneAPI.oneAPIBackend) = oneAPI.oneArray(v)

    ExaModels.sort!(array::A; lt = isless) where {A <: oneAPI.oneArray} =
        copyto!(array, sort!(Array(array); lt = lt))

    # below is type piracy
    function Base.findall(f::F, bitarray::A) where {F <: Function, A <: oneAPI.oneArray}
        a = Array(bitarray)
        b = findall(f, a)
        c = similar(bitarray, eltype(b), length(b))
        return copyto!(c, b)
    end
    Base.findall(bitarray::A) where {A <: oneAPI.oneArray} = Base.findall(identity, bitarray)
end

end # module
