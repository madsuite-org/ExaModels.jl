module ExaModelsOpenCL

import ExaModels, OpenCL

ExaModels.sort!(array::A; lt = isless) where {A <: OpenCL.CLArray} =
    copyto!(array, sort!(Array(array); lt = lt))

# below is type piracy
function Base.findall(f::F, bitarray::A) where {F <: Function, A <: OpenCL.CLArray}
    a = Array(bitarray)
    b = findall(f, a)
    c = similar(bitarray, eltype(b), length(b))
    return copyto!(c, b)
end
Base.findall(bitarray::A) where {A <: OpenCL.CLArray} = Base.findall(identity, bitarray)

end # module
