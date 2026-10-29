module ExaModelsMetal

import ExaModels, Metal

ExaModels.sort!(array::A; lt = isless) where {A<:Metal.MtlArray} =
    copyto!(array, sort!(Array(array); lt = lt))

ExaModels.convert_array(v::Metal.MtlArray, ::Metal.MetalBackend) = v
# A recipe's iterator is an argument placeholder until the core is
# instantiated; `instantiate` converts the computed iterator then.
ExaModels.convert_array(v::ExaModels.AbstractArgNode, ::Metal.MetalBackend) = v
ExaModels.convert_array(v, ::Metal.MetalBackend) =
    Metal.MtlArray(ExaModels.replace_float_64_by_32.(v))

ExaModels.default_T(::Metal.MetalBackend) = Float32

end # module
