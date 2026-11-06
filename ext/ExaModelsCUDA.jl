# Stores the model-independent device code once, for every package built on ExaModels.
# `KernelAbstractions` is a trigger so that ExaModels' extension for it is loaded while this
# one precompiles.
module ExaModelsCUDA

import ExaModels, CUDA, KernelAbstractions
import PrecompileTools

PrecompileTools.@compile_workload begin
    if CUDA.functional()
        ExaModels.precompile_backend(CUDA.CUDABackend())
    end
end

end # module
