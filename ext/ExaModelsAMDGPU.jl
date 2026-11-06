# Stores the model-independent device code once, for every package built on ExaModels.
# `KernelAbstractions` is a trigger so that ExaModels' extension for it is loaded while this
# one precompiles.
module ExaModelsAMDGPU

import ExaModels, AMDGPU, KernelAbstractions
import PrecompileTools

PrecompileTools.@compile_workload begin
    if AMDGPU.functional()
        ExaModels.precompile_backend(AMDGPU.ROCBackend())
    end
end

end # module
