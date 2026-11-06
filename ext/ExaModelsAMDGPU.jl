# Stores the model-independent device code once, for every package built on ExaModels.
# `KernelAbstractions` is a trigger so that ExaModels' extension for it is loaded while this
# one precompiles.
module ExaModelsAMDGPU

import ExaModels, AMDGPU, KernelAbstractions
import PrecompileTools

PrecompileTools.@compile_workload begin
    if AMDGPU.functional()
        ExaModels.precompile_backend(AMDGPU.ROCBackend())
        # Reached by runtime dispatch in a session's first synchronization. GPUToolbox is not
        # a dependency here, so it comes from the loaded modules.
        # These are internals, so a version without one only loses the stored call.
        try
            loaded(name) = only(m for (id, m) in Base.loaded_modules if id.name == name)
            toolbox = loaded("GPUToolbox")
            wait = toolbox.WaitRequest{typeof(AMDGPU.HIP.worker_device_synchronize), AMDGPU.HIP.HIPDevice, AMDGPU.HIP.hipError_t}
            precompile(toolbox.perform!, (wait,))
        catch err
            @debug "skipping a stored call" exception = err
        end
    end
end

end # module
