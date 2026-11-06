# Stores the model-independent device code once, for every package built on ExaModels.
# `KernelAbstractions` is a trigger so that ExaModels' extension for it is loaded while this
# one precompiles.
module ExaModelsCUDA

import ExaModels, CUDA, KernelAbstractions
import PrecompileTools

PrecompileTools.@compile_workload begin
    if CUDA.functional()
        ExaModels.precompile_backend(CUDA.CUDABackend())
        # Reached by runtime dispatch in a session's first launch and synchronization. Some
        # live in packages that are not dependencies here, so they come from the loaded modules.
        # These are internals, so a version without one only loses the stored call.
        try
            loaded(name) = only(m for (id, m) in Base.loaded_modules if id.name == name)
            core, compiler = CUDA.CUDACore, loaded("CUDA_Compiler_jll")
            toolbox, arrays = loaded("GPUToolbox"), loaded("GPUArrays")
            wait = toolbox.WaitRequest{typeof(core.worker_synchronize), Tuple{core.CuEvent, core.CuContext}, core.cudaError_enum}
            precompile(NamedTuple{(:backend,)}, (Tuple{core.CUDAKernels.CUDABackend},))
            precompile(compiler.is_available, ())
            Base.invokelatest(compiler.ptxas)
            precompile(toolbox.perform!, (wait,))
            # Last: `Base.ScopedValues` lacks these below Julia 1.13.
            scoped = Base.ScopedValues.ScopedValue{Union{Nothing, arrays.AllocCache}}
            precompile(Base.ScopedValues.hasdefault, (scoped,))
            precompile(Base.ScopedValues.getdefault, (scoped,))
        catch err
            @debug "skipping a stored call" exception = err
        end
    end
end

end # module
