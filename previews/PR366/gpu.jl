# # CPU and GPU acceleration
# One of the key features of ExaModels.jl is being able to evaluate derivatives either on multi-threaded CPUs or GPU accelerators. Currently, GPU acceleration is only tested for NVIDIA GPUs. If you'd like to use multi-threaded CPU acceleration, start julia with
# ```
# $ julia -t 4 # using 4 threads
# ```
# Also, if you're using NVIDIA GPUs, make sure to have installed appropriate drivers.

# Let's say that our CPU code is as follows.
using ExaModels
function luksan_vlcek_obj(x, i)
    return 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2
end

function luksan_vlcek_con(x, i)
    return 3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
           x[i]exp(x[i] - x[i+1]) - 3
end

function luksan_vlcek_x0(i)
    return mod(i, 2) == 1 ? -1.2 : 1.0
end

function luksan_vlcek_model(N)

    c = ExaCore(concrete = Val(true))
    @add_var(c, x, N; start = (luksan_vlcek_x0(i) for i = 1:N))
    @add_con(c, luksan_vlcek_con(x, i) for i = 1:(N-2))
    @add_obj(c, luksan_vlcek_obj(x, i) for i = 2:N)

    return ExaModel(c)
end

# Now we simply modify this by
function luksan_vlcek_model(N, backend = nothing)

    c = ExaCore(; backend = backend, concrete = Val(true)) # specify the backend
    @add_var(c, x, N; start = (luksan_vlcek_x0(i) for i = 1:N))
    @add_con(c, luksan_vlcek_con(x, i) for i = 1:(N-2))
    @add_obj(c, luksan_vlcek_obj(x, i) for i = 2:N)

    return ExaModel(c)
end


# The acceleration can be done simply by specifying the backend. In particular, for multi-threaded CPUs,
using ExaModels, NLPModelsIpopt, KernelAbstractions

m = luksan_vlcek_model(10, CPU())
ipopt(m)

# For NVIDIA GPUs, we can use `CUDABackend`. However, currently, there are not many optimization solvers that are capable of solving problems on GPUs. The only option right now is using [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl). To use this, first install
# ```julia
# import Pkg; Pkg.add("MadNLPGPU")
# ```
# Then, we can run:
# ```julia
# using CUDA, MadNLPGPU
# 
# m = luksan_vlcek_model(10, CUDABackend())
# madnlp(m)
# ```

# In the case we have arrays for the data, what we need to do is to simply convert the array types to the corresponding device array types. In particular,

function cuda_luksan_vlcek_model(N)
    c = ExaCore(; backend = CUDABackend(), concrete = Val(true))
    d1 = CuArray(1:(N-2))
    d2 = CuArray(2:N)
    d3 = CuArray([luksan_vlcek_x0(i) for i = 1:N])

    @add_var(c, x, N; start = d3)
    @add_con(c, luksan_vlcek_con(x, i) for i in d1)
    @add_obj(c, luksan_vlcek_obj(x, i) for i in d2)

    return ExaModel(c)
end

# ```julia
# m = cuda_luksan_vlcek_model(10)
# madnlp(m)
# ```
# Since ExaModels builds on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
# it can in principle target multiple hardware backends.
# The following backends are provided by the JuliaGPU ecosystem:
#
# - `CPU()` for multi-threaded CPU execution
# - `CUDABackend()` for NVIDIA GPUs ([CUDA.jl](https://github.com/JuliaGPU/CUDA.jl))
# - `ROCBackend()` for AMD GPUs ([AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl))
# - `OneAPIBackend()` for Intel GPUs ([oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl))
# - `MetalBackend()` for Apple GPUs ([Metal.jl](https://github.com/JuliaGPU/Metal.jl))
# - `OpenCLBackend()` for generic OpenCL devices ([OpenCL.jl](https://github.com/JuliaGPU/OpenCL.jl))
