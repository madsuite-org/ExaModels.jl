# MiniAOT: run PTX kernels from a trimmed library, through the CUDA driver only.
#
# Vendored into every library compiled with `device = :cuda`; it has no
# dependencies, so nothing here moves when CUDA.jl does.  The contract with the
# build step (ext/ExaModelsCompilerCUDAExt.jl) is:
#   - a kernel's first PTX parameter is CUDA.jl's kernel state, passed as zeros;
#   - every other argument is an isbits value passed by value, in Julia layout;
#   - zero-size arguments have no PTX parameter and are skipped.
# A kernel that throws therefore faults instead of reporting where; the
# callbacks it serves are written not to throw.
module MiniAOT

const libcuda = "libcuda.so.1"

struct CUDAError <: Exception
    call::Symbol
    code::Cint
end
@inline check(r::Cint, call::Symbol) = (r == 0 || throw(CUDAError(call, r)); nothing)

const CONTEXT = Ref{Ptr{Cvoid}}(C_NULL)

# The primary context of device 0: the same context CUDA.jl and the CUDA
# runtime use, so device pointers from the caller are valid here.
function init()
    CONTEXT[] == C_NULL || return nothing
    check(ccall((:cuInit, libcuda), Cint, (Cuint,), 0), :cuInit)
    dev = Ref{Cint}(0)
    check(ccall((:cuDeviceGet, libcuda), Cint, (Ptr{Cint}, Cint), dev, 0), :cuDeviceGet)
    ctx = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:cuDevicePrimaryCtxRetain, libcuda), Cint, (Ptr{Ptr{Cvoid}}, Cint), ctx, dev[]),
          :cuDevicePrimaryCtxRetain)
    check(ccall((:cuCtxSetCurrent, libcuda), Cint, (Ptr{Cvoid},), ctx[]), :cuCtxSetCurrent)
    CONTEXT[] = ctx[]
    return nothing
end

mutable struct Buffer{T}
    ptr::UInt64
    len::Int
    function Buffer{T}(len::Integer) where {T}
        init()
        p = Ref{UInt64}(0)
        len > 0 && check(ccall((:cuMemAlloc_v2, libcuda), Cint, (Ptr{UInt64}, Csize_t), p,
                               len * sizeof(T)), :cuMemAlloc_v2)
        b = new{T}(p[], Int(len))
        finalizer(free!, b)
        return b
    end
end
Buffer(a::Vector{T}) where {T} = upload!(Buffer{T}(length(a)), a)
Base.length(b::Buffer) = b.len
free!(b::Buffer) = (b.ptr == 0 || ccall((:cuMemFree_v2, libcuda), Cint, (UInt64,), b.ptr); b.ptr = 0; nothing)
devptr(b::Buffer{T}) where {T} = reinterpret(Core.LLVMPtr{T,1}, b.ptr)

function upload!(d::Buffer{T}, s::Vector{T}) where {T}
    isempty(s) && return d
    GC.@preserve s check(ccall((:cuMemcpyHtoD_v2, libcuda), Cint, (UInt64, Ptr{T}, Csize_t),
                               d.ptr, s, sizeof(T) * length(s)), :cuMemcpyHtoD_v2)
    return d
end
function download(s::Buffer{T}) where {T}
    d = Vector{T}(undef, s.len)
    isempty(d) && return d
    GC.@preserve d check(ccall((:cuMemcpyDtoH_v2, libcuda), Cint, (Ptr{T}, UInt64, Csize_t),
                               d, s.ptr, sizeof(T) * s.len), :cuMemcpyDtoH_v2)
    return d
end
function zero!(p::Core.LLVMPtr{T,1}, n::Int) where {T}
    n > 0 && check(ccall((:cuMemsetD8_v2, libcuda), Cint, (UInt64, UInt8, Csize_t),
                         reinterpret(UInt64, p), 0x00, sizeof(T) * n), :cuMemsetD8_v2)
    return nothing
end

mutable struct Kernel
    entry::String
    state_bytes::Int
    ptx::Vector{UInt8}           # NUL-terminated
    func::Ptr{Cvoid}
end
Kernel(entry::String, state_bytes::Int, ptx::String) =
    Kernel(entry, state_bytes, Vector{UInt8}(ptx * "\0"), C_NULL)

function getfunction(k::Kernel)
    k.func == C_NULL || return k.func
    init()
    mod = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve k check(ccall((:cuModuleLoadData, libcuda), Cint, (Ptr{Ptr{Cvoid}}, Ptr{UInt8}),
                               mod, k.ptx), :cuModuleLoadData)
    fn = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:cuModuleGetFunction, libcuda), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}, Cstring),
                fn, mod[], k.entry), :cuModuleGetFunction)
    k.func = fn[]
    return k.func
end

# zero-size arguments have no PTX parameter
_nonghost(::Tuple{}) = ()
_nonghost(a::Tuple) = sizeof(typeof(first(a))) == 0 ? _nonghost(Base.tail(a)) :
                      (first(a), _nonghost(Base.tail(a))...)

"""
    launch(k, n, args...)

Launch `k` over `n` threads, 256 per block, on the default stream.
"""
function launch(k::Kernel, n::Int, args::Vararg{Any,N}) where {N}
    n <= 0 && return nothing
    f = getfunction(k)
    refs = map(Ref, _nonghost(args))
    state = zeros(UInt8, max(k.state_bytes, 1))
    GC.@preserve refs state begin
        params = Ptr{Cvoid}[pointer(state), map(r -> Base.unsafe_convert(Ptr{Cvoid}, r), refs)...]
        GC.@preserve params check(ccall((:cuLaunchKernel, libcuda), Cint,
                    (Ptr{Cvoid}, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, Ptr{Cvoid},
                     Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
                    f, Cuint(cld(n, 256)), 1, 1, Cuint(256), 1, 1, 0, C_NULL,
                    params, C_NULL), :cuLaunchKernel)
    end
    return nothing
end

synchronize() = (init(); check(ccall((:cuCtxSynchronize, libcuda), Cint, ()), :cuCtxSynchronize))

end # module MiniAOT
