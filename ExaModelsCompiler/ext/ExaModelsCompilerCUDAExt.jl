"""
Build-time half of `compile_library(...; device = :cuda)`.

Compiles ExaModels' per-element evaluation (`f(itr[I], x, θ)`, `grpass`, `jrpass`,
`hrpass0`) into PTX kernels, one set per block, and generates the library code that
launches them through the vendored MiniAOT runtime (`src/miniaot.jl`).

Kernels are generic: the block function, its data (a range by value, or a device
pointer and length), constraint dims and every array arrive as arguments.  So the
PTX depends only on TYPES, never on sizes, and one library serves every instance.

CUDA.jl is used here only.  The public pieces are `GPUCompiler.compile` and
`GPUCompiler.CompilerJob`; `CUDACore.compiler_config` and `CUDACore.methodinstance`
are the only internals touched, and only at build time, so a CUDA.jl change can
break a build loudly but never a compiled library.
"""
module ExaModelsCompilerCUDAExt

using ExaModelsCompiler, ExaModels
using CUDA
import CUDACore, GPUCompiler
const EM = ExaModels

const LP{T} = Core.LLVMPtr{T,1}
const PTX_ARCH = v"7.0"     # lowest target; the driver compiles the PTX for newer GPUs

# ── device code ─────────────────────────────────────────────────────────────────
@inline _tid() = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
@inline _w(p::LP{T}, n::Int) where {T} = CuDeviceArray{T,1,CUDA.AS.Global}(p, (n,))
@inline _it(r::UnitRange{Int}) = r
@inline _it(t::Tuple{LP{T},Int}) where {T} = _w(t[1], t[2])

function k_obj(F, it, L::Int, o::LP{Float64}, xp::LP{Float64}, xn::Int, tp::LP{Float64}, tn::Int)
    I = _tid()
    if I <= L
        itr = _it(it); x = _w(xp, xn); θ = _w(tp, tn); out = _w(o, 1)
        @inbounds CUDA.@atomic out[1] += F(itr[I], x, θ)
    end
    return
end
function k_grad(F, it, L::Int, bp::LP{Float64}, bn::Int, xp::LP{Float64}, xn::Int, tp::LP{Float64}, tn::Int)
    I = _tid()
    if I <= L
        itr = _it(it); x = _w(xp, xn); θ = _w(tp, tn); buf = _w(bp, bn)
        @inbounds EM.grpass(F(itr[I], EM.AdjointNodeSource(x), θ), F.comp1, buf, EM.offset1(F, I), 0, 1.0)
    end
    return
end
function k_hessobj(F, it, L::Int, hp::LP{Float64}, hn::Int, xp::LP{Float64}, xn::Int, tp::LP{Float64}, tn::Int, w::Float64)
    I = _tid()
    if I <= L
        itr = _it(it); x = _w(xp, xn); θ = _w(tp, tn); hv = _w(hp, hn)
        @inbounds EM.hrpass0(F(itr[I], EM.SecondAdjointNodeSource(x), θ), F.comp2, hv, nothing,
                             EM.offset2(F, I), 0, w, 0.0)
    end
    return
end
function k_con(F, it, D, L::Int, cp::LP{Float64}, cn::Int, xp::LP{Float64}, xn::Int, tp::LP{Float64}, tn::Int)
    I = _tid()
    if I <= L
        itr = _it(it); x = _w(xp, xn); θ = _w(tp, tn); c = _w(cp, cn)
        @inbounds CUDA.@atomic c[EM.offset0(F, itr, I, D)] += F(itr[I], x, θ)
    end
    return
end
function k_jac(F, it, D, L::Int, jp::LP{Float64}, jn::Int, xp::LP{Float64}, xn::Int, tp::LP{Float64}, tn::Int)
    I = _tid()
    if I <= L
        itr = _it(it); x = _w(xp, xn); θ = _w(tp, tn); jv = _w(jp, jn)
        @inbounds EM.jrpass(F(itr[I], EM.AdjointNodeSource(x), θ), F.comp1, EM.offset0(F, itr, I, D),
                            jv, nothing, EM.offset1(F, I), 0, 1.0)
    end
    return
end
function k_hesscon(F, it, D, L::Int, hp::LP{Float64}, hn::Int, xp::LP{Float64}, xn::Int, tp::LP{Float64}, tn::Int,
                   yp::LP{Float64}, yn::Int)
    I = _tid()
    if I <= L
        itr = _it(it); x = _w(xp, xn); θ = _w(tp, tn); hv = _w(hp, hn); y = _w(yp, yn)
        @inbounds EM.hrpass0(F(itr[I], EM.SecondAdjointNodeSource(x), θ), F.comp2, hv, nothing,
                             EM.offset2(F, I), 0, y[EM.offset0(F, itr, I, D)], 0.0)
    end
    return
end
function k_scatter(gp::LP{Float64}, gn::Int, bp::LP{Float64}, bn::Int, vp::LP{Int}, pp::LP{Int}, n::Int)
    I = _tid()
    if I <= n
        g = _w(gp, gn); buf = _w(bp, bn); gvar = _w(vp, n); gpos = _w(pp, n)
        @inbounds CUDA.@atomic g[gvar[I]] += buf[gpos[I]]
    end
    return
end

# ── PTX emission ────────────────────────────────────────────────────────────────
const DRIVER_SYMBOLS = ("vprintf",)          # unresolved externs the driver provides

function _param_bytes(p::AbstractString)
    m = match(r"\.b8\s+\S+\[(\d+)\]", p)
    m !== nothing && return parse(Int, m.captures[1])
    m = match(r"\.param\s+\.(?:u|s|b|f)(\d+)\s", p)
    m !== nothing && return parse(Int, m.captures[1]) ÷ 8
    error("cannot size PTX parameter: $p")
end

"""
    ptx(f, argtypes) -> (entry, state_bytes, ptx)

PTX for `f(argtypes...)`, checked against MiniAOT's calling convention: a leading
kernel-state parameter, then one parameter per non-ghost argument whose size equals
the host type's.  A mismatch here would otherwise be a silent miscompute.
"""
function ptx(f, argtypes::Tuple)
    job = GPUCompiler.CompilerJob(CUDACore.methodinstance(typeof(f), Tuple{argtypes...}),
                                  CUDACore.compiler_config(CUDA.device(); arch = PTX_ARCH))
    asm, _ = CUDACore.JuliaContext() do ctx
        GPUCompiler.compile(:asm, job)
    end
    for e in eachmatch(r"\.extern[^\n]*?(\w+)\(", asm)
        e.captures[1] in DRIVER_SYMBOLS || error("$(nameof(f)): PTX needs unresolved `$(e.captures[1])`")
    end
    m = match(r"\.entry\s+(\S+)\(([^)]*)\)", asm)
    m === nothing && error("$(nameof(f)): no .entry in PTX")
    params = split(strip(m.captures[2]), ",\n")
    host = [T for T in argtypes if sizeof(T) > 0]
    length(params) == length(host) + 1 ||
        error("$(nameof(f)): $(length(params)) PTX parameters for $(length(host)) arguments + state")
    for (p, T) in zip(params[2:end], host)
        _param_bytes(p) == sizeof(T) || error("$(nameof(f)): PTX parameter `$p` is not sizeof($T) = $(sizeof(T))")
    end
    return String(m.captures[1]), _param_bytes(params[1]), asm
end

# ── library source for one model ────────────────────────────────────────────────
_isrange(itr) = itr isa UnitRange{Int}
_itrtype(itr) = _isrange(itr) ? UnitRange{Int} : Tuple{LP{eltype(collect(itr))},Int}

function ExaModelsCompiler._device_source(::Val{:cuda}, p::AbstractString, probe)
    objs, cons = probe.objs, probe.cons
    (objs isa Tuple && cons isa Tuple) || error(
        "`$p`: device = :cuda needs the model's blocks stored as a Tuple; this model " *
        "has too many blocks for that ($(length(objs)) objective, $(length(cons)) constraint)")
    V = (LP{Float64}, Int)                          # a vector argument: pointer, length
    kernels = Tuple{String,Any,Tuple}[]
    for (k, b) in enumerate(objs)
        F = typeof(b.f); IT = _itrtype(b.itr)
        push!(kernels, ("obj_$k", k_obj, (F, IT, Int, LP{Float64}, V..., V...)))
        push!(kernels, ("grad_$k", k_grad, (F, IT, Int, V..., V..., V...)))
        push!(kernels, ("hessobj_$k", k_hessobj, (F, IT, Int, V..., V..., V..., Float64)))
    end
    for (k, b) in enumerate(cons)
        F = typeof(b.f); IT = _itrtype(b.itr); D = typeof(EM._constraint_dims(b))
        push!(kernels, ("con_$k", k_con, (F, IT, D, Int, V..., V..., V...)))
        push!(kernels, ("jac_$k", k_jac, (F, IT, D, Int, V..., V..., V...)))
        push!(kernels, ("hesscon_$k", k_hesscon, (F, IT, D, Int, V..., V..., V..., V...)))
    end
    push!(kernels, ("scatter", k_scatter, (V..., V..., LP{Int}, LP{Int}, Int)))

    io = IOBuffer()
    pr(s...) = println(io, s...)
    pr("    # ── device tier for `$p` (ExaModelsCompilerCUDAExt; PTX for sm_$(PTX_ARCH.major)$(PTX_ARCH.minor)+) ──")
    pr("    module KERNELS_$p")
    pr("    import ..MiniAOT")
    for (name, f, args) in kernels
        entry, sb, asm = ptx(f, args)
        pr("    const $name = MiniAOT.Kernel($(repr(entry)), $sb, $(repr(asm)))")
    end
    pr("    end")
    # Per-instance device state.  Block data types are read off the instance type,
    # so nothing here names a type the generated module cannot see.
    arrblocks = [(kind, k) for (kind, list) in ((:obj, objs), (:con, cons)) for (k, b) in enumerate(list) if !_isrange(b.itr)]
    blk(kind, k) = kind === :obj ? "m.objs[$k]" : "m.cons[$k]"
    pr("    const _MT_$p = eltype(typeof(MODELS_$p))")
    pr("    _itrT_$p(B) = eltype(fieldtype(B, :itr))")
    pr("    mutable struct GPU_$(p)_T")
    pr("        theta::MiniAOT.Buffer{Float64}; out::MiniAOT.Buffer{Float64}; gbuf::MiniAOT.Buffer{Float64}")
    pr("        gvar::MiniAOT.Buffer{Int}; gpos::MiniAOT.Buffer{Int}")
    for (kind, k) in arrblocks
        pr("        $(kind)_$k::MiniAOT.Buffer{_itrT_$p(fieldtype(fieldtype(_MT_$p, :$(kind)s), $k))}")
    end
    pr("    end")
    pr("    const GPU_$p = GPU_$(p)_T[]")
    pr("    function _mkgpu_$p(m)")
    # Gradient scatter plan: which variable each per-term buffer slot belongs to,
    # computed on the host with the same pass the KernelAbstractions backend uses.
    pr("        nb = 0")
    for (k, b) in enumerate(objs); pr("        nb += length($(blk(:obj, k)).itr) * $(blk(:obj, k)).f.o1step"); end
    pr("        gs = fill((0, 0), nb)")
    for (k, b) in enumerate(objs)
        pr("        let b = $(blk(:obj, k)), itr = vec(collect(b.itr))")
        pr("            for i in eachindex(itr)")
        pr("                ExaModels.grpass(b.f.f(itr[i], ExaModels.AdjointNodeSource(ExaModels.NaNSource{Float64}()), ExaModels.NaNSource{Float64}()),")
        pr("                                 b.f.comp1, gs, ExaModels.offset1(b.f, i), 0, NaN)")
        pr("            end")
        pr("        end")
    end
    pr("        GPU_$(p)_T(MiniAOT.Buffer(Vector{Float64}(m.θ)), MiniAOT.Buffer{Float64}(1), MiniAOT.Buffer{Float64}(nb),")
    pr("                  MiniAOT.Buffer(Int[first(q) for q in gs]), MiniAOT.Buffer(Int[last(q) for q in gs])",
       join([", MiniAOT.Buffer(vec(collect($(blk(kind, k)).itr)))" for (kind, k) in arrblocks]), ")")
    pr("    end")
    pr("    function _gpu_$p(id::Int)")
    pr("        while length(GPU_$p) < id")
    pr("            push!(GPU_$p, _mkgpu_$p(MODELS_$p[length(GPU_$p) + 1]))")
    pr("        end")
    pr("        g = GPU_$p[id]")
    pr("        # parameters are live instance state (P_set_value): refresh before every evaluation")
    pr("        isempty(MODELS_$p[id].θ) || MiniAOT.upload!(g.theta, Vector{Float64}(MODELS_$p[id].θ))")
    pr("        return g")
    pr("    end")
    pr("    _dev_$p(ptr::Ptr{Cdouble}) = reinterpret(Core.LLVMPtr{Float64,1}, ptr)")
    itarg(kind, k, b) = _isrange(b.itr) ? "$(blk(kind, k)).itr" : "(MiniAOT.devptr(g.$(kind)_$k), length(g.$(kind)_$k))"
    L(kind, k, b) = _isrange(b.itr) ? "length($(blk(kind, k)).itr)" : "length(g.$(kind)_$k)"
    common = "x, n, MiniAOT.devptr(g.theta), length(g.theta)"
    callback(name, sig, body) = begin
        pr("    Base.@ccallable function $(p)_$name($sig)::Cint")
        pr("        _valid_$p(id) || return Cint(1)")
        pr("        try")
        pr("            m = MODELS_$p[Int(id)]; g = _gpu_$p(Int(id))")
        pr("            n = m.meta.nvar; x = _dev_$p(xh)")
        print(io, body)
        pr("            MiniAOT.synchronize()")
        pr("            return Cint(0)")
        pr("        catch")
        pr("            return Cint(2)")
        pr("        end")
        pr("    end")
    end
    pr("    # witness of the device tier: 1 = CUDA, device 0, primary context")
    pr("    Base.@ccallable function $(p)_device(id::Cint)::Cint")
    pr("        return _valid_$p(id) ? Cint(1) : Cint(-1)")
    pr("    end")
    b_obj = join(["            MiniAOT.launch(KERNELS_$p.obj_$k, $(L(:obj,k,b)), $(blk(:obj,k)).f, $(itarg(:obj,k,b)), $(L(:obj,k,b)), MiniAOT.devptr(g.out), $common)\n" for (k, b) in enumerate(objs)])
    callback("obj_d", "id::Cint, xh::Ptr{Cdouble}, out::Ptr{Cdouble}",
             "            MiniAOT.zero!(MiniAOT.devptr(g.out), 1)\n" * b_obj *
             "            MiniAOT.synchronize(); unsafe_store!(out, MiniAOT.download(g.out)[1])\n")
    b_grad = join(["            MiniAOT.launch(KERNELS_$p.grad_$k, $(L(:obj,k,b)), $(blk(:obj,k)).f, $(itarg(:obj,k,b)), $(L(:obj,k,b)), MiniAOT.devptr(g.gbuf), length(g.gbuf), $common)\n" for (k, b) in enumerate(objs)])
    callback("grad_d", "id::Cint, xh::Ptr{Cdouble}, gh::Ptr{Cdouble}",
             "            gd = _dev_$p(gh); MiniAOT.zero!(gd, n); MiniAOT.zero!(MiniAOT.devptr(g.gbuf), length(g.gbuf))\n" * b_grad *
             "            MiniAOT.launch(KERNELS_$p.scatter, length(g.gvar), gd, n, MiniAOT.devptr(g.gbuf), length(g.gbuf), MiniAOT.devptr(g.gvar), MiniAOT.devptr(g.gpos), length(g.gvar))\n")
    b_con = join(["            MiniAOT.launch(KERNELS_$p.con_$k, $(L(:con,k,b)), $(blk(:con,k)).f, $(itarg(:con,k,b)), ExaModels._constraint_dims($(blk(:con,k))), $(L(:con,k,b)), cd, m.meta.ncon, $common)\n" for (k, b) in enumerate(cons)])
    callback("cons_d", "id::Cint, xh::Ptr{Cdouble}, ch::Ptr{Cdouble}",
             "            cd = _dev_$p(ch); MiniAOT.zero!(cd, m.meta.ncon)\n" * b_con)
    b_jac = join(["            MiniAOT.launch(KERNELS_$p.jac_$k, $(L(:con,k,b)), $(blk(:con,k)).f, $(itarg(:con,k,b)), ExaModels._constraint_dims($(blk(:con,k))), $(L(:con,k,b)), jd, m.meta.nnzj, $common)\n" for (k, b) in enumerate(cons)])
    callback("jac_d", "id::Cint, xh::Ptr{Cdouble}, vh::Ptr{Cdouble}",
             "            jd = _dev_$p(vh); MiniAOT.zero!(jd, m.meta.nnzj)\n" * b_jac)
    b_hess = join(vcat(
        ["            MiniAOT.launch(KERNELS_$p.hessobj_$k, $(L(:obj,k,b)), $(blk(:obj,k)).f, $(itarg(:obj,k,b)), $(L(:obj,k,b)), hd, m.meta.nnzh, $common, obj_weight)\n" for (k, b) in enumerate(objs)],
        ["            MiniAOT.launch(KERNELS_$p.hesscon_$k, $(L(:con,k,b)), $(blk(:con,k)).f, $(itarg(:con,k,b)), ExaModels._constraint_dims($(blk(:con,k))), $(L(:con,k,b)), hd, m.meta.nnzh, $common, _dev_$p(yh), m.meta.ncon)\n" for (k, b) in enumerate(cons)]))
    callback("hess_d", "id::Cint, xh::Ptr{Cdouble}, yh::Ptr{Cdouble}, obj_weight::Cdouble, vh::Ptr{Cdouble}",
             "            hd = _dev_$p(vh); MiniAOT.zero!(hd, m.meta.nnzh)\n" * b_hess)
    return String(take!(io))
end

end # module
