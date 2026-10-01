# Buffered subexpressions on KernelAbstractions backends.
#
# Every serial sweep of the CPU implementation becomes a segmented-scatter
# kernel: program entries are sorted by destination at model build, segment
# pointers are computed with the same (dstu, ptr) pattern as compress_to_dense,
# and each workitem owns one destination, so no atomics are needed (which also
# keeps Float32-only backends viable).  Tree-evaluation kernels (stage sync,
# stage Jacobian fills, stage Hessian replays) are plain per-element launches
# with slot-exclusive writes.
#
# Seed accumulation differs from the CPU path: device passes write plain
# buffers, so the λ seeds come from a separate weighted first-order pass
# (σ over objectives, y over constraints) scattered from the compressed slots,
# and the μ seeds are scattered from the dead (-j,-j) Hessian slots, which the
# generic buffered-leaf pass methods populate when driven without a HessTarget.

# ── segmented scatter programs ────────────────────────────────────────────────

struct SegCopy{VI}
    dstu::VI
    ptr::VI
    src::VI
end
struct SegProd{VI}
    dstu::VI
    ptr::VI
    a::VI
    b::VI
end
struct SegProdC{VI,VT}
    dstu::VI
    ptr::VI
    a::VI
    b::VI
    c::VT
end

_seg_isempty(s) = isempty(s.dstu)

function _segment_perm(dst::Vector{Int})
    p = sortperm(dst)
    d = dst[p]
    dstu = Int[]
    ptr = Int[1]
    for k in eachindex(d)
        if k == 1 || d[k] != d[k-1]
            push!(dstu, d[k])
            push!(ptr, ptr[end])
        end
        ptr[end] += 1
    end
    return p, dstu, ptr
end

_dev(x0, v::Vector{Int}) = copyto!(similar(x0, Int, length(v)), v)
_dev(x0, v::Vector{T}) where {T<:AbstractFloat} = copyto!(similar(x0, length(v)), v)

function _seg_copy(x0, dst::Vector{Int}, src::Vector{Int})
    p, dstu, ptr = _segment_perm(dst)
    return SegCopy(_dev(x0, dstu), _dev(x0, ptr), _dev(x0, src[p]))
end
function _seg_prod(x0, dst::Vector{Int}, a::Vector{Int}, b::Vector{Int})
    p, dstu, ptr = _segment_perm(dst)
    return SegProd(_dev(x0, dstu), _dev(x0, ptr), _dev(x0, a[p]), _dev(x0, b[p]))
end
function _seg_prodc(x0, dst::Vector{Int}, a::Vector{Int}, b::Vector{Int}, c::Vector)
    p, dstu, ptr = _segment_perm(dst)
    return SegProdC(_dev(x0, dstu), _dev(x0, ptr), _dev(x0, a[p]), _dev(x0, b[p]), _dev(x0, c[p]))
end

@kernel function kseg_copy(y, @Const(dstu), @Const(ptr), @Const(src), @Const(S))
    I = @index(Global)
    acc = zero(eltype(y))
    @inbounds for j = ptr[I]:(ptr[I+1]-1)
        acc += S[src[j]]
    end
    @inbounds y[dstu[I]] += acc
end
@kernel function kseg_prod(y, @Const(dstu), @Const(ptr), @Const(a), @Const(b), @Const(A), @Const(B))
    I = @index(Global)
    acc = zero(eltype(y))
    @inbounds for j = ptr[I]:(ptr[I+1]-1)
        acc += A[a[j]] * B[b[j]]
    end
    @inbounds y[dstu[I]] += acc
end
@kernel function kseg_prodc(y, @Const(dstu), @Const(ptr), @Const(a), @Const(b), @Const(c), @Const(A), @Const(B))
    I = @index(Global)
    acc = zero(eltype(y))
    @inbounds for j = ptr[I]:(ptr[I+1]-1)
        acc += c[j] * A[a[j]] * B[b[j]]
    end
    @inbounds y[dstu[I]] += acc
end

_run!(backend, s::SegCopy, y, S) =
    _seg_isempty(s) || kseg_copy(backend)(y, s.dstu, s.ptr, s.src, S; ndrange = length(s.dstu))
_run!(backend, s::SegProd, y, A, B) =
    _seg_isempty(s) || kseg_prod(backend)(y, s.dstu, s.ptr, s.a, s.b, A, B; ndrange = length(s.dstu))
_run!(backend, s::SegProdC, y, A, B) =
    _seg_isempty(s) ||
    kseg_prodc(backend)(y, s.dstu, s.ptr, s.a, s.b, s.c, A, B; ndrange = length(s.dstu))

# ── per-element stage kernels ─────────────────────────────────────────────────

@kernel function ksync(θ, @Const(f), @Const(itr), @Const(x), @Const(off))
    I = @index(Global)
    @inbounds θ[off+I] = f(itr[I], x, θ)
end

@kernel function kstagej(y1, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(o1base))
    I = @index(Global)
    @inbounds ExaModels.jrpass(
        f(itr[I], ExaModels.AdjointNodeSource(x), θ),
        f.comp1,
        0,
        y1,
        nothing,
        o1base + (I - 1) * f.o1step,
        0,
        one(eltype(y1)),
    )
end

@kernel function kstageh(y1, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(λ), @Const(μ), @Const(θoff), @Const(o2base))
    I = @index(Global)
    @inbounds ExaModels.hrpass(
        f(itr[I], ExaModels.SecondAdjointNodeSource(x), θ),
        f.comp2,
        y1,
        nothing,
        o2base + (I - 1) * f.o2step,
        0,
        λ[θoff+I],
        μ[θoff+I],
    )
end

# first-order constraint pass with per-row adjoints (λ seeds from y)
@kernel function kerjs(y1, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adjs), @Const(dims))
    I = @index(Global)
    @inbounds ExaModels.jrpass(
        f(itr[I], ExaModels.AdjointNodeSource(x), θ),
        f.comp1,
        ExaModels.offset0(f, itr, I, dims),
        y1,
        nothing,
        ExaModels.offset1(f, I),
        0,
        adjs[ExaModels.offset0(f, itr, I, dims)],
    )
end

_lamgrad!(backend, y, ::Tuple{}, x, θ, w) = nothing
function _lamgrad!(backend, y, (obj, objs...), x, θ, w)
    _lamgrad!(backend, y, objs, x, θ, w)
    if !isempty(obj.itr)
        kerg(backend)(y, obj.f, obj.itr, x, θ, w; ndrange = length(obj.itr))
    end
end

_lamjac!(backend, y, ::Tuple{}, x, θ, yv) = nothing
function _lamjac!(backend, y, (con, cons...), x, θ, yv)
    _lamjac!(backend, y, cons, x, θ, yv)
    if con isa ExaModels.Constraint && !isempty(con.itr)
        kerjs(backend)(y, con.f, con.itr, x, θ, yv, ExaModels._constraint_dims(con); ndrange = length(con.itr))
    end
end

# host-side gradient structure pass (objective comp1 slots → (target, slot))
function _grad_structure_host!(::Type{T}, objs, gsp) where {T}
    for obj in objs
        f = obj.f
        for k in eachindex(obj.itr)
            graph = f.f(
                obj.itr[k],
                ExaModels.AdjointNodeSource(ExaModels.NaNSource{T}()),
                ExaModels.NaNSource{T}(),
            )
            ExaModels.grpass(graph, f.comp1, gsp, ExaModels.offset1(obj, k), 0, T(NaN))
        end
    end
    return gsp
end

# ── build-time artifact ───────────────────────────────────────────────────────

struct SubexprKA{T,VT<:AbstractVector{T},VI}
    nvar::Int
    nθ::Int
    # device buffers
    stage_ext::VT
    res::VT
    cons_ext::VT
    hess_ext::VT
    C::VT
    abuf::VT
    abuf2::VT
    y_ext::VT                  # [gradient (nvar); adjoint buffer (nθ)]
    # stage metadata, newest-first (aligned with the stage tuple)
    θoff::Vector{Int}
    o1base::Vector{Int}
    o2base::Vector{Int}
    # segmented programs
    grad_scatter::SegCopy{VI}              # gradbuffer → y_ext (shifted targets)
    grad_chain::Vector{SegProd{VI}}        # per level: y_ext += y_ext[nvar+row]·stage_ext
    lam_obj::SegCopy{VI}                   # gradbuffer (weighted) → abuf
    lam_con::SegCopy{VI}                   # cons_ext (y-weighted jrpass) → abuf
    lam_chain::Vector{SegProd{VI}}         # per level: abuf += abuf[row]·stage_ext
    mu_cons::SegCopy{VI}                   # consumer dead slots → abuf2
    mu_stage::Vector{SegCopy{VI}}          # per level: stage dead slots → abuf2
    res_copy::Vector{SegCopy{VI}}          # per stage, oldest-first execution
    res_prod::Vector{SegProd{VI}}
    jac_copy::SegCopy{VI}
    jac_prod::SegProd{VI}
    ccopy::Vector{SegCopy{VI}}             # per level (newest-first)
    cmul1::Vector{SegProd{VI}}             # rank-2 phase
    cmul2::Vector{SegProd{VI}}             # rank-1 phase
    amul::Vector{SegProdC{VI,VT}}
    hmul::Vector{SegProdC{VI,VT}}
    hess_copy::SegCopy{VI}
end

function ExaModels.build_subexpr_ka(
    c::C,
    sjac,
    shess,
    nvar,
) where {T,VT<:AbstractVector{T},B<:KernelAbstractions.Backend,C<:ExaModels.ExaCore{T,VT,B}}
    sjac === nothing && return nothing
    x0 = c.x0
    stages_h = ExaModels._host_blocks(c.subexprs)
    objs_h = ExaModels._host_blocks(c.obj)
    cons_h = ExaModels._host_blocks(c.cons)
    ordered = reverse(collect(Any, stages_h))      # oldest first; level l = index
    nlv = length(ordered)
    nθ = length(c.θ)

    slotlvl = zeros(Int, nθ)
    for (i, s) in enumerate(ordered)
        slotlvl[(s.offset+1):(s.offset+length(s.itr))] .= i
    end
    lvlof(ξ) = ξ > 0 ? 0 : slotlvl[-ξ]

    # ---- host structure passes for the scatter maps ----
    # gradient slots (objective comp1 space): (target, slot) with -slot markers
    gsp = fill((0, 0), c.nnzg)
    _grad_structure_host!(T, objs_h, gsp)
    gs_dst = Int[]; gs_src = Int[]
    lo_dst = Int[]; lo_src = Int[]
    for (t, l) in gsp
        l == 0 && continue
        push!(gs_dst, t > 0 ? t : nvar - t)
        push!(gs_src, l)
        if t < 0
            push!(lo_dst, -t)
            push!(lo_src, l)
        end
    end
    # constraint jacobian slots: markers from the extended structure pass
    jrows_h = zeros(Int, c.nnzj)
    jcols_h = zeros(Int, c.nnzj)
    ExaModels._jac_structure!(T, cons_h, jrows_h, jcols_h)
    lc_dst = Int[]; lc_src = Int[]
    for ind in eachindex(jcols_h)
        if jcols_h[ind] < 0
            push!(lc_dst, -jcols_h[ind])
            push!(lc_src, ind)
        end
    end
    # hessian extended structure: dead (-j,-j) slots per source segment
    nnzh_ext = length(shess.hess_ext)
    herows = zeros(Int, nnzh_ext)
    hecols = zeros(Int, nnzh_ext)
    ExaModels._obj_hess_structure!(T, objs_h, herows, hecols)
    ExaModels._con_hess_structure!(T, cons_h, herows, hecols)
    stage_o2_old = reverse(shess.stage_o2)          # oldest-first bases
    for (si, s) in enumerate(ordered)
        step = s.f.o2step
        for k in eachindex(s.itr)
            ExaModels._stage_shessian!(
                herows,
                hecols,
                s.f,
                s.itr[k],
                ExaModels.NaNSource{T}(),
                ExaModels.NaNSource{T}(),
                s.f.comp2,
                stage_o2_old[si] + (k - 1) * step,
                T(NaN),
                T(NaN),
            )
        end
    end
    mu_cons_dst = Int[]; mu_cons_src = Int[]
    mu_stage_dst = [Int[] for _ in 1:nlv]           # by SOURCE stage level
    for ind in eachindex(herows)
        r = herows[ind]
        (r < 0 && r == hecols[ind]) || continue
        if ind <= c.nnzh
            push!(mu_cons_dst, -r); push!(mu_cons_src, ind)
        else
            si = searchsortedlast(stage_o2_old, ind - 1)
            push!(mu_stage_dst[si], -r)
            push!(mu_stage_dst[si], ind)            # interleaved (dst, src)
        end
    end
    mu_stage = Vector{Any}(undef, nlv)
    for l in 1:nlv
        v = mu_stage_dst[l]
        mu_stage[l] = _seg_copy(x0, v[1:2:end], v[2:2:end])
    end

    # localrow chain programs per SOURCE level: full (gradient) and slot-only (λ)
    localrows = Dict{Int,Vector{Tuple{Int,Int}}}()
    let res_probe = ExaModels._build_subexpr_jac(T, c.subexprs, c.cons, c.nnzj)
        localrows = res_probe[3]
    end
    gc_dst = [Int[] for _ in 1:nlv]; gc_a = [Int[] for _ in 1:nlv]; gc_b = [Int[] for _ in 1:nlv]
    lch_dst = [Int[] for _ in 1:nlv]; lch_a = [Int[] for _ in 1:nlv]; lch_b = [Int[] for _ in 1:nlv]
    for (row, lst) in localrows
        l = slotlvl[row]
        for (col, pos) in lst
            tgt = col > 0 ? col : nvar - col
            push!(gc_dst[l], tgt); push!(gc_a[l], nvar + row); push!(gc_b[l], pos)
            if col < 0
                push!(lch_dst[l], -col); push!(lch_a[l], row); push!(lch_b[l], pos)
            end
        end
    end

    seg_res_copy = [
        _seg_copy(x0, sjac.res_copy_dst[r], sjac.res_copy_src[r]) for
        r in sjac.lvl_res_copy
    ]
    seg_res_prod = [
        _seg_prod(x0, sjac.res_prod_dst[r], sjac.res_prod_a[r], sjac.res_prod_b[r])
        for r in sjac.lvl_res_prod
    ]

    seg_ccopy = [_seg_copy(x0, shess.ccopy_dst[r], shess.ccopy_src[r]) for r in shess.lvl_ccopy]
    seg_cmul1 = SegProd[]
    seg_cmul2 = SegProd[]
    for (i, r) in enumerate(shess.lvl_cmul)
        ph = shess.cmul_ph[i]
        r1 = first(r):(first(r)+ph-1)
        r2 = (first(r)+ph):last(r)
        push!(seg_cmul1, _seg_prod(x0, shess.cmul_dst[r1], shess.cmul_src[r1], shess.cmul_a[r1]))
        push!(seg_cmul2, _seg_prod(x0, shess.cmul_dst[r2], shess.cmul_src[r2], shess.cmul_a[r2]))
    end
    seg_amul = [
        _seg_prodc(x0, shess.amul_dst[r], shess.amul_src[r], shess.amul_a[r], fill(T(2), length(r)))
        for r in shess.lvl_amul
    ]
    seg_hmul = [
        _seg_prodc(x0, shess.hmul_dst[r], shess.hmul_src[r], shess.hmul_a[r], shess.hmul_c[r])
        for r in shess.lvl_hmul
    ]

    VI = typeof(similar(x0, Int, 0))
    return SubexprKA(
        Int(nvar),
        nθ,
        similar(x0, length(sjac.stage_ext)),
        similar(x0, length(sjac.res)),
        similar(x0, length(sjac.cons_ext)),
        similar(x0, length(shess.hess_ext)),
        similar(x0, length(shess.C)),
        similar(x0, nθ),
        similar(x0, nθ),
        similar(x0, nvar + nθ),
        reverse([s.offset for s in ordered]),
        copy(sjac.stage_o1),
        copy(shess.stage_o2),
        _seg_copy(x0, gs_dst, gs_src),
        [_seg_prod(x0, gc_dst[l], gc_a[l], gc_b[l]) for l in nlv:-1:1],
        _seg_copy(x0, lo_dst, lo_src),
        _seg_copy(x0, lc_dst, lc_src),
        [_seg_prod(x0, lch_dst[l], lch_a[l], lch_b[l]) for l in nlv:-1:1],
        _seg_copy(x0, mu_cons_dst, mu_cons_src),
        SegCopy{VI}[mu_stage[l] for l in nlv:-1:1],
        SegCopy{VI}[seg_res_copy...],
        SegProd{VI}[seg_res_prod...],
        _seg_copy(x0, sjac.jac_copy_dst, sjac.jac_copy_src),
        _seg_prod(x0, sjac.jac_prod_dst, sjac.jac_prod_a, sjac.jac_prod_b),
        SegCopy{VI}[seg_ccopy...],
        SegProd{VI}[seg_cmul1...],
        SegProd{VI}[seg_cmul2...],
        SegProdC{VI,typeof(similar(x0, 0))}[seg_amul...],
        SegProdC{VI,typeof(similar(x0, 0))}[seg_hmul...],
        _seg_copy(x0, shess.hess_copy_dst, shess.hess_copy_src),
    )
end

# ── drivers ───────────────────────────────────────────────────────────────────

_sync_ka!(backend, ::Tuple{}, θoff, i, x, θ) = nothing
function _sync_ka!(backend, stages::Tuple, θoff, i, x, θ)
    _sync_ka!(backend, Base.tail(stages), θoff, i + 1, x, θ)
    s = first(stages)
    if !isempty(s.itr)
        ksync(backend)(θ, s.f, s.itr, x, θoff[i]; ndrange = length(s.itr))
    end
    return nothing
end

_stagej_ka!(backend, ::Tuple{}, ska, i, x, θ) = nothing
function _stagej_ka!(backend, stages::Tuple, ska, i, x, θ)
    s = first(stages)
    if !isempty(s.itr)
        kstagej(backend)(ska.stage_ext, s.f, s.itr, x, θ, ska.o1base[i]; ndrange = length(s.itr))
    end
    _stagej_ka!(backend, Base.tail(stages), ska, i + 1, x, θ)
    return nothing
end

function _sync_point_ka!(backend, m, x)
    ska = m.ska
    _sync_ka!(backend, m.subexprs, ska.θoff, 1, x, m.θ)
    return nothing
end

function _obj_buffered(m, x)
    _sync_point_ka!(m.ext.backend, m, x)
    if !isempty(m.ext.objbuffer)
        _obj(m.ext.backend, m.ext.objbuffer, m.objs, x, m.θ)
        return ExaModels.sum(m.ext.objbuffer)
    else
        return zero(eltype(x))
    end
end

function _cons_buffered!(m, x, y)
    _sync_point_ka!(m.ext.backend, m, x)
    _cons_nln!(m.ext.backend, y, m.cons, x, m.θ)
    _conaugs!(m.ext.backend, m.ext.conbuffer, m.cons, x, m.θ)
    if length(m.ext.conaugptr) > 1
        compress_to_dense(m.ext.backend)(
            y,
            m.ext.conbuffer,
            m.ext.conaugptr,
            m.ext.conaugsparsity;
            ndrange = length(m.ext.conaugptr) - 1,
        )
    end
    return y
end

function _grad_buffered!(m, x, y)
    backend = m.ext.backend
    ska = m.ska
    _sync_point_ka!(backend, m, x)
    fill!(ska.stage_ext, zero(eltype(ska.stage_ext)))
    _stagej_ka!(backend, m.subexprs, ska, 1, x, m.θ)
    fill!(ska.y_ext, zero(eltype(ska.y_ext)))
    gradbuffer = m.ext.gradbuffer
    if !isempty(gradbuffer)
        fill!(gradbuffer, zero(eltype(gradbuffer)))
        _grad!(backend, gradbuffer, m.objs, x, m.θ)
        _run!(backend, ska.grad_scatter, ska.y_ext, gradbuffer)
    end
    for seg in ska.grad_chain          # newest-first
        _run!(backend, seg, ska.y_ext, ska.y_ext, ska.stage_ext)
    end
    copyto!(y, view(ska.y_ext, 1:ska.nvar))
    return y
end

function _jac_buffered!(m, x, y)
    backend = m.ext.backend
    ska = m.ska
    _sync_point_ka!(backend, m, x)
    fill!(ska.stage_ext, zero(eltype(ska.stage_ext)))
    _stagej_ka!(backend, m.subexprs, ska, 1, x, m.θ)
    fill!(ska.res, zero(eltype(ska.res)))
    for i in eachindex(ska.res_copy)   # oldest-first
        _run!(backend, ska.res_copy[i], ska.res, ska.stage_ext)
        _run!(backend, ska.res_prod[i], ska.res, ska.stage_ext, ska.res)
    end
    fill!(ska.cons_ext, zero(eltype(ska.cons_ext)))
    _jac_coord!(backend, ska.cons_ext, m.cons, x, m.θ)
    fill!(y, zero(eltype(y)))
    _run!(backend, ska.jac_copy, y, ska.cons_ext)
    _run!(backend, ska.jac_prod, y, ska.cons_ext, ska.res)
    return y
end

_stageh_ka!(backend, ::Tuple{}, ska, i, x, θ) = nothing
function _stageh_ka!(backend, stages::Tuple, ska, i, x, θ)
    s = first(stages)
    if !isempty(s.itr)
        kstageh(backend)(
            ska.hess_ext,
            s.f,
            s.itr,
            x,
            θ,
            ska.abuf,
            ska.abuf2,
            ska.θoff[i],
            ska.o2base[i];
            ndrange = length(s.itr),
        )
    end
    _run!(backend, ska.mu_stage[i], ska.abuf2, ska.hess_ext)
    _run!(backend, ska.lam_chain[i], ska.abuf, ska.abuf, ska.stage_ext)
    _run!(backend, ska.ccopy[i], ska.C, ska.hess_ext)
    _run!(backend, ska.cmul1[i], ska.C, ska.C, ska.stage_ext)
    _run!(backend, ska.cmul2[i], ska.C, ska.C, ska.stage_ext)
    _run!(backend, ska.amul[i], ska.abuf2, ska.C, ska.stage_ext)
    _stageh_ka!(backend, Base.tail(stages), ska, i + 1, x, θ)
    return nothing
end

_stageh_hmul!(backend, ska, hess) =
    for seg in ska.hmul
        _run!(backend, seg, hess, ska.C, ska.stage_ext)
    end

function _hess_buffered!(m, x, yv, hess, obj_weight)
    backend = m.ext.backend
    ska = m.ska
    _sync_point_ka!(backend, m, x)
    fill!(ska.stage_ext, zero(eltype(ska.stage_ext)))
    _stagej_ka!(backend, m.subexprs, ska, 1, x, m.θ)
    # λ seeds
    fill!(ska.abuf, zero(eltype(ska.abuf)))
    gradbuffer = m.ext.gradbuffer
    if !isempty(gradbuffer)
        fill!(gradbuffer, zero(eltype(gradbuffer)))
        _lamgrad!(backend, gradbuffer, m.objs, x, m.θ, obj_weight)
        _run!(backend, ska.lam_obj, ska.abuf, gradbuffer)
    end
    if yv !== nothing
        fill!(ska.cons_ext, zero(eltype(ska.cons_ext)))
        _lamjac!(backend, ska.cons_ext, m.cons, x, m.θ, yv)
        _run!(backend, ska.lam_con, ska.abuf, ska.cons_ext)
    end
    # extended-space entries + μ seeds from the consumer segment
    fill!(ska.abuf2, zero(eltype(ska.abuf2)))
    fill!(ska.hess_ext, zero(eltype(ska.hess_ext)))
    fill!(ska.C, zero(eltype(ska.C)))
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(backend, ska.hess_ext, m.objs, x, m.θ, obj_weight)
    yv === nothing || _con_hess_coord!(backend, ska.hess_ext, m.cons, x, m.θ, yv)
    _run!(backend, ska.mu_cons, ska.abuf2, ska.hess_ext)
    # interleaved replays + per-level elimination, newest first
    _stageh_ka!(backend, m.subexprs, ska, 1, x, m.θ)
    _stageh_hmul!(backend, ska, hess)
    _run!(backend, ska.hess_copy, hess, ska.hess_ext)
    KernelAbstractions.synchronize(backend)
    return hess
end
