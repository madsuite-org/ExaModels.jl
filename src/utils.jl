# WrapperNLPModel serves as a wrapper for ExaNLPModel, or even any NLPModels.
# This is useful when you want to use a solver that does not support non-stardard array data types.
# TODO: make this as an independent package

"""
    WrapperNLPModel

An `NLPModels.AbstractNLPModel` wrapper that bridges a model whose internal array
type differs from the host array type (e.g. a GPU-backed `ExaModel` wrapped for a
CPU-only solver).  All NLP callbacks are forwarded to the inner model through
intermediate copy buffers.

Use [`WrapperNLPModel(m)`](@ref) or [`WrapperNLPModel(VT, m)`](@ref) rather than
constructing this struct directly.
"""
struct WrapperNLPModel{
    T,
    VT,
    T2,
    VT2<:AbstractVector{T2},
    VI,
    I<:NLPModels.AbstractNLPModel{T2,VT2},
} <: NLPModels.AbstractNLPModel{T,VT}

    inner::I

    x_result::VT
    x_result2::VT
    y_result::VT

    x_buffer::VT2
    y_buffer::VT2
    v_buffer::VT2

    cons_buffer::VT2
    grad_buffer::VT2

    jac_buffer::VT2
    jac_I_buffer::VI
    jac_J_buffer::VI

    hess_buffer::VT2
    hess_buffer2::VT
    hess_I_buffer::VI
    hess_J_buffer::VI

    meta::NLPModels.AbstractNLPModelMeta{T,VT}
    counters::NLPModels.Counters
end

"""
    WrapperNLPModel(m)

Returns a `WrapperNLPModel{Float64,Vector{Float64}}` wrapping `m`, forwarding
all NLP callbacks through `Float64` CPU copy buffers.
"""
WrapperNLPModel(m) = WrapperNLPModel(Vector{Float64}, m)

"""
    WrapperNLPModel(VT, m)

Returns a `WrapperModel{T,VT}` wrapping `m <: AbstractNLPModel{T}`
"""
function WrapperNLPModel(VT, m)
    nvar = NLPModels.get_nvar(m)
    ncon = NLPModels.get_ncon(m)
    nnzj = NLPModels.get_nnzj(m)
    nnzh = NLPModels.get_nnzh(m)

    x_result = VT(undef, nvar)
    x_result2 = VT(undef, nvar)
    y_result = VT(undef, ncon)

    x0 = VT(undef, nvar)
    lvar = VT(undef, nvar)
    uvar = VT(undef, nvar)

    y0 = VT(undef, ncon)
    lcon = VT(undef, ncon)
    ucon = VT(undef, ncon)

    copyto!(x0, m.meta.x0)
    copyto!(lvar, m.meta.lvar)
    copyto!(uvar, m.meta.uvar)

    copyto!(y0, m.meta.y0)
    copyto!(lcon, m.meta.lcon)
    copyto!(ucon, m.meta.ucon)

    x_buffer = similar(m.meta.x0, nvar)
    y_buffer = similar(m.meta.x0, ncon)
    v_buffer = similar(m.meta.x0, nvar)
    cons_buffer = similar(m.meta.x0, ncon)
    grad_buffer = similar(m.meta.x0, nvar)
    jac_buffer = similar(m.meta.x0, nnzj)
    jac_I_buffer = similar(m.meta.x0, Int, nnzj)
    jac_J_buffer = similar(m.meta.x0, Int, nnzj)
    hess_buffer = similar(m.meta.x0, nnzh)
    hess_buffer2 = VT(undef, nnzh)
    hess_I_buffer = similar(m.meta.x0, Int, nnzh)
    hess_J_buffer = similar(m.meta.x0, Int, nnzh)

    return WrapperNLPModel(
        m,
        x_result,
        x_result2,
        y_result,
        x_buffer,
        y_buffer,
        v_buffer,
        cons_buffer,
        grad_buffer,
        jac_buffer,
        jac_I_buffer,
        jac_J_buffer,
        hess_buffer,
        hess_buffer2,
        hess_I_buffer,
        hess_J_buffer,
        NLPModels.NLPModelMeta(
            nvar,
            x0 = x0,
            lvar = lvar,
            uvar = uvar,
            ncon = ncon,
            y0 = y0,
            lcon = lcon,
            ucon = ucon,
            nnzj = nnzj,
            nnzh = nnzh,
            minimize = m.meta.minimize,
        ),
        NLPModels.Counters(),
    )
end

function NLPModels.jac_structure!(
    m::WrapperNLPModel,
    rows::AbstractVector,
    cols::AbstractVector,
)
    NLPModels.jac_structure!(m.inner, m.jac_I_buffer, m.jac_J_buffer)
    copyto!(rows, m.jac_I_buffer)
    copyto!(cols, m.jac_J_buffer)
    return rows, cols
end

function NLPModels.hess_structure!(
    m::WrapperNLPModel,
    rows::AbstractVector,
    cols::AbstractVector,
)
    NLPModels.hess_structure!(m.inner, m.hess_I_buffer, m.hess_J_buffer)
    copyto!(rows, m.hess_I_buffer)
    copyto!(cols, m.hess_J_buffer)
    return rows, cols
end

function NLPModels.obj(m::WrapperNLPModel, x::AbstractVector)
    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    o = NLPModels.obj(m.inner, m.x_buffer)
    return o
end
function NLPModels.cons!(m::WrapperNLPModel, x::AbstractVector, g::AbstractVector)
    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    NLPModels.cons!(m.inner, m.x_buffer, m.cons_buffer)
    copyto!(m.y_result, m.cons_buffer)
    copyto!(g, m.y_result)
    return g
end
function NLPModels.grad!(m::WrapperNLPModel, x::AbstractVector, f::AbstractVector)
    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    NLPModels.grad!(m.inner, m.x_buffer, m.grad_buffer)
    copyto!(m.x_result, m.grad_buffer)
    copyto!(f, m.x_result)
    return f
end
function NLPModels.jac_coord!(m::WrapperNLPModel, x::AbstractVector, jac::AbstractVector)
    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    NLPModels.jac_coord!(m.inner, m.x_buffer, m.jac_buffer)
    copyto!(jac, m.jac_buffer)
    return jac
end
function NLPModels.hess_coord!(
    m::WrapperNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)
    copyto!(m.x_buffer, x)
    copyto!(m.y_buffer, y)
    NLPModels.hess_coord!(
        m.inner,
        m.x_buffer,
        m.y_buffer,
        m.hess_buffer;
        obj_weight = eltype(m.hess_buffer)(obj_weight),
    )
    copyto!(m.hess_buffer2, m.hess_buffer)
    copyto!(hess, m.hess_buffer2)

    return hess
end

function buffered_copyto!(a, b, c)
    copyto!(b, c)
    copyto!(a, b)
end
function NLPModels.jprod!(
    m::WrapperNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    buffered_copyto!(m.x_buffer, m.x_result, x)
    buffered_copyto!(m.grad_buffer, m.x_result2, v)

    NLPModels.jprod!(m.inner, m.x_buffer, m.grad_buffer, m.cons_buffer)

    buffered_copyto!(Jv, m.y_result, m.cons_buffer)
    return Jv
end
function NLPModels.jtprod_nln!(
    m::WrapperNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)

    buffered_copyto!(m.x_buffer, m.x_result, x)
    buffered_copyto!(m.cons_buffer, m.y_result, v)

    NLPModels.jtprod_nln!(m.inner, m.x_buffer, m.cons_buffer, m.grad_buffer)

    buffered_copyto!(Jtv, m.x_result, m.grad_buffer)
    return Jtv
end
function NLPModels.hprod!(
    m::WrapperNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
)

    buffered_copyto!(m.x_buffer, m.x_result, x)
    buffered_copyto!(m.y_buffer, m.y_result, y)
    buffered_copyto!(m.grad_buffer, m.x_result2, v)

    NLPModels.hprod!(
        m.inner,
        m.x_buffer,
        m.y_buffer,
        m.grad_buffer,
        m.v_buffer;
        obj_weight = eltype(m.x_buffer)(obj_weight),
    )

    buffered_copyto!(Hv, m.x_result, m.v_buffer)
    return Hv
end

# TimedNLPModels

Base.@kwdef mutable struct CallbackStats
    obj_cnt::Int = 0
    cons_cnt::Int = 0
    grad_cnt::Int = 0
    jac_coord_cnt::Int = 0
    hess_coord_cnt::Int = 0
    jac_structure_cnt::Int = 0
    hess_structure_cnt::Int = 0
    obj_time::Float64 = 0.0
    cons_time::Float64 = 0.0
    grad_time::Float64 = 0.0
    jac_coord_time::Float64 = 0.0
    hess_coord_time::Float64 = 0.0
    jac_structure_time::Float64 = 0.0
    hess_structure_time::Float64 = 0.0
end

"""
    TimedNLPModel

A transparent wrapper around any `AbstractNLPModel` that records wall-clock
timings and call counts for each NLP callback (`obj`, `cons!`, `grad!`,
`jac_coord!`, `hess_coord!`, `jac_structure!`, `hess_structure!`).

Statistics accumulate in the `stats` field and can be printed with `print(m)`.
Construct via [`TimedNLPModel(m)`](@ref).
"""
struct TimedNLPModel{T,VT,I<:NLPModels.AbstractNLPModel{T,VT}} <:
       NLPModels.AbstractNLPModel{T,VT}
    inner::I
    meta::NLPModels.AbstractNLPModelMeta{T,VT}
    stats::CallbackStats
    counters::NLPModels.Counters
end

"""
    TimedNLPModel(m)

Wraps `m` in a [`TimedNLPModel`](@ref) with all counters and timers reset to zero.
"""
function TimedNLPModel(m)
    return TimedNLPModel(m, m.meta, CallbackStats(), NLPModels.Counters())
end

"""
    TimedNLPModel(core::ExaCore; kwargs...)

Builds an [`ExaModel`](@ref) from `core` (forwarding `kwargs...`) and wraps it in
a [`TimedNLPModel`](@ref).
"""
function TimedNLPModel(c::ExaModels.ExaCore; kwargs...)
    m = ExaModels.Model(c; kwargs...)
    return TimedNLPModel(m)
end

function NLPModels.jac_structure!(
    m::M,
    rows::V,
    cols::V,
) where {M<:TimedNLPModel,V<:AbstractVector}

    m.stats.jac_structure_cnt += 1
    t = time()
    NLPModels.jac_structure!(m.inner, rows, cols)
    m.stats.jac_structure_time += time() - t
    return rows, cols
end

function NLPModels.hess_structure!(
    m::M,
    rows::V,
    cols::V,
) where {M<:TimedNLPModel,V<:AbstractVector}

    m.stats.hess_structure_cnt += 1
    t = time()
    NLPModels.hess_structure!(m.inner, rows, cols)
    m.stats.hess_structure_time += time() - t
    return rows, cols
end

function NLPModels.obj(m::TimedNLPModel, x::AbstractVector)
    m.stats.obj_cnt += 1
    t = time()
    o = NLPModels.obj(m.inner, x)
    m.stats.obj_time += time() - t
    return o
end
function NLPModels.cons!(m::TimedNLPModel, x::AbstractVector, g::AbstractVector)
    m.stats.cons_cnt += 1
    t = time()
    NLPModels.cons!(m.inner, x, g)
    m.stats.cons_time += time() - t
    return g
end
function NLPModels.grad!(m::TimedNLPModel, x::AbstractVector, f::AbstractVector)
    m.stats.grad_cnt += 1
    t = time()
    NLPModels.grad!(m.inner, x, f)
    m.stats.grad_time += time() - t
    return f
end
function NLPModels.jac_coord!(m::TimedNLPModel, x::AbstractVector, jac::AbstractVector)
    m.stats.jac_coord_cnt += 1
    t = time()
    NLPModels.jac_coord!(m.inner, x, jac)
    m.stats.jac_coord_time += time() - t
    return jac
end
function NLPModels.hess_coord!(
    m::TimedNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)
    m.stats.hess_coord_cnt += 1
    t = time()
    NLPModels.hess_coord!(m.inner, x, y, hess; obj_weight = obj_weight)
    m.stats.hess_coord_time += time() - t
    return hess
end

function Base.print(io::IO, e::TimedNLPModel)
    tot = 0.0
    for f in fieldnames(CallbackStats)
        if endswith(string(f), "cnt")
            Printf.@printf "%20s:  %13i times\n" f getfield(e.stats, f)
        else
            t = getfield(e.stats, f)
            Printf.@printf "%20s:  %13.6f secs\n" f t
            tot += t
        end
    end
    println("------------------------------------------")
    Printf.@printf "       total AD time:  %13.6f secs\n" tot
end
Base.show(io::IO, ::MIME"text/plain", e::TimedNLPModel) = Base.print(io, e);


"""
    CompressedNLPModel

A wrapper around an `AbstractNLPModel` that sums duplicate `(row, col)` entries
in the sparse Jacobian and Hessian.

Duplicates arise when multiple constraint or objective patterns contribute to the
same matrix position (e.g. after augmentation via [`add_con!`](@ref)).
`CompressedNLPModel` detects them once at construction and accumulates them on
every subsequent `jac_coord!` / `hess_coord!` call, so callers receive a matrix
with no repeated coordinates.

Construct via [`CompressedNLPModel(m)`](@ref).
"""
struct CompressedNLPModel{
    T,
    VT<:AbstractVector{T},
    B,
    VI<:AbstractVector{Int},
    VI2<:AbstractVector{Tuple{Tuple{Int,Int},Int}},
    M<:NLPModels.AbstractNLPModel{T,VT},
} <: NLPModels.AbstractNLPModel{T,VT}

    inner::M
    jptr::VI
    jsparsity::VI2
    hptr::VI
    hsparsity::VI2
    buffer::VT

    backend::B
    meta::NLPModels.NLPModelMeta{T,VT}
    counters::NLPModels.Counters
end

# Comparisons for sorting sparsity entries and finding where runs start. They are named
# functions rather than closures so that their names, and so the names of the GPU kernels
# and methods specialized on them, stay the same from one version to the next; precompile
# statements and ahead-of-time compiled code can then refer to them.
_first_lt(a, b) = a[1] < b[1]
_first_ne(a, b) = a[1] != b[1]
_row_lt(a, b) = a[1][1] < b[1][1]
_col_lt(a, b) = a[1][2] < b[1][2]
_row_ne(a, b) = a[1][1] != b[1][1]
_col_ne(a, b) = a[1][2] != b[1][2]

function getptr(backend::Nothing, array; cmp = !=)
    return push!(
        pushfirst!(findall(cmp.(@view(array[1:(end-1)]), @view(array[2:end]))) .+= 1, 1),
        length(array) + 1,
    )
end

"""
    CompressedNLPModel(m)

Wraps `m` in a [`CompressedNLPModel`](@ref).

Queries the full Jacobian and Hessian sparsity patterns from `m`, identifies
duplicate `(row, col)` pairs, builds pointer arrays for O(nnz) accumulation on
subsequent `jac_coord!` / `hess_coord!` calls.
"""
function CompressedNLPModel(m)

    nnzj = NLPModels.get_nnzj(m)
    nnzh = NLPModels.get_nnzh(m)

    Ibuffer = similar(m.meta.x0, Int, max(nnzj, nnzh))
    Jbuffer = similar(m.meta.x0, Int, max(nnzj, nnzh))
    buffer = similar(m.meta.x0, max(nnzj, nnzh))

    NLPModels.jac_structure!(m, Ibuffer, Jbuffer)

    backend = getbackend(m)

    jsparsity = get_compressed_sparsity(nnzj, Ibuffer, Jbuffer, backend)
    sort!(jsparsity; lt = _first_lt)
    jptr = getptr(backend, jsparsity; cmp = _first_ne)

    NLPModels.hess_structure!(m, Ibuffer, Jbuffer)

    hsparsity = get_compressed_sparsity(nnzh, Ibuffer, Jbuffer, backend)
    sort!(hsparsity; lt = _first_lt)
    hptr = getptr(backend, hsparsity; cmp = _first_ne)


    meta = NLPModels.NLPModelMeta(
        m.meta.nvar,
        ncon = m.meta.ncon,
        nnzj = length(jptr) - 1,
        nnzh = length(hptr) - 1,
        x0 = m.meta.x0,
        lvar = m.meta.lvar,
        uvar = m.meta.uvar,
        y0 = m.meta.y0,
        lcon = m.meta.lcon,
        ucon = m.meta.ucon,
    )

    counters = NLPModels.Counters()

    return CompressedNLPModel(
        m,
        jptr,
        jsparsity,
        hptr,
        hsparsity,
        buffer,
        backend,
        meta,
        counters,
    )
end

getbackend(m) = nothing
get_compressed_sparsity(nnz, Ibuffer, Jbuffer, backend::Nothing) =
    map((k, i, j) -> ((j, i), k), 1:nnz, Ibuffer, Jbuffer)

function NLPModels.obj(m::CompressedNLPModel, x::AbstractVector)
    NLPModels.obj(m.inner, x)
end

function NLPModels.grad!(m::CompressedNLPModel, x::AbstractVector, y::AbstractVector)
    NLPModels.grad!(m.inner, x, y)
end

function NLPModels.cons!(m::CompressedNLPModel, x::AbstractVector, g::AbstractVector)
    NLPModels.cons!(m.inner, x, g)
end

function NLPModels.jac_coord!(m::CompressedNLPModel, x::AbstractVector, j::AbstractVector)
    NLPModels.jac_coord!(m.inner, x, m.buffer)
    _compress!(j, m.buffer, m.jptr, m.jsparsity, m.backend)
    return j
end

function NLPModels.hess_coord!(
    m::CompressedNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    h::AbstractVector;
    obj_weight = one(eltype(x)),
)
    NLPModels.hess_coord!(m.inner, x, y, m.buffer; obj_weight = obj_weight)
    _compress!(h, m.buffer, m.hptr, m.hsparsity, m.backend)
    return h
end

function NLPModels.jac_structure!(
    m::CompressedNLPModel,
    I::AbstractVector,
    J::AbstractVector,
)
    _structure!(I, J, m.jptr, m.jsparsity, m.backend)
    return I, J
end

function NLPModels.hess_structure!(
    m::CompressedNLPModel,
    I::AbstractVector,
    J::AbstractVector,
)
    _structure!(I, J, m.hptr, m.hsparsity, m.backend)
    return I, J
end

function _compress!(V, buffer, ptr, sparsity, backend::Nothing)
    fill!(V, zero(eltype(V)))
    @simd for i = 1:(length(ptr)-1)
        for j = ptr[i]:(ptr[i+1]-1)
            V[i] += buffer[sparsity[j][2]]
        end
    end
end

function _structure!(I, J, ptr, sparsity, backend::Nothing)
    @simd for i = 1:(length(ptr)-1)
        J[i], I[i] = sparsity[ptr[i]][1]
    end
end

export WrapperNLPModel, TimedNLPModel, CompressedNLPModel
