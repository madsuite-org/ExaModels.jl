module ExaModelsKernelAbstractions

import Adapt
import ExaModels: ExaModels, NLPModels
import KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, synchronize, CPU

function getitr(gen::UnitRange{Int64})
    return gen
end
function getitr(gen::Base.Iterators.ProductIterator{NTuple{N,UnitRange{Int64}}}) where {N} end

function ExaModels.getptr(backend, array; cmp = (x, y) -> x != y)

    bitarray = similar(array, Bool, length(array) + 1)
    kergetptr(backend)(cmp, bitarray, array; ndrange = length(array) + 1)

    return ExaModels.findall(identity, bitarray)
end


struct KAExtension{T,VT<:AbstractVector{T},H,VI1,VI2,B}
    backend::B
    objbuffer::VT
    gradbuffer::VT
    gsparsity::VI1
    gptr::VI2
    conbuffer::VT
    conaugsparsity::VI1
    conaugptr::VI2
    prodhelper::H
end

function ExaModels.build_extension(
    c::C;
    prod = false,
    kwargs...,
) where {T,VT<:AbstractVector{T},B<:KernelAbstractions.Backend,C<:ExaModels.ExaCore{T,VT,B}}

    gsparsity = similar(c.x0, Tuple{Int,Int}, c.nnzg)

    _grad_structure!(T, c.backend, c.obj, gsparsity)

    if !isempty(gsparsity)
        ExaModels.sort!(gsparsity; lt = ((i, j), (k, l)) -> i < k)
    end
    gptr = ExaModels.getptr(c.backend, gsparsity; cmp = (x, y) -> x[1] != y[1])

    conaugsparsity = similar(c.x0, Tuple{Int,Int}, c.nconaug)
    _conaug_structure!(T, c.backend, c.cons, conaugsparsity)
    if !isempty(conaugsparsity)
        ExaModels.sort!(conaugsparsity; lt = ((i, j), (k, l)) -> i < k)
    end
    conaugptr = ExaModels.getptr(c.backend, conaugsparsity; cmp = (x, y) -> x[1] != y[1])


    if prod
        # c.nnzj / c.nnzh / c.ncon contain only SIMD contributions; oracle
        # contributions are deferred and accumulated in _build_with_oracle.
        # The SIMD structure builders below only populate entries for c.cons / c.obj.
        simd_nnzj = c.nnzj
        simd_nnzh = c.nnzh
        simd_ncon = c.ncon

        total_nnzj = c.nnzj + sum(getproperty(o, :nnzj) for o in c.oracles; init = 0)
        total_nnzh = c.nnzh + sum(getproperty(o, :nnzh) for o in c.oracles; init = 0) +
            sum(getproperty(o, :nnzh) for o in c.scalar_oracles; init = 0)
        jacbuffer = similar(c.x0, total_nnzj)
        hessbuffer = similar(c.x0, total_nnzh)
        jacsparsityi = similar(c.x0, Tuple{Tuple{Int, Int}, Int}, simd_nnzj)
        hesssparsityi = similar(c.x0, Tuple{Tuple{Int, Int}, Int}, simd_nnzh)

        _jac_structure!(T, c.backend, c.cons, jacsparsityi, nothing)

        jacsparsityj = copy(jacsparsityi)
        _obj_hess_structure!(T, c.backend, c.obj, hesssparsityi, nothing)
        _con_hess_structure!(T, c.backend, c.cons, hesssparsityi, nothing)
        hesssparsityj = copy(hesssparsityi)

        if !isempty(jacsparsityi)
            ExaModels.sort!(jacsparsityi; lt = (((i, j), k), ((n, m), l)) -> i < n)
        end
        jacptri =
            ExaModels.getptr(c.backend, jacsparsityi; cmp = (x, y) -> x[1][1] != y[1][1])

        if !isempty(jacsparsityj)
            ExaModels.sort!(jacsparsityj; lt = (((i, j), k), ((n, m), l)) -> j < m)
        end
        jacptrj =
            ExaModels.getptr(c.backend, jacsparsityj; cmp = (x, y) -> x[1][2] != y[1][2])


        if !isempty(hesssparsityi)
            ExaModels.sort!(hesssparsityi; lt = (((i, j), k), ((n, m), l)) -> i < n)
        end
        hessptri =
            ExaModels.getptr(c.backend, hesssparsityi; cmp = (x, y) -> x[1][1] != y[1][1])
        if !isempty(hesssparsityj)
            ExaModels.sort!(hesssparsityj; lt = (((i, j), k), ((n, m), l)) -> j < m)
        end
        hessptrj =
            ExaModels.getptr(c.backend, hesssparsityj; cmp = (x, y) -> x[1][2] != y[1][2])

        # Per-oracle sorted sparsity + row/col pointers, precomputed once.
        # Used in jprod_nln! / jtprod_nln! / hprod! so we never rebuild them
        # at inference time.  adapt=Val(true) oracles store nothing.
        _o_jac_off = let s = simd_nnzj
            v = [
                begin
                        r = s; s += o.nnzj; r
                    end for o in c.oracles
            ]; v
        end
        _o_hess_off = let s = simd_nnzh
            v = [
                begin
                        r = s; s += o.nnzh; r
                    end for o in c.oracles
            ]; v
        end
        _o_con_off = let s = simd_ncon
            v = [
                begin
                        r = s; s += o.ncon; r
                    end for o in c.oracles
            ]; v
        end
        _SP = Tuple{Tuple{Int, Int}, Int}  # element type shared with SIMD sparsity
        oracle_prod = ntuple(length(c.oracles)) do i
            oracle = c.oracles[i]
            # Skip precomputed sparsity for non-GPU oracles, zero-nnzj, or
            # matrix-free-only oracles (they use jvp!/vjp!/hvp! directly).
            (oracle.adapt === Val(true) || oracle.nnzj == 0 || oracle.jac! === nothing) && return nothing
            off_j = _o_jac_off[i];  off_c = _o_con_off[i];  off_h = _o_hess_off[i]
            # Jac: row-sorted (for Jv) and col-sorted (for J'v)
            jac_sp_i = _oracle_jac_sparsity(oracle, off_j, off_c, c.backend, c.x0)
            ExaModels.sort!(jac_sp_i; lt = (((r1, c1), _), ((r2, c2), __)) -> r1 < r2)
            jac_ptr_i = ExaModels.getptr(c.backend, jac_sp_i; cmp = (a, b) -> a[1][1] != b[1][1])
            jac_sp_j = copy(jac_sp_i)
            ExaModels.sort!(jac_sp_j; lt = (((r1, c1), _), ((r2, c2), __)) -> c1 < c2)
            jac_ptr_j = ExaModels.getptr(c.backend, jac_sp_j; cmp = (a, b) -> a[1][2] != b[1][2])
            # Hess: row-sorted (kersyspmv) and col-sorted (kersyspmv2)
            if oracle.nnzh > 0
                hess_sp_i = _oracle_hess_sparsity(oracle, off_h, c.backend, c.x0)
                ExaModels.sort!(hess_sp_i; lt = (((r1, c1), _), ((r2, c2), __)) -> r1 < r2)
                hess_ptr_i = ExaModels.getptr(c.backend, hess_sp_i; cmp = (a, b) -> a[1][1] != b[1][1])
                hess_sp_j = copy(hess_sp_i)
                ExaModels.sort!(hess_sp_j; lt = (((r1, c1), _), ((r2, c2), __)) -> c1 < c2)
                hess_ptr_j = ExaModels.getptr(c.backend, hess_sp_j; cmp = (a, b) -> a[1][2] != b[1][2])
            else
                hess_sp_i = similar(c.x0, _SP, 0)
                hess_ptr_i = ExaModels.getptr(c.backend, hess_sp_i; cmp = (a, b) -> a[1][1] != b[1][1])
                hess_sp_j = similar(c.x0, _SP, 0)
                hess_ptr_j = hess_ptr_i
            end
            (
                jacsparsityi = jac_sp_i, jacptri = jac_ptr_i,
                jacsparsityj = jac_sp_j, jacptrj = jac_ptr_j,
                hesssparsityi = hess_sp_i, hessptri = hess_ptr_i,
                hesssparsityj = hess_sp_j, hessptrj = hess_ptr_j,
            )
        end

        prodhelper = (
            jacbuffer = jacbuffer,
            jacsparsityi = jacsparsityi,
            jacsparsityj = jacsparsityj,
            jacptri = jacptri,
            jacptrj = jacptrj,
            hessbuffer = hessbuffer,
            hesssparsityi = hesssparsityi,
            hesssparsityj = hesssparsityj,
            hessptri = hessptri,
            hessptrj = hessptrj,
            oracle_prod = oracle_prod,
        )
    else
        prodhelper = nothing
    end

    return KAExtension(
        c.backend,
        similar(c.x0, c.nobj),
        similar(c.x0, c.nnzg),
        gsparsity,
        gptr,
        similar(c.x0, c.nconaug),
        conaugsparsity,
        conaugptr,
        prodhelper,
    )
end

_conaug_structure!(T, backend, ::Tuple{}, sparsity) = nothing
function _conaug_structure!(T, backend, (con, cons...), sparsity)
    _conaug_structure!(T, backend, cons, sparsity)
    con isa ExaModels.ConstraintAugmentation && !isempty(con.itr) &&
        kers(backend)(sparsity, con.f, con.itr, con.oa, con.dims; ndrange = length(con.itr))
end
@kernel function kers(sparsity, @Const(f), @Const(itr), @Const(oa), @Const(dims))
    I = @index(Global)
    @inbounds sparsity[oa+I] = (ExaModels.offset0(f, itr, I, dims), oa + I)
end



_grad_structure!(T, backend, ::Tuple{}, gsparsity) = nothing
function _grad_structure!(T, backend, (obj, objs...), gsparsity)
    _grad_structure!(T, backend, objs, gsparsity)
    ExaModels.sgradient!(backend, gsparsity, obj, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), T(NaN))
end

function ExaModels.jac_structure!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    rows::V,
    cols::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    _ska(m) === nothing ||
        return ExaModels._jac_structure_impl!(m, ExaModels._subexprs(m), rows, cols)
    if !isempty(rows)
        _jac_structure!(T, m.ext.backend, m.cons, rows, cols)
    end
    return rows, cols
end
_jac_structure!(T, backend, ::Tuple{}, rows, cols) = nothing
function _jac_structure!(T, backend, (con, cons...), rows, cols)
    _jac_structure!(T, backend, cons, rows, cols)
    ExaModels.sjacobian!(backend, rows, cols, con, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), T(NaN))
end


function ExaModels.hess_structure!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    rows::V,
    cols::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    _ska(m) === nothing ||
        return ExaModels._hess_structure_impl!(m, ExaModels._subexprs(m), rows, cols)
    if !isempty(rows)
        _obj_hess_structure!(T, m.ext.backend, m.objs, rows, cols)
        _con_hess_structure!(T, m.ext.backend, m.cons, rows, cols)
        end
    return rows, cols
end

_obj_hess_structure!(T, backend, ::Tuple{}, rows, cols) = nothing
function _obj_hess_structure!(T, backend, (obj, objs...), rows, cols)
    _obj_hess_structure!(T, backend, objs, rows, cols)
    ExaModels.shessian!(backend, rows, cols, obj, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), T(NaN), T(NaN))
end
_con_hess_structure!(T, backend, ::Tuple{}, rows, cols) = nothing
function _con_hess_structure!(T, backend, (con, cons...), rows, cols)
    _con_hess_structure!(T, backend, cons, rows, cols)
    ExaModels.shessian!(backend, rows, cols, con, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), T(NaN), T(NaN))
end


_ska(m::ExaModels.ExaModel) = m.ska
_ska(m) = nothing

function ExaModels.obj(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
) where {T,VT,E<:KAExtension}
    _ska(m) === nothing || return _obj_buffered(m, x)
    if !isempty(m.ext.objbuffer)
        _obj(m.ext.backend, m.ext.objbuffer, m.objs, x, m.θ)
            result = ExaModels.sum(m.ext.objbuffer)
        return result
    else
        return zero(T)
    end
end
_obj(backend, objbuffer, ::Tuple{}, x, θ) = nothing
function _obj(backend, objbuffer, (obj, objs...), x, θ)
    _obj(backend, objbuffer, objs, x, θ)
    if !isempty(obj.itr)
        kerf(backend)(objbuffer, obj.f, obj.itr, x, θ; ndrange = length(obj.itr))
    end
end

function ExaModels.cons_nln!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    y::AbstractVector,
) where {T,VT,E<:KAExtension}
    _ska(m) === nothing || return _cons_buffered!(m, x, y)
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
_cons_nln!(backend, y, ::Tuple{}, x, θ) = nothing
function _cons_nln!(backend, y, (con, cons...), x, θ)
    _cons_nln!(backend, y, cons, x, θ)
    if con isa ExaModels.Constraint && !isempty(con.itr)
        kerf(backend)(y, con.f, con.itr, x, θ; ndrange = length(con.itr))
    end
end



_conaugs!(backend, y, ::Tuple{}, x, θ) = nothing
function _conaugs!(backend, y, (con, cons...), x, θ)
    _conaugs!(backend, y, cons, x, θ)
    if con isa ExaModels.ConstraintAugmentation && !isempty(con.itr)
        kerf2(backend)(y, con.f, con.itr, x, θ, con.oa; ndrange = length(con.itr))
    end
end

function ExaModels.grad!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::V,
    y::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    _ska(m) === nothing || return _grad_buffered!(m, x, y)
    gradbuffer = m.ext.gradbuffer

    fill!(y, zero(eltype(y)))
    if !isempty(gradbuffer)
        fill!(gradbuffer, zero(eltype(gradbuffer)))
        _grad!(m.ext.backend, m.ext.gradbuffer, m.objs, x, m.θ)
        compress_to_dense(m.ext.backend)(
            y,
            gradbuffer,
            m.ext.gptr,
            m.ext.gsparsity;
            ndrange = length(m.ext.gptr) - 1,
        )
    end

    return y
end
_grad!(backend, y, ::Tuple{}, x, θ) = nothing
function _grad!(backend, y, (obj, objs...), x, θ)
    _grad!(backend, y, objs, x, θ)
    ExaModels.sgradient!(backend, y, obj, x, θ, one(eltype(y)))
end

function ExaModels.jac_coord!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::V,
    y::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    _ska(m) === nothing || return _jac_buffered!(m, x, y)
    fill!(y, zero(eltype(y)))
    _jac_coord!(m.ext.backend, y, m.cons, x, m.θ)
    return y
end
_jac_coord!(backend, y, ::Tuple{}, x, θ) = nothing
function _jac_coord!(backend, y, (con, cons...), x, θ)
    _jac_coord!(backend, y, cons, x, θ)
    ExaModels.sjacobian!(backend, y, nothing, con, x, θ, one(eltype(y)))
end

function ExaModels.jprod_nln!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
) where {T,VT,E<:KAExtension{T,VT,Nothing}}
    error("Prodhelper is not defined. Use ExaModels(c; prod=true) to use jprod_nln!")
end
function ExaModels.jtprod_nln!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
) where {T,VT,E<:KAExtension{T,VT,Nothing}}
    error("Prodhelper is not defined. Use ExaModels(c; prod=true) to use jtprod_nln!")
end
function ExaModels.jprod_nln!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}
    _ska(m) === nothing || error(
        "jprod/jtprod/hprod with buffered subexpressions are not yet supported on device backends",
    )

    fill!(Jv, zero(eltype(Jv)))
    fill!(m.ext.prodhelper.jacbuffer, zero(eltype(Jv)))
    _jac_coord!(m.ext.backend, m.ext.prodhelper.jacbuffer, m.cons, x, m.θ)
    kerspmv(m.ext.backend)(
        Jv,
        v,
        m.ext.prodhelper.jacsparsityi,
        m.ext.prodhelper.jacbuffer,
        m.ext.prodhelper.jacptri,
        ndrange = length(m.ext.prodhelper.jacptri) - 1,
    )
    return Jv
end
function ExaModels.jtprod_nln!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}
    _ska(m) === nothing || error(
        "jprod/jtprod/hprod with buffered subexpressions are not yet supported on device backends",
    )

    fill!(Jtv, zero(eltype(Jtv)))
    fill!(m.ext.prodhelper.jacbuffer, zero(eltype(Jtv)))
    _jac_coord!(m.ext.backend, m.ext.prodhelper.jacbuffer, m.cons, x, m.θ)
    kerspmv2(m.ext.backend)(
        Jtv,
        v,
        m.ext.prodhelper.jacsparsityj,
        m.ext.prodhelper.jacbuffer,
        m.ext.prodhelper.jacptrj,
        ndrange = length(m.ext.prodhelper.jacptrj) - 1,
    )
    return Jtv
end
function ExaModels.hprod!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    y::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}
    _ska(m) === nothing || error(
        "jprod/jtprod/hprod with buffered subexpressions are not yet supported on device backends",
    )

    if isnothing(m.ext.prodhelper)
        error("Prodhelper is not defined. Use ExaModels(c; prod=true) to use hprod!")
    end

    fill!(Hv, zero(eltype(Hv)))
    fill!(m.ext.prodhelper.hessbuffer, zero(eltype(Hv)))

    _obj_hess_coord!(m.ext.backend, m.ext.prodhelper.hessbuffer, m.objs, x, m.θ, obj_weight)
    _con_hess_coord!(m.ext.backend, m.ext.prodhelper.hessbuffer, m.cons, x, m.θ, y)
    kersyspmv(m.ext.backend)(
        Hv,
        v,
        m.ext.prodhelper.hesssparsityi,
        m.ext.prodhelper.hessbuffer,
        m.ext.prodhelper.hessptri,
        ndrange = length(m.ext.prodhelper.hessptri) - 1,
    )
    kersyspmv2(m.ext.backend)(
        Hv,
        v,
        m.ext.prodhelper.hesssparsityj,
        m.ext.prodhelper.hessbuffer,
        m.ext.prodhelper.hessptrj,
        ndrange = length(m.ext.prodhelper.hessptrj) - 1,
    )

    return Hv
end
function ExaModels.hprod!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}
    _ska(m) === nothing || error(
        "jprod/jtprod/hprod with buffered subexpressions are not yet supported on device backends",
    )

    if isnothing(m.ext.prodhelper)
        error("Prodhelper is not defined. Use ExaModels(c; prod=true) to use hprod!")
    end

    fill!(Hv, zero(eltype(Hv)))
    fill!(m.ext.prodhelper.hessbuffer, zero(eltype(Hv)))

    _obj_hess_coord!(m.ext.backend, m.ext.prodhelper.hessbuffer, m.objs, x, m.θ, obj_weight)
    kersyspmv(m.ext.backend)(
        Hv,
        v,
        m.ext.prodhelper.hesssparsityi,
        m.ext.prodhelper.hessbuffer,
        m.ext.prodhelper.hessptri,
        ndrange = length(m.ext.prodhelper.hessptri) - 1,
    )
    kersyspmv2(m.ext.backend)(
        Hv,
        v,
        m.ext.prodhelper.hesssparsityj,
        m.ext.prodhelper.hessbuffer,
        m.ext.prodhelper.hessptrj,
        ndrange = length(m.ext.prodhelper.hessptrj) - 1,
    )

    return Hv
end

@kernel function kerspmv(y, @Const(x), @Const(coord), @Const(V), @Const(ptr))
    idx = @index(Global)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        y[i] += V[ind] * x[j]
    end
end
@kernel function kerspmv2(y, @Const(x), @Const(coord), @Const(V), @Const(ptr))
    idx = @index(Global)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        y[j] += V[ind] * x[i]
    end
end
@kernel function kersyspmv(y, @Const(x), @Const(coord), @Const(V), @Const(ptr))
    idx = @index(Global)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        y[i] += V[ind] * x[j]
    end
end
@kernel function kersyspmv2(y, @Const(x), @Const(coord), @Const(V), @Const(ptr))
    idx = @index(Global)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        if i != j
            y[j] += V[ind] * x[i]
        end
    end
end



function ExaModels.hess_coord!(
    m::ExaModels.AbstractExaModel{T,VT,E},
        x::AbstractVector,
        hess::AbstractVector;
        obj_weight = one(T),
    ) where {T, VT, E <: KAExtension}
    _ska(m) === nothing || return _hess_buffered!(m, x, nothing, hess, T(obj_weight))
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.ext.backend, hess, m.objs, x, m.θ, T(obj_weight))
    return hess
end

function ExaModels.hess_coord!(
        m::ExaModels.AbstractExaModel{T, VT, E},
        x::AbstractVector,
        y::AbstractVector,
        hess::AbstractVector;
        obj_weight = one(T),
    ) where {T, VT, E <: KAExtension}
    _ska(m) === nothing || return _hess_buffered!(m, x, y, hess, T(obj_weight))
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.ext.backend, hess, m.objs, x, m.θ, T(obj_weight))
    _con_hess_coord!(m.ext.backend, hess, m.cons, x, m.θ, y)
    return hess
end
_obj_hess_coord!(backend, hess, ::Tuple{}, x, θ, obj_weight) = nothing
function _obj_hess_coord!(backend, hess, (obj, objs...), x, θ, obj_weight)
    _obj_hess_coord!(backend, hess, objs, x, θ, obj_weight)
    ExaModels.shessian!(backend, hess, nothing, obj, x, θ, obj_weight, zero(eltype(hess)))
end
_con_hess_coord!(backend, hess, ::Tuple{}, x, θ, y) = nothing
function _con_hess_coord!(backend, hess, (con, cons...), x, θ, y)
    _con_hess_coord!(backend, hess, cons, x, θ, y)
    ExaModels.shessian!(backend, hess, nothing, con, x, θ, y, zero(eltype(hess)))
end


function ExaModels.sgradient!(
    backend::B,
    y,
    f,
    x,
    θ,
    adj,
) where {B<:KernelAbstractions.Backend}

    if !isempty(f.itr)
        kerg(backend)(y, f.f, f.itr, x, θ, adj; ndrange = length(f.itr))
    end
end

function ExaModels.sjacobian!(
    backend::B,
    y1,
    y2,
    f,
    x,
    θ,
    adj,
) where {B<:KernelAbstractions.Backend}
    if !isempty(f.itr)
        kerj(backend)(y1, y2, f.f, f.itr, x, θ, adj, ExaModels._constraint_dims(f); ndrange = length(f.itr))
    end
end

function ExaModels.shessian!(
    backend::B,
    y1,
    y2,
    f,
    x,
    θ,
    adj,
    adj2,
) where {B<:KernelAbstractions.Backend}
    if !isempty(f.itr)
        kerh(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2; ndrange = length(f.itr))
    end
end

function ExaModels.shessian!(
    backend::B,
    y1,
    y2,
    f,
    x,
    θ,
    adj::V,
    adj2,
) where {B<:KernelAbstractions.Backend,V<:AbstractVector}
    if !isempty(f.itr)
        kerh2(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2, ExaModels._constraint_dims(f); ndrange = length(f.itr))
    end
end

@kernel function kerh(
    y1,
    y2,
    @Const(f),
    @Const(itr),
    @Const(x),
    @Const(θ),
    @Const(adj1),
    @Const(adj2)
)
    I = @index(Global)
    @inbounds ExaModels.hrpass0(
        f(itr[I], ExaModels.SecondAdjointNodeSource(x), θ),
        f.comp2,
        y1,
        y2,
        ExaModels.offset2(f, I),
        0,
        adj1,
        adj2,
    )
end

@kernel function kerh2(
    y1,
    y2,
    @Const(f),
    @Const(itr),
    @Const(x),
    @Const(θ),
    @Const(adjs1),
    @Const(adj2),
    @Const(dims)
)
    I = @index(Global)
    @inbounds ExaModels.hrpass0(
        f(itr[I], ExaModels.SecondAdjointNodeSource(x), θ),
        f.comp2,
        y1,
        y2,
        ExaModels.offset2(f, I),
        0,
        adjs1[ExaModels.offset0(f, itr, I, dims)],
        adj2,
    )
end

@kernel function kerj(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj), @Const(dims))
    I = @index(Global)
    @inbounds ExaModels.jrpass(
        f(itr[I], ExaModels.AdjointNodeSource(x), θ),
        f.comp1,
        ExaModels.offset0(f, itr, I, dims),
        y1,
        y2,
        ExaModels.offset1(f, I),
        0,
        adj,
    )
end

@kernel function kerg(y, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj))
    I = @index(Global)
    @inbounds ExaModels.grpass(
        f(itr[I], ExaModels.AdjointNodeSource(x), θ),
        f.comp1,
        y,
        ExaModels.offset1(f, I),
        0,
        adj,
    )
end

@kernel function kerf(y, @Const(f), @Const(itr), @Const(x), @Const(θ))
    I = @index(Global)
    @inbounds y[ExaModels.offset0(f, itr, I)] = f(itr[I], x, θ)
end
@kernel function kerf2(y, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(oa))
    I = @index(Global)
    @inbounds y[oa+I] = f(itr[I], x, θ)
end


@kernel function compress_to_dense(y, @Const(y0), @Const(ptr), @Const(sparsity))
    I = @index(Global)
    @inbounds for j = ptr[I]:(ptr[I+1]-1)
        (k, l) = sparsity[j]
        y[k] += y0[l]
    end
end

@kernel function kergetptr(cmp, bitarray, @Const(array))
    I = @index(Global)
    @inbounds if I == 1
        bitarray[I] = true
    elseif I == length(array) + 1
        bitarray[I] = true
    else
        i0 = array[I-1]
        i1 = array[I]

        if cmp(i0, i1)
            bitarray[I] = true
        else
            bitarray[I] = false
        end
    end
end

ExaModels.getbackend(m::ExaModels.AbstractExaModel{T,VT,E}) where {T,VT,E<:KAExtension} =
    m.ext.backend

# ── GPU-specific overrides for ExaModelWithOracle ────────────────────────────
#
# When the backend is a KA GPU (CUDA, ROCm, …) the generic oracle.jl methods
# would fall back to CPU scalar loops on device arrays.  The specialisations
# below dispatch on E<:KAExtension and are therefore *more specific* than the
# plain ExaModelWithOracle methods, so Julia's method resolution picks them for
# GPU models.
#
# Design principle:
#   • SIMD symbolic part  → GPU kernels (same as the non-oracle path)
#   • Oracle callback part → CPU bridge: Array(gpu) → callback → copyto!(gpu, cpu)
#     Oracle callbacks are always CPU functions (including PyCall wrappers).

function ExaModels.obj(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        x::AbstractVector,
    ) where {T, VT, E <: KAExtension}
    # SIMD part via GPU kernel (avoids scalar indexing on device arrays)
    if !isempty(m.ext.objbuffer)
        _obj(m.ext.backend, m.ext.objbuffer, m.objs, x, m.θ)
        val = ExaModels.sum(m.ext.objbuffer)
    else
        val = zero(T)
    end
    # Oracle callbacks are CPU functions; adapt device array to CPU first
    for oracle in m.scalar_oracles
        xin = ExaModels._oracle_input(oracle, x)
        val += oracle.f(xin)
    end
    return val
end

function ExaModels.grad!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        x::V,
        y::V,
    ) where {T, VT, E <: KAExtension, V <: AbstractVector}
    gradbuffer = m.ext.gradbuffer
    fill!(y, zero(eltype(y)))
    if !isempty(gradbuffer)
        fill!(gradbuffer, zero(eltype(gradbuffer)))
        _grad!(m.ext.backend, m.ext.gradbuffer, m.objs, x, m.θ)
        compress_to_dense(m.ext.backend)(
            y,
            gradbuffer,
            m.ext.gptr,
            m.ext.gsparsity;
            ndrange = length(m.ext.gptr) - 1,
        )
    end
    backend = m.ext.backend
    for (i, oracle) in enumerate(m.scalar_oracles)
        cache = m.scalar_oracle_caches[i]
        xin = ExaModels._oracle_input(oracle, x)
        ExaModels._run_with_buf!(oracle, backend, cache.buf_nvar, cache.cpu_nvar) do b
            oracle.grad!(b, xin)
        end
        y .+= cache.buf_nvar
    end
    return y
end

function ExaModels.jac_structure!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        rows::V,
        cols::V,
    ) where {T, VT, E <: KAExtension, V <: AbstractVector}
    if !isempty(rows)
        _jac_structure!(T, m.ext.backend, m.cons, rows, cols)
    end
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        if oracle.nnzj > 0
            rows[cache.jac_idx] .= cache.jac_rows_shifted
            cols[cache.jac_idx] .= cache.jac_cols_copy
        end
    end
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        if ev.nnzj > 0
            rows[cache.jac_idx] .= cache.jac_rows_global
            cols[cache.jac_idx] .= cache.jac_cols_global
        end
    end
    return rows, cols
end

function ExaModels.hess_structure!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        rows::V,
        cols::V,
    ) where {T, VT, E <: KAExtension, V <: AbstractVector}
    if !isempty(rows)
        _obj_hess_structure!(T, m.ext.backend, m.objs, rows, cols)
        _con_hess_structure!(T, m.ext.backend, m.cons, rows, cols)
    end
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        if oracle.nnzh > 0
            rows[cache.hess_idx] .= cache.hess_rows_copy
            cols[cache.hess_idx] .= cache.hess_cols_copy
        end
    end
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        if ev.nnzh > 0
            rows[cache.hess_idx] .= cache.hess_rows_global
            cols[cache.hess_idx] .= cache.hess_cols_global
        end
    end
    return rows, cols
end

function ExaModels.cons_nln!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        x::AbstractVector,
        y::AbstractVector,
    ) where {T, VT, E <: KAExtension}
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
    backend = m.ext.backend
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        xin = ExaModels._oracle_input(oracle, x)
        ExaModels._run_with_buf!(oracle, backend, cache.buf_ncon, cache.cpu_ncon) do b
            oracle.f!(b, xin)
        end
        y[cache.con_idx] .= cache.buf_ncon
    end
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        xin = ExaModels._eval_input(ev, x[cache.var_global_idx])
        ExaModels._run_with_buf!(ev, backend, cache.buf_ncon, cache.cpu_ncon) do b
            ev.f!(b, xin)
        end
        y[cache.con_global_idx] .+= cache.buf_ncon
    end
    return y
end

function ExaModels.jac_coord!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        x::V,
        jac::V,
    ) where {T, VT, E <: KAExtension, V <: AbstractVector}
    fill!(jac, zero(eltype(jac)))
    # SIMD part: GPU-accelerated.
    _jac_coord!(m.ext.backend, jac, m.cons, x, m.θ)
    backend = m.ext.backend
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        oracle.nnzj == 0 && continue
        if oracle.jac! !== nothing
            xin = ExaModels._oracle_input(oracle, x)
            ExaModels._run_with_buf!(oracle, backend, cache.buf_nnzj, cache.cpu_nnzj) do b
                oracle.jac!(b, xin)
            end
            jac[cache.jac_idx] .= cache.buf_nnzj
        else
            ExaModels._jac_reconstruct_via_jvp!(oracle, x, jac, cache, backend)
        end
    end
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        ev.nnzj == 0 && continue
        xin = ExaModels._eval_input(ev, x[cache.var_global_idx])
        if ev.jac! !== nothing
            ExaModels._run_with_buf!(ev, backend, cache.buf_nnzj, cache.cpu_nnzj) do b
                ev.jac!(b, xin)
            end
            jac[cache.jac_idx] .= cache.buf_nnzj
        elseif ev.jvp! !== nothing
            ExaModels._eval_jac_reconstruct_via_jvp!(ev, xin, jac, cache, backend)
        end
    end
    return jac
end

function ExaModels.hess_coord!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        x::AbstractVector,
        y::AbstractVector,
        hess::AbstractVector;
        obj_weight = one(T),
    ) where {T, VT, E <: KAExtension}
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.ext.backend, hess, m.objs, x, m.θ, T(obj_weight))
    _con_hess_coord!(m.ext.backend, hess, m.cons, x, m.θ, y)
    backend = m.ext.backend
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        oracle.nnzh == 0 && continue
        if oracle.hess! !== nothing
            xin = ExaModels._oracle_input(oracle, x)
            win = ExaModels._oracle_input(oracle, y[cache.con_idx])
            ExaModels._run_with_buf!(oracle, backend, cache.buf_nnzh, cache.cpu_nnzh) do b
                oracle.hess!(b, xin, win)
            end
            hess[cache.hess_idx] .= cache.buf_nnzh
        else
            ExaModels._hess_reconstruct_via_hvp!(oracle, x, y, hess, cache, backend)
        end
    end
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        ev.nnzh == 0 && continue
        xin = ExaModels._eval_input(ev, x[cache.var_global_idx])
        win = ExaModels._eval_input(ev, y[cache.con_global_idx])
        if ev.hess! !== nothing
            ExaModels._run_with_buf!(ev, backend, cache.buf_nnzh, cache.cpu_nnzh) do b
                ev.hess!(b, xin, win)
            end
            hess[cache.hess_idx] .= cache.buf_nnzh
        elseif ev.hvp! !== nothing
            ExaModels._eval_hess_reconstruct_via_hvp!(ev, xin, win, hess, cache, backend)
        end
    end
    return hess
end

# ── jprod_nln! / jtprod_nln! / hprod! for ExaModelWithOracle on GPU ──────────
#
# Strategy:
#   • SIMD part: re-use the existing prod-helper GPU kernels (kerspmv etc.)
#     which operate entirely on device memory with no scalar indexing.
#   • Oracle part (gpu=true):  call jac!/hess! with device arrays, then use
#     kerspmv/kersyspmv directly on the oracle's own device sparsity buffers.
#   • Oracle part (gpu=false): CPU bridge; accumulate on host and copyto! back.
#
# For the "prod=false" case (no prodhelper) we still handle the oracle part
# correctly; the SIMD part falls back to the cpu path via the base method.

# Helper: build the ((row,col), global_index) sparsity vector for one oracle's
# Jacobian, in the format expected by kerspmv / kerspmv2.
function _oracle_jac_sparsity(oracle, off_j, off_c, backend, x0)
    n = oracle.nnzj
    cpu = [
        ((oracle.jac_rows[k] + off_c, oracle.jac_cols[k]), off_j + k)
            for k in 1:n
    ]
    dev = similar(x0, eltype(cpu), n)
    copyto!(dev, cpu)
    return dev
end

# Helper: build the symmetric sparsity vector for one oracle's Hessian.
function _oracle_hess_sparsity(oracle, off_h, backend, x0)
    n = oracle.nnzh
    cpu = [
        ((oracle.hess_rows[k], oracle.hess_cols[k]), off_h + k)
            for k in 1:n
    ]
    dev = similar(x0, eltype(cpu), n)
    copyto!(dev, cpu)
    return dev
end

# jprod_nln! – error if no prodhelper for the SIMD part
function ExaModels.jprod_nln!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        x::AbstractVector,
        v::AbstractVector,
        Jv::AbstractVector,
    ) where {T, VT, E <: KAExtension{T, VT, Nothing}}
    error("Prodhelper is not defined. Use ExaModel(c; prod=true) to use jprod_nln!")
end
function ExaModels.jtprod_nln!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        x::AbstractVector,
        v::AbstractVector,
        Jtv::AbstractVector,
    ) where {T, VT, E <: KAExtension{T, VT, Nothing}}
    error("Prodhelper is not defined. Use ExaModel(c; prod=true) to use jtprod_nln!")
end

function ExaModels.jprod_nln!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        x::AbstractVector,
        v::AbstractVector,
        Jv::AbstractVector,
    ) where {T, VT, N <: NamedTuple, E <: KAExtension{T, VT, N}}
    fill!(Jv, zero(eltype(Jv)))
    # SIMD symbolic part via GPU sparse-matvec kernel.
    fill!(m.ext.prodhelper.jacbuffer, zero(eltype(Jv)))
    _jac_coord!(m.ext.backend, m.ext.prodhelper.jacbuffer, m.cons, x, m.θ)
    let _n = length(m.ext.prodhelper.jacptri) - 1
        _n > 0 && kerspmv(m.ext.backend)(
            Jv, v,
            m.ext.prodhelper.jacsparsityi,
            m.ext.prodhelper.jacbuffer,
            m.ext.prodhelper.jacptri;
            ndrange = _n,
        )
    end
    backend = m.ext.backend
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        oracle.nnzj == 0 && continue
        xin = ExaModels._oracle_input(oracle, x)
        vin = ExaModels._oracle_input(oracle, v)
        if ExaModels.has_matfree_jac(oracle)
            ExaModels._run_with_buf!(oracle, backend, cache.buf_ncon, cache.cpu_ncon) do b
                oracle.jvp!(b, xin, vin)
            end
            Jv[cache.con_idx] .+= cache.buf_ncon
        elseif oracle.adapt === Val(false)
            prod_cache = m.ext.prodhelper.oracle_prod[i]
            # `view` (not `getindex`) so the user's `jac!` writes into the buffer
            # in place; `jacbuffer[cache.jac_idx]` would return a fresh copy and
            # the kerspmv below would read zeros from the oracle slice.
            oracle.jac!(view(m.ext.prodhelper.jacbuffer, cache.jac_idx), xin)
            let _n = length(prod_cache.jacptri) - 1
                _n > 0 && kerspmv(m.ext.backend)(
                    Jv, v, prod_cache.jacsparsityi, m.ext.prodhelper.jacbuffer, prod_cache.jacptri;
                    ndrange = _n,
                )
            end
        else
            off_c = m.oracle_con_offsets[i]
            ExaModels._run_with_buf!(oracle, backend, cache.buf_nnzj, cache.cpu_nnzj) do b
                oracle.jac!(b, xin)
            end
            jac_cpu = ExaModels._to_cpu!(backend, cache.buf_nnzj, cache.cpu_nnzj)
            v_cpu = ExaModels._vec_to_cpu(oracle, backend, vin, cache.cpu_nvar)
            delta = m.work_ncon_cpu
            fill!(delta, zero(eltype(delta)))
            for k in 1:oracle.nnzj
                delta[oracle.jac_rows[k] + off_c] += jac_cpu[k] * v_cpu[oracle.jac_cols[k]]
            end
            copyto!(m.work_ncon, delta)
            Jv .+= m.work_ncon
        end
    end
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        ev.nnzj == 0 && continue
        xin = ExaModels._eval_input(ev, x[cache.var_global_idx])
        vin_local = ExaModels._eval_input(ev, v[cache.var_global_idx])
        if ev.jvp! !== nothing
            ExaModels._run_with_buf!(ev, backend, cache.buf_ncon, cache.cpu_ncon) do b
                ev.jvp!(b, xin, vin_local)
            end
            Jv[cache.con_global_idx] .+= cache.buf_ncon
        elseif ev.jac! !== nothing
            ExaModels._run_with_buf!(ev, backend, cache.buf_nnzj, cache.cpu_nnzj) do b
                ev.jac!(b, xin)
            end
            jac_cpu = ExaModels._to_cpu!(backend, cache.buf_nnzj, cache.cpu_nnzj)
            v_cpu = ExaModels._vec_to_cpu(ev, backend, vin_local, cache.cpu_nvar)
            delta = cache.cpu_ncon
            fill!(delta, zero(eltype(delta)))
            for k in 1:ev.nnzj
                delta[ev.jac_rows[k]] += jac_cpu[k] * v_cpu[ev.jac_cols[k]]
            end
            copyto!(cache.buf_ncon, delta)
            Jv[cache.con_global_idx] .+= cache.buf_ncon
        end
    end
    return Jv
end

function ExaModels.jtprod_nln!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        x::AbstractVector,
        v::AbstractVector,
        Jtv::AbstractVector,
    ) where {T, VT, N <: NamedTuple, E <: KAExtension{T, VT, N}}
    fill!(Jtv, zero(eltype(Jtv)))
    fill!(m.ext.prodhelper.jacbuffer, zero(eltype(Jtv)))
    _jac_coord!(m.ext.backend, m.ext.prodhelper.jacbuffer, m.cons, x, m.θ)
    let _n = length(m.ext.prodhelper.jacptrj) - 1
        _n > 0 && kerspmv2(m.ext.backend)(
            Jtv, v,
            m.ext.prodhelper.jacsparsityj,
            m.ext.prodhelper.jacbuffer,
            m.ext.prodhelper.jacptrj;
            ndrange = _n,
        )
    end
    backend = m.ext.backend
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        oracle.nnzj == 0 && continue
        xin = ExaModels._oracle_input(oracle, x)
        if ExaModels.has_matfree_jac(oracle)
            w = ExaModels._oracle_input(oracle, v[cache.con_idx])
            ExaModels._run_with_buf!(oracle, backend, cache.buf_nvar, cache.cpu_nvar) do b
                oracle.vjp!(b, xin, w)
            end
            Jtv .+= cache.buf_nvar
        elseif oracle.adapt === Val(false)
            prod_cache = m.ext.prodhelper.oracle_prod[i]
            # See `jprod_nln!` for the `view` vs `getindex` rationale.
            oracle.jac!(view(m.ext.prodhelper.jacbuffer, cache.jac_idx), xin)
            let _n = length(prod_cache.jacptrj) - 1
                _n > 0 && kerspmv2(m.ext.backend)(
                    Jtv, v, prod_cache.jacsparsityj, m.ext.prodhelper.jacbuffer, prod_cache.jacptrj;
                    ndrange = _n,
                )
            end
        else
            off_c = m.oracle_con_offsets[i]
            vin = ExaModels._oracle_input(oracle, v)
            ExaModels._run_with_buf!(oracle, backend, cache.buf_nnzj, cache.cpu_nnzj) do b
                oracle.jac!(b, xin)
            end
            jac_cpu = ExaModels._to_cpu!(backend, cache.buf_nnzj, cache.cpu_nnzj)
            v_cpu = ExaModels._vec_to_cpu(oracle, backend, vin, m.work_ncon_cpu)
            delta = m.work_nvar_cpu
            fill!(delta, zero(eltype(delta)))
            for k in 1:oracle.nnzj
                delta[oracle.jac_cols[k]] += jac_cpu[k] * v_cpu[oracle.jac_rows[k] + off_c]
            end
            copyto!(m.work_nvar, delta)
            Jtv .+= m.work_nvar
        end
    end
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        ev.nnzj == 0 && continue
        xin = ExaModels._eval_input(ev, x[cache.var_global_idx])
        w_local = ExaModels._eval_input(ev, v[cache.con_global_idx])
        if ev.vjp! !== nothing
            ExaModels._run_with_buf!(ev, backend, cache.buf_nvar, cache.cpu_nvar) do b
                ev.vjp!(b, xin, w_local)
            end
            Jtv[cache.var_global_idx] .+= cache.buf_nvar
        elseif ev.jac! !== nothing
            ExaModels._run_with_buf!(ev, backend, cache.buf_nnzj, cache.cpu_nnzj) do b
                ev.jac!(b, xin)
            end
            jac_cpu = ExaModels._to_cpu!(backend, cache.buf_nnzj, cache.cpu_nnzj)
            w_cpu = ExaModels._vec_to_cpu(ev, backend, w_local, cache.cpu_ncon)
            delta = cache.cpu_nvar
            fill!(delta, zero(eltype(delta)))
            for k in 1:ev.nnzj
                delta[ev.jac_cols[k]] += jac_cpu[k] * w_cpu[ev.jac_rows[k]]
            end
            copyto!(cache.buf_nvar, delta)
            Jtv[cache.var_global_idx] .+= cache.buf_nvar
        end
    end
    return Jtv
end

function ExaModels.hprod!(
        m::ExaModels.ExaModelWithOracle{T, VT, E},
        x::AbstractVector,
        y::AbstractVector,
        v::AbstractVector,
        Hv::AbstractVector;
        obj_weight = one(eltype(x)),
    ) where {T, VT, N <: NamedTuple, E <: KAExtension{T, VT, N}}
    fill!(Hv, zero(eltype(Hv)))
    fill!(m.ext.prodhelper.hessbuffer, zero(eltype(Hv)))
    # SIMD symbolic part.
    _obj_hess_coord!(m.ext.backend, m.ext.prodhelper.hessbuffer, m.objs, x, m.θ, obj_weight)
    _con_hess_coord!(m.ext.backend, m.ext.prodhelper.hessbuffer, m.cons, x, m.θ, y)
    let _n = length(m.ext.prodhelper.hessptri) - 1
        _n > 0 && kersyspmv(m.ext.backend)(
            Hv, v,
            m.ext.prodhelper.hesssparsityi,
            m.ext.prodhelper.hessbuffer,
            m.ext.prodhelper.hessptri;
            ndrange = _n,
        )
    end
    let _n = length(m.ext.prodhelper.hessptrj) - 1
        _n > 0 && kersyspmv2(m.ext.backend)(
            Hv, v,
            m.ext.prodhelper.hesssparsityj,
            m.ext.prodhelper.hessbuffer,
            m.ext.prodhelper.hessptrj;
            ndrange = _n,
        )
    end
    backend = m.ext.backend
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        oracle.nnzh == 0 && continue
        xin = ExaModels._oracle_input(oracle, x)
        win = ExaModels._oracle_input(oracle, y[cache.con_idx])
        vin = ExaModels._oracle_input(oracle, v)
        if ExaModels.has_matfree_hess(oracle)
            ExaModels._run_with_buf!(oracle, backend, cache.buf_nvar, cache.cpu_nvar) do b
                oracle.hvp!(b, xin, win, vin)
            end
            Hv .+= cache.buf_nvar
        elseif oracle.adapt === Val(false)
            prod_cache = m.ext.prodhelper.oracle_prod[i]
            # In-place: `[idx] .= rhs` is `setindex!`, so the buffer slice is
            # actually written. (Contrast with `jprod_nln!`/`jtprod_nln!` where
            # the user's `jac!` is called positionally with `view(...)`.)
            buf_slice = view(m.ext.prodhelper.hessbuffer, cache.hess_idx)
            oracle.hess!(buf_slice, xin, win)
            let _n = length(prod_cache.hessptri) - 1
                _n > 0 && kersyspmv(m.ext.backend)(
                    Hv, v, prod_cache.hesssparsityi, m.ext.prodhelper.hessbuffer, prod_cache.hessptri;
                    ndrange = _n,
                )
            end
            let _n = length(prod_cache.hessptrj) - 1
                _n > 0 && kersyspmv2(m.ext.backend)(
                    Hv, v, prod_cache.hesssparsityj, m.ext.prodhelper.hessbuffer, prod_cache.hessptrj;
                    ndrange = _n,
                )
            end
        else
            ExaModels._run_with_buf!(oracle, backend, cache.buf_nnzh, cache.cpu_nnzh) do b
                oracle.hess!(b, xin, win)
            end
            hess_cpu = ExaModels._to_cpu!(backend, cache.buf_nnzh, cache.cpu_nnzh)
            v_cpu = ExaModels._vec_to_cpu(oracle, backend, vin, cache.cpu_nvar)
            delta = m.work_nvar_cpu
            fill!(delta, zero(eltype(delta)))
            for k in 1:oracle.nnzh
                r, c_ = oracle.hess_rows[k], oracle.hess_cols[k]
                delta[r] += hess_cpu[k] * v_cpu[c_]
                if r != c_
                    delta[c_] += hess_cpu[k] * v_cpu[r]
                end
            end
            copyto!(m.work_nvar, delta)
            Hv .+= m.work_nvar
        end
    end
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        ev.nnzh == 0 && continue
        xin = ExaModels._eval_input(ev, x[cache.var_global_idx])
        win = ExaModels._eval_input(ev, y[cache.con_global_idx])
        vin = ExaModels._eval_input(ev, v[cache.var_global_idx])
        if ev.hvp! !== nothing
            ExaModels._run_with_buf!(ev, backend, cache.buf_nvar, cache.cpu_nvar) do b
                ev.hvp!(b, xin, win, vin)
            end
            Hv[cache.var_global_idx] .+= cache.buf_nvar
        elseif ev.hess! !== nothing
            ExaModels._run_with_buf!(ev, backend, cache.buf_nnzh, cache.cpu_nnzh) do b
                ev.hess!(b, xin, win)
            end
            hess_cpu = ExaModels._to_cpu!(backend, cache.buf_nnzh, cache.cpu_nnzh)
            v_cpu = ExaModels._vec_to_cpu(ev, backend, vin, cache.cpu_nvar)
            # `cpu_nvar2` is a separate buffer from `cpu_nvar` (which `v_cpu`
            # may alias) so we can both read v_cpu and write the delta.
            delta = cache.cpu_nvar2
            fill!(delta, zero(eltype(delta)))
            for k in 1:ev.nnzh
                r, c_ = ev.hess_rows[k], ev.hess_cols[k]
                delta[r] += hess_cpu[k] * v_cpu[c_]
                if r != c_
                    delta[c_] += hess_cpu[k] * v_cpu[r]
                end
            end
            copyto!(cache.buf_nvar2, delta)
            Hv[cache.var_global_idx] .+= cache.buf_nvar2
        end
    end
    return Hv
end

function ExaModels._compress!(V, buffer, ptr, sparsity, backend)
    fill!(V, zero(eltype(V)))
    ker_compress!(backend)(V, buffer, ptr, sparsity; ndrange = length(ptr) - 1)
end

@kernel function ker_compress!(V, @Const(buffer), @Const(ptr), @Const(sparsity))
    i = @index(Global)
    @inbounds for j = ptr[i]:(ptr[i+1]-1)
        V[i] += buffer[sparsity[j][2]]
    end
end

function ExaModels._structure!(I, J, ptr, sparsity, backend)
    ker_structure!(backend)(I, J, ptr, sparsity, ndrange = length(ptr) - 1)
end

@kernel function ker_structure!(I, J, @Const(ptr), @Const(sparsity))
    i = @index(Global)
    @inbounds J[i], I[i] = sparsity[ptr[i]][1]
end

function ExaModels.get_compressed_sparsity(nnz, Ibuffer, Jbuffer, backend)
    sparsity = similar(Ibuffer, Tuple{Tuple{Int,Int},Int}, nnz)
    ker_get_compressed_sparsity(backend)(sparsity, Ibuffer, Jbuffer; ndrange = nnz)
    return sparsity
end
@kernel function ker_get_compressed_sparsity(sparsity, @Const(I), @Const(J))
    i = @index(Global)
    @inbounds sparsity[i] = ((J[i], I[i]), i)
end





# ── merged (family-merged) constraint and objective blocks ──────────────────
# A merged block's row/nonzero offsets live in its MergedRow elements, not in
# the SIMDFunction's affine fields, so the generic kernels' offset1/offset2
# calls would be wrong for them.  These variants read the element.  (The cons
# path needs no variant: kerf resolves the row through the 3-argument
# offset0, which dispatches on the MergedRow iterator.)

@kernel function kerj_m(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj))
    I = @index(Global)
    @inbounds begin
        el = itr[I]
        ExaModels.jrpass(
            f(el, ExaModels.AdjointNodeSource(x), θ),
            f.comp1, el.o0, y1, y2, el.o1, 0, adj,
        )
    end
end

@kernel function kerh_m(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj1), @Const(adj2))
    I = @index(Global)
    @inbounds begin
        el = itr[I]
        ExaModels.hrpass0(
            f(el, ExaModels.SecondAdjointNodeSource(x), θ),
            f.comp2, y1, y2, el.o2, 0, adj1, adj2,
        )
    end
end

@kernel function kerh2_m(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adjs1), @Const(adj2))
    I = @index(Global)
    @inbounds begin
        el = itr[I]
        ExaModels.hrpass0(
            f(el, ExaModels.SecondAdjointNodeSource(x), θ),
            f.comp2, y1, y2, el.o2, 0, adjs1[el.o0], adj2,
        )
    end
end

@kernel function kerg_m(y, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj))
    I = @index(Global)
    @inbounds begin
        el = itr[I]
        ExaModels.grpass(
            f(el, ExaModels.AdjointNodeSource(x), θ),
            f.comp1, y, el.o1, 0, adj,
        )
    end
end

const _MergedCon = ExaModels.Constraint{F, I} where {F, I <: AbstractVector{<:ExaModels.MergedRow}}
const _MergedObj = ExaModels.Objective{F, I} where {F, I <: AbstractVector{<:ExaModels.MergedRow}}

# objectives: the gradient and Hessian passes (values and structure both go
# through these two) read the element's o1/o2.  The objective Hessian weight
# is always scalar, so kerh_m covers it; kerh2_m stays constraint-only.
function ExaModels.sgradient!(backend::B, y, f::_MergedObj, x, θ, adj) where {B <: KernelAbstractions.Backend}
    if !isempty(f.itr)
        kerg_m(backend)(y, f.f, f.itr, x, θ, adj; ndrange = length(f.itr))
    end
end

function ExaModels.shessian!(backend::B, y1, y2, f::_MergedObj, x, θ, adj, adj2) where {B <: KernelAbstractions.Backend}
    if !isempty(f.itr)
        kerh_m(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2; ndrange = length(f.itr))
    end
end

function ExaModels.sjacobian!(backend::B, y1, y2, f::_MergedCon, x, θ, adj) where {B <: KernelAbstractions.Backend}
    if !isempty(f.itr)
        kerj_m(backend)(y1, y2, f.f, f.itr, x, θ, adj; ndrange = length(f.itr))
    end
end

function ExaModels.shessian!(backend::B, y1, y2, f::_MergedCon, x, θ, adj, adj2) where {B <: KernelAbstractions.Backend}
    if !isempty(f.itr)
        kerh_m(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2; ndrange = length(f.itr))
    end
end

function ExaModels.shessian!(backend::B, y1, y2, f::_MergedCon, x, θ, adj::V, adj2) where {B <: KernelAbstractions.Backend, V <: AbstractVector}
    if !isempty(f.itr)
        kerh2_m(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2; ndrange = length(f.itr))
    end
end

include("ExaModelsKABuffered.jl")


end # module ExaModelsKernelAbstractions
