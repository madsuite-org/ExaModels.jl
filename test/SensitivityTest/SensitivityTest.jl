module SensitivityTest

using Test
using ExaModels
using NLPModelsExtensions
import NLPModels: jac_structure!, jac_coord!, hess_structure!, hess_coord!, DimensionError

import ..BACKENDS

function _model(backend, a0, θ0; sensitivity = Val(true), oracle = nothing)
    c = ExaCore(; backend)
    @add_par(c, a, a0)
    @add_par(c, θ, θ0; sensitivity = sensitivity)
    @add_var(c, x, 2)
    @add_obj(c, θ[1] * x[1]^2)
    @add_con(c, s, θ[i] * x[i]^2 - a[1] for i = 1:2)
    @add_con!(c, s, i => θ[2] * x[i] for i = 1:2)
    oracle === nothing || (c = constraint(c, oracle))
    return ExaModel(c), θ, a
end

_oracle() = VectorNonlinearOracle(
    nvar = 2, ncon = 1, nnzj = 2, nnzh = 0, jac_rows = [1, 1], jac_cols = [1, 2],
    lcon = [0.0], ucon = [0.0], adapt = Val(true),
    f! = (cv, xv) -> (cv[1] = xv[1] - xv[2]; nothing),
    jac! = (vv, xv) -> (vv[1] = 1.0; vv[2] = -1.0; nothing),
)

# manually calculated dc/dp, d^2L/dxdp w/ L = w f + y^T c
function _expected(x, y, w)
    J = [x[1]^2 x[1]; 0.0 x[2]^2+x[2]]
    H = [2w*x[1]+2x[1]*y[1] y[1]; 0.0 2x[2]*y[2]+y[2]]
    return J, H
end

function _dense(structure!, coord!, nnz, nrow, ncol, m, args...; kw...)
    rows, cols = similar(m.meta.x0, Int, nnz), similar(m.meta.x0, Int, nnz)
    vals = similar(m.meta.x0, nnz)
    structure!(m, rows, cols)
    coord!(m, args..., vals; kw...)
    A = zeros(nrow, ncol)
    for (i, j, v) in zip(Array(rows), Array(cols), Array(vals))
        A[i, j] += v
    end
    return A
end
_jac(m, x) = _dense(jac_structure!, jac_coord!, m.meta.nnzj, m.meta.ncon, m.meta.nvar, m, x)
_hess(m, x, y) = _dense(hess_structure!, hess_coord!, m.meta.nnzh, m.meta.nvar, m.meta.nvar, m, x, y)

# check that the dc/dp, d^2L/dxdp calculations are as expected
function runtests()
    @testset "Parametric sensitivity" begin
        @testset "graph" begin
            c = ExaCore()
            c, θ = add_par(c, [2.0, 3.0]; sensitivity = Val(true))
            c, a = add_par(c, [1.0])
            c, x = add_var(c, 1:2)
            @test ExaModels._sensitivity(θ) === Val(true) && ExaModels._sensitivity(a) === Val(false)
            @test θ[1] isa ExaModels.ParameterNode{Int,true} && a[1] isa ExaModels.ParameterNode{Int,false}
            @test ExaModels._has_par(typeof(θ[1] * x[1]^2)) === Val(true)
            @test ExaModels._has_par(typeof(a[1] * x[1]^2)) === Val(false)

            src = ExaModels.ParameterAdjointSource([2.0, 3.0], 5)
            node = ExaModels.ParameterNode(2, Val(true))
            @test node(1, ExaModels.AdjointNodeSource(nothing), src) === ExaModels.AdjointNodeVar(7, 3.0)
            @test node(1, ExaModels.SecondAdjointNodeSource(nothing), src) === ExaModels.SecondAdjointNodeVar(7, 3.0)
            @test ExaModels.ParameterNode(2, Val(false))(1, ExaModels.AdjointNodeSource(nothing), src) === 3.0

            k1, k2, k3 = Ref(1), Ref(2), Ref(3)
            comp, n = ExaModels._compressor(Any[k1, k2, k1, k3], Val(4), k -> k !== k3)
            @test comp.inner == (1, 2, 1, 3) && n == 2

            add_par_val(c, s) = add_par(c, 1:2; sensitivity = s)
            @test (@inferred add_par_val(c, Val(true)))[2] isa ExaModels.Parameter{<:Any,<:Any,<:Any,<:Any,true}
        end

        for backend in BACKENDS
            a0, θ0 = [1.0], [2.0, 3.0]
            xh, yh, w = [1.5, -2.0], [0.5, -1.0], 0.3
            vh, uh = [0.7, -0.4], [1.2, 0.8]
            m, θ, a = _model(backend, a0, θ0)
            x, y = ExaModels.convert_array(xh, backend), ExaModels.convert_array(yh, backend)
            v, u = ExaModels.convert_array(vh, backend), ExaModels.convert_array(uh, backend)
            J, H = _expected(xh, yh, 1.0)
            Hw = _expected(xh, yh, w)[2]

            @test get_npar(m) == 2
            @test get_par_meta(m) == ParametricNLPModelMeta(; npar = 2, nnzj_par = 4, nnzh_par = 5, grad_par_available = false)
            @test Array(jac_par_dense!(m, x, similar(x, 2, 2))) ≈ J
            @test Array(hess_par_dense!(m, x, y, similar(x, 2, 2))) ≈ H
            @test Array(hess_par_dense!(m, x, y, similar(x, 2, 2); obj_weight = w)) ≈ Hw
            @test Array(jprod_par(m, x, v)) ≈ J * vh
            @test Array(jtprod_par(m, x, u)) ≈ J' * uh
            @test Array(hprod_par(m, x, y, v)) ≈ H * vh
            @test Array(hprod_par(m, x, y, v; obj_weight = w)) ≈ Hw * vh
            @test Array(htprod_par(m, x, y, u)) ≈ H' * uh
            @test Array(htprod_par(m, x, y, u; obj_weight = w)) ≈ Hw' * uh
            if backend === nothing
                @test Matrix(jac_par(m, x)) ≈ J && Matrix(hess_par(m, x, y)) ≈ H
            end

            @test Array(get_par(m)) == θ0
            set_par!(m, 2θ0)
            @test Array(get_par(m)) == 2θ0
            set_par!(m, ExaModels.convert_array(θ0, backend))
            @test Array(get_par(m)) == θ0
            @test_throws DimensionMismatch set_par!(m, [1.0, 2.0, 3.0])
            @test_throws DimensionError jprod_par!(m, x, similar(x, 1), similar(x, 2))
            if backend !== nothing
                @test_throws DimensionMismatch jac_par_dense!(m, x, similar(x, 1, 1))
            end

            m0 = _model(backend, a0, θ0; sensitivity = Val(false))[1]
            @test get_npar(m0) == 0 && get_nnzj_par(m0) == 0 && get_nnzh_par(m0) == 0
            @test size(jac_par_dense!(m0, x, similar(x, 2, 0))) == (2, 0)
            @test iszero(Array(jprod_par(m0, x, similar(x, 0))))
            @test (m0.meta.nnzj, m0.meta.nnzh) == (m.meta.nnzj, m.meta.nnzh)
            @test _jac(m0, x) == _jac(m, x) && _hess(m0, x, y) == _hess(m, x, y)

            mo = _model(backend, a0, θ0; oracle = _oracle())[1]
            yo = ExaModels.convert_array([yh; 0.7], backend)
            @test mo isa ExaModels.ExaModelWithOracle && get_npar(mo) == 2
            @test Array(jac_par_dense!(mo, x, similar(x, 3, 2))) ≈ [J; 0.0 0.0]
            @test Array(hess_par_dense!(mo, x, yo, similar(x, 2, 2))) ≈ H
            @test Array(jprod_par(mo, x, v)) ≈ [J * vh; 0.0]
        end
    end
end

end # module
