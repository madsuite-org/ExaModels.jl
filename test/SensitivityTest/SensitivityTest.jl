module SensitivityTest

using Test
using ExaModels
import NLPModels: jac_structure!, jac_coord!, hess_structure!, hess_coord!
import ExaModels: get_nnzj_par, get_nnzh_par, jac_structure_par!, jac_coord_par!,
    hess_structure_par!, hess_coord_par!, jac_par, hess_par

import ..BACKENDS

function _model(backend, a0, θ0; sensitivity = true)
    c = ExaCore(; backend)
    @add_par(c, a, a0)
    @add_par(c, θ, θ0; sensitivity = sensitivity)
    @add_var(c, x, 2)
    @add_obj(c, θ[1] * x[1]^2)
    @add_con(c, s, θ[i] * x[i]^2 - a[1] for i = 1:2)
    @add_con!(c, s, i => θ[2] * x[i] for i = 1:2)
    return ExaModel(c), θ, a
end

# manually calculate dc/dp, d^2L/dxdp w/ L = w f + y^T c
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
_jac_par(m, x) = _dense(jac_structure_par!, jac_coord_par!, get_nnzj_par(m), m.meta.ncon, length(m.θ), m, x)
_hess_par(m, x, y; kw...) =
    _dense(hess_structure_par!, hess_coord_par!, get_nnzh_par(m), m.meta.nvar, length(m.θ), m, x, y; kw...)
_jac(m, x) = _dense(jac_structure!, jac_coord!, m.meta.nnzj, m.meta.ncon, m.meta.nvar, m, x)
_hess(m, x, y) = _dense(hess_structure!, hess_coord!, m.meta.nnzh, m.meta.nvar, m.meta.nvar, m, x, y)

# check that the dc/dp, d^2L/dxdp calculations are as expected
function runtests()
    @testset "Parametric sensitivity" begin
        @testset "graph" begin
            c = ExaCore()
            c, θ = add_par(c, [2.0, 3.0]; sensitivity = true)
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

            s = ExaModels.ParameterAdjointResult(zeros(2))
            s[1], s[3] = 1.0, 5.0
            @test s.v == [1.0, 0.0] && s[1] == 1.0 && s[3] == 0.0

            k1, k2, k3 = Ref(1), Ref(2), Ref(3)
            comp, n = ExaModels._compressor(Any[k1, k2, k1, k3], Val(4), k -> k !== k3)
            @test comp.inner == (1, 2, 1, ExaModels._EMPTY) && n == 2
        end

        for backend in BACKENDS
            a0, θ0 = [1.0], [2.0, 3.0]
            xh, yh, w = [1.5, -2.0], [0.5, -1.0], 0.3
            m, θ, a = _model(backend, a0, θ0)
            x, y = ExaModels.convert_array(xh, backend), ExaModels.convert_array(yh, backend)
            J, H = _expected(xh, yh, 1.0)
            Hw = _expected(xh, yh, w)[2]
            zero_a = zeros(2)

            @test _jac_par(m, x) ≈ [zero_a J]
            @test _hess_par(m, x, y) ≈ [zero_a H]
            @test _hess_par(m, x, y; obj_weight = w) ≈ [zero_a Hw]

            A = fill!(similar(x, 2, 2), 0.0)
            rows = ExaModels.convert_array([1, 1, 2, 2], backend)
            cols = ExaModels.convert_array([2, 2, 1, 3], backend)
            vals = ExaModels.convert_array([1.0, 2.0, 3.0, 4.0], backend)
            colmap = ExaModels.convert_array([0, 1, 2], backend)
            @test Array(ExaModels._scatter_par!(m.ext, A, rows, cols, vals, colmap)) == [3.0 0.0; 0.0 4.0]
            @test Array(ExaModels._scatter_par!(m.ext, A, rows[1:0], cols[1:0], vals[1:0], colmap)) == [3.0 0.0; 0.0 4.0]

            colmap, n = ExaModels._par_colmap(m)
            @test Array(colmap) == [0, 1, 2] && n == 2
            @test Array(ExaModels._par_colmap(m, θ)[1]) == [0, 1, 2]
            @test_throws ArgumentError ExaModels._par_colmap(m, a)

            @test Array(jac_par(m, x)) ≈ J && Array(jac_par(m, x, θ)) ≈ J
            @test Array(hess_par(m, x, y)) ≈ H && Array(hess_par(m, x, y, θ; obj_weight = w)) ≈ Hw
            @test m(x, y) == (hess_par(m, x, y), jac_par(m, x)) == m(θ, x, y)
            @test_throws ArgumentError jac_par(m, x, a)

            m0 = _model(backend, a0, θ0; sensitivity = false)[1]
            @test get_nnzj_par(m0) == get_nnzh_par(m0) == 0
            @test size(jac_par(m0, x)) == (2, 0) && size(hess_par(m0, x, y)) == (2, 0)
            @test (m0.meta.nnzj, m0.meta.nnzh) == (m.meta.nnzj, m.meta.nnzh)
            @test _jac(m0, x) == _jac(m, x) && _hess(m0, x, y) == _hess(m, x, y)

            @test Array(m[θ]) == θ0
            m[θ] = 2θ0
            @test Array(m[θ]) == 2θ0
        end
    end
end

end # module
