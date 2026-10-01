module SubexprTest

using Test, ExaModels
import NLPModels

# Builds the same nested two-stage model with inlined (buffered = false) or
# buffered (buffered = true) subexpressions; results must be identical.
function _nested_model(; buffered = false)
    c = ExaCore(concrete = Val(true))
    c, x = add_var(c, 10)
    c, s = add_expr(c, (x[i]^2 + sin(x[i+1]) for i in 1:9); buffered)
    c, t = add_expr(c, (s[i] * s[i+1] + x[i] for i in 1:8); buffered)
    c, _ = add_obj(c, (exp(t[i] / 100) * t[i] + cos(x[i]) for i in 1:8))
    c, _ = add_con(c, (t[i] + x[i+2]^2 for i in 1:8))
    return ExaModel(c)
end

function _twodim_model(; buffered = false)
    c = ExaCore(concrete = Val(true))
    c, x = add_var(c, 1:3, 1:4)
    itr = [(i, k) for i in 1:3, k in 1:4]
    c, s = add_expr(c, (x[i, k]^2 + i * k for (i, k) in itr); buffered)
    c, _ = add_obj(c, (s[i, k] * x[i, k] for (i, k) in itr))
    return ExaModel(c)
end

# Assemble the (duplicate-summing) COO Jacobian into a dense matrix at the
# model's reference point xv (captured by closure argument).
function _dense_jac(m, xv = [sin(3i) + 0.5 for i in 1:10])
    rows = zeros(Int, m.meta.nnzj)
    cols = zeros(Int, m.meta.nnzj)
    NLPModels.jac_structure!(m, rows, cols)
    vals = zeros(m.meta.nnzj)
    NLPModels.jac_coord!(m, xv, vals)
    J = zeros(m.meta.ncon, m.meta.nvar)
    for (r, c, v) in zip(rows, cols, vals)
        J[r, c] += v
    end
    return J
end

# Dense symmetric Hessian from the lower-triangular COO (duplicates summed).
function _dense_hess(m, xv, yv; obj_weight = 1.0)
    rows = zeros(Int, m.meta.nnzh)
    cols = zeros(Int, m.meta.nnzh)
    NLPModels.hess_structure!(m, rows, cols)
    vals = zeros(m.meta.nnzh)
    if yv === nothing
        NLPModels.hess_coord!(m, xv, vals; obj_weight)
    else
        NLPModels.hess_coord!(m, xv, yv, vals; obj_weight)
    end
    H = zeros(m.meta.nvar, m.meta.nvar)
    for (r, c, v) in zip(rows, cols, vals)
        H[r, c] += v
        r != c && (H[c, r] += v)
    end
    return H
end

# Gradient of the Lagrangian σf + yᵀc via grad! and Jacobian transpose.
function _lag_grad(m, xv, yv)
    g = zeros(m.meta.nvar)
    NLPModels.grad!(m, xv, g)
    J = _dense_jac(m, xv)
    return g .+ J' * yv
end

function runtests()
    @testset "Buffered subexpressions (first-order)" begin
        m0 = _nested_model(buffered = false)
        m1 = _nested_model(buffered = true)
        xv = [sin(3i) + 0.5 for i in 1:10]
        h = 1e-6

        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1e-14

        g0 = zeros(8)
        g1 = zeros(8)
        NLPModels.cons_nln!(m0, xv, g0)
        NLPModels.cons_nln!(m1, xv, g1)
        @test g1 ≈ g0 rtol = 1e-14

        d0 = zeros(10)
        d1 = zeros(10)
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1e-14

        # independent check: finite differences on the buffered model
        h = 1e-6
        fd = [
            (
                NLPModels.obj(m1, [xv[j] + (j == k ? h : 0.0) for j in 1:10]) -
                NLPModels.obj(m1, [xv[j] - (j == k ? h : 0.0) for j in 1:10])
            ) / 2h for k in 1:10
        ]
        @test d1 ≈ fd rtol = 1e-6

        # Jacobian: assemble dense from COO on both paths and compare
        @test _dense_jac(m1) ≈ _dense_jac(m0) rtol = 1e-14

        # Jacobian against finite differences of the constraints
        J1 = _dense_jac(m1)
        for k in 1:10
            gp = zeros(8); gm = zeros(8)
            NLPModels.cons_nln!(m1, [xv[j] + (j == k ? h : 0.0) for j in 1:10], gp)
            NLPModels.cons_nln!(m1, [xv[j] - (j == k ? h : 0.0) for j in 1:10], gm)
            @test J1[:, k] ≈ (gp .- gm) ./ 2h rtol = 1e-5
        end

        # Lagrangian Hessian: assemble dense (symmetric) from COO on both paths
        yv = [cos(2i) for i in 1:8]
        @test _dense_hess(m1, xv, yv) ≈ _dense_hess(m0, xv, yv) rtol = 1e-14
        @test _dense_hess(m1, xv, yv; obj_weight = 0.3) ≈
              _dense_hess(m0, xv, yv; obj_weight = 0.3) rtol = 1e-14
        @test _dense_hess(m1, xv, nothing) ≈ _dense_hess(m0, xv, nothing) rtol = 1e-14

        # Hessian against finite differences of the Lagrangian gradient
        H1 = _dense_hess(m1, xv, yv)
        for k in 1:10
            gp = _lag_grad(m1, [xv[j] + (j == k ? h : 0.0) for j in 1:10], yv)
            gm = _lag_grad(m1, [xv[j] - (j == k ? h : 0.0) for j in 1:10], yv)
            @test H1[:, k] ≈ (gp .- gm) ./ 2h rtol = 1e-5
        end
    end

    # Adversarial sharing: duplicate columns inside one stage element's
    # Jacobian (x[i+1] enters twice), squared subexpressions (diagonal ss
    # expansion), and products of adjacent elements (off-diagonal ss with
    # overlapping columns) — exercises every doubling coefficient.
    @testset "Buffered subexpressions (coefficient edge cases)" begin
        function _edge_model(; buffered = false)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 6)
            c, s = add_expr(c, (x[i] * x[i+1] + sin(x[i]) * x[i+1] for i in 1:5); buffered)
            c, _ = add_obj(c, (s[i]^2 + s[i] * s[i+1] + s[i] * x[i] for i in 1:4))
            c, _ = add_con(c, (s[i]^2 + x[i] * s[i+1] for i in 1:4))
            return ExaModel(c)
        end
        m0 = _edge_model(buffered = false)
        m1 = _edge_model(buffered = true)
        xv = [0.3 + 0.1i for i in 1:6]
        yv = [1.0, -2.0, 0.5, 3.0]

        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1e-14
        d0 = zeros(6); d1 = zeros(6)
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1e-14
        @test _dense_jac(m1, xv) ≈ _dense_jac(m0, xv) rtol = 1e-14
        @test _dense_hess(m1, xv, yv) ≈ _dense_hess(m0, xv, yv) rtol = 1e-14
    end

    @testset "Buffered subexpressions (multi-dimensional)" begin
        m0 = _twodim_model(buffered = false)
        m1 = _twodim_model(buffered = true)
        xv = [cos(i) for i in 1:12]

        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1e-14

        d0 = zeros(12)
        d1 = zeros(12)
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1e-14
    end
end

end # module
