module SubexprTest

using Test, ExaModels
using KernelAbstractions
using MadNLP, NLPModelsIpopt
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

        # matrix-vector products against the inlined model
        vv = [cos(5i) for i in 1:10]
        vc = [sin(7i) for i in 1:8]
        Jv0 = zeros(8); Jv1 = zeros(8)
        NLPModels.jprod_nln!(m0, xv, vv, Jv0)
        NLPModels.jprod_nln!(m1, xv, vv, Jv1)
        @test Jv1 ≈ Jv0 rtol = 1e-14
        Jtv0 = zeros(10); Jtv1 = zeros(10)
        NLPModels.jtprod_nln!(m0, xv, vc, Jtv0)
        NLPModels.jtprod_nln!(m1, xv, vc, Jtv1)
        @test Jtv1 ≈ Jtv0 rtol = 1e-14
        Hv0 = zeros(10); Hv1 = zeros(10)
        NLPModels.hprod!(m0, xv, yv, vv, Hv0; obj_weight = 0.7)
        NLPModels.hprod!(m1, xv, yv, vv, Hv1; obj_weight = 0.7)
        @test Hv1 ≈ Hv0 rtol = 1e-14
        NLPModels.hprod!(m0, xv, vv, Hv0)
        NLPModels.hprod!(m1, xv, vv, Hv1)
        @test Hv1 ≈ Hv0 rtol = 1e-14
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

    # A LINEAR stage used nonlinearly: its curvature contribution is entirely
    # μ·∇s∇sᵀ.  This fails if stage replays go through hrpass0, which skips
    # leaves on purely linear paths (valid only for a zero second-order seed).
    @testset "Buffered subexpressions (linear stage)" begin
        function _lin_model(; buffered = false)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 5)
            c, s = add_expr(c, (2.0x[i] + x[i+1] for i in 1:4); buffered)
            c, t = add_expr(c, (s[i] + s[i+1] for i in 1:3); buffered)  # linear in s
            c, _ = add_obj(c, (t[i]^2 + exp(s[i]) for i in 1:3))
            c, _ = add_con(c, (t[i] * s[i+1] for i in 1:3))
            return ExaModel(c)
        end
        m0 = _lin_model(buffered = false)
        m1 = _lin_model(buffered = true)
        xv = [0.1i - 0.2 for i in 1:5]
        yv = [1.0, -1.5, 2.0]

        d0 = zeros(5); d1 = zeros(5)
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1e-14
        @test _dense_jac(m1, xv) ≈ _dense_jac(m0, xv) rtol = 1e-14
        @test _dense_hess(m1, xv, yv) ≈ _dense_hess(m0, xv, yv) rtol = 1e-14
        @test _dense_hess(m1, xv, nothing) ≈ _dense_hess(m0, xv, nothing) rtol = 1e-14
    end

    # Deep chain (4 stage levels): exercises multi-level sequential
    # elimination, including same-level pairs and chained μ seeds.
    @testset "Buffered subexpressions (deep chain)" begin
        function _chain_model(; buffered = false, n = 8, K = 4)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, n)
            c, s = add_expr(c, (x[i]^2 + sin(x[i+1]) for i in 1:(n-1)); buffered)
            for _ in 2:K
                sp = s
                m = length(sp.size[1])
                c, s = add_expr(c, (sp[i] * sp[i+1] + 0.5sp[i] for i in 1:(m-1)); buffered)
            end
            m = length(s.size[1])
            c, _ = add_obj(c, (s[i] / (1 + s[i]^2) for i in 1:m))
            c, _ = add_con(c, (s[i] * x[i] + x[i+1]^2 for i in 1:m))
            return ExaModel(c)
        end
        # The inlined reference is unusable here by construction (its compile
        # explodes at this depth), so check against finite differences of the
        # buffered model itself.
        m1 = _chain_model(buffered = true)
        xv = [0.2 + 0.05i for i in 1:8]
        yv = [1.0, -2.0, 0.5, 1.5]
        h = 1e-6
        pert(k, d) = [xv[j] + (j == k ? d : 0.0) for j in 1:8]

        d1 = zeros(8)
        NLPModels.grad!(m1, xv, d1)
        fd = [(NLPModels.obj(m1, pert(k, h)) - NLPModels.obj(m1, pert(k, -h))) / 2h for k in 1:8]
        @test d1 ≈ fd rtol = 1e-5

        J1 = _dense_jac(m1, xv)
        for k in 1:8
            gp = zeros(4); gm = zeros(4)
            NLPModels.cons_nln!(m1, pert(k, h), gp)
            NLPModels.cons_nln!(m1, pert(k, -h), gm)
            @test J1[:, k] ≈ (gp .- gm) ./ 2h rtol = 1e-4
        end

        H1 = _dense_hess(m1, xv, yv)
        for k in 1:8
            gp = _lag_grad(m1, pert(k, h), yv)
            gm = _lag_grad(m1, pert(k, -h), yv)
            @test H1[:, k] ≈ (gp .- gm) ./ 2h rtol = 1e-4
        end
    end

    # API surface: macro + named form, kwarg pass-through, refs retrieval,
    # parameters interleaved around buffered stages, set_parameter! after build.
    @testset "Buffered subexpressions (API surface + parameters)" begin
        function _par_model(; buffered = false)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 4)
            c, p = add_par(c, 3; value = 0.5)
            c, s = add_expr(c, (p[i] * x[i]^2 + sin(x[i+1]) for i in 1:3); buffered)
            c, q = add_par(c, 2; value = 2.0)
            c, t = add_expr(c, (s[i] * q[1] + s[i+1] * q[2] for i in 1:2); buffered)
            c, _ = add_obj(c, ((t[i] - 1.0)^2 for i in 1:2))
            return c, p, ExaModel(c)
        end
        c0, p0, m0 = _par_model(buffered = false)
        c1, p1, m1 = _par_model(buffered = true)
        xv = [0.3, -0.2, 0.7, 0.4]
        d0 = zeros(4); d1 = zeros(4)

        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1e-14
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1e-14
        @test _dense_hess(m1, xv, nothing) ≈ _dense_hess(m0, xv, nothing) rtol = 1e-14

        # updating a parameter must not disturb the buffered segments
        set_parameter!(c0, p0, [1.5, -0.5, 2.5])
        set_parameter!(c1, p1, [1.5, -0.5, 2.5])
        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1e-14
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1e-14

        # macro named form registers the handle in the core/model
        c = ExaCore(concrete = Val(true))
        @add_var(c, z, 4)
        @add_expr(c, sb, z[i]^2 + z[i+1] for i in 1:3; buffered = true)
        @add_obj(c, sb[i]^2 for i in 1:3)
        @test sb isa BufferedExpression
        m = ExaModel(c)
        @test m.sb isa BufferedExpression
        @test NLPModels.obj(m, ones(4)) ≈ 12.0 rtol = 1e-14  # 3 terms, (1+1)^2 each

        # non-default backends are refused at add time
        cb = ExaCore(concrete = Val(true), backend = CPU())
        cb, xb = add_var(cb, 3)
        @test_throws ErrorException add_expr(cb, (xb[i]^2 for i in 1:2); buffered = true)
    end

    # End-to-end: a real solver consumes the buffered callbacks; identical
    # problems must reach the same solution as the inlined formulation.
    @testset "Buffered subexpressions (solver round-trip)" begin
        function _solve_model(; buffered = false)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 6)
            c, s = add_expr(c, (x[i]^2 + sin(x[i+1]) for i in 1:5); buffered)
            c, t = add_expr(c, (s[i] * s[i+1] + x[i] for i in 1:4); buffered)
            c, _ = add_obj(c, ((t[i] - 1.0)^2 + 0.1 * x[i]^2 for i in 1:4))
            c, _ = add_con(c, (t[i] + x[i+2] for i in 1:4); lcon = -10.0, ucon = 10.0)
            return ExaModel(c)
        end
        m0 = _solve_model(buffered = false)
        m1 = _solve_model(buffered = true)

        r0 = madnlp(m0; print_level = MadNLP.ERROR)
        r1 = madnlp(m1; print_level = MadNLP.ERROR)
        @test r1.status == r0.status
        @test r1.objective ≈ r0.objective rtol = 1e-8
        @test r1.solution ≈ r0.solution rtol = 1e-6

        i0 = ipopt(m0; print_level = 0)
        i1 = ipopt(m1; print_level = 0)
        @test i1.objective ≈ i0.objective rtol = 1e-8
        @test i1.solution ≈ i0.solution rtol = 1e-6
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
