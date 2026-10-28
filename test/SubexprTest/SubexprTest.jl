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
    c, s = add_expr(c, (x[i]^2 + sin(x[i + 1]) for i in 1:9); buffered)
    c, t = add_expr(c, (s[i] * s[i + 1] + x[i] for i in 1:8); buffered)
    c, _ = add_obj(c, (exp(t[i] / 100.0) * t[i] + cos(x[i]) for i in 1:8))
    c, _ = add_con(c, (t[i] + x[i + 2]^2 for i in 1:8))
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

# ── add_sum fixtures ───────────────────────────────────────────────────────
#
# Sum model: e[i] = x[i]^2 + sin(x[i+1]) buffered, then a ragged weighted
# sum with variable terms and per-slot init:
#   v[j] = init[j] + 2 e[j] + 3 e[j+1] + 0.5 x[j]  (j = 1:4)
#   v[1] += 4 e[4] + 0.25 x[5];  v[2] += 0.5 e[3] twice;  v[3] -= e[1]
# consumed nonlinearly by the objective and constraints.
_sum_elt(x, i) = x[i]^2 + sin(x[i + 1])
function _sum_vexpr(x, j)
    base = 1.5 + 2.0 * _sum_elt(x, j) + 3.0 * _sum_elt(x, j + 1) + 0.5 * x[j]
    return j == 1 ? base + 4.0 * _sum_elt(x, 4) + 0.25 * x[5] :
        j == 2 ? base + 1.0 * _sum_elt(x, 3) :
        j == 3 ? base - 1.0 * _sum_elt(x, 1) : base
end

# Inlined reference: one concrete tree per j (a generator traces ONE symbolic
# tree, so the ragged per-j branches must be baked in with literal indices).
function _sum_ref_model()
    c = ExaCore(concrete = Val(true))
    c, x = add_var(c, 6)
    for j in 1:4
        c, _ = add_obj(c, (_sum_vexpr(x, j)^2 + exp(_sum_vexpr(x, j) / 10.0) for _ in 1:1))
    end
    for j in 1:4
        c, _ = add_con(c, (_sum_vexpr(x, j) * x[j + 1] + x[j]^2 for _ in 1:1))
    end
    return ExaModel(c)
end

function _sum_core(backend)
    c = backend === nothing ? ExaCore(concrete = Val(true)) :
        ExaCore(concrete = Val(true), backend = backend)
    c, x = add_var(c, 6)
    c, e = add_expr(c, (x[i]^2 + sin(x[i + 1]) for i in 1:5); buffered = true)
    pairs = vcat(
        [j => 2.0 * e[j] for j in 1:4],
        [j => 3.0 * e[j + 1] for j in 1:4],
        [1 => 4.0 * e[4], 3 => -1.0 * e[1], 2 => 0.5 * e[3], 2 => 0.5 * e[3]],
        [j => 0.5 * x[j] for j in 1:4],
        [1 => 0.25 * x[5]],
    )
    c, v = add_sum(c, 4, pairs; init = 1.5)
    c, _ = add_obj(c, (v[j]^2 + exp(v[j] / 10.0) for j in 1:4))
    c, _ = add_con(c, (v[j] * x[j + 1] + x[j]^2 for j in 1:4))
    return c
end
_sum_model() = ExaModel(_sum_core(nothing))

# Chained sums interleaved with traced stages, in both directions
# (sum over stage slots, stage over sum slots, sum over sum +
# stage + variable); ragged, so it has no inlined reference.
function _sum_chain_core(backend)
    c = backend === nothing ? ExaCore(concrete = Val(true)) :
        ExaCore(concrete = Val(true), backend = backend)
    c, x = add_var(c, 6)
    c, e = add_expr(c, (x[i]^2 + sin(x[i + 1]) for i in 1:5); buffered = true)
    c, r1 = add_sum(
        c, 3, vcat(
            [j => 2.0 * e[j] for j in 1:3],
            [j => -0.5 * e[j + 2] for j in 1:3],
            [2 => 0.3 * x[6]]
        ); init = [0.1, 0.2, 0.3]
    )
    c, t = add_expr(c, (r1[i] * r1[i + 1] + exp(r1[i] / 10.0) for i in 1:2); buffered = true)
    c, r2 = add_sum(
        c, 2, vcat(
            [j => 1.5 * t[j] for j in 1:2],
            [j => 0.25 * r1[j] for j in 1:2],
            [1 => 0.5 * x[1]]
        )
    )
    c, _ = add_obj(c, (r2[j]^2 + cos(r2[j]) for j in 1:2))
    c, _ = add_con(c, (r2[j] * x[j + 1] + r1[j]^2 for j in 1:2))
    return c
end

# Device comparisons: a Float32 backend (Metal) gets the Float32 AD tolerance.
_dev_rtol(m) = eltype(m.meta.x0) === Float32 ? 1.0e-4 : 1.0e-13

# Device model `m1` against the host model `m0` of the same problem, every
# evaluation entry point through second order.
function _device_vs_host(m0, m1, xh, yh; obj_weight = 0.7)
    tol = _dev_rtol(m1)
    nv, nc = m0.meta.nvar, m0.meta.ncon
    xd = copyto!(similar(m1.meta.x0, nv), xh)
    yd = copyto!(similar(m1.meta.x0, nc), yh)
    @test NLPModels.obj(m1, xd) ≈ NLPModels.obj(m0, xh) rtol = tol
    c0 = zeros(nc)
    cd = fill!(similar(m1.meta.x0, nc), 0)
    NLPModels.cons_nln!(m0, xh, c0)
    NLPModels.cons_nln!(m1, xd, cd)
    @test Array(cd) ≈ c0 rtol = tol
    g0 = zeros(nv)
    gd = fill!(similar(m1.meta.x0, nv), 0)
    NLPModels.grad!(m0, xh, g0)
    NLPModels.grad!(m1, xd, gd)
    @test Array(gd) ≈ g0 rtol = tol
    r = zeros(Int, m1.meta.nnzj)
    cl = zeros(Int, m1.meta.nnzj)
    NLPModels.jac_structure!(m1, r, cl)
    vd = fill!(similar(m1.meta.x0, m1.meta.nnzj), 0)
    NLPModels.jac_coord!(m1, xd, vd)
    J1 = zeros(nc, nv)
    for (i, j, v) in zip(r, cl, Array(vd))
        J1[i, j] += v
    end
    @test J1 ≈ _dense_jac(m0, xh) rtol = tol
    hr = zeros(Int, m1.meta.nnzh)
    hc = zeros(Int, m1.meta.nnzh)
    NLPModels.hess_structure!(m1, hr, hc)
    hv = fill!(similar(m1.meta.x0, m1.meta.nnzh), 0)
    NLPModels.hess_coord!(m1, xd, yd, hv; obj_weight)
    H1 = zeros(nv, nv)
    for (i, j, v) in zip(hr, hc, Array(hv))
        H1[i, j] += v
        i != j && (H1[j, i] += v)
    end
    @test H1 ≈ _dense_hess(m0, xh, yh; obj_weight) rtol = 10tol
    return nothing
end

# Device-backend comparison: the same nested model built on a KA backend must
# match the host inlined reference for every evaluation entry point.
function _device_tests(backend)
    m0 = _nested_model(buffered = false)
    c = ExaCore(concrete = Val(true), backend = backend)
    c, x = add_var(c, 10)
    c, s = add_expr(c, (x[i]^2 + sin(x[i + 1]) for i in 1:9); buffered = true)
    c, t = add_expr(c, (s[i] * s[i + 1] + x[i] for i in 1:8); buffered = true)
    # 100.0, not 100: the `/` derivative rule evaluates `1 / x2`, which for an Int
    # divisor is Int / Int = Float64 and does not compile on Metal (same fixture
    # as the host model above, so the two stay comparable)
    c, _ = add_obj(c, (exp(t[i] / 100.0) * t[i] + cos(x[i]) for i in 1:8))
    c, _ = add_con(c, (t[i] + x[i + 2]^2 for i in 1:8))
    m1 = ExaModel(c)
    tol = _dev_rtol(m1)

    xh = [sin(3i) + 0.5 for i in 1:10]
    xd = copyto!(similar(m1.meta.x0, 10), xh)

    @test NLPModels.obj(m1, xd) ≈ NLPModels.obj(m0, xh) rtol = tol

    g0 = zeros(8)
    gd = fill!(similar(m1.meta.x0, 8), 0)
    NLPModels.cons_nln!(m0, xh, g0)
    NLPModels.cons_nln!(m1, xd, gd)
    @test Array(gd) ≈ g0 rtol = tol

    d0 = zeros(10)
    dd = fill!(similar(m1.meta.x0, 10), 0)
    NLPModels.grad!(m0, xh, d0)
    NLPModels.grad!(m1, xd, dd)
    @test Array(dd) ≈ d0 rtol = tol

    r = zeros(Int, m1.meta.nnzj); cl = zeros(Int, m1.meta.nnzj)
    NLPModels.jac_structure!(m1, r, cl)
    vd = fill!(similar(m1.meta.x0, m1.meta.nnzj), 0)
    NLPModels.jac_coord!(m1, xd, vd)
    J1 = zeros(8, 10)
    for (i, j, v) in zip(r, cl, Array(vd))
        J1[i, j] += v
    end
    @test J1 ≈ _dense_jac(m0, xh) rtol = tol

    yh = [cos(2i) for i in 1:8]
    yd = copyto!(similar(m1.meta.x0, 8), yh)
    hr = zeros(Int, m1.meta.nnzh); hc = zeros(Int, m1.meta.nnzh)
    NLPModels.hess_structure!(m1, hr, hc)
    hv = fill!(similar(m1.meta.x0, m1.meta.nnzh), 0)
    NLPModels.hess_coord!(m1, xd, yd, hv; obj_weight = 0.7)
    H1 = zeros(10, 10)
    for (i, j, v) in zip(hr, hc, Array(hv))
        H1[i, j] += v
        i != j && (H1[j, i] += v)
    end
    @test H1 ≈ _dense_hess(m0, xh, yh; obj_weight = 0.7) rtol = tol

    pvh = [sin(5i) + 0.25 for i in 1:10]
    pvd = copyto!(similar(m1.meta.x0, 10), pvh)
    hp0 = zeros(10)
    hp1 = fill!(similar(m1.meta.x0, 10), 0)
    NLPModels.hprod!(m0, xh, yh, pvh, hp0; obj_weight = 0.7)
    NLPModels.hprod!(m1, xd, yd, pvd, hp1; obj_weight = 0.7)
    @test Array(hp1) ≈ hp0 rtol = tol
    NLPModels.hprod!(m0, xh, pvh, hp0)
    NLPModels.hprod!(m1, xd, pvd, hp1)
    @test Array(hp1) ≈ hp0 rtol = tol

    # A device model with no buffered stage and no prodhelper keeps the host
    # hprod! it had before the buffered methods existed. That fallback indexes
    # the arrays on the host, so it is only usable where the device arrays are
    # host arrays (the CPU backend); on a GPU it scalar-indexes, as it always did.
    backend isa CPU || return nothing
    function plain(bk)
        c2 = ExaCore(concrete = Val(true), backend = bk)
        c2, x2 = add_var(c2, 10)
        c2, _ = add_obj(c2, (x2[i]^2 * x2[i + 1] for i in 1:9))
        c2, _ = add_con(c2, (x2[i] * x2[i + 2] - 1.0 for i in 1:8))
        return ExaModel(c2)
    end
    m2, m3 = plain(backend), plain(nothing)
    hp2 = fill!(similar(m2.meta.x0, 10), 0)
    hp3 = zeros(10)
    NLPModels.hprod!(m3, xh, yh, pvh, hp3; obj_weight = 0.7)
    NLPModels.hprod!(m2, xd, yd, pvd, hp2; obj_weight = 0.7)
    @test Array(hp2) ≈ hp3 rtol = tol
    NLPModels.hprod!(m3, xh, pvh, hp3)
    NLPModels.hprod!(m2, xd, pvd, hp2)
    return @test Array(hp2) ≈ hp3 rtol = tol
end

function runtests(; backends = Any[CPU()])
    @testset "Buffered subexpressions (first-order)" begin
        m0 = _nested_model(buffered = false)
        m1 = _nested_model(buffered = true)
        xv = [sin(3i) + 0.5 for i in 1:10]
        h = 1.0e-6

        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1.0e-14

        g0 = zeros(8)
        g1 = zeros(8)
        NLPModels.cons_nln!(m0, xv, g0)
        NLPModels.cons_nln!(m1, xv, g1)
        @test g1 ≈ g0 rtol = 1.0e-14

        d0 = zeros(10)
        d1 = zeros(10)
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1.0e-14

        # independent check: finite differences on the buffered model
        h = 1.0e-6
        fd = [
            (
                    NLPModels.obj(m1, [xv[j] + (j == k ? h : 0.0) for j in 1:10]) -
                    NLPModels.obj(m1, [xv[j] - (j == k ? h : 0.0) for j in 1:10])
                ) / 2h for k in 1:10
        ]
        @test d1 ≈ fd rtol = 1.0e-6

        # Jacobian: assemble dense from COO on both paths and compare
        @test _dense_jac(m1) ≈ _dense_jac(m0) rtol = 1.0e-14

        # Jacobian against finite differences of the constraints
        J1 = _dense_jac(m1)
        for k in 1:10
            gp = zeros(8); gm = zeros(8)
            NLPModels.cons_nln!(m1, [xv[j] + (j == k ? h : 0.0) for j in 1:10], gp)
            NLPModels.cons_nln!(m1, [xv[j] - (j == k ? h : 0.0) for j in 1:10], gm)
            @test J1[:, k] ≈ (gp .- gm) ./ 2h rtol = 1.0e-5
        end

        # Lagrangian Hessian: assemble dense (symmetric) from COO on both paths
        yv = [cos(2i) for i in 1:8]
        @test _dense_hess(m1, xv, yv) ≈ _dense_hess(m0, xv, yv) rtol = 1.0e-14
        @test _dense_hess(m1, xv, yv; obj_weight = 0.3) ≈
            _dense_hess(m0, xv, yv; obj_weight = 0.3) rtol = 1.0e-14
        @test _dense_hess(m1, xv, nothing) ≈ _dense_hess(m0, xv, nothing) rtol = 1.0e-14

        # Hessian against finite differences of the Lagrangian gradient
        H1 = _dense_hess(m1, xv, yv)
        for k in 1:10
            gp = _lag_grad(m1, [xv[j] + (j == k ? h : 0.0) for j in 1:10], yv)
            gm = _lag_grad(m1, [xv[j] - (j == k ? h : 0.0) for j in 1:10], yv)
            @test H1[:, k] ≈ (gp .- gm) ./ 2h rtol = 1.0e-5
        end

        # matrix-vector products against the inlined model
        vv = [cos(5i) for i in 1:10]
        vc = [sin(7i) for i in 1:8]
        Jv0 = zeros(8); Jv1 = zeros(8)
        NLPModels.jprod_nln!(m0, xv, vv, Jv0)
        NLPModels.jprod_nln!(m1, xv, vv, Jv1)
        @test Jv1 ≈ Jv0 rtol = 1.0e-14
        Jtv0 = zeros(10); Jtv1 = zeros(10)
        NLPModels.jtprod_nln!(m0, xv, vc, Jtv0)
        NLPModels.jtprod_nln!(m1, xv, vc, Jtv1)
        @test Jtv1 ≈ Jtv0 rtol = 1.0e-14
        Hv0 = zeros(10); Hv1 = zeros(10)
        NLPModels.hprod!(m0, xv, yv, vv, Hv0; obj_weight = 0.7)
        NLPModels.hprod!(m1, xv, yv, vv, Hv1; obj_weight = 0.7)
        @test Hv1 ≈ Hv0 rtol = 1.0e-14
        NLPModels.hprod!(m0, xv, vv, Hv0)
        NLPModels.hprod!(m1, xv, vv, Hv1)
        @test Hv1 ≈ Hv0 rtol = 1.0e-14
    end

    # Adversarial sharing: duplicate columns inside one stage element's
    # Jacobian (x[i+1] enters twice), squared subexpressions (diagonal ss
    # expansion), and products of adjacent elements (off-diagonal ss with
    # overlapping columns) — exercises every doubling coefficient.
    @testset "Buffered subexpressions (coefficient edge cases)" begin
        function _edge_model(; buffered = false)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 6)
            c, s = add_expr(c, (x[i] * x[i + 1] + sin(x[i]) * x[i + 1] for i in 1:5); buffered)
            c, _ = add_obj(c, (s[i]^2 + s[i] * s[i + 1] + s[i] * x[i] for i in 1:4))
            c, _ = add_con(c, (s[i]^2 + x[i] * s[i + 1] for i in 1:4))
            return ExaModel(c)
        end
        m0 = _edge_model(buffered = false)
        m1 = _edge_model(buffered = true)
        xv = [0.3 + 0.1i for i in 1:6]
        yv = [1.0, -2.0, 0.5, 3.0]

        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1.0e-14
        d0 = zeros(6); d1 = zeros(6)
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1.0e-14
        @test _dense_jac(m1, xv) ≈ _dense_jac(m0, xv) rtol = 1.0e-14
        @test _dense_hess(m1, xv, yv) ≈ _dense_hess(m0, xv, yv) rtol = 1.0e-14
    end

    # A LINEAR stage used nonlinearly: its curvature contribution is entirely
    # μ·∇s∇sᵀ.  This fails if stage replays go through hrpass0, which skips
    # leaves on purely linear paths (valid only for a zero second-order seed).
    @testset "Buffered subexpressions (linear stage)" begin
        function _lin_model(; buffered = false)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 5)
            c, s = add_expr(c, (2.0x[i] + x[i + 1] for i in 1:4); buffered)
            c, t = add_expr(c, (s[i] + s[i + 1] for i in 1:3); buffered)  # linear in s
            c, _ = add_obj(c, (t[i]^2 + exp(s[i]) for i in 1:3))
            c, _ = add_con(c, (t[i] * s[i + 1] for i in 1:3))
            return ExaModel(c)
        end
        m0 = _lin_model(buffered = false)
        m1 = _lin_model(buffered = true)
        xv = [0.1i - 0.2 for i in 1:5]
        yv = [1.0, -1.5, 2.0]

        d0 = zeros(5); d1 = zeros(5)
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1.0e-14
        @test _dense_jac(m1, xv) ≈ _dense_jac(m0, xv) rtol = 1.0e-14
        @test _dense_hess(m1, xv, yv) ≈ _dense_hess(m0, xv, yv) rtol = 1.0e-14
        @test _dense_hess(m1, xv, nothing) ≈ _dense_hess(m0, xv, nothing) rtol = 1.0e-14
    end

    # Deep chain (4 stage levels): exercises multi-level sequential
    # elimination, including same-level pairs and chained μ seeds.
    @testset "Buffered subexpressions (deep chain)" begin
        function _chain_model(; buffered = false, n = 8, K = 4)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, n)
            c, s = add_expr(c, (x[i]^2 + sin(x[i + 1]) for i in 1:(n - 1)); buffered)
            for _ in 2:K
                sp = s
                m = length(sp.size[1])
                c, s = add_expr(c, (sp[i] * sp[i + 1] + 0.5sp[i] for i in 1:(m - 1)); buffered)
            end
            m = length(s.size[1])
            c, _ = add_obj(c, (s[i] / (1 + s[i]^2) for i in 1:m))
            c, _ = add_con(c, (s[i] * x[i] + x[i + 1]^2 for i in 1:m))
            return ExaModel(c)
        end
        # The inlined reference is unusable here by construction (its compile
        # explodes at this depth), so check against finite differences of the
        # buffered model itself.
        m1 = _chain_model(buffered = true)
        xv = [0.2 + 0.05i for i in 1:8]
        yv = [1.0, -2.0, 0.5, 1.5]
        h = 1.0e-6
        pert(k, d) = [xv[j] + (j == k ? d : 0.0) for j in 1:8]

        d1 = zeros(8)
        NLPModels.grad!(m1, xv, d1)
        fd = [(NLPModels.obj(m1, pert(k, h)) - NLPModels.obj(m1, pert(k, -h))) / 2h for k in 1:8]
        @test d1 ≈ fd rtol = 1.0e-5

        J1 = _dense_jac(m1, xv)
        for k in 1:8
            gp = zeros(4); gm = zeros(4)
            NLPModels.cons_nln!(m1, pert(k, h), gp)
            NLPModels.cons_nln!(m1, pert(k, -h), gm)
            @test J1[:, k] ≈ (gp .- gm) ./ 2h rtol = 1.0e-4
        end

        H1 = _dense_hess(m1, xv, yv)
        for k in 1:8
            gp = _lag_grad(m1, pert(k, h), yv)
            gm = _lag_grad(m1, pert(k, -h), yv)
            @test H1[:, k] ≈ (gp .- gm) ./ 2h rtol = 1.0e-4
        end
    end

    # API surface: macro + named form, kwarg pass-through, refs retrieval,
    # parameters interleaved around buffered stages, set_value! after build.
    @testset "Buffered subexpressions (API surface + parameters)" begin
        function _par_model(; buffered = false)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 4)
            c, p = add_par(c, 3; value = 0.5)
            c, s = add_expr(c, (p[i] * x[i]^2 + sin(x[i + 1]) for i in 1:3); buffered)
            c, q = add_par(c, 2; value = 2.0)
            c, t = add_expr(c, (s[i] * q[1] + s[i + 1] * q[2] for i in 1:2); buffered)
            c, _ = add_obj(c, ((t[i] - 1.0)^2 for i in 1:2))
            return c, p, ExaModel(c)
        end
        c0, p0, m0 = _par_model(buffered = false)
        c1, p1, m1 = _par_model(buffered = true)
        xv = [0.3, -0.2, 0.7, 0.4]
        d0 = zeros(4); d1 = zeros(4)

        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1.0e-14
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1.0e-14
        @test _dense_hess(m1, xv, nothing) ≈ _dense_hess(m0, xv, nothing) rtol = 1.0e-14

        # updating a parameter must not disturb the buffered segments. A model
        # owns its θ, so the update goes to the models, not the cores, and the
        # objective must actually move.
        o_before = NLPModels.obj(m0, xv)
        set_value!(m0, p0, [1.5, -0.5, 2.5])
        set_value!(m1, p1, [1.5, -0.5, 2.5])
        @test !isapprox(NLPModels.obj(m0, xv), o_before; rtol = 1.0e-6)
        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1.0e-14
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1.0e-14

        # macro named form registers the handle in the core/model
        c = ExaCore(concrete = Val(true))
        @add_var(c, z, 4)
        @add_expr(c, sb, z[i]^2 + z[i + 1] for i in 1:3; buffered = true)
        @add_obj(c, sb[i]^2 for i in 1:3)
        @test sb isa BufferedExpression
        m = ExaModel(c)
        @test m.sb isa BufferedExpression
        @test NLPModels.obj(m, ones(4)) ≈ 12.0 rtol = 1.0e-14  # 3 terms, (1+1)^2 each

        # prod = true is refused with buffered subexpressions
        cp = ExaCore(concrete = Val(true))
        cp, xp = add_var(cp, 3)
        cp, sp = add_expr(cp, (xp[i]^2 for i in 1:2); buffered = true)
        cp, _ = add_obj(cp, (sp[i] for i in 1:2))
        @test_throws ErrorException ExaModel(cp; prod = true)
    end

    # End-to-end: a real solver consumes the buffered callbacks; identical
    # problems must reach the same solution as the inlined formulation.
    @testset "Buffered subexpressions (solver round-trip)" begin
        function _solve_model(; buffered = false)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 6)
            c, s = add_expr(c, (x[i]^2 + sin(x[i + 1]) for i in 1:5); buffered)
            c, t = add_expr(c, (s[i] * s[i + 1] + x[i] for i in 1:4); buffered)
            c, _ = add_obj(c, ((t[i] - 1.0)^2 + 0.1 * x[i]^2 for i in 1:4))
            c, _ = add_con(c, (t[i] + x[i + 2] for i in 1:4); lcon = -10.0, ucon = 10.0)
            return ExaModel(c)
        end
        m0 = _solve_model(buffered = false)
        m1 = _solve_model(buffered = true)

        r0 = madnlp(m0; print_level = MadNLP.ERROR)
        r1 = madnlp(m1; print_level = MadNLP.ERROR)
        @test r1.status == r0.status
        @test r1.objective ≈ r0.objective rtol = 1.0e-8
        @test r1.solution ≈ r0.solution rtol = 1.0e-6

        i0 = ipopt(m0; print_level = 0)
        i1 = ipopt(m1; print_level = 0)
        @test i1.objective ≈ i0.objective rtol = 1.0e-8
        @test i1.solution ≈ i0.solution rtol = 1.0e-6
    end

    @testset "Buffered subexpressions (multi-dimensional)" begin
        m0 = _twodim_model(buffered = false)
        m1 = _twodim_model(buffered = true)
        xv = [cos(i) for i in 1:12]

        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1.0e-14

        d0 = zeros(12)
        d1 = zeros(12)
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1.0e-14
    end

    # add_sum: ragged weighted sum over slots and variables with init,
    # duplicate (slot, source) terms, nonlinear consumers — against the inlined
    # per-group reference.
    @testset "add_sum (vs inlined reference)" begin
        m0 = _sum_ref_model()
        m1 = _sum_model()
        xv = [0.3 + 0.1i for i in 1:6]
        yv = [1.0, -2.0, 0.5, 3.0]

        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1.0e-14
        c0 = zeros(4); c1 = zeros(4)
        NLPModels.cons_nln!(m0, xv, c0)
        NLPModels.cons_nln!(m1, xv, c1)
        @test c1 ≈ c0 rtol = 1.0e-14
        d0 = zeros(6); d1 = zeros(6)
        NLPModels.grad!(m0, xv, d0)
        NLPModels.grad!(m1, xv, d1)
        @test d1 ≈ d0 rtol = 1.0e-13
        @test _dense_jac(m1, xv) ≈ _dense_jac(m0, xv) rtol = 1.0e-13
        @test _dense_hess(m1, xv, yv) ≈ _dense_hess(m0, xv, yv) rtol = 1.0e-13
        @test _dense_hess(m1, xv, yv; obj_weight = 0.3) ≈
            _dense_hess(m0, xv, yv; obj_weight = 0.3) rtol = 1.0e-13
        @test _dense_hess(m1, xv, nothing) ≈ _dense_hess(m0, xv, nothing) rtol = 1.0e-13

        # matrix-vector products
        vv = [cos(5i) for i in 1:6]
        vc = [sin(7i) for i in 1:4]
        Jv0 = zeros(4); Jv1 = zeros(4)
        NLPModels.jprod_nln!(m0, xv, vv, Jv0)
        NLPModels.jprod_nln!(m1, xv, vv, Jv1)
        @test Jv1 ≈ Jv0 rtol = 1.0e-13
        Jtv0 = zeros(6); Jtv1 = zeros(6)
        NLPModels.jtprod_nln!(m0, xv, vc, Jtv0)
        NLPModels.jtprod_nln!(m1, xv, vc, Jtv1)
        @test Jtv1 ≈ Jtv0 rtol = 1.0e-13
        Hv0 = zeros(6); Hv1 = zeros(6)
        NLPModels.hprod!(m0, xv, yv, vv, Hv0; obj_weight = 0.7)
        NLPModels.hprod!(m1, xv, yv, vv, Hv1; obj_weight = 0.7)
        @test Hv1 ≈ Hv0 rtol = 1.0e-13
    end

    # The chained fixture has no inlined reference, so check against finite
    # differences on the host; the device loop below compares each backend
    # against this host model.
    @testset "add_sum (chained, FD)" begin
        m = ExaModel(_sum_chain_core(nothing))

        xv = [0.2 + 0.07i for i in 1:6]
        yv = [1.3, -0.8]
        h = 1.0e-6
        pert(k, d) = [xv[i] + (i == k ? d : 0.0) for i in 1:6]

        g = zeros(6)
        NLPModels.grad!(m, xv, g)
        fd = [(NLPModels.obj(m, pert(k, h)) - NLPModels.obj(m, pert(k, -h))) / 2h for k in 1:6]
        @test g ≈ fd rtol = 1.0e-5

        J = _dense_jac(m, xv)
        for k in 1:6
            gp = zeros(2); gm = zeros(2)
            NLPModels.cons_nln!(m, pert(k, h), gp)
            NLPModels.cons_nln!(m, pert(k, -h), gm)
            @test J[:, k] ≈ (gp .- gm) ./ 2h rtol = 1.0e-4
        end

        H = _dense_hess(m, xv, yv)
        for k in 1:6
            gp = _lag_grad(m, pert(k, h), yv)
            gm = _lag_grad(m, pert(k, -h), yv)
            @test H[:, k] ≈ (gp .- gm) ./ 2h rtol = 1.0e-4
        end
    end

    # μ reaching a sum slot only through the cross-pair collapse: two sums over
    # a common older sum slot, coupled bilinearly, with no traced consumer
    # touching the older slot. In the other fixtures every sum slot is also a
    # direct nonlinear consumer leaf, so a seed-time entry marks it μ-active
    # and the collapse path is never the only route.
    @testset "add_sum (μ through cross-pair collapse)" begin
        function build(; buffered)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 4)
            c, e1 = add_expr(c, (x[i]^2 for i in 1:1); buffered)
            c, e2 = add_expr(c, (sin(x[i + 2]) for i in 1:1); buffered)
            if buffered
                c, v = add_sum(c, 1, [1 => e1[1] + 3.0 * e2[1]])
                c, a = add_sum(c, 1, [1 => 3.0 * v[1] + x[2]])
                c, q = add_sum(c, 1, [1 => 2.0 * v[1] + x[1]])
                c, _ = add_obj(c, (a[d] * q[d] for d in 1:1))
                c, _ = add_con(c, (a[d] * q[d] - 1.0 for d in 1:1))
            else
                c, _ = add_obj(c, ((3.0 * (e1[d] + 3.0 * e2[d]) + x[2]) * (2.0 * (e1[d] + 3.0 * e2[d]) + x[1]) for d in 1:1))
                c, _ = add_con(c, ((3.0 * (e1[d] + 3.0 * e2[d]) + x[2]) * (2.0 * (e1[d] + 3.0 * e2[d]) + x[1]) - 1.0 for d in 1:1))
            end
            return ExaModel(c)
        end
        m1 = build(buffered = true)
        m0 = build(buffered = false)
        xv = [0.3, -0.4, 0.7, 1.1]
        yv = [0.6]
        @test NLPModels.obj(m1, xv) ≈ NLPModels.obj(m0, xv) rtol = 1.0e-13
        @test _dense_hess(m1, xv, yv) ≈ _dense_hess(m0, xv, yv) rtol = 1.0e-12
        @test _dense_hess(m1, xv, nothing) ≈ _dense_hess(m0, xv, nothing) rtol = 1.0e-12
    end

    # A single group argument summing many elements (the issue-#331 pattern):
    # v = Σₜ x[2t-1]·x[2t], obj = v² — one element stage, one sum stage, and
    # a genuinely dense rank-1 μ·∇v∇vᵀ Hessian block.
    @testset "add_sum (single dense group)" begin
        n = 12
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, 2n)
        c, s = add_expr(c, (x[2t - 1] * x[2t] for t in 1:n); buffered = true)
        c, v = add_sum(c, 1, (1 => s[t] for t in 1:n))
        c, _ = add_obj(c, ((v[d])^2 for d in 1:1))
        m = ExaModel(c)

        xv = ones(2n)
        @test NLPModels.obj(m, xv) ≈ Float64(n)^2 rtol = 1.0e-14
        g = zeros(2n)
        NLPModels.grad!(m, xv, g)
        @test all(≈(2.0 * n), g)

        xv = [0.1 + 0.05i for i in 1:2n]
        h = 1.0e-6
        pert(k, d) = [xv[i] + (i == k ? d : 0.0) for i in 1:2n]
        H = _dense_hess(m, xv, nothing)
        for k in 1:2n
            gp = zeros(2n); gm = zeros(2n)
            NLPModels.grad!(m, pert(k, h), gp)
            NLPModels.grad!(m, pert(k, -h), gm)
            @test H[:, k] ≈ (gp .- gm) ./ 2h rtol = 1.0e-4
        end
    end

    @testset "add_sum (API surface + errors)" begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, 4)
        c, p = add_par(c, 2; value = 1.0)
        c, s = add_expr(c, (x[i]^2 for i in 1:3); buffered = true)

        # constants fold into init; affine nesting with - / unary minus / division
        c, v = add_sum(c, 2, [1 => 2.0 * s[1] + 1.0, 2 => -(s[2] - s[3]) / 2.0, 2 => 0.5])
        c, _ = add_obj(c, (v[j]^2 for j in 1:2))
        m = ExaModel(c)
        xv = [0.5, -0.3, 0.8, 0.1]
        v1 = 2.0 * xv[1]^2 + 1.0
        v2 = -(xv[2]^2 - xv[3]^2) / 2.0 + 0.5
        @test NLPModels.obj(m, xv) ≈ v1^2 + v2^2 rtol = 1.0e-14

        # non-affine terms, parameter sources, bad indices, init mismatch
        @test_throws ArgumentError add_sum(c, 2, [1 => s[1] * s[2]])
        @test_throws ArgumentError add_sum(c, 2, [1 => s[1]^2])
        @test_throws ArgumentError add_sum(c, 2, [1 => p[1] * s[1]])
        @test_throws ArgumentError add_sum(c, 2, [3 => s[1]])
        @test_throws DimensionMismatch add_sum(c, 2, [1 => s[1]]; init = [1.0])
        @test_throws ArgumentError add_sum(c, 2, [x[1] => s[1]])
        @test_throws ArgumentError add_sum(c, 2, [1 => s[1] / x[2]])
        @test occursin("buffered sum", sprint(show, v))

        # a sum node (exa_sum, or sum(... for ...) inside the macros) is affine
        c, u = add_sum(c, 1, [1 => exa_sum(k -> 2.0 * s[k], Val(1:3)) - 1.0])
        c, _ = add_obj(c, (u[1]^2 for _ in 1:1))
        mu = ExaModel(c)
        u1 = 2.0 * (xv[1]^2 + xv[2]^2 + xv[3]^2) - 1.0
        @test NLPModels.obj(mu, xv) ≈ v1^2 + v2^2 + u1^2 rtol = 1.0e-14
    end

    @testset "add_sum (solver round-trip)" begin
        function _sum_solve(; summed = false)
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 6; start = fill(0.5, 6))
            if summed
                c, e = add_expr(c, (x[i]^2 + sin(x[i + 1]) for i in 1:5); buffered = true)
                c, v = add_sum(
                    c, 4, vcat(
                        [j => 2.0 * e[j] for j in 1:4],
                        [j => 3.0 * e[j + 1] for j in 1:4],
                        [j => 0.5 * x[j] for j in 1:4]
                    ); init = 1.5
                )
                c, _ = add_obj(c, ((v[j] - 1.0)^2 + 0.1 * x[j]^2 for j in 1:4))
                c, _ = add_con(c, (v[j] + x[j + 2] for j in 1:4); lcon = -10.0, ucon = 10.0)
            else
                vex(j) = 1.5 + 2.0 * _sum_elt(x, j) + 3.0 * _sum_elt(x, j + 1) + 0.5 * x[j]
                c, _ = add_obj(c, ((vex(j) - 1.0)^2 + 0.1 * x[j]^2 for j in 1:4))
                c, _ = add_con(c, (vex(j) + x[j + 2] for j in 1:4); lcon = -10.0, ucon = 10.0)
            end
            return ExaModel(c)
        end
        m0 = _sum_solve(summed = false)
        m1 = _sum_solve(summed = true)
        r0 = madnlp(m0; print_level = MadNLP.ERROR)
        r1 = madnlp(m1; print_level = MadNLP.ERROR)
        @test r1.status == r0.status
        @test r1.objective ≈ r0.objective rtol = 1.0e-8
        @test r1.solution ≈ r0.solution rtol = 1.0e-6
    end

    for backend in backends
        @testset "Buffered subexpressions (device: $backend)" begin
            _device_tests(backend)
            _device_merged_plus_buffered(backend)
            _device_augmented_con_buffered(backend)
        end
        @testset "add_sum (device: $backend)" begin
            _device_vs_host(
                _sum_ref_model(), ExaModel(_sum_core(backend)),
                [0.3 + 0.1i for i in 1:6], [1.0, -2.0, 0.5, 3.0],
            )
        end
        @testset "add_sum (device chained: $backend)" begin
            _device_vs_host(
                ExaModel(_sum_chain_core(nothing)), ExaModel(_sum_chain_core(backend)),
                [0.2 + 0.07i for i in 1:6], [1.3, -0.8],
            )
        end
    end
    return
end

# A buffered subexpression together with a plain mergeable constraint added
# after the consumer: on device backends the plain block is family-merged from
# its first add, so the λ-seed Jacobian pass of the Hessian elimination must
# read the nonzero offsets from the MergedRow elements (kerjs_m). The generic
# kernel's offset1(f, I) wrote them into the consumer's cons_ext slots, giving
# a wrong device Hessian while the host was right.
function _device_merged_plus_buffered(backend)
    function build(bk)
        c = ExaCore(concrete = Val(true), backend = bk)
        c, x = add_var(c, 10)
        c, s = add_expr(c, (x[i]^2 + sin(x[i + 1]) for i in 1:9); buffered = true)
        c, _ = add_obj(c, (s[i] * x[i] for i in 1:9))
        c, _ = add_con(c, (s[i] + x[i + 2]^2 for i in 1:8))      # consumer: stays plain
        c, _ = add_con(c, (sin(x[i]) * x[i + 1] for i in 1:9))   # mergeable family
        return ExaModel(c)
    end
    m0 = build(nothing)
    m1 = build(backend)
    @test m1.meta.ncon == 17 == m0.meta.ncon
    xh = [sin(3i) + 0.5 for i in 1:10]
    xd = copyto!(similar(m1.meta.x0, 10), xh)
    yh = [cos(2i) for i in 1:17]
    yd = copyto!(similar(m1.meta.x0, 17), yh)
    H0 = zeros(10, 10)
    hr0 = zeros(Int, m0.meta.nnzh); hc0 = zeros(Int, m0.meta.nnzh)
    NLPModels.hess_structure!(m0, hr0, hc0)
    hv0 = zeros(m0.meta.nnzh)
    NLPModels.hess_coord!(m0, xh, yh, hv0; obj_weight = 0.7)
    for (i, j, v) in zip(hr0, hc0, hv0)
        H0[i, j] += v
    end
    H1 = zeros(10, 10)
    hr = zeros(Int, m1.meta.nnzh); hc = zeros(Int, m1.meta.nnzh)
    NLPModels.hess_structure!(m1, hr, hc)
    hv = fill!(similar(m1.meta.x0, m1.meta.nnzh), 0)
    NLPModels.hess_coord!(m1, xd, yd, hv; obj_weight = 0.7)
    for (i, j, v) in zip(hr, hc, Array(hv))
        H1[i, j] += v
    end
    return @test H1 ≈ H0 rtol = (eltype(m1.meta.x0) == Float32 ? 1.0e-4 : 1.0e-13)
end

# A constraint row added with `add_con!` into a pre-declared block is a
# ConstraintAugmentation rather than a Constraint. The Hessian elimination seeds
# each stage slot with ∂c/∂s weighted by y and replays the stage against that
# seed, which is the only place the (∂c/∂s)·∇²s term of the row comes from; a
# λ-seed pass restricted to plain Constraint blocks leaves it out. The μ seeds
# travel by a separate path and keep the off-diagonals right on their own, so
# the assertion has to be the full dense Hessian against the inlined reference,
# not a norm over the stored nonzeros.
function _device_augmented_con_buffered(backend)
    function build(bk; buffered = true)
        c = ExaCore(concrete = Val(true), backend = bk)
        c, x = add_var(c, 8)
        c, con = add_con(c, 6; lcon = 0.0, ucon = 0.0)
        c, s = add_expr(c, (x[i]^3 + sin(x[i + 1]) for i in 1:7); buffered)
        c, _ = add_obj(c, (s[i] * x[i] for i in 1:7))
        c, _ = add_con!(c, con, (i => s[i] + s[i + 1] * x[i] for i in 1:6))
        return ExaModel(c)
    end
    m0 = build(nothing; buffered = false)
    m1 = build(backend; buffered = true)
    @test m1.meta.ncon == 6 == m0.meta.ncon
    xh = [0.4 + 0.1i for i in 1:8]
    xd = copyto!(similar(m1.meta.x0, 8), xh)
    yh = [cos(2i) + 1.5 for i in 1:6]
    yd = copyto!(similar(m1.meta.x0, 6), yh)
    H0 = _dense_hess(m0, xh, yh; obj_weight = 0.7)
    hr = zeros(Int, m1.meta.nnzh)
    hc = zeros(Int, m1.meta.nnzh)
    NLPModels.hess_structure!(m1, hr, hc)
    hv = fill!(similar(m1.meta.x0, m1.meta.nnzh), 0)
    NLPModels.hess_coord!(m1, xd, yd, hv; obj_weight = 0.7)
    H1 = zeros(8, 8)
    for (i, j, v) in zip(hr, hc, Array(hv))
        H1[i, j] += v
        i != j && (H1[j, i] += v)
    end
    # Float32 backends (Metal) need a looser comparison than Float64 ones
    return @test H1 ≈ H0 rtol = (eltype(m1.meta.x0) == Float32 ? 1.0e-4 : 1.0e-13)
end

end # module
