module OracleTest

using ExaModels
using NLPModels
using SparseArrays
using LinearAlgebra
using Test

import ..BACKENDS

# Backend-aware tolerance: Metal uses Float32 (~1e-7 machine eps), so we relax
# from 1e-12 to 1e-4. Pass a `tight` keyword to use a less-strict default on Float64.
_atol(backend, tight = 1.0e-12) =
    (ExaModels.default_T(backend) == Float32) ? 1.0e-4 : tight

# ── Test problem ───────────────────────────────────────────────────────────────
#
# A small 4-variable problem mixing SIMD symbolic constraints with two
# VectorNonlinearOracle blocks, so every offset calculation can be verified.
#
# Variables:  x ∈ R^4  (all free)
# Objective:  f(x) = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
#   grad(f)  = 2x
#   H_f      = 2·I  (diagonal, 4 NNZ)
#
# Constraint layout:
#   row 1 (SIMD):    x[1] + x[2] = 1           (linear, 2 Jac NNZ, 0 Hess NNZ)
#   row 2 (oracle A): x[3] - x[4] = 0           (linear oracle, 2 Jac NNZ, 0 Hess NNZ)
#   row 3 (oracle B): x[3]^2 + x[4]^2 ≥ 0      (nonlinear oracle, 2 Jac NNZ, 2 Hess NNZ)
#
# Totals: ncon=3, nnzj=6, nnzh = 4(obj) + 2(oracleB) = 6

function _build_oracle_model(backend)
    c = ExaCore(Float64; backend = backend)
    c, x = add_var(c, 4; lvar = -Inf, uvar = Inf, start = [0.5, 0.5, 0.6, 0.4])

    # Objective: sum x[i]^2
    c, _ = add_obj(c, x[i]^2 for i in 1:4)

    # SIMD constraint: x[1] + x[2] = 1
    c, _ = add_con(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)

    # Oracle A: linear  c = x[3] - x[4] = 0
    oracle_A = VectorNonlinearOracle(
        nvar = 4,
        ncon = 1,
        nnzj = 2,
        nnzh = 0,
        jac_rows = [1, 1],
        jac_cols = [3, 4],
        lcon = [0.0],
        ucon = [0.0],
        adapt = Val(true),
        f! = (cv, xv) -> (cv[1] = xv[3] - xv[4]; nothing),
        jac! = (vv, xv) -> (vv[1] = 1.0; vv[2] = -1.0; nothing),
    )
    c = constraint(c, oracle_A)

    # Oracle B: nonlinear  c = x[3]^2 + x[4]^2 ≥ 0  (lcon=-Inf, ucon=Inf for a free residual)
    oracle_B = VectorNonlinearOracle(
        nvar = 4,
        ncon = 1,
        nnzj = 2,
        nnzh = 2,
        jac_rows = [1, 1],
        jac_cols = [3, 4],
        hess_rows = [3, 4],
        hess_cols = [3, 4],
        lcon = [-Inf],
        ucon = [Inf],
        adapt = Val(true),
        f! = (cv, xv) -> (cv[1] = xv[3]^2 + xv[4]^2; nothing),
        jac! = (vv, xv) -> (vv[1] = 2 * xv[3]; vv[2] = 2 * xv[4]; nothing),
        hess! = (hv, xv, yv) -> (hv[1] = 2 * yv[1]; hv[2] = 2 * yv[1]; nothing),
    )
    c = constraint(c, oracle_B)

    return ExaModel(c)
end

# ── Analytic reference values at x0 = [0.5, 0.5, 0.6, 0.4] ───────────────────
const X0 = [0.5, 0.5, 0.6, 0.4]
const OBJ0 = 0.5^2 + 0.5^2 + 0.6^2 + 0.4^2          # 1.02
const GRAD0 = 2 .* X0                                  # [1.0,1.0,1.2,0.8]
const CONS0 = [1.0, 0.2, 0.52]                         # [x1+x2, x3-x4, x3^2+x4^2]

# Dense Jacobian at x0: shape (ncon=3, nvar=4)
#   row 1: ∂(x1+x2)/∂xi       = [1, 1, 0, 0]
#   row 2: ∂(x3-x4)/∂xi       = [0, 0, 1,-1]
#   row 3: ∂(x3^2+x4^2)/∂xi   = [0, 0, 2*x3, 2*x4] = [0,0,1.2,0.8]
function analytic_jac_dense(x)
    J = zeros(3, 4)
    J[1, 1] = 1.0; J[1, 2] = 1.0
    J[2, 3] = 1.0; J[2, 4] = -1.0
    J[3, 3] = 2 * x[3]; J[3, 4] = 2 * x[4]
    return J
end

# Dense Lagrangian Hessian (lower triangle) at x, y=[λ1,λ2,λ3]:
#   H = 2·I + λ3·diag([0,0,2,2])  (SIMD and oracleA are linear → no Hessian)
function analytic_hess_dense(x, y)
    H = diagm(0 => fill(2.0, 4))
    H[3, 3] += 2 * y[3]
    H[4, 4] += 2 * y[3]
    return H
end

# ── Tests ──────────────────────────────────────────────────────────────────────

function test_model_type()
    return @testset "ExaModelWithOracle type" begin
        m = _build_oracle_model(nothing)
        @test m isa ExaModelWithOracle

        # plain model (no oracle) still returns ExaModel
        c2 = ExaCore(Float64)
        c2, x2 = add_var(c2, 3)
        @test ExaModel(c2) isa ExaModel
    end
end

function test_meta(backend)
    return @testset "meta" begin
        m = _build_oracle_model(backend)
        @test m.meta.nvar == 4
        @test m.meta.ncon == 3          # 1 SIMD + 1 oracleA + 1 oracleB
        @test m.meta.nnzj == 6          # 2 SIMD + 2 oracleA + 2 oracleB
        @test m.meta.nnzh == 6          # 4 obj diag + 2 oracleB
        @test Array(m.meta.lcon) ≈ [1.0, 0.0, -Inf]
        @test Array(m.meta.ucon) ≈ [1.0, 0.0, Inf]
    end
end

function test_obj_and_grad(backend)
    return @testset "obj and grad" begin
        m = _build_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)
        @test NLPModels.obj(m, x0) ≈ OBJ0 atol = _atol(backend)
        @test Array(NLPModels.grad(m, x0)) ≈ GRAD0 atol = _atol(backend)
    end
end

function test_cons(backend)
    return @testset "cons_nln!" begin
        m = _build_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)
        g = similar(x0, m.meta.ncon)
        ExaModels.cons_nln!(m, x0, g)
        @test Array(g) ≈ CONS0 atol = _atol(backend)
    end
end

function test_jac(backend)
    return @testset "jac_structure! and jac_coord!" begin
        m = _build_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)

        rows = similar(x0, Int, m.meta.nnzj)
        cols = similar(x0, Int, m.meta.nnzj)
        ExaModels.jac_structure!(m, rows, cols)

        vals = similar(x0, m.meta.nnzj)
        ExaModels.jac_coord!(m, x0, vals)

        # Assemble to dense and compare with analytic
        J_assembled = zeros(3, 4)
        rows_cpu = Array(rows); cols_cpu = Array(cols); vals_cpu = Array(vals)
        for k in eachindex(rows_cpu)
            J_assembled[rows_cpu[k], cols_cpu[k]] += vals_cpu[k]
        end
        @test J_assembled ≈ analytic_jac_dense(X0) atol = _atol(backend)

        # Row indices: SIMD uses rows [1,1], oracleA shifts to [2,2], oracleB to [3,3]
        @test all(1 .<= rows_cpu .<= 3)
        @test all(1 .<= cols_cpu .<= 4)
    end
end

function test_hess(backend)
    return @testset "hess_structure! and hess_coord!" begin
        m = _build_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)
        y0 = ExaModels.convert_array([0.0, 0.0, 1.0], backend)   # λ3=1 activates oracleB Hess

        rows = similar(x0, Int, m.meta.nnzh)
        cols = similar(x0, Int, m.meta.nnzh)
        ExaModels.hess_structure!(m, rows, cols)

        vals = similar(x0, m.meta.nnzh)
        ExaModels.hess_coord!(m, x0, y0, vals)

        # Assemble to dense (lower + upper for symmetry, values accumulate)
        H_assembled = zeros(4, 4)
        rows_cpu = Array(rows); cols_cpu = Array(cols); vals_cpu = Array(vals)
        for k in eachindex(rows_cpu)
            H_assembled[rows_cpu[k], cols_cpu[k]] += vals_cpu[k]
            if rows_cpu[k] != cols_cpu[k]
                H_assembled[cols_cpu[k], rows_cpu[k]] += vals_cpu[k]
            end
        end
        @test H_assembled ≈ analytic_hess_dense(X0, [0.0, 0.0, 1.0]) atol = _atol(backend)

        # obj-only variant (no constraint contribution)
        fill!(vals, 0)
        ExaModels.hess_coord!(m, x0, vals; obj_weight = 1.0)
        H_obj = zeros(4, 4)
        vals_cpu = Array(vals)
        for k in eachindex(rows_cpu)
            H_obj[rows_cpu[k], cols_cpu[k]] += vals_cpu[k]
            if rows_cpu[k] != cols_cpu[k]
                H_obj[cols_cpu[k], rows_cpu[k]] += vals_cpu[k]
            end
        end
        @test H_obj ≈ diagm(0 => fill(2.0, 4)) atol = _atol(backend)
    end
end

function test_jprod_jtprod(backend)
    return @testset "jprod_nln! and jtprod_nln!" begin
        c = ExaCore(Float64; backend = backend)
        c, x = add_var(c, 4; lvar = -Inf, uvar = Inf, start = [0.5, 0.5, 0.6, 0.4])
        c, _ = add_obj(c, x[i]^2 for i in 1:4)
        c, _ = add_con(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)
        oracle_A = VectorNonlinearOracle(
            nvar = 4, ncon = 1, nnzj = 2, nnzh = 0,
            jac_rows = [1, 1], jac_cols = [3, 4],
            lcon = [0.0], ucon = [0.0],
            adapt = Val(true),
            f! = (cv, xv) -> (cv[1] = xv[3] - xv[4]; nothing),
            jac! = (vv, xv) -> (vv[1] = 1.0; vv[2] = -1.0; nothing),
        )
        c = constraint(c, oracle_A)
        oracle_B = VectorNonlinearOracle(
            nvar = 4, ncon = 1, nnzj = 2, nnzh = 2,
            jac_rows = [1, 1], jac_cols = [3, 4],
            hess_rows = [3, 4], hess_cols = [3, 4],
            lcon = [-Inf], ucon = [Inf],
            adapt = Val(true),
            f! = (cv, xv) -> (cv[1] = xv[3]^2 + xv[4]^2; nothing),
            jac! = (vv, xv) -> (vv[1] = 2 * xv[3]; vv[2] = 2 * xv[4]; nothing),
            hess! = (hv, xv, yv) -> (hv[1] = 2 * yv[1]; hv[2] = 2 * yv[1]; nothing),
        )
        c = constraint(c, oracle_B)
        m = ExaModel(c; prod = true)
        x0 = ExaModels.convert_array(X0, backend)
        J = analytic_jac_dense(X0)

        v = ExaModels.convert_array([1.0, -1.0, 2.0, -0.5], backend)  # nvar-dim
        w = ExaModels.convert_array([1.0, 2.0, -1.0], backend)  # ncon-dim

        Jv = similar(x0, m.meta.ncon)
        Jtw = similar(x0, m.meta.nvar)
        ExaModels.jprod_nln!(m, x0, v, Jv)
        ExaModels.jtprod_nln!(m, x0, w, Jtw)

        @test Array(Jv) ≈ J * X0 * 0 .+ J * [1.0, -1.0, 2.0, -0.5] atol = _atol(backend, 1.0e-10)
        @test Array(Jtw) ≈ J' * [1.0, 2.0, -1.0]                    atol = _atol(backend, 1.0e-10)
    end
end

function test_hprod(backend)
    return @testset "hprod!" begin
        c = ExaCore(Float64; backend = backend)
        c, x = add_var(c, 4; lvar = -Inf, uvar = Inf, start = [0.5, 0.5, 0.6, 0.4])
        c, _ = add_obj(c, x[i]^2 for i in 1:4)
        c, _ = add_con(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)
        oracle_A = VectorNonlinearOracle(
            nvar = 4, ncon = 1, nnzj = 2, nnzh = 0,
            jac_rows = [1, 1], jac_cols = [3, 4],
            lcon = [0.0], ucon = [0.0],
            adapt = Val(true),
            f! = (cv, xv) -> (cv[1] = xv[3] - xv[4]; nothing),
            jac! = (vv, xv) -> (vv[1] = 1.0; vv[2] = -1.0; nothing),
        )
        c = constraint(c, oracle_A)
        oracle_B = VectorNonlinearOracle(
            nvar = 4, ncon = 1, nnzj = 2, nnzh = 2,
            jac_rows = [1, 1], jac_cols = [3, 4],
            hess_rows = [3, 4], hess_cols = [3, 4],
            lcon = [-Inf], ucon = [Inf],
            adapt = Val(true),
            f! = (cv, xv) -> (cv[1] = xv[3]^2 + xv[4]^2; nothing),
            jac! = (vv, xv) -> (vv[1] = 2 * xv[3]; vv[2] = 2 * xv[4]; nothing),
            hess! = (hv, xv, yv) -> (hv[1] = 2 * yv[1]; hv[2] = 2 * yv[1]; nothing),
        )
        c = constraint(c, oracle_B)
        m = ExaModel(c; prod = true)
        x0 = ExaModels.convert_array(X0, backend)
        y0 = ExaModels.convert_array([0.0, 0.0, 1.0], backend)
        v = ExaModels.convert_array([1.0, -1.0, 2.0, -0.5], backend)

        Hv = similar(x0, m.meta.nvar)
        ExaModels.hprod!(m, x0, y0, v, Hv)

        H = analytic_hess_dense(X0, [0.0, 0.0, 1.0])
        @test Array(Hv) ≈ H * [1.0, -1.0, 2.0, -0.5] atol = _atol(backend, 1.0e-10)
    end
end

function test_val_false_jprod_jtprod_hprod(backend)
    # Regression test for the `oracle.adapt === Val(false)` branch in
    # `jprod_nln!` / `jtprod_nln!` / `hprod!` on KA backends.  Earlier code wrote
    # `oracle.jac!(jacbuffer[cache.jac_idx], xin)` — `getindex` returns a copy,
    # so the buffer was never updated and the subsequent kerspmv silently read
    # zeros for the oracle's slice.  Mirror `test_jprod_jtprod` / `test_hprod`
    # but with `adapt = Val(false)` and broadcast callbacks so the path is
    # exercised end-to-end against the analytic Jacobian / Hessian.
    return @testset "adapt=Val(false) jprod/jtprod/hprod (prod=true)" begin
        c = ExaCore(Float64; backend = backend)
        c, x = add_var(c, 4; lvar = -Inf, uvar = Inf, start = [0.5, 0.5, 0.6, 0.4])
        c, _ = add_obj(c, x[i]^2 for i in 1:4)
        c, _ = add_con(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)
        oracle_A = VectorNonlinearOracle(
            nvar = 4, ncon = 1, nnzj = 2, nnzh = 0,
            jac_rows = [1, 1], jac_cols = [3, 4],
            lcon = [0.0], ucon = [0.0],
            adapt = Val(false),
            f! = (cv, xv) -> (cv .= xv[3:3] .- xv[4:4]; nothing),
            # Constants written one element at a time via 1-element broadcast —
            # `vv .= [1.0, -1.0]` would pass a CPU `Vector` into a CUDA kernel.
            jac! = (vv, xv) -> begin
                vv[1:1] .= one(eltype(xv))
                vv[2:2] .= -one(eltype(xv))
                nothing
            end,
        )
        c = constraint(c, oracle_A)
        oracle_B = VectorNonlinearOracle(
            nvar = 4, ncon = 1, nnzj = 2, nnzh = 2,
            jac_rows = [1, 1], jac_cols = [3, 4],
            hess_rows = [3, 4], hess_cols = [3, 4],
            lcon = [-Inf], ucon = [Inf],
            adapt = Val(false),
            f! = (cv, xv) -> (cv .= xv[3:3] .^ 2 .+ xv[4:4] .^ 2; nothing),
            jac! = (vv, xv) -> (vv .= 2 .* xv[3:4]; nothing),
            hess! = (hv, xv, yv) -> (hv .= 2 .* yv[1:1]; nothing),
        )
        c = constraint(c, oracle_B)
        m = ExaModel(c; prod = true)

        x0 = ExaModels.convert_array(X0, backend)
        v = ExaModels.convert_array([1.0, -1.0, 2.0, -0.5], backend)
        w = ExaModels.convert_array([1.0, 2.0, -1.0], backend)
        y0 = ExaModels.convert_array([0.0, 0.0, 1.0], backend)
        J = analytic_jac_dense(X0)
        H = analytic_hess_dense(X0, [0.0, 0.0, 1.0])

        Jv = similar(x0, m.meta.ncon)
        Jtw = similar(x0, m.meta.nvar)
        Hv = similar(x0, m.meta.nvar)
        ExaModels.jprod_nln!(m, x0, v, Jv)
        ExaModels.jtprod_nln!(m, x0, w, Jtw)
        ExaModels.hprod!(m, x0, y0, v, Hv)

        @test Array(Jv) ≈ J * [1.0, -1.0, 2.0, -0.5] atol = _atol(backend, 1.0e-10)
        @test Array(Jtw) ≈ J' * [1.0, 2.0, -1.0]       atol = _atol(backend, 1.0e-10)
        @test Array(Hv) ≈ H * [1.0, -1.0, 2.0, -0.5] atol = _atol(backend, 1.0e-10)
    end
end

function test_multiple_oracles(backend)
    return @testset "multiple oracles offsets" begin
        c = ExaCore(Float64; backend = backend)
        c, x = add_var(c, 4; lvar = -Inf, uvar = Inf)
        c, _ = add_obj(c, x[i]^2 for i in 1:4)

        # 2 SIMD constraints
        c, _ = add_con(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)
        c, _ = add_con(c, x[3] - x[4]; lcon = 0.0, ucon = 0.0)

        # Oracle 1 (ncon=1): x[1] - x[2] = 0
        o1 = VectorNonlinearOracle(
            nvar = 4, ncon = 1, nnzj = 2, nnzh = 0,
            jac_rows = [1, 1], jac_cols = [1, 2],
            lcon = [0.0], ucon = [0.0],
            adapt = Val(true),
            f! = (cv, xv) -> (cv[1] = xv[1] - xv[2]; nothing),
            jac! = (vv, xv) -> (vv[1] = 1.0; vv[2] = -1.0; nothing),
        )
        c = constraint(c, o1)

        # Oracle 2 (ncon=2): [x[1]*x[3], x[2]*x[4]]
        o2 = VectorNonlinearOracle(
            nvar = 4, ncon = 2, nnzj = 4, nnzh = 2,
            jac_rows = [1, 1, 2, 2], jac_cols = [1, 3, 2, 4],
            hess_rows = [1, 2], hess_cols = [3, 4],
            lcon = [0.0, 0.0], ucon = [Inf, Inf],
            adapt = Val(true),
            f! = (cv, xv) -> (cv[1] = xv[1] * xv[3]; cv[2] = xv[2] * xv[4]; nothing),
            jac! = (vv, xv) -> (vv[1] = xv[3]; vv[2] = xv[1]; vv[3] = xv[4]; vv[4] = xv[2]; nothing),
            hess! = (hv, xv, yv) -> (hv[1] = yv[1]; hv[2] = yv[2]; nothing),
        )
        c = constraint(c, o2)

        m = ExaModel(c)
        @test m isa ExaModelWithOracle
        @test m.meta.ncon == 5    # 2 SIMD + 1 + 2
        @test m.meta.nnzj == 2 + 2 + 2 + 4   # SIMD 2+2, o1 2, o2 4
        @test m.meta.nnzh == 4 + 0 + 2     # obj 4, o1 0, o2 2

        # Verify constraint values at x0=[0.5,0.5,0.6,0.4]
        x0 = ExaModels.convert_array(X0, backend)
        g = similar(x0, 5)
        ExaModels.cons_nln!(m, x0, g)
        g_cpu = Array(g)
        @test g_cpu[1] ≈ 1.0  atol = _atol(backend)   # SIMD: x1+x2
        @test g_cpu[2] ≈ 0.2  atol = _atol(backend)   # SIMD: x3-x4
        @test g_cpu[3] ≈ 0.0  atol = _atol(backend)   # o1: x1-x2
        @test g_cpu[4] ≈ X0[1] * X0[3]  atol = _atol(backend)  # o2[1]: x1*x3
        @test g_cpu[5] ≈ X0[2] * X0[4]  atol = _atol(backend)  # o2[2]: x2*x4

        # Verify oracle row offsets in Jacobian
        rows = similar(x0, Int, m.meta.nnzj)
        cols = similar(x0, Int, m.meta.nnzj)
        ExaModels.jac_structure!(m, rows, cols)
        rows_cpu = Array(rows)
        @test all(1 .<= rows_cpu .<= 5)
        # Oracle 2's first constraint block must have rows ≥ 4 (SIMD=2, o1 offset=2→row 3)
        @test 4 ∈ rows_cpu && 5 ∈ rows_cpu
    end
end

function test_gpu_oracle(backend)
    # Only meaningful for GPU backends; skip on CPU (nothing / CPU()).
    # We simulate a "GPU-native black-box" by setting adapt=Val(false) and providing
    # callbacks that operate directly on whatever array type the backend uses.
    # For CPU backends the callbacks just receive plain Arrays, which is fine.
    return @testset "adapt=Val(false) oracle" begin
        c = ExaCore(Float64; backend = backend)
        c, x = add_var(c, 4; lvar = -Inf, uvar = Inf, start = [0.5, 0.5, 0.6, 0.4])
        c, _ = add_obj(c, x[i]^2 for i in 1:4)
        c, _ = add_con(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)

        # Oracle with adapt=Val(false): callbacks receive the native array type of the
        # backend (CuArray on CUDA, Array on CPU) without any host copy.
        oracle_gpu = VectorNonlinearOracle(
            nvar = 4,
            ncon = 1,
            nnzj = 2,
            nnzh = 2,
            jac_rows = [1, 1],
            jac_cols = [3, 4],
            hess_rows = [3, 4],
            hess_cols = [3, 4],
            lcon = [-Inf],
            ucon = [Inf],
            adapt = Val(false),
            f! = (cv, xv) -> (cv .= xv[3:3] .^ 2 .+ xv[4:4] .^ 2; nothing),
            jac! = (vv, xv) -> (vv .= 2 .* xv[3:4]; nothing),
            hess! = (hv, xv, yv) -> (hv .= 2 .* yv; nothing),
        )
        c = constraint(c, oracle_gpu)
        m = ExaModel(c)
        @test m isa ExaModelWithOracle

        x0 = ExaModels.convert_array(X0, backend)
        y0 = ExaModels.convert_array([0.0, 1.0], backend)

        # cons
        g = similar(x0, m.meta.ncon)
        ExaModels.cons_nln!(m, x0, g)
        @test Array(g)[1] ≈ 1.0   atol = _atol(backend)
        @test Array(g)[2] ≈ 0.52  atol = _atol(backend)   # x3^2 + x4^2

        # jac_coord
        jac = similar(x0, m.meta.nnzj)
        ExaModels.jac_coord!(m, x0, jac)
        rows = similar(x0, Int, m.meta.nnzj)
        cols = similar(x0, Int, m.meta.nnzj)
        ExaModels.jac_structure!(m, rows, cols)
        J = zeros(2, 4)
        for k in eachindex(Array(rows))
            J[Array(rows)[k], Array(cols)[k]] += Array(jac)[k]
        end
        @test J[1, 1] ≈ 1.0  atol = _atol(backend)
        @test J[1, 2] ≈ 1.0  atol = _atol(backend)
        @test J[2, 3] ≈ 2 * X0[3]  atol = _atol(backend)
        @test J[2, 4] ≈ 2 * X0[4]  atol = _atol(backend)

        # hess_coord with λ₂=1
        hess = similar(x0, m.meta.nnzh)
        ExaModels.hess_coord!(m, x0, y0, hess)
        hrows = similar(x0, Int, m.meta.nnzh)
        hcols = similar(x0, Int, m.meta.nnzh)
        ExaModels.hess_structure!(m, hrows, hcols)
        H = zeros(4, 4)
        for k in eachindex(Array(hrows))
            r, c_ = Array(hrows)[k], Array(hcols)[k]
            H[r, c_] += Array(hess)[k]
            r != c_ && (H[c_, r] += Array(hess)[k])
        end
        # obj: 2I; oracle hess: λ_oracle*2 on (3,3) and (4,4);
        # y0=[0.0,1.0] so λ_oracle=y0[2]=1.0, contribution = 2.0
        @test H[1, 1] ≈ 2.0  atol = _atol(backend)
        @test H[2, 2] ≈ 2.0  atol = _atol(backend)
        @test H[3, 3] ≈ 4.0  atol = _atol(backend)   # 2 (obj) + 2*1.0 (oracle)
        @test H[4, 4] ≈ 4.0  atol = _atol(backend)   # 2 (obj) + 2*1.0 (oracle)
    end
end

# ── Matrix-free oracle tests ──────────────────────────────────────────────────
#
# Same test problem as above but using jvp!/vjp!/hvp! instead of jac!/hess!.

function _build_matfree_oracle_model(backend)
    c = ExaCore(Float64; backend = backend)
    c, x = add_var(c, 4; lvar = -Inf, uvar = Inf, start = [0.5, 0.5, 0.6, 0.4])

    # Objective: sum x[i]^2
    c, _ = add_obj(c, x[i]^2 for i in 1:4)

    # SIMD constraint: x[1] + x[2] = 1
    c, _ = add_con(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)

    # Oracle A (linear, matrix-free only): x[3] - x[4] = 0
    oracle_A = VectorNonlinearOracle(
        nvar = 4,
        ncon = 1,
        adapt = Val(true),
        f! = (cv, xv) -> (cv[1] = xv[3] - xv[4]; nothing),
        jvp! = (Jv, xv, v) -> (Jv[1] = v[3] - v[4]; nothing),
        vjp! = (Jtv, xv, w) -> (fill!(Jtv, 0.0); Jtv[3] = w[1]; Jtv[4] = -w[1]; nothing),
        lcon = [0.0],
        ucon = [0.0],
    )
    c = constraint(c, oracle_A)

    # Oracle B (nonlinear, matrix-free): x[3]^2 + x[4]^2 ≥ 0
    oracle_B = VectorNonlinearOracle(
        nvar = 4,
        ncon = 1,
        adapt = Val(true),
        f! = (cv, xv) -> (cv[1] = xv[3]^2 + xv[4]^2; nothing),
        jvp! = (Jv, xv, v) -> (Jv[1] = 2 * xv[3] * v[3] + 2 * xv[4] * v[4]; nothing),
        vjp! = (Jtv, xv, w) -> (fill!(Jtv, 0.0); Jtv[3] = 2 * xv[3] * w[1]; Jtv[4] = 2 * xv[4] * w[1]; nothing),
        hvp! = (Hv, xv, w, v) -> (fill!(Hv, 0.0); Hv[3] = 2 * w[1] * v[3]; Hv[4] = 2 * w[1] * v[4]; nothing),
        lcon = [-Inf],
        ucon = [Inf],
    )
    c = constraint(c, oracle_B)

    return ExaModel(c)
end

function test_matfree_meta(backend)
    return @testset "matfree meta" begin
        m = _build_matfree_oracle_model(backend)
        @test m isa ExaModelWithOracle
        @test m.meta.nvar == 4
        @test m.meta.ncon == 3
        # Matrix-free oracles declare dense structure:
        # oracle_A: nnzj = 1*4 = 4, nnzh = 0 (linear, no hvp!)
        # oracle_B: nnzj = 1*4 = 4, nnzh = 4*(4+1)/2 = 10
        @test m.meta.nnzj == 2 + 4 + 4    # SIMD + oA + oB
        @test m.meta.nnzh == 4 + 0 + 10   # obj + oA + oB
    end
end

function test_matfree_cons(backend)
    return @testset "matfree cons_nln!" begin
        m = _build_matfree_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)
        g = similar(x0, m.meta.ncon)
        ExaModels.cons_nln!(m, x0, g)
        @test Array(g) ≈ CONS0 atol = _atol(backend)
    end
end

function test_matfree_jac(backend)
    return @testset "matfree jac_coord! (reconstructed)" begin
        m = _build_matfree_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)

        rows = similar(x0, Int, m.meta.nnzj)
        cols = similar(x0, Int, m.meta.nnzj)
        ExaModels.jac_structure!(m, rows, cols)

        vals = similar(x0, m.meta.nnzj)
        ExaModels.jac_coord!(m, x0, vals)

        J_assembled = zeros(3, 4)
        rows_cpu = Array(rows); cols_cpu = Array(cols); vals_cpu = Array(vals)
        for k in eachindex(rows_cpu)
            J_assembled[rows_cpu[k], cols_cpu[k]] += vals_cpu[k]
        end
        @test J_assembled ≈ analytic_jac_dense(X0) atol = _atol(backend)
    end
end

function test_matfree_hess(backend)
    return @testset "matfree hess_coord! (reconstructed)" begin
        m = _build_matfree_oracle_model(backend)
        x0 = ExaModels.convert_array(X0, backend)
        y0 = ExaModels.convert_array([0.0, 0.0, 1.0], backend)

        rows = similar(x0, Int, m.meta.nnzh)
        cols = similar(x0, Int, m.meta.nnzh)
        ExaModels.hess_structure!(m, rows, cols)

        vals = similar(x0, m.meta.nnzh)
        ExaModels.hess_coord!(m, x0, y0, vals)

        H_assembled = zeros(4, 4)
        rows_cpu = Array(rows); cols_cpu = Array(cols); vals_cpu = Array(vals)
        for k in eachindex(rows_cpu)
            H_assembled[rows_cpu[k], cols_cpu[k]] += vals_cpu[k]
            if rows_cpu[k] != cols_cpu[k]
                H_assembled[cols_cpu[k], rows_cpu[k]] += vals_cpu[k]
            end
        end
        @test H_assembled ≈ analytic_hess_dense(X0, [0.0, 0.0, 1.0]) atol = _atol(backend)
    end
end

function test_matfree_jprod_jtprod(backend)
    return @testset "matfree jprod_nln! and jtprod_nln!" begin
        c = ExaCore(Float64; backend = backend)
        c, x = add_var(c, 4; lvar = -Inf, uvar = Inf, start = [0.5, 0.5, 0.6, 0.4])
        c, _ = add_obj(c, x[i]^2 for i in 1:4)
        c, _ = add_con(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)
        oracle_A = VectorNonlinearOracle(
            nvar = 4, ncon = 1,
            adapt = Val(true),
            f! = (cv, xv) -> (cv[1] = xv[3] - xv[4]; nothing),
            jvp! = (Jv, xv, v) -> (Jv[1] = v[3] - v[4]; nothing),
            vjp! = (Jtv, xv, w) -> (fill!(Jtv, 0.0); Jtv[3] = w[1]; Jtv[4] = -w[1]; nothing),
            lcon = [0.0], ucon = [0.0],
        )
        c = constraint(c, oracle_A)
        oracle_B = VectorNonlinearOracle(
            nvar = 4, ncon = 1,
            adapt = Val(true),
            f! = (cv, xv) -> (cv[1] = xv[3]^2 + xv[4]^2; nothing),
            jvp! = (Jv, xv, v) -> (Jv[1] = 2 * xv[3] * v[3] + 2 * xv[4] * v[4]; nothing),
            vjp! = (Jtv, xv, w) -> (fill!(Jtv, 0.0); Jtv[3] = 2 * xv[3] * w[1]; Jtv[4] = 2 * xv[4] * w[1]; nothing),
            hvp! = (Hv, xv, w, v) -> (fill!(Hv, 0.0); Hv[3] = 2 * w[1] * v[3]; Hv[4] = 2 * w[1] * v[4]; nothing),
            lcon = [-Inf], ucon = [Inf],
        )
        c = constraint(c, oracle_B)
        m = ExaModel(c; prod = true)
        x0 = ExaModels.convert_array(X0, backend)
        J = analytic_jac_dense(X0)

        v = ExaModels.convert_array([1.0, -1.0, 2.0, -0.5], backend)
        w = ExaModels.convert_array([1.0, 2.0, -1.0], backend)

        Jv = similar(x0, m.meta.ncon)
        Jtw = similar(x0, m.meta.nvar)
        ExaModels.jprod_nln!(m, x0, v, Jv)
        ExaModels.jtprod_nln!(m, x0, w, Jtw)

        @test Array(Jv) ≈ J * [1.0, -1.0, 2.0, -0.5] atol = _atol(backend, 1.0e-10)
        @test Array(Jtw) ≈ J' * [1.0, 2.0, -1.0]      atol = _atol(backend, 1.0e-10)
    end
end

function test_matfree_hprod(backend)
    return @testset "matfree hprod!" begin
        c = ExaCore(Float64; backend = backend)
        c, x = add_var(c, 4; lvar = -Inf, uvar = Inf, start = [0.5, 0.5, 0.6, 0.4])
        c, _ = add_obj(c, x[i]^2 for i in 1:4)
        c, _ = add_con(c, x[1] + x[2]; lcon = 1.0, ucon = 1.0)
        oracle_A = VectorNonlinearOracle(
            nvar = 4, ncon = 1,
            adapt = Val(true),
            f! = (cv, xv) -> (cv[1] = xv[3] - xv[4]; nothing),
            jvp! = (Jv, xv, v) -> (Jv[1] = v[3] - v[4]; nothing),
            vjp! = (Jtv, xv, w) -> (fill!(Jtv, 0.0); Jtv[3] = w[1]; Jtv[4] = -w[1]; nothing),
            lcon = [0.0], ucon = [0.0],
        )
        c = constraint(c, oracle_A)
        oracle_B = VectorNonlinearOracle(
            nvar = 4, ncon = 1,
            adapt = Val(true),
            f! = (cv, xv) -> (cv[1] = xv[3]^2 + xv[4]^2; nothing),
            jvp! = (Jv, xv, v) -> (Jv[1] = 2 * xv[3] * v[3] + 2 * xv[4] * v[4]; nothing),
            vjp! = (Jtv, xv, w) -> (fill!(Jtv, 0.0); Jtv[3] = 2 * xv[3] * w[1]; Jtv[4] = 2 * xv[4] * w[1]; nothing),
            hvp! = (Hv, xv, w, v) -> (fill!(Hv, 0.0); Hv[3] = 2 * w[1] * v[3]; Hv[4] = 2 * w[1] * v[4]; nothing),
            lcon = [-Inf], ucon = [Inf],
        )
        c = constraint(c, oracle_B)
        m = ExaModel(c; prod = true)
        x0 = ExaModels.convert_array(X0, backend)
        y0 = ExaModels.convert_array([0.0, 0.0, 1.0], backend)
        v = ExaModels.convert_array([1.0, -1.0, 2.0, -0.5], backend)

        Hv = similar(x0, m.meta.nvar)
        ExaModels.hprod!(m, x0, y0, v, Hv)

        H = analytic_hess_dense(X0, [0.0, 0.0, 1.0])
        @test Array(Hv) ≈ H * [1.0, -1.0, 2.0, -0.5] atol = _atol(backend, 1.0e-10)
    end
end

function test_matfree_backward_compat()
    return @testset "backward compatibility (jac! only)" begin
        # Existing code that only passes jac!/hess! should still work.
        oracle = VectorNonlinearOracle(
            nvar = 2, ncon = 1, nnzj = 2, nnzh = 0,
            jac_rows = [1, 1], jac_cols = [1, 2],
            lcon = [0.0], ucon = [0.0],
            f! = (cv, xv) -> (cv[1] = xv[1] + xv[2]; nothing),
            jac! = (vv, xv) -> (vv[1] = 1.0; vv[2] = 1.0; nothing),
        )
        @test oracle.jvp! === nothing
        @test oracle.vjp! === nothing
        @test oracle.hvp! === nothing
        @test !has_matfree_jac(oracle)
        @test !has_matfree_hess(oracle)
    end
end

# ── Minimal happy-path tests for the rest of the oracle API surface ────────────
# CPU-only (backend = nothing). One assertion per API, just enough to catch
# regressions in code paths the backend-looped tests above don't exercise.

function test_scalar_oracle()
    return @testset "ScalarNonlinearOracle" begin
        c = ExaCore(Float64; concrete = Val(true))
        c, x = add_var(c, 2; start = [1.0, 2.0])
        oracle = ScalarNonlinearOracle(
            nvar = 2,
            f = xv -> xv[1]^2 + xv[2]^2,
            grad! = (g, xv) -> (g[1] = 2 * xv[1]; g[2] = 2 * xv[2]; nothing),
            adapt = Val(true),
        )
        c = objective(c, oracle)
        m = ExaModel(c)
        @test m isa ExaModelWithOracle
        @test NLPModels.obj(m, [3.0, 4.0]) ≈ 25.0
        @test Array(NLPModels.grad(m, [3.0, 4.0])) ≈ [6.0, 8.0]

        # Lock in zero-allocation steady state for grad! — Issue 6 regression.
        # Was ~oracle.nvar * sizeof(Float64) bytes per call before preallocation.
        xv = [3.0, 4.0]
        g = zeros(2)
        for _ in 1:3
            NLPModels.grad!(m, xv, g)
        end   # warm up JIT
        @test (@allocated NLPModels.grad!(m, xv, g)) == 0
    end
end

function test_add_eval()
    return @testset "add_eval (function form)" begin
        c = ExaCore(Float64; concrete = Val(true))
        c, x = add_var(c, 2)
        c, con = add_con(c, 0.0 * x[i] for i in 1:2; lcon = 0.0, ucon = 0.0)
        c, _ = add_eval(
            c, (con,), (x,),
            (res, xv) -> (res .= xv .^ 2; nothing);
            jac! = (vals, xv) -> (vals .= 2 .* xv; nothing),
            nnzj = 2,
            jac_structure! = (rows, cols) -> (append!(rows, 1:2); append!(cols, 1:2)),
            adapt = Val(true),
        )
        m = ExaModel(c)
        g = zeros(2)
        NLPModels.cons_nln!(m, [3.0, 4.0], g)
        @test g ≈ [9.0, 16.0]                    # SIMD writes 0, evaluator adds x.^2
    end
end

function test_add_con_macro_oracle()
    return @testset "@add_con! oracle form" begin
        c = ExaCore(Float64; concrete = Val(true))
        @add_var(c, x, 2)
        con = @add_con(c, 0.0 * x[i] for i in 1:2; lcon = 0.0, ucon = 0.0)
        @add_con!(
            c, (con,), (x,),
            (res, xv) -> (res .= xv .^ 2; nothing);
            jac! = (vals, xv) -> (vals .= 2 .* xv; nothing),
            nnzj = 2,
            jac_structure! = (rows, cols) -> (append!(rows, 1:2); append!(cols, 1:2)),
            adapt = Val(true),
        )
        m = ExaModel(c)
        g = zeros(2)
        NLPModels.cons_nln!(m, [3.0, 4.0], g)
        @test g ≈ [9.0, 16.0]
    end
end

function test_embed_oracle()
    return @testset "embed_oracle" begin
        c = ExaCore(Float64; concrete = Val(true))
        c, x = add_var(c, 2)
        c, z, _ = embed_oracle(
            c, x, 2;
            f! = (y, xv) -> (y .= xv .^ 2; nothing),
            jvp! = (Jv, xv, v) -> (Jv .= 2 .* xv .* v; nothing),
            vjp! = (Jtv, xv, w) -> (Jtv .= 2 .* xv .* w; nothing),
            adapt = Val(true),
        )
        m = ExaModel(c)
        @test m isa ExaModelWithOracle
        @test m.meta.nvar == 4 && m.meta.ncon == 2          # 2 x's + 2 z's, 2 oracle rows
        g = zeros(2)
        NLPModels.cons_nln!(m, [1.0, 2.0, 3.0, 4.0], g)     # x = [1,2]; z = [3,4]
        @test g ≈ [2.0, 0.0]                                 # z .- x.^2 = [3-1, 4-4]
    end
end

# Scalar + vector oracles together: hess_structure!/hess_coord!/hprod! must
# fill every block (the scalar-oracle block used to be allocated but never
# written), on the host and on every device backend.
function test_scalar_oracle_hessian(backend)
    # Float64 only: oracle models cannot yet be built with T = Float32 on the
    # host (the oracle caches and work buffers are allocated as Float64, and the
    # ExaModelWithOracle fields require VT). The device legs still exercise the
    # obj_weight -> T conversion in the KA overrides, which Metal requires.
    return @testset "scalar-oracle Hessian assembly" begin
        c = ExaCore(Float64; backend = backend)
        c, x = add_var(c, 2; start = [1.0, 2.0])
        sor = ScalarNonlinearOracle(;
            nvar = 2,
            f = x_ -> x_[1]^2 * x_[2],
            grad! = (g, x_) -> (g[1] = 2x_[1] * x_[2]; g[2] = x_[1]^2),
            hvp! = (Hv, x_, v) -> (
                Hv[1] = 2x_[2] * v[1] + 2x_[1] * v[2];
                Hv[2] = 2x_[1] * v[1]
            ),
            hess_rows = [1, 2, 2], hess_cols = [1, 1, 2],
            adapt = Val(true),
        )
        c = objective(c, sor)
        vor = VectorNonlinearOracle(;
            nvar = 2, ncon = 1,
            jac_rows = [1, 1], jac_cols = [1, 2],
            hess_rows = [2], hess_cols = [1],
            lcon = [0.0], ucon = [0.0],
            f! = (cv, x_) -> (cv[1] = x_[1] * x_[2] - 1.0),
            jac! = (vals, x_) -> (vals[1] = x_[2]; vals[2] = x_[1]),
            hess! = (vals, x_, y_) -> (vals[1] = y_[1]),
            adapt = Val(true),
        )
        c = constraint(c, vor)
        m = ExaModel(c; prod = true)
        rows = similar(m.meta.x0, Int, m.meta.nnzh)
        colsv = similar(rows)
        NLPModels.hess_structure!(m, rows, colsv)
        @test sort(collect(zip(Array(rows), Array(colsv)))) == [(1, 1), (2, 1), (2, 1), (2, 2)]
        xv = similar(m.meta.x0, 2)
        copyto!(xv, [1.0, 2.0])
        y = similar(m.meta.x0, 1)
        fill!(y, 1.0)
        vals = similar(m.meta.x0, m.meta.nnzh)
        NLPModels.hess_coord!(m, xv, y, vals; obj_weight = 0.5)
        H = zeros(2, 2)
        for (i, j, vv) in zip(Array(rows), Array(colsv), Array(vals))
            H[i, j] += vv
        end
        # 0.5 * obj hessian [4 2; 2 0] + y1 * con hessian [0 1; 1 0] (lower)
        @test H[1, 1] == 2.0 && H[2, 1] == 2.0 && H[2, 2] == 0.0
        v = similar(xv)
        copyto!(v, [1.0, 0.0])
        Hv = similar(xv)
        NLPModels.hprod!(m, xv, y, v, Hv; obj_weight = 0.5)
        @test Array(Hv) == [2.0, 2.0]   # 0.5*[4,2] + 1.0*[0,1]
        NLPModels.hprod!(m, xv, v, Hv; obj_weight = 0.5)   # objective-only
        @test Array(Hv) == [2.0, 1.0]   # 0.5*[4,2]
    end
end

function runtests()
    return @testset "OracleTest" begin
        @testset "VectorNonlinearOracle type checks" begin
            test_model_type()
        end

        for backend in BACKENDS
            @testset "backend=$backend" begin
                test_meta(backend)
                test_obj_and_grad(backend)
                test_cons(backend)
                test_jac(backend)
                test_hess(backend)
                test_jprod_jtprod(backend)
                test_hprod(backend)
                test_scalar_oracle_hessian(backend)
                test_val_false_jprod_jtprod_hprod(backend)
                test_multiple_oracles(backend)
                test_gpu_oracle(backend)
            end
        end

        @testset "Matrix-free oracle" begin
            test_matfree_backward_compat()
            for backend in BACKENDS
                @testset "matfree backend=$backend" begin
                    test_matfree_meta(backend)
                    test_matfree_cons(backend)
                    test_matfree_jac(backend)
                    test_matfree_hess(backend)
                    test_matfree_jprod_jtprod(backend)
                    test_matfree_hprod(backend)
                end
            end
        end

        @testset "Other oracle APIs" begin
            test_scalar_oracle()
            test_add_eval()
            test_add_con_macro_oracle()
            test_embed_oracle()
        end
    end
end

end # module OracleTest
