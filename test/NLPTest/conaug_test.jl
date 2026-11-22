using ExaModels
using Test
import NLPModels

# Tests for:
#   - add_con!(core, c1[i] += expr for ...) — single-generator form
#   - add_con(core, N, M) / add_con(core, r1, r2) — multi-dimensional empty constraints
#   - @add_con! two-argument form

function test_conaug_1d(backend)
    @testset "add_con! single-generator matches old syntax (1D)" begin
        N = 9

        # Old syntax: add_con!(c, g, i => expr for ...)
        c1 = ExaCore(; backend, concrete = Val(true))
        c1, x1 = add_var(c1, N + 1)
        c1, g1 = add_con(c1, N; lcon = -1.0, ucon = 1.0)
        c1, _ = add_con!(c1, g1, i => x1[i] + x1[i + 1] for i in 1:N)

        # New syntax: add_con!(c, g[i] += expr for ...)
        c2 = ExaCore(; backend, concrete = Val(true))
        c2, x2 = add_var(c2, N + 1)
        c2, g2 = add_con(c2, N; lcon = -1.0, ucon = 1.0)
        c2, _ = add_con!(c2, g2[i] += x2[i] + x2[i + 1] for i in 1:N)

        m1 = ExaModel(c1)
        m2 = ExaModel(c2)

        @test m1.meta.nnzj == m2.meta.nnzj
        @test m1.meta.nnzh == m2.meta.nnzh
        x0 = ExaModels.convert_array(rand(N + 1), backend)
        @test NLPModels.cons(m1, x0) ≈ NLPModels.cons(m2, x0)
        @test NLPModels.grad(m1, x0) ≈ NLPModels.grad(m2, x0)
    end

    @testset "@add_con! two-argument form (1D)" begin
        N = 9
        c = ExaCore(; backend, concrete = Val(true))
        @add_var(c, x, N + 1)
        @add_con(c, g, Constant(0) for _ in 1:N; lcon = -1.0, ucon = 1.0)
        @add_con!(c, g[i] += x[i] + x[i + 1] for i in 1:N)

        m = ExaModel(c)
        x0 = ExaModels.convert_array(ones(N + 1), backend)
        g_vals = Array(NLPModels.cons(m, x0))
        @test all(≈(2.0), g_vals)
    end

    return @testset "add_con! with -= syntax (1D)" begin
        N = 9
        # g[i] -= expr should be equivalent to g[i] += (-expr)
        c1 = ExaCore(; backend, concrete = Val(true))
        c1, x1 = add_var(c1, N + 1)
        c1, g1 = add_con(c1, N; lcon = -1.0, ucon = 1.0)
        c1, _ = add_con!(c1, g1[i] -= x1[i] + x1[i + 1] for i in 1:N)

        # Reference: same result via +=(-expr)
        c2 = ExaCore(; backend, concrete = Val(true))
        c2, x2 = add_var(c2, N + 1)
        c2, g2 = add_con(c2, N; lcon = -1.0, ucon = 1.0)
        c2, _ = add_con!(c2, g2[i] += -(x2[i] + x2[i + 1]) for i in 1:N)

        m1 = ExaModel(c1)
        m2 = ExaModel(c2)
        @test m1.meta.nnzj == m2.meta.nnzj
        @test m1.meta.nnzh == m2.meta.nnzh
        x0 = ExaModels.convert_array(rand(N + 1), backend)
        @test NLPModels.cons(m1, x0) ≈ NLPModels.cons(m2, x0)
        @test NLPModels.grad(m1, x0) ≈ NLPModels.grad(m2, x0)
    end
end

function test_conaug_2d(backend)
    @testset "add_con(core, dims...) — 2D integer dims" begin
        N, M = 3, 4
        c = ExaCore(; backend, concrete = Val(true))
        c, x = add_var(c, N, M)
        c, g = add_con(c, N, M; lcon = -Inf, ucon = 0.0)
        @test Base.size(g.itr) == (N, M)
        itr = [(i, j) for i in 1:(N - 1), j in 1:M][:]
        c, _ = add_con!(c, g[i, j] += x[i, j] - x[i + 1, j] for (i, j) in itr)
        m = ExaModel(c)
        @test m.meta.ncon == N * M
        @test m.meta.nnzj == length(itr) * 2

        # Verify actual constraint values: g[i,j] = x[i,j] - x[i+1,j] for i<N, else 0
        x0 = ExaModels.convert_array(collect(Float64, 1:(N * M)), backend)
        g_vals = Array(NLPModels.cons(m, x0))
        # constraints stored in column-major order: linear index (i,j) → (j-1)*N + i
        for j in 1:M, i in 1:N
            k = (j - 1) * N + i
            expected = i < N ? Float64((j - 1) * N + i) - Float64((j - 1) * N + i + 1) : 0.0
            @test g_vals[k] ≈ expected
        end
    end

    @testset "add_con(core, dims...) — range-based 2D dims (non-unit start)" begin
        N, M = 3, 4
        r1, r2 = 1:N, 2:(M + 1)   # r2 starts at 2: j ∈ 2..5 in range space, 1..4 block-local
        # x has M+1 columns so x[i,j] is valid for j up to M+1
        c = ExaCore(; backend, concrete = Val(true))
        c, x = add_var(c, N, M + 1)
        c, g = add_con(c, r1, r2; lcon = -Inf, ucon = 0.0)
        @test Base.size(g.itr) == (length(r1), length(r2))
        @test length(g.itr) == N * M

        # For range-based constraints, augmentation indices must use range values:
        # i ∈ r1 = 1:N, j ∈ r2 = 2:M+1.  _con_adjust then maps j → j-(start-1).
        itr = [(i, j) for i in first(r1):(last(r1) - 1), j in r2][:]
        c, _ = add_con!(c, g[i, j] += x[i, j] - x[i + 1, j] for (i, j) in itr)
        m = ExaModel(c)
        @test m.meta.ncon == N * M
        @test m.meta.nnzj == length(itr) * 2

        # x0[k] = k, so x[i,j] = (j-1)*N+i → x[i,j]-x[i+1,j] = -1 for all valid (i,j)
        x0 = ExaModels.convert_array(collect(Float64, 1:(N * (M + 1))), backend)
        g_vals = Array(NLPModels.cons(m, x0))
        for j_col in 1:M, i in 1:N
            k = (j_col - 1) * N + i
            expected = i < N ? -1.0 : 0.0
            @test g_vals[k] ≈ expected
        end
    end

    @testset "add_con(core, dims...) — 3D integer dims with value check" begin
        N, M, K = 2, 3, 4
        c = ExaCore(; backend, concrete = Val(true))
        c, x = add_var(c, N * M * K)
        c, g = add_con(c, N, M, K; lcon = 0.0, ucon = 0.0)
        @test Base.size(g.itr) == (N, M, K)
        @test length(g.itr) == N * M * K

        # Augment with a non-trivial expression: g[i,j,k] += x[linidx(i,j,k)] * 2 for all (i,j,k)
        itr3 = [(i, j, k) for i in 1:N, j in 1:M, k in 1:K][:]
        c, _ = add_con!(c, g[i, j, kk] += x[(kk - 1) * N * M + (j - 1) * N + i] * 2 for (i, j, kk) in itr3)
        m = ExaModel(c)
        @test m.meta.ncon == N * M * K
        @test m.meta.nnzj == length(itr3)

        x0 = ExaModels.convert_array(ones(N * M * K), backend)
        g_vals = Array(NLPModels.cons(m, x0))
        @test all(≈(2.0), g_vals)
    end

    @testset "range-based 2D matches integer 2D (same result)" begin
        N, M = 3, 4
        itr = [(i, j) for i in 1:(N - 1), j in 1:M][:]

        # Integer dims
        c1 = ExaCore(; backend, concrete = Val(true))
        c1, x1 = add_var(c1, N, M)
        c1, g1 = add_con(c1, N, M; lcon = -1.0, ucon = 1.0)
        c1, _ = add_con!(c1, g1[i, j] += x1[i, j] - x1[i + 1, j] for (i, j) in itr)

        # Range dims (1:N, 1:M) — same as integer N, M
        c2 = ExaCore(; backend, concrete = Val(true))
        c2, x2 = add_var(c2, N, M)
        c2, g2 = add_con(c2, 1:N, 1:M; lcon = -1.0, ucon = 1.0)
        c2, _ = add_con!(c2, g2[i, j] += x2[i, j] - x2[i + 1, j] for (i, j) in itr)

        m1 = ExaModel(c1)
        m2 = ExaModel(c2)
        @test m1.meta.nnzj == m2.meta.nnzj
        @test m1.meta.nnzh == m2.meta.nnzh

        x0 = ExaModels.convert_array(rand(N * M), backend)
        @test NLPModels.cons(m1, x0) ≈ NLPModels.cons(m2, x0)

        # Also verify against CPU reference (only when backend isn't already CPU)
        c_ref = ExaCore(; concrete = Val(true))
        c_ref, x_ref = add_var(c_ref, N, M)
        c_ref, g_ref = add_con(c_ref, N, M; lcon = -1.0, ucon = 1.0)
        c_ref, _ = add_con!(c_ref, g_ref[i, j] += x_ref[i, j] - x_ref[i + 1, j] for (i, j) in itr)
        m_ref = ExaModel(c_ref)
        x0_cpu = collect(Float64, Array(x0))
        @test Array(NLPModels.cons(m1, x0)) ≈ NLPModels.cons(m_ref, x0_cpu)
    end

    return @testset "multiple augmentations on same 2D constraint" begin
        N, M = 4, 5
        # Two separate augmentations: forward and backward differences
        itr_fwd = [(i, j) for i in 1:(N - 1), j in 1:M][:]
        itr_bwd = [(i, j) for i in 2:N, j in 1:M][:]

        c = ExaCore(; backend, concrete = Val(true))
        c, x = add_var(c, N, M)
        c, g = add_con(c, N, M; lcon = -Inf, ucon = Inf)
        # Add forward diff: g[i,j] += x[i,j] - x[i+1,j]
        c, _ = add_con!(c, g[i, j] += x[i, j] - x[i + 1, j] for (i, j) in itr_fwd)
        # Add backward diff: g[i,j] += x[i-1,j] - x[i,j]  (cancels forward for interior rows)
        c, _ = add_con!(c, g[i, j] += x[i - 1, j] - x[i, j] for (i, j) in itr_bwd)
        m = ExaModel(c)
        @test m.meta.ncon == N * M
        @test m.meta.nnzj == (length(itr_fwd) + length(itr_bwd)) * 2

        # Interior rows 2..N-1 get both augmentations: (x[i,j]-x[i+1,j]) + (x[i-1,j]-x[i,j])
        # = x[i-1,j] - x[i+1,j]
        # Row 1: only forward diff:  x[1,j] - x[2,j]
        # Row N: only backward diff: x[N-1,j] - x[N,j]
        x0 = ExaModels.convert_array(collect(Float64, 1:(N * M)), backend)
        g_vals = Array(NLPModels.cons(m, x0))
        for j in 1:M, i in 1:N
            k = (j - 1) * N + i
            xval = (j - 1) * N  # base for column j
            if i == 1
                expected = Float64(xval + 1) - Float64(xval + 2)
            elseif i == N
                expected = Float64(xval + N - 1) - Float64(xval + N)
            else
                expected = Float64(xval + i - 1) - Float64(xval + i + 1)
            end
            @test g_vals[k] ≈ expected
        end
    end
end

function test_conaug(backend)
    @testset "1D constraint augmentation" begin
        test_conaug_1d(backend)
    end
    return @testset "Multi-dimensional constraint" begin
        test_conaug_2d(backend)
    end
end
