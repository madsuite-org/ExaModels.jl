using ExaModels
using Test
import NLPModels

# Tests for Constant{T}, add_var(gen), and named variable/constraint access

function test_const(backend)
    return @testset "Constant basic usage" begin
        c = ExaCore(; backend, concrete = Val(true))
        N = 5
        @add_var(c, x, N; start = 1.0)

        # Const(N) keeps type stable — N is not captured in the type parameter
        @add_obj(c, Constant(N) * x[i]^2 for i in 1:N)

        m = ExaModel(c)
        @test m.meta.nvar == N
        @test m.meta.nnzo > 0

        x0 = ExaModels.convert_array(ones(N), backend)
        @test NLPModels.obj(m, x0) ≈ N * N  # sum of N*x[i]^2 = N*N*1
    end
end

function test_add_var_gen(backend)
    return @testset "add_var with generator (equality constraints)" begin
        N = 4
        targets = [2.0, 3.0, 4.0, 5.0]
        pars = collect(enumerate(targets))

        c = ExaCore(; backend, concrete = Val(true))
        @add_var(c, y, N; start = 1.0)

        # add_var(gen) creates variables x[i] with x[i] == targets[i]
        c, x = add_var(c, i - 0.0 for i in targets; start = 2.0)

        m = ExaModel(c)

        # N original vars + N new vars; N equality constraints from generator
        @test m.meta.nvar == 2 * N
        @test m.meta.ncon == N
        # equality constraints: lcon == ucon == 0
        @test all(iszero, m.meta.lcon)
        @test all(iszero, m.meta.ucon)
    end
end

function test_named_var_access(backend)
    @testset "Named variable access via core.name and model.name" begin
        c = ExaCore(; backend, concrete = Val(true))
        N = 5
        @add_var(c, x, N; start = 0.5)
        @add_obj(c, (x[i] - 1)^2 for i in 1:N)

        m = ExaModel(c)

        # Access Variable via core.x and model.x
        @test c.x isa ExaModels.Variable
        @test m.x isa ExaModels.Variable
        @test c.x === m.x
    end

    @testset "Named constraint access via model.name" begin
        c = ExaCore(; backend, concrete = Val(true))
        @add_var(c, x, 4; start = 0.0)
        @add_con(c, g, x[i] + x[i + 1] for i in 1:3)

        m = ExaModel(c)

        @test c.g isa ExaModels.Constraint
        @test m.g isa ExaModels.Constraint
    end

    return @testset "Named objective access via model.name" begin
        c = ExaCore(; backend, concrete = Val(true))
        @add_var(c, x, 3; start = 1.0)
        @add_obj(c, f, x[i]^2 for i in 1:3)

        m = ExaModel(c)

        @test c.f isa ExaModels.Objective
        @test m.f isa ExaModels.Objective
    end
end

function test_add_par_dims(backend)
    @testset "add_par(core, array) — backward compat" begin
        c = ExaCore(; backend, concrete = Val(true))
        c, θ = add_par(c, ones(5))
        @test θ.length == 5
        @test θ.size == (5,)
    end

    @testset "add_par(core, n; value)" begin
        c = ExaCore(; backend, concrete = Val(true))
        c, θ = add_par(c, 5; value = 1.0)
        @test θ.length == 5
    end

    @testset "add_par(core, range; value) — non-unit start" begin
        c = ExaCore(; backend, concrete = Val(true))
        c, x = add_var(c, 5)
        c, θ = add_par(c, 2:4; value = [10.0, 20.0, 30.0])
        @test θ.length == 3
        @test θ.size == (2:4,)

        c, g = add_con(c, θ[j] * x[1] for j in 2:4; lcon = 0.0, ucon = 0.0)
        m = ExaModel(c)
        x0 = ExaModels.convert_array(ones(5), backend)
        g_vals = Array(NLPModels.cons(m, x0))
        @test g_vals ≈ [10.0, 20.0, 30.0]
    end

    @testset "add_par(core, n, range; value) — multi-dim" begin
        c = ExaCore(; backend, concrete = Val(true))
        c, x = add_var(c, 12)
        c, θ = add_par(c, 3, 2:5; value = collect(Float64, 1:12))
        @test θ.length == 12
        @test θ.size == (3, 2:5)

        c, g = add_con(c, θ[i, j] * x[1] for (i, j) in [(1, 2), (2, 3), (3, 4)]; lcon = 0.0, ucon = 0.0)
        m = ExaModel(c)
        x0 = ExaModels.convert_array(ones(12), backend)
        g_vals = Array(NLPModels.cons(m, x0))
        @test g_vals ≈ [1.0, 5.0, 9.0]
    end

    @testset "data-axis block sizes" begin
        # A constraint may iterate a DATA collection (`for (i, tau) in nodes`),
        # which then forms one axis of its block's size field. `size`/`total`
        # must treat that axis like a range's — by its length — or every
        # accessor downstream (multipliers, a compiled library's layout
        # query) throws a MethodError. Six of COPS's seventeen models carry
        # such an axis; catmix's collocation nodes found this.
        c = ExaCore(; backend, concrete = Val(true))
        c, x = add_var(c, 4; start = 1.0)
        nodes = [(1, 0.5), (2, 1.5), (3, 2.5)]
        c, s = add_con(
            c, x[t[1]] - t[2] for t in nodes, j in 1:2;
            lcon = -10.0, ucon = 10.0, name = Val(:dax)
        )
        m = ExaModel(c)
        b = get_cons(m, :dax)
        @test ExaModels.size(b.size) == (3, 2)
        @test ExaModels.total(b.size) == 6
        # A ZIP axis measures the same way — steering's boundary conditions,
        # `for (i, v) in zip(2:4, angles)`, were the second iterator type to
        # arrive as a swallowed status 2; the rule is now general (an axis is
        # measured by its length) rather than per-type.
        c2 = ExaCore(; backend, concrete = Val(true))
        c2, y = add_var(c2, 6; start = 0.0)
        c2, z = add_con(
            c2, y[i] - v for (i, v) in zip(2:4, [5.0, 45.0, 0.0]);
            lcon = 0.0, ucon = 0.0, name = Val(:zax)
        )
        m2 = ExaModel(c2)
        b2 = get_cons(m2, :zax)
        @test ExaModels.size(b2.size) == (3,)
        @test ExaModels.total(b2.size) == 3
    end

    return @testset "set_value! with range-sized parameter" begin
        c = ExaCore(; backend, concrete = Val(true))
        c, θ = add_par(c, 2:4; value = ones(3))
        set_value!(c, θ, [5.0, 6.0, 7.0])
        c, x = add_var(c, 1)
        c, g = add_con(c, θ[j] * x[1] for j in 2:4; lcon = 0.0, ucon = 0.0)
        m = ExaModel(c)
        x0 = ExaModels.convert_array(ones(1), backend)
        g_vals = Array(NLPModels.cons(m, x0))
        @test g_vals ≈ [5.0, 6.0, 7.0]
    end
end

function test_nonunit_expr(backend)
    @testset "Expression with non-unit-start range" begin
        c = ExaCore(; backend, concrete = Val(true))
        c, x = add_var(c, 5)
        c, s = add_expr(c, x[i]^2 for i in 2:4)
        @test s.size == (2:4,)
        c, g = add_con(c, s[j] for j in 2:4; lcon = 0.0, ucon = 0.0)
        m = ExaModel(c)
        x0 = ExaModels.convert_array(collect(Float64, 1:5), backend)
        g_vals = Array(NLPModels.cons(m, x0))
        @test g_vals ≈ [4.0, 9.0, 16.0]
    end

    return @testset "Expression with ProductIterator non-unit ranges" begin
        c = ExaCore(; backend, concrete = Val(true))
        c, x = add_var(c, 3, 4)
        c, s = add_expr(c, x[i, j]^2 for (i, j) in Iterators.product(1:3, 2:4))
        @test s.size == (1:3, 2:4)
        c, g = add_con(c, s[i, j] for (i, j) in Iterators.product(1:3, 2:4); lcon = 0.0, ucon = 0.0)
        m = ExaModel(c)
        x0 = ExaModels.convert_array(collect(Float64, 1:12), backend)
        g_vals = Array(NLPModels.cons(m, x0))
        # x[i,j] = (j-1)*3 + i, s[i,j] = x[i,j]^2
        expected = [Float64((j - 1) * 3 + i)^2 for j in 2:4, i in 1:3][:]  # product order: i fast, j slow
        # ProductIterator iterates i first, j second → column-major
        expected = [Float64((j - 1) * 3 + i)^2 for i in 1:3 for j in 2:4]
        @test length(g_vals) == 9
    end
end

function test_colon_index(backend)
    return @testset "Colon expands to the tuple of entries along an axis" begin
        N, K = 4, 3
        itr = [(i, k, exp10(i / N)) for i in 1:N, k in 1:K]
        f(x...) = -x[1] * 2 + x[2] * x[3] - x[4] + x[5]
        g(x, θ) = x[1] * θ[1] - x[5] * θ[2]
        function build(con)
            c = ExaCore(; backend, concrete = Val(true))
            c, z = add_var(c, 1:5, 1:N, 0:K; start = 1.0)
            c, θ = add_par(c, [2.0, 3.0])
            c, _ = con(c, z, θ)
            ExaModel(c)
        end
        m = build(
            (c, z, θ) ->
            add_con(c, -hi * (f(z[:, i, k]...) + g(z[:, i, k], θ[:])) for (i, k, hi) in itr),
        )
        mref = build(
            (c, z, θ) -> add_con(
                c,
                -hi * (
                    f(z[1, i, k], z[2, i, k], z[3, i, k], z[4, i, k], z[5, i, k]) +
                        g((z[1, i, k], z[2, i, k], z[3, i, k], z[4, i, k], z[5, i, k]), (θ[1], θ[2]))
                ) for (i, k, hi) in itr
            ),
        )
        x0 = ExaModels.convert_array([sin(i) for i in 1:m.meta.nvar], backend)
        @test Array(NLPModels.cons(m, x0)) ≈ Array(NLPModels.cons(mref, x0))

        # column-major order, non-unit starts, `x[:]` as in Base, Expression, wrong rank
        c = ExaCore(; backend, concrete = Val(true))
        c, x = add_var(c, 2, 3)
        c, y = add_var(c, 2:4, 0:1)
        c, s = add_expr(c, x[i, j]^2 for (i, j) in Iterators.product(1:2, 1:3))
        @test [v.i for v in x[:, 2]] == [3, 4]
        @test [v.i for v in x[2, :]] == [2, 4, 6]
        @test [v.i for v in y[:, 1]] == [10, 11, 12]
        @test [v.i for v in x[:]] == [v.i for v in x[:, :]] == 1:6
        @test length(s[:, 1]) == 2
        @test_throws Exception x[:, 1, 1]
    end
end

function test_generator_free_constr(backend)
    return @testset "add_con(core, expr, itr)" begin
        c = ExaCore(; backend, concrete = Val(true))
        @add_var(c, x, 2)
        gen = (sin(x[i]) for i in 1:2)
        expr = gen.f(ExaModels.DataSource())
        @add_con(c, g, expr, 1:2)
        @test c.g isa ExaModels.Constraint
    end
end

function test_features(backend)
    @testset "Const" begin
        test_const(backend)
    end
    @testset "add_var(gen)" begin
        test_add_var_gen(backend)
    end
    @testset "Named variable/constraint access" begin
        test_named_var_access(backend)
    end
    @testset "add_par with dims" begin
        test_add_par_dims(backend)
    end
    @testset "Non-unit expression indexing" begin
        test_nonunit_expr(backend)
    end
    @testset "Colon indexing" begin
        test_colon_index(backend)
    end
    return @testset "Generator-free constraint" begin
        test_generator_free_constr(backend)
    end
end
