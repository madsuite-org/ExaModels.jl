using Test
using ExaModelsIO, ExaModels, NLPModels
include("roundtrip.jl")

# A function outside the operator vocabulary, registered with ExaModels.
myf(x) = x^2 + x
ExaModels.@register_univariate(myf, x -> 2x + 1, x -> 2.0)

@testset "ExaModelsIO" begin
    @testset "rosenbrock with parameters" begin
        c = ExaCore()
        x = @add_var(c, x, 10; start = 0.5, lvar = -1, uvar = 2)
        θ = @add_par(c, θ, [1.0, 2.0])
        @add_obj(c, 100 * (x[i+1] - x[i]^2)^2 + (x[i] - θ[1])^2 for i = 1:9)
        @add_con(c, g, x[i] + x[i+1] for i = 1:9; lcon = -1.5, ucon = 1.5)
        m2 = roundtrip_check(ExaModel(c))
        @test m2.x isa ExaModels.Variable && m2.g isa ExaModels.Constraint
    end

    @testset "2-D variables, named-tuple data, augmentation" begin
        c = ExaCore()
        x = @add_var(c, x, 10)
        y = @add_var(c, y, 1:2, 1:3; start = 0.1)
        data = [(i = i, j = j, a = 0.5i) for i in 1:2 for j in 1:3]
        @add_obj(c, p.a * sin(y[p.i, p.j]) + exp(x[p.i] / 3) for p in data)
        g = @add_con(c, g, x[i] * x[i+1] for i = 1:9; ucon = 1.0)
        @add_con!(c, g, i => y[1, 1] * 2 for i in 1:3)
        @add_con!(c, g, p.i => cos(y[p.i, p.j]) for p in data)
        roundtrip_check(ExaModel(c))
    end

    @testset "2-D constraints, tuple data, dims form, subexpressions" begin
        c = ExaCore()
        x = @add_var(c, x, 2, 3; lvar = 0)
        @add_con(c, h, x[i, j]^3 for i in 1:2, j in 1:3; lcon = 0.0)
        # an axis that is a collection rather than a range
        @add_con(c, k, x[i, t[1]] * t[2] for i in 1:2, t in [(1, 0.5), (3, 2.0)])
        g = @add_con(c, g, 2, 2)
        @add_con!(c, g, (i, j) => x[i, 1] * x[j, 2] for i in 1:2, j in 1:2)
        e = @add_expr(c, e, x[1, i]^2 + 1 for i in 1:3)
        @add_obj(c, e[i] * log(x[2, i]) for i in 1:3)
        @add_obj(c, sum(x[1, j] for j in 1:3))
        roundtrip_check(ExaModel(c; ), )
    end

    @testset "constant augmentation" begin
        c = ExaCore()
        x = @add_var(c, x, 6; start = 0.3)
        con = @add_con(c, con, 1; lcon = 0.0, ucon = 0.0)
        @add_con!(c, con[1] += x[i] for i in 1:6)
        @add_con!(c, con[1] += -3.0 for _ in 1:1)
        # the form ExaModelsMOI builds: no index, element k adds to row k
        g = @add_con(c, g, 3; ucon = 1.0)
        c, _ = ExaModels.add_con!(c, g, Base.Generator(p -> p[2] * x[p[1]]^2, [(1, 2.0), (4, 3.0), (6, 1.5)]))
        # constant terms over empty-tuple data, as ExaModels' MOI converter builds them
        c, _ = ExaModels.add_con!(c, g, Base.Generator(_ -> ExaModels.Null(0.25), [(), (), ()]))
        data = [(q = i, w = (0.5i, 1.0, -i)) for i in 1:4]
        @add_obj(c, x[d.q] * sum(d.w[k] * x[k] for k in 1:3) for d in data)
        roundtrip_check(ExaModel(c))
    end

    @testset "Float32 literals, non-finite constants" begin
        c = ExaCore(Float32)
        x = @add_var(c, x, 4; start = 0.5f0, lvar = -Inf32, uvar = 2.5f0)
        @add_obj(c, 1.5f0 * x[i]^3 + 0.1f0 * exp(x[i]) for i in 1:4)
        @add_con(c, g, min(x[i], Inf32) - 0.3f0 * x[i+1] for i in 1:3; lcon = -Inf32)
        m = ExaModel(c)
        path = tempname() * ".h5"
        write_model(path, m)
        m2 = read_model(path)
        x0 = m.meta.x0
        @test eltype(m2.meta.x0) == Float32
        @test obj(m, x0) == obj(m2, x0) && grad(m, x0) == grad(m2, x0) && cons(m, x0) == cons(m2, x0)
        @test m2.meta.lvar == m.meta.lvar && m2.meta.lcon == m.meta.lcon
    end

    @testset "operator vocabulary and user functions" begin
        c = ExaCore()
        x = @add_var(c, x, 3; start = 0.7)
        @add_obj(c, sin(x[i]) + hypot(x[i], 2.0) + myf(x[i]) for i in 1:3)
        m = ExaModel(c)
        path = tempname() * ".h5"
        write_model(path, m)
        s = ExaModelsIO.HDF5.h5open(f -> read(f["structure"]), path)
        @test occursin("\"sin\"", s) && occursin("\"hypot\"", s) && !occursin("Base.", s)
        @test occursin("\"extensions\":[\"Main.myf\"]", s)
        @test_throws ArgumentError read_model(path)          # user function not supplied
        m2 = read_model(path; functions = Dict("Main.myf" => myf))
        x0 = m.meta.x0
        @test obj(m, x0) ≈ obj(m2, x0) && grad(m, x0) ≈ grad(m2, x0)
    end

    @testset "format version" begin
        c = ExaCore(); x = @add_var(c, x, 2); @add_obj(c, x[i]^2 for i in 1:2)
        path = tempname() * ".h5"
        write_model(path, ExaModel(c))
        H = ExaModelsIO.HDF5
        V = ExaModelsIO.FORMAT_VERSION
        @test H.h5open(f -> read(H.attributes(f)["format_version"]), path) == "$(V.major).$(V.minor)"
        setversion(v) = H.h5open(path, "r+") do f
            a = H.attrs(f); delete!(a, "format_version"); a["format_version"] = v
        end
        setversion("1.99")
        @test read_model(path).meta.nvar == 2                  # newer minor version: read
        setversion("2.0")
        @test_throws ArgumentError read_model(path)             # other major version: refuse
    end

    if isdefined(ExaModels, :BufferedExpression)
        @testset "buffered subexpressions" begin
            c = ExaCore()
            x = @add_var(c, x, 6, 2; start = 0.3)
            θ = @add_par(c, θ, [2.0, 0.5])
            # two levels, as in WM_CFy: partial sums of squares, then a norm per column
            c, q = add_expr(c, (sum(x[3(b - 1) + i, k]^2 for i in 1:3) for b in 1:2, k in 1:2); buffered = true)
            c, r = add_expr(c, (sqrt(θ[2] * (q[1, k] + q[2, k])) for k in 1:2); buffered = true, name = Val(:r))
            @add_obj(c, (x[i, k] - θ[1])^2 for i in 1:6, k in 1:2)
            @add_con(c, g, x[i, k] * r[k] for i in 1:6, k in 1:2; ucon = 1.0)
            m2 = roundtrip_check(ExaModel(c))
            @test length(m2.subexprs) == 2 && m2.r isa ExaModels.BufferedExpression
        end
    end

    @testset "Float32, maximize, solution" begin
        c = ExaCore(Float32; minimize = false)
        x = @add_var(c, x, 3; start = 1)
        @add_obj(c, -(x[i] - 1)^2 for i in 1:3)
        m = ExaModel(c)
        path = tempname() * ".h5"
        sol = (solution = ones(Float32, 3), multipliers = Float32[], multipliers_L = zeros(Float32, 3),
               multipliers_U = zeros(Float32, 3), objective = 0.0f0, status = :first_order)
        write_model(path, m; solution = sol)
        @test read_model(path).meta.x0 == m.meta.x0
        @test eltype(read_model(path).meta.x0) == Float32
        s = read_solution(path)
        @test s.solution == ones(3) && s.status == "first_order"
    end
end
