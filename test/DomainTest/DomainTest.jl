module DomainTest

using Test, ExaModels, NLPModels

# Each model evaluates one function at a point outside its real domain.
const CASES = [
    (x -> sqrt(x), -1.0, 4.0),
    (x -> log(x), -1.0, 2.0),
    (x -> log2(x), -1.0, 2.0),
    (x -> log10(x), -1.0, 2.0),
    (x -> log1p(x), -2.0, 2.0),
    (x -> asin(x), 2.0, 0.5),
    (x -> acos(x), 2.0, 0.5),
    (x -> acosh(x), 0.5, 2.0),
    (x -> atanh(x), 2.0, 0.5),
    (x -> acoth(x), 0.5, 2.0),
    (x -> x^0.5, -1.0, 4.0),
    (x -> x^2.5, -1.0, 4.0),
]

function model(f)
    c = ExaCore()
    @add_var(c, x, 1)
    @add_obj(c, f(x[i]) for i in 1:1)
    @add_con(c, f(x[i]) for i in 1:1)
    return ExaModel(c)
end

function runtests()
    @testset "Evaluation outside a function's domain" begin
        for (f, outside, inside) in CASES
            m = model(f)
            nj, nh = NLPModels.get_nnzj(m), NLPModels.get_nnzh(m)
            x = [outside]
            @test isnan(NLPModels.obj(m, x))
            @test isnan(NLPModels.cons(m, x)[1])
            @test isnan(NLPModels.grad(m, x)[1])
            @test all(isnan, NLPModels.jac_coord(m, x))
            @test all(isnan, NLPModels.hess_coord(m, x, [1.0]))

            # Inside the domain, the values are Base's.
            x = [inside]
            @test NLPModels.obj(m, x) == f(inside)
            @test NLPModels.cons(m, x)[1] == f(inside)
            @test isfinite(NLPModels.grad(m, x)[1])
            @test all(isfinite, NLPModels.jac_coord(m, x))
            @test all(isfinite, NLPModels.hess_coord(m, x, [1.0]))
        end
    end

    return @testset "An integer power of a negative number stays defined" begin
        m = model(x -> x^2.0)
        @test NLPModels.obj(m, [-3.0]) == 9.0
        @test NLPModels.grad(m, [-3.0])[1] == -6.0
    end
end

end # module
