module DomainTest

using Test, ExaModels, NLPModels
import ..BACKENDS

# Each case is a function, a point outside its real domain, and a point inside it.
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

function model(f, backend)
    c = ExaCore(; backend)
    @add_var(c, x, 1)
    @add_obj(c, f(x[i]) for i in 1:1)
    @add_con(c, f(x[i]) for i in 1:1)
    return WrapperNLPModel(ExaModel(c))
end

function power_model(backend)
    c = ExaCore(; backend)
    @add_var(c, x, 2)
    @add_obj(c, x[1]^x[2] for i in 1:1)
    return WrapperNLPModel(ExaModel(c))
end

function runtests()
    for backend in BACKENDS
        @testset "Evaluation outside a function's domain ($backend)" begin
            for (f, outside, inside) in CASES
                m = model(f, backend)
                x = [outside]
                @test isnan(NLPModels.obj(m, x))
                @test isnan(NLPModels.cons(m, x)[1])
                @test isnan(NLPModels.grad(m, x)[1])
                @test all(isnan, NLPModels.jac_coord(m, x))
                @test all(isnan, NLPModels.hess_coord(m, x, [1.0]))

                # Inside the domain, the values are Base's.
                x = [inside]
                @test NLPModels.obj(m, x) ≈ f(inside)
                @test NLPModels.cons(m, x)[1] ≈ f(inside)
                @test isfinite(NLPModels.grad(m, x)[1])
                @test all(isfinite, NLPModels.jac_coord(m, x))
                @test all(isfinite, NLPModels.hess_coord(m, x, [1.0]))
            end
        end

        @testset "Powers of a negative number ($backend)" begin
            m = model(x -> x^2.0, backend)
            @test NLPModels.obj(m, [-3.0]) ≈ 9.0
            @test NLPModels.grad(m, [-3.0])[1] ≈ -6.0

            # The value is defined, and the derivative in the exponent is not.
            m = power_model(backend)
            @test NLPModels.obj(m, [-1.0, 2.0]) ≈ 1.0
            g = NLPModels.grad(m, [-1.0, 2.0])
            @test g[1] ≈ -2.0
            @test isnan(g[2])

            # Base gives these a value, and the wrapper must not change that.
            @test ExaModels._nan_outside(^)(-2.0, Inf) == (-2.0)^Inf
            @test isnan(ExaModels._nan_outside(^)(-Inf, 0.5))
        end
    end
    return
end

end # module
