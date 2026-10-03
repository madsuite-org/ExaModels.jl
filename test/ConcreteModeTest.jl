module ConcreteModeTest

using Test, ExaModels
import NLPModels

# The same small model (variables, parameters, a subexpression, constraints,
# an augmentation, and an objective) built on either storage mode.
function _build(; concrete)
    n = 10
    c = ExaCore(; concrete)
    c, x = add_var(c, n; lvar = -1.0, uvar = 1.0, start = 0.5)
    c, θ = add_par(c, ones(n))
    c, s = add_expr(c, θ[i] * x[i]^2 for i in 1:n)
    c, g = add_con(c, x[i] + x[i+1] for i in 1:n-1; lcon = -1.0, ucon = 1.0)
    c, _ = add_con!(c, g, i => sin(x[i+1]) for i in 1:n-1)
    c, _ = add_obj(c, s[i] for i in 1:n)
    return c
end

function runtests()
    @testset "Concrete mode" begin
        @testset "default is type-erased storage" begin
            c = ExaCore()
            @test c.var isa Vector{Any}
            @test c.par isa Vector{Any}
            @test c.obj isa Vector{Any}
            @test c.cons isa Vector{Any}
            # The point of the erased storage: adding a block leaves the
            # core's type unchanged, so the builder compiles once rather
            # than once per block.
            c1, x = add_var(c, 10)
            @test typeof(c1) === typeof(c)
            c2, _ = add_obj(c1, x[i]^2 for i in 1:10)
            @test typeof(c2) === typeof(c1)
        end

        @testset "_concretize recovers the Val(true) core" begin
            cf = _build(concrete = Val(false))
            ct = _build(concrete = Val(true))
            @test typeof(ExaModels._concretize(cf)) === typeof(ct)
            # Identity on a core that is already concrete.
            @test ExaModels._concretize(ct) === ct
        end

        @testset "both modes build the same model" begin
            mf = ExaModel(_build(concrete = Val(false)))
            mt = ExaModel(_build(concrete = Val(true)))
            @test typeof(mf) === typeof(mt)
            x = [0.1i for i in 1:10]
            @test NLPModels.obj(mf, x) == NLPModels.obj(mt, x)
            @test NLPModels.grad(mf, x) == NLPModels.grad(mt, x)
            @test NLPModels.cons(mf, x) == NLPModels.cons(mt, x)
            @test NLPModels.jac_coord(mf, x) == NLPModels.jac_coord(mt, x)
            y = ones(mf.meta.ncon)
            @test NLPModels.hess_coord(mf, x, y) == NLPModels.hess_coord(mt, x, y)
        end

        @testset "recipe core in default mode" begin
            c, N = ExaCore(nargs = Val(1))
            @add_var(c, x, N; start = 1.5)
            @add_obj(c, (x[i] - 1)^2 for i in 1:N)
            m = ExaModel(c, 4)
            @test m.meta.nvar == 4
            @test NLPModels.obj(m, m.meta.x0) ≈ 1.0
        end
    end
end

end # module
