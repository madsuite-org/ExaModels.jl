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

        @testset "recipe core in default mode, instantiated" begin
            # default (Vector{Any}) storage built against an argument, without ranges
            c, d = ExaCore(nargs = Val(1))
            @add_var(c, y, length(d.v); start = d.v)
            @add_obj(c, y[i]^2 for i in d.idx)
            m = ExaModel(c, (v = [1.0, 2.0, 3.0], idx = 1:3))
            @test m.meta.nvar == 3
            @test NLPModels.obj(m, m.meta.x0) == 14.0
        end

        @testset "dynamic storage builds the same blocks as concrete" begin
            # the dynamic build path is compiled once rather than per block; it must
            # still produce blocks of the same types (so the same kernels) and values
            m1 = ExaModel(_build(; concrete = Val(true)))
            m2 = ExaModel(_build(; concrete = Val(false)))
            @test typeof(m1.cons) === typeof(m2.cons)
            @test typeof(m1.objs) === typeof(m2.objs)
            @test all(((a, b),) -> a.f.o1step == b.f.o1step && a.f.o2step == b.f.o2step &&
                a.f.comp1.inner == b.f.comp1.inner && a.f.comp2.inner == b.f.comp2.inner,
                zip(m1.cons, m2.cons))
        end

        @testset "erased handles build the same model" begin
            c1 = _build(; concrete = Val(false))
            c2 = _build(; concrete = Val(:erased))
            # the caller's handles carry no expression type; the core's blocks do
            c = ExaCore(; concrete = Val(:erased))
            c, x = add_var(c, 3)
            c, e = add_expr(c, x[i]^2 for i in 1:3)
            c, g = add_con(c, e[i] + x[i] for i in 1:3)
            c, a = add_con!(c, g, i => sin(x[i]) for i in 1:3)
            c, o = add_obj(c, e[i] * x[i] for i in 1:3)
            @test typeof(o).parameters[1] === Any
            @test all(b -> isconcretetype(typeof(b.f)), c.obj)
            @test typeof(e).parameters[2] === Any
            @test typeof(g).parameters[1] === Any
            @test typeof(a).parameters[1] === Any
            @test all(b -> isconcretetype(typeof(b.f)), c.cons)
            m1 = ExaModel(c1)
            m2 = ExaModel(c2)
            x0 = m1.meta.x0
            y0 = collect(1.0:m1.meta.ncon)
            @test m1.meta.nnzj == m2.meta.nnzj && m1.meta.nnzh == m2.meta.nnzh
            @test NLPModels.obj(m1, x0) == NLPModels.obj(m2, x0)
            @test NLPModels.cons(m1, x0) == NLPModels.cons(m2, x0)
            @test NLPModels.grad(m1, x0) == NLPModels.grad(m2, x0)
            @test NLPModels.jac_structure(m1) == NLPModels.jac_structure(m2)
            @test NLPModels.jac_coord(m1, x0) == NLPModels.jac_coord(m2, x0)
            @test NLPModels.hess_structure(m1) == NLPModels.hess_structure(m2)
            @test NLPModels.hess_coord(m1, x0, y0) == NLPModels.hess_coord(m2, x0, y0)
        end
    end
end

end # module
