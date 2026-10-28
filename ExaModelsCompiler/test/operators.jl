# Operators on argument nodes that ExaModelsCompiler defines: ranges and type-directed
# conversions and constructors. (Recipe arithmetic and queries are tested in ExaModels.)

import ExaModels: instantiate, ArgSource

@testset "argument-node operators" begin
    arg = ArgSource()
    A = (N = 10, nh = 2, v = [1.0, 2.0, 3.0], x = 2.7)

    @testset "ranges and type-directed constructors" begin
        @test instantiate(1:arg.N, A) == 1:10
        @test instantiate(arg.nh:arg.N, A) == 2:10
        @test instantiate(zeros(Float32, arg.nh), A) == zeros(Float32, 2)
        @test instantiate(floor(Int, arg.x), A) === 2
        @test instantiate(convert(Float32, arg.nh), A) === 2.0f0
        @test (@inferred instantiate(1:arg.N, A)) == 1:10
        @test repr(1:arg.N) == "(1:arg.N)"
    end

    @testset "building a model against arg" begin
        # The stated use case: sizes and starting values supplied later.
        c = ExaModels.ExaCore(concrete = Val(true))
        ExaModels.@add_var(c, y, arg.N; start = arg.v)
        ExaModels.@add_obj(c, y[i]^2 for i in 1:arg.N)
        ExaModels.@add_con(c, y[i] - 2.0 for i in 1:arg.N; lcon = 0.0, ucon = 0.0)

        m = ExaModels.ExaModel(c, (N = 3, v = [1.0, 2.0, 3.0]))
        @test m.meta.nvar == 3
        @test m.meta.ncon == 3
        @test m.meta.x0 == [1.0, 2.0, 3.0]
        @test ExaModels.obj(m, m.meta.x0) == 14.0     # 1 + 4 + 9

        # The same core instantiates again, at a different size, without
        # having been consumed by the first instantiation.
        m2 = ExaModels.ExaModel(c, (N = 2, v = [4.0, 5.0]))
        @test m2.meta.nvar == 2
        @test m2.meta.x0 == [4.0, 5.0]
        @test ExaModels.obj(m2, m2.meta.x0) == 41.0   # 16 + 25
        @test m.meta.x0 == [1.0, 2.0, 3.0]            # first model untouched

        # Bounds too, not just sizes and starts.
        cb = ExaModels.ExaCore(concrete = Val(true))
        ExaModels.@add_var(cb, xb, arg.N; lvar = arg.lo, uvar = 0.0)
        mb = ExaModels.ExaModel(cb, (N = 2, lo = [-1.0, -2.0]))
        @test mb.meta.lvar == [-1.0, -2.0]
        @test mb.meta.uvar == [0.0, 0.0]
    end

    @testset "per-iteration indexing of arg is rejected with a reason" begin
        c = ExaModels.ExaCore(concrete = Val(true))
        ExaModels.@add_var(c, y, arg.N; start = 0.0)
        err = try
            ExaModels.@add_con(c, y[i] - arg.b[i] for i in 1:arg.N; lcon = 0.0, ucon = 0.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("generator", err.msg)
    end

    @testset "a model built against two argument objects" begin
        c, sz, dat = ExaModels.ExaCore(concrete = Val(true), nargs = Val(2))
        ExaModels.@add_var(c, y, sz.N; lvar = dat.lo, start = dat.v)
        ExaModels.@add_obj(c, y[i]^2 for i in 1:sz.N)

        m = ExaModels.ExaModel(c, (N = 3,), (lo = [-1.0, -2.0, -3.0], v = [4.0, 5.0, 6.0]))
        @test m.meta.nvar == 3
        @test m.meta.lvar == [-1.0, -2.0, -3.0]
        @test m.meta.x0 == [4.0, 5.0, 6.0]
        @test ExaModels.obj(m, m.meta.x0) == 77.0     # 16 + 25 + 36

        # The two are independent: re-instantiate with different data only.
        m2 = ExaModels.ExaModel(c, (N = 2,), (lo = [0.0, 0.0], v = [1.0, 2.0]))
        @test m2.meta.nvar == 2
        @test m2.meta.x0 == [1.0, 2.0]
    end
    @testset "an argument scalar inside an expression" begin
        # The LuksanVlcek shape: a loop-invariant scalar computed from the
        # size and used as a coefficient, so an argument node lands in the
        # expression graph next to `Var` nodes.  Built twice from one
        # source and compared, which is the only check that catches a
        # sparsity pattern that is plausible but wrong.
        function build(src)
            h = 1 / (src.N + 1)
            c = ExaModels.ExaCore(concrete = Val(true))
            ExaModels.@add_var(c, x, src.N; start = 1.0)
            ExaModels.@add_obj(c, h * (x[i] - 2.0)^2 for i in 1:src.N)
            ExaModels.@add_con(
                c, h * x[i] + x[i] for i in 1:src.N;
                lcon = 0.0, ucon = 1.0
            )
            return c
        end

        # NOTE: the evaluation point must not be called `x`.  `build` is a
        # nested function, so its `@add_var(c, x, ...)` binds `x` in the
        # *enclosing* local of that name rather than to one of its own —
        # silently turning the point vector into a `Variable`, and failing
        # far away as `+(::UnitRange, ::Int)` inside `getindex`.
        a = (N = 6,)
        ref = ExaModels.ExaModel(build(a))
        m = ExaModels.ExaModel(build(arg), a)
        pt = collect(0.1:0.1:0.6)
        mult = collect(0.6:-0.1:0.1)

        @test m.meta.nvar == ref.meta.nvar
        @test ExaModels.obj(m, pt) ≈ ExaModels.obj(ref, pt)
        @test NLPModels.grad(m, pt) ≈ NLPModels.grad(ref, pt)
        @test NLPModels.cons(m, pt) ≈ NLPModels.cons(ref, pt)
        # The recipe path stores blocks plain while the direct build
        # family-merges them; equal-valued Integer leaves share one slot in
        # the merged tree, so both have the same sparsity down to the
        # coordinate order
        @test m.meta.nnzj == ref.meta.nnzj
        @test m.meta.nnzh == ref.meta.nnzh
        # the structure equalities are the only check that catches a
        # plausible but wrong pattern
        @test NLPModels.jac_structure(m) == NLPModels.jac_structure(ref)
        @test NLPModels.jac_coord(m, pt) ≈ NLPModels.jac_coord(ref, pt)
        @test NLPModels.hess_structure(m) == NLPModels.hess_structure(ref)
        @test NLPModels.hess_coord(m, pt, mult) ≈ NLPModels.hess_coord(ref, pt, mult)

        # The coefficient really is deferred: the same core at another size
        # uses a different h, so a baked-in value would show as a wrong
        # objective rather than a wrong shape.
        a9 = (N = 9,)
        m9 = ExaModels.ExaModel(build(arg), a9)
        ref9 = ExaModels.ExaModel(build(a9))
        pt9 = collect(0.1:0.1:0.9)
        @test ExaModels.obj(m9, pt9) ≈ ExaModels.obj(ref9, pt9)
        # h differs between the two sizes, so a baked-in coefficient would
        # make these agree.
        @test !isapprox(ExaModels.obj(m9, pt9), ExaModels.obj(ref, pt))
    end
    @testset "a second block's offset is deferred too" begin
        # The first block's length is symbolic, so the second block's
        # offset is — and that offset is baked into every `Var` index of
        # the second block's expressions.
        function build(src)
            c = ExaModels.ExaCore(concrete = Val(true))
            ExaModels.@add_var(c, x, src.N; start = 1.0)
            ExaModels.@add_var(c, z, 3; start = 2.0)
            ExaModels.@add_obj(c, z[j]^2 for j in 1:3)
            ExaModels.@add_con(
                c, x[i] + z[1] for i in 1:src.N;
                lcon = 0.0, ucon = Inf
            )
            return c
        end

        for N in (4, 7)
            a = (N = N,)
            ref = ExaModels.ExaModel(build(a))
            m = ExaModels.ExaModel(build(arg), a)
            pt = collect(1.0:(N + 3)) ./ 10     # not `x`; see the note above
            @test m.meta.nvar == ref.meta.nvar == N + 3
            @test ExaModels.obj(m, pt) ≈ ExaModels.obj(ref, pt)
            @test NLPModels.cons(m, pt) ≈ NLPModels.cons(ref, pt)
            @test NLPModels.jac_structure(m) == NLPModels.jac_structure(ref)
            @test NLPModels.jac_coord(m, pt) ≈ NLPModels.jac_coord(ref, pt)
        end
    end
    @testset "recipe core in default mode" begin
        c, N = ExaModels.ExaCore(nargs = Val(1))
        ExaModels.@add_var(c, x, N; start = 1.5)
        ExaModels.@add_obj(c, (x[i] - 1)^2 for i in 1:N)
        m = ExaModels.ExaModel(c, 4)
        @test m.meta.nvar == 4
        @test NLPModels.obj(m, m.meta.x0) ≈ 1.0
    end
    @testset "arity is checked against the arity the core declared" begin
        c, N = ExaModels.ExaCore(concrete = Val(true), nargs = Val(1))
        ExaModels.@add_var(c, xb, N; start = 0.0)
        ExaModels.@add_obj(c, (xb[i] - 1.0)^2 for i in 1:N)
        @test ExaModels.ExaModel(c, 4).meta.nvar == 4
        # Surplus values were accepted in silence before the core carried
        # its arity: nothing consumes them, so a transposed or duplicated
        # argument built a different model without a word.
        @test_throws ArgumentError ExaModels.ExaModel(c, 4, 9)
        @test_throws ArgumentError ExaModels.ExaModel(c, 4, 9, 11)
        # A core that declared NOTHING is deliberately not held to an
        # arity: `Val(0)` records "did not say", not "has no
        # placeholders", because the documented bare-`ArgSource()` style
        # writes `arg.N` into a plain `ExaCore()`. Enforcing here would
        # reject the stated use case.
        plain = ExaModels.ExaCore(concrete = Val(true))
        ExaModels.@add_var(plain, xc, 3; start = 0.0)
        @test plain.nargs === Val(0)
        @test ExaModels.ExaModel(plain).meta.nvar == 3
        # Too few still reports a recipe rather than a BoundsError from
        # somewhere inside the build.
        @test_throws ArgumentError ExaModels.ExaModel(c)
    end
end
