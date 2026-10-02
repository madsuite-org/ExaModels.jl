module ArgumentTest

import SparseArrays

using Test
import ExaModels
import NLPModels
import ExaModels: instantiate, ArgSource, ArgIndexed, ArgNode1, ArgNode2, _anyarg
using Test: @inferred

# `ArgSource` placeholders come from `ExaCore(nargs = ...)`; the bare
# constructor is equivalent and is what most of these unit tests use.
const arg = ArgSource()

# The argument object every test instantiates against.
const A = (
    N = 10,
    nh = 2,
    v = [1.0, 2.0, 3.0],
    x = 2.7,
    inner = (k = 4, w = [5.0, 6.0]),
)

mutable struct _Mutable
    a::Int
end
Base.:(==)(x::_Mutable, y::_Mutable) = x.a == y.a

function runtests()
    @testset "Argument tests" begin

        @testset "instantiate is the identity without an arg dependency" begin
            v = [1.0, 2.0]
            m = _Mutable(3)

            # `===` is a real identity claim only for mutable/reference values.
            # For an immutable — a number, a tuple of them — `===` is
            # egal-by-value, so it cannot tell "handed back untouched" from
            # "rebuilt with equal contents".
            #
            # Measured, not assumed: making `instantiate(::Tuple, _)` map
            # unconditionally instead of short-circuiting left this whole file
            # green, while a control mutation of `_arg_access` turned 45
            # assertions red.  The tuple `===` claims here therefore prove
            # nothing about rebuilding, and nothing below relies on them to.
            # What *is* tested is the property that does bite: a container must
            # not copy the mutable values it holds.
            for x in (v, m, Ref(1.0), Dict(:a => 1))
                @test instantiate(x, A) === x
            end

            for x in (7, 3.5, :sym, "str", v, m, nothing, 1:4, (1, v), (a = 1, b = v))
                @test instantiate(x, A) == x
                @test _anyarg(x) === Val(false)
            end

            # A container with no arg dependency must not copy what it holds.
            @test instantiate((1, v), A)[2] === v
            @test instantiate((a = 1, b = v), A).b === v
        end

        @testset "field and index access" begin
            @test instantiate(arg, A) === A
            @test instantiate(arg.N, A) == 10
            @test instantiate(arg.v, A) === A.v
            @test instantiate(arg.inner.k, A) == 4
            @test instantiate(arg.inner.w, A) === A.inner.w
            @test instantiate(arg[1], A) == 10          # first field of the NamedTuple
            @test instantiate(arg.v[2], A) == 2.0
            @test _anyarg(arg.inner.k) === Val(true)

            # Access paths stay in the type, mirroring DataIndexed.
            @test arg.N isa ArgIndexed{ArgSource{1}, :N}
            @test arg.inner.k isa ArgIndexed{ArgIndexed{ArgSource{1}, :inner}, :k}
        end

        @testset "basic algebra" begin
            @test instantiate(arg.N + 1, A) == 11
            @test instantiate(1 + arg.N, A) == 11
            @test instantiate(arg.N - arg.nh, A) == 8
            @test instantiate(2 * arg.N, A) == 20
            @test instantiate(arg.N / arg.nh, A) == 5.0
            @test instantiate(arg.N ÷ arg.nh, A) == 5
            @test instantiate(-arg.N, A) == -10
            @test instantiate(arg.N^2, A) == 100          # via literal_pow
            @test instantiate(arg.N^arg.nh, A) == 100
            @test instantiate(abs(-arg.N), A) == 10
            @test instantiate(sqrt(arg.x), A) ≈ sqrt(2.7)
            @test instantiate(min(arg.N, arg.nh), A) == 2
            @test instantiate(max(arg.N, arg.nh), A) == 10
        end

        @testset "size queries" begin
            @test instantiate(length(arg.v), A) == 3
            @test instantiate(size(arg.v), A) == (3,)
            @test instantiate(size(arg.v, 1), A) == 3
            @test instantiate(sum(arg.v), A) == 6.0
            @test instantiate(first(arg.v), A) == 1.0
            @test instantiate(length(arg.v) + 1, A) == 4
        end

        @testset "mixing with concrete arrays" begin
            @test instantiate(arg.nh * zeros(4), A) == zeros(4)
            @test instantiate(arg.nh * ones(3), A) == fill(2.0, 3)
            @test instantiate(ones(3) * arg.nh, A) == fill(2.0, 3)
            # Concrete container, symbolic index.
            w = [10.0, 20.0, 30.0]
            @test instantiate(w[arg.nh], A) == 20.0
        end

        @testset "broadcasting is refused, and says what to do instead" begin
            # Deliberately unsupported: an array you want elementwise arithmetic
            # on is one you can build in the args function.  The refusal has to
            # be explicit, because Base's `broadcastable(x) = collect(x)`
            # fallback would otherwise wrap the node in a `collect` and fail
            # later as something unrecognisable.
            for bad in (() -> arg.v .+ 1, () -> 2 .* arg.v, () -> sqrt.(arg.v))
                err = try
                    bad()
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("args", err.msg)
            end
        end

        @testset "ranges and constructors" begin
            @test instantiate(1:arg.N, A) == 1:10
            @test instantiate(arg.nh:arg.N, A) == 2:10
            @test instantiate(zeros(arg.N), A) == zeros(10)
            @test instantiate(ones(arg.nh), A) == ones(2)
            @test instantiate(zeros(Float32, arg.nh), A) == zeros(Float32, 2)
            @test instantiate(fill(1.5, arg.nh), A) == fill(1.5, 2)
            @test instantiate(floor(Int, arg.x), A) === 2
            @test instantiate(convert(Float32, arg.nh), A) === 2.0f0
        end

        @testset "containers holding arg nodes" begin
            t = (arg.N, 3)
            @test instantiate(t, A) == (10, 3)
            @test _anyarg(t...) === Val(true)

            nt = (n = arg.N, m = 3)
            @test instantiate(nt, A) == (n = 10, m = 3)
            @test _anyarg(nt...) === Val(true)

            # A tuple of tuples, only one of which is symbolic.
            @test instantiate(((arg.nh,), (1,)), A) == ((2,), (1,))
        end

        @testset "type stability" begin
            @test (@inferred instantiate(arg.N, A)) == 10
            @test (@inferred instantiate(arg.N + 1, A)) == 11
            @test (@inferred instantiate(length(arg.v), A)) == 3
            @test (@inferred instantiate(arg.nh * ones(3), A)) == fill(2.0, 3)
            @test (@inferred instantiate(1:arg.N, A)) == 1:10
            @test (@inferred instantiate([1.0, 2.0], A)) == [1.0, 2.0]
            @test (@inferred instantiate((arg.N, 3), A)) == (10, 3)
        end

        @testset "placeholders come from the core, and are numbered" begin
            # `nargs = Val(0)` is the default and must be exactly main's
            # behaviour: the core alone, not a one-element tuple.
            @test ExaModels.ExaCore(concrete = Val(true)) isa ExaModels.ExaCore
            @test ExaModels.ExaCore(concrete = Val(true), nargs = Val(0)) isa ExaModels.ExaCore

            c1, a1 = ExaModels.ExaCore(concrete = Val(true), nargs = Val(1))
            @test c1 isa ExaModels.ExaCore
            @test a1 === ArgSource{1}()
            @test ArgSource() === ArgSource{1}()

            c3, b1, b2, b3 = ExaModels.ExaCore(concrete = Val(true), nargs = Val(3))
            @test (b1, b2, b3) === (ArgSource{1}(), ArgSource{2}(), ArgSource{3}())

            # Each placeholder resolves to its own argument object, by position.
            @test instantiate(b1.v, (v = 1,), (v = 2,), (v = 3,)) == 1
            @test instantiate(b2.v, (v = 1,), (v = 2,), (v = 3,)) == 2
            @test instantiate(b3.v, (v = 1,), (v = 2,), (v = 3,)) == 3

            # How many values come back is known statically.
            f() = ExaModels.ExaCore(concrete = Val(true), nargs = Val(2))
            @test (@inferred f()) isa Tuple{ExaModels.ExaCore, ArgSource{1}, ArgSource{2}}
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
                ExaModels.@add_con(c, h * x[i] + x[i] for i in 1:src.N;
                                   lcon = 0.0, ucon = 1.0)
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
            # the recipe path stores blocks plain while the direct build
            # family-merges them, so nnz counts and coordinate lists are
            # representation-dependent; the assembled matrices are the
            # contract (duplicate COO coordinates sum)
            _assemble(mm, rows, cols, vals, nr, nc) = SparseArrays.sparse(rows, cols, vals, nr, nc)
            jm_r, jm_c = NLPModels.jac_structure(m)
            jr_r, jr_c = NLPModels.jac_structure(ref)
            @test SparseArrays.sparse(jm_r, jm_c, NLPModels.jac_coord(m, pt), m.meta.ncon, m.meta.nvar) ≈
                  SparseArrays.sparse(jr_r, jr_c, NLPModels.jac_coord(ref, pt), ref.meta.ncon, ref.meta.nvar)
            hm_r, hm_c = NLPModels.hess_structure(m)
            hr_r, hr_c = NLPModels.hess_structure(ref)
            @test SparseArrays.sparse(hm_r, hm_c, NLPModels.hess_coord(m, pt, mult), m.meta.nvar, m.meta.nvar) ≈
                  SparseArrays.sparse(hr_r, hr_c, NLPModels.hess_coord(ref, pt, mult), ref.meta.nvar, ref.meta.nvar)

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
                ExaModels.@add_con(c, x[i] + z[1] for i in 1:src.N;
                                   lcon = 0.0, ucon = Inf)
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

        @testset "a core with no arg dependency is built exactly as before" begin
            build() = begin
                c = ExaModels.ExaCore(concrete = Val(true))
                ExaModels.@add_var(c, x, 4; start = 1.0)
                ExaModels.@add_obj(c, x[i]^2 for i in 1:4)
                c
            end
            c = build()

            # `ExaModel(core)` and `ExaModel(core, nothing)` must agree, and the
            # second must not route through instantiation at all.
            m1 = ExaModels.ExaModel(c)
            m2 = ExaModels.ExaModel(c, nothing)
            @test typeof(m1) === typeof(m2)
            @test m1.meta.nvar == m2.meta.nvar == 4
            @test ExaModels.obj(m1, m1.meta.x0) == ExaModels.obj(m2, m2.meta.x0) == 4.0

            # Instantiating it anyway is a no-op on every slot.
            ci = instantiate(c, (N = 99,))
            @test typeof(ci) === typeof(c)
            @test ci.nvar === c.nvar
            @test ci.x0 === c.x0
        end

        @testset "an un-walkable field is reported, not silently kept" begin
            # `instantiate` is the identity on types it has no method for, so a
            # placeholder could otherwise survive into a model that looks built.
            #
            # The core must DECLARE `nargs = Val(1)` here, even though these
            # tests use the bare `arg`.  On a core declaring `Val(0)` the arity
            # check rejects the one-value call before `instantiate` ever runs,
            # and this testset would pass without exercising the thing it is
            # named after — the same ArgumentError, from the wrong guard.
            c, _ = ExaModels.ExaCore(concrete = Val(true), nargs = Val(1))
            ExaModels.@add_var(c, xa, arg.N; start = 0.0)
            # Smuggle a placeholder into a slot nothing walks.
            bad = ExaModels.ExaCore(c; tag = Ref(arg.N))
            @test bad.nargs === Val(1)          # so the arity guard passes it through
            @test_throws ArgumentError ExaModels.ExaModel(bad, (N = 2,))
            # ... and the check is what rejects it, not a downstream accident.
            @test ExaModels._mentions_arg(typeof(instantiate(bad, (N = 2,))))
            @test !ExaModels._mentions_arg(typeof(instantiate(c, (N = 2,))))
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

        @testset "display" begin
            @test repr(arg) == "arg"
            @test repr(arg.N) == "arg.N"
            @test repr(arg.inner.k) == "arg.inner.k"
            @test repr(arg.N + 1) == "(arg.N + 1)"
            @test repr(2 * arg.N) == "(2 * arg.N)"
            @test repr(1:arg.N) == "(1:arg.N)"
            @test repr(length(arg.v)) == "length(arg.v)"
        end

        @testset "a converted iterable stays on its backend" begin
            # The deferred collect in _adapt_gen must not migrate an argument
            # back to the host: collect(::CuArray) is a Vector, so a recipe
            # instantiated with device-converted arguments produced a model
            # whose kernels met a non-bitstype host array. Invisible on the
            # CPU, where collect of a Vector is just a copy — which is why
            # this asserts RESIDENCE, not values.
            for backend in Main.BACKENDS
                backend === nothing && continue
                c, d = ExaModels.ExaCore(nargs = Val(1))
                ExaModels.@add_var(c, x, length(d.rows))
                ExaModels.@add_obj(c, (x[r.i] - r.w)^2 for r in d.rows)
                rows = [(i = i, w = Float64(i)) for i = 1:5]
                dev = ExaModels.convert_array(rows, backend)
                m = ExaModels.ExaModel(c, (; rows = dev); backend = backend)
                @testset "$backend" begin
                    # Residence via typeof equality. Only a device backend
                    # discriminates — on CPU() the converted array IS a Vector,
                    # so the pre-fix collect was a same-type copy there; the
                    # CUDA leg is the one this test exists for.
                    @test typeof(m.objs[1].itr) == typeof(dev)
                end
            end
        end
    end
end

end # module ArgumentTest
