module ArgumentTest

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
    return @testset "Argument tests" begin

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

        # ranges and type-directed constructors are tested in ExaModelsCompiler
        @testset "constructors" begin
            @test instantiate(zeros(arg.N), A) == zeros(10)
            @test instantiate(ones(arg.nh), A) == ones(2)
            @test instantiate(fill(1.5, arg.nh), A) == fill(1.5, 2)
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

        @testset "display" begin
            @test repr(arg) == "arg"
            @test repr(arg.N) == "arg.N"
            @test repr(arg.inner.k) == "arg.inner.k"
            @test repr(arg.N + 1) == "(arg.N + 1)"
            @test repr(2 * arg.N) == "(2 * arg.N)"
            @test repr(length(arg.v)) == "length(arg.v)"
        end

        @testset "a deferred iterator keeps an undeclared recipe's block plain" begin
            # a mergeable tree (concrete-size variable) over `arg.data`: the family
            # merge must leave the iterator alone, since its length and eltype
            # are deferred nodes until instantiation
            c = ExaModels.ExaCore()
            c, p = ExaModels.add_var(c, 3)
            c, _ = ExaModels.add_obj(c, (p[1] * exp(p[2] * d.t) - d.y)^2 for d in arg.data)
            c, _ = ExaModels.add_con(c, p[1] * d.t + p[3] for d in arg.data; lcon = -1.0, ucon = 1.0)
            data = [(t = 0.1k, y = 1.0 + k) for k in 1:4]
            m = ExaModels.ExaModel(c, (data = data,))
            xv = [1.0, 0.5, -0.25]
            @test NLPModels.obj(m, xv) ≈ sum((xv[1] * exp(xv[2] * d.t) - d.y)^2 for d in data)
            @test NLPModels.cons(m, xv) ≈ [xv[1] * d.t + xv[3] for d in data]

            # after a deferred-length block the core's counters are deferred, so
            # a later block with a concrete iterator has deferred row/nonzero
            # bases: it stays plain too and instantiates with them
            c = ExaModels.ExaCore(concrete = Val(true))
            c, p = ExaModels.add_var(c, 3)
            c, _ = ExaModels.add_obj(c, (p[1] * d.t - d.y)^2 for d in arg.data)
            c, _ = ExaModels.add_obj(c, p[2]^2 for _ in 1:1)
            c, _ = ExaModels.add_con(c, p[1] * d.t + p[3] for d in arg.data; lcon = -1.0, ucon = 1.0)
            c, _ = ExaModels.add_con(c, p[1] - 1.0 for _ in 1:1)
            m = ExaModels.ExaModel(c, (data = data,))
            @test NLPModels.obj(m, xv) ≈ sum((xv[1] * d.t - d.y)^2 for d in data) + xv[2]^2
            @test NLPModels.cons(m, xv) ≈ [[xv[1] * d.t + xv[3] for d in data]; xv[1] - 1.0]
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
                rows = [(i = i, w = Float64(i)) for i in 1:5]
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

        @testset "an iterable computed on the host moves to the core's backend" begin
            # An iterable built from an argument node (`ArgNode1(f, N)`) is
            # computed on the host when the recipe is instantiated. `add_con`
            # converts iterables when a block is added, but a recipe core's
            # iterables only exist after `instantiate`, which must convert them
            # too; otherwise the first kernel launch meets a host array.
            idx(N) = collect(1:2:(N - 1))
            for backend in Main.BACKENDS
                backend === nothing && continue
                c, N = ExaModels.ExaCore(; backend = backend, nargs = Val(1))
                ExaModels.@add_var(c, x, N; start = 1.0)
                ExaModels.@add_con(c, x[i] * x[i+1] - 1 for i in ExaModels.ArgNode1(idx, N))
                ExaModels.@add_obj(c, x[i]^2 for i in ExaModels.ArgNode1(idx, N))
                m = ExaModels.ExaModel(c, 10)
                host = ExaModels.convert_array(idx(10), backend)
                @testset "$backend" begin
                    @test typeof(m.cons[1].itr) == typeof(host)
                    @test typeof(m.objs[1].itr) == typeof(host)
                    x0 = ExaModels.convert_array(collect(1.0:10.0), backend)
                    @test Array(NLPModels.cons(m, x0)) ≈ [i * (i + 1) - 1 for i = 1:2:9]
                    @test NLPModels.obj(m, x0) ≈ sum(i^2 for i = 1:2:9)
                end
            end
        end
    end
end

end # module ArgumentTest
