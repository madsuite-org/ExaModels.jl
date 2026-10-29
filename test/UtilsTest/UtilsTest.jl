module UtilsTest

using Test
import ExaModels, NLPModelsIpopt
import ..NLPTest: _exa_luksan_vlcek_model

UTIL_MODELS = [ExaModels.TimedNLPModel, ExaModels.CompressedNLPModel]

FIELDS = [:solution, :multipliers, :multipliers_L, :multipliers_U, :objective]

# Test struct for replace_float_64_by_32 coverage
struct _RFTestBus{T}
    id::Int
    vmax::T
    vmin::T
    name::Symbol
end

struct _RFTestNetwork{T}
    baseMVA::T
    bus::Vector{_RFTestBus{T}}
    coefs::NTuple{3, T}
end

# Float64-free nested isbits index structs (the shape of the NLPTest luksan_struct
# iterator): these must pass through replace_float_64_by_32 unchanged, also
# when broadcast over an array.
struct _RFTestI2
    i::Tuple{Int, Int}
end
struct _RFTestI1
    i::_RFTestI2
end

# whether an index tree still holds a product of two integers
_intprod(n::ExaModels.Node2{typeof(*), I1, I2}) where {I1 <: Integer, I2 <: Integer} = true
_intprod(n::ExaModels.Node2) = _intprod(n.inner1) || _intprod(n.inner2)
_intprod(n::ExaModels.Node1) = _intprod(n.inner)
_intprod(::Any) = false

function runtests()
    @testset "Utils tests" begin
        @testset "multi-dimensional index" begin
            for (sz, is) in [((3, 4, 5), (2, 3, 4)), ((2:4, 0:3), (3, 1)), ((5,), (4,)),
                    ((1:2, 3:5, -1:1), (2, 4, 0))]
                st = ExaModels._start.(sz)
                ln = ExaModels._length.(sz)
                @test ExaModels._lin_index(7, is, st, ln) == 7 + ExaModels.idxx(is .- (st .- 1), ln)
            end
            @test ExaModels._lin_index(7, (), (), ()) == 7 + ExaModels.idxx((), ())
            # an integer axis among symbolic ones folds into the constant: no
            # product of two integers is left in the index tree
            d = ExaModels.DataSource()
            ix = ExaModels._lin_index(5, (d[1], 2, d[2]), (2, 0, 1), (3, 3, 3))
            @test !_intprod(ix)
            c = ExaModels.ExaCore()
            c, z = ExaModels.add_var(c, 3; start = 0.3)
            c, x = ExaModels.add_var(c, 2:4, 0:2, 3; start = ((0.1 * (i + 2j + 5k)) for i in 2:4, j in 0:2, k in 1:3))
            c, p = ExaModels.add_par(c, [Float64(i + 10j) for i in 1:2, j in 1:3])
            c, _ = ExaModels.add_con(c, x[i, j, k]^2 * p[2, k] + z[k] + x[i, 1, k] for i in 2:4, j in 0:2, k in 1:3)
            m = ExaModels.ExaModel(c)
            x0 = m.meta.x0
            X = reshape(x0[4:end], 3, 3, 3)
            @test ExaModels.NLPModels.cons(m, x0) ≈
                vec([X[i - 1, j + 1, k]^2 * (2.0 + 10k) + x0[k] + X[i - 1, 2, k] for i in 2:4, j in 0:2, k in 1:3])
        end
        m, ~ = _exa_luksan_vlcek_model(nothing, 3)

        result = NLPModelsIpopt.ipopt(m; print_level = 0)

        for util_model in UTIL_MODELS
            util_result = NLPModelsIpopt.ipopt(util_model(m); print_level = 0)

            @testset "$util_model" begin
                for field in FIELDS
                    @testset "$field" begin
                        @test getfield(util_result, field) ≈ getfield(result, field) atol =
                            1e-6
                    end
                end
            end
        end

        @testset "replace_float_64_by_32" begin
            # Scalar leaf
            @test ExaModels.replace_float_64_by_32(1.5) === 1.5f0
            @test ExaModels.replace_float_64_by_32(1.5f0) === 1.5f0
            @test ExaModels.replace_float_64_by_32(3) === 3
            @test ExaModels.replace_float_64_by_32(:foo) === :foo

            # Tuple / NamedTuple recursion
            @test ExaModels.replace_float_64_by_32((1.0, 2, "x")) === (1.0f0, 2, "x")
            @test ExaModels.replace_float_64_by_32((a = 1.0, b = 2)) === (a = 1.0f0, b = 2)

            # Array of Float64
            @test ExaModels.replace_float_64_by_32([1.0, 2.0]) == Float32[1.0, 2.0]
            @test eltype(ExaModels.replace_float_64_by_32([1.0, 2.0])) === Float32

            # Mixed-eltype array routed elementwise
            arr = Any[1.0, 2]
            out = ExaModels.replace_float_64_by_32(arr)
            @test out[1] === 1.0f0 && out[2] === 2

            # Parametric struct rebuild
            b = _RFTestBus{Float64}(1, 1.1, 0.9, :b1)
            b32 = ExaModels.replace_float_64_by_32(b)
            @test b32 isa _RFTestBus{Float32}
            @test b32.id === 1
            @test b32.vmax === 1.1f0
            @test b32.name === :b1

            # Nested struct + array field
            net = _RFTestNetwork(100.0,
                [_RFTestBus{Float64}(1, 1.1, 0.9, :b1),
                 _RFTestBus{Float64}(2, 1.2, 0.8, :b2)],
                (0.1, 0.2, 0.3))
            net32 = ExaModels.replace_float_64_by_32(net)
            @test net32 isa _RFTestNetwork{Float32}
            @test net32.baseMVA === 100.0f0
            @test eltype(net32.bus) === _RFTestBus{Float32}
            @test net32.coefs === (0.1f0, 0.2f0, 0.3f0)

            # Identity on type with no Float64 anywhere
            @test ExaModels.replace_float_64_by_32((1, 2, 3)) === (1, 2, 3)

            # Nested isbits structs without Float64: returned as they are, element
            # by element, whether the array method or a broadcast is used. On
            # Julia 1.13.2 the broadcast of the struct rebuild returned corrupt
            # elements for exactly this shape (the luksan_struct iterator), which
            # the Float64-free short-circuit now avoids.
            idx = [_RFTestI1(_RFTestI2((i, j))) for i in 1:3, j in 1:2]
            @test ExaModels._contains_float64(_RFTestI1) === false
            @test ExaModels._contains_float64(_RFTestBus{Float64}) === true
            @test ExaModels._contains_float64(_RFTestBus{Float32}) === false
            @test ExaModels.replace_float_64_by_32(idx) === idx
            @test ExaModels.replace_float_64_by_32.(idx) == idx
            @test all(ExaModels.replace_float_64_by_32(x) === x for x in idx)
            sl = idx[1:(end - 1), :]
            @test ExaModels.replace_float_64_by_32(sl) == sl
            @test ExaModels.replace_float_64_by_32.(sl)[1] === _RFTestI1(_RFTestI2((1, 1)))
        end

        # The register macros must not assume `ExaModels` or its names are in
        # scope at the call site.
        @testset "register macro hygiene" begin
            m = Module(:RegisterHygiene)
            Core.eval(m, :(import ExaModels as EM))
            Core.eval(
                m, quote
                    f(x) = exp(x)
                    g(x, y) = x * y
                    EM.@register_univariate(f, f, f)
                    EM.@register_bivariate(g, (x, y) -> y, (x, y) -> x, (x, y) -> zero(x), (x, y) -> one(x), (x, y) -> zero(x))
                end
            )
            @test !isdefined(m, :ExaModels)
            # the methods were just defined, so build and evaluate in the latest world
            obj, grad = Base.invokelatest() do
                c = ExaModels.ExaCore()
                c, x = ExaModels.add_var(c, 2; start = [0.5, 2.0])
                c, _ = ExaModels.add_obj(c, m.f(x[1]) + m.g(x[1], x[2]) for _ in 1:1)
                nlp = ExaModels.ExaModel(c)
                x0 = [0.5, 2.0]
                ExaModels.NLPModels.obj(nlp, x0), ExaModels.NLPModels.grad(nlp, x0)
            end
            @test obj ≈ exp(0.5) + 1.0
            @test grad ≈ [exp(0.5) + 2.0, 0.5]
        end
    end
end

end
