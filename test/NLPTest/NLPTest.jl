module NLPTest

using ExaModels
using ExaPowerIO, Test
using NLPModels, NLPModelsJuMP, NLPModelsIpopt, NLPModelsTest
using JuMP, PowerModels, MadNLP, Percival

import ..BACKENDS, ..BACKEND_TESTS_ONLY
import ..ad_tolerance, ..sol_tolerance, ..solver_tolerance

const NLP_TEST_ARGUMENTS = [
    ("luksan_struct", 3),
    ("luksan_struct", 20),
    ("luksan_vlcek", 3),
    ("luksan_vlcek", 20),
    ("ac_power", "pglib_opf_case3_lmbd.m"),
    ("ac_power", "pglib_opf_case14_ieee.m"),
    ("trivialmax", 1), # Issue #518 in MadNLP
]

const SOLVERS = [
    ("ipopt", (nlp; kwargs...) -> ipopt(nlp; print_level = 0, kwargs...)),
    ("madnlp", (nlp; kwargs...) -> madnlp(nlp; print_level = MadNLP.ERROR, kwargs...)),
    ("percival", (nlp; kwargs...) -> percival(nlp, kwargs...)),
]

const EXCLUDE1 = [("ac_power", "percival"), ("struct_ac_power", "percival"), ("trivialmax", "percival")]

include("luksan.jl")
include("power.jl")
include("luksan_struct.jl")
include("parameter_test.jl")
include("subexpr_test.jl")
include("trivialmax.jl")
include("feature_test.jl")
include("conaug_test.jl")

function test_nlp(m1, m2; full = false, tol = sol_tolerance(eltype(m1.meta.x0), eltype(m2.meta.x0)))
    @testset "NLP meta tests" begin
        list = [:nvar, :ncon, :x0, :lvar, :uvar, :y0, :lcon, :ucon]

        if full
            append!(list, [:nnzj, :nnzh])
        end

        for field in list
            @testset "$field" begin
                A = getfield(m1.meta, field)
                B = getfield(m2.meta, field)
                if eltype(A) <: AbstractFloat
                    @test A ≈ B atol = sol_tolerance(eltype(A), eltype(B)) rtol = sol_tolerance(eltype(A), eltype(B))
                else
                    @test A == B
                end
            end
        end
    end

    @testset "NLP callback tests" begin
        x0 = copy(m2.meta.x0)
        y0 = randn(eltype(m2.meta.x0), m2.meta.ncon)
        u = randn(eltype(m2.meta.x0), m2.meta.nvar)
        v = randn(eltype(m2.meta.x0), m2.meta.ncon)

        @test NLPModels.obj(m1, x0) ≈ NLPModels.obj(m2, x0) atol = tol rtol = tol
        @test NLPModels.cons(m1, x0) ≈ NLPModels.cons(m2, x0) atol = tol rtol = tol
        @test NLPModels.grad(m1, x0) ≈ NLPModels.grad(m2, x0) atol = tol rtol = tol
        @test NLPModels.jprod(m1, x0, u) ≈ NLPModels.jprod(m2, x0, u) atol = tol rtol = tol
        @test NLPModels.jtprod(m1, x0, v) ≈ NLPModels.jtprod(m2, x0, v) atol = tol rtol = tol
        @test NLPModels.hprod(m1, x0, y0, u) ≈ NLPModels.hprod(m2, x0, y0, u) atol = tol rtol = tol

        if full
            jac_buffer1 = zeros(m1.meta.nnzj)
            jac_buffer2 = zeros(m2.meta.nnzj)
            jac_I_buffer1 = zeros(Int, m1.meta.nnzj)
            jac_I_buffer2 = zeros(Int, m2.meta.nnzj)
            jac_J_buffer1 = zeros(Int, m1.meta.nnzj)
            jac_J_buffer2 = zeros(Int, m2.meta.nnzj)

            hess_buffer1 = zeros(m1.meta.nnzh)
            hess_buffer2 = zeros(m2.meta.nnzh)
            hess_I_buffer1 = zeros(Int, m1.meta.nnzh)
            hess_I_buffer2 = zeros(Int, m2.meta.nnzh)
            hess_J_buffer1 = zeros(Int, m1.meta.nnzh)
            hess_J_buffer2 = zeros(Int, m2.meta.nnzh)

            NLPModels.jac_coord!(m1, x0, jac_buffer1)
            NLPModels.jac_coord!(m2, x0, jac_buffer2)
            NLPModels.hess_coord!(m1, x0, y0, hess_buffer1)
            NLPModels.hess_coord!(m2, x0, y0, hess_buffer2)
            NLPModels.jac_structure!(m1, jac_I_buffer1, jac_J_buffer1)
            NLPModels.jac_structure!(m2, jac_I_buffer2, jac_J_buffer2)
            NLPModels.hess_structure!(m1, hess_I_buffer1, hess_J_buffer1)
            NLPModels.hess_structure!(m2, hess_I_buffer2, hess_J_buffer2)

            @test jac_buffer1 ≈ jac_buffer2 atol = tol rtol = tol
            @test hess_buffer1 ≈ hess_buffer2 atol = tol rtol = tol
            @test jac_I_buffer1 == jac_I_buffer2
            @test jac_J_buffer1 == jac_J_buffer2
            @test hess_I_buffer1 == hess_I_buffer2
            @test hess_J_buffer1 == hess_J_buffer2
        end
    end
end

function test_nlp_solution(result1, result2; tol = sol_tolerance(eltype(result1.solution),eltype(result2.solution)))

    @testset "solution test" begin
        @test result1.status == result2.status
        for field in [:solution, :multipliers, :multipliers_L, :multipliers_U]
            @testset "$field" begin
                @test getfield(result1, field) ≈ getfield(result2, field) atol = tol rtol = tol
            end
        end
    end
end

dual_lb(x) = has_lower_bound(x) ? dual(LowerBoundRef(x)) : 0.0
dual_ub(x) = has_upper_bound(x) ? dual(UpperBoundRef(x)) : 0.0

function test_api(result1, vars1, cons1, vars2, cons2, minimize::Bool; tol = sol_tolerance(eltype(result1.solution)))
    @testset "API test" begin
        for (var1, var2) in zip(vars1, vars2)
            @test solution(result1, var1) ≈ [value(var) for var in var2] atol = tol rtol = tol
            @test multipliers_L(result1, var1) ≈ [dual_lb(var) for var in var2] atol = tol rtol = tol
            @test multipliers_U(result1, var1) ≈ [-dual_ub(var) for var in var2] atol = tol rtol = tol
        end
        for (con1, con2) in zip(cons1, cons2)
            if minimize
                @test multipliers(result1, con1) ≈ [-dual.(con) for con in con2] atol = tol rtol = tol
            else
                @test multipliers(result1, con1) ≈ [dual.(con) for con in con2] atol = tol rtol = tol
            end
        end
    end
end

function runtests()
    @testset "NLP test" begin
        # The JuMP comparisons build only host models, so they sit outside the
        # backend loop.
        BACKEND_TESTS_ONLY || @testset "Comparison to JuMP" begin
            for (name, args) in NLP_TEST_ARGUMENTS
                @testset "$name $args" begin
                    exa_model = getfield(@__MODULE__, Symbol("_exa_$(name)_model"))
                    jump_model = getfield(@__MODULE__, Symbol("_jump_$(name)_model"))

                    m0 = WrapperNLPModel(first(exa_model(nothing, args)))
                    m2 = MathOptNLPModel(first(jump_model(nothing, args)))

                    test_nlp(m0, m2; full = false)

                    for (sname, solver) in SOLVERS
                        (name, sname) in EXCLUDE1 && continue

                        result1 = solver(m0)
                        result2 = solver(m2)

                        @testset "$sname" begin
                            test_nlp_solution(result1, result2)
                        end
                    end
                end
            end

            @testset "Multi-column constraints" begin
                m3 = WrapperNLPModel(exa_luksan_vlcek_model(nothing, 3; M = 2))
                m4 = jump_luksan_vlcek_model(nothing, 3; M = 2)
                test_nlp(m3, m4; full = false)
            end
        end

        for backend in BACKENDS
            @testset "$backend" begin
                for (name, args) in NLP_TEST_ARGUMENTS
                    @testset "$name $args" begin

                        exa_model = getfield(@__MODULE__, Symbol("_exa_$(name)_model"))
                        jump_model = getfield(@__MODULE__, Symbol("_jump_$(name)_model"))

                        m0 = WrapperNLPModel(first(exa_model(nothing, args)))

                        jm, vars2, cons2 = jump_model(nothing, args)
                        set_optimizer(jm, MadNLP.Optimizer)
                        set_optimizer_attribute(jm, "print_level", MadNLP.ERROR)
                        optimize!(jm)

                        m, vars1, cons1 = exa_model(backend, args)
                        m1 = WrapperNLPModel(m)

                        @testset "Backend test" begin
                            test_nlp(m0, m1; full = true, tol = sol_tolerance(eltype(m0.meta.x0), eltype(m1.inner.meta.x0)))
                        end
                        @testset "API test" begin
                            result1 = madnlp(m1; print_level = MadNLP.ERROR, tol = solver_tolerance(eltype(m1.inner.meta.x0)))
                            test_api(result1, vars1, cons1, vars2, cons2, objective_sense(jm) == MIN_SENSE; tol = sol_tolerance(eltype(m1.inner.meta.x0)))
                        end
                    end
                end

                @testset "Parameter Test" begin
                    test_parametric_vs_nonparametric(backend)
                end

                @testset "Subexpr Test" begin
                    test_subexpr(backend)
                end

                @testset "Feature Test" begin
                    test_features(backend)
                end

                @testset "Constraint Augmentation" begin
                    test_conaug(backend)
                end
            end
        end
    end
end

function __init__()
    PowerModels.silence()
end


end # NLPTest
