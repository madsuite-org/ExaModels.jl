module SIMDModeTest

using Test
import ExaModels
import Ipopt
import MathOptInterface as MOI

function runtests()
    @testset "SIMDMode adapter" begin
        mode = ExaModels.SIMDMode()
        @test mode isa MOI.Nonlinear.AbstractAutomaticDifferentiation
        model = MOI.Nonlinear.model(mode)
        @test model isa MOI.ModelLike
        x, y = MOI.add_variables(model, 2)
        @test !MOI.supports_constraint(
            model,
            MOI.VectorOfVariables,
            MOI.VectorNonlinearOracle{Float64},
        )
        # Objective: x^2 (handled natively by ExaModels, no quad layer).
        MOI.Nonlinear.set_objective(
            model,
            MOI.ScalarQuadraticFunction(
                [MOI.ScalarQuadraticTerm(2.0, x, x)],
                MOI.ScalarAffineTerm{Float64}[],
                0.0,
            ),
        )
        MOI.Nonlinear.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(2.0, x), MOI.ScalarAffineTerm(3.0, y)],
                0.0,
            ),
            MOI.LessThan(4.0),
        )
        MOI.Nonlinear.add_constraint(
            model,
            MOI.ScalarQuadraticFunction(
                [
                    MOI.ScalarQuadraticTerm(2.0, x, x),
                    MOI.ScalarQuadraticTerm(1.0, x, y),
                ],
                [MOI.ScalarAffineTerm(1.0, y)],
                0.0,
            ),
            MOI.Interval(0.0, 1.0),
        )
        sin_x = MOI.ScalarNonlinearFunction(:sin, Any[x])
        MOI.Nonlinear.add_constraint(model, sin_x, MOI.LessThan(0.5))
        d = MOI.Nonlinear.Evaluator(model, mode, [x, y])
        @test d isa MOI.AbstractNLPEvaluator
        @test MOI.features_available(d) == [:Grad, :Jac, :JacVec, :Hess, :HessVec]
        # Row queries work before MOI.initialize.
        @test length(MOI.Nonlinear._constraint_bounds(d)) == 3
        @test MOI.Nonlinear._constraint_bounds(d) == [
            MOI.NLPBoundsPair(-Inf, 4.0),
            MOI.NLPBoundsPair(0.0, 1.0),
            MOI.NLPBoundsPair(-Inf, 0.5),
        ]
        MOI.initialize(d, [:Grad, :Jac, :Hess])
        xv = [1.0, 2.0]
        @test MOI.eval_objective(d, xv) == 1.0
        grad = fill(NaN, 2)
        MOI.eval_objective_gradient(d, grad, xv)
        @test grad == [2.0, 0.0]
        g = fill(NaN, 3)
        MOI.eval_constraint(d, g, xv)
        @test g ≈ [8.0, 5.0, sin(1.0)]
        J_structure = MOI.jacobian_structure(d)
        J_values = fill(NaN, length(J_structure))
        MOI.eval_constraint_jacobian(d, J_values, xv)
        J = zeros(3, 2)
        for ((row, col), value) in zip(J_structure, J_values)
            J[row, col] += value
        end
        @test J ≈ [
            2.0 3.0
            4.0 2.0
            cos(1.0) 0.0
        ]
        H_structure = MOI.hessian_lagrangian_structure(d)
        σ, μ = 2.0, [100.0, 1_000.0, 10_000.0]
        H_values = fill(NaN, length(H_structure))
        MOI.eval_hessian_lagrangian(d, H_values, xv, σ, μ)
        H = zeros(2, 2)
        for ((row, col), value) in zip(H_structure, H_values)
            H[row, col] += value
            if row != col
                H[col, row] += value
            end
        end
        @test H[1, 1] ≈ 2σ + 2 * μ[2] - sin(1.0) * μ[3]
        @test H[1, 2] ≈ μ[2]
        @test H[2, 2] ≈ 0.0
    end
    @testset "Ipopt with SIMDMode" begin
        model = Ipopt.Optimizer()
        MOI.set(model, MOI.Silent(), true)
        mode = ExaModels.SIMDMode()
        MOI.set(model, MOI.AutomaticDifferentiationBackend(), mode)
        @test MOI.get(model, MOI.AutomaticDifferentiationBackend()) === mode
        x, y = MOI.add_variables(model, 2)
        MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
        MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
        objective = MOI.ScalarQuadraticFunction(
            [
                MOI.ScalarQuadraticTerm(2.0, x, x),
                MOI.ScalarQuadraticTerm(2.0, y, y),
            ],
            [
                MOI.ScalarAffineTerm(-2.0, x),
                MOI.ScalarAffineTerm(-4.0, y),
            ],
            5.0,
        )
        MOI.set(model, MOI.ObjectiveFunction{typeof(objective)}(), objective)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        constraint = MOI.ScalarNonlinearFunction(:+, Any[x, y])
        MOI.add_constraint(model, constraint, MOI.GreaterThan(3.0))
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 1.0 atol = 1e-4
        @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 2.0 atol = 1e-4
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 0.0 atol = 1e-8
    end
    @testset "SIMDMode requires identity variable order" begin
        x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
        mode = ExaModels.SIMDMode()
        model = MOI.Nonlinear.model(mode)
        MOI.Nonlinear.add_constraint(
            model,
            MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
            MOI.LessThan(1.0),
        )
        d = MOI.Nonlinear.Evaluator(model, mode, [y, x])
        @test_throws(
            ErrorException(
                "`ExaModels.SIMDMode` requires the variables of the model " *
                "to be `MOI.VariableIndex.(1:n)`, in order.",
            ),
            MOI.initialize(d, [:Grad, :Jac]),
        )
    end
    return
end

end # module
