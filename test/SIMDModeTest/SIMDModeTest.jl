module SIMDModeTest

using Test
import ExaModels
import MathOptInterface as MOI

function runtests()



    
    @testset "SIMDMode adapter" begin
        x, y = MOI.VariableIndex(1), MOI.VariableIndex(2)
        mode = ExaModels.SIMDMode()
        @test mode isa MOI.Nonlinear.AbstractAutomaticDifferentiation
        model = MOI.Nonlinear.model(mode)
        @test model isa MOI.Nonlinear.ModelWithOracles
        # Objective: x^2 (handled natively by ExaModels, no quad layer).
        MOI.Nonlinear.set_objective(
            model,
            MOI.ScalarQuadraticFunction(
                [MOI.ScalarQuadraticTerm(2.0, x, x)],
                MOI.ScalarAffineTerm{Float64}[],
                0.0,
            ),
        )
        # Constraints, in row order: the oracle layer's rows come first.
        oracle = MOI.VectorNonlinearOracle(;
            dimension = 1,
            l = [0.0],
            u = [1.0],
            eval_f = (ret, z) -> (ret[1] = z[1]^2),
            jacobian_structure = [(1, 1)],
            eval_jacobian = (ret, z) -> (ret[1] = 2.0 * z[1]),
            hessian_lagrangian_structure = [(1, 1)],
            eval_hessian_lagrangian = (ret, z, μ) -> (ret[1] = 2.0 * μ[1]),
        )
        MOI.Nonlinear.add_constraint(model, MOI.VectorOfVariables([x]), oracle)
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
        @test d isa MOI.Nonlinear.EvaluatorWithOracles
        @test MOI.features_available(d) == [:Grad, :Jac, :Hess]
        # Row queries work before MOI.initialize.
        @test MOI.Nonlinear.num_constraints(d) == 4
        @test MOI.Nonlinear.constraint_bounds(d) == [
            MOI.NLPBoundsPair(0.0, 1.0),
            MOI.NLPBoundsPair(-Inf, 4.0),
            MOI.NLPBoundsPair(0.0, 1.0),
            MOI.NLPBoundsPair(-Inf, 0.5),
        ]
        @test MOI.Nonlinear.constraint_linearity(d) == [
            MOI.Nonlinear.NONLINEAR,
            MOI.Nonlinear.LINEAR,
            MOI.Nonlinear.QUADRATIC,
            MOI.Nonlinear.NONLINEAR,
        ]
        @test MOI.Nonlinear.objective_linearity(d) == MOI.Nonlinear.QUADRATIC
        MOI.initialize(d, [:Grad, :Jac, :Hess])
        xv = [1.0, 2.0]
        @test MOI.eval_objective(d, xv) == 1.0
        grad = fill(NaN, 2)
        MOI.eval_objective_gradient(d, grad, xv)
        @test grad == [2.0, 0.0]
        g = fill(NaN, 4)
        MOI.eval_constraint(d, g, xv)
        @test g ≈ [1.0, 8.0, 5.0, sin(1.0)]
        J_structure = MOI.jacobian_structure(d)
        J_values = fill(NaN, length(J_structure))
        MOI.eval_constraint_jacobian(d, J_values, xv)
        J = zeros(4, 2)
        for ((row, col), value) in zip(J_structure, J_values)
            J[row, col] += value
        end
        @test J ≈ [
            2.0 0.0
            2.0 3.0
            4.0 2.0
            cos(1.0) 0.0
        ]
        H_structure = MOI.hessian_lagrangian_structure(d)
        σ, μ = 2.0, [10.0, 100.0, 1_000.0, 10_000.0]
        H_values = fill(NaN, length(H_structure))
        MOI.eval_hessian_lagrangian(d, H_values, xv, σ, μ)
        H = zeros(2, 2)
        for ((row, col), value) in zip(H_structure, H_values)
            H[row, col] += value
            if row != col
                H[col, row] += value
            end
        end
        @test H[1, 1] ≈ 2σ + 2 * μ[1] + 2 * μ[3] - sin(1.0) * μ[4]
        @test H[1, 2] ≈ μ[3]
        @test H[2, 2] ≈ 0.0
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
