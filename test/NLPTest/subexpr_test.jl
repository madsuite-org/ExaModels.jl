# Test subexpression functionality

"""
Test basic 1D subexpression creation and solving.
Compares a model with subexpressions to an equivalent model without.
"""

function test_subexpr_concrete_index_constant_term(backend)
    # A standalone scalar term in the body (`i - 1`) plus a CONCRETE index
    # into the expression: substitution then hands the stored `-` node two
    # Real children. The node storage that replaced the body closure must
    # fold that case eagerly, as the closure did — left as a node it has no
    # evaluation method (the one-sided Real methods tie), and the failure
    # surfaced at the ADJOINT call, so the gradient is the assertion here.
    c = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c, x, 4; start = [1.0, 2.0, 3.0, 4.0])
    @add_expr(c, s, x[i] + (i - 1) for i in 1:4)
    @add_obj(c, (s[2] + x[i])^2 for i in 1:4)
    m = ExaModel(c)
    x0 = Array(m.meta.x0)                      # host copy for reference math
    want = sum((x0[2] + 1 + x0[i])^2 for i in 1:4)
    @test NLPModels.obj(m, m.meta.x0) ≈ want
    g = NLPModels.grad(m, m.meta.x0)
    gwant = [2 * (x0[2] + 1 + x0[i]) for i in 1:4]
    gwant[2] += sum(2 * (x0[2] + 1 + x0[i]) for i in 1:4)
    @test Array(g) ≈ gwant
end

function test_subexpr_basic(backend)
    # Model WITHOUT subexpressions
    c1 = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c1, x1, 10; start = 1.0)
    # Objective: sum of (x[i]^2 + x[i+1]^2)^2
    @add_obj(c1, (x1[i]^2 + x1[i + 1]^2)^2 for i in 1:9)
    # Constraint: x[i]^2 - 1 >= 0
    @add_con(c1, x1[i]^2 - 1 for i in 1:10; lcon = 0.0)
    m1 = ExaModel(c1)

    # Model WITH subexpressions
    c2 = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c2, x2, 10; start = 1.0)
    # Create subexpression for x[i]^2
    @add_expr(c2, s, x2[i]^2 for i in 1:10)
    # Objective using subexpression
    @add_obj(c2, (s[i] + s[i + 1])^2 for i in 1:9)
    # Constraint using subexpression
    @add_con(c2, s[i] - 1 for i in 1:10; lcon = 0.0)
    m2 = ExaModel(c2)

    # The model with subexpressions has more variables and constraints
    @test m2.meta.nvar == m1.meta.nvar + 10  # 10 auxiliary variables
    @test m2.meta.ncon == m1.meta.ncon + 10  # 10 defining constraints

    # Solve both models (wrap in WrapperNLPModel for GPU compatibility with Ipopt)
    result1 = NLPModelsIpopt.ipopt(WrapperNLPModel(m1); print_level = 0, tol = solver_tolerance(eltype(c1.x0)))
    result2 = NLPModelsIpopt.ipopt(WrapperNLPModel(m2); print_level = 0, tol = solver_tolerance(eltype(c2.x0)))

    # Solutions for original variables should match
    @test result1.status == result2.status
    @test solution(result1, x1) ≈ solution(result2, x2) atol = sol_tolerance(eltype(c1.x0)) rtol = sol_tolerance(eltype(c1.x0))

    # Subexpression values should equal x^2
    subexpr_vals = solution(result2, s)
    x_vals = solution(result2, x2)
    return @test subexpr_vals ≈ x_vals .^ 2 atol = sol_tolerance(eltype(c1.x0)) rtol = sol_tolerance(eltype(c1.x0))
end
 
"""
Test multi-dimensional subexpressions with automatic dimension inference.
"""
function test_subexpr_multidim(backend)
    T, N = 5, 4

    c = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c, x, 0:T, 0:N; start = 0.5)

    # Create 2D subexpression with Cartesian product - dimensions inferred automatically
    @add_expr(c, dx, x[t, i] - x[t - 1, i] for t in 1:T, i in 1:N)

    # Test that subexpression has correct dimensions (inferred from Cartesian product)
    @test dx.size == (1:T, 1:N)
    @test dx.length == T * N

    # Use in objective
    @add_obj(c, dx[t, i]^2 for t in 1:T, i in 1:N)

    # Add some constraints to make it non-trivial
    @add_con(c, x[0, i] - 0.0 for i in 0:N)  # Initial condition
    @add_con(c, x[T, i] - 1.0 for i in 0:N)  # Final condition

    m = ExaModel(c)

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0, tol = solver_tolerance(eltype(c.x0)))
    @test result.status == :first_order

    # Check subexpression values match the definition
    x_sol = solution(result, x)
    dx_sol = solution(result, dx)

    for t in 1:T, i in 1:N
        expected = x_sol[t + 1, i + 1] - x_sol[t, i + 1]  # +1 for 0-based to 1-based
        @test dx_sol[t, i] ≈ expected atol = sol_tolerance(eltype(c.x0)) rtol = sol_tolerance(eltype(c.x0))
    end
    return
end

"""
Test subexpression with automatic dimension inference from Cartesian product.
"""
function test_subexpr_auto_dims(backend)
    T, N = 3, 2

    c = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c, x, 1:T, 1:N; start = 1.0)

    # Subexpr with Cartesian product syntax - dimensions inferred automatically
    @add_expr(c, s, x[t, i]^2 + t * i for t in 1:T, i in 1:N)

    # Check dimensions were inferred
    @test s.size == (1:T, 1:N)
    @test s.length == T * N

    # Use in objective
    @add_obj(c, s[t, i] for t in 1:T, i in 1:N)

    m = ExaModel(c)

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0, tol = solver_tolerance(eltype(c.x0)))
    return @test result.status == :first_order
end

"""
Test that subexpressions can be used in both objectives and constraints.
"""
function test_subexpr_in_obj_and_con(backend)
    c = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c, x, 5; start = 2.0, lvar = 0.0)

    # Subexpression
    @add_expr(c, s, sqrt(x[i]) for i in 1:5)

    # Use in objective
    @add_obj(c, (s[i] - 1)^2 for i in 1:5)

    # Use in constraint
    @add_con(c, s[i] + s[i + 1] for i in 1:4; lcon = 1.0, ucon = 3.0)

    m = ExaModel(c)

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0, tol = solver_tolerance(eltype(c.x0)))
    @test result.status == :first_order

    # sqrt(x) = 1 at optimum, so x = 1
    @test solution(result, x) ≈ ones(5) atol = sol_tolerance(eltype(c.x0)) rtol = sol_tolerance(eltype(c.x0))
    return @test solution(result, s) ≈ ones(5) atol = sol_tolerance(eltype(c.x0)) rtol = sol_tolerance(eltype(c.x0))
end

# """
# Test that lifted subexpressions inherit start values from main variables.
# The auxiliary variables should get start values computed from the expression
# evaluated at the main variables' start values.
# """
# function test_subexpr_lifted_start_values(backend)
#     c = ExaCore(; backend = backend, concrete = Val(true))

#     # Create variables with specific start values
#     @add_var(c, x, 5; start = 3.0)

#     # Create lifted subexpression s[i] = x[i]^2
#     # With x start = 3.0, the subexpr auxiliary vars should start at 9.0
#     @add_expr(c, s, x[i]^2 for i in 1:5)

#     # Check that the subexpression auxiliary variables have computed start values
#     # The start values are stored in c.x0 at the subexpr's offset
#     start_vals = c.x0[(s.offset+1):(s.offset+s.length)]
#     @test all(Array(start_vals) .≈ 9.0)

#     # Also test with parameters
#     c2 = ExaCore(; backend = backend, concrete = Val(true))
#     @add_par(c2, θ, [1.0, 2.0, 3.0])
#     @add_var(c2, x2, 3; start = 2.0)

#     # Subexpression uses both x and θ: s[i] = x[i] * θ[i]
#     # With x start = 2.0 and θ = [1,2,3], expect start = [2,4,6]
#     @add_expr(c2, s2, x2[i] * θ[i] for i in 1:3)

#     start_vals2 = c2.x0[(s2.offset+1):(s2.offset+s2.length)]
#     @test Array(start_vals2) ≈ [2.0, 4.0, 6.0]

#     # Test that the model solves correctly with these start values
#     @add_obj(c2, (x2[i] - s2[i])^2 for i in 1:3)
#     @add_con(c2, x2[i] - 1 for i in 1:3; lcon = 0.0, ucon = Inf)

#     m = ExaModel(c2)
#     result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0, tol = solver_tolerance(eltype(c2.x0)))
#     return @test result.status == :first_order
# end

"""
Test reduced subexpressions (no extra variables/constraints).
"""
function test_subexpr_reduced_basic(backend)
    # Model WITHOUT subexpressions
    c1 = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c1, x1, 10; start = 1.0)
    @add_obj(c1, (x1[i]^2 + x1[i + 1]^2)^2 for i in 1:9)
    @add_con(c1, x1[i]^2 - 1 for i in 1:10; lcon = 0.0)
    m1 = ExaModel(c1)

    # Model WITH reduced subexpressions
    c2 = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c2, x2, 10; start = 1.0)
    @add_expr(c2, s, x2[i]^2 for i in 1:10)
    @add_obj(c2, (s[i] + s[i + 1])^2 for i in 1:9)
    @add_con(c2, s[i] - 1 for i in 1:10; lcon = 0.0)
    m2 = ExaModel(c2)

    # Reduced subexpressions should NOT add variables or constraints
    @test m2.meta.nvar == m1.meta.nvar
    @test m2.meta.ncon == m1.meta.ncon

    # Solve both models (wrap in WrapperNLPModel for GPU compatibility with Ipopt)
    result1 = NLPModelsIpopt.ipopt(WrapperNLPModel(m1); print_level = 0, tol = solver_tolerance(eltype(c1.x0)))
    result2 = NLPModelsIpopt.ipopt(WrapperNLPModel(m2); print_level = 0, tol = solver_tolerance(eltype(c2.x0)))

    # Solutions should match
    @test result1.status == result2.status
    return @test solution(result1, x1) ≈ solution(result2, x2) atol = sol_tolerance(eltype(c1.x0)) rtol = sol_tolerance(eltype(c1.x0))
end

# """
# Test that reduced and lifted subexpressions produce equivalent solutions.
# """
# function test_subexpr_lifted_vs_reduced(backend)
#     # Model with LIFTED subexpressions
#     c1 = ExaCore(; backend = backend, concrete = Val(true))
#     @add_var(c1, x1, 5; start = 2.0, lvar = 0.0)
#     @add_expr(c1, s1, sqrt(x1[i]) for i in 1:5)  # lifted (default)
#     @add_obj(c1, (s1[i] - 1)^2 for i in 1:5)
#     @add_con(c1, s1[i] + s1[i + 1] for i in 1:4; lcon = 1.0, ucon = 3.0)
#     m1 = ExaModel(c1)

#     # Model with REDUCED subexpressions
#     c2 = ExaCore(; backend = backend, concrete = Val(true))
#     @add_var(c2, x2, 5; start = 2.0, lvar = 0.0)
#     @add_expr(c2, s2, sqrt(x2[i]) for i in 1:5)
#     @add_obj(c2, (s2[i] - 1)^2 for i in 1:5)
#     @add_con(c2, s2[i] + s2[i + 1] for i in 1:4; lcon = 1.0, ucon = 3.0)
#     m2 = ExaModel(c2)

#     # Lifted has more vars/cons
#     @test m1.meta.nvar > m2.meta.nvar
#     @test m1.meta.ncon > m2.meta.ncon

#     # Solve both (wrap in WrapperNLPModel for GPU compatibility with Ipopt)
#     result1 = NLPModelsIpopt.ipopt(WrapperNLPModel(m1); print_level = 0, tol = solver_tolerance(eltype(c1.x0)))
#     result2 = NLPModelsIpopt.ipopt(WrapperNLPModel(m2); print_level = 0, tol = solver_tolerance(eltype(c2.x0)))

#     # Both should converge to same solution
#     @test result1.status == :first_order
#     @test result2.status == :first_order
#     return @test solution(result1, x1) ≈ solution(result2, x2) atol = sol_tolerance(eltype(c1.x0)) rtol = sol_tolerance(eltype(c1.x0))
# end

"""
Test multi-dimensional reduced subexpressions.
"""
function test_subexpr_reduced_multidim(backend)
    T, N = 3, 2

    c = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c, x, 0:T, 0:N; start = 0.5)

    # Reduced 2D subexpression
    @add_expr(c, dx, x[t, i] - x[t - 1, i] for t in 1:T, i in 1:N)

    # Check dimensions
    @test dx.size == (1:T, 1:N)
    @test dx.length == T * N

    # No extra vars/cons from reduced subexpr
    nvar_before = c.nvar
    ncon_before = c.ncon

    # Use in objective
    @add_obj(c, dx[t, i]^2 for t in 1:T, i in 1:N)

    # Add constraints
    @add_con(c, x[0, i] - 0.0 for i in 0:N)
    @add_con(c, x[T, i] - 1.0 for i in 0:N)

    m = ExaModel(c)

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0, tol = solver_tolerance(eltype(c.x0)))
    return @test result.status == :first_order
end

"""
Test nested reduced subexpressions.
"""
function test_subexpr_reduced_nested(backend)
    c = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c, x, 5; start = 1.0, lvar = 0.1)

    # Nested reduced subexpressions
    @add_expr(c, s1, x[i]^2 for i in 1:5)
    @add_expr(c, s2, s1[i] + s1[i] for i in 1:5)  # 2*x[i]^2

    @add_obj(c, (s2[i] - 2)^2 for i in 1:5)  # minimize (2*x^2 - 2)^2

    m = ExaModel(c)

    # No extra vars/cons from reduced subexprs
    @test m.meta.nvar == 5
    @test m.meta.ncon == 0

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0, tol = solver_tolerance(eltype(c.x0)))
    @test result.status == :first_order

    # 2*x^2 = 2 => x = 1
    return @test solution(result, x) ≈ ones(5) atol = sol_tolerance(eltype(c.x0)) rtol = sol_tolerance(eltype(c.x0))
end

# """
# Test mixed lifted and reduced subexpressions.
# """
# function test_subexpr_mixed(backend)
#     c = ExaCore(; backend = backend, concrete = Val(true))
#     @add_var(c, x, 5; start = 1.0, lvar = 0.1)

#     # First subexpr is lifted
#     @add_expr(c, s_lifted, x[i]^2 for i in 1:5)  # adds 5 vars + 5 cons

#     # Second subexpr is reduced, uses the lifted one
#     @add_expr(c, s_reduced, s_lifted[i] * 2 for i in 1:5)

#     @add_obj(c, (s_reduced[i] - 2)^2 for i in 1:5)

#     m = ExaModel(c)

#     # Only lifted subexpr adds vars/cons
#     @test m.meta.nvar == 10  # 5 original + 5 lifted
#     @test m.meta.ncon == 5   # 5 defining constraints

#     # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
#     result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0, tol = solver_tolerance(eltype(c.x0)))
#     @test result.status == :first_order

#     # 2*x^2 = 2 => x = 1
#     return @test solution(result, x) ≈ ones(5) atol = sol_tolerance(eltype(c.x0)) rtol = sol_tolerance(eltype(c.x0))
# end

"""
Test reduced subexpressions with 0-based ranges (like in distillation example).
This tests the symbolic indexing when ranges don't start at 1.
"""
function test_subexpr_reduced_0based(backend)
    T = 3

    c = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c, u, 0:T; start = 1.0)

    # Reduced subexpressions with 0-based ranges (like distillation column)
    # V[t] should equal u[t] * 2 + 1, NOT u[t-1] * 2 + 1
    @add_expr(c, V, u[t] * 2 + 1 for t in 0:T)

    # Objective: minimize sum of (V[t] - 3)^2
    # If V[t] = u[t]*2+1, optimal is u[t] = 1 (V[t] = 3)
    @add_obj(c, (V[t] - 3)^2 for t in 0:T)

    m = ExaModel(c)

    # No extra vars from reduced subexpr
    @test m.meta.nvar == T + 1  # just u[0:T]
    @test m.meta.ncon == 0

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0, tol = solver_tolerance(eltype(c.x0)))
    @test result.status == :first_order

    # V[t] = u[t]*2+1 = 3 => u[t] = 1
    return @test solution(result, u) ≈ ones(T + 1) atol = sol_tolerance(eltype(c.x0)) rtol = sol_tolerance(eltype(c.x0))
end

"""
Test nested reduced subexpressions with 0-based ranges (like distillation VdyA).
"""
function test_subexpr_reduced_0based_nested(backend)
    T, N = 2, 2

    c = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c, x, 0:T, 0:N; start = 1.0)

    # First reduced subexpr with 0-based range
    @add_expr(c, s1, x[t, i] + 1 for t in 0:T, i in 0:N)

    # Second reduced subexpr uses the first one
    # s2[t,i] should be s1[t,i] * 2 = (x[t,i] + 1) * 2
    @add_expr(c, s2, s1[t, i] * 2 for t in 0:T, i in 0:N)

    # Objective: minimize sum of (s2[t,i] - 4)^2
    # s2 = (x+1)*2 = 4 => x = 1
    @add_obj(c, (s2[t, i] - 4)^2 for t in 0:T, i in 0:N)

    m = ExaModel(c)

    # No extra vars from reduced subexprs
    @test m.meta.nvar == (T + 1) * (N + 1)
    @test m.meta.ncon == 0

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0, tol = solver_tolerance(eltype(c.x0)))
    @test result.status == :first_order

    # (x+1)*2 = 4 => x = 1
    return @test solution(result, x) ≈ ones(T + 1, N + 1) atol = sol_tolerance(eltype(c.x0)) rtol = sol_tolerance(eltype(c.x0))
end

"""
Symbolic indexing of a subexpression whose iterator is not its own index set
(#293): `y[i]` must not silently mean `f(i)` instead of `f(p[i])`.
"""
function test_subexpr_permuted_iterator(backend)
    p = [3, 1, 2]
    x0 = [1.0, 2.0, 3.0]
    want = sum(i * x0[p[i]] for i in 1:3)

    c = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c, x, 3; start = x0)
    @add_expr(c, y, x[p_i] for p_i in p)
    @test_throws ArgumentError add_obj(c, i * y[i] for i in 1:3)
    # concrete indices are positional and stay supported
    @add_obj(c, y[1] + 2 * y[2] + 3 * y[3] for _ in 1:1)
    m = ExaModel(c)
    @test NLPModels.obj(m, m.meta.x0) ≈ want

    c = ExaCore(; backend = backend, concrete = Val(true))
    @add_var(c, x, 3; start = x0)
    @add_expr(c, y, x[p_i] for p_i in p; buffered = true)
    @add_obj(c, i * y[i] for i in 1:3)
    m = ExaModel(c)
    return @test NLPModels.obj(m, m.meta.x0) ≈ want
end

"""
Run all subexpression tests.
"""
function test_subexpr(backend)
    # @testset "Subexpr basic (lifted)" begin
    #     test_subexpr_basic(backend)
    # end

    # @testset "Subexpr multi-dim (lifted)" begin
    #     test_subexpr_multidim(backend)
    # end

    # @testset "Subexpr auto dims (lifted)" begin
    #     test_subexpr_auto_dims(backend)
    # end

    # @testset "Subexpr in obj and con (lifted)" begin
    #     test_subexpr_in_obj_and_con(backend)
    # end

    # @testset "Subexpr lifted start values" begin
    #     test_subexpr_lifted_start_values(backend)
    # end

    @testset "Subexpr reduced basic" begin
        test_subexpr_reduced_basic(backend)
    end

    # @testset "Subexpr lifted vs reduced" begin
    #     test_subexpr_lifted_vs_reduced(backend)
    # end

    @testset "Subexpr reduced multi-dim" begin
        test_subexpr_reduced_multidim(backend)
    end

    @testset "Subexpr reduced nested" begin
        test_subexpr_reduced_nested(backend)
    end

    # @testset "Subexpr mixed lifted and reduced" begin
    #     test_subexpr_mixed(backend)
    # end

    @testset "Subexpr reduced 0-based ranges" begin
        test_subexpr_reduced_0based(backend)
    end

    @testset "Subexpr reduced 0-based nested" begin
        test_subexpr_reduced_0based_nested(backend)
    end

    @testset "Subexpr concrete index with constant term" begin
        test_subexpr_concrete_index_constant_term(backend)
    end

    @testset "Subexpr permuted iterator" begin
        test_subexpr_permuted_iterator(backend)
    end

end

