# # [Example: Distillation Column](@id distillation)
#
# This example demonstrates the use of [`@add_expr`](@ref) to define reusable subexpressions
# that simplify complex models.  We compare two formulations of a distillation column:
# one written by hand and one that uses named subexpressions for readability.
#
# ## Subexpressions in ExaModels
#
# The [`@add_expr`](@ref) macro (or [`add_expr`](@ref)) creates an **inlined subexpression**:
# a named expression template that is substituted directly wherever it is indexed.
# No auxiliary variables or extra constraints are added to the problem.
#
# ```julia
# @add_expr(c, s, x[i]^2 for i in 1:n)   # s[i] expands to x[i]^2 at each use site
# ```
#
# The benefit is purely notational: repeated sub-expressions like finite-difference
# stencils can be named once and reused across many constraints, keeping model code concise
# without changing the NLP structure.

# ## Original Model (without subexpressions)
# This is the original formulation where expressions like `(xA[t, i] - xA[t-1, i]) / dt`
# are repeated in multiple constraints.

using ExaModels, NLPModelsIpopt

function distillation_column_model(T = 3; backend = nothing)

    NT = 30
    FT = 17
    Ac = 0.5
    At = 0.25
    Ar = 1.0
    D = 0.2
    F = 0.4
    ybar = 0.8958
    ubar = 2.0
    alpha = 1.6
    dt = 10 / T
    xAf = 0.5
    xA0s = ExaModels.convert_array([(i, 0.5) for i = 0:(NT+1)], backend)

    itr0 = ExaModels.convert_array(collect(Iterators.product(1:T, 1:(FT-1))), backend)
    itr1 = ExaModels.convert_array(collect(Iterators.product(1:T, (FT+1):NT)), backend)
    itr2 = ExaModels.convert_array(collect(Iterators.product(0:T, 0:(NT+1))), backend)

    c = ExaCore(; backend, concrete = Val(true))

    @add_var(c, xA, 0:T, 0:(NT+1); start = 0.5)
    @add_var(c, yA, 0:T, 0:(NT+1); start = 0.5)
    @add_var(c, u, 0:T; start = 1.0)
    @add_var(c, V, 0:T; start = 1.0)
    @add_var(c, L2, 0:T; start = 1.0)

    @add_obj(c, (yA[t, 1] - ybar)^2 for t = 0:T)
    @add_obj(c, (u[t] - ubar)^2 for t = 0:T)

    @add_con(c, xA[0, i] - xA0 for (i, xA0) in xA0s)
    @add_con(
        c,
        (xA[t, 0] - xA[t-1, 0]) / dt - (1 / Ac) * (yA[t, 1] - xA[t, 0]) for t = 1:T
    )
    @add_con(
        c,
        (xA[t, i] - xA[t-1, i]) / dt -
        (1 / At) * (u[t] * D * (yA[t, i-1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i+1])) for
        (t, i) in itr0
    )
    @add_con(
        c,
        (xA[t, FT] - xA[t-1, FT]) / dt -
        (1 / At) * (
            F * xAf + u[t] * D * xA[t, FT-1] - L2[t] * xA[t, FT] -
            V[t] * (yA[t, FT] - yA[t, FT+1])
        ) for t = 1:T
    )
    @add_con(
        c,
        (xA[t, i] - xA[t-1, i]) / dt -
        (1 / At) * (L2[t] * (yA[t, i-1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i+1])) for
        (t, i) in itr1
    )
    @add_con(
        c,
        (xA[t, NT+1] - xA[t-1, NT+1]) / dt -
        (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT+1] - V[t] * yA[t, NT+1]) for
        t = 1:T
    )
    @add_con(c, V[t] - u[t] * D - D for t = 0:T)
    @add_con(c, L2[t] - u[t] * D - F for t = 0:T)
    @add_con(
        c,
        yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) for (t, i) in itr2
    )

    return ExaModel(c)
end

# ## Model with Lifted Subexpressions
# Uses subexpressions for time derivatives and vapor differences.
# This adds auxiliary variables and constraints but makes the model more readable.

function distillation_column_model_with_subexpr(T = 3; backend = nothing)

    NT = 30
    FT = 17
    Ac = 0.5
    At = 0.25
    Ar = 1.0
    D = 0.2
    F = 0.4
    ybar = 0.8958
    ubar = 2.0
    alpha = 1.6
    dt = 10 / T
    xAf = 0.5
    xA0s = ExaModels.convert_array([(i, 0.5) for i in 0:(NT + 1)], backend)

    c = ExaCore(; backend, concrete = Val(true))

    ## Decision variables
    @add_var(c, xA, 0:T, 0:(NT + 1); start = 0.5)
    @add_var(c, yA, 0:T, 0:(NT + 1); start = 0.5)
    @add_var(c, u, 0:T; start = 1.0)
    @add_var(c, V, 0:T; start = 1.0)
    @add_var(c, L2, 0:T; start = 1.0)

    ## Subexpressions - define common terms once
    @add_expr(c, dxA, (xA[t, i] - xA[t - 1, i]) / dt for t in 1:T, i in 0:(NT + 1))
    @add_expr(c, dyA, yA[t, i] - yA[t, i + 1] for t in 0:T, i in 0:NT)

    ## Objectives
    @add_obj(c, (yA[t, 1] - ybar)^2 for t in 0:T)
    @add_obj(c, (u[t] - ubar)^2 for t in 0:T)

    ## Initial conditions
    @add_con(c, xA[0, i] - xA0 for (i, xA0) in xA0s)

    ## Condenser - now using dxA subexpression
    @add_con(c, dxA[t, 0] - (1 / Ac) * (yA[t, 1] - xA[t, 0]) for t in 1:T)

    ## Rectifying section - cleaner with dxA and dyA
    itr_rect = ExaModels.convert_array(collect(Iterators.product(1:T, 1:(FT - 1))), backend)
    @add_con(c, dxA[t, i] - (1 / At) * (u[t] * D * (yA[t, i - 1] - xA[t, i]) - V[t] * dyA[t, i]) for (t, i) in itr_rect)

    ## Feed tray
    @add_con(c, dxA[t, FT] - (1 / At) * (F * xAf + u[t] * D * xA[t, FT - 1] - L2[t] * xA[t, FT] - V[t] * dyA[t, FT]) for t in 1:T)

    ## Stripping section
    itr_strip = ExaModels.convert_array(collect(Iterators.product(1:T, (FT + 1):NT)), backend)
    @add_con(c, dxA[t, i] - (1 / At) * (L2[t] * (yA[t, i - 1] - xA[t, i]) - V[t] * dyA[t, i]) for (t, i) in itr_strip)

    ## Reboiler
    @add_con(c, dxA[t, NT + 1] - (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT + 1] - V[t] * yA[t, NT + 1]) for t in 1:T)

    ## Flow relationships
    @add_con(c, V[t] - u[t] * D - D for t in 0:T)
    @add_con(c, L2[t] - u[t] * D - F for t in 0:T)

    ## VLE
    itr_vle = ExaModels.convert_array(collect(Iterators.product(0:T, 0:(NT + 1))), backend)
    @add_con(c, yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) for (t, i) in itr_vle)

    return ExaModel(c)
end

#-

# ## Running the Models
#
# Let's compare both formulations and verify they converge to the same solution.

T_val = 10

# Without subexpressions:
m_orig = distillation_column_model(T_val)
result_orig = ipopt(m_orig; print_level = 0)

#-

# With subexpressions:
m_subexpr = distillation_column_model_with_subexpr(T_val)
result_subexpr = ipopt(m_subexpr; print_level = 0)

#-

# ## Comparison Results

println("="^62)
println("Distillation Column Model Comparison (T=$T_val)")
println("="^62)
println()
println("| Model               | Variables | Constraints | Iterations | Objective |")
println("|---------------------|-----------|-------------|------------|-----------|")
println("| Original            | $(lpad(m_orig.meta.nvar, 9)) | $(lpad(m_orig.meta.ncon, 11)) | $(lpad(result_orig.iter, 10)) | $(round(result_orig.objective, digits = 6)) |")
println("| With subexpressions | $(lpad(m_subexpr.meta.nvar, 9)) | $(lpad(m_subexpr.meta.ncon, 11)) | $(lpad(result_subexpr.iter, 10)) | $(round(result_subexpr.objective, digits = 6)) |")
println()

#-

# Both formulations are equivalent NLPs and converge to the same solution:

println("Same objective: ", isapprox(result_orig.objective, result_subexpr.objective, rtol = 1.0e-4))
