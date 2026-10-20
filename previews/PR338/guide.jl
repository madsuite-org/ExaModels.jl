# # [Getting Started](@id guide)
# ExaModels can create nonlinear programming models and allows solving the created models using NLP solvers (in particular, those that are interfaced with `NLPModels`, such as [NLPModelsIpopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl) and [MadNLP](https://github.com/MadNLP/MadNLP.jl). This documentation page will describe how to use `ExaModels` to model and solve nonlinear optimization problems.

# We will first consider the following simple nonlinear program [lukvsan1998indefinitely](@cite):
# ```math
# \begin{aligned}
# \min_{\{x_i\}_{i=0}^N} &\sum_{i=2}^N  100(x_{i-1}^2-x_i)^2+(x_{i-1}-1)^2\\
# \text{s.t.} &  3x_{i+1}^3+2x_{i+2}-5+\sin(x_{i+1}-x_{i+2})\sin(x_{i+1}+x_{i+2})+4x_{i+1}-x_i e^{x_i-x_{i+1}}-3 = 0
# \end{aligned}
# ```
# We will follow the following Steps to create the model/solve this optimization problem.
# - Step 0: import ExaModels.jl
# - Step 1: create a [`ExaCore`](@ref) object, wherein we can progressively build an optimization model.
# - Step 2: create optimization variables with [`@add_var`](@ref) (or [`add_var`](@ref)), while attaching it to previously created `ExaCore`.
# - Step 3 (interchangable with Step 3): create objective function with [`@add_obj`](@ref) (or [`add_obj`](@ref)), while attaching it to previously created `ExaCore`.
# - Step 4 (interchangable with Step 2): create constraints with [`@add_con`](@ref) (or [`add_con`](@ref)), while attaching it to previously created `ExaCore`.
# - Step 5: create an [`ExaModel`](@ref) based on the `ExaCore`.

# Now, let's jump right in. We import ExaModels via (Step 0):
using ExaModels
# Now, all the functions that are necessary for creating model are imported to into `Main`.


# ## ExaCore
# An `ExaCore` object can be created simply by (Step 1):
c = ExaCore(concrete = Val(true))
# This is where our optimization model information will be progressively stored. This object is not yet an `NLPModel`, but it will essentially store all the necessary information.

# ## Variables
# Now, let's create the optimization variables. From the problem definition, we can see that we will need $N$ scalar variables. We will choose $N=10$, and create the variable $x\in\mathbb{R}^{N}$ with the following command:
N = 10
@add_var(c, x, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
# This creates the variable `x`, which we will be able to refer to when we create constraints/objective constraints. Also, this modifies the information in the `ExaCore` object properly so that later an optimization model can be properly created with the necessary information. Observe that we have used the keyword argument `start` to specify the initial guess for the solution. The variable upper and lower bounds can be specified in a similar manner. For example, if we wanted to set the lower bound of the variable `x` to 0.0 and the upper bound to 10.0, we could do it as follows:
# ```julia
# @add_var(c, x, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N), lvar = 0.0, uvar = 10.0)
# ```

# ## Objective
# The objective can be set as follows:
@add_obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
# !!! note
#     Note that the terms here are summed, without explicitly using `sum( ... )` syntax.

# ## Constraints
# The constraints can be set as follows:
@add_con(
    c,
    3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
    x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2)
)
# 
# Note that `ExaModels` always assume that the constraints are doubly-bounded inequalities. That is, the constraint above is treated as
# ```math
#  g^\flat \leq \left[g^{(m)}(x; q_j)\right]_{j\in [J_m]} +\sum_{n\in [N_m]}\sum_{k\in [K_n]}h^{(n)}(x; s^{(n)}_{k}) \leq g^\sharp
# ```
# where `g^\flat` and `g^\sharp` are the lower and upper bounds of the constraint, respectively. In this case, both bounds are zero, i.e., `g^\flat = g^\sharp = 0`.
#
# You can use the keyword arguments `lcon` and `ucon` to specify the lower and upper bounds of the constraints, respectively. For example, if we wanted to set the lower bound of the constraint to -1 and the upper bound to 1, we could do it as follows:
@add_con(
    c,
    3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
    x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2);
    lcon = -1.0, ucon = 1.0
)

# If you want to create a single-bounded constraint, you can set `lcon` to `-Inf` or `ucon` to `Inf`. For example, if we wanted to set the lower bound of the constraint to -1 and the upper bound to infinity, we could do it as follows:
@add_con(
    c,
    3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
    x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2);
    lcon = -1.0, ucon = Inf
)

# ## Subexpressions
# When complex expressions are reused across multiple objectives or constraints, you can define them as subexpressions using [`@add_expr`](@ref) (or [`add_expr`](@ref)). This improves code readability and avoids duplicating the expression tree.
#
# Subexpressions are **inlined**: when you index `s[i]`, the original expression is substituted directly — no auxiliary variables or equality constraints are created. Derivative code is shared across all uses of the same subexpression pattern.
#
# Here's a simple example:
c2 = ExaCore(concrete = Val(true))
@add_var(c2, y, N; start = 1.0)
# Define a subexpression for y[i]^2
@add_expr(c2, s, y[i]^2 for i in 1:N)
# Now s[i] can be used in objectives and constraints
@add_obj(c2, (s[i] - 1)^2 for i in 1:N)
@add_con(c2, s[i] + s[i + 1] for i in 1:(N - 1); lcon = 0.0)

# Multi-dimensional subexpressions are also supported with automatic dimension inference:
# ```julia
# @add_expr(c, dx, x[t, i] - x[t-1, i] for t in 1:T, i in 1:N)
# # dx[t, i] can now be used in constraints
# @add_con(c, dx[t, i] - something for t in 1:T, i in 1:N)
# ```
#
# For a comprehensive example using subexpressions, see the [Distillation Column example](@ref distillation).

# ## ExaModel
# Finally, we are ready to create an `ExaModel` from the data we have collected in `ExaCore`. Since `ExaCore` includes all the necessary information, we can do this simply by:
m = ExaModel(c)

# Now, we got an optimization model ready to be solved. This problem can be solved with for example, with the Ipopt solver, as follows.
using NLPModelsIpopt
result = ipopt(m)

# Here, `result` is an `AbstractExecutionStats`, which typically contains the solution information. We can check several information as follows.
println("Status: $(result.status)")
println("Number of iterations: $(result.iter)")

# ## Solutions
# The solution values for variable `x` can be inquired by:
sol = solution(result, x)
# This will return the primal solution of the variable `x` as a vector. Dual solutions can be inquired similarly, by using the `multipliers` function. 



# ExaModels provide several APIs similar to this:
# - [`solution`](@ref) inquires the primal solution.
# - [`multipliers`](@ref) inquires the dual solution.
# - [`multipliers_L`](@ref) inquires the lower bound dual solution.
# - [`multipliers_U`](@ref) inquires the upper bound dual solution.

# This concludes a short tutorial on how to use ExaModels to model and solve optimization problems. Want to learn more? Take a look at the following examples, which provide further tutorial on how to use ExaModels.jl. Each of the examples are designed to instruct a few additional techniques.
# - [Example: Quadrotor](@ref quad): modeling multiple types of objective values and constraints.
# - [Example: Distillation Column](@ref distillation): using subexpressions and two-dimensional index sets.
# - [Example: Optimal Power Flow](@ref opf): handling complex data and using constraint augmentation.
