# # [Two-Stage Optimization](@id two_stage)
# ExaModels supports two-stage optimization problems. This feature enables efficient modeling of optimization problems where decisions are made in two stages:
#
# - **Design (first-stage) variables**: Decisions made before uncertainty is revealed, shared across all scenarios
# - **Recourse (second-stage) variables**: Scenario-specific decisions made after uncertainty is revealed
# - **Scenarios**: Each scenario has its own parameters θ that affect the objective and constraints

using ExaModels, NLPModelsIpopt

# Define the problem dimensions and scenario parameters:
ns = 3   ## number of scenarios
nv = 2   ## recourse variables per scenario
nd = 1   ## design variables
weight = 1.0 / ns

# To annotate the scenario for each variable and constraint, we need to start with a special
# ExaCore that supports such scenario annotations. Pass `ns` as the first argument to set
# the number of scenarios.
core = TwoStageExaCore(ns, concrete = Val(true))

# Now we can define the first-stage (design) variable `d`, shared across all scenarios.
# These are added without `EachScenario()`.
@add_var(core, d, nd; start = 1.0, lvar = 0.0, uvar = Inf)

# Second-stage (recourse) variables are added with `EachScenario()` as the first argument
# after the core. This creates `nv * ns` variables in total — one block of `nv` per scenario.
# Variables for scenario `s` occupy flat indices `(s-1)*nv+1 : s*nv`.
v = @add_var(core, EachScenario(), nv; start = 1.0, lvar = 0.0, uvar = Inf)

# Second-stage constraints are also added with `EachScenario()`. Here we add the constraint
# v[s,1] - v[s,2]^2 = 0 for each scenario s, using flat variable indices.
con_data = [(s, (s - 1) * nv + 1, (s - 1) * nv + 2) for s in 1:ns]
@add_con(core, EachScenario(), (v[i1] - v[i2]^2 for (s, i1, i2) in con_data); lcon = 0.0, ucon = 0.0)

# The objective can mix first- and second-stage variables. Here we minimize
# d[1]^2 + weight * sum_{s,i} (v[s,i] - d[1])^2.
@add_obj(core, d[1]^2)
obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
@add_obj(core, weight * (v[vidx] - d[1])^2 for (i, j, vidx) in obj_data)

m = ExaModel(core)

# Now we can solve the model as usual.
ipopt(m)
# If the solver knows how to exploit the scenario structure, the structure-exploiting method can be used.
