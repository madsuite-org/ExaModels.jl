"""
    ExaModels

An algebraic modeling and automatic differentiation tool in Julia, specialized
for SIMD-parallel evaluation of nonlinear programs on CPUs and GPUs.

## Computation graph

Expressions are represented as trees of [`AbstractNode`](@ref) subtypes:

- [`Var`](@ref) / [`DataIndexed`](@ref) — decision variables and data fields
- [`Node1`](@ref) / [`Node2`](@ref) — unary / binary operations
- [`Constant{T}`](@ref) — compile-time scalar constant; the value `T` is
  stored as a **type parameter** so it is visible to the compiler and to
  `juliac --trim=safe` without any runtime storage.
- [`SumNode`](@ref) / [`ProdNode`](@ref) — reduction over a tuple of nodes

## Algebraic simplification

Operations involving `Constant{T}` are simplified at model-construction time:
`x * Constant(1) → x`, `x ^ Constant(2) → abs2(x)`, etc. (see
`specialization.jl`).  Combined `Constant OP Constant` expressions are folded
to a new `Constant` via the rules in `register.jl`.

For more information, please visit https://github.com/madsuite-org/ExaModels.jl
"""
module ExaModels

import Adapt: adapt
import NLPModels:
    NLPModels,
    obj,
    cons!,
    grad!,
    jac_coord!,
    hess_coord!,
    jprod!,
    jtprod!,
    hprod!,
    jac_structure!,
    hess_structure!,
    cons_nln!,
    jtprod_nln!,
    jprod_nln!
import SolverCore
import Printf

include("templates.jl")
include("argument.jl")
include("graph.jl")
include("register.jl")
include("specialization.jl")
include("functionlist.jl")
include("simdfunction.jl")
include("prettyprint.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")
include("nlp.jl")
include("oracle.jl")
include("utils.jl")
include("tags.jl")
include("two_stage.jl")
include("wrapper.jl")

export ExaModel,
    ExaCore,
    Expression,
    add_var,
    add_par,
    add_con,
    add_con!,
    add_obj,
    add_expr,
    @add_var,
    @add_expr,
    @add_par,
    @add_obj,
    @add_con,
    @add_con!,
    get_vars,
    get_cons,
    get_pars,
    solution,
    multipliers,
    multipliers_L,
    multipliers_U,
    Constant,
    ArgSource,
    instantiate,
    SumNode,
    ProdNode,
    exa_sum,
    exa_prod,
    @register_univariate,
    @register_bivariate,
    VectorNonlinearOracle,
    OracleEvaluator,
    objective,
    constraint,
    ExaModelWithOracle,
    has_matfree_jac,
    has_matfree_hess,
    embed_oracle,
    add_eval,
    ScalarNonlinearOracle,
    EachScenario,
    TwoStageExaModel,
    get_nscen,
    get_var_scen,
    get_con_scen,
    AbstractVariableTag,
    AbstractConstraintTag,
    get_value,
    set_value!,
    get_start,
    set_start!,
    get_lvar,
    set_lvar!,
    get_uvar,
    set_uvar!,
    get_lcon,
    set_lcon!,
    get_ucon,
    set_ucon!

end # module ExaModels
