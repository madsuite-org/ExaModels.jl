# Extension points used by ExaModelsMOI and ExaModelsGenOpt extensions

"""
    is_extension_type(::Type{F}) -> Bool

Return `true` if `F` is a function type handled by an extension.
Used by `MOI.supports` and `MOI.supports_constraint` in ExaModelsMOI to
whitelist extension types.
"""
function is_extension_type end
is_extension_type(::Type) = false

"""
    exafy_extension_obj_arg(f) -> Tuple

Convert an objective function (or a term of a `+` objective) `f` to an
`(expr, data)` tuple for ExaModels so that the objective term is
`sum(expr(d) for d in data)`.
"""
function exafy_extension_obj_arg end

"""
    exafy_extension_con(f, row::Int) -> Tuple

Convert a vector-valued constraint function `f` whose first row is `row` to an
`(row_expr => expr, data)` tuple for ExaModels so that constraint `row_expr(d)`
is `expr(d)` for each `d` in `data`.
"""
function exafy_extension_con end
