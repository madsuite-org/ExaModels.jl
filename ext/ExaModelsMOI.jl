module ExaModelsMOI

import ExaModels
import MathOptInterface as MOI

function __init__()
    setglobal!(ExaModels, :Optimizer, Optimizer)
    return
end

const PARAMETER_INDEX_THRESHOLD = div(typemax(Int64), 2) + 1

"""
    struct Bin{E,P}
        head::E
        data::Vector{P}
    end

This struct represents `head(d) for d in data`

`head` will be one of two things:

 1) `DataIndexed() => e`: this means that the generator is a constraint
    augmentation. It maps the row index to an expression. The `e` is a symbolic
    function. This de-duplicates structural constraints, so two constraints with
    the same symbolic form will get automatically added as elements to the
    vector `data`.

 2) `e::ExaModels.AbstractNode`: this means that the generator is part of a
    summation. This is used for the objective function.
"""
struct Bin{E,P}
    head::E
    data::Vector{P}
end

"""
    update_bin!(bin::Vector{Bin}, head, data)

This function loops through the list of `bin` looking for a matching `head`. If
found, it updates the bin in place. Othersise, it appends a new bin.
"""
function update_bin!(
    bins::Vector{Bin},
    head::Union{
        ExaModels.AbstractNode,
        Pair{<:ExaModels.AbstractNode,<:ExaModels.AbstractNode},
    },
    data::Tuple,
)
    for bin in bins
        if _update_bin!(bin, head, data)
            return
        end
    end
    push!(bins, Bin(head, [data]))
    return bins
end

# The types match for `head(data)` to be appended to this bin. We check if the
# head's match with `==`, otherwise we recurse to the next bin.
function _update_bin!(bin::Bin{E,P}, head::E, data::P) where {E,P}
    if head == bin.head
        push!(bin.data, data)
        return true
    end
    return false
end

# The head does not match. We can't update this bin in-place.
_update_bin!(::Bin, ::Any, ::Any) = false

# A method for the objective function. First convert the MOI function `f` into
# an `ExaModels.AbstractNode`, then add that.
function update_bin!(bins::Vector{Bin}, f)
    head, data = _exafy(f, (), nothing)
    return update_bin!(bins, head, data)
end

# A method for adding to a constraint. First convert the MOI function `f` into
# an `ExaModels.AbstractNode`, then add that.
function update_bin!(bins::Vector{Bin}, (row, f)::Pair{Int,F}) where {F}
    head, data = _exafy(f, (), nothing)
    e = ExaModels.DataIndexed(ExaModels.DataSource(), length(data) + 1)
    return update_bin!(bins, e => head, (data..., row))
end

# This is a type that lets us dispatch on the difference between `row => expr`
# and `expr`.
abstract type AbstractBin end

# This is an `expr`. It gets added to the objective.
struct ObjectiveBin <:AbstractBin end

# Things are passed through unchanged.
(::ObjectiveBin)(f) = f

# Except objective constants are passed as `Null`.
(::ObjectiveBin)(f::Real) = ExaModels.Null(f)

# This is a `row => expr`. We keep the row in a closure.
struct ConstraintBin <: AbstractBin
    row::Int
end

# When passing, we convert to a pair.
(bin::ConstraintBin)(f) = bin.row => f

# VariableIndices are handled directly.
function update_bin!(
    bins::Vector{Bin},
    fn::AbstractBin,
    f::Union{Real,MOI.VariableIndex},
)
    return update_bin!(bins, fn(f))
end

# Add the additive terms separately, instead of creating a single +(args...)
# expression.
function update_bin!(
    bins::Vector{Bin},
    fn::AbstractBin,
    f::MOI.ScalarAffineFunction,
)
    for term in f.terms
        update_bin!(bins, fn(term))
    end
    if !iszero(f.constant)
        update_bin!(bins, fn(f.constant))
    end
    return bins
end

# Add the additive terms separately, instead of creating a single +(args...)
# expression.
function update_bin!(
    bins::Vector{Bin},
    fn::AbstractBin,
    f::MOI.ScalarQuadraticFunction,
)
    for term in f.affine_terms
        update_bin!(bins, fn(term))
    end
    for term in f.quadratic_terms
        update_bin!(bins, fn(term))
    end
    if !iszero(f.constant)
        update_bin!(bins, fn(f.constant))
    end
    return bins
end

_is_zero(x::Real) = iszero(x)

_is_zero(::Any) = false

function update_bin!(
    bins::Vector{Bin},
    fn::AbstractBin,
    f::MOI.ScalarNonlinearFunction,
)
    if f.head == :- && length(f.args) == 2
        # Optimization: :(x - y) -> :(+(x, -y))
        # This allows additive terms in the left-hand side to be added
        # separately. This is a common case in JuMP because
        # `@constraint(model, lhs <= rhs)` normalizes to `lhs - rhs <= 0`.
        update_bin!(bins, fn, f.args[1])
        if !_is_zero(f.args[2])
            rhs = MOI.Utilities.operate(-, Float64, f.args[2])
            update_bin!(bins, fn(rhs))
        end
        return bins
    elseif f.head != :+
        return update_bin!(bins, fn(f))
    end
    # Optimization: if the expression is a `:+`, add the child arguments as
    # separate terms. This keeps the size of the expressions small for ExaModels.
    constant = 0.0
    for arg in f.args
        if arg isa MOI.ScalarAffineFunction
            for term in arg.terms
                update_bin!(bins, fn(term))
            end
            constant += arg.constant
        elseif arg isa MOI.ScalarQuadraticFunction
            for term in arg.affine_terms
                update_bin!(bins, fn(term))
            end
            for term in arg.quadratic_terms
                update_bin!(bins, fn(term))
            end
            constant += arg.constant
        else
            # This is NOT fn(arg) here because we want to be able to lift any
            # nested `+(+(args...), args....)`.
            update_bin!(bins, fn, arg)
        end
    end
    if !iszero(constant)
        update_bin!(bins, fn(constant))
    end
    return bins
end

# Function types handled by extensions, e.g., `GenOpt.SumGenerator` in the
# objective. See `ExaModels.exafy_extension_obj_arg`.
function update_bin!(bins::Vector{Bin}, fn::AbstractBin, f)
    if !(fn isa ObjectiveBin && ExaModels.is_extension_type(typeof(f)))
        throw(ArgumentError("Unsupported function of type $(typeof(f))"))
    end
    head, data = ExaModels.exafy_extension_obj_arg(f)
    push!(bins, Bin(head, data))
    return bins
end

# _exafy

# This method is used for objective constants.
_exafy(f::ExaModels.Null, data::Tuple, ::Any) = f, data

# This method is used when a constant appears in a function.
function _exafy(f::Real, data::Tuple, ::Any)
    e = ExaModels.DataIndexed(ExaModels.DataSource(), length(data) + 1)
    return e, (data..., f)
end

function _exafy(
    f::MOI.VariableIndex,
    data::Tuple,
    var_to_data::Union{Nothing,Dict{Int,Int}},
)
    if f.value > PARAMETER_INDEX_THRESHOLD
        e = ExaModels.DataIndexed(ExaModels.DataSource(), length(data) + 1)
        idx = f.value - PARAMETER_INDEX_THRESHOLD
        return ExaModels.ParameterNode(e), (data..., idx)
    end
    if var_to_data !== nothing
        # An optimization: the variable `f` may already appear in the tuple
        # `data`. If so, we want to re-use the slot instead of appending a new
        # element to `data`. (ExaModels could be clever here and check for
        # duplicates.)
        #
        # We don't have this optimization for ParameterNode's because the main
        # problem with duplicates are they they show up as duplicated elements
        # in the Jacobian and Hessian.
        if (pidx = get(var_to_data, f.value, nothing)) !== nothing
            p_cache = ExaModels.DataIndexed(ExaModels.DataSource(), pidx)
            return ExaModels.Var(p_cache), data
        end
        var_to_data[f.value] = length(data) + 1
    end
    e = ExaModels.DataIndexed(ExaModels.DataSource(), length(data) + 1)
    return ExaModels.Var(e), (data..., f.value)
end

function _exafy(
    f::MOI.ScalarAffineTerm,
    data::Tuple,
    var_to_data::Union{Nothing,Dict{Int,Int}},
)
    x_head, data = _exafy(f.variable, data, var_to_data)
    c_head, data = _exafy(f.coefficient, data, var_to_data)
    return c_head * x_head, data
end

# This method is used when a ScalarAffineFunction appears inside a
# ScalarNonlinearFunction. For that reason we don't do anything clever with the
# additive terms.
function _exafy(
    f::MOI.ScalarAffineFunction,
    data::Tuple,
    var_to_data::Union{Nothing,Dict{Int,Int}},
)
    head, data = _exafy(f.constant, data, var_to_data)
    if !isempty(f.terms)
        y = sum(begin
            c1, data = _exafy(term, data, var_to_data)
            c1
        end for term in f.terms)
        head += y
    end
    return head, data
end

function _exafy(
    f::MOI.ScalarQuadraticTerm,
    data::Tuple,
    var_to_data::Union{Nothing,Dict{Int,Int}},
)
    if f.variable_1 == f.variable_2
        x_head, data = _exafy(f.variable_1, data, var_to_data)
        c_head, data = _exafy(f.coefficient / 2, data, var_to_data)
        return c_head * abs2(x_head), data
    end
    x1_head, data = _exafy(f.variable_1, data, var_to_data)
    x2_head, data = _exafy(f.variable_2, data, var_to_data)
    c_head, data = _exafy(f.coefficient, data, var_to_data)
    return c_head * x1_head * x2_head, data
end

# This method is used when a ScalarQuadraticFunction appears inside a
# ScalarNonlinearFunction. For that reason we don't do anything clever with the
# additive terms.
function _exafy(
    f::MOI.ScalarQuadraticFunction,
    data::Tuple,
    var_to_data::Union{Nothing,Dict{Int,Int}},
)
    head, data = _exafy(f.constant, data, var_to_data)
    if !isempty(f.affine_terms)
        head += sum(begin
            c1, data = _exafy(term, data, var_to_data)
            c1
        end for term in f.affine_terms)
    end
    if !isempty(f.quadratic_terms)
        head += sum(begin
            c1, data = _exafy(term, data, var_to_data)
            c1
        end for term in f.quadratic_terms)
    end
    return head, data
end

function _exafy(f::MOI.ScalarNonlinearFunction, data::Tuple, ::Nothing)
    # Replace the incoming `var_to_data === nothing` with a dictionary that maps
    # the variable index with the element in `data`. This is used when there are
    # repeated variable indices in `f`. See `_exafy(::VariableIndex, args...)`.
    return _exafy(f, data, Dict{Int,Int}())
end

function _exafy(
    f::MOI.ScalarNonlinearFunction,
    data::Tuple,
    var_to_data::Dict{Int,Int},
)
    # This assumes that we support only the default functions in `MOI.Nonlinear`
    op = getfield(MOI.Nonlinear, f.head)
    if length(f.args) == 1
        # A special case when there is one argument.
        arg, data = _exafy(only(f.args), data, var_to_data)
        return op(arg), data
    elseif length(f.args) == 2
        # A special case when there are two arguments
        arg1, data = _exafy(f.args[1], data, var_to_data)
        arg2, data = _exafy(f.args[2], data, var_to_data)
        return op(arg1, arg2), data
    end
    args = ()
    for arg in f.args
        head, data = _exafy(arg, data, var_to_data)
        args = (args..., head)
    end
    return op(args...), data
end

"""
    ExaModels.ExaModel(
        src::MOI.ModelLike;
        backend = nothing,
        prod::Bool = false,
        T = ExaModels.default_T(backend),
    )

Convert `src` to an `ExaModel`.
"""
function ExaModels.ExaModel(
    src::MOI.ModelLike;
    backend = nothing,
    prod = false,
    T = ExaModels.default_T(backend),
)
    dest = Optimizer{T}(nothing; backend)
    MOI.copy_to(dest, src)
    c = to_exacore(dest, backend)
    return ExaModels.ExaModel(c; prod)
end

# Now comes the MOI interface

"""
    ExaModels.Optimizer(solver, backend = nothing)

Create a new ExaModels.Optimizer object.

## Examples

```julia-repl
julia> import ExaModels, NLPModelsIpopt, KernelAbstractions

julia> optimizer = () -> ExaModels.Optimizer(NLPModelsIpopt.ipopt);

julia> optimizer = () -> ExaModels.Optimizer(NLPModelsIpopt.ipopt, KernelAbstractions.CPU());
```
"""
mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    solver::Any
    backend::Any
    result::Any
    solve_time::Float64
    options::Dict{Symbol,Any}
    # Problem cache
    sense::MOI.OptimizationSense
    lvar::Vector{T}
    uvar::Vector{T}
    startvar::Vector{T}
    pstart::Vector{T}
    lcon::Vector{T}
    ucon::Vector{T}
    objs::Vector{Bin}
    cons::Vector{Bin}

    function Optimizer{T}(solver, backend = nothing; kwargs...) where {T}
        return new(
            solver,
            backend,
            nothing,
            0.0,
            Dict{Symbol,Any}(kwargs...),
            # Problem cache
            MOI.FEASIBILITY_SENSE,
            T[],
            T[],
            T[],
            T[],
            T[],
            T[],
            Bin[],
            Bin[],
        )
    end
end

function Optimizer(solver, backend = nothing; kwargs...)
    T = ExaModels.default_T(backend)
    return Optimizer{T}(solver, backend; kwargs...)
end

function MOI.empty!(model::ExaModelsMOI.Optimizer)
    model.result = nothing
    model.solve_time = 0.0
    model.sense = MOI.FEASIBILITY_SENSE
    empty!(model.lvar)
    empty!(model.uvar)
    empty!(model.startvar)
    empty!(model.pstart)
    empty!(model.lcon)
    empty!(model.ucon)
    empty!(model.cons)
    empty!(model.objs)
    return
end

function MOI.is_empty(model::Optimizer)
    return isempty(model.lvar) &&
           isempty(model.pstart) &&
           isempty(model.cons) &&
           isempty(model.objs)
end

# MOI.ObjectiveSense

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    model.sense = sense
    if sense == MOI.FEASIBILITY_SENSE
        empty!(model.objs)
    end
    return
end

# MOI.ObjectiveFunction

function MOI.supports(
    ::Optimizer{T},
    ::MOI.ObjectiveFunction{F},
) where {
    T,
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{T},
        MOI.ScalarQuadraticFunction{T},
        MOI.ScalarNonlinearFunction,
    },
}
    return true
end

function MOI.set(
    model::Optimizer{T},
    ::MOI.ObjectiveFunction{F},
    f::F,
) where {
    T,
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{T},
        MOI.ScalarQuadraticFunction{T},
        MOI.ScalarNonlinearFunction,
    },
}
    empty!(model.objs)
    update_bin!(model.objs, ObjectiveBin(), f)
    return
end

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{F}) where {F}
    return ExaModels.is_extension_type(F)
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveFunction{F}, f::F) where {F}
    empty!(model.objs)
    update_bin!(model.objs, ObjectiveBin(), f)
    return
end

# MOI.add_variable

function MOI.add_variable(model::Optimizer{T}) where {T}
    push!(model.lvar, typemin(T))
    push!(model.uvar, typemax(T))
    push!(model.startvar, zero(T))
    return MOI.VariableIndex(length(model.lvar))
end

# MOI.add_constrained_variable

function MOI.supports_add_constrained_variable(
    ::Optimizer{T},
    ::Type{MOI.Parameter{T}},
) where {T}
    return true
end

function MOI.add_constrained_variable(
    model::Optimizer{T},
    set::MOI.Parameter{T},
) where {T}
    push!(model.pstart, set.value)
    index = PARAMETER_INDEX_THRESHOLD + length(model.pstart)
    ci = MOI.ConstraintIndex{MOI.VariableIndex,typeof(set)}(index)
    return MOI.VariableIndex(index), ci
end

# VariableIndex-in-Set constraints

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VariableIndex},
    ::Type{S},
) where {
    T,
    S<:Union{
        MOI.GreaterThan{T},
        MOI.LessThan{T},
        MOI.EqualTo{T},
        MOI.Interval{T},
    },
}
    return true
end

function _update_bound(model::Optimizer, col::Int, set::MOI.GreaterThan)
    model.lvar[col] = set.lower
    return
end

function _update_bound(model::Optimizer, col::Int, set::MOI.LessThan)
    model.uvar[col] = set.upper
    return
end

function _update_bound(model::Optimizer, col::Int, set::MOI.EqualTo)
    model.lvar[col] = model.uvar[col] = set.value
    return
end

function _update_bound(model::Optimizer, col::Int, set::MOI.Interval)
    model.lvar[col], model.uvar[col] = set.lower, set.upper
    return
end

function MOI.add_constraint(
    model::Optimizer{T},
    f::MOI.VariableIndex,
    s::Union{
        MOI.GreaterThan{T},
        MOI.LessThan{T},
        MOI.EqualTo{T},
        MOI.Interval{T},
    },
) where {T}
    @assert f.value < PARAMETER_INDEX_THRESHOLD
    _update_bound(model, f.value, s)
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(f.value)
end

# MOI.VariablePrimalStart

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.set(
    model::Optimizer{T},
    ::MOI.VariablePrimalStart,
    x::MOI.VariableIndex,
    value::Union{Nothing,T},
) where {T}
    @assert x.value < PARAMETER_INDEX_THRESHOLD
    model.startvar[x.value] = something(value, zero(T))
    return
end

# Function-in-Set constraints

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{F},
    ::Type{S},
) where {
    T,
    F<:Union{
        MOI.ScalarAffineFunction{T},
        MOI.ScalarQuadraticFunction{T},
        MOI.ScalarNonlinearFunction,
    },
    S<:Union{
        MOI.GreaterThan{T},
        MOI.LessThan{T},
        MOI.EqualTo{T},
        MOI.Interval{T},
    },
}
    return true
end

_bounds(s::MOI.Interval) = (s.lower, s.upper)

_bounds(s::MOI.EqualTo) = (s.value, s.value)

_bounds(s::MOI.GreaterThan{T}) where {T} = (s.lower, typemax(T))

_bounds(s::MOI.LessThan{T}) where {T} = (typemin(T), s.upper)

function MOI.add_constraint(
    model::Optimizer{T},
    f::Union{
        MOI.ScalarAffineFunction{T},
        MOI.ScalarQuadraticFunction{T},
        MOI.ScalarNonlinearFunction,
    },
    s::Union{
        MOI.GreaterThan{T},
        MOI.LessThan{T},
        MOI.EqualTo{T},
        MOI.Interval{T},
    },
) where {T}
    row = length(model.lcon) + 1
    update_bin!(model.cons, ConstraintBin(row), f)
    l, u = _bounds(s)
    push!(model.lcon, l)
    push!(model.ucon, u)
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(row)
end

# Vector-valued constraints with function types handled by extensions, e.g.,
# `GenOpt.FunctionGenerator`. See `ExaModels.exafy_extension_con`.

const _VectorSets = Union{MOI.Zeros,MOI.Nonnegatives,MOI.Nonpositives}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{F},
    ::Type{<:_VectorSets},
) where {F<:MOI.AbstractVectorFunction}
    return ExaModels.is_extension_type(F)
end

_bounds(::MOI.Zeros, ::Type{T}) where {T} = (zero(T), zero(T))

_bounds(::MOI.Nonnegatives, ::Type{T}) where {T} = (zero(T), typemax(T))

_bounds(::MOI.Nonpositives, ::Type{T}) where {T} = (typemin(T), zero(T))

function MOI.add_constraint(
    model::Optimizer{T},
    f::MOI.AbstractVectorFunction,
    s::_VectorSets,
) where {T}
    row = length(model.lcon) + 1
    head, data = ExaModels.exafy_extension_con(f, row)
    push!(model.cons, Bin(head, data))
    l, u = _bounds(s, T)
    append!(model.lcon, fill(l, MOI.dimension(s)))
    append!(model.ucon, fill(u, MOI.dimension(s)))
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(row)
end

function to_exacore(model::Optimizer{T}, backend) where {T}
    c = ExaModels.ExaCore(
        T;
        backend,
        minimize = model.sense != MOI.MAX_SENSE,
        concrete = Val(true),
    )
    if !isempty(model.pstart)
        c, _ = ExaModels.add_par(c, model.pstart)
    end
    c, _ = ExaModels.add_var(
        c,
        length(model.lvar);
        start = model.startvar,
        lvar = model.lvar,
        uvar = model.uvar,
    )
    if !isempty(model.cons)
        c, cons = ExaModels.add_con(c, length(model.lcon); model.lcon, model.ucon)
        for bin in model.cons
            c, _ = ExaModels.add_con!(c, cons, (bin.head for _ in bin.data))
        end
    end
    for bin in model.objs
        c, _ = ExaModels.add_obj(c, bin.head, bin.data)
    end
    return c
end

# MOI.copy_to

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

# MOI.optimize!

function MOI.optimize!(model::Optimizer)
    core = to_exacore(model, model.backend)
    exa_model = ExaModels.ExaModel(core; prod = true)
    start_time = time()
    result = model.solver(exa_model; model.options...)
    model.result = (
        objective = result.objective,
        solution = Array(result.solution),
        multipliers = Array(result.multipliers),
        multipliers_L = Array(result.multipliers_L),
        multipliers_U = Array(result.multipliers_U),
        status = result.status,
    )
    model.solve_time = time() - start_time
    return
end

# MOI.TerminationStatus

# SolverCore returns a `Symbol` in `result.status` for any solver implementing
# the NLPModels callable interface (e.g. `madnlp(::AbstractNLPModel)`,
# `ipopt(::AbstractNLPModel)`). The vocabulary is defined by SolverCore.jl
# (see SolverCore/src/stats.jl). Solvers that emit something else can be
# supported by extending these dicts or overriding the two `MOI.get` methods.
const _TERMINATION_STATUS_CODES = Dict{Symbol, MOI.TerminationStatusCode}(
    :first_order => MOI.LOCALLY_SOLVED,
    :acceptable => MOI.ALMOST_LOCALLY_SOLVED,
    :small_step => MOI.SLOW_PROGRESS,
    :infeasible => MOI.INFEASIBLE,
    :max_iter => MOI.ITERATION_LIMIT,
    :max_time => MOI.TIME_LIMIT,
    :user => MOI.INTERRUPTED,
    :exception => MOI.OTHER_ERROR,
)

MOI.get(model::Optimizer, ::MOI.RawStatusString) = string(model.result.status)

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.result === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return get(_TERMINATION_STATUS_CODES, model.result.status, MOI.OTHER_ERROR)
end

# MOI.PrimalStatus, MOI.DualStatus

const _RESULT_STATUS_CODES = Dict{Symbol, MOI.ResultStatusCode}(
    :first_order => MOI.FEASIBLE_POINT,
    :acceptable => MOI.NEARLY_FEASIBLE_POINT,
    :infeasible => MOI.INFEASIBLE_POINT,
)

function MOI.get(model::Optimizer, attr::Union{MOI.PrimalStatus,MOI.DualStatus})
    if model.result === nothing || attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    return get(
        _RESULT_STATUS_CODES,
        model.result.status,
        MOI.UNKNOWN_RESULT_STATUS,
    )
end

# MOI.VariablePrimal

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    if vi.value > PARAMETER_INDEX_THRESHOLD
        return model.pstart[vi.value]
    end
    return model.result.solution[vi.value]
end

# MOI.ConstraintDual

function _scale(model::Optimizer{T}) where {T}
    return model.sense == MOI.MAX_SENSE ? -one(T) : one(T)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    return -_scale(model) * model.result.multipliers[ci.value]
end

function _reduced_cost(model, col)
    return model.result.multipliers_L[col] - model.result.multipliers_U[col]
end
function MOI.get(
    model::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{T}},
) where {T}
    MOI.check_result_index_bounds(model, attr)
    rc = _reduced_cost(model, ci.value)
    return min(zero(rc), _scale(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{T}},
) where {T}
    MOI.check_result_index_bounds(model, attr)
    rc = _reduced_cost(model, ci.value)
    return max(zero(rc), _scale(model) * rc)
end

function MOI.get(
    model::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
) where {T,S<:Union{MOI.Interval{T},MOI.EqualTo{T}}}
    MOI.check_result_index_bounds(model, attr)
    return _scale(model) * _reduced_cost(model, ci.value)
end

# MOI.ResultCount

MOI.get(model::Optimizer, ::MOI.ResultCount) = model.result !== nothing ? 1 : 0

# MOI.ObjectiveValue

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.result.objective
end

# MOI.SolveTimeSec

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

# MOI.SolverName

function MOI.get(model::Optimizer, ::MOI.SolverName)
    return "$(string(model.solver)) running with ExaModels"
end

# MOI.RawOptimizerAttribute

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    model.options[Symbol(attr.name)] = value
    # No need to reset model.solver because this gets handled in optimize!.
    return
end

# MOI.NLPBlock

function MOI.set(::Optimizer, ::MOI.NLPBlock, ::MOI.NLPBlockData)
    return error(
        """
        The legacy nonlinear model interface is not supported.

        Please use the new MOI-based interface.
        """,
    )
end

end # module
