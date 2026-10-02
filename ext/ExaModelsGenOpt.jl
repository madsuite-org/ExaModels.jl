module ExaModelsGenOpt

import ExaModels
import GenOpt:
    FunctionGenerator, SumGenerator, ContiguousArrayOfVariables, IteratorIndex
import MathOptInterface as MOI

# Mark GenOpt function types as extension types
ExaModels.is_extension_type(::Type{<:FunctionGenerator}) = true
ExaModels.is_extension_type(::Type{<:SumGenerator}) = true

# Handle SumGenerator in objective expressions
function ExaModels.exafy_extension_obj_arg(m::SumGenerator)
    lengths = map(it -> length(first(it.values)), m.iterators)
    if length(lengths) == 1 && lengths[] == 1
        return exagen(m.func, nothing), only.(m.iterators[].values)
    end
    return exagen(m.func, _offsets(lengths)), _pars(m.iterators)
end

# Handle FunctionGenerator constraints. The row index is appended to each
# element of the data.
function ExaModels.exafy_extension_con(func::FunctionGenerator, row::Int)
    lengths = map(it -> length(first(it.values)), func.iterators)
    expr = exagen(func.func, _offsets(lengths))
    row_expr = ExaModels.DataIndexed(ExaModels.DataSource(), sum(lengths) + 1)
    data = [(p..., row + i - 1) for (i, p) in enumerate(_pars(func.iterators))]
    return row_expr => expr, data
end

_offsets(lengths) = [0; cumsum(lengths)[1:(end - 1)]]

function _pars(iterators)
    return vec(
        map(
            Base.Iterators.ProductIterator(
                ntuple(i -> iterators[i].values, length(iterators)),
            ),
        ) do I
            return reduce((i, j) -> tuple(i..., j...), I)
        end,
    )
end

# Convert GenOpt expression trees to ExaModels format

exagen(α::Number, _) = α

function exagen(f::MOI.ScalarNonlinearFunction, offsets)
    if f.head == :getindex
        v = f.args[1]
        if v isa ContiguousArrayOfVariables
            # ExaModelsMOI uses the MOI variable index as ExaModels index
            idx = exagen(f.args[2], offsets)
            if !iszero(v.offset)
                idx = v.offset + idx
            end
            cp = cumprod(v.size)
            for i in 3:length(f.args)
                idx += cp[i - 2] * (exagen(f.args[i], offsets) - 1)
            end
            return ExaModels.Var(idx)
        elseif v isa IteratorIndex
            @assert length(f.args) == 2
            @assert f.args[2] isa Integer
            if isnothing(offsets)
                @assert isone(f.args[2])
                return ExaModels.DataSource()
            else
                return ExaModels.DataIndexed(
                    ExaModels.DataSource(),
                    offsets[v.value] + f.args[2],
                )
            end
        else
            error(
                "Unexpected the first operand of `getindex` to be of type `$(typeof(v))`",
            )
        end
    else
        # This assumes that we support only the default functions in
        # `MOI.Nonlinear`, like `_exafy` in ExaModelsMOI
        op = getfield(MOI.Nonlinear, f.head)
        return op((exagen(e, offsets) for e in f.args)...)
    end
end

end # module
