# Expression trees <-> plain dictionaries (the grammar is in SPEC.md).
#
# Every node becomes a Dict with exactly one key naming its kind:
#
#   "op"    => [name, args...]       function application; name from the operator
#                                    vocabulary (vocabulary.jl), or "ext:<name>"
#   "sum"   => [args...]             sum of the arguments
#   "prod"  => [args...]             product of the arguments
#   "var"   => index                 decision variable, global 1-based index
#   "par"   => index                 parameter, global 1-based index into θ
#   "sub"   => index                 buffered subexpression value, global 1-based index into θ
#   "data"  => [path...]             generator element; path entries are field
#                                    names (strings) or positions (integers)
#   "const" => number                a literal
#   "null"  => number or null        a constant leaf standing where ExaModels needs a node
#   "static"=> number                a constant ExaModels keeps in the type domain
#   "val"   => number                Val(value), the exponent of a literal power x^3
#   "tuple" => [args...]             a tuple, e.g. the row index (i, j) of a 2-D constraint
#
# A number is [type, value]: value is a JSON integer or boolean for integer and
# Bool types, and for floating-point types a JSON number holding the value
# converted to Float64 (exact for Float16/Float32/Float64), or "inf", "-inf", "nan".
# `index` is itself a node, since a variable index usually depends on the data.

const _NUMBER_TYPES = Dict(
    "Float64" => Float64, "Float32" => Float32, "Float16" => Float16,
    "Int64" => Int64, "Int32" => Int32, "Int16" => Int16, "Int8" => Int8,
    "UInt64" => UInt64, "UInt32" => UInt32, "UInt16" => UInt16, "UInt8" => UInt8,
    "Bool" => Bool,
)

_number_type_name(::Type{T}) where {T} =
    haskey(_NUMBER_TYPES, string(T)) ? string(T) :
    throw(ArgumentError("ExaModelsIO cannot store numbers of type $T"))

_number_value(v::Union{Bool,Integer}) = v
_number_value(v::AbstractFloat) =
    isnan(v) ? "nan" : isinf(v) ? (v > 0 ? "inf" : "-inf") : Float64(v)

number(v::Number) = Any[_number_type_name(typeof(v)), _number_value(v)]

function parse_number(n)
    T = _NUMBER_TYPES[n[1]]
    v = n[2]
    v isa AbstractString || return T(v)
    v == "inf" ? T(Inf) : v == "-inf" ? T(-Inf) : v == "nan" ? T(NaN) :
        throw(ArgumentError("invalid number $(n)"))
end

# ---------------------------------------------------------------- encode

encode(v::Number) = Dict("const" => number(v))
encode(::ExaModels.Constant{T}) where {T} = Dict("static" => number(T))
# A Null stays a Null: a whole expression may be a constant (`con[1] += -N`), and
# ExaModels needs a node there, not a number.
encode(n::ExaModels.Null) =
    Dict("null" => n.value === nothing ? nothing : number(n.value))
encode(n::ExaModels.Var) = Dict("var" => encode(n.i))
encode(n::ExaModels.ParameterNode) = Dict("par" => encode(n.i))
encode(::ExaModels.DataSource) = Dict("data" => Any[])
function encode(n::ExaModels.DataIndexed{I,J}) where {I,J}
    path = encode(getfield(n, :inner))["data"]
    return Dict("data" => push!(copy(path), J isa Symbol ? string(J) : J))
end
encode(n::ExaModels.Node1{F}) where {F} =
    Dict("op" => Any[operator_name(F.instance), encode(n.inner)])
encode(n::ExaModels.Node2{F}) where {F} =
    Dict("op" => Any[operator_name(F.instance), encode(n.inner1), encode(n.inner2)])
_has(:SubexprNode) && (encode(n::ExaModels.SubexprNode) = Dict("sub" => encode(n.i)))
encode(n::ExaModels.SumNode) = Dict("sum" => Any[encode(a) for a in n.inners])
encode(n::ExaModels.ProdNode) = Dict("prod" => Any[encode(a) for a in n.inners])
encode(::Val{V}) where {V} = Dict("val" => number(V))
encode(t::Tuple) = Dict("tuple" => Any[encode(a) for a in t])
# ArgLeaf exists from ExaModels 0.13 on; checking for it keeps the writer usable on older versions.
isdefined(ExaModels, :ArgLeaf) && (encode(n::ExaModels.ArgLeaf) = throw(ArgumentError(
    "the model still holds a recipe argument; instantiate it before writing")))
encode(n) = throw(ArgumentError("ExaModelsIO cannot store expression node $(typeof(n))"))

# ---------------------------------------------------------------- decode

# Rebuild the node for generator element `p`.  Called with `p` set to the
# symbolic element ExaModels passes while tracing the generator, so the result
# is built by ExaModels' own operator overloads.
function decode(d, p, functions)
    k, v = only(d)
    if k == "op"
        f = operator_function(v[1], functions)
        args = Tuple(decode(a, p, functions) for a in v[2:end])
        # x^3 with a literal exponent is stored as ^(x, Val(3)); rebuild it the same way
        f === (^) && args[2] isa Val && return Base.literal_pow(^, args[1], args[2])
        return f(args...)
    elseif k == "var"
        return ExaModels.Var(decode(v, p, functions))
    elseif k == "par"
        return ExaModels.ParameterNode(decode(v, p, functions))
    elseif k == "data"
        q = p
        for s in v
            q = s isa AbstractString ? getproperty(q, Symbol(s)) : q[Int(s)]
        end
        return q
    elseif k == "null"
        return v === nothing ? ExaModels.Null() : ExaModels.Null(parse_number(v))
    elseif k == "const"
        return parse_number(v)
    elseif k == "static"
        return ExaModels.Constant(parse_number(v))
    elseif k == "val"
        return Val(parse_number(v))
    elseif k == "tuple"
        return Tuple(decode(a, p, functions) for a in v)
    elseif k == "sub"
        _has(:SubexprNode) || throw(ArgumentError(
            "the file uses buffered subexpressions, which ExaModels $(pkgversion(ExaModels)) does not have"))
        return ExaModels.SubexprNode(decode(v, p, functions))
    elseif k == "sum"
        return ExaModels.SumNode(Tuple(decode(a, p, functions) for a in v))
    elseif k == "prod"
        return ExaModels.ProdNode(Tuple(decode(a, p, functions) for a in v))
    end
    throw(ArgumentError("unknown node kind `$k`"))
end
