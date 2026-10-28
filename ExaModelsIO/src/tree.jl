# Expression trees <-> plain dictionaries.
#
# Every node becomes a Dict with exactly one key naming its kind:
#
#   "op"    => [name, args...]       function application; name is the function's
#                                    qualified name, e.g. "Base.+", "Base.Math.sin"
#   "sum"   => [args...]             ExaModels.SumNode
#   "prod"  => [args...]             ExaModels.ProdNode
#   "var"   => index                 decision variable, global 1-based index
#   "par"   => index                 parameter, global 1-based index
#   "data"  => [path...]             generator element; path entries are field
#                                    names (strings) or positions (integers)
#   "const" => [type, value]         literal number; value is a string so that
#                                    Inf/NaN and exact floats survive JSON
#   "null"  => [type, value] or null ExaModels.Null, a constant leaf
#   "static"=> [type, value]         ExaModels.Constant (value in the type domain)
#   "val"   => [type, value]         Val(value), e.g. the exponent of a literal power x^3
#   "tuple" => [args...]             a tuple, e.g. the row index (i, j) of a 2-D constraint
#
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

_parse_number(t::AbstractString, v::AbstractString) =
    (T = _NUMBER_TYPES[t]; T === Bool ? parse(Bool, v) : parse(T, v))

_const(v::Number) = Dict("const" => [_number_type_name(typeof(v)), repr(v)])

_function_name(f) = join((string.(fullname(parentmodule(f)))..., string(nameof(f))), ".")

# ---------------------------------------------------------------- encode

encode(v::Number) = _const(v)
encode(::ExaModels.Constant{T}) where {T} =
    Dict("static" => [_number_type_name(typeof(T)), repr(T)])
# A Null stays a Null: a whole expression may be a constant (`con[1] += -N`), and
# ExaModels needs a node there, not a number.
encode(n::ExaModels.Null) =
    Dict("null" => n.value === nothing ? nothing : _const(n.value)["const"])
encode(n::ExaModels.Var) = Dict("var" => encode(n.i))
encode(n::ExaModels.ParameterNode) = Dict("par" => encode(n.i))
encode(::ExaModels.DataSource) = Dict("data" => Any[])
function encode(n::ExaModels.DataIndexed{I,J}) where {I,J}
    path = encode(getfield(n, :inner))["data"]
    return Dict("data" => push!(copy(path), J isa Symbol ? string(J) : J))
end
encode(n::ExaModels.Node1{F}) where {F} =
    Dict("op" => Any[_function_name(F.instance), encode(n.inner)])
encode(n::ExaModels.Node2{F}) where {F} =
    Dict("op" => Any[_function_name(F.instance), encode(n.inner1), encode(n.inner2)])
encode(n::ExaModels.SumNode) = Dict("sum" => Any[encode(a) for a in n.inners])
encode(n::ExaModels.ProdNode) = Dict("prod" => Any[encode(a) for a in n.inners])
encode(::Val{V}) where {V} = Dict("val" => [_number_type_name(typeof(V)), repr(V)])
encode(t::Tuple) = Dict("tuple" => Any[encode(a) for a in t])
# ArgLeaf exists from ExaModels 0.13 on; checking for it keeps the writer usable on older versions.
isdefined(ExaModels, :ArgLeaf) && (encode(n::ExaModels.ArgLeaf) = throw(ArgumentError(
    "the model still holds a recipe argument; instantiate it before writing")))
encode(n) = throw(ArgumentError("ExaModelsIO cannot store expression node $(typeof(n))"))

# ---------------------------------------------------------------- decode

# Resolve a qualified function name.  `functions` (name => function) wins, so
# a caller can supply functions that were registered with ExaModels by the user.
function resolve_function(name::AbstractString, functions)
    haskey(functions, name) && return functions[name]
    parts = split(name, ".")
    root = Symbol(parts[1])
    mod = root === :Base ? Base : root === :Core ? Core :
        get(Dict(Symbol(m.name) => mod for (m, mod) in Base.loaded_modules), root, nothing)
    mod === nothing && throw(ArgumentError(
        "function `$name`: module $root is not loaded; load it or pass `functions`"))
    obj = mod
    for p in parts[2:end]
        s = Symbol(p)
        isdefined(obj, s) || throw(ArgumentError(
            "function `$name` not found; pass it with `functions = Dict(\"$name\" => f)`"))
        obj = getfield(obj, s)
    end
    return obj
end

# Rebuild the node for generator element `p`.  Called with `p` set to the
# symbolic element ExaModels passes while tracing the generator, so the result
# is built by ExaModels' own operator overloads.
function decode(d, p, functions)
    k, v = only(d)
    if k == "op"
        f = resolve_function(v[1], functions)
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
        return v === nothing ? ExaModels.Null() : ExaModels.Null(_parse_number(v[1], v[2]))
    elseif k == "const"
        return _parse_number(v[1], v[2])
    elseif k == "static"
        return ExaModels.Constant(_parse_number(v[1], v[2]))
    elseif k == "val"
        return Val(_parse_number(v[1], v[2]))
    elseif k == "tuple"
        return Tuple(decode(a, p, functions) for a in v)
    elseif k == "sum"
        return ExaModels.SumNode(Tuple(decode(a, p, functions) for a in v))
    elseif k == "prod"
        return ExaModels.ProdNode(Tuple(decode(a, p, functions) for a in v))
    end
    throw(ArgumentError("unknown node kind `$k`"))
end
