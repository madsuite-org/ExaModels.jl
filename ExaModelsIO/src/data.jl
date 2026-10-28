# Generator data <-> columns.
#
# A block's iterator is described by a schema and stored as one array per leaf
# field.  Schemas:
#
#   {"range": [start, stop]}                      a UnitRange; no arrays
#   {"array": dims, "elem": elem}                 an array of elements
#
# and an element schema is one of
#
#   {"scalar": type}                              column at the block's path
#   {"namedtuple": [[name, elem], ...]}           columns under <path>/<name>
#   {"tuple": [elem, ...]}                        columns under <path>/<i>
#
# A struct (e.g. `ExaPowerIO.GenData`) is stored as a named tuple of its fields
# and read back as a NamedTuple: expressions only access fields by name or
# position, and the struct's type may not exist when the file is read.

function data_schema(itr::AbstractUnitRange{<:Integer})
    return Dict("range" => [first(itr), last(itr)]), Pair{String,Vector}[]
end

function data_schema(itr)
    a = collect(itr)  # also copies GPU arrays to the CPU
    cols = Pair{String,Vector}[]
    elem = _elem_schema!(cols, "", vec(a))
    return Dict("array" => collect(size(a)), "elem" => elem), cols
end

_join(path, name) = isempty(path) ? string(name) : string(path, "/", name)

function _elem_schema!(cols, path, v::AbstractVector{T}) where {T}
    if T <: Number
        push!(cols, (isempty(path) ? "value" : path) => Vector{T}(v))
        return Dict("scalar" => _number_type_name(T))
    elseif T <: NamedTuple || (isconcretetype(T) && isstructtype(T) && !(T <: Tuple))
        names = fieldnames(T)
        return Dict("namedtuple" => Any[
            [string(n), _elem_schema!(cols, _join(path, n), [getfield(e, n) for e in v])]
            for n in names])
    elseif T <: Tuple
        return Dict("tuple" => Any[
            _elem_schema!(cols, _join(path, i), [e[i] for e in v]) for i in 1:fieldcount(T)])
    end
    throw(ArgumentError(
        "ExaModelsIO can store generator data made of numbers, tuples, named tuples and structs of these; got $T"))
end

# Zip typed columns into rows. A function barrier: `parts` is a concretely typed
# tuple of vectors here, so the loop compiles to plain loads.
function _rows(wrap, parts::Tuple)
    n = length(parts[1])
    all(p -> length(p) == n, parts) || error("columns of one data block have different lengths")
    return [wrap(map(p -> @inbounds(p[k]), parts)) for k in 1:n]
end

# Inverse of `data_schema`.  `col(path)` returns the stored column.
function data_value(schema, col)
    haskey(schema, "range") && return schema["range"][1]:schema["range"][2]
    v = _elem_value(schema["elem"], "", col)
    dims = Tuple(Int.(schema["array"]))
    return length(dims) == 1 ? v : reshape(v, dims)
end

function _elem_value(s, path, col)
    if haskey(s, "scalar")
        return col(isempty(path) ? "value" : path)
    elseif haskey(s, "namedtuple")
        names = Tuple(Symbol(n) for (n, _) in s["namedtuple"])
        parts = Tuple(_elem_value(e, _join(path, n), col) for (n, e) in s["namedtuple"])
        return _rows(NamedTuple{names}, parts)
    elseif haskey(s, "tuple")
        parts = Tuple(_elem_value(e, _join(path, i), col) for (i, e) in enumerate(s["tuple"]))
        return _rows(identity, parts)
    end
    throw(ArgumentError("unknown data schema $(keys(s))"))
end
