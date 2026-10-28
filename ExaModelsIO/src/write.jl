"""
    write_model(path, model::ExaModel; solution = nothing)

Write `model` to the HDF5 file `path`.

`solution`, if given, is a solver result with fields `solution`, `multipliers`,
`multipliers_L`, `multipliers_U`, `objective` and `status` (for example the
output of `madnlp` or `ipopt`); it is stored under `/solution` and read back
with [`read_solution`](@ref).

File layout:

| path                 | contents                                              |
|:---------------------|:------------------------------------------------------|
| attributes of `/`    | `format`, `format_version`, `examodels_version`       |
| `/structure`         | JSON: variable/parameter blocks, objective and        |
|                      | constraint expressions, and the schema of their data  |
| `/arrays/<name>`     | `x0`, `lvar`, `uvar`, `y0`, `lcon`, `ucon`, `theta`   |
| `/data/<block>/...`  | the data each block iterates over, one array per field|
| `/solution/<name>`   | optional solver output                                |
"""
function write_model(path::AbstractString, m::ExaModel; solution = nothing)
    structure, columns = describe(m)
    meta = m.meta
    HDF5.h5open(path, "w") do f
        HDF5.attributes(f)["format"] = FORMAT
        HDF5.attributes(f)["format_version"] = FORMAT_VERSION
        HDF5.attributes(f)["examodels_version"] = string(pkgversion(ExaModels))
        f["structure"] = JSON3.write(structure)
        arrays = HDF5.create_group(f, "arrays")
        for (k, v) in (
            "x0" => meta.x0, "lvar" => meta.lvar, "uvar" => meta.uvar,
            "y0" => meta.y0, "lcon" => meta.lcon, "ucon" => meta.ucon, "theta" => m.θ,
        )
            arrays[k] = Array(v)
        end
        for (k, v) in columns
            f[k] = v
        end
        solution === nothing || _write_solution(f, solution)
    end
    return path
end

function _write_solution(f, s)
    g = HDF5.create_group(f, "solution")
    for k in (:solution, :multipliers, :multipliers_L, :multipliers_U)
        hasproperty(s, k) && (g[string(k)] = Array(getproperty(s, k)))
    end
    hasproperty(s, :objective) && (g["objective"] = getproperty(s, :objective))
    hasproperty(s, :status) && (g["status"] = string(getproperty(s, :status)))
end

# ExaModels keeps blocks in reverse order of addition.
_in_order(t::Tuple) = reverse(t)

_dims(size) = Any[d isa Integer ? d : [first(d), last(d)] for d in size]

# The name under which `block` is registered in `refs`, or nothing.
function _ref_name(refs, block)
    for (k, v) in pairs(refs)
        typeof(v) === typeof(block) && v.offset == block.offset && return string(k)
    end
    return nothing
end

function describe(m::ExaModel{T}) where {T}
    refs = m.refs === nothing ? (;) : m.refs
    columns = Pair{String,Vector}[]
    _check_tag(b) = hasproperty(b, :tag) && b.tag !== nothing &&
        @warn "ExaModelsIO does not store block tags; dropping tag $(b.tag)" maxlog = 1

    variables = map(collect(_in_order(m.vars))) do v
        _check_tag(v)
        Dict("name" => _ref_name(refs, v), "size" => _dims(v.size), "offset" => v.offset)
    end
    parameters = map(collect(_in_order(m.pars))) do p
        _check_tag(p)
        Dict("name" => _ref_name(refs, p), "size" => _dims(p.size), "offset" => p.offset)
    end

    function store(group, itr)
        schema, cols = data_schema(itr)
        append!(columns, [string(group, "/", c) => v for (c, v) in cols])
        return schema
    end

    # A generator `for i in A, j in B` iterates the product of its axes, which
    # ExaModels keeps as the constraint's `size`.  Storing the axes instead of
    # the product rebuilds the same generator, and with it the same `size`.
    function block(kind, b, k)
        _check_tag(b)
        group = "data/$(kind)_$k"
        d = Dict{String,Any}("group" => group)
        if b isa ExaModels.Constraint && length(b.size) > 1 && !(b.f.f isa ExaModels.Null) &&
                isequal(vec(collect(b.itr)), vec(collect(Iterators.product(b.size...))))
            d["axes"] = [store("$group/axis_$i", a) for (i, a) in enumerate(b.size)]
        else
            d["data"] = store(group, b.itr)
        end
        return d
    end

    objectives = map(enumerate(_in_order(m.objs))) do (k, o)
        d = block("obj", o, k)
        d["expr"] = encode(o.f.f)
        d
    end

    constraints = map(enumerate(_in_order(m.cons))) do (k, c)
        d = block("con", c, k)
        if c isa ExaModels.ConstraintAugmentation
            d["kind"] = "augmentation"
            d["base_offset"] = c.f.o0  # offset of the constraint it adds to
            if c.f.f isa Pair
                d["index"] = encode(c.f.f.first)
                d["expr"] = encode(c.f.f.second)
            else  # element k adds to row k of the base constraint (as built by ExaModelsMOI)
                d["index"] = nothing
                d["expr"] = encode(c.f.f)
            end
        else
            d["kind"] = "constraint"
            d["name"] = _ref_name(refs, c)
            d["offset"] = c.offset
            d["shape"] = collect(size(c.itr))
            if c.f.f isa ExaModels.Null  # built as add_con(core, dims...), filled by add_con!
                all(a -> a isa Integer || a isa AbstractUnitRange, c.size) || throw(ArgumentError(
                    "ExaModelsIO cannot store constraint dimensions $(c.size)"))
                d["size"] = _dims(c.size)
                d["expr"] = nothing
            else
                d["expr"] = encode(c.f.f)
            end
        end
        d
    end

    structure = Dict(
        "eltype" => _number_type_name(T),
        "name" => string(m.name),
        "minimize" => m.meta.minimize,
        "nvar" => m.meta.nvar, "ncon" => m.meta.ncon, "npar" => length(m.θ),
        "variables" => variables, "parameters" => parameters,
        "objectives" => objectives, "constraints" => constraints,
    )
    return structure, columns
end
