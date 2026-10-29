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
        HDF5.attributes(f)["format_version"] = string(FORMAT_VERSION.major, ".", FORMAT_VERSION.minor)
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
    return hasproperty(s, :status) && (g["status"] = string(getproperty(s, :status)))
end

_dims(size) = Any[d isa Integer ? d : [first(d), last(d)] for d in size]

# The name under which a block is registered in `refs`, or nothing.
function _ref_name(refs, T::Type, offset)
    for (k, v) in pairs(refs)
        v isa T && v.offset == offset && return string(k)
    end
    return nothing
end

_warn_tag(tag) = tag === nothing ||
    @warn "ExaModelsIO does not store block tags; dropping tag $(tag)" maxlog = 1

function describe(m::ExaModel{T}) where {T}
    refs = m.refs === nothing ? (;) : m.refs
    columns = Pair{String, Vector}[]

    variables = map(collect(_in_order(m.vars))) do v
        _warn_tag(v.tag)
        Dict(
            "name" => _ref_name(refs, ExaModels.Variable, v.offset), "size" => _dims(v.size),
            "offset" => v.offset
        )
    end
    parameters = map(collect(_in_order(m.pars))) do p
        _warn_tag(p.tag)
        Dict(
            "name" => _ref_name(refs, ExaModels.Parameter, p.offset), "size" => _dims(p.size),
            "offset" => p.offset
        )
    end

    function store(group, itr)
        schema, cols = data_schema(itr)
        append!(columns, [string(group, "/", c) => v for (c, v) in cols])
        return schema
    end

    # A generator `for i in A, j in B` iterates the product of its axes, which
    # ExaModels keeps as the constraint's `size`.  Storing the axes instead of
    # the product rebuilds the same generator, and with it the same `size`.
    function block(kind, b::PlainBlock, k)
        _warn_tag(b.tag)
        group = "data/$(kind)_$k"
        d = Dict{String, Any}("group" => group)
        dims_form = b.tree isa ExaModels.Null && b.tree.value === nothing
        if b.size !== nothing && length(b.size) > 1 && !dims_form &&
                isequal(vec(collect(b.itr)), vec(collect(Iterators.product(b.size...))))
            d["iterator"] = Dict("product" => [store("$group/axis_$i", a) for (i, a) in enumerate(b.size)])
        elseif b.size !== nothing && length(b.size) == 1 && b.size[1] isa AbstractUnitRange &&
                !dims_form && isequal(collect(b.itr), collect(b.size[1]))
            # a collected range, e.g. a buffered block over `3:4`: store the range so
            # the rebuilt block has the same axis
            d["iterator"] = Dict("collection" => store(group, b.size[1]))
        else
            d["iterator"] = Dict("collection" => store(group, b.itr))
        end
        return d
    end

    objectives = map(enumerate(objective_blocks(m))) do (k, b)
        d = block("obj", b, k)
        d["expr"] = encode(b.tree)
        d
    end

    cons, augs = constraint_blocks(m)
    constraints = map(enumerate(cons)) do (k, b)
        d = block("con", b, k)
        d["kind"] = "constraint"
        d["name"] = _ref_name(refs, ExaModels.Constraint, b.offset)
        d["offset"] = b.offset
        d["shape"] = collect(size(b.itr))
        if b.tree isa ExaModels.Null && b.tree.value === nothing  # add_con(core, dims...), filled by add_con!
            all(a -> a isa Integer || a isa AbstractUnitRange, b.size) || throw(
                ArgumentError(
                    "ExaModelsIO cannot store constraint dimensions $(b.size)"
                )
            )
            d["size"] = _dims(b.size)
            d["expr"] = nothing
        else
            d["expr"] = encode(b.tree)
        end
        d
    end
    for (k, b) in enumerate(augs)
        d = block("aug", b, k)
        d["kind"] = "augmentation"
        d["base_offset"] = b.offset  # offset of the constraint it adds to
        if b.tree isa Pair
            d["index"] = encode(b.tree.first)
            d["expr"] = encode(b.tree.second)
        else  # element k adds to row k of the base constraint (as built by ExaModelsMOI)
            d["index"] = nothing
            d["expr"] = encode(b.tree)
        end
        push!(constraints, d)
    end

    subexpressions = map(enumerate(stage_blocks(m))) do (k, b)
        d = block("sub", b, k)
        d["offset"] = b.offset
        d["name"] = _has(:BufferedExpression) ?
            _ref_name(refs, ExaModels.BufferedExpression, b.offset) : nothing
        d["size"] = b.size !== nothing && all(a -> a isa Integer || a isa AbstractUnitRange, b.size) ?
            _dims(b.size) : nothing
        d["expr"] = encode(b.tree)
        d
    end

    exts = Set{String}()
    for b in Iterators.flatten((objectives, constraints, subexpressions)), k in ("expr", "index")
        _collect_extensions!(exts, get(b, k, nothing))
    end

    structure = Dict(
        "extensions" => sort!(collect(exts)),
        "eltype" => _number_type_name(T),
        "name" => string(m.name),
        "minimize" => m.meta.minimize,
        "nvar" => m.meta.nvar, "ncon" => m.meta.ncon, "npar" => length(m.θ),
        "variables" => variables, "parameters" => parameters,
        "subexpressions" => subexpressions,
        "objectives" => objectives, "constraints" => constraints,
    )
    return structure, columns
end

# The "ext:" operator names used anywhere in a tree.
function _collect_extensions!(exts, d)
    d isa AbstractDict || return exts
    for (k, v) in d
        if k == "op"
            startswith(v[1], EXT_PREFIX) && push!(exts, v[1][(length(EXT_PREFIX) + 1):end])
            foreach(a -> _collect_extensions!(exts, a), v[2:end])
        elseif v isa AbstractVector
            foreach(a -> _collect_extensions!(exts, a), v)
        else
            _collect_extensions!(exts, v)
        end
    end
    return exts
end
