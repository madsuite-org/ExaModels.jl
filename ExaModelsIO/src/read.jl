"""
    read_model(path; backend = nothing, functions = Dict())

Read a model written by [`write_model`](@ref) and return a new `ExaModel`.

The model is rebuilt through ExaModels' modelling functions (`add_var`,
`add_par`, `add_obj`, `add_con`, `add_con!`), so it gets the derivative
structures of the ExaModels version doing the reading. `backend` places it on a
device as in `ExaCore(; backend)`.

Functions are stored by qualified name (`"Base.Math.sin"`) and looked up in the
loaded modules. A function registered by the user with ExaModels must be
registered again before reading, and if its module is not loaded it can be
passed as `functions = Dict("MyModule.f" => f)`.
"""
function read_model(path::AbstractString; backend = nothing, functions = Dict{String,Any}())
    HDF5.h5open(path, "r") do f
        _check_format(f)
        s = JSON3.read(read(f["structure"]), Dict{String,Any})
        a(k) = read(f["arrays/$k"])
        return _build(s, a, k -> read(f[k]), backend, functions)
    end
end

"""
    read_solution(path)

Return the solver output stored by `write_model(path, model; solution)` as a
`NamedTuple`, or `nothing` if the file has none.
"""
function read_solution(path::AbstractString)
    HDF5.h5open(path, "r") do f
        haskey(f, "solution") || return nothing
        g = f["solution"]
        return NamedTuple(Symbol(k) => read(g[k]) for k in keys(g))
    end
end

function _check_format(f)
    attrs = HDF5.attributes(f)
    (haskey(attrs, "format") && read(attrs["format"]) == FORMAT) ||
        throw(ArgumentError("not an ExaModelsIO file"))
    v = read(attrs["format_version"])
    v <= FORMAT_VERSION || throw(ArgumentError(
        "file has format version $v; this ExaModelsIO reads up to $FORMAT_VERSION"))
end

_lengths(dims) = Tuple(d isa Integer ? Int(d) : Int(d[2] - d[1] + 1) for d in dims)
_axes(dims) = Tuple(d isa Integer ? Int(d) : Int(d[1]):Int(d[2]) for d in dims)
_name(n) = n === nothing ? nothing : Val(Symbol(n))
_shape(v, shape) = length(shape) == 1 ? v : reshape(v, Tuple(shape))

function _build(s, array, column, backend, functions)
    T = _NUMBER_TYPES[s["eltype"]]
    x0, lvar, uvar = array("x0"), array("lvar"), array("uvar")
    y0, lcon, ucon, θ = array("y0"), array("lcon"), array("ucon"), array("theta")

    c = ExaCore(T; backend, minimize = s["minimize"], name = Symbol(s["name"]))

    for v in s["variables"]
        r = v["offset"] .+ (1:prod(_lengths(v["size"])))
        c, _ = ExaModels.add_var(c, _axes(v["size"])...;
            start = _shape(x0[r], _lengths(v["size"])), lvar = _shape(lvar[r], _lengths(v["size"])),
            uvar = _shape(uvar[r], _lengths(v["size"])), name = _name(v["name"]))
    end
    for p in s["parameters"]
        r = p["offset"] .+ (1:prod(_lengths(p["size"])))
        c, _ = ExaModels.add_par(c, _shape(θ[r], _lengths(p["size"])); name = _name(p["name"]))
    end

    iter(b) = haskey(b, "axes") ?
        Iterators.product((data_value(a, k -> column(string(b["group"], "/axis_$i/", k)))
                           for (i, a) in enumerate(b["axes"]))...) :
        data_value(b["data"], k -> column(string(b["group"], "/", k)))
    gen(b) = Base.Generator(q -> decode(b["expr"], q, functions), iter(b))

    for o in s["objectives"]
        c, _ = ExaModels.add_obj(c, gen(o))
    end

    base = Dict{Int,Any}()
    for b in s["constraints"]
        if b["kind"] == "constraint"
            shape = Int.(b["shape"])
            r = b["offset"] .+ (1:prod(shape))
            kw = (start = _shape(y0[r], shape), lcon = _shape(lcon[r], shape),
                  ucon = _shape(ucon[r], shape), name = _name(b["name"]))
            c, con = b["expr"] === nothing ?
                ExaModels.add_con(c, _axes(b["size"])...; kw...) :
                ExaModels.add_con(c, gen(b); kw...)
            con.offset == b["offset"] || error("constraint offset mismatch while reading")
            base[b["offset"]] = con
        else
            g = b["index"] === nothing ? gen(b) : Base.Generator(
                q -> decode(b["index"], q, functions) => decode(b["expr"], q, functions), iter(b))
            c, _ = ExaModels.add_con!(c, base[b["base_offset"]], g)
        end
    end
    return ExaModel(c)
end
