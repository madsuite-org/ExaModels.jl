"""
    read_model(path; backend = nothing, functions = Dict())

Read a model written by [`write_model`](@ref) and return a new `ExaModel`.

The model is rebuilt through ExaModels' modelling functions (`add_var`,
`add_par`, `add_obj`, `add_con`, `add_con!`), so it gets the derivative
structures of the ExaModels version doing the reading. `backend` places it on a
device as in `ExaCore(; backend)`.

Functions are stored by their name in the operator vocabulary (SPEC.md).
Functions from SpecialFunctions need that package loaded. A function the user
registered with ExaModels is stored as an extension: register it again and pass
it as `functions = Dict("MyModule.f" => f)`; the file lists the names it needs.
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
    v = VersionNumber(read(attrs["format_version"]))
    v.major == FORMAT_VERSION.major || throw(ArgumentError(
        "file has format version $(v.major).$(v.minor); this ExaModelsIO reads " *
        "version $(FORMAT_VERSION.major).x"))
end

_lengths(dims) = Tuple(d isa Integer ? Int(d) : Int(d[2] - d[1] + 1) for d in dims)
_axes(dims) = Tuple(d isa Integer ? Int(d) : Int(d[1]):Int(d[2]) for d in dims)
_name(n) = n === nothing ? nothing : Val(Symbol(n))
_shape(v, shape) = length(shape) == 1 ? v : reshape(v, Tuple(shape))

function _build(s, array, column, backend, functions)
    missing_ext = filter(k -> !haskey(functions, k), s["extensions"])
    isempty(missing_ext) || throw(ArgumentError(
        "the file uses user functions $(join(missing_ext, ", ")); register them with " *
        "ExaModels and pass them as `functions = Dict(name => f)`"))
    T = _NUMBER_TYPES[s["eltype"]]
    x0, lvar, uvar = array("x0"), array("lvar"), array("uvar")
    y0, lcon, ucon, θ = array("y0"), array("lcon"), array("ucon"), array("theta")

    c = ExaCore(T; backend, minimize = s["minimize"], name = Symbol(s["name"]))

    for v in s["variables"]
        r = v["offset"] .+ (1:prod(_lengths(v["size"])))
        c, var = ExaModels.add_var(c, _axes(v["size"])...;
            start = _shape(x0[r], _lengths(v["size"])), lvar = _shape(lvar[r], _lengths(v["size"])),
            uvar = _shape(uvar[r], _lengths(v["size"])), name = _name(v["name"]))
        _check_offset(var.offset, v["offset"], "variable")
    end
    for p in s["parameters"]
        r = p["offset"] .+ (1:prod(_lengths(p["size"])))
        c, par = ExaModels.add_par(c, _shape(θ[r], _lengths(p["size"])); name = _name(p["name"]))
        _check_offset(par.offset, p["offset"], "parameter")
    end

    iter(b) = haskey(b["iterator"], "product") ?
        Iterators.product((data_value(a, k -> column(string(b["group"], "/axis_$i/", k)))
                           for (i, a) in enumerate(b["iterator"]["product"]))...) :
        data_value(b["iterator"]["collection"], k -> column(string(b["group"], "/", k)))
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
            _check_offset(con.offset, b["offset"], "constraint")
            base[b["offset"]] = con
        else
            g = b["index"] === nothing ? gen(b) : Base.Generator(
                q -> decode(b["index"], q, functions) => decode(b["expr"], q, functions), iter(b))
            c, _ = ExaModels.add_con!(c, base[b["base_offset"]], g)
        end
    end
    m = ExaModel(c)
    # Variable and parameter indices in the file are global, so the rebuilt model
    # must lay out its blocks exactly as the file says; a mismatch would be silent.
    (m.meta.nvar, m.meta.ncon, length(m.θ)) == (s["nvar"], s["ncon"], s["npar"]) || error(
        "rebuilt model has $(m.meta.nvar) variables, $(m.meta.ncon) constraints, " *
        "$(length(m.θ)) parameters; the file says $(s["nvar"]), $(s["ncon"]), $(s["npar"])")
    return m
end

_check_offset(got, want, what) = got == want ||
    error("rebuilt $what block starts at offset $got; the file says $want")
