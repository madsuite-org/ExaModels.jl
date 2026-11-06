"""
    precompile_callbacks(m::AbstractExaModel; prod = false)

Evaluate at `m`'s starting point every callback a solver uses: `jac_structure!`,
`hess_structure!`, `obj`, `grad!`, `cons!`, `jac_coord!`, and `hess_coord!` with and
without multipliers, also with the views of a longer buffer that MadNLP passes. Buffers
have `m`'s own array type, so on a device the kernels run there. Returns `nothing`.

With `prod = true`, also evaluate `jprod!`, `jtprod!` and `hprod!`. On a device these need
a model built with `ExaModel(c; prod = true)`. They are off by default because they add
to the code stored for each model.

Called from a `PrecompileTools.@compile_workload`, this stores the code of `m` in the
caller's package image. A model's type depends on its expressions, not on its data, so a
model of the same formulation but other data then builds and evaluates without compiling.

Each callback is called through `Base.invokelatest`, as from the top level or from behind
a solver's function barrier, so that it is compiled as a standalone method and not only
inlined into this function.

An error raised at the starting point, such as a `DomainError`, propagates.
"""
function precompile_callbacks(m::AbstractExaModel; prod = false)
    call = Base.invokelatest
    for get in _META_GETTERS
        call(get, m)
    end
    call(getproperty, call(getproperty, m, :meta), :x0)
    x = call(copy, call(NLPModels.get_x0, m))
    T = eltype(x)
    call(length, x)
    ncon = call(NLPModels.get_ncon, m)
    nnzj, nnzh = call(NLPModels.get_nnzj, m), call(NLPModels.get_nnzh, m)
    y = call(fill!, call(similar, x, ncon), zero(T))
    call(jac_structure!, m, call(similar, x, Int, nnzj), call(similar, x, Int, nnzj))
    call(hess_structure!, m, call(similar, x, Int, nnzh), call(similar, x, Int, nnzh))
    call(obj, m, x)
    call(grad!, m, x, call(similar, x))
    call(cons!, m, x, call(similar, x, ncon))
    call(jac_coord!, m, x, call(similar, x, nnzj))
    hess = call(similar, x, nnzh)
    call(hess_coord!, m, x, hess)
    call(hess_coord!, m, x, hess; obj_weight = one(T))
    call(hess_coord!, m, x, y, hess)
    call(hess_coord!, m, x, y, hess; obj_weight = one(T))
    call(jac_coord!, m, x, view(call(similar, x, nnzj + 1), 1:nnzj))
    call(hess_coord!, m, x, y, view(call(similar, x, nnzh + 1), 1:nnzh); obj_weight = one(T))
    if prod
        call(jprod!, m, x, call(similar, x), call(similar, x, ncon))
        call(jtprod!, m, x, y, call(similar, x))
        call(hprod!, m, x, call(similar, x), call(similar, x))
        call(hprod!, m, x, y, call(similar, x), call(similar, x))
    end
    # A keyword call from the top level builds this `NamedTuple` by itself.
    call(NamedTuple{(:obj_weight,)}, (one(T),))
    return nothing
end

const _META_GETTERS = (
    NLPModels.get_nvar,
    NLPModels.get_ncon,
    NLPModels.get_nnzj,
    NLPModels.get_nnzh,
    NLPModels.get_x0,
    NLPModels.get_y0,
    NLPModels.get_lvar,
    NLPModels.get_uvar,
    NLPModels.get_lcon,
    NLPModels.get_ucon,
)

# Model-independent code: the core, each kind of block, and model construction. A model's
# own expressions are compiled by the package that defines them (see
# `precompile_callbacks`). The callbacks run here as well: a method inlined in this image
# but not compiled standalone stays without native code, as a package loaded later cannot
# add it.
function _precompile_workload(backend = nothing; n = 4, prod = false)
    c = ExaCore(; backend)
    c, p = add_par(c, ones(n))
    c, x = add_var(c, n; start = ones(n), lvar = -2.0, uvar = 2.0)
    c, s = add_expr(c, x[i] * p[i] for i in 1:n)
    c, _ = add_obj(c, (s[i] - 1)^2 for i in 1:n)
    data = [(i = i, a = 0.5 * i) for i in 1:(n - 1)]
    c, g = add_con(c, x[d.i] * x[d.i + 1] + d.a for d in data; ucon = 5.0)
    c, _ = add_con!(c, g, d.i => x[d.i + 1]^2 for d in data)
    m = ExaModel(c; prod)
    precompile_callbacks(m; prod)
    return m
end

"""
    precompile_backend(backend)

Compile the model-independent code of `backend` for use in a
`PrecompileTools.@compile_workload`: building a model, sorting its sparsity patterns, and
the `sum` and `findall` kernels. The sizes reach the kernel variants that large models use.

The extensions for CUDA and AMDGPU call this when a device works during precompilation. For
another backend, call it from the workload of a package that needs it, in a process where
that backend works. Device kernels are stored only by backends whose compiled kernels
persist across sessions.
"""
function precompile_backend(backend)
    m = Base.invokelatest(_precompile_workload, backend; n = 1 << 12, prod = true)
    x = NLPModels.get_x0(m)
    sum(fill!(similar(x, 1 << 16), zero(eltype(x))))
    findall(fill!(similar(x, Bool, 1 << 21), true))
    return nothing
end

PrecompileTools.@compile_workload Base.invokelatest(_precompile_workload)
