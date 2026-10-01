# # [Model recipes: building once, instantiating with data](@id recipe)

# Ordinarily an `ExaCore` is built from data you already have: the sizes,
# starting point and bounds are known when you write `@add_var`, and the model
# that comes out is tied to them.

# A **model recipe** separates the two. You write the model against
# [`ArgSource`](@ref) placeholders obtained from `ExaCore(nargs = ...)` — leaving sizes, starting points, bounds and
# even the sets that generators iterate over open — and supply the data later:

# ```julia
# ExaModel(core, data)
# ```

# ## Why bother

# The same recipe instantiating at any size is convenient. The reason the
# concept exists, though, is **ahead-of-time compilation**.

# You can of course write your own model-building function and try to compile
# it. Nothing stops you, and for a modelling library of your own that may be
# exactly right. But AOT compilation imposes real requirements — `juliac
# --trim=safe` has to resolve the entire call graph statically — and which
# constructs survive that is rarely obvious while you are writing a model.
# Finding out usually means compiling, reading a trimming error that points at
# some inferred type deep in the machinery, and guessing what to change. Model
# developers should not have to hold that in their heads.

# A recipe is the mechanism that takes the question away. It draws a line
# between the part of a model that is *structure* — which becomes compiled code
# — and the part that is *data* — which crosses a boundary at run time. Writing
# a model in that form is the whole of what AOT compilation requires here: if a
# recipe builds and instantiates, it is on the compilation pathway, and you
# never reason about trimming yourself.

# What that pathway gives you:

# 1. the recipe, compiled once by `ExaModelsCompiler` into a self-contained shared
#    library, with the data left open;
# 2. a plain C interface on that library, so the caller needs no Julia;
# 3. consumption through
#    [CNLPModels.jl](https://github.com/MadNLP/CNLPModels.jl) or
#    [cnlpmodels](https://github.com/MadNLP/cnlpmodels-py), which present it as
#    an ordinary NLP to solvers on either side.

# The rest of this page walks that through: writing a recipe, instantiating it,
# and compiling it.

using ExaModels, NLPModels

# ## Argument placeholders

# Ask [`ExaCore`](@ref) for placeholders alongside the core. `nargs` says how
# many, and each one stands for one argument you will supply later:

core, N, x0 = ExaCore(concrete = Val(true), nargs = Val(2))

# `nargs = Val(0)` is the default and returns the core alone, exactly as
# before — nothing changes for models that do not use this.

# A placeholder is used *as the value it stands for*. `N` here is the number of
# variables, not a namespace to reach into, and ordinary arithmetic on it is
# deferred:

N, N - 1, 1:N

# ## Writing a recipe

# Write the model exactly as you would with real values:

@add_var(core, x, N; start = x0)
@add_obj(core, 100 * (x[i+1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:N-1)
@add_con(core, x[i] + x[i+1] for i = 1:N-1; lcon = -1.5, ucon = 1.5)

# That core is a recipe. It is **not a new kind of object** — it is an ordinary
# `ExaCore`, the very type you get without placeholders, holding symbolic
# expressions in some of its slots instead of numbers:

typeof(core) <: ExaCore

# which is why its variable count reads as an expression rather than a count:

core.nvar

# ## Instantiating it

# Supply one concrete value per placeholder, in the order they came back:

m10 = ExaModel(core, 10, fill(-1.2, 10))

# The *same* recipe instantiates again at a different size. It is not consumed
# by the first use, and the first model is unaffected:

m50 = ExaModel(core, 50, fill(-1.2, 50))

(m10.meta.nvar, m50.meta.nvar)

# Instantiated models are ordinary `ExaModel`s, so every solver in the
# JuliaSmoothOptimizers ecosystem takes them as usual.

NLPModels.obj(m10, m10.meta.x0)

# ## What can be deferred, and what should just be data

# Sizes, bounds, starting points, and the collections generators iterate over
# can all be placeholders. So can a loop-invariant scalar used as a coefficient
# inside an expression — `h` below resolves to an ordinary number when the model
# is built.

# What is *not* worth deferring is anything you can simply compute and pass in.
# A placeholder supports arithmetic, not arbitrary Julia: a comprehension over a
# deferred size, a random draw, or a table assembled from a file has no symbolic
# form. Compute those and hand them over as arguments — that is what the extra
# `nargs` are for:

function chain_pairs(n)
    ## whatever it takes — comprehensions, randomness, file parsing
    return [(i = i, j = j) for i = 1:n-1 for j = i+1:n]
end

chain, n, pairs, h = ExaCore(concrete = Val(true), nargs = Val(3))
@add_var(chain, z, n; start = 0.5)
## each `p` is an element of whatever is passed for `pairs`
@add_obj(chain, h * (z[p.i] - z[p.j])^2 for p in pairs)

m = ExaModel(chain, 6, chain_pairs(6), 1 / 7)
(m.meta.nvar, NLPModels.obj(m, m.meta.x0))

# This split — placeholders for the structure, computed values for everything
# else — is the shape to aim for. It is also exactly what a compiled library
# needs: the recipe becomes code, the arguments become the data crossing the
# boundary.

# !!! note
#     Per-row data belongs in the collection a generator iterates, not in a
#     field lookup on a placeholder. Indexing a placeholder with an iteration
#     index is an error: an argument is resolved once, while the index varies
#     per row. Pass the values as one of the arguments and iterate them —
#     `for p in pairs`, then `p.i` — which is the same idiom used for ordinary
#     (non-recipe) models.

# ## Compiling a recipe into a shared library

# Because a recipe is a complete model with its data factored out, it can be
# compiled ahead of time. `ExaModelsCompiler`, a subdirectory package of this
# repository, does that with
# [JuliaC](https://github.com/JuliaLang/juliac.jl), producing a self-contained
# `.so` that exposes the model over a plain C interface:

# ```julia
# using ExaModels, ExaModelsCompiler
#
# compile_library(recipe, "/opt/models/rosen";
#                 arg = (N = 10, x0 = zeros(10)))
# ```

# The `arg` you pass here is an *example*. Its values are never baked in — its
# **types** are. `juliac --trim=safe` has to resolve the whole call graph
# statically, so the example fixes what `N` *is* (an `Int`), while what it
# *equals* is supplied per instance at run time.

# The library exports, for prefix `P` (defaulting to the output directory's
# name): `P_new(n)` returning an instance id, then `P_nvar`, `P_ncon`,
# `P_nnzj`, `P_nnzh`, `P_meta`, `P_obj`, `P_grad`, `P_cons`, `P_jac`,
# `P_hess` and the two `_structure` functions. Indices are 1-based, the
# Hessian is the lower triangle of `obj_weight * ∇²f + Σᵢ yᵢ ∇²cᵢ`, and every
# function returns a `Cint` status rather than throwing across the boundary.

# ## Consuming the library

# The C interface is the one used by two companion packages, neither of which
# is part of ExaModels:

# * [CNLPModels.jl](https://github.com/MadNLP/CNLPModels.jl) — loads the
#   library as an `NLPModels.AbstractNLPModel`, so any
#   JuliaSmoothOptimizers-compatible solver can solve it.
# * [cnlpmodels](https://github.com/MadNLP/cnlpmodels-py) — the same consumer
#   for Python, over ctypes and numpy, requiring no Julia runtime on the
#   caller's side.

# From Julia:

# ```julia
# using CNLPModels, NLPModelsIpopt
#
# lib = CNLPModels.load("/opt/models/rosen/lib/librosen.so")
# m = CNLPModel(lib, 1000; prefix = "rosen")          # rosen_new(1000)
# result = ipopt(m)
# ```

# From Python:

# ```python
# import cnlpmodels
#
# lib = cnlpmodels.load("/opt/models/rosen/lib/librosen.so")
# m = cnlpmodels.CModel(lib, 1000, prefix="rosen")
# x, info = cnlpmodels.solve_ipopt(m)      # via cyipopt, if installed
# ```

# !!! warning
#     Both consumers default the symbol prefix to `"rec"` when given a library
#     handle, while `compile_library` defaults it to the output directory's
#     name. Pass `prefix` explicitly unless the two happen to agree.

# `ExaModelsCompiler` currently emits the scalar instantiation ABI, so the example
# `arg` must be an `Integer` or a `NamedTuple` holding exactly one integer
# field. An example of any other shape is refused with an explanation rather
# than compiled into a library that would fail to load.
