# [Saving and Loading Models](@id io)

The `ExaModelsIO` subpackage saves an `ExaModel` to a file that later versions
of ExaModels can read. It lives in the `ExaModelsIO/` directory of the
ExaModels.jl repository and is installed from the General registry:

```julia
using Pkg
Pkg.add("ExaModelsIO")
```

`write_model` writes a model, and optionally a solver result, to an HDF5 file.
`read_model` rebuilds the model, on any backend, and `read_solution` returns the
stored result:

```julia
using ExaModels, ExaModelsIO

write_model("model.h5", m)                     # m::ExaModel
m2 = read_model("model.h5")                    # a new ExaModel
m3 = read_model("model.h5"; backend = CUDABackend())

write_model("model.h5", m; solution = result)  # also store a solver result
read_solution("model.h5")                      # (solution = ..., multipliers = ..., ...)
```

The file holds what the model is, not how ExaModels lays it out in memory: the
variable and parameter blocks, the expression of each objective and constraint
block as a tree of plain JSON objects, and the data each block iterates over as
plain arrays. Derivative structures and sparsity patterns are not stored.
`read_model` rebuilds the model with `add_var`, `add_par`, `add_obj`, `add_con`
and `add_con!`, so the model it returns has the internals of the ExaModels
version doing the reading.

The format is defined in
[`ExaModelsIO/SPEC.md`](https://github.com/madsuite-org/ExaModels.jl/blob/main/ExaModelsIO/SPEC.md).
The supported model features and the current limitations are listed in
[`ExaModelsIO/README.md`](https://github.com/madsuite-org/ExaModels.jl/blob/main/ExaModelsIO/README.md).
