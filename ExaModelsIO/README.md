# ExaModelsIO.jl

Save an `ExaModel` to a file that later versions of ExaModels can read.

ExaModelsIO lives in the `ExaModelsIO/` directory of the ExaModels.jl repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/madsuite-org/ExaModels.jl", subdir = "ExaModelsIO")
```

```julia
using ExaModels, ExaModelsIO

write_model("model.h5", m)                     # m::ExaModel
m2 = read_model("model.h5")                    # a new ExaModel
m3 = read_model("model.h5"; backend = CUDABackend())

write_model("model.h5", m; solution = result)  # also store a solver result
read_solution("model.h5")                      # (solution = ..., multipliers = ..., ...)
```

## What is stored

The file is HDF5. It holds what the model is, not how ExaModels lays it out in
memory.

| path                | contents                                                     |
|:--------------------|:-------------------------------------------------------------|
| attributes of `/`   | `format`, `format_version`, `examodels_version`              |
| `/structure`        | JSON: variable and parameter blocks, and for each objective and constraint block its expression and the layout of its data |
| `/arrays`           | `x0`, `lvar`, `uvar`, `y0`, `lcon`, `ucon`, `theta`          |
| `/data/<block>/...` | the data each block iterates over, one array per field       |
| `/solution`         | optional solver output                                       |

An expression is a tree of plain JSON objects. Functions are named by their
qualified name (`"Base.+"`, `"Base.Math.sin"`), variables and parameters by
their global index, and generator data by its field path. For example,
`p.a * sin(y[p.i, p.j])` is stored as

```json
{"op": ["Base.*", {"data": ["a"]},
        {"op": ["Base.Math.sin", {"var": {"op": ["Base.+", ...]}}]}]}
```

Derivative structures and sparsity patterns are not stored. `read_model`
rebuilds the model with `add_var`, `add_par`, `add_obj`, `add_con` and
`add_con!`, so the model it returns has the internals of the ExaModels version
doing the reading. The arrays in `/arrays` and `/data` can be read by any HDF5
tool without Julia.

Generator data may be numbers, tuples, named tuples, or structs of these. A
struct (for example `ExaPowerIO.GenData`) is read back as a `NamedTuple` with
the same fields, so reading does not need the package that defined it.

## Limitations

- Functions registered with `@register_univariate` / `@register_bivariate`
  must be registered again before reading. If their module is not loaded, pass
  them as `read_model(path; functions = Dict("MyModule.f" => f))`.
- Block tags are not stored (a warning is printed).
- Models with oracles (`ExaModelWithOracle`) and two-stage models are not
  supported.
