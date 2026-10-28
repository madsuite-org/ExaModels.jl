"""
    ExaModelsIO

Saves an `ExaModel` to a file that a later ExaModels version can read back.

The file stores what the model *is*, not how ExaModels lays it out in memory:

- the expression of each objective and constraint block, as a small tree with
  operators named by string (`"+"`, `"sin"`), variables and parameters by global
  index, and generator data by field path;
- the data each block iterates over, one array per field;
- the arrays `x0`, `lvar`, `uvar`, `y0`, `lcon`, `ucon` and the parameter values.

Derivative structures, sparsity patterns and other internals are not stored;
[`read_model`](@ref) rebuilds them through ExaModels' modelling API.
"""
module ExaModelsIO

import ExaModels
import ExaModels: ExaCore, ExaModel
import NLPModels
import HDF5
import JSON3

export write_model, read_model, read_solution

const FORMAT = "ExaModelsIO"
const FORMAT_VERSION = 1

include("tree.jl")
include("data.jl")
include("write.jl")
include("read.jl")

end
