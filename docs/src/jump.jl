# # JuMP Interface

# This tutorial explains how to use ExaModels with [JuMP](https://jump.dev).

# ## As an Optimizer

# ExaModels can be called from JuMP using `ExaModels.Optimizer`. The first
# argument to `Optimizer` is any NLPModels-compatible solver.

using JuMP
import ExaModels
import NLPModelsIpopt

N = 10
model = Model(() -> ExaModels.Optimizer(NLPModelsIpopt.ipopt))
@variable(model, x[i in 1:N], start = mod(i, 2) == 1 ? -1.2 : 1.0)
@constraint(
    model,
    [i in 1:(N-2)],
    (3 * x[i+1]^3 + 2 * x[i+2] - 5) +
    sin(x[i+1] - x[i+2]) * sin(x[i+1] + x[i+2]) +
    4 * x[i+1] - x[i] * exp(x[i] - x[i+1]) - 3 == 0.0
)
@objective(
    model,
    Min,
    sum(100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i in 2:N),
)
optimize!(model)

# Behind the scenes, `ExaModels.Optimizer` converts the JuMP model into an
# equivalent `ExaModels.ExaModel` before passing it to NLPModelsIpopt.

# For large structured nonlinear models, using `ExaModels.Optimizer` can be
# significantly faster than using `Ipopt.Optimizer` directly because
# `ExaModels.Optimizer` uses ExaModel's automatic differentiation routines
# instead using JuMP's default automatic differentiation library.

# ## Accessing the ExaModel

# You can also construct the ExaModel directly:

exa_model = ExaModels.ExaModel(model)

# ## Backends

# Pass a backend as the second argument of `ExaModels.Optimizer` to change the
# backend. For example, to create use a multi-threaded CPU routine, do:

import KernelAbstractions
set_optimizer(
    model,
    () -> ExaModels.Optimizer(NLPModelsIpopt.ipopt, KernelAbstractions.CPU()),
)
optimize!(model)

# ## GPUs

# To use MadNLP's GPU support, do:

import CUDA
import CUDSS
import MadNLP
import MadNLPGPU

set_optimizer(
    model,
    () -> ExaModels.Optimizer(MadNLP.madnlp, CUDA.CUDABackend()),
)
optimize!(model)
