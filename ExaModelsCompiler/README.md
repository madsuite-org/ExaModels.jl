# ExaModelsCompiler

Compiles an ExaModels `ExaCore` into a shared library implementing the
[cnlp ABI](https://github.com/madsuite-org/cnlp-abi) 0.2: the model is evaluated through
a plain C interface, on the CPU and optionally on an NVIDIA GPU (`device = :cuda`), with
no Julia installation needed by the consumer and no compilation at run time. Libraries
are consumed from Julia by [CNLPLib.jl](https://github.com/madsuite-org/CNLPLib.jl),
from Python by `cnlpmodels`, and from C through `cnlp.h`.

```julia
using ExaModels, ExaModelsCompiler, CNLPLib, MadNLP

c, N = ExaCore(nargs = Val(1))
@add_var(c, x, N; start = 0.5)
@add_obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i in 2:N)
compile_library("rosen", c, 10; bundle = true)

madnlp(CNLPModel("rosen", 100_000))
```

Documentation: the [Compiling Models to Shared Libraries](https://madsuite-org.github.io/ExaModels.jl/dev/compiler/)
page of the ExaModels docs covers instantiation, several models per library, GPU
libraries, calling a library from C, and model file formats.
