# [Compiling Models to Shared Libraries](@id compiler)

`ExaModelsCompiler` compiles an `ExaCore` into a shared library that evaluates the
model through a plain C interface, the [cnlp ABI](https://cnlplib.madsuite.org).
The library needs no Julia installation on the consumer's side and no compilation at
run time. It can be used from Julia through
[CNLPLib.jl](https://github.com/madsuite-org/CNLPLib.jl), from Python through
`cnlpmodels`, and from C or any language that can call C.

A library compiled with `device = :cuda` also evaluates the model on an NVIDIA GPU, on
device pointers. The same file serves both: the CPU entry points are always present.

ExaModelsCompiler is registered; CNLPLib is installed from its repository:

```julia
using Pkg
Pkg.add("ExaModelsCompiler")
Pkg.add(url = "https://github.com/madsuite-org/CNLPLib.jl")
```

This page describes ExaModelsCompiler 0.4, whose libraries implement cnlp ABI 0.2 and are
read by CNLPLib 0.4.

Compiling requires Julia 1.12 or later (it uses `juliac` through JuliaC.jl, which is a
dependency) and takes from one to a few minutes per library.

## A first library

A model to be compiled is written as a recipe: a core built with `nargs = Val(1)`
whose size, or other data, is an argument that is only known when the model is
instantiated.

```julia
using ExaModels, ExaModelsCompiler

function rosenbrock()
    c, N = ExaCore(nargs = Val(1))
    @add_var(c, x, N; start = 0.5)
    @add_obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i in 2:N)
    @add_con(c, 3x[i+1]^3 + 2x[i+2] - 5 + sin(x[i+1] - x[i+2]) * sin(x[i+1] + x[i+2]) +
                4x[i+1] - x[i] * exp(x[i] - x[i+1]) - 3 for i in 1:N-2)
    return c
end

r = compile_library("rosen", rosenbrock(), 10; bundle = true)
r.libpath    # ".../rosen/lib/librosen.so"
```

The arguments after the core are an example, given as they would be to
`ExaModel(core, 10)`. The compiler uses it to check that the core instantiates and to
fix the argument types; the library is then instantiated at any size.

The library is consumed by name or by path:

```julia
using CNLPLib, MadNLP

m = CNLPModel("rosen", 100_000)      # rosen_new(100000)
madnlp(m)
```

`compile_library("@rosen", ...)` installs the library on the CNLPLib search path
(`CNLPLIB_PATH`), where `CNLPModel("@rosen", n)` finds it.

### Bundled or not

`bundle = true` writes a directory holding the library and a private copy of the Julia
runtime (about 80 MB), which runs anywhere. `bundle = false` (the default) writes a
single small library that links the Julia installation it was built with; the
consumer's machine needs the same Julia version. Use `bundle = true` for anything you
distribute, and for Julia consumers on macOS.

## Instantiation

What a model is instantiated from decides the entry points the library exports:

| Example arguments | Entry point | From Julia |
|:--|:--|:--|
| none (a core without placeholders) | `P_new(n)`, `n` ignored | `CNLPModel(lib)` |
| one integer | `P_new(n)` | `CNLPModel(lib, 1000)` |
| several values, arrays, tables | builder: `P_schema`, `P_data_begin`, `P_set_*`, `P_new_from_data` | `CNLPModel(lib, 3, [0.1, 0.2], table)` |
| `argfun = f` with a string example | `P_new_str(path)` | `CNLPModel(lib, "case118.m")` |

`P` is the model's symbol prefix, by default the library's name. Builder examples must
be `Int64` or `Float64` (scalars, vectors, or tables given as vectors of named tuples).

An argument function moves the work of reading input files into the library. `f` must
be a named function owned by a package (not a closure or a script function), and it
returns the argument tuple the core is instantiated with:

```julia
# in a package
case_args(path::AbstractString) = (parse_case(path),)

compile_library("acopf", ac_core, "case118.m"; argfun = case_args)
```

## Several models in one library

Models that share one library also share one runtime copy. Give them as name pairs; the
names are the prefixes:

```julia
compile_library("@grid",
    :acp => (acp_core, case_args, "case118.m"),
    :dcp => (dcp_core, case_args, "case118.m"),
    :fixed => small_core,                      # no instantiation data
)
CNLPModel("@grid", :acp, "case300.m")
```

A package that provides many models can implement `compile_all` (see its docstring) so
that one call compiles all of them.

## GPU evaluation

With CUDACore and GPUCompiler loaded at build time, `device = :cuda` adds a device tier
to the library:

```julia
using CUDACore, GPUCompiler, ExaModels, ExaModelsCompiler

compile_library("rosen", rosenbrock(), 10; device = :cuda, bundle = true)
```

The library then exports, in addition to the CPU entry points, `P_device(id)` (1 for
CUDA) and `P_obj_d`, `P_grad_d`, `P_cons_d`, `P_jac_d`, `P_hess_d`, which take device
pointers (except the objective's output, one host `double`). The kernels are compiled to
PTX at build time and take the problem size as an argument, so one library serves every
instance size. At run time the library needs only the NVIDIA driver (`libcuda`); it uses
the primary context of device 0, the same context CUDA.jl and the CUDA runtime use, so
device pointers from either work.

Prefer `CUDACore` over the `CUDA` umbrella package in the build environment: the
umbrella brings in packages (through `CUDATools`) that `juliac` cannot trim.

From Julia, the same library gives a model whose vectors live on the GPU, for a GPU
solver:

```julia
using CUDA, CUDSS, CNLPLib, MadNLPGPU         # MadNLPGPU needs CUDSS for GPU KKT systems

m = CNLPModel("rosen", 100_000; device = :cuda)   # a CNLPModel{CuVector{Float64}}
madnlp(m)
```

Any solver for `AbstractNLPModel{Float64, CuVector{Float64}}` works the same way.

Not supported on the device tier: models with buffered subexpressions
(`add_expr(...; buffered = Val(true))`) and models with more blocks than fit in a
tuple. Both are refused when compiling.

## Calling the library from C

The entry points are declared by [`cnlp.h`](https://github.com/madsuite-org/cnlplib/blob/main/cnlp.h).
Every integer is `int64_t`, indices are 1-based, the Hessian is the lower triangle of
``\sigma \nabla^2 f + \sum_i y_i \nabla^2 c_i`` with ``\sigma`` given as `obj_weight`,
and every function returns 0 on success. Nothing throws across the boundary.

This program evaluates the Luksan-Vlcek model of a library `libmodels.so`, compiled
with `:lv => (lv_core, 10)` and `device = :cuda`, on the CPU and on the GPU:

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cnlp.h"

CNLP_DECLARE_MODEL(lv)
CNLP_DECLARE_DEVICE(lv)

int main(void) {
    int64_t id = lv_new(1000);                 /* instance of size 1000 */
    if (id <= 0) return 1;
    int64_t n = lv_nvar(id), m = lv_ncon(id);
    double *x = malloc(n * sizeof *x), *lvar = malloc(n * sizeof *lvar),
           *uvar = malloc(n * sizeof *uvar), *lcon = malloc(m * sizeof *lcon),
           *ucon = malloc(m * sizeof *ucon), *g = malloc(n * sizeof *g);
    lv_meta(id, x, lvar, uvar, lcon, ucon);    /* x = starting point */

    double f;                                  /* CPU: host pointers */
    lv_obj(id, x, &f);
    lv_grad(id, x, g);

    if (lv_device(id) != 1) return 1;          /* GPU: device pointers */
    double *dx, *dg, fd, *gd = malloc(n * sizeof *gd);
    cudaMalloc((void **)&dx, n * sizeof(double));
    cudaMalloc((void **)&dg, n * sizeof(double));
    cudaMemcpy(dx, x, n * sizeof(double), cudaMemcpyHostToDevice);
    lv_obj_d(id, dx, &fd);                     /* the objective value comes back to the host */
    lv_grad_d(id, dx, dg);
    cudaMemcpy(gd, dg, n * sizeof(double), cudaMemcpyDeviceToHost);

    double e = 0;
    for (int64_t i = 0; i < n; i++) e = fmax(e, fabs(g[i] - gd[i]));
    printf("n = %lld, m = %lld, f = %.6f (GPU %.6f), max |grad - grad_d| = %.1e\n",
           (long long)n, (long long)m, f, fd, e);
    return 0;
}
```

```sh
cc -std=c99 -O2 -o lvdemo main.c -Lmodels/lib -lmodels \
   -Wl,-rpath,models/lib -lcudart -lm
./lvdemo
# n = 1000, m = 998, f = 6493.500000 (GPU 6493.500000), max |grad - grad_d| = 0.0e+00
```

(`cnlp.h` is not shipped with the library; take it from the cnlplib repository and
put it next to `main.c`.) A
consumer that only learns the model names at run time loads the library with `dlopen`
and asks `cnlp_nmodels` and `cnlp_model_name`, as the Julia and Python consumers do.
The first call starts the library's own runtime. Calls on the device tier read inputs
on the default stream, so a caller that wrote them on another stream synchronizes
first.

## Model sense, file formats and solutions

- **Sense.** `P_minimize(id)` is 1 or 0; maximization models are supported.
- **Solution writers.** `writers = (:matpower => write_case, ...)` compiles functions
  that write a solution in the model's own format, exported as `P_write_<name>`. Each
  is a named package function `f(out, arg, model, x, y, zl, zu; objective, status)`,
  called once on the example before compiling. `:default => :matpower` names the
  writer consumers use when no format is given.
- **Readers.** `readers = (instance = (:matpower => f,), start = (...), parameters = (...), default = :matpower)`
  compiles readers of the model's own formats: `P_new_from_<name>(path)` instantiates,
  `P_start_from_<name>` fills a starting point, `P_set_from_<name>` sets parameters.
- **Multiplier signs.** The ABI uses the convention of the
  [cnlp solution format](https://github.com/madsuite-org/cnlplib/blob/main/formats/solution.md),
  ``\sigma \nabla f + J^\top y - z_L + z_U = 0`` with ``\sigma = -1`` when maximizing.
  Writer and reader functions are written in the NLPModels convention (the one solver
  results use); the library converts between the two.

From Julia, `write_solution(m, "sol.h5", result)` writes the standard HDF5 solution
file, or uses the model's default writer, and `write_solution(m, :matpower, "sol.m", result)`
picks a writer by name.

## Limitations

- A library built on Windows works from C and Python but cannot be loaded back into
  Julia (`juliac` does not privatize the runtime there).
- An unbundled library can be loaded into Julia on Linux only; use `bundle = true`
  elsewhere.
- Functions that end up in a compiled library (argument functions, writers, readers,
  functions inside the core) must be named functions owned by a package, so that
  `juliac` can resolve them.
