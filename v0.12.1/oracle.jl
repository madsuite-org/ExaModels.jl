# # [Oracle Constraints](@id oracle)

# ExaModels can combine its SIMD symbolic constraints with user-supplied
# "oracle" constraint blocks via [`VectorNonlinearOracle`](@ref).  An oracle
# encapsulates an opaque nonlinear function together with derivative
# callbacks — either sparse coordinate matrices (`jac!`/`hess!`) or
# matrix-free products (`jvp!`/`vjp!`/`hvp!`).
#
# This is useful when part of the model comes from an external source that
# ExaModels cannot trace symbolically: a neural network, a PDE solver,
# a GPU simulation, etc.

# ## Low-level API: `VectorNonlinearOracle`
#
# At the lowest level, you construct a `VectorNonlinearOracle` with callbacks
# that operate on the **full** problem variable vector, then register it with
# `constraint(core, oracle)`.  This gives full control over sparsity patterns
# and variable indexing.

# ## High-level API: `embed_oracle`
#
# For the common pattern of embedding an opaque function `f: ℝⁿ → ℝᵐ` into
# a model, `embed_oracle` handles all the plumbing automatically:
#
# 1. Creates auxiliary output variables `z ∈ ℝᵐ`
# 2. Registers an oracle constraint `z − f(x) = 0`
# 3. Returns the updated core and `z` as a regular `Variable` for use in any SIMD expression
#
# The callbacks operate on **local** vectors — no offset management needed:
#
# - `f!(y, x)`:          `y[1:m] = f(x[1:n])`
# - `jvp!(Jv, x, v)`:    `Jv[1:m] = J_f(x) * v[1:n]`
# - `vjp!(Jtv, x, w)`:   `Jtv[1:n] = J_f(x)' * w[1:m]`
# - `hvp!(Hv, x, w, v)`:  `Hv[1:n] = (Σ wᵢ ∇²fᵢ(x)) * v[1:n]`  (optional)
#
# When `adapt=Val(false)` (the default), all callbacks receive device arrays (e.g. `CuArray`)
# and must use broadcast operations — no scalar indexing.  Use `adapt=Val(true)` to have
# arrays automatically copied to CPU before each callback invocation.

# ## Example
#
# We embed an opaque element-wise squaring function `f(x)ᵢ = xᵢ²` and mix
# its output `z` with symbolic ExaModels expressions in the objective and
# constraints.  Using broadcast in the callbacks makes them work on both
# CPU and GPU without changes.
#
# ```math
# \min_{x} \sum_i \bigl(z_i + \sin(x_i)\bigr) \quad
# \text{s.t.} \quad z = f(x),\; x_i \ge 1 \;\forall i
# ```

using ExaModels
using CUDA
using MadNLP
using MadNLPGPU

N = 100

core = ExaCore(Float64; concrete = Val(true), backend = CUDABackend())
@add_var(core, x, N; start = (1.0 for _ in 1:N))

core, z, _ = embed_oracle(
    core, x, N;
    f! = (y, xv) -> (y .= xv .^ 2; nothing),
    jvp! = (Jv, xv, v) -> (Jv .= 2 .* xv .* v; nothing),
    vjp! = (Jtv, xv, w) -> (Jtv .= 2 .* xv .* w; nothing),
    hvp! = (Hv, xv, w, v) -> (Hv .= 2 .* w .* v; nothing),
    adapt = Val(false),
)

# The oracle output `z` can now be used in SIMD generator expressions,
# freely mixed with symbolic operations on `x`:

@add_obj(core, z[i] + sin(x[i]) for i in 1:N)
@add_con(core, x[i] for i in 1:N; lcon = 1.0, ucon = Inf)

model = ExaModel(core)
result = madnlp(model; print_level = MadNLP.INFO, linear_solver = LapackCUDASolver)

println("\nStatus:    ", result.status)
println("Objective: ", result.objective)
println("x[1:5]:    ", round.(result.solution[1:5]; digits = 6))
println("z[1:5]:    ", round.(result.solution[(N + 1):(N + 5)]; digits = 6))

# The optimal solution is xᵢ = 1, zᵢ = 1 for all i, with objective
# N × (1 + sin(1)) ≈ 184.15.
