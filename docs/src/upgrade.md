# Upgrading from v0.9 to v0.10

This page documents the breaking API changes between ExaModels v0.9 and v0.10.
It is written to be directly usable as a prompt for a code-migration agent.

---

## Summary for an automated migration agent

The following is a complete, machine-readable description of every rename.
Apply each substitution globally to all `.jl` files that import ExaModels.

### 1. Model-building function and macro renames

Every old name has a direct one-to-one replacement.
The old mutating style (`x = variable(c, ...)`) is now deprecated; use the
functional style (`c, x = add_var(c, ...)`) or the macro style
(`@add_var(c, x, ...)`).

| v0.9 function | v0.10 function | v0.10 macro |
|---|---|---|
| `variable(c, ...)` | `add_var(c, ...)` → returns `(c, var)` | `@add_var(c, x, ...)` |
| `parameter(c, ...)` | `add_par(c, ...)` → returns `(c, par)` | `@add_par(c, θ, ...)` |
| `constraint(c, ...)` | `add_con(c, ...)` → returns `(c, con)` | `@add_con(c, g, ...)` |
| `constraint!(c, c1, ...)` | `add_con!(c, c1, ...)` → returns `(c, aug)` | `@add_con!(c, g, ...)` |
| `objective(c, ...)` | `add_obj(c, ...)` → returns `(c, obj)` | `@add_obj(c, f, ...)` |
| `subexpr(c, ...)` | `add_expr(c, ...)` → returns `(c, expr)` | `@add_expr(c, s, ...)` |

### 2. `ExaCore` is immutable

In v0.9, `ExaCore` was a mutable struct that was modified in-place by each
model-building call.

Since v0.10, `ExaCore` is an immutable struct: every model-building call
returns a new core, and the old mutating wrappers (`variable`, `parameter`,
`objective`, `constraint`, `constraint!`, `subexpr`) have been removed.

The `concrete` keyword selects how the core accumulates blocks.  The default,
`concrete = Val(false)`, uses type-erased storage: the core's type does not
change as blocks are added, so model construction compiles once instead of
once per block.  Pass `concrete = Val(true)` to keep every block in the
core's type, which is required for AOT compilation with `juliac --trim=safe`.
`concrete = Val(:erased)` stores blocks as the default does and also erases the
expression type from the handles `add_obj`, `add_con`, `add_con!` and
(non-buffered, non-lifted) `add_expr` return, so a model builder that holds many handles is not compiled
against their expression trees; it is not compatible with `juliac --trim=safe`.
All modes produce the same `ExaModel`:

```julia
# v0.9
c = ExaCore()
x = variable(c, 10; lvar = 0.0)
objective(c, x[i]^2 for i in 1:10)
m = ExaModel(c)

# v0.10 — functional style (recommended)
c = ExaCore()
c, x = add_var(c, 10; lvar = 0.0)
c, _  = add_obj(c, x[i]^2 for i in 1:10)
m = ExaModel(c)

# v0.10 — macro style (most concise)
c = ExaCore()
@add_var(c, x, 10; lvar = 0.0)
@add_obj(c, x[i]^2 for i in 1:10)
m = ExaModel(c)
```

### 3. Mechanical substitution rules

Apply these regex/string substitutions in order to any v0.9 file.
Each rule is written as `OLD → NEW`.

```
variable(     →  add_var(          # only when first arg is an ExaCore
parameter(    →  add_par(
objective(    →  add_obj(
constraint!(  →  add_con!(         # must come before constraint( rule
constraint(   →  add_con(
subexpr(      →  add_expr(
```

After renaming the call sites, update the call pattern and switch to the
immutable `ExaCore`:

```
# Old: result assigned directly, c mutated in-place
c = ExaCore()
x = variable(c, ...)

# New: functional pair destructuring, c rebound to updated immutable
c = ExaCore()
c, x = add_var(c, ...)
```

### 4. Complete before/after example

```julia
# ── v0.9 ────────────────────────────────────────────────────────────────────
using ExaModels

n = 100
c = ExaCore()
x = variable(c, n; lvar = -1.0, uvar = 1.0, start = 0.0)
θ = parameter(c, ones(n))
s = subexpr(c, θ[i] * x[i]^2 for i in 1:n)
g = constraint(c, x[i] + x[i+1] for i in 1:n-1; lcon = -1.0, ucon = 1.0)
constraint!(c, g, i => sin(x[i+1]) for i in 1:n-1)
objective(c, s[i] for i in 1:n)
m = ExaModel(c)

# ── v0.10 (functional) ───────────────────────────────────────────────────────
using ExaModels

n = 100
c = ExaCore()
c, x = add_var(c, n; lvar = -1.0, uvar = 1.0, start = 0.0)
c, θ = add_par(c, ones(n))
c, s = add_expr(c, θ[i] * x[i]^2 for i in 1:n)
c, g = add_con(c, x[i] + x[i+1] for i in 1:n-1; lcon = -1.0, ucon = 1.0)
c, _ = add_con!(c, g, i => sin(x[i+1]) for i in 1:n-1)
c, _ = add_obj(c, s[i] for i in 1:n)
m = ExaModel(c)

# ── v0.10 (macro) ────────────────────────────────────────────────────────────
using ExaModels

n = 100
c = ExaCore()
@add_var(c, x, n; lvar = -1.0, uvar = 1.0, start = 0.0)
@add_par(c, θ, ones(n))
@add_expr(c, s, θ[i] * x[i]^2 for i in 1:n)
@add_con(c, g, x[i] + x[i+1] for i in 1:n-1; lcon = -1.0, ucon = 1.0)
@add_con!(c, g, i => sin(x[i+1]) for i in 1:n-1)
@add_obj(c, s[i] for i in 1:n)
m = ExaModel(c)
```
