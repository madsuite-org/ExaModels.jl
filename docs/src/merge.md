# Automatic family merging

When a model is built with the default (non-concrete) storage, `ExaModel(c)`
merges every group of constraint blocks that share one expression tree type
into a single block. The user does not request this; it happens on every
build, and any group the transform cannot prove safe is left as separate
blocks.

```julia
c = ExaCore()
c, x = add_var(c, 10)
c, y = add_var(c, 10)
c, _ = add_con(c, sin(x[i]) * x[i+1] for i in 1:9)
c, _ = add_con(c, sin(y[i]) * y[i+1] for i in 1:9)   # same structure, other variables
m = ExaModel(c)
length(m.cons)   # 1: one merged family
```

## Why

Every structurally distinct constraint block contributes a set of derivative
kernels, and every block, distinct or not, lengthens the tuple that the
evaluation callbacks are specialized over. A flowsheet that instantiates the
same unit model many times, a scenario loop, or any `add_con` in a loop
therefore pays compilation per block. After merging, the model carries one
block per constraint *family*, so compilation cost depends on how many kinds
of constraints the model has, not on how many times each kind is
instantiated. On GPUs the same collapse means one kernel launch per family
instead of one per block.

## What merging does

Two blocks are the same family when their expression trees have the same
type. The tree type fixes the structure; the values that differ between
blocks live in fields: variable-index offsets, scalar coefficients. Those
are hoisted out of the tree:

- every scalar leaf becomes per-row data, so the merged tree, element, and
  block types are a function of the family's tree type alone — building the
  same model at a different replication count produces the *same* model
  type, and all compiled code is reused;
- `Integer` leaves (they feed variable indices, which sparsity-structure
  evaluation needs with no parameter values available) travel in the
  iterator element;
- `AbstractFloat` leaves are appended to the parameter vector `θ`, stored
  once per source block and read as `θ[pbase + s]`. A merged coefficient is
  therefore also an updatable parameter;
- each row carries the row and nonzero offsets its source block gave it, so
  no constraint row, bound, sparsity position, or name handle moves.

For families whose row targets are affine (every plain `add_con` block), the
merged iterator is lazy: one small descriptor per source block, with the
original iterators (`UnitRange`s included) kept alive and rows materialized
on the fly. Memory cost is proportional to the number of blocks, not rows.
Augmentation (`add_con!`) families and all families on GPU backends use a
materialized element array instead.

## What does not merge

A group is silently left unmerged when the transform cannot prove the merged
block equivalent:

- trees containing node kinds outside the hoisting walk (`SumNode`,
  `ProdNode`, recipe placeholders);
- blocks whose merged tree would change the per-row sparsity footprint;
- blocks with mismatched tags or iterator element types;
- augmentation families on GPU backends (their accumulation uses the
  extension's collision-handling pipeline);
- everything, when the core was built with `concrete = Val(true)`: the merge
  pass is dynamic, so the statically compilable path never runs it.

Set `ExaModel(c; merge = false)` to skip merging entirely, and
`ENV["EXAMODELS_MERGE_DEBUG"] = "1"` to print why groups were refused.

## Cost model

The merge pass itself compiles once per family set per session; rebuilding
the same families, at any replication count, reuses all of it. Evaluation
performance is unchanged within measurement noise in most regimes and
faster where many blocks previously paid per-block overhead; the one
measured regression is 1.1 to 1.2 times on Jacobian/Hessian evaluation for
models consisting of a few shallow range-iterated blocks, where the per-row
offset loads are comparable to the kernel's arithmetic.
