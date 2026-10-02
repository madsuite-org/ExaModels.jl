# Automatic family merging

Merging happens during construction: when `add_con` or `add_con!` receives a
block whose expression tree type matches a family already in the core, the
block is folded into that family's single merged block, and the core's type
does not change. The first block of a family is stored plain, so models
without repeated structures are untouched. A block whose subtree sharing
diverges from the family's representative cannot be folded incrementally
(the slot layout is already fixed); those leftovers are merged in one batch
pass when `ExaModel(c)` is built. Nothing is requested by the user, and any
block the transform cannot prove safe stays separate.

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
- everything, when merging is off. `concrete = Val(true)` defaults to off so
  that model builders compiled with `juliac --trim` never reach the dynamic
  merge machinery; a concrete core built in a normal session can opt in with
  `ExaCore(concrete = Val(true), merge = true)`.

Set `ExaCore(merge = false)` to disable merging for a core, and
`ENV["EXAMODELS_MERGE_DEBUG"] = "1"` to print why blocks were refused.

## Footprint of spliced subexpressions

Slots are per leaf position, with no value-dependent sharing, so a tree that
splices a subexpression at several reference sites (`add_expr` without
`buffered` or `lift`) carries one derivative entry per site: the Jacobian
and Hessian COO buffers grow accordingly (duplicate coordinates, summed on
assembly; the assembled matrices are unchanged to the last bit).  On a
splice-heavy model this is a real memory cost — the CO2-capture MESH unit
measures nnzj 567 to 2087 and nnzh 2520 to 41640 — and the remedy is to
mark those subexpressions `buffered = true`, which evaluates them once per
row into a θ-backed stage and brings the footprint below the spliced
original (nnzj 804, nnzh 550 on the same model), or `lift = true`.  Models
without spliced subexpressions see no footprint change.

## Cost model

The merge machinery compiles once per family set per session; rebuilding
the same families, at any replication count, reuses all of it. Evaluation
performance is unchanged within measurement noise in most regimes and
faster where many blocks previously paid per-block overhead; the one
measured regression is 1.1 to 1.2 times on Jacobian/Hessian evaluation for
models consisting of a few shallow range-iterated blocks, where the per-row
offset loads are comparable to the kernel's arithmetic.
