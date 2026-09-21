# ── Abstract node types ───────────────────────────────────────────────────────

"""
    AbstractNode

Root abstract type for all nodes in the ExaModels symbolic computation graph.
Every node is callable as `node(i, x, θ)` and returns the scalar value of the
expression at the current evaluation point, where `i` is the iteration index,
`x` the primal variable vector, and `θ` the parameter vector.
"""
abstract type AbstractNode end

"""
    AbstractAdjointNode

Abstract type for nodes in the first-order (gradient) forward-pass tree.
"""
abstract type AbstractAdjointNode end

"""
    AbstractSecondAdjointNode

Abstract type for nodes in the second-order (Hessian) reverse-pass tree.
"""
abstract type AbstractSecondAdjointNode end

"""
    Null{T} <: AbstractNode

A leaf node that always evaluates to a fixed scalar.  The value is stored in a
field (as opposed to [`Constant`](@ref), which encodes the value as a type
parameter).  Used when the constant is not known at compile time.

`Null()` creates a zero-valued node; `Null(v)` creates a node that returns
`T(v)` where `T` is the element type of the primal vector.
"""
struct Null{T} <: AbstractNode
    value::T
end
Null() = Null(nothing)

"""
    Constant{T} <: AbstractNode

A compile-time constant node whose value `T` is embedded as a **type
parameter**.

## Design rationale

Because `T` is a type parameter rather than a struct field, the Julia compiler
(and `juliac --trim=safe`) can propagate `T` concretely through every
downstream `Node1` / `Node2` type, making the entire computation graph
type-stable with no runtime storage for the constant.

## Construction

```julia
Constant(2)      # → Constant{2}()
Constant(3.14)   # → Constant{3.14}()
```

## Evaluation

```julia
c = Constant(42)
c(i, x, θ)  # → 42
```

## Algebraic simplification

`specialization.jl` registers identities for `Constant` combined with
[`AbstractNode`](@ref) operands:

| expression             | simplification       |
|:-----------------------|:---------------------|
| `x + Constant(0)`      | `x`                  |
| `x * Constant(1)`      | `x`                  |
| `x / Constant(1)`      | `x`                  |
| `x ^ Constant(0)`      | `Constant(1)`        |
| `x ^ Constant(1)`      | `x`                  |
| `x ^ Constant(2)`      | `Node1(abs2, x)`     |
| `x ^ Constant(-1)`     | `inv(x)`             |
| `Constant(0) * x`      | `Constant(0)`        |
| `Constant(0) / x`      | `Constant(0)`        |

These rules fire at **model-construction time** (when the `ExaCore` is being
built), collapsing trivial branches before the graph is finalized.
"""
struct Constant{T} <: AbstractNode end
@inline Constant(T) = Constant{T}()
@inline (::Constant{T})(i, x, θ) where {T} = T

Base.:^(::Constant{0}, x::AbstractNode) = Constant(0)
Base.:^(::Constant{1}, x::AbstractNode) = Constant(1)




"""
    AdjointNull{V} <: AbstractAdjointNode

Leaf node in the first-order adjoint tree representing a constant (zero
contribution to the gradient).  Carries the primal value `x` so that upstream
nodes can read it without recomputation.
"""
struct AdjointNull{V} <: AbstractAdjointNode
    x::V
end

"""
    SecondAdjointNull{V} <: AbstractSecondAdjointNode

Leaf node in the second-order adjoint tree representing a constant (zero
contribution to both the gradient and the Hessian).  Carries the primal value
`x` for upstream consumption.
"""
struct SecondAdjointNull{V} <: AbstractSecondAdjointNode
    x::V
end

"""
    VarSource <: AbstractNode

Sentinel node used as the root of a variable array.  Indexing a `VarSource`
with an integer `i` returns `Var(i)`, building the leaf node that refers to
the `i`-th decision variable.
"""
struct VarSource <: AbstractNode end

"""
    Var{I} <: AbstractNode

A leaf node representing the `i`-th decision variable.

When evaluated as `v(i, x, θ)`, returns `x[v.i]` (or `x[v.i(i,x,θ)]` when
the index is itself a node).
"""
struct Var{I} <: AbstractNode
    i::I
end

struct ParameterSource <: AbstractNode end
"""
    ParameterNode{I, S} <: AbstractNode

A leaf node representing the `i`-th parameter.

`S` flags whether the sensitivity should be calculated for this parameter block.
"""
struct ParameterNode{I,S} <: AbstractNode
    i::I
end
@inline ParameterNode(i::I, ::Val{S} = Val(false)) where {I,S} = ParameterNode{I,S}(i)

"""
    ArgLeaf{A} <: AbstractNode

A scalar drawn from the model arguments, sitting inside the expression graph —
`h = 1 / (arg.N + 1)` used as a coefficient, or a variable offset that is not
known until the core is instantiated.

It borrows [`DataIndexed`](@ref)'s handling without being the same thing.  A
`DataIndexed` is a *parameter* of the pattern — its value arrives on every
kernel iteration, and the tree is complete without it.  An `ArgLeaf` is a
placeholder that will be gone: it is substituted once at instantiation, leaving
the plain `Real` a core built with concrete sizes would have had.

What the two share is that neither has a value while sparsity is being
discovered.  That discovery evaluates the graph at `Identity` against a
`NaNSource`, and structure does not depend on values, so both answer `NaN`
there — which is why this leaf needs no separate treatment in the probe.

Wrapping happens automatically: an [`AbstractArgNode`](@ref) combined with a
graph node through any registered operation becomes an `ArgLeaf` child of the
resulting [`Node2`](@ref).  Arithmetic *between* argument nodes stays in the
argument world and never reaches the graph.
"""
struct ArgLeaf{A} <: AbstractNode
    a::A
end


"""
    DataSource <: AbstractNode

Sentinel node used as the root of a parameter (data) array.  Indexing or
accessing fields on a `DataSource` returns a [`DataIndexed`](@ref) node that
encodes the access path as nested type parameters for zero-overhead data
lookup.
"""
struct DataSource <: AbstractNode end

"""
    DataIndexed{I, J} <: AbstractNode

A node representing the lookup `inner.J` (or `inner[J]`) where `inner` is a
`DataSource` or another `DataIndexed`.  The field/index `J` is encoded as a type
parameter so that `getfield` / `getindex` can be resolved at compile time.

Constructed implicitly via `getproperty` / `getindex` on a [`DataSource`](@ref).
"""
struct DataIndexed{I, J} <: AbstractNode
    inner::I
end

@inline DataIndexed(inner::I, n) where {I} = DataIndexed{I, n}(inner)
@inline DataIndexed(inner::I, s::Constant{n}) where {I, n} = DataIndexed{I, n}(inner)
"""
    Node1{F, I} <: AbstractNode

An interior node representing a unary operation `F` applied to one child node.

Evaluated as `F(inner(i, x, θ))`.

# Fields
- `inner::I`: child node
"""
struct Node1{F,I} <: AbstractNode
    inner::I
end

"""
    Node2{F, I1, I2} <: AbstractNode

An interior node representing a binary operation `F` applied to two children.
Either child can be a node or a plain scalar (e.g. `Real`), allowing numeric
coefficients to be stored directly without wrapping in [`Constant`](@ref).

Evaluated as `F(inner1(i, x, θ), inner2(i, x, θ))`.

# Fields
- `inner1::I1`: left child
- `inner2::I2`: right child
"""
struct Node2{F,I1,I2} <: AbstractNode
    inner1::I1
    inner2::I2
end

# ── Re-indexing a built subexpression ─────────────────────────────────────────
#
# `@add_expr` stores a body that is spliced in wherever it is referenced, so the
# stored form has to be re-indexable.  A closure did that by being called again
# with a new index; a node does it by substituting the `DataSource` it was built
# against.
#
# Storing a node rather than a closure is what makes subexpressions usable in a
# recipe.  A closure captures the `Variable`s it mentions, so when a size is a
# placeholder the `ArgSource` ends up inside the closure's *type*, where
# `instantiate` cannot reach it — and Julia emits no constructor for a closure,
# so it cannot be rebuilt field-wise either.  Nodes have constructors and
# dispatch, and the operation lives in the type parameter, so this walk is
# static: no reflection, no `@generated`, nothing `--trim=safe` will refuse.
@inline _reindex(::DataSource, i) = i
@inline _reindex(n::DataIndexed{I, J}, i) where {I, J} =
    _reindexed_access(_reindex(getfield(n, :inner), i), J)
# Once the source has been replaced by a concrete element — an integer, or the
# tuple a multi-dimensional or product-iterator generator destructures — the
# lookup is no longer symbolic and has to be performed, not rebuilt.  Leaving it
# as a `DataIndexed` around a raw tuple is what broke the multi-dim cases.
@inline _reindexed_access(inner::AbstractNode, J) = DataIndexed(inner, J)
@inline _reindexed_access(inner, J::Symbol) = getproperty(inner, J)
@inline _reindexed_access(inner, J) = getindex(inner, J)
@inline _reindex(n::Var, i) = Var(_reindex(getfield(n, :i), i))
@inline _reindex(n::ParameterNode{I,S}, i) where {I,S} = ParameterNode(_reindex(getfield(n, :i), i), Val(S))
@inline function _reindex(n::Node1{F}, i) where {F}
    # Julia synthesises no partially-parameterised constructor, so the child
    # types are given explicitly; they are known here, so this stays static.
    inner = _reindex(getfield(n, :inner), i)
    # A fully-concrete subterm is COMPUTED, not wrapped — the closure this
    # storage replaces evaluated such terms eagerly, and a node with only
    # Real children has no evaluation method (register.jl's one-sided Real
    # methods tie for two: `s[2]` substituting into `i - 1` was ambiguous at
    # the adjoint call). `F.instance` exists because ops are named functions.
    inner isa Real && return F.instance(inner)
    return Node1{F, typeof(inner)}(inner)
end
@inline function _reindex(n::Node2{F}, i) where {F}
    a = _reindex(getfield(n, :inner1), i)
    b = _reindex(getfield(n, :inner2), i)
    a isa Real && b isa Real && return F.instance(a, b)
    return Node2{F, typeof(a), typeof(b)}(a, b)
end
# Leaves with no data dependence — constants, variable/parameter sources, plain
# numbers — are their own re-indexing.
@inline _reindex(x, i) = x

struct FirstFixed{F}
    inner::F
end
struct SecondFixed{F}
    inner::F
end

@inline Base.getproperty(n::DataSource, s::Symbol) = DataIndexed(n, s)
@inline Base.getindex(n::DataSource, i) = DataIndexed(n, i)
@inline Base.indexed_iterate(n::P, idx, start = 1) where {P <: Union{DataSource, DataIndexed}} = (DataIndexed(n, idx), idx + 1)

@inline Base.getproperty(v::DataIndexed{I, n}, s::Symbol) where {I, n} = DataIndexed(v, s)
@inline Base.getindex(v::DataIndexed{I, n}, i) where {I, n} = DataIndexed(v, i)
@inline Base.indexed_iterate(v::DataIndexed{I, n}, idx, start = 1) where {I, n} = (DataIndexed(v, idx), idx + 1)


@inline Base.getindex(n::VarSource, i) = Var(i)
@inline Base.getindex(::ParameterSource, i) = ParameterNode(i)

@inline Node1(f::F, inner::I) where {F,I} = Node1{F,I}(inner)
@inline Node2(f::F, inner1::I1, inner2::I2) where {F,I1,I2} = Node2{F,I1,I2}(inner1, inner2)


struct Identity end

@inline (v::Var{I})(i, x, θ) where {I<:AbstractNode} = @inbounds x[v.i(i, x, θ)]
@inline (v::Var{I})(i, x, θ) where {I} = @inbounds x[v.i]
@inline (v::Var{I})(i::Identity, x, θ) where {I <: AbstractNode} = x[v]
@inline (v::Var{I})(i::Identity, x, θ) where {I <: Real} = x[v]

@inline (v::ParameterNode{I})(i, x, θ) where {I<:AbstractNode} = @inbounds θ[v.i(i, x, θ)]
@inline (v::ParameterNode{I})(::Any, x, θ) where {I} = @inbounds θ[v.i]
@inline (v::ParameterNode{I})(::Identity, x, θ) where {I<:AbstractNode} = @inbounds θ[v.i]

@inline (v::DataSource)(i, x, θ) = i
@inline (v::DataIndexed{I, n})(i, x, θ) where {I, n} = @inbounds getfield(getfield(v, :inner)(i, x, θ), n)

@inline (v::DataIndexed)(i::Identity, x, θ) = eltype(θ)(NaN)
@inline (v::DataSource)(i::Identity, x, θ) = eltype(θ)(NaN)

# An argument scalar is unknown at build time in exactly the sense a data field
# is, and answers the sparsity probe identically.
@inline (v::ArgLeaf)(i, x, θ) = eltype(θ)(NaN)
@inline (v::ArgLeaf)(i::Identity, x, θ) = eltype(θ)(NaN)

"""
    AdjointNode1{F, T, I} <: AbstractAdjointNode

Gradient-pass node for a unary operation `F`.  Stores the primal value and the
partial derivative alongside the child so a single forward sweep simultaneously
computes both.

# Fields
- `x::T`: primal value `F(inner.x)`
- `y::T`: derivative `F'(inner.x)`
- `inner::I`: child node
"""
struct AdjointNode1{F,T,I} <: AbstractAdjointNode
    x::T
    y::T
    inner::I
end

"""
    AdjointNode2{F, T, I1, I2} <: AbstractAdjointNode

Gradient-pass node for a binary operation `F`.  Stores the primal value and
both partial derivatives.

# Fields
- `x::T`: primal value `F(inner1.x, inner2.x)`
- `y1::T`: partial derivative w.r.t. first argument
- `y2::T`: partial derivative w.r.t. second argument
- `inner1::I1`: left child
- `inner2::I2`: right child
"""
struct AdjointNode2{F,T,I1,I2} <: AbstractAdjointNode
    x::T
    y1::T
    y2::T
    inner1::I1
    inner2::I2
end

"""
    AdjointNodeVar{I, T} <: AbstractAdjointNode

Leaf node in the gradient-pass tree corresponding to a decision variable.

# Fields
- `i::I`: variable index
- `x::T`: primal value `x[i]` at the current evaluation point
"""
struct AdjointNodeVar{I,T} <: AbstractAdjointNode
    i::I
    x::T
end

"""
    AdjointNodeSource{VT}

Factory for [`AdjointNodeVar`](@ref) leaves.  Indexing with `i` returns
`AdjointNodeVar(i, inner[i])`, seeding the gradient-pass tree with the current
primal value.

# Fields
- `inner::VT`: primal variable vector (or `nothing` for a zero-valued seed)
"""
struct AdjointNodeSource{VT}
    inner::VT
end

@inline AdjointNode1(f::F, x::T, y, inner::I) where {F,T,I} =
    AdjointNode1{F,T,I}(x, y, inner)
@inline AdjointNode2(f::F, x::T, y1, y2, inner1::I1, inner2::I2) where {F,T,I1,I2} =
    AdjointNode2{F,T,I1,I2}(x, y1, y2, inner1, inner2)

@inline Base.getindex(x::I, i) where {I<:AdjointNodeSource{Nothing}} =
    AdjointNodeVar(i, 0)
@inline Base.getindex(x::I, i) where {I<:AdjointNodeSource} =
    @inbounds AdjointNodeVar(i, x.inner[i])


"""
    SecondAdjointNode1{F, T, I} <: AbstractSecondAdjointNode

Hessian-pass node for a unary operation `F`.  A single forward sweep stores
the primal value, first derivative, and second derivative needed for the
reverse Hessian accumulation.

# Fields
- `x::T`: primal value `F(inner.x)`
- `y::T`: first derivative `F'(inner.x)`
- `h::T`: second derivative `F''(inner.x)`
- `inner::I`: child node
"""
struct SecondAdjointNode1{F,T,I} <: AbstractSecondAdjointNode
    x::T
    y::T
    h::T
    inner::I
end

"""
    SecondAdjointNode2{F, T, I1, I2} <: AbstractSecondAdjointNode

Hessian-pass node for a binary operation `F`.

# Fields
- `x::T`: primal value `F(inner1.x, inner2.x)`
- `y1::T`: partial derivative w.r.t. first argument
- `y2::T`: partial derivative w.r.t. second argument
- `h11::T`: second partial w.r.t. first argument
- `h12::T`: mixed second partial
- `h22::T`: second partial w.r.t. second argument
- `inner1::I1`: left child
- `inner2::I2`: right child
"""
struct SecondAdjointNode2{F,T,I1,I2} <: AbstractSecondAdjointNode
    x::T
    y1::T
    y2::T
    h11::T
    h12::T
    h22::T
    inner1::I1
    inner2::I2
end

"""
    SecondAdjointNodeVar{I, T} <: AbstractSecondAdjointNode

Leaf node in the Hessian-pass tree corresponding to a decision variable.

# Fields
- `i::I`: variable index
- `x::T`: primal value `x[i]` at the current evaluation point
"""
struct SecondAdjointNodeVar{I,T} <: AbstractSecondAdjointNode
    i::I
    x::T
end

"""
    SecondAdjointNodeSource{VT}

Factory for [`SecondAdjointNodeVar`](@ref) leaves.  Indexing with `i` returns
`SecondAdjointNodeVar(i, inner[i])`, seeding the Hessian-pass tree.

# Fields
- `inner::VT`: primal variable vector (or `nothing` for a zero-valued seed)
"""
struct SecondAdjointNodeSource{VT}
    inner::VT
end

@inline SecondAdjointNode1(f::F, x::T, y, h, inner::I) where {F,T,I} =
    SecondAdjointNode1{F,T,I}(x, y, h, inner)
@inline SecondAdjointNode2(
    f::F,
    x::T,
    y1,
    y2,
    h11,
    h12,
    h22,
    inner1::I1,
    inner2::I2,
) where {F,T,I1,I2} =
    SecondAdjointNode2{F,T,I1,I2}(x, y1, y2, h11, h12, h22, inner1, inner2)

@inline Base.getindex(x::I, i) where {I<:SecondAdjointNodeSource{Nothing}} =
    SecondAdjointNodeVar(i, 0)
@inline Base.getindex(x::I, i) where {I<:SecondAdjointNodeSource} =
    @inbounds SecondAdjointNodeVar(i, x.inner[i])


"""
    ParameterAdjointSource{VT}

Parameter vector that turns a `sensitivity = true` [`ParameterNode`](@ref) into an
[`AdjointNodeVar`](@ref) or [`SecondAdjointNodeVar`](@ref) leaf at index `i + offset`.

# Fields
- `inner::VT`: parameter vector
- `offset::Int`: leaf index offset
"""
struct ParameterAdjointSource{VT}
    inner::VT
    offset::Int
end
@inline Base.getindex(s::ParameterAdjointSource, i) = @inbounds s.inner[i]
Base.eltype(::Type{ParameterAdjointSource{VT}}) where {VT} = eltype(VT)

_leaf(::AdjointNodeSource) = AdjointNodeVar
_leaf(::SecondAdjointNodeSource) = SecondAdjointNodeVar

@inline function (v::ParameterNode{I,true})(i, x, θ::ParameterAdjointSource) where {I<:AbstractNode}
    j = v.i(i, x, θ)
    @inbounds _leaf(x)(j + θ.offset, θ.inner[j])
end
@inline (v::ParameterNode{I,true})(i, x, θ::ParameterAdjointSource) where {I} =
    @inbounds _leaf(x)(v.i + θ.offset, θ.inner[v.i])
@inline (v::ParameterNode{I,true})(::Identity, x, θ::ParameterAdjointSource) where {I<:AbstractNode} = x[v]
@inline (v::ParameterNode{I,true})(::Identity, x, θ::ParameterAdjointSource) where {I<:Real} = x[v]

"""
    ParameterAdjointResult{T, V} <: AbstractVector{T}

Output vector `v` of the `∂c/∂θ` and `∂²L/∂x∂θ` passes.

Keeps the entries of `θ` in `v` and removes the entires in `x`.

# Fields
- `v::V`: output vector
"""
struct ParameterAdjointResult{T,V} <: AbstractVector{T}
    v::V
end
@inline ParameterAdjointResult(v::V) where {V} = ParameterAdjointResult{eltype(V),V}(v)
Base.size(s::ParameterAdjointResult) = size(s.v)
@inline Base.getindex(s::ParameterAdjointResult{T}, i::Integer) where {T} =
    @inbounds i <= length(s.v) ? s.v[i] : zero(T)
@inline function Base.setindex!(s::ParameterAdjointResult, x, i::Integer)
    @inbounds i <= length(s.v) && (s.v[i] = x)
    return x
end

@inline (v::Null{Nothing})(i, x::V, θ) where {T,V<:AbstractVector{T}} = zero(T)
@inline (v::Null{N})(i, x::V, θ) where {N,T,V<:AbstractVector{T}} = T(v.value)
@inline (v::Null{Nothing})(i, x::AdjointNodeSource{T}, θ) where {T} = AdjointNull(zero(eltype(T)))
@inline (v::Null{N})(i, x::AdjointNodeSource{T}, θ) where {N, T} = AdjointNull(eltype(T)(v.value))
@inline (v::Null{Nothing})(i, x::SecondAdjointNodeSource{T}, θ) where {T} = SecondAdjointNull(zero(eltype(T)))
@inline (v::Null{N})(i, x::SecondAdjointNodeSource{T}, θ) where {N, T} = SecondAdjointNull(eltype(T)(v.value))

# ── SumNode / ProdNode ────────────────────────────────────────────────────────

"""
    SumNode{I} <: AbstractNode

A node representing the sum of a tuple of child nodes.

Constructed by [`exa_sum`](@ref).  Within `@add_obj`, `@add_con`, and `@add_expr` macros,
`sum(body for k in range)` is automatically rewritten to
`exa_sum(k -> body, Val(range))` with the `Val` hoisted outside the generator
closure for type stability under `juliac --trim=safe`.

In adjoint / second-adjoint mode the children are evaluated and folded via
`reduce(+, …)` (or `reduce(*, …)` for [`ProdNode`](@ref)), reusing the existing
registered `+` / `*` dispatch.  No dedicated adjoint node types are needed.
"""
struct SumNode{I} <: AbstractNode
    inners::I
end

"""
    ProdNode{I} <: AbstractNode

A node representing the product of a tuple of child nodes.

Constructed by [`exa_prod`](@ref).  See [`SumNode`](@ref) for design notes.
"""
struct ProdNode{I} <: AbstractNode
    inners::I
end

# ── Primal evaluation (x::AbstractVector → scalar) ───────────────────────────

@inline (n::SumNode{Tuple{}})(i, x::V, θ) where {T, V<:AbstractVector{T}} = zero(T)
@inline (n::SumNode)(i, x::V, θ) where {T, V<:AbstractVector{T}} =
    mapreduce(inner -> inner(i, x, θ), +, n.inners)

@inline (n::ProdNode{Tuple{}})(i, x::V, θ) where {T, V<:AbstractVector{T}} = one(T)
@inline (n::ProdNode)(i, x::V, θ) where {T, V<:AbstractVector{T}} =
    mapreduce(inner -> inner(i, x, θ), *, n.inners)

# ── Adjoint tree (gradient) ───────────────────────────────────────────────────

@inline (n::SumNode{Tuple{}})(i, x::AdjointNodeSource{VT}, θ) where {VT} =
    AdjointNull(zero(eltype(VT)))
@inline (n::SumNode)(i, x::AdjointNodeSource, θ) =
    reduce(+, map(inner -> inner(i, x, θ), n.inners))

@inline (n::ProdNode{Tuple{}})(i, x::AdjointNodeSource{VT}, θ) where {VT} =
    AdjointNull(one(eltype(VT)))
@inline (n::ProdNode)(i, x::AdjointNodeSource, θ) =
    reduce(*, map(inner -> inner(i, x, θ), n.inners))

# ── Second-adjoint tree (Hessian) ─────────────────────────────────────────────

@inline (n::SumNode{Tuple{}})(i, x::SecondAdjointNodeSource{VT}, θ) where {VT} =
    SecondAdjointNull(zero(eltype(VT)))
@inline (n::SumNode)(i, x::SecondAdjointNodeSource, θ) =
    reduce(+, map(inner -> inner(i, x, θ), n.inners))

@inline (n::ProdNode{Tuple{}})(i, x::SecondAdjointNodeSource{VT}, θ) where {VT} =
    SecondAdjointNull(one(eltype(VT)))
@inline (n::ProdNode)(i, x::SecondAdjointNodeSource, θ) =
    reduce(*, map(inner -> inner(i, x, θ), n.inners))

# Indexing an argument node with a *graph* node — `arg.b[i]` inside a generator,
# where `i` is the iteration index — cannot mean anything: `arg` is resolved
# once, at instantiation, while `i` varies per row.  Per-iteration data is what
# the generator's own iterator is for.  Caught here so it says so, rather than
# surfacing later as a MethodError on an unrelated operator.
@inline Base.getindex(::AbstractArgNode, ::AbstractNode) = throw(
    ArgumentError(
        "cannot index an `arg` expression with an iteration index: `arg` is " *
        "resolved once when the model is instantiated, so it cannot vary per " *
        "row. Pass per-iteration data through the generator instead — " *
        "`@add_con(core, x[d.i] - d.b for d in data)` — and keep `arg` for " *
        "sizes, starting points and bounds.",
    ),
)

# ── Instantiating a graph built against `arg` ────────────────────────────────
#
# Variable offsets are baked into `Var` indices at build time, so a graph built
# after a symbolic block carries argument nodes inside it.  These mirror
# `replace_T` in simdfunction.jl: rebuild the node with the child's new type,
# keeping the operation in the type parameter.  Leaves with no children
# (`Constant`, `Null`, `VarSource`, `DataSource`, …) fall through to the generic
# identity in argument.jl.

@inline function instantiate(n::Var{I}, a::Vararg{Any,N}) where {I, N}
    i = instantiate(n.i, a...)
    return Var{typeof(i)}(i)
end
@inline function instantiate(n::ParameterNode{I,S}, a::Vararg{Any,N}) where {I, S, N}
    i = instantiate(n.i, a...)
    return ParameterNode(i, Val(S))
end
@inline function instantiate(n::Node1{F,I}, a::Vararg{Any,N}) where {F,I, N}
    i = instantiate(n.inner, a...)
    return Node1{F,typeof(i)}(i)
end
@inline function instantiate(n::Node2{F,I1,I2}, a::Vararg{Any,N}) where {F,I1,I2, N}
    i1 = instantiate(n.inner1, a...)
    i2 = instantiate(n.inner2, a...)
    return Node2{F,typeof(i1),typeof(i2)}(i1, i2)
end
@inline instantiate(n::SumNode, a::Vararg{Any,N}) where {N} = SumNode(instantiate(n.inners, a...))
@inline instantiate(n::ProdNode, a::Vararg{Any,N}) where {N} = ProdNode(instantiate(n.inners, a...))
# `DataIndexed` overrides `getproperty` to keep building access paths, so its
# own field has to be read with `getfield`.
@inline instantiate(n::DataIndexed{I,J}, a::Vararg{Any,N}) where {I,J, N} =
    DataIndexed(instantiate(getfield(n, :inner), a...), J)
@inline instantiate(p::Pair, a::Vararg{Any,N}) where {N} = instantiate(p.first, a...) => instantiate(p.second, a...)
# An `ArgLeaf` does not survive instantiation: it *becomes* the scalar, leaving
# a graph indistinguishable from one built with concrete sizes.
@inline instantiate(n::ArgLeaf, a::Vararg{Any,N}) where {N} = instantiate(getfield(n, :a), a...)
