# ── Symbolic build-time arguments ─────────────────────────────────────────────
#
# WHY THIS EXISTS
#
# An `ExaCore` is normally built from data already in hand, so the model and its
# data are finished together and cannot be separated afterwards.  That is fine
# until you want to compile the model ahead of time.
#
# Ahead-of-time compilation needs the opposite arrangement: the *structure* has
# to become code, while the *data* stays a run-time input.  Nothing stopped a
# user writing their own builder and pointing `juliac` at it, but
# `--trim=safe` requires the whole call graph to resolve statically, and which
# modelling constructs survive that is not something a model author can see
# while writing.  The usual way to find out is to compile, read a trimming
# error naming an inferred type somewhere in the machinery, and guess.
#
# A recipe removes the question.  Writing a model against `ArgSource`
# placeholders *is* the separation that compilation requires: what stays in the
# core becomes code, what goes into the arguments crosses the boundary as data.
# A recipe that builds and instantiates is on the compilation pathway, and the
# author never reasons about trimming.  `ExaModelsCompiler` takes it from there —
# shared library, C ABI, consumed by CNLPModels.jl or cnlpmodels.
#
# The same mechanism happens to make one core reusable at any size, which is
# convenient, but it is a consequence rather than the motivation.
#
# ── HOW IT RELATES TO DataSource ──
#
# `ArgSource` borrows `DataSource`'s *encoding* — a sentinel whose field/index
# access returns a node carrying the access path in a type parameter, so the
# lookup compiles away — but the two are different ideas, and it is worth being
# precise about which is which.
#
#   DataSource  — the PARAMETERIZATION of an algebraic pattern.  One expression
#                 tree stands for many constraint rows or objective terms, and
#                 the data point is what that tree is parameterized by: it is
#                 bound afresh on every kernel iteration, as `node(i, x, θ)`.
#                 Nothing is missing from such a tree; it is complete, and being
#                 evaluated repeatedly is the whole point.
#
#   ArgSource   — a genuine PLACEHOLDER.  It stands for a quantity not yet known
#                 while the model is being written — a size, a starting point, a
#                 bound, the set a generator runs over.  It is substituted once,
#                 by `instantiate(node, arg)`, and is gone afterwards: nothing
#                 ever evaluates it.  It lives in the slots the expression graph
#                 is built *from*, not inside the graph.
#
# So a core carrying `DataSource` nodes is a finished model; a core carrying
# `ArgSource` nodes is a recipe for one.
#
# This lets an `ExaCore` be written against arguments that are not yet known:
#
#     @add_var(c, x, arg.N; start = arg.v)
#     @add_con(c, x[i]^2 for i in 1:arg.N; lcon = 0.0, ucon = arg.nh * ones(10))
#
# (the range `1:arg.N` needs ExaModelsCompiler) and instantiated later.  Anything with no `ArgSource` in it is passed through
# by `instantiate` untouched, so ordinary (non-symbolic) models are unaffected.

"""
    AbstractArgNode

Root type for the symbolic *argument* expression tree.  Nodes are built by
ordinary Julia operations on an [`ArgSource`](@ref) (`arg.N`, `length(arg.v)`,
`arg.nh * zeros(10)`, …) and are resolved to concrete values by
[`instantiate`](@ref).  Ranges and type-directed constructors on arguments
(`1:arg.N`, `zeros(T, arg.N)`, `floor(Int, ·)`, `convert(T, ·)`) are defined by
ExaModelsCompiler, which needs Julia 1.12.
"""
abstract type AbstractArgNode end

"""
    ArgSource <: AbstractArgNode

Sentinel node standing for the whole argument object supplied at instantiation
time.  Field access or indexing returns an [`ArgIndexed`](@ref) node.

Create one where you need it and give it whatever name reads best — it carries
no state, so every instance is interchangeable:

```julia
arg = ArgSource()
@add_var(c, x, arg.N; start = arg.x0)
```
"""
struct ArgSource{K} <: AbstractArgNode end

@inline ArgSource() = ArgSource{1}()

"""
    ArgIndexed{I, J} <: AbstractArgNode

The lookup `inner.J` (or `inner[J]`) where `inner` is an [`ArgSource`](@ref) or
another `ArgIndexed`.  `J` is a type parameter, so `getproperty` / `getindex`
resolves at compile time — the same encoding [`DataIndexed`](@ref) uses.
"""
struct ArgIndexed{I, J} <: AbstractArgNode
    inner::I
end
@inline ArgIndexed(inner::I, j) where {I} = ArgIndexed{I, j}(inner)

"""
    ArgNode1{F, I} <: AbstractArgNode

A unary operation `f(inner)` deferred until instantiation.  `f` is stored as a
field; for the singleton function types this always holds in practice, that
field is zero-size and the call is statically resolved.
"""
struct ArgNode1{F, I} <: AbstractArgNode
    f::F
    inner::I
end

"""
    ArgNode2{F, I1, I2} <: AbstractArgNode

A binary operation `f(inner1, inner2)` deferred until instantiation.  Either
child may be a plain value (a `Number`, an `AbstractArray`, …) so that mixed
expressions such as `arg.nh * zeros(10)` are representable.
"""
struct ArgNode2{F, I1, I2} <: AbstractArgNode
    f::F
    inner1::I1
    inner2::I2
end

"""
    ArgCall{F, A} <: AbstractArgNode

A deferred call `f(args...)` of arbitrary arity.  `ArgNode1` / `ArgNode2` cover
the operator cases that users write and print nicely; `ArgCall` is what the
model-building internals use when an existing helper simply needs to run later
against instantiated inputs.

`args` is always a `Tuple`, and is written out at every call site.  A varargs
constructor would be shadowed by the default two-field one for the arity-1 case
— `ArgCall(f, x)` would quietly store `x` itself rather than `(x,)` — so there
deliberately isn't one.
"""
struct ArgCall{F, A <: Tuple} <: AbstractArgNode
    f::F
    args::A
end

# ── Access path construction ──────────────────────────────────────────────────
#
# Defined on the abstract type so every node stays symbolic under further
# access.  All internal field reads below therefore go through `getfield`.

@inline Base.getproperty(n::AbstractArgNode, s::Symbol) = ArgIndexed(n, s)
@inline Base.getindex(n::AbstractArgNode, j) = ArgIndexed(n, j)
@inline Base.indexed_iterate(n::AbstractArgNode, idx, state = 1) =
    (ArgIndexed(n, idx), idx + 1)

# ── Instantiation ─────────────────────────────────────────────────────────────

"""
    instantiate(x, args...)

Resolve every [`AbstractArgNode`](@ref) inside `x` against the concrete
argument objects `args`, and return the resulting value.  An
[`ArgSource`](@ref)`{K}` resolves to `args[K]`, so the order here is the order
the placeholders came out of [`ExaCore`](@ref).

`instantiate` is the identity on anything that carries no `ArgSource`
dependency — including containers, which are returned unchanged (`===`) rather
than rebuilt — so it is safe to apply to every slot of a model unconditionally.

## Example
```jldoctest
julia> using ExaModels

julia> arg = ArgSource();

julia> ExaModels.instantiate(arg.nh * ones(3), (nh = 2,))
3-element Vector{Float64}:
 2.0
 2.0
 2.0

julia> x = [1.0, 2.0];

julia> ExaModels.instantiate(x, (nh = 2,)) === x     # identity, no arg dependency
true
```
"""
@inline instantiate(x, a::Vararg{Any,N}) where {N} = x
@inline instantiate(::ArgSource{K}, a::Vararg{Any,N}) where {K, N} = a[K]
@inline instantiate(n::ArgIndexed{I, J}, a::Vararg{Any,N}) where {I, J, N} =
    _arg_access(instantiate(getfield(n, :inner), a...), J)
@inline instantiate(n::ArgNode1, a::Vararg{Any,N}) where {N} =
    getfield(n, :f)(instantiate(getfield(n, :inner), a...))
@inline instantiate(n::ArgNode2, a::Vararg{Any,N}) where {N} = getfield(n, :f)(
    instantiate(getfield(n, :inner1), a...),
    instantiate(getfield(n, :inner2), a...),
)
@inline instantiate(n::ArgCall, a::Vararg{Any,N}) where {N} =
    getfield(n, :f)(map(x -> instantiate(x, a...), getfield(n, :args))...)

@inline _arg_access(x, j::Symbol) = getproperty(x, j)
@inline _arg_access(x, j) = getindex(x, j)

# Containers map unconditionally — never guarded by a "does this contain an
# argument node?" predicate.  Such a predicate can only see what is *directly*
# in the tuple, so `core.var` — a tuple of `Variable` structs each holding
# placeholders — reads as clean and is handed back untouched, leaving a core
# that looks fully instantiated and is not.  Mapping costs nothing observable:
# an immutable tuple rebuilds `===` to itself, and each element is passed
# through by identity when it has no dependency of its own.
# `Vararg{Any,N}` forces specialization on the arguments' concrete types:
# a vararg that is only splatted through is otherwise left unspecialized
# (Julia's passthrough heuristic), and the `map` below then carries a dynamic
# call that `juliac --trim=safe` cannot resolve. Harmless under the JIT,
# load-bearing for AOT — same reason on the `ExaCore` method in nlp.jl.
# Unrolled as well as specialized: `Vararg{Any,N}` is necessary — without it
# the vararg is not specialized at all — but on a core the size of an AC
# OPF's the `map` closure is still an unresolved call under `--trim=safe`.
# The recursion gives each element a direct `instantiate` call.
#
# THE PARAMETER ROLES ARE LOAD-BEARING, both of them. The ELEMENTS stay
# structural (`t::Tuple`, recursed via `first`/`Base.tail`): splatting them
# into a bare vararg (`_f(a, x, xs...)`) puts the heterogeneous element
# types into a pass-through-only vararg, which Julia leaves unspecialized —
# the per-element types are lost, and a core whose objectives project
# fields out of a deferred data call (gasoil's shape) widens at the
# instantiate boundary: 6 verifier errors, reproduced and bisected to this
# hunk alone (6 → 0 on the swap, real COPS gasoil as the instrument). The
# ARGUMENTS ride the specialized `Vararg{Any,N}`. Inverting either role
# reintroduces the widening; the ExaModelsCompiler suite's models are too small
# and too homogeneous to see it — the COPS CI pin on this branch is the
# regression gate for this class.
@inline _instantiate_each(::Tuple{}, a::Vararg{Any,N}) where {N} = ()
@inline _instantiate_each(t::Tuple, a::Vararg{Any,N}) where {N} =
    (instantiate(first(t), a...), _instantiate_each(Base.tail(t), a...)...)
@inline instantiate(t::Tuple, a::Vararg{Any,N}) where {N} = _instantiate_each(t, a...)
@inline instantiate(t::NamedTuple{K}, a::Vararg{Any,N}) where {K, N} =
    NamedTuple{K}(_instantiate_each(Tuple(t), a...))

"""
    _anyarg(xs...)

`Val(true)` if any of `xs` is an [`AbstractArgNode`](@ref), `Val(false)`
otherwise — used to pick between the eager and the deferred form of a
build-time operation.

Resolved entirely by dispatch: each method returns a literal `Val`, and the
recursion is over the argument list, so the answer is part of the method
signature rather than a `Bool` that inference has to fold.  That matters for
`juliac --trim=safe`, where a `Val(::Bool)` built from a runtime value would
leave the choice — and both branches' return types — unresolved.
"""
@inline _anyarg() = Val(false)
@inline _anyarg(::AbstractArgNode, xs...) = Val(true)
@inline _anyarg(x, xs...) = _anyarg(xs...)

# ── Algebra ───────────────────────────────────────────────────────────────────
#
# Operands that may sit opposite an argument node.  Deliberately a closed union
# rather than `Any`: it keeps these methods from claiming dispatch they have no
# business in, and avoids ambiguities with Base's own array/number arithmetic.
const ArgOperand = Union{Number, AbstractArray, Tuple}

# Binary operations. Each gets three methods (node⊗node, node⊗value,
# value⊗node) so either or both sides may be symbolic.
#
# NOTE: the *stored* function must be spelled `Base.$op` too, not the bare
# symbol.  ExaModels shadows a few Base names in its own namespace (`size` in
# nlp.jl, `sort!` in templates.jl, `append!`), so a bare `$op` inside this
# module would capture the ExaModels method and `instantiate` would then call
# the wrong function on a perfectly ordinary array.
for op in (:+, :-, :*, :/, :\, :^, :%, :÷, :fld, :cld, :mod, :min, :max)
    @eval begin
        @inline Base.$op(a::AbstractArgNode, b::AbstractArgNode) = ArgNode2(Base.$op, a, b)
        @inline Base.$op(a::AbstractArgNode, b::ArgOperand) = ArgNode2(Base.$op, a, b)
        @inline Base.$op(a::ArgOperand, b::AbstractArgNode) = ArgNode2(Base.$op, a, b)
    end
end

# `x^2` lowers to `Base.literal_pow(^, x, Val(2))`, which would otherwise miss
# the `^` methods above.
@inline Base.literal_pow(::typeof(^), a::AbstractArgNode, ::Val{p}) where {p} =
    ArgNode2(^, a, p)

# Unary operations, and the queries that make sizes usable symbolically
# (`length(arg.v)`, `size(arg.A, 2)` via the binary table below, …).
for op in (
    :+, :-, :abs, :abs2, :sqrt, :cbrt, :inv, :exp, :log, :log2, :log10,
    :sin, :cos, :tan, :floor, :ceil, :round, :trunc, :sign,
    :length, :size, :ndims, :eltype, :first, :last, :axes, :collect,
    :sum, :prod, :maximum, :minimum, :zeros, :ones, :zero, :one, :transpose,
)
    @eval @inline Base.$op(a::AbstractArgNode) = ArgNode1(Base.$op, a)
end

# Two-argument forms of the same queries / constructors.  `getindex` is absent
# deliberately: indexing *into* an argument node is the access-path case above
# and must stay an `ArgIndexed`, so only the "concrete container, symbolic
# index" direction (`v[arg.i]`) is added here.
for op in (:size, :axes, :reshape, :fill, :repeat)
    @eval begin
        @inline Base.$op(a::AbstractArgNode, b::AbstractArgNode) = ArgNode2(Base.$op, a, b)
        @inline Base.$op(a::AbstractArgNode, b::ArgOperand) = ArgNode2(Base.$op, a, b)
        @inline Base.$op(a::ArgOperand, b::AbstractArgNode) = ArgNode2(Base.$op, a, b)
    end
end
# Split by concrete operand type rather than reusing `ArgOperand`: Base already
# has `getindex(::AbstractArray, I...)`, so the union form is ambiguous with it.
@inline Base.getindex(a::AbstractArray, b::AbstractArgNode) = ArgNode2(Base.getindex, a, b)
@inline Base.getindex(a::Tuple, b::AbstractArgNode) = ArgNode2(Base.getindex, a, b)
@inline Base.getindex(a::Number, b::AbstractArgNode) = ArgNode2(Base.getindex, a, b)

# Type-directed conversions and constructors (`floor(Int, arg.x)`, `zeros(Float64, arg.N)`,
# `convert(T, arg.x)`) and ranges (`1:arg.N`) are defined in ExaModelsCompiler, not here.
# They extend Base functions whose call sites are everywhere (`convert`, `:`) or dispatch
# on `::Type{T}` for any `T`, and loading them invalidates compiled code across the
# session, which costs every ExaModels user, including those whose recipes never use them.

# ── Broadcasting ──────────────────────────────────────────────────────────────
#
# Deliberately unsupported.  A deferred broadcast is implementable — it needs a
# `BroadcastStyle`, a `broadcastable`, and an override of Broadcast's own
# `instantiate` to skip axis computation — but nothing needs it: an array you
# want elementwise arithmetic on is an array you can build in the function that
# produces the arguments.  Refused explicitly, because the fallback
# `broadcastable(x) = collect(x)` would otherwise wrap the node in a `collect`
# and fail later with something unrecognisable.
Base.broadcastable(::AbstractArgNode) = throw(
    ArgumentError(
        "cannot broadcast over an argument placeholder. Compute the array in " *
        "the function that builds your arguments and pass it in — e.g. " *
        "`myargs(N) = (; N = N, scaled = 2 .* v)`, then refer to `arg.scaled` " *
        "— rather than writing the elementwise operation against `arg`.",
    ),
)

# ── Display ───────────────────────────────────────────────────────────────────

const _ARG_INFIX = Dict{Any, String}(
    (+) => " + ", (-) => " - ", (*) => " * ", (/) => " / ", (\) => " \\ ",
    (^) => "^", (%) => " % ", (÷) => " ÷ ", Colon() => ":",
)

_arg_string(x) = repr(x)
_arg_string(::ArgSource) = "arg"
function _arg_string(n::ArgIndexed{I, J}) where {I, J}
    inner = _arg_string(getfield(n, :inner))
    return J isa Symbol ? "$inner.$J" : "$inner[$(repr(J))]"
end
_arg_string(n::ArgNode1) =
    "$(_arg_fname(getfield(n, :f)))($(_arg_string(getfield(n, :inner))))"
function _arg_string(n::ArgNode2)
    f = getfield(n, :f)
    l = _arg_string(getfield(n, :inner1))
    r = _arg_string(getfield(n, :inner2))
    infix = get(_ARG_INFIX, f, nothing)
    return infix === nothing ? "$(_arg_fname(f))($l, $r)" : "($l$infix$r)"
end
_arg_string(n::ArgCall) =
    "$(_arg_fname(getfield(n, :f)))($(join(map(_arg_string, getfield(n, :args)), ", ")))"

_arg_fname(f) = string(f)
_arg_fname(f::Base.Fix1) = "$(string(f.f))($(string(f.x)), ·)"

Base.show(io::IO, n::AbstractArgNode) = print(io, _arg_string(n))
