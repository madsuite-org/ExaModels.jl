# ── AST transformation helpers (used by @add_obj, @add_con, @add_con!, @add_expr macros) ──────
#
# These functions walk generator expressions at macro-expansion time and perform
# two transformations on the generator body:
#
#   1. sum/prod rewriting — `sum(body for j in range)` is replaced by
#      `exa_sum(j -> body, Val(range))`.  The `Val`-wrapped range is hoisted
#      into a `let`-binding outside the generator closure so that
#      `juliac --trim=safe` can resolve `Val{N}` concretely via constant
#      propagation.
#
#   2. Selective Constant wrapping — non-iterator sub-expressions that evaluate
#      to a plain number at macro-expansion time (e.g. literal `2`, `3.14`, or
#      `2*pi`) are wrapped in [`Constant`](@ref) so that algebraic
#      simplification rules in `specialization.jl` (x*1 → x, x^2 → abs2(x),
#      etc.) fire at model-construction time.  Sub-expressions that reference
#      iterator variables are left as-is; they become plain scalars at runtime
#      and are handled by the `AbstractNode * Real` dispatch in
#      `register.jl`.

const _EXA_SUM_REF = GlobalRef(@__MODULE__, :exa_sum)
const _EXA_PROD_REF = GlobalRef(@__MODULE__, :exa_prod)

# Collect all iterator variable names from the LHS of a for-clause.
# Handles both simple symbols (`j`) and tuple destructuring (`(i, j)`).
function _collect_iter_vars!(vars::Set{Symbol}, lhs)
    return if lhs isa Symbol
        push!(vars, lhs)
    elseif lhs isa Expr && lhs.head == :tuple
        for arg in lhs.args
            _collect_iter_vars!(vars, arg)
        end
    end
end

# Entry point: locate generators inside an expression and process each one.
# Non-generator nodes are returned unchanged.
function _auto_const_gen(expr)
    expr isa Expr || return expr
    if expr.head == :generator
        return _process_generator(expr, Set{Symbol}())
    elseif expr.head == :flatten
        return Expr(:flatten, _auto_const_gen(expr.args[1]))
    elseif expr.head == :call
        # e.g. sum(generator) — recurse into call arguments but keep function name
        new_args = Any[expr.args[1]]
        for i in 2:length(expr.args)
            push!(new_args, _auto_const_gen(expr.args[i]))
        end
        return Expr(:call, new_args...)
    else
        return expr
    end
end

# Process a single generator node.  Collects its iterator variables (unioned
# with `outer_vars` from enclosing generators), then delegates to
# `_wrap_free_symbols` for the body.  Any `sum`/`prod` calls in the body are
# rewritten to `exa_sum`/`exa_prod` with their Val-ranges hoisted into a
# surrounding `let`-block.
function _process_generator(gen, outer_vars::Set{Symbol})
    iter_vars = copy(outer_vars)
    for spec in gen.args[2:end]
        _collect_iter_vars!(iter_vars, spec.args[1])
    end

    # Accumulates (gensym => Val(range)) pairs that must be bound OUTSIDE the
    # generator closure so that juliac --trim=safe can resolve Val{N} concretely
    # via constant propagation at the call site.
    hoisted = Pair{Symbol, Any}[]

    body = gen.args[1]
    if body isa Expr && body.head == :generator
        new_body = _process_generator(body, iter_vars)
    elseif body isa Expr && body.head == :flatten
        new_body = Expr(:flatten, _process_generator(body.args[1], iter_vars))
    else
        new_body = _wrap_free_symbols(body, iter_vars, hoisted)
    end

    result = Expr(:generator, new_body, gen.args[2:end]...)

    # Hoist Val bindings outside the generator so Julia's inference can
    # propagate Val{N} concretely into exa_sum / exa_prod.
    if !isempty(hoisted)
        binds = Expr(:block, [Expr(:(=), p.first, p.second) for p in hoisted]...)
        result = Expr(:let, binds, result)
    end

    return result
end

# Walk the body of a generator and apply one transformation:
#
#   • sum/prod(body for j in range) → exa_sum(j -> body, val_sym)
#     where val_sym is a hoisted Val(range) binding (see _process_generator).
#
# All other sub-expressions — including number literals and free symbols — are
# left as-is.  Numeric scalars are handled by the `AbstractNode * Real` dispatch
# in `register.jl`, which stores them directly in `Node2` as a concrete field.
# Users who want value-in-type optimisation (algebraic folding via
# `Constant{T}`) can call `Constant(v)` explicitly in their model expressions.
#
# Nodes that are NOT recursed into:
#   • Function names in call position
#   • Array bases in indexing expressions  (`a` in `a[i]`)
#   • Dot-access targets and quoted expressions
function _wrap_free_symbols(
        expr, iter_vars::Set{Symbol},
        hoisted::Vector{Pair{Symbol, Any}} = Pair{Symbol, Any}[]
    )
    if !(expr isa Expr)
        # Number literals, Symbols, LineNumberNodes — return as-is.
        # Numbers will be handled by the AbstractNode * Real dispatch at
        # model-construction time.
        return expr
    end

    h = expr.head
    if h == :call
        fn = expr.args[1]
        # sum(body for j in range) / prod(body for j in range) — single-spec generator only.
        # Transform to exa_sum(j -> wrapped_body, lo, val_sym) where val_sym is a
        # hoisted Val(length(range)) binding.  Hoisting outside the generator closure
        # lets juliac --trim=safe resolve Val{N} concretely via constant propagation.
        if length(expr.args) == 2 &&
                expr.args[2] isa Expr && expr.args[2].head == :generator &&
                length(expr.args[2].args) == 2 &&   # body + exactly one for-spec
                (
                fn === :sum || fn === :prod ||
                    fn isa GlobalRef && (fn.name === :sum || fn.name === :prod)
            )
            gen = expr.args[2]
            body = gen.args[1]
            spec = gen.args[2]            # Expr(:(=), iter_var_lhs, range_expr)
            iter_lhs = spec.args[1]
            range_expr = spec.args[2]
            nested_vars = copy(iter_vars)
            _collect_iter_vars!(nested_vars, iter_lhs)
            wrapped_body = _wrap_free_symbols(body, nested_vars, hoisted)
            ref = (fn === :sum || fn isa GlobalRef && fn.name === :sum) ?
                _EXA_SUM_REF : _EXA_PROD_REF
            # Hoist Val(range) outside the generator closure so juliac can
            # resolve the concrete type (e.g. Val{1:3}).
            val_sym = gensym(:exa_sum_val)
            push!(hoisted, val_sym => Expr(:call, :Val, range_expr))
            return Expr(:call, ref, Expr(:->, iter_lhs, wrapped_body), val_sym)
        end
        new_args = Any[expr.args[1]]  # keep function name as-is
        for i in 2:length(expr.args)
            push!(new_args, _wrap_free_symbols(expr.args[i], iter_vars, hoisted))
        end
        return Expr(:call, new_args...)
    elseif h == :ref
        new_args = Any[expr.args[1]]  # keep array base as-is
        for i in 2:length(expr.args)
            push!(new_args, _wrap_free_symbols(expr.args[i], iter_vars, hoisted))
        end
        return Expr(:ref, new_args...)
    elseif h == :generator
        nested_vars = copy(iter_vars)
        for spec in expr.args[2:end]
            _collect_iter_vars!(nested_vars, spec.args[1])
        end
        new_body = _wrap_free_symbols(expr.args[1], nested_vars, hoisted)
        return Expr(:generator, new_body, expr.args[2:end]...)
    elseif h == :->
        lambda_vars = copy(iter_vars)
        _collect_iter_vars!(lambda_vars, expr.args[1])
        new_body = _wrap_free_symbols(expr.args[2], lambda_vars, hoisted)
        return Expr(:->, expr.args[1], new_body)
    elseif h == :. || h == :quote || h == :macrocall
        return expr
    else
        return Expr(h, [_wrap_free_symbols(a, iter_vars, hoisted) for a in expr.args]...)
    end
end

# ── Operator dispatch for AbstractNode ────────────────────────────────────────
#
# The generic bivariate registrations in `register.jl` cover
# `AbstractNode OP AbstractNode`, `AbstractNode OP Real`, and
# `Real OP AbstractNode`.  The specialisations below handle cases where a
# more specific rule must take precedence.

# ── Constant algebraic simplifications ────────────────────────────────────────
# These fire at model-construction time whenever a Constant{T} node meets an
# AbstractNode operand, collapsing trivial graph branches before the model is
# finalised.  They are registered for the most common numeric zero/one/two
# values across all standard floating-point types.

# ── Power with compile-time integer exponent ──────────────────────────────────
# Julia's literal_pow fast-path keeps the exponent as Val{P}, enabling
# juliac --trim=safe to trace the exact Node2{^, …, Val{P}} constructor.
# The fallback `^(node, ::Integer)` / `^(node, ::Real)` embeds the exponent
# directly in Node2 as a concrete scalar.
@inline Base.literal_pow(::typeof(^), d1::AbstractNode, ::Val{P}) where {P} =
    _pow_val(d1, Val{P}())
@inline Base.:^(d1::AbstractNode, d2::Integer) = Node2(^, d1, d2)
@inline Base.:^(d1::AbstractNode, d2::Real) = Node2(^, d1, d2)
@inline _pow_val(d1::AbstractNode, ::Val{1}) = d1
@inline _pow_val(d1::AbstractNode, ::Val{2}) = Node1(abs2, d1)
@inline _pow_val(d1::AbstractNode, ::Val{V}) where {V} = Node2(^, d1, Val{V}())
# Make Val{V}() callable in the eval context so Node2(^, node, Val{V}())
# evaluates correctly as node^V.
@inline (::Val{V})(i, x, θ) where {V} = V

# ── exa_sum / exa_prod ────────────────────────────────────────────────────────
#
# Type-stable sum/prod for use inside @add_obj / @add_con / @add_expr generators.
#
# The macro transformation (_wrap_free_symbols) rewrites:
#   sum(body for j in range)  →  exa_sum(j -> body, Val(range))
#   prod(body for j in range) →  exa_prod(j -> body, Val(range))
#
# The Val(range) is hoisted into a let-binding outside the generator closure so
# that juliac --trim=safe can resolve Val{N} concretely via constant propagation.
#
# Supported iterator forms:
#   • Tuple        — tail-recursive _exa_map builds a concrete NTuple
#   • UnitRange{Int} — ntuple(f, Val{N}()) is type-stable when N is a
#                      compile-time constant (literal or Val-encoded length)
#   • Val{range}   — preferred for juliac; embeds the range in the type

# Helper: type-stable map over a tuple, returning a concrete NTuple of results.
@inline _exa_map(f, ::Tuple{}) = ()
@inline _exa_map(f, t::Tuple) = (f(t[1]), _exa_map(f, Base.tail(t))...)

# ── Tuple iterator ────────────────────────────────────────────────────────────

"""
    exa_sum(f, itr)
    exa_sum(f, ::Val{range})
    exa_sum(gen::Base.Generator)

Build a [`SumNode`](@ref) representing `∑ f(k)` for `k ∈ itr`. Inside `@add_obj`,
`@add_con`, and `@add_expr` macros, `sum(body for k in range)` is automatically
rewritten to `exa_sum(k -> body, Val(range))` with the `Val` hoisted outside
the generator closure.

# Supported iterators
- `Tuple`: type-stable via tail recursion.
- `UnitRange{Int}`: type-stable when the length is a compile-time constant.
- `Val{range}` (preferred for juliac): `exa_sum(f, Val(1:nc))` embeds the range
  in the type parameter, ensuring `juliac --trim=safe` can resolve `Val{N}`.

# juliac / AOT usage
Julia's inference cannot propagate constants like `nc = 3` through a generator
closure boundary.  When calling `add_con`/`add_obj` programmatically (not via
macro), hoist `Val(range)` outside the generator:

```julia
v = Val(1:nc)                                          # outside generator
c, con = add_con(c, (exa_sum(j -> x[j], v) for i in 1:nh))  # v captured
```
"""
@inline exa_sum(f, ::Tuple{}) = SumNode(())
@inline exa_sum(f, t::Tuple) = SumNode(_exa_map(f, t))

"""
    exa_prod(f, itr)
    exa_prod(f, ::Val{range})
    exa_prod(gen::Base.Generator)

Build a [`ProdNode`](@ref) representing `∏ f(k)` for `k ∈ itr`. Inside `@add_obj`,
`@add_con`, and `@add_expr` macros, `prod(body for k in range)` is automatically
rewritten to `exa_prod(k -> body, Val(range))` with the `Val` hoisted outside
the generator closure.

See [`exa_sum`](@ref) for supported iterators and juliac usage notes.
"""
@inline exa_prod(f, ::Tuple{}) = ProdNode(())
@inline exa_prod(f, t::Tuple) = ProdNode(_exa_map(f, t))

# ── UnitRange{Int}: Val{N}-based recursion ────────────────────────────────────

# UnitRange form (for direct use / Generator fallback — NOT type-stable inside
# generator closures for juliac; prefer the Val form below).
@inline exa_sum(f, r::UnitRange{Int}) = _exa_sum_range(f, first(r), Val(length(r)))
@inline exa_prod(f, r::UnitRange{Int}) = _exa_prod_range(f, first(r), Val(length(r)))

# Val-wrapped range form: Val(1:nc) embeds the range in the type parameter.
# The macro hoists `Val(range)` outside the generator closure so juliac can
# resolve the concrete Val{UnitRange{Int64}(1,3)} type.
# Users can also call this directly: `exa_sum(j -> x[j], Val(1:nc))`
@inline function exa_sum(f, ::Val{r}) where {r}
    return _exa_sum_range(f, first(r), Val(length(r)))
end
@inline function exa_prod(f, ::Val{r}) where {r}
    return _exa_prod_range(f, first(r), Val(length(r)))
end

@inline _exa_sum_range(f, lo::Int, ::Val{N}) where {N} =
    SumNode(ntuple(i -> f(lo + i - 1), Val{N}()))

@inline _exa_prod_range(f, lo::Int, ::Val{N}) where {N} =
    ProdNode(ntuple(i -> f(lo + i - 1), Val{N}()))

# ── Generator form (direct use outside macro) ─────────────────────────────────

@inline exa_sum(gen::Base.Generator) = exa_sum(gen.f, gen.iter)
@inline exa_prod(gen::Base.Generator) = exa_prod(gen.f, gen.iter)


# ── Algebraic simplification rules for Constant{T} ───────────────────────────
#
# These methods fire when a Constant{T} node appears in an arithmetic expression
# with another AbstractNode at model-construction time, eliminating unnecessary
# Node1/Node2 allocations before the graph is finalised.
#
# Rules are registered for the most common numeric zero/one/two values across
# Int, Float64, Float32, and Float16 so that expressions written with any of
# those literal types are simplified regardless of the calling context.

for zero in (0, 0.0, Float32(0.0), Float16(0.0))
    @eval begin
        Base.:+(::Constant{$zero}, x::AbstractNode) = x           # 0 + x  →  x
        Base.:+(x::AbstractNode, ::Constant{$zero}) = x           # x + 0  →  x
        Base.:-(::Constant{$zero}, x::AbstractNode) = -x          # 0 - x  → -x
        Base.:-(x::AbstractNode, ::Constant{$zero}) = x           # x - 0  →  x
        Base.:*(::Constant{$zero}, x::AbstractNode) = Constant(0) # 0 * x  →  0
        Base.:*(x::AbstractNode, ::Constant{$zero}) = Constant(0) # x * 0  →  0
        Base.:/(::Constant{$zero}, x::AbstractNode) = Constant(0) # 0 / x  →  0
        Base.:^(x::AbstractNode, ::Constant{$zero}) = Constant(1) # x ^ 0  →  1
    end
end

for one in (1, 1.0, Float32(1.0), Float16(1.0))
    @eval begin
        Base.:*(::Constant{$one}, x::AbstractNode) = x            # 1 * x  →  x
        Base.:*(x::AbstractNode, ::Constant{$one}) = x            # x * 1  →  x
        Base.:/(x::AbstractNode, ::Constant{$one}) = x            # x / 1  →  x
        Base.:/(::Constant{$one}, x::AbstractNode) = inv(x)       # 1 / x  →  inv(x)
        Base.:^(x::AbstractNode, ::Constant{-$one}) = inv(x)      # x ^ -1 →  inv(x)
        Base.:^(x::AbstractNode, ::Constant{$one}) = x            # x ^ 1  →  x
    end
end

for two in (2, 2.0, Float32(2.0), Float16(2.0))
    @eval begin
        Base.:^(x::AbstractNode, ::Constant{$two}) = abs2(x)      # x ^ 2  →  abs2(x)
    end
end
