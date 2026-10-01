# Pretty printing for node types

const _FULLTYPE_DISPLAY = Ref(false)

"""
    fulltype_display!(enabled::Bool)

Globally enable or disable full-type display for ExaModels node types.

When enabled, `typeof(node)` and error messages (MethodErrors, stacktraces)
show the complete parametric type instead of the abbreviated `…` form.

```julia
ExaModels.fulltype_display!(true)   # enable — e.g. typeof(x) → Node2{+,Var{Int64},Var{Int64}}
ExaModels.fulltype_display!(false)  # disable — e.g. typeof(x) → Node2{+,…}
```

For one-off inspection without changing the global setting, use [`fulltype`](@ref).
"""
fulltype_display!(v::Bool) = (_FULLTYPE_DISPLAY[] = v)

function _opname(::Type{F}) where {F}
    if F <: Function && hasmethod(nameof, Tuple{F})
        return string(nameof(F.instance))
    end
    return string(F)
end
_opname(::Type{FirstFixed{F}}) where {F} = _opname(F)
_opname(::Type{SecondFixed{F}}) where {F} = _opname(F)
_opname(::Type{typeof(+)}) = "+"
_opname(::Type{typeof(-)}) = "-"
_opname(::Type{typeof(*)}) = "*"
_opname(::Type{typeof(/)}) = "/"
_opname(::Type{typeof(^)}) = "^"

# Helper to print expression tree
function _print_tree(io::IO, node::Constant{T}, indent::Int) where {T}
    print(io, " "^indent, T)
end
function _print_tree(io::IO, node::Val{T}, indent::Int) where {T}
    print(io, " "^indent, T)
end
function _print_tree(io::IO, node::Null{Nothing}, indent::Int)
    print(io, " "^indent, "0")
end
function _print_tree(io::IO, node::Null{T}, indent::Int) where {T}
    print(io, " "^indent, node.value)
end
function _print_tree(io::IO, node::Var{I}, indent::Int) where {I<:AbstractNode}
    print(io, " "^indent, "x[")
    _print_tree(io, node.i, 0)
    print(io, "]")
end
function _print_tree(io::IO, node::Var{I}, indent::Int) where {I}
    print(io, " "^indent, "x[", node.i, "]")
end
function _print_tree(io::IO, node::ParameterNode{I}, indent::Int) where {I<:AbstractNode}
    print(io, " "^indent, "θ[")
    _print_tree(io, node.i, 0)
    print(io, "]")
end
function _print_tree(io::IO, node::ParameterNode{I}, indent::Int) where {I}
    print(io, " "^indent, "θ[", node.i, "]")
end
function _print_tree(io::IO, node::DataSource, indent::Int)
    print(io, " "^indent, "i")
end
function _print_tree(io::IO, node::DataIndexed{I,n}, indent::Int) where {I,n}
    _print_tree(io, getfield(node, :inner), 0)  # getfield bypasses getproperty override
    if n isa Integer
        print(io, "[", n, "]")
    else
        print(io, ".", n)
    end
end
function _print_tree(io::IO, node::VarSource, indent::Int)
    print(io, " "^indent, "x")
end
function _print_tree(io::IO, node::ParameterSource, indent::Int)
    print(io, " "^indent, "θ")
end
function _print_tree(io::IO, node::Node1{typeof(abs2)}, indent::Int)
    print(io, " "^indent)
    _print_tree(io, node.inner, 0)
    print(io, "^2")
end
function _print_tree(io::IO, node::Node1{F}, indent::Int) where {F}
    print(io, " "^indent, _opname(F), "(")
    _print_tree(io, node.inner, 0)
    print(io, ")")
end
# Check if a node is a zero constant
_is_zero(::Null{Nothing}) = true
_is_zero(n::Null) = n.value == 0
_is_zero(::Constant{0}) = true
_is_zero(::Constant) = false
_is_zero(::Val{0}) = true
_is_zero(::Val) = false
_is_zero(n::Real) = n == 0
_is_zero(_) = false

# Check if a node is a one constant
_is_one(n::Null) = n.value == 1
_is_one(::Constant{1}) = true
_is_one(::Constant) = false
_is_one(::Val{1}) = true
_is_one(::Val) = false
_is_one(n::Real) = n == 1
_is_one(_) = false

# Operator precedence for parenthesization
_op_prec(::Type{typeof(+)}) = 1
_op_prec(::Type{typeof(-)}) = 1
_op_prec(::Type{typeof(*)}) = 2
_op_prec(::Type{typeof(/)}) = 2
_op_prec(::Type{typeof(^)}) = 3
_op_prec(::Type) = 0

_node_prec(::Node2{F}) where {F} = _op_prec(F)
_node_prec(_) = 10  # leaves / function calls never need parens

# Print child, wrapping in parens if its precedence is lower than the parent's.
function _print_child(io::IO, child, parent_prec::Int)
    if _node_prec(child) < parent_prec
        print(io, "(")
        _print_tree(io, child, 0)
        print(io, ")")
    else
        _print_tree(io, child, 0)
    end
end

function _print_tree(io::IO, node::Node2{F}, indent::Int) where {F}
    op = _opname(F)
    a, b = node.inner1, node.inner2
    prec = _op_prec(F)
    # Simplify identity operations
    if op == "+" && _is_zero(b)
        return _print_tree(io, a, indent)
    elseif op == "+" && _is_zero(a)
        return _print_tree(io, b, indent)
    elseif op == "-" && _is_zero(b)
        return _print_tree(io, a, indent)
    elseif op == "*" && _is_one(b)
        return _print_tree(io, a, indent)
    elseif op == "*" && _is_one(a)
        return _print_tree(io, b, indent)
    elseif op == "*" && (_is_zero(a) || _is_zero(b))
        return print(io, " "^indent, "0")
    elseif op == "^" && _is_one(b)
        return _print_tree(io, a, indent)
    elseif op == "/" && _is_one(b)
        return _print_tree(io, a, indent)
    end
    if op == "^"
        print(io, " "^indent)
        _print_child(io, a, prec)
        print(io, "^")
        _print_child(io, b, prec)
    elseif op in ("*", "/", "+", "-")
        print(io, " "^indent)
        _print_child(io, a, prec)
        print(io, " ", op, " ")
        _print_child(io, b, prec)
    else
        print(io, " "^indent, op, "(")
        _print_tree(io, a, 0)
        print(io, ", ")
        _print_tree(io, b, 0)
        print(io, ")")
    end
end

# Handle Pair nodes (used in constraint augmentation: index => expr)
function _print_tree(io::IO, node::Pair{P,S}, indent::Int) where {P,S<:AbstractNode}
    _print_tree(io, node.second, indent)
end

# Compact expression string for a node
function _expr_string(node)
    buf = IOBuffer()
    _print_tree(buf, node, 0)
    s = String(take!(buf))
    # Strip outermost parentheses
    if length(s) >= 2 && s[1] == '(' && s[end] == ')'
        # Verify they are matching (not e.g. "(a) + (b)")
        depth = 0
        matched = true
        for (k, ch) in enumerate(s)
            depth += (ch == '(') - (ch == ')')
            if depth == 0 && k < length(s)
                matched = false
                break
            end
        end
        matched && return s[2:end-1]
    end
    return s
end

# Fallback for non-node types (e.g., Real constants in SIMDFunction.f)
function _print_tree(io::IO, node::Real, indent::Int)
    print(io, " "^indent, node)
end

# Generic fallback for unknown types
function _print_tree(io::IO, node, indent::Int)
    print(io, " "^indent, node)
end

# --- Symbolic expression nodes ---

function Base.show(io::IO, node::Constant{T}) where {T}
    print(io, T)
end
# Type show for new types
function Base.show(io::IO, t::Type{<:Constant})
    if t isa DataType && !isempty(t.parameters)
        print(io, "Constant{", t.parameters[1], "}")
    else
        print(io, "Constant")
    end
end

function Base.show(io::IO, node::Null{Nothing})
    print(io, "Null(0)")
end
function Base.show(io::IO, node::Null{T}) where {T}
    print(io, "Null(", node.value, ")")
end

function Base.show(io::IO, node::VarSource)
    print(io, "VarSource()")
end


function Base.show(io::IO, node::Var{I}) where {I}
    print(io, _expr_string(node))
end

function Base.show(io::IO, node::ParameterSource)
    print(io, "ParameterSource()")
end

function Base.show(io::IO, node::ParameterNode{I}) where {I}
    print(io, _expr_string(node))
end

function Base.show(io::IO, node::DataSource)
    print(io, "i")
end

function Base.show(io::IO, node::DataIndexed{I,n}) where {I,n}
    print(io, _expr_string(node))
end

function Base.show(io::IO, node::Node1{F,I}) where {F,I}
    print(io, _expr_string(node))
end

function Base.show(io::IO, node::Node2{F,I1,I2}) where {F,I1,I2}
    print(io, _expr_string(node))
end

# --- First-order adjoint nodes ---

function Base.show(io::IO, node::AdjointNull{V}) where {V}
    print(io, "AdjointNull(x = ", node.x, ")")
end

function Base.show(io::IO, node::AdjointNodeVar{I,T}) where {I,T}
    print(io, "AdjointNodeVar(i = ", node.i, ", x = ", node.x, ")")
end

function Base.show(io::IO, node::AdjointNode1{F,T,I}) where {F,T,I}
    print(io, "AdjointNode1{", _opname(F), "}(x = ", node.x, ", y = ", node.y, ")")
end

function Base.show(io::IO, node::AdjointNode2{F,T,I1,I2}) where {F,T,I1,I2}
    print(io, "AdjointNode2{", _opname(F), "}(x = ", node.x, ", y1 = ", node.y1, ", y2 = ", node.y2, ")")
end

function Base.show(io::IO, node::AdjointNodeSource{VT}) where {VT}
    print(io, "AdjointNodeSource{", VT, "}()")
end

# --- Second-order adjoint nodes ---

function Base.show(io::IO, node::SecondAdjointNull{V}) where {V}
    print(io, "SecondAdjointNull(x = ", node.x, ")")
end

function Base.show(io::IO, node::SecondAdjointNodeVar{I,T}) where {I,T}
    print(io, "SecondAdjointNodeVar(i = ", node.i, ", x = ", node.x, ")")
end

function Base.show(io::IO, node::SecondAdjointNode1{F,T,I}) where {F,T,I}
    print(io, "SecondAdjointNode1{", _opname(F), "}(x = ", node.x, ", y = ", node.y, ", h = ", node.h, ")")
end

function Base.show(io::IO, node::SecondAdjointNode2{F,T,I1,I2}) where {F,T,I1,I2}
    print(io, "SecondAdjointNode2{", _opname(F), "}(x = ", node.x,
        ", y1 = ", node.y1, ", y2 = ", node.y2,
        ", h11 = ", node.h11, ", h12 = ", node.h12, ", h22 = ", node.h22, ")")
end

function Base.show(io::IO, node::SecondAdjointNodeSource{VT}) where {VT}
    print(io, "SecondAdjointNodeSource{", VT, "}()")
end

# --- Short type name (use typeof(node) for full info) ---

"""
    fulltype(node)
    fulltype(io::IO, node)

Print the full unabbreviated type of a node.

By default, `show` for ExaModels node types abbreviates inner graph-structure
parameters with `…` to keep output readable (e.g. `Node2{+,…}` instead of
`Node2{typeof(+),Var{Int64},Var{Int64}}`).  `fulltype` bypasses this by
passing the `IOContext` key `:fulltype => true`, which all `show(::IO,
::Type{...})` overloads in this file respect.

## Examples

```julia
node = Var(1) + Var(2)          # Node2{typeof(+),Var{Int64},Var{Int64}}
show(stdout, typeof(node))      # → Node2{+,…}          (default)
fulltype(node)                  # → Node2{+,Var{Int64},Var{Int64}}

# Equivalent manual form:
show(IOContext(stdout, :fulltype => true), typeof(node))
```

## IOContext flag

Any `show` call that passes `:fulltype => true` in its `IOContext` will render
the complete parametric type.  This is useful when you need full type output
programmatically without calling `fulltype` directly:

```julia
io = IOContext(stderr, :fulltype => true)
show(io, typeof(node))
```
"""
fulltype(node) = fulltype(stdout, node)
function fulltype(io::IO, node)
    show(IOContext(io, :fulltype => true), typeof(node))
    println(io)
end

_short_type(::Null{Nothing}) = "Null"
_short_type(::Null{T}) where {T} = "Null{$T}"
_short_type(::Var) = "Var"
_short_type(::VarSource) = "VarSource"
_short_type(::ParameterSource) = "ParameterSource"
_short_type(::ParameterNode) = "ParameterNode"
_short_type(::DataSource) = "DataSource"
_short_type(::DataIndexed) = "DataIndexed"
_short_type(::Constant{T}) where {T} = "Constant{$T}"
_short_type(::Node1{F}) where {F} = "Node1{$(_opname(F))}"
_short_type(::Node2{F}) where {F} = "Node2{$(_opname(F))}"
_short_type(::AdjointNull) = "AdjointNull"
_short_type(::AdjointNodeVar) = "AdjointNodeVar"
_short_type(::AdjointNode1{F}) where {F} = "AdjointNode1{$(_opname(F))}"
_short_type(::AdjointNode2{F}) where {F} = "AdjointNode2{$(_opname(F))}"
_short_type(::AdjointNodeSource) = "AdjointNodeSource"
_short_type(::SecondAdjointNull) = "SecondAdjointNull"
_short_type(::SecondAdjointNodeVar) = "SecondAdjointNodeVar"
_short_type(::SecondAdjointNode1{F}) where {F} = "SecondAdjointNode1{$(_opname(F))}"
_short_type(::SecondAdjointNode2{F}) where {F} = "SecondAdjointNode2{$(_opname(F))}"
_short_type(::SecondAdjointNodeSource) = "SecondAdjointNodeSource"

# --- Short type printing (for stacktraces, MethodErrors, etc.) ---
#
# Each overload checks `get(io, :fulltype, _FULLTYPE_DISPLAY[])`.  When false (the default),
# inner graph-structure parameters are replaced with `…` for readability.
# When true (set by `fulltype(node)` or `IOContext(io, :fulltype => true)`),
# the complete parametric type is printed instead.

# Use Type{<:X} (subtype constraint) instead of Type{X{A,B,...}} where {A,B,...}.
# The latter pattern with unconstrained type variables causes Julia's is_derived_type
# to recurse infinitely when checking if unrelated types (e.g. Symbol) match the
# signature during method dispatch compilation.
function Base.show(io::IO, t::Type{<:Null})
    if t isa DataType && !isempty(t.parameters)
        T = t.parameters[1]
        T === Nothing ? print(io, "Null") : print(io, "Null{", T, "}")
    else
        print(io, "Null")
    end
end
function Base.show(io::IO, t::Type{<:Var})
    if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && t isa DataType && !isempty(t.parameters)
        print(io, "Var{", t.parameters[1], "}")
    else
        print(io, "Var{…}")
    end
end
function Base.show(io::IO, t::Type{<:ParameterNode})
    if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && t isa DataType && !isempty(t.parameters)
        print(io, "ParameterNode{", t.parameters[1], "}")
    else
        print(io, "ParameterNode{…}")
    end
end
function Base.show(io::IO, ::Type{<:DataIndexed})
    print(io, "DataIndexed{…}")
end
function Base.show(io::IO, t::Type{<:Node1})
    if t isa DataType && !isempty(t.parameters)
        F = t.parameters[1]
        if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && length(t.parameters) >= 2
            print(io, "Node1{", F, ",", t.parameters[2], "}")
        else
            print(io, "Node1{", _opname(F), ",…}")
        end
    else
        print(io, "Node1{…}")
    end
end
function Base.show(io::IO, t::Type{<:Node2})
    if t isa DataType && !isempty(t.parameters)
        F = t.parameters[1]
        if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && length(t.parameters) >= 3
            print(io, "Node2{", F, ",", t.parameters[2], ",", t.parameters[3], "}")
        else
            print(io, "Node2{", _opname(F), ",…}")
        end
    else
        print(io, "Node2{…}")
    end
end

function Base.show(io::IO, t::Type{<:AdjointNull})
    t isa DataType && !isempty(t.parameters) ? print(io, "AdjointNull{", t.parameters[1], "}") : print(io, "AdjointNull")
end
function Base.show(io::IO, t::Type{<:AdjointNodeVar})
    if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && t isa DataType && length(t.parameters) >= 2
        print(io, "AdjointNodeVar{", t.parameters[1], ",", t.parameters[2], "}")
    else
        T = (t isa DataType && length(t.parameters) >= 2) ? t.parameters[2] : "?"
        print(io, "AdjointNodeVar{…,", T, "}")
    end
end
function Base.show(io::IO, t::Type{<:AdjointNode1})
    if t isa DataType && !isempty(t.parameters)
        F, T = t.parameters[1], (length(t.parameters) >= 2 ? t.parameters[2] : "?")
        if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && length(t.parameters) >= 3
            print(io, "AdjointNode1{", F, ",", T, ",", t.parameters[3], "}")
        else
            print(io, "AdjointNode1{", _opname(F), ",", T, ",…}")
        end
    else
        print(io, "AdjointNode1{…}")
    end
end
function Base.show(io::IO, t::Type{<:AdjointNode2})
    if t isa DataType && !isempty(t.parameters)
        F, T = t.parameters[1], (length(t.parameters) >= 2 ? t.parameters[2] : "?")
        if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && length(t.parameters) >= 4
            print(io, "AdjointNode2{", F, ",", T, ",", t.parameters[3], ",", t.parameters[4], "}")
        else
            print(io, "AdjointNode2{", _opname(F), ",", T, ",…}")
        end
    else
        print(io, "AdjointNode2{…}")
    end
end
function Base.show(io::IO, t::Type{<:AdjointNodeSource})
    if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && t isa DataType && !isempty(t.parameters)
        print(io, "AdjointNodeSource{", t.parameters[1], "}")
    else
        print(io, "AdjointNodeSource{…}")
    end
end

function Base.show(io::IO, t::Type{<:SecondAdjointNull})
    t isa DataType && !isempty(t.parameters) ? print(io, "SecondAdjointNull{", t.parameters[1], "}") : print(io, "SecondAdjointNull")
end
function Base.show(io::IO, t::Type{<:SecondAdjointNodeVar})
    if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && t isa DataType && length(t.parameters) >= 2
        print(io, "SecondAdjointNodeVar{", t.parameters[1], ",", t.parameters[2], "}")
    else
        T = (t isa DataType && length(t.parameters) >= 2) ? t.parameters[2] : "?"
        print(io, "SecondAdjointNodeVar{…,", T, "}")
    end
end
function Base.show(io::IO, t::Type{<:SecondAdjointNode1})
    if t isa DataType && !isempty(t.parameters)
        F, T = t.parameters[1], (length(t.parameters) >= 2 ? t.parameters[2] : "?")
        if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && length(t.parameters) >= 3
            print(io, "SecondAdjointNode1{", F, ",", T, ",", t.parameters[3], "}")
        else
            print(io, "SecondAdjointNode1{", _opname(F), ",", T, ",…}")
        end
    else
        print(io, "SecondAdjointNode1{…}")
    end
end
function Base.show(io::IO, t::Type{<:SecondAdjointNode2})
    if t isa DataType && !isempty(t.parameters)
        F, T = t.parameters[1], (length(t.parameters) >= 2 ? t.parameters[2] : "?")
        if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && length(t.parameters) >= 4
            print(io, "SecondAdjointNode2{", F, ",", T, ",", t.parameters[3], ",", t.parameters[4], "}")
        else
            print(io, "SecondAdjointNode2{", _opname(F), ",", T, ",…}")
        end
    else
        print(io, "SecondAdjointNode2{…}")
    end
end
function Base.show(io::IO, t::Type{<:SecondAdjointNodeSource})
    if get(io, :fulltype, _FULLTYPE_DISPLAY[]) && t isa DataType && !isempty(t.parameters)
        print(io, "SecondAdjointNodeSource{", t.parameters[1], "}")
    else
        print(io, "SecondAdjointNodeSource{…}")
    end
end

# --- MIME "text/plain" for detailed display ---

function Base.show(io::IO, ::MIME"text/plain", node::AbstractNode)
    println(io, _short_type(node), "  (use fulltype(node) or IOContext(io, :fulltype => true) for full type)")
    print(io, "  ", _expr_string(node))
end

function Base.show(io::IO, ::MIME"text/plain", node::AdjointNode1{F,T,I}) where {F,T,I}
    println(io, _short_type(node))
    println(io, "  x = ", node.x, ", y = ", node.y)
    print(io, "  inner: ", node.inner)
end

function Base.show(io::IO, ::MIME"text/plain", node::AdjointNode2{F,T,I1,I2}) where {F,T,I1,I2}
    println(io, _short_type(node))
    println(io, "  x = ", node.x, ", y1 = ", node.y1, ", y2 = ", node.y2)
    println(io, "  inner1: ", node.inner1)
    print(io, "  inner2: ", node.inner2)
end

function Base.show(io::IO, ::MIME"text/plain", node::AdjointNull{V}) where {V}
    print(io, _short_type(node), "(x = ", node.x, ")")
end

function Base.show(io::IO, ::MIME"text/plain", node::AdjointNodeVar{I,T}) where {I,T}
    print(io, _short_type(node), "(i = ", node.i, ", x = ", node.x, ")")
end

function Base.show(io::IO, ::MIME"text/plain", node::SecondAdjointNode1{F,T,I}) where {F,T,I}
    println(io, _short_type(node))
    println(io, "  x = ", node.x, ", y = ", node.y, ", h = ", node.h)
    print(io, "  inner: ", node.inner)
end

function Base.show(io::IO, ::MIME"text/plain", node::SecondAdjointNode2{F,T,I1,I2}) where {F,T,I1,I2}
    println(io, _short_type(node))
    println(io, "  x = ", node.x, ", y1 = ", node.y1, ", y2 = ", node.y2)
    println(io, "  h11 = ", node.h11, ", h12 = ", node.h12, ", h22 = ", node.h22)
    println(io, "  inner1: ", node.inner1)
    print(io, "  inner2: ", node.inner2)
end

function Base.show(io::IO, ::MIME"text/plain", node::SecondAdjointNull{V}) where {V}
    print(io, _short_type(node), "(x = ", node.x, ")")
end

function Base.show(io::IO, ::MIME"text/plain", node::SecondAdjointNodeVar{I,T}) where {I,T}
    print(io, _short_type(node), "(i = ", node.i, ", x = ", node.x, ")")
end

_short_type(::SumNode) = "SumNode"
_short_type(::ProdNode) = "ProdNode"

# Pretty printing and show for SumNode / ProdNode (must be after struct definitions)
function Base.show(io::IO, node::SumNode)
    print(io, _expr_string(node))
end
function Base.show(io::IO, node::ProdNode)
    print(io, _expr_string(node))
end
function Base.show(io::IO, ::Type{SumNode{I}}) where {I}
    print(io, "SumNode{…}")
end
function Base.show(io::IO, ::Type{ProdNode{I}}) where {I}
    print(io, "ProdNode{…}")
end
function _print_tree(io::IO, node::SumNode, indent::Int)
    print(io, " "^indent)
    for (k, inner) in enumerate(node.inners)
        k > 1 && print(io, " + ")
        _print_tree(io, inner, 0)
    end
end
function _print_tree(io::IO, node::ProdNode, indent::Int)
    print(io, " "^indent)
    for (k, inner) in enumerate(node.inners)
        k > 1 && print(io, " * ")
        _print_tree(io, inner, 0)
    end
end
