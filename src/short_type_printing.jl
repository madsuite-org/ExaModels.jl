# Short printing of expression-tree types (`Node2{+,…}` instead of the full parametric
# type), defined by `enable_short_type_printing()`. These are `show` methods for types,
# so defining them invalidates compiled code that prints types; they are not defined
# at load. The methods are kept as a quoted block in the package image, so enabling them
# does not need the source files at run time (e.g. in a sysimage or app).
#
# Each method checks `get(io, :fulltype, _FULLTYPE_DISPLAY[])`. When false (the
# default), inner graph-structure parameters are replaced with `…` for readability.
# When true (set by `fulltype(node)` or `IOContext(io, :fulltype => true)`), the
# complete parametric type is printed instead.
#
# Use Type{<:X} (subtype constraint) instead of Type{X{A,B,...}} where {A,B,...}.
# The latter pattern with unconstrained type variables causes Julia's is_derived_type
# to recurse infinitely when checking if unrelated types (e.g. Symbol) match the
# signature during method dispatch compilation.

const _SHORT_TYPE_PRINTING_CODE = quote
    function Base.show(io::IO, t::Type{<:Constant})
        if t isa DataType && !isempty(t.parameters)
            print(io, "Constant{", t.parameters[1], "}")
        else
            print(io, "Constant")
        end
    end

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

    function Base.show(io::IO, ::Type{SumNode{I}}) where {I}
        print(io, "SumNode{…}")
    end

    function Base.show(io::IO, ::Type{ProdNode{I}}) where {I}
        print(io, "ProdNode{…}")
    end
end
