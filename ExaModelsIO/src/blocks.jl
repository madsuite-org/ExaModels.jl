# A model as plain blocks: one expression over one collection each, the form a
# file stores. ExaModels may hold a block in another form (merged families,
# where several source blocks share one tree with their scalar leaves hoisted
# into the rows); this file undoes that, so the writer never sees it.

struct PlainBlock
    tree::Any          # expression; for an augmentation, `idx => expr` or a bare expr
    itr::Any           # the collection iterated over
    offset::Int        # first row (constraint), base row offset (augmentation), θ offset (stage)
    size::Any          # constraint dimensions, or nothing
    tag::Any
end

_has(s) = isdefined(ExaModels, s)

# ---------------------------------------------------------------- merged families

_is_merged(itr) = _has(:SegmentedItr) && itr isa ExaModels.SegmentedItr
_is_merged_rows(itr) = (_has(:MergedRows) && itr isa ExaModels.MergedRows) ||
    (_has(:DeviceSegItr) && itr isa ExaModels.DeviceSegItr)

# The source tree of one segment: hoisted leaves get their values back, and the
# row element `row.d` becomes the element itself again.
_unhoist(n, iv, fv) = n
_unhoist(n::ExaModels.Node1{F}, iv, fv) where {F} = F.instance(_unhoist(n.inner, iv, fv))
function _unhoist(n::ExaModels.Node2{F}, iv, fv) where {F}
    a, b = _unhoist(n.inner1, iv, fv), _unhoist(n.inner2, iv, fv)
    # x^3 with a literal exponent holds Val(3); rebuild it as Julia does
    F === typeof(^) && b isa Val && return Base.literal_pow(^, a, b)
    return F.instance(a, b)
end
_unhoist(n::ExaModels.Var, iv, fv) = ExaModels.Var(_unhoist(n.i, iv, fv))
_unhoist(n::ExaModels.ParameterNode, iv, fv) = ExaModels.ParameterNode(_unhoist(n.i, iv, fv))
_unhoist(n::ExaModels.SumNode, iv, fv) = ExaModels.SumNode(map(a -> _unhoist(a, iv, fv), n.inners))
_unhoist(n::ExaModels.ProdNode, iv, fv) = ExaModels.ProdNode(map(a -> _unhoist(a, iv, fv), n.inners))
_unhoist(n::Pair, iv, fv) = _unhoist(n.first, iv, fv) => _unhoist_root(n.second, iv, fv)
function _unhoist(n::ExaModels.DataIndexed{I, J}, iv, fv) where {I, J}
    inner = getfield(n, :inner)
    # DataIndexed(DataSource(), _MROW_D) is the source element
    inner isa ExaModels.DataSource && J == ExaModels._MROW_D && return ExaModels.DataSource()
    return ExaModels.DataIndexed(_unhoist(inner, iv, fv), J)
end
if _has(:IvRef)
    _unhoist(n::ExaModels.IvRef, iv, fv) = iv[n.k]
    _unhoist(n::ExaModels.FvRef, iv, fv) = fv[n.k]
end
# A whole expression that was a constant must stay a node.
function _unhoist_root(n, iv, fv)
    t = _unhoist(n, iv, fv)
    return t isa Real ? ExaModels.Null(t) : t
end

# The plain blocks a merged constraint or objective was built from, in order.
function _segments(b)
    tree = b.f.f
    return [
        PlainBlock(
            _unhoist_root(tree, s.iv, s.fv), s.itr, s.o0,
            _segment_size(s.itr),
            b isa ExaModels.Constraint ? b.tag : nothing
        )
            for s in b.itr.segs
    ]
end

# A merged block records only its row count; the segment's own iterator still
# has the source block's dimensions, which tuple-indexed augmentations need.
_segment_size(itr::Iterators.ProductIterator) = itr.iterators
_segment_size(itr::AbstractArray) = size(itr)
_segment_size(itr) = (length(itr),)

function _unsupported_merged(b)
    throw(
        ArgumentError(
            "ExaModelsIO cannot store a merged block with row-wise targets or a " *
                "device layout; write the model built on the CPU"
        )
    )
end

# ---------------------------------------------------------------- the model

# ExaModels keeps blocks in reverse order of addition.
_in_order(t::Tuple) = reverse(t)

objective_blocks(m) = reduce(
    vcat, (
        _is_merged(o.itr) ? _segments(o) :
            _is_merged_rows(o.itr) ? _unsupported_merged(o) :
            [PlainBlock(o.f.f, o.itr, 0, nothing, nothing)]
            for o in _in_order(m.objs)
    ); init = PlainBlock[]
)

# Constraint blocks in row order, then augmentations in the order they were added.
function constraint_blocks(m)
    cons, augs = PlainBlock[], PlainBlock[]
    for c in _in_order(m.cons)
        if c isa ExaModels.ConstraintAugmentation
            push!(augs, PlainBlock(c.f.f, c.itr, c.f.o0, nothing, c.tag))
        elseif _is_merged(c.itr)
            append!(cons, _segments(c))
        elseif _is_merged_rows(c.itr)
            _unsupported_merged(c)
        else
            push!(cons, PlainBlock(c.f.f, c.itr, c.offset, c.size, c.tag))
        end
    end
    sort!(cons; by = b -> b.offset)
    return cons, augs
end

# Buffered subexpression stages, in θ order.
function stage_blocks(m)
    hasproperty(m, :subexprs) || return PlainBlock[]
    out = PlainBlock[]
    for s in _in_order(m.subexprs)
        s isa ExaModels.SubexprStage || throw(
            ArgumentError(
                "ExaModelsIO cannot store a $(typeof(s).name.name) (buffered sum); use add_expr(...; buffered = true)"
            )
        )
        push!(out, PlainBlock(s.f.f, s.itr, s.offset, nothing, nothing))
    end
    return out
end
