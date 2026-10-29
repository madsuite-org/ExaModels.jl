# A model as plain blocks: one expression over one collection each, the form a
# file stores. ExaModels may hold a block in another form (a merged family,
# where several source blocks share one tree with their scalar leaves hoisted
# into the rows); `ExaModels.expand_family` undoes that, so the writer never
# sees it.

struct PlainBlock
    tree::Any          # expression; for an augmentation, `idx => expr` or a bare expr
    itr::Any           # the collection iterated over
    offset::Int        # first row (constraint), base row offset (augmentation), θ offset (stage)
    size::Any          # constraint dimensions, or nothing
    tag::Any
end

_has(s) = isdefined(ExaModels, s)

# A block as its source blocks; ExaModels versions without merging have no families.
_expand(b) = _has(:expand_family) ? ExaModels.expand_family(b) : Any[b]

# ExaModels keeps blocks in reverse order of addition.
_in_order(t::Tuple) = reverse(t)

objective_blocks(m) = PlainBlock[
    PlainBlock(o.f.f, o.itr, 0, nothing, nothing)
        for b in _in_order(m.objs) for o in _expand(b)
]

# Constraint blocks in row order (a family sits at its first block's position,
# so add order is not row order), then augmentations in the order they were added.
function constraint_blocks(m)
    cons, augs = PlainBlock[], PlainBlock[]
    for b in _in_order(m.cons), c in _expand(b)
        if c isa ExaModels.ConstraintAugmentation
            push!(augs, PlainBlock(c.f.f, c.itr, c.f.o0, nothing, c.tag))
        else
            push!(cons, PlainBlock(c.f.f, c.itr, c.offset, c.size, c.tag))
        end
    end
    sort!(cons; by = b -> b.offset)
    return cons, augs
end

# The handle a buffered stage was created with, if it is named: its size and tag
# live there, not on the stage.
function _stage_handle(m, offset)
    m.refs === nothing && return nothing
    for v in values(m.refs)
        v isa ExaModels.BufferedExpression && v.offset == offset && return v
    end
    return nothing
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
        h = _stage_handle(m, s.offset)
        push!(
            out, PlainBlock(
                s.f.f, s.itr, s.offset, h === nothing ? nothing : h.size,
                h === nothing ? nothing : h.tag
            )
        )
    end
    return out
end
