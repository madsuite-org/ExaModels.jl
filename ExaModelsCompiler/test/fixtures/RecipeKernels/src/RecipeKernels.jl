"""
    RecipeKernels

A stand-in for a modelling library, in the one respect that matters here: it
owns a function that ends up *inside* a recipe rather than beside it.

A starting point that varies with the index cannot be a number in the core, and
the size it runs over is not known while the structure is written — so it is
deferred, and the deferred thing is a function this package wrote.  The core
that comes out therefore names a `RecipeKernels` type, and a process that
cannot load `RecipeKernels` cannot read that core back.

The functions are deliberately named rather than anonymous: a `Base.Generator`
over `alternating` carries `typeof(alternating)`, which is stable across
recompiles, whereas a closure carries a gensym that is not.
"""
module RecipeKernels

export alternating, offsets, doubled_args, parsed_args, table2d, datant, summary_writer

"Alternating starting point — the shape a per-index start generator takes."
@inline alternating(i) = isodd(i) ? -1.2 : 1.0

"An index set computed from the size: a comprehension, so it has no symbolic
form and has to run once the size is known."
offsets(n) = [2 * div(i - 1, 2) for i in 1:2:max(n - 2, 0)]

# Argument functions for the argfun surface: named, package-owned, returning
# the argument TUPLE the core is instantiated with — the contract
# `compile_library` requires so the generated library can call them by name.

"The integer-kind argument function: `P_new(n)` hands `n` to it."
doubled_args(n::Integer) = (2 * Int(n),)

"The string-kind argument function: `P_new_str(s)` hands the string to it —
standing in for a case-file path parsed on the far side of the boundary."
parsed_args(s::AbstractString) = (parse(Int, s),)

"A 2-D table of heterogeneous tuples over an open size — the COPS-shaped
deferred data (a Matrix of (Int,Int,Int,F64,F64,F64) rows over `1:n × 1:3`)
that exercises product iterators and per-element instantiate typing."
table2d(n) = [(i, j, i + j, 0.5, 1.5, 2.5) for i in 1:n, j in 1:3]

"A NamedTuple-returning data function — the shape a modelling library's
`*_data(n)` takes, with fields PROJECTED out of the deferred call
(`d.v_start` for starts, `d.tab` as an iterator): the projections are
`ArgIndexed` nodes over the one `ArgNode1`, which is the structure the
gasoil-class cores carry."
datant(n) = (
    v_start = fill(0.5, 3 * n),
    tab = [(i, j, 0.5 + i) for i in 1:n, j in 1:3],
)

"A solution writer for the writer surface: records what it was handed, so the
test can check every argument crossed the boundary."
function summary_writer(out::AbstractString, arg, model, x, y, zl, zu;
                        objective::Float64 = NaN, status::AbstractString = "")
    # `open(f, path, mode)` splats its arguments, which `--trim=safe` refuses.
    io = open(out, "w")
    try
        # One string: a long `println` argument list is a `Vararg` call trim cannot resolve.
        words = (string(arg), string(model.meta.nvar), string(length(x)), string(sum(x)),
                 string(length(y)), string(sum(y)), string(sum(zl)), string(sum(zu)),
                 string(objective), String(status))
        println(io, join(words, ' '))
    finally
        close(io)
    end
    return out
end

end # module
