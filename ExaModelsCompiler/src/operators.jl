# Type-directed conversions and constructors, and ranges, on ExaModels argument nodes
# (`floor(Int, arg.x)`, `zeros(Float64, arg.N)`, `convert(T, arg.x)`, `1:arg.N`), for
# recipes that compute with their arguments. They live here rather than in ExaModels
# because they extend Base functions whose call sites are everywhere (`convert`, `:`) or
# that dispatch on `::Type{T}` for any `T`, and loading them invalidates compiled code
# across the session.

using ExaModels: AbstractArgNode, ArgNode1, ArgNode2

# Type-directed conversions: `floor(Int, arg.x)`, `zeros(Float64, arg.N)`.
# `Base.Fix1` keeps the type parameter in the node's type instead of capturing a
# closure, so the call stays statically resolvable.
for op in (:floor, :ceil, :round, :trunc, :zeros, :ones, :convert)
    @eval @inline Base.$op(::Type{T}, a::AbstractArgNode) where {T} =
        ArgNode1(Base.Fix1(Base.$op, T), a)
end
# `convert` is the one name above that Base also uses for *storage* rather than
# for conversion: `convert(::Type{Any}, x)` is what runs when a value is merely
# put into an `Any` slot, which on 1.10 includes `push!` to a `Vector{Any}` — so
# a node landing in one of those hits the deferred method above, and the two are
# ambiguous.  Storing a node is not converting it.
@inline Base.convert(::Type{Any}, a::AbstractArgNode) = a

# Ranges: `1:arg.N`, `arg.lo:arg.hi`.
@inline Base.:(:)(a::AbstractArgNode, b::AbstractArgNode) = ArgNode2(Colon(), a, b)
@inline Base.:(:)(a::AbstractArgNode, b::Number) = ArgNode2(Colon(), a, b)
@inline Base.:(:)(a::Number, b::AbstractArgNode) = ArgNode2(Colon(), a, b)

