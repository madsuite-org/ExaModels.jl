"""
    _needs_overload(f, types)

Return `true` when ExaModels should add a method for `f` with the given
argument `types`.

Plain `hasmethod` is too conservative for Base generics such as
`max(x,y) = ifelse(isless(x,y),y,x)`: those definitions match any
`AbstractNode` argument, so `hasmethod` returns `true` and would prevent the
ExaModels-specific overload from being added.  We instead check the *owner
module* of the matching method and only skip when ExaModels already owns it.
"""
function _needs_overload(f, types)
    hasmethod(f, types) || return true
    return which(f, types).module !== @__MODULE__
end

"""
    @register_univariate(f, df, ddf)

Register a univariate function `f` so it can be used inside `@add_obj` / `@add_con`
expressions.  The macro adds three method groups:

1. **Primal graph node** — `f(n::AbstractNode) → Node1(f, n)`.
   Applied at model-construction time to build the symbolic graph.

2. **Constant folding** — `f(::Constant{T}) → Constant(f(T))`.
   When the argument is a [`Constant`](@ref) (value encoded as a type
   parameter), the result is evaluated immediately and stored as a new
   `Constant`, avoiding a `Node1` allocation entirely.

3. **Adjoint / second-adjoint nodes** — forward-pass nodes for gradient and
   Hessian computation, using `df` and `ddf` as the derivative functions.

# Arguments
- `f`:   the function to register
- `df`:  first derivative `f'(x)`
- `ddf`: second derivative `f''(x)`

## Example
```jldoctest
julia> using ExaModels

julia> relu3(x) = x > 0 ? x^3 : zero(x)
relu3 (generic function with 1 method)

julia> drelu3(x) = x > 0 ? 3*x^2 : zero(x)
drelu3 (generic function with 1 method)

julia> ddrelu3(x) = x > 0 ? 6*x : zero(x)
ddrelu3 (generic function with 1 method)

julia> @register_univariate(relu3, drelu3, ddrelu3)
```
"""
macro register_univariate(f, df, ddf)
    return esc(
        quote
            if ExaModels._needs_overload($f, Tuple{ExaModels.AbstractNode})
                @inline $f(n::N) where {N<:ExaModels.AbstractNode} = ExaModels.Node1($f, n)
            end
            # Constant folding: f(Constant{T}()) → Constant(f(T))
            @inline $f(n::ExaModels.Constant{I}) where {I} = ExaModels.Constant($f(I))

            @inline $f(d::D) where {D<:ExaModels.AbstractAdjointNode} =
                ExaModels.AdjointNode1($f, $f(d.x), $df(d.x), d)
            @inline $f(t::T) where {T<:ExaModels.AbstractSecondAdjointNode} =
                ExaModels.SecondAdjointNode1($f, $f(t.x), $df(t.x), $ddf(t.x), t)

            @inline (n::ExaModels.Node1{typeof($f),I})(i, x, θ) where {I} =
                $f(n.inner(i, x, θ))
            # structure-pass twin (θ is a NaNSource only in structure passes: the add-time
            # sparsity probe and the device structure kernels, which run once): not
            # inlined, so the probe each add_con compiles stays small; evaluation inlines the tree
            (n::ExaModels.Node1{typeof($f),I})(i, x, θ::ExaModels.NaNSource) where {I} =
                $f(n.inner(i, x, θ))
        end,
    )
end

"""
    @register_bivariate(f, df1, df2, ddf11, ddf12, ddf22)

Register a bivariate function `f` so it can be used inside `@add_obj` / `@add_con`
expressions.  The macro adds four method groups:

1. **Node OP Node** — `f(d1::AbstractNode, d2::AbstractNode) → Node2(f, d1, d2)`.
2. **Node OP Real** — `f(d1::AbstractNode, d2::Real) → Node2(f, d1, d2)`.
   Numeric scalars (including iterator-derived values like `factorial(j)`) are
   stored directly in `Node2` without wrapping in [`Constant`](@ref).
3. **Real OP Node** — symmetric counterpart of the above.
4. **Constant folding** — `f(Constant{I1}(), Constant{I2}()) → Constant(f(I1, I2))`.
   Both-constant expressions are evaluated immediately.

# Arguments
- `f`:     the function to register
- `df1`:   partial derivative w.r.t. the first argument
- `df2`:   partial derivative w.r.t. the second argument
- `ddf11`: second partial w.r.t. the first argument
- `ddf12`: mixed second partial
- `ddf22`: second partial w.r.t. the second argument

## Example
```jldoctest
julia> using ExaModels

julia> relu23(x,y) = (x > 0 || y > 0) ? (x + y)^3 : zero(x)
relu23 (generic function with 1 method)

julia> drelu231(x,y) = (x > 0 || y > 0) ? 3 * (x + y)^2 : zero(x)
drelu231 (generic function with 1 method)

julia> drelu232(x,y) = (x > 0 || y > 0) ? 3 * (x + y)^2  : zero(x)
drelu232 (generic function with 1 method)

julia> ddrelu2311(x,y) = (x > 0 || y > 0) ? 6 * (x + y) : zero(x)
ddrelu2311 (generic function with 1 method)

julia> ddrelu2312(x,y) = (x > 0 || y > 0) ? 6 * (x + y) : zero(x)
ddrelu2312 (generic function with 1 method)

julia> ddrelu2322(x,y) = (x > 0 || y > 0) ? 6 * (x + y) : zero(x)
ddrelu2322 (generic function with 1 method)

julia> @register_bivariate(relu23, drelu231, drelu232, ddrelu2311, ddrelu2312, ddrelu2322)
```
"""
macro register_bivariate(f, df1, df2, ddf11, ddf12, ddf22)
    return esc(
        quote
            if ExaModels._needs_overload($f, Tuple{ExaModels.AbstractNode,ExaModels.AbstractNode})
                @inline function $f(
                    d1::D1,
                    d2::D2,
                ) where {D1<:ExaModels.AbstractNode,D2<:ExaModels.AbstractNode}
                    ExaModels.Node2($f, d1, d2)
                end
            end
            # Constant folding: f(Constant{I1}(), Constant{I2}()) → Constant(f(I1,I2))
            @inline $f(d1::ExaModels.Constant{I1}, d2::ExaModels.Constant{I2}) where {I1, I2} = ExaModels.Constant($f(I1, I2))

            if ExaModels._needs_overload($f, Tuple{ExaModels.AbstractNode,Real})
                @inline function $f(
                    d1::D1,
                    d2::D2,
                ) where {D1<:ExaModels.AbstractNode,D2<:Real}
                    ExaModels.Node2($f, d1, d2)
                end
            end

            if ExaModels._needs_overload($f, Tuple{Real,ExaModels.AbstractNode})
                @inline function $f(
                    d1::D1,
                    d2::D2,
                ) where {D1<:Real,D2<:ExaModels.AbstractNode}
                    ExaModels.Node2($f, d1, d2)
                end
            end

            # An argument scalar meeting a graph node — `h * x[i]` where
            # `h = 1/(arg.N+1)`.  It enters the graph as an `ArgLeaf`, which is
            # a leaf of unknown value in the same sense `DataIndexed` is, and
            # becomes a plain `Real` at instantiation.  Arithmetic between two
            # argument nodes is not this case and stays in the argument world.
            @inline function $f(
                d1::D1,
                d2::D2,
            ) where {D1<:ExaModels.AbstractNode,D2<:ExaModels.AbstractArgNode}
                ExaModels.Node2($f, d1, ExaModels.ArgLeaf(d2))
            end

            @inline function $f(
                d1::D1,
                d2::D2,
            ) where {D1<:ExaModels.AbstractArgNode,D2<:ExaModels.AbstractNode}
                ExaModels.Node2($f, ExaModels.ArgLeaf(d1), d2)
            end

            @inline function $f(
                d1::D1,
                d2::D2,
            ) where {D1<:ExaModels.AbstractAdjointNode,D2<:ExaModels.AbstractAdjointNode}

                x1 = d1.x
                x2 = d2.x
                ExaModels.AdjointNode2($f, $f(x1, x2), $df1(x1, x2), $df2(x1, x2), d1, d2)
            end
            @inline function $f(
                d1::D1,
                d2::D2,
            ) where {
                D1<:ExaModels.AbstractAdjointNode,
                D2<:Union{Real,ExaModels.ParameterNode},
            }

                x1 = d1.x
                x2 = d2

                ExaModels.AdjointNode1($f, $f(x1, x2), $df1(x1, x2), d1)
            end
            @inline function $f(
                d1::D1,
                d2::D2,
            ) where {
                D1<:Union{Real,ExaModels.ParameterNode},
                D2<:ExaModels.AbstractAdjointNode,
            }

                x1 = d1
                x2 = d2.x
                ExaModels.AdjointNode1($f, $f(x1, x2), $df2(x1, x2), d2)
            end

            @inline function $f(
                t1::T1,
                t2::T2,
            ) where {
                T1<:ExaModels.AbstractSecondAdjointNode,
                T2<:ExaModels.AbstractSecondAdjointNode,
            }

                x1 = t1.x
                x2 = t2.x
                ExaModels.SecondAdjointNode2(
                    $f,
                    $f(x1, x2),
                    $df1(x1, x2),
                    $df2(x1, x2),
                    $ddf11(x1, x2),
                    $ddf12(x1, x2),
                    $ddf22(x1, x2),
                    t1,
                    t2,
                )
            end
            @inline function $f(
                t1::T1,
                t2::T2,
            ) where {
                T1<:ExaModels.AbstractSecondAdjointNode,
                T2<:Union{Real,ExaModels.ParameterNode},
            }

                x1 = t1.x
                x2 = t2
                ExaModels.SecondAdjointNode1(
                    ExaModels.SecondFixed($f),
                    $f(x1, x2),
                    $df1(x1, x2),
                    $ddf11(x1, x2),
                    t1,
                )
            end
            @inline function $f(
                t1::T1,
                t2::T2,
            ) where {
                T1<:Union{Real,ExaModels.ParameterNode},
                T2<:ExaModels.AbstractSecondAdjointNode,
            }

                x1 = t1
                x2 = t2.x
                ExaModels.SecondAdjointNode1(
                    ExaModels.FirstFixed($f),
                    $f(x1, x2),
                    $df2(x1, x2),
                    $ddf22(x1, x2),
                    t2,
                )
            end

            @inline (n::ExaModels.Node2{typeof($f),I1,I2})(i, x, θ) where {I1,I2} =
                $f(n.inner1(i, x, θ), n.inner2(i, x, θ))
            @inline (n::ExaModels.Node2{typeof($f),I1,I2})(i, x, θ) where {I1<:Real,I2} =
                $f(n.inner1, n.inner2(i, x, θ))
            @inline (n::ExaModels.Node2{typeof($f),I1,I2})(i, x, θ) where {I1,I2<:Real} =
                $f(n.inner1(i, x, θ), n.inner2)
            # sparsity-probe twins, as for Node1 above
            (n::ExaModels.Node2{typeof($f),I1,I2})(i, x, θ::ExaModels.NaNSource) where {I1,I2} =
                $f(n.inner1(i, x, θ), n.inner2(i, x, θ))
            (n::ExaModels.Node2{typeof($f),I1,I2})(i, x, θ::ExaModels.NaNSource) where {I1<:Real,I2} =
                $f(n.inner1, n.inner2(i, x, θ))
            (n::ExaModels.Node2{typeof($f),I1,I2})(i, x, θ::ExaModels.NaNSource) where {I1,I2<:Real} =
                $f(n.inner1(i, x, θ), n.inner2)
        end,
    )
end

@inline _mone(x) = -one(x)
@inline _one(x1, x2) = one(x1)
@inline _zero(x1, x2) = zero(x1)
@inline _mone(x1, x2) = -one(x1)
@inline _x1(x1, x2) = x1
@inline _x2(x1, x2) = x2
@inline _and(x::Bool, y::Bool) = x && y
@inline _or(x::Bool, y::Bool) = x || y
@inline _and(x, y::Bool) = x == 1 && y
@inline _or(x, y::Bool) = x == 1 || y
@inline _and(x::Bool, y) = x && y == 1
@inline _or(x::Bool, y) = x || y == 1
@inline _and(x, y) = x == 1 && y == 1
@inline _or(x, y) = x == 1 || y == 1

# Type-generic constant helpers (avoid Float64 literals for Float32 compatibility)
@inline _clog2(x) = oftype(x, log(2))
@inline _clog10(x) = oftype(x, log(10))
@inline _cpi(x) = oftype(x, π)
@inline _cd2r(x) = oftype(x, π / 180)
@inline _cr2d(x) = oftype(x, 180 / π)

