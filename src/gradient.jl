"""
    drpass(d::D, y, adj)

Performs dense gradient evaluation via the reverse pass on the computation (sub)graph formed by forward pass

# Arguments:
- `d`: first-order computation (sub)graph
- `y`: result vector
- `adj`: adjoint propagated up to the current node
"""
@inline function drpass(d::D, y, adj) where {D<:Union{Real,AdjointNull}}
    nothing
end
@inline function drpass(d::D, y, adj) where {D<:AdjointNode1}
    offset = drpass(d.inner, y, _amul(adj, d.y))
    nothing
end
@inline function drpass(d::D, y, adj) where {D<:AdjointNode2}
    offset = drpass(d.inner1, y, _amul(adj, d.y1))
    offset = drpass(d.inner2, y, _amul(adj, d.y2))
    nothing
end
@inline function drpass(d::D, y, adj) where {D<:AdjointNodeVar}
    @inbounds y[d.i] += adj
    nothing
end

"""
    gradient!(y, f, x, adj)

Performs dense gradient evaluation

# Arguments:
- `y`: result vector
- `f`: the function to be differentiated in `SIMDFunction` format
- `x`: variable vector
- `adj`: initial adjoint
"""
function gradient!(y, f, x, θ, adj)
    @simd for k in eachindex(f.itr)
        @inbounds gradient!(y, f.f, x, θ, f.itr[k], adj)
    end
    return y
end
function gradient!(y, f, x, θ, p, adj)
    graph = f(p, AdjointNodeSource(x), θ)
    drpass(graph, y, adj)
    return y
end

"""
    grpass(d::D, comp, y, o1, cnt, adj)

Performs dsparse gradient evaluation via the reverse pass on the computation (sub)graph formed by forward pass

# Arguments:
- `d`: first-order computation (sub)graph
- `comp`: a `Compressor`, which helps map counter to sparse vector index
- `y`: result vector
- `o1`: index offset
- `cnt`: counter
- `adj`: adjoint propagated up to the current node
    """
@inline function grpass(
    d::D,
    comp,
    y,
    o1,
    cnt,
    adj,
    ) where {D <: Union{AdjointNull, DataIndexed, Real}}
    return cnt
end
@inline function grpass(d::D, comp, y, o1, cnt, adj) where {D<:AdjointNode1}
    cnt = grpass(d.inner, comp, y, o1, cnt, _amul(adj, d.y))
    return cnt
end
@inline function grpass(d::D, comp, y, o1, cnt, adj) where {D<:AdjointNode2}
    cnt = grpass(d.inner1, comp, y, o1, cnt, _amul(adj, d.y1))
    cnt = grpass(d.inner2, comp, y, o1, cnt, _amul(adj, d.y2))
    return cnt
end
@inline function grpass(d::D, comp, y, o1, cnt, adj) where {D<:AdjointNodeVar}
    @inbounds y[o1+comp(cnt+=1)] += adj
    return cnt
end
@inline function grpass(d::AdjointNodeVar, comp::Nothing, y, o1, cnt::Vector, adj)
    push!(cnt, d.i)
    return cnt
end

# Tuple-based sparsity detection: cnt = (mapping_acc::Tuple, unique_acc::Tuple)
# Returns (mapping_tuple, unique_tuple) without any mutable state.
@inline _grpass_find_ident(x, ::Tuple{}, i) = 0
@inline function _grpass_find_ident(x, t::Tuple, i)
    t[1] === x && return i
    return _grpass_find_ident(x, Base.tail(t), i + 1)
end

@inline function grpass(
    d::D,
    comp::Nothing,
    y,
    o1,
    cnt::Tuple{<:Tuple, <:Tuple},
    adj,
    ) where {D <: Union{AdjointNull, DataIndexed, Real}}
    return cnt
end
@inline function grpass(
    d::D,
    comp::Nothing,
    y,
    o1,
    cnt::Tuple{<:Tuple,<:Tuple},
    adj,
) where {D<:AdjointNode1}
    return grpass(d.inner, nothing, y, o1, cnt, _amul(adj, d.y))
end
@inline function grpass(
    d::D,
    comp::Nothing,
    y,
    o1,
    cnt::Tuple{<:Tuple,<:Tuple},
    adj,
) where {D<:AdjointNode2}
    cnt = grpass(d.inner1, nothing, y, o1, cnt, _amul(adj, d.y1))
    return grpass(d.inner2, nothing, y, o1, cnt, _amul(adj, d.y2))
end
@inline function grpass(
    d::D,
    comp::Nothing,
    y,
    o1,
    cnt::Tuple{<:Tuple,<:Tuple},
    adj,
) where {D<:AdjointNodeVar}
    mapping, uniques = cnt
    idx = _grpass_find_ident(d.i, uniques, 1)
    if idx === 0
        return ((mapping..., length(uniques) + 1), (uniques..., d.i))
    else
        return ((mapping..., idx), uniques)
    end
end
@inline function grpass(
    d::D,
    comp,
    y::V,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,V<:AbstractVector{Tuple{Int,Int}}}
    ind = o1 + comp(cnt += 1)
    @inbounds y[ind] = (d.i, ind)
    return cnt
end

""" sgradient!(y, f, x, adj)

Performs sparse gradient evaluation

# Arguments:
- `y`: result vector
- `f`: the function to be differentiated in `SIMDFunction` format
- `x`: variable vector
- `adj`: initial adjoint
"""
function sgradient!(y, f, x, θ, adj)
    @simd for k in eachindex(f.itr)
        @inbounds begin
            p = f.itr[k]
            sgradient!(y, f.f, p, x, θ, _comp1(f.f, p), offset1(f, k), adj)
        end
    end
    return y
end

function sgradient!(y, f, p, x, θ, comp, o1, adj)
    graph = f(p, AdjointNodeSource(x), θ)
    grpass(graph, comp, y, o1, 0, adj)
    return y
end
