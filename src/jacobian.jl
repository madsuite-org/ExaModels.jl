"""
    jrpass(d::D, comp, i, y1, y2, o1, cnt, adj)

Performs sparse jacobian evaluation via the reverse pass on the computation (sub)graph formed by forward pass

# Arguments:
- `d`: first-order computation (sub)graph
- `comp`: a `Compressor`, which helps map counter to sparse vector index
- `i`: constraint index (this is `i`-th constraint)
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity)
- `o1`: index offset
- `cnt`: counter
- `adj`: adjoint propagated up to the current node
"""
@inline function jrpass(
    d::D,
    comp,
    i,
    y1,
    y2,
    o1,
    cnt,
    adj,
) where {D<:Union{AdjointNull,Real}}
    return cnt
end
@inline function jrpass(d::D, comp, i, y1, y2, o1, cnt, adj) where {D<:AdjointNode1}
    cnt = jrpass(d.inner, comp, i, y1, y2, o1, cnt, _amul(adj, d.y))
    return cnt
end
@inline function jrpass(d::D, comp, i, y1, y2, o1, cnt, adj) where {D<:AdjointNode2}
    cnt = jrpass(d.inner1, comp, i, y1, y2, o1, cnt, _amul(adj, d.y1))
    cnt = jrpass(d.inner2, comp, i, y1, y2, o1, cnt, _amul(adj, d.y2))
    return cnt
end
@inline function jrpass(d::D, comp, i, y1, y2, o1, cnt, adj) where {D<:AdjointNodeVar}
    @inbounds y1[o1+comp(cnt+=1)] += adj
    return cnt
end
@inline function jrpass(
    d::D,
    comp,
    i,
    y1::Tuple{V1,V2},
    y2,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,V1<:AbstractVector,V2<:AbstractVector}
    (y, v) = y1
    @inbounds y[i] += adj * v[d.i]
    return (cnt += 1)
end
@inline function jrpass(
    d::D,
    comp,
    i,
    y1,
    y2::Tuple{V1,V2},
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,V1<:AbstractVector,V2<:AbstractVector}
    y, v = y2
    @inbounds y[d.i] += adj * v[i]
    return (cnt += 1)
end
@inline function jrpass(
    d::D,
    comp,
    i,
    y1::V,
    y2::V,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,I<:Integer,V<:AbstractVector{I}}
    ind = o1 + comp(cnt += 1)
    @inbounds y1[ind] = i
    @inbounds y2[ind] = d.i
    return cnt
end
@inline function jrpass(
    d::D,
    comp,
    i,
    y1::V,
    y2,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,I<:Tuple{Tuple{Int,Int},Int},V<:AbstractVector{I}}
    ind = o1 + comp(cnt += 1)
    @inbounds y1[ind] = ((i, d.i), ind)
    return cnt
end


"""
    sjacobian!(y1, y2, f, x, adj)

Performs sparse jacobian evaluation

# Arguments:
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity)
- `f`: the function to be differentiated in `SIMDFunction` format
- `x`: variable vector
- `adj`: initial adjoint
"""
function sjacobian!(y1, y2, f, x, θ, adj)
    @simd for i in eachindex(f.itr)
        @inbounds sjacobian!(
            y1,
            y2,
            f.f,
            f.itr[i],
            x,
            θ,
            f.f.comp1,
            offset0(f, i),
            offset1(f, i),
            adj,
        )
    end
end

function sjacobian!(y1, y2, f, p, x, θ, comp, o0, o1, adj)
    graph = f(p, AdjointNodeSource(x), θ)
    jrpass(graph, comp, o0, y1, y2, o1, 0, adj)
end
