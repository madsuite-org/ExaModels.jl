"""
    hdrpass(t1::T1, t2::T2, comp, y1, y2, o2, cnt, adj)

Performs sparse hessian evaluation (`(df1/dx)(df2/dx)'` portion) via the reverse pass on the computation (sub)graph formed by second-order forward pass

# Arguments:
- `t1`: second-order computation (sub)graph regarding f1
- `t2`: second-order computation (sub)graph regarding f2
- `comp`: a `Compressor`, which helps map counter to sparse vector index
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity)
- `o2`: index offset
- `cnt`: counter
- `adj`: second adjoint propagated up to the current node
"""
@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode1,T2<:SecondAdjointNode1}
    cnt = hdrpass(t1.inner, t2.inner, comp, y1, y2, o2, cnt, _amul(adj, t1.y * t2.y))
    cnt
end
@inline function hdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNode1,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
)  # despecialized
    cnt = hdrpass(t1.inner, t2.inner, comp, y1, y2, o2, cnt, _amul(adj, t1.y * t2.y))
    cnt
end


@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNode1}
    cnt = hdrpass(t1, t2.inner, comp, y1, y2, o2, cnt, _amul(adj, t2.y))
    cnt
end
@inline function hdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointNode1,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
)  # despecialized
    cnt = hdrpass(t1, t2.inner, comp, y1, y2, o2, cnt, _amul(adj, t2.y))
    cnt
end


@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode1,T2<:SecondAdjointNodeVar}
    cnt = hdrpass(t1.inner, t2, comp, y1, y2, o2, cnt, _amul(adj, t1.y))
    cnt
end
@inline function hdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNodeVar,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
)  # despecialized
    cnt = hdrpass(t1.inner, t2, comp, y1, y2, o2, cnt, _amul(adj, t1.y))
    cnt
end


@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode2,T2<:SecondAdjointNode2}
    cnt = hdrpass(t1.inner1, t2.inner1, comp, y1, y2, o2, cnt, _amul(adj, t1.y1 * t2.y1))
    cnt = hdrpass(t1.inner1, t2.inner2, comp, y1, y2, o2, cnt, _amul(adj, t1.y1 * t2.y2))
    cnt = hdrpass(t1.inner2, t2.inner1, comp, y1, y2, o2, cnt, _amul(adj, t1.y2 * t2.y1))
    cnt = hdrpass(t1.inner2, t2.inner2, comp, y1, y2, o2, cnt, _amul(adj, t1.y2 * t2.y2))
    cnt
end
@inline function hdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNode2,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
) # despecialized
    cnt = hdrpass(t1.inner1, t2.inner1, comp, y1, y2, o2, cnt, _amul(adj, t1.y1 * t2.y1))
    cnt = hdrpass(t1.inner1, t2.inner2, comp, y1, y2, o2, cnt, _amul(adj, t1.y1 * t2.y2))
    cnt = hdrpass(t1.inner2, t2.inner1, comp, y1, y2, o2, cnt, _amul(adj, t1.y2 * t2.y1))
    cnt = hdrpass(t1.inner2, t2.inner2, comp, y1, y2, o2, cnt, _amul(adj, t1.y2 * t2.y2))
    cnt
end


@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode1,T2<:SecondAdjointNode2}
    cnt = hdrpass(t1.inner, t2.inner1, comp, y1, y2, o2, cnt, _amul(adj, t1.y * t2.y1))
    cnt = hdrpass(t1.inner, t2.inner2, comp, y1, y2, o2, cnt, _amul(adj, t1.y * t2.y2))
    cnt
end
@inline function hdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNode2,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
) # despecialized
    cnt = hdrpass(t1.inner, t2.inner1, comp, y1, y2, o2, cnt, _amul(adj, t1.y * t2.y1))
    cnt = hdrpass(t1.inner, t2.inner2, comp, y1, y2, o2, cnt, _amul(adj, t1.y * t2.y2))
    cnt
end

@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode2,T2<:SecondAdjointNode1}
    cnt = hdrpass(t1.inner1, t2.inner, comp, y1, y2, o2, cnt, _amul(adj, t1.y1 * t2.y))
    cnt = hdrpass(t1.inner2, t2.inner, comp, y1, y2, o2, cnt, _amul(adj, t1.y2 * t2.y))
    cnt
end
function hdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNode1,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
) # despecialized
    cnt = hdrpass(t1.inner1, t2.inner, comp, y1, y2, o2, cnt, _amul(adj, t1.y1 * t2.y))
    cnt = hdrpass(t1.inner2, t2.inner, comp, y1, y2, o2, cnt, _amul(adj, t1.y2 * t2.y))
    cnt
end

@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNode2}
    cnt = hdrpass(t1, t2.inner1, comp, y1, y2, o2, cnt, _amul(adj, t2.y1))
    cnt = hdrpass(t1, t2.inner2, comp, y1, y2, o2, cnt, _amul(adj, t2.y2))
    cnt
end
@inline function hdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointNode2,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
) # despecialized
    cnt = hdrpass(t1, t2.inner1, comp, y1, y2, o2, cnt, _amul(adj, t2.y1))
    cnt = hdrpass(t1, t2.inner2, comp, y1, y2, o2, cnt, _amul(adj, t2.y2))
    cnt
end

@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode2,T2<:SecondAdjointNodeVar}
    cnt = hdrpass(t1.inner1, t2, comp, y1, y2, o2, cnt, _amul(adj, t1.y1))
    cnt = hdrpass(t1.inner2, t2, comp, y1, y2, o2, cnt, _amul(adj, t1.y2))
    cnt
end
@inline function hdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNodeVar,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
) # despecialized
    cnt = hdrpass(t1.inner1, t2, comp, y1, y2, o2, cnt, _amul(adj, t1.y1))
    cnt = hdrpass(t1.inner2, t2, comp, y1, y2, o2, cnt, _amul(adj, t1.y2))
    cnt
end


@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNodeVar}
    i, j = t1.i, t2.i
    @inbounds if i == j
        y1[o2+comp(cnt+=1)] += 2 * adj
    else
        y1[o2+comp(cnt+=1)] += adj
    end
    cnt
end

@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt::Tuple{<:Tuple,<:Tuple},
    adj,
) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNodeVar}
    pair = (t1.i, t2.i)
    mapping, uniques = cnt
    idx = _hpass_find_pair(pair, uniques, 1)
    if idx === 0
        return ((mapping..., length(uniques) + 1), (uniques..., pair))
    else
        return ((mapping..., idx), uniques)
    end
end


@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1::Tuple{V1,V2},
    y2,
    o2,
    cnt,
    adj,
) where {
    T1<:SecondAdjointNodeVar,
    T2<:SecondAdjointNodeVar,
    V1<:AbstractVector,
    V2<:AbstractVector,
}
    i, j = t1.i, t2.i
    y, v = y1
    @inbounds if i == j
        y[i] += 2 * adj * v[i]
    else
        y[i] += adj * v[j]
        y[j] += adj * v[i]
    end
    return (cnt += 1)
end

# SecondAdjointNull: constant branch contributes zero cross-derivatives
@inline hdrpass(::Any, ::SecondAdjointNull, comp, y1, y2, o2, cnt, adj) = cnt
@inline hdrpass(::SecondAdjointNull, ::Any, comp, y1, y2, o2, cnt, adj) = cnt
@inline hdrpass(::SecondAdjointNull, ::SecondAdjointNull, comp, y1, y2, o2, cnt, adj) = cnt

"""
    hrpass(t::D, comp, y1, y2, o2, cnt, adj, adj2)

Performs sparse hessian evaluation (`d²f/dx²` portion) via the reverse pass on the computation (sub)graph formed by second-order forward pass

# Arguments:
- `comp`: a `Compressor`, which helps map counter to sparse vector index
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity)
- `o2`: index offset
- `cnt`: counter
- `adj`: first adjoint propagated up to the current node
- `adj`: second adjoint propagated up to the current node
"""

@inline function hrpass(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:Union{SecondAdjointNull,Real}}
    cnt
end
@inline function hrpass(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode1}

    cnt = hrpass(t.inner, comp, y1, y2, o2, cnt, _amul(adj, t.y), _amul(adj2, (t.y)^2) + _amul(adj, t.h))
    cnt
end
@inline function hrpass(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode2}

    adj2y1y2 = _amul(adj2, t.y1 * t.y2)
    adjh12 = _amul(adj, t.h12)
    cnt = hrpass(t.inner1, comp, y1, y2, o2, cnt, _amul(adj, t.y1), _amul(adj2, (t.y1)^2) + _amul(adj, t.h11))
    cnt = hrpass(t.inner2, comp, y1, y2, o2, cnt, _amul(adj, t.y2), _amul(adj2, (t.y2)^2) + _amul(adj, t.h22))
    cnt = hdrpass(t.inner1, t.inner2, comp, y1, y2, o2, cnt, adj2y1y2 + adjh12)
    cnt
end

@inline hrpass0(args...) = hrpass(args...)


@inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {N<:Union{FirstFixed{typeof(*)},SecondFixed{typeof(*)}},D<:SecondAdjointNode1{N}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, _amul(adj, t.y), _amul(adj2, (t.y)^2))
    cnt
end
@inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {N<:Union{FirstFixed{typeof(+)},SecondFixed{typeof(+)}},D<:SecondAdjointNode1{N}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, adj, adj2)
    cnt
end
@inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode1{FirstFixed{typeof(-)}}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, -adj, adj2)
    cnt
end
@inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode1{SecondFixed{typeof(-)}}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, adj, adj2)
    cnt
end

@inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode1{typeof(+)}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, adj, adj2)
    cnt
end
@inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode1{typeof(-)}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, -adj, adj2)
    cnt
end

@inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode2{typeof(+)}}
    cnt = hrpass0(t.inner1, comp, y1, y2, o2, cnt, adj, adj2)
    cnt = hrpass0(t.inner2, comp, y1, y2, o2, cnt, adj, adj2)
    cnt
end

@inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode2{typeof(-)}}
    cnt = hrpass0(t.inner1, comp, y1, y2, o2, cnt, adj, adj2)
    cnt = hrpass0(t.inner2, comp, y1, y2, o2, cnt, -adj, adj2)
    cnt
end
@inline function hrpass0(
    t::T,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar}
    cnt
end
@inline function hrpass0(
    t::T,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar}
    cnt
end


@inline function hdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointNodeVar,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt::Vector,
    adj,
)
    push!(cnt, (t1.i, t2.i))
    return cnt
end
function hrpass(t::SecondAdjointNodeVar, comp::Nothing, y1, y2, o2, cnt::Vector, adj, adj2)
    push!(cnt, (t.i, t.i))
    return cnt
end

# Tuple-based sparsity detection for Hessian: cnt = (mapping_acc::Tuple, unique_acc::Tuple)
# Unique elements are (i,j) pairs (lower-triangular: i >= j).
@inline _hpass_find_pair(x, ::Tuple{}, i) = 0
@inline function _hpass_find_pair(x, t::Tuple, i)
    t[1] === x && return i
    return _hpass_find_pair(x, Base.tail(t), i + 1)
end

@inline function hrpass(
    t::T,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt::Tuple{<:Tuple,<:Tuple},
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar}
    pair = (t.i, t.i)
    mapping, uniques = cnt
    idx = _hpass_find_pair(pair, uniques, 1)
    if idx === 0
        return ((mapping..., length(uniques) + 1), (uniques..., pair))
    else
        return ((mapping..., idx), uniques)
    end
end

@inline function hrpass(
    t::T,
    comp,
    y1::Tuple{V1,V2},
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar,V1<:AbstractVector,V2<:AbstractVector}
    y, v = y1
    @inbounds y[t.i] += adj2 * v[t.i]
    return (cnt += 1)
end
@inline function hrpass(
    t::T,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar}
    @inbounds y1[o2+comp(cnt+=1)] += adj2
    cnt
end
@inline function hrpass(
    t::T,
    comp,
    y1::V,
    y2::V,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar,I<:Integer,V<:AbstractVector{I}}
    ind = o2 + comp(cnt += 1)
    @inbounds y1[ind] = t.i
    @inbounds y2[ind] = t.i
    cnt
end
@inline function hrpass(
    t::T,
    comp,
    y1::V,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar,I<:Tuple{Tuple{Int,Int},Int},V<:AbstractVector{I}}
    ind = o2 + comp(cnt += 1)
    @inbounds y1[ind] = ((t.i, t.i), ind)
    cnt
end
@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1::V,
    y2::V,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNodeVar,I<:Integer,V<:AbstractVector{I}}
    i, j = t1.i, t2.i
    ind = o2 + comp(cnt += 1)
    @inbounds if i >= j
        y1[ind] = i
        y2[ind] = j
    else
        y1[ind] = j
        y2[ind] = i
    end
    cnt
end
@inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1::V,
    y2,
    o2,
    cnt,
    adj,
) where {
    T1<:SecondAdjointNodeVar,
    T2<:SecondAdjointNodeVar,
    I<:Tuple{Tuple{Int,Int},Int},
    V<:AbstractVector{I},
}
    i, j = t1.i, t2.i
    ind = o2 + comp(cnt += 1)
    @inbounds if i >= j
        y1[ind] = ((i, j), ind)
    else
        y1[ind] = ((j, i), ind)
    end
    cnt
end

"""
    shessian!(y1, y2, f, x, adj1, adj2)

Performs sparse jacobian evaluation

# Arguments:
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity)
- `f`: the function to be differentiated in `SIMDFunction` format
- `x`: variable vector
- `adj1`: initial first adjoint
- `adj2`: initial second adjoint
"""
function shessian!(y1, y2, f, x, θ, adj1, adj2)
    @simd for k in eachindex(f.itr)
        @inbounds begin
            p = f.itr[k]
            shessian!(y1, y2, f.f, p, x, θ, _comp2(f.f, p), offset2(f, k), adj1, adj2)
        end
    end
end
function shessian!(y1, y2, f, x, θ, adj1s::V, adj2) where {V<:AbstractVector}
    @simd for k in eachindex(f.itr)
        @inbounds begin
            p = f.itr[k]
            shessian!(y1, y2, f.f, p, x, θ, _comp2(f.f, p), offset2(f, k), adj1s[offset0(f, k)], adj2)
        end
    end
end

function shessian!(y1, y2, f, p, x, θ, comp, o2, adj1, adj2)
    graph = f(p, SecondAdjointNodeSource(x), θ)
    hrpass0(graph, comp, y1, y2, o2, 0, adj1, adj2)
end
