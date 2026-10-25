module LongTupleTest

using Test, ExaModels
import NLPModels

# Backend list comes from the suite driver when available; standalone runs
# cover the plain CPU path.
const LT_BACKENDS = isdefined(Main, :BACKENDS) ? Main.BACKENDS : [nothing]

# Models whose block tuples exceed 32 entries are walked by the `_LongTuple`
# loop methods instead of the dispatch recursion (nlp.jl and the KA
# extension). The fixture builds 39 constraint blocks (36 structurally
# distinct one-block families, a named block, and two augmentations) and 33
# objective blocks, so every long-tuple method is on the evaluation path;
# values are checked against finite differences on the host and against the
# host result per device backend.
const FUNCS = (sin, cos, exp, tanh)

function _build(backend)
    n = 16
    c = ExaCore(; backend)
    c, x = add_var(c, n; start = 0.3)
    for a in FUNCS
        c, _ = add_con(c, a(x[i]) - 0.1 for i in 1:(n - 1); lcon = -5.0, ucon = 5.0)
    end
    for a in FUNCS, b in FUNCS
        c, _ = add_con(c, a(b(x[i])) - 0.1 for i in 1:(n - 1); lcon = -9.0, ucon = 9.0)
    end
    for a in FUNCS, b in FUNCS
        c, _ = add_con(c, 0.5 * a(x[i]) + 0.25 * b(x[i + 1]) for i in 1:(n - 1); lcon = -9.0, ucon = 9.0)
    end
    c, g = add_con(c, x[i] + x[i + 8] for i in 1:4; lcon = -2.0, ucon = 2.0, name = Val(:glong))
    c, _ = add_con!(c, g, i => sin(x[i + 1]) for i in 1:4)
    c, _ = add_con!(c, g, i => cos(x[i + 2]) for i in 1:4)
    for a in FUNCS, b in FUNCS
        c, _ = add_obj(c, a(x[i]) * b(x[i + 1]) for i in 1:(n - 1))
    end
    for a in FUNCS, b in FUNCS
        c, _ = add_obj(c, 0.5 * a(b(x[i])) for i in 1:(n - 1))
    end
    c, _ = add_obj(c, abs2(x[i]) for i in 1:n)
    return c
end

function _evalall(m)
    T = eltype(m.meta.x0)
    x0 = copy(m.meta.x0) .+ T(0.01)   # a Float64 literal would promote a Float32 device vector
    g = fill!(similar(x0), zero(T))
    cv = fill!(similar(x0, m.meta.ncon), zero(T))
    jv = fill!(similar(x0, m.meta.nnzj), zero(T))
    hv = fill!(similar(x0, m.meta.nnzh), zero(T))
    jr = fill!(similar(x0, Int, m.meta.nnzj), 0)
    jc = fill!(similar(jr), 0)
    hr = fill!(similar(x0, Int, m.meta.nnzh), 0)
    hc = fill!(similar(hr), 0)
    y = fill!(similar(cv), one(T))
    o = NLPModels.obj(m, x0)
    NLPModels.grad!(m, x0, g)
    NLPModels.cons!(m, x0, cv)
    NLPModels.jac_structure!(m, jr, jc)
    NLPModels.jac_coord!(m, x0, jv)
    NLPModels.hess_structure!(m, hr, hc)
    NLPModels.hess_coord!(m, x0, y, hv)
    return (o, Array(g), Array(cv), Array(jr), Array(jc), Array(jv), Array(hr), Array(hc), Array(hv))
end

function _fd_jac(m, x0)
    n = length(x0)
    J = zeros(m.meta.ncon, n)
    h = 1.0e-6
    cp = zeros(m.meta.ncon)
    cm = zeros(m.meta.ncon)
    for j in 1:n
        xp = copy(x0); xp[j] += h
        xm = copy(x0); xm[j] -= h
        NLPModels.cons!(m, xp, cp)
        NLPModels.cons!(m, xm, cm)
        J[:, j] .= (cp .- cm) ./ (2h)
    end
    return J
end

function _gradL(m, x, y)
    g = zeros(length(x))
    NLPModels.grad!(m, x, g)
    jr = zeros(Int, m.meta.nnzj); jc = zeros(Int, m.meta.nnzj)
    jv = zeros(m.meta.nnzj)
    NLPModels.jac_structure!(m, jr, jc)
    NLPModels.jac_coord!(m, x, jv)
    for k in 1:m.meta.nnzj
        g[jc[k]] += jv[k] * y[jr[k]]
    end
    return g
end

function _test_host()
    m = ExaModel(_build(nothing))
    # the loop methods only run if the tuples really are long
    @test m.cons isa ExaModels._LongTuple
    @test m.objs isa ExaModels._LongTuple
    @test length(m.cons) == 39
    @test length(m.objs) == 33

    x0 = copy(m.meta.x0) .+ 0.01
    n = length(x0)

    # gradient against central finite differences of obj
    g = zeros(n)
    NLPModels.grad!(m, x0, g)
    h = 1.0e-6
    gfd = map(1:n) do j
        xp = copy(x0); xp[j] += h
        xm = copy(x0); xm[j] -= h
        (NLPModels.obj(m, xp) - NLPModels.obj(m, xm)) / (2h)
    end
    @test isapprox(g, gfd; rtol = 1.0e-6, atol = 1.0e-8)

    # Jacobian values against central finite differences of cons!
    jr = zeros(Int, m.meta.nnzj); jc = zeros(Int, m.meta.nnzj)
    jv = zeros(m.meta.nnzj)
    NLPModels.jac_structure!(m, jr, jc)
    NLPModels.jac_coord!(m, x0, jv)
    J = zeros(m.meta.ncon, n)
    for k in 1:m.meta.nnzj
        J[jr[k], jc[k]] += jv[k]
    end
    @test isapprox(J, _fd_jac(m, x0); rtol = 1.0e-6, atol = 1.0e-8)

    # Lagrangian Hessian values against central finite differences of
    # grad(obj) + J(x)'y (the Jacobian is validated just above)
    y = fill!(zeros(m.meta.ncon), 1.0)
    hr = zeros(Int, m.meta.nnzh); hc = zeros(Int, m.meta.nnzh)
    hv = zeros(m.meta.nnzh)
    NLPModels.hess_structure!(m, hr, hc)
    NLPModels.hess_coord!(m, x0, y, hv)
    H = zeros(n, n)
    for k in 1:m.meta.nnzh
        H[hr[k], hc[k]] += hv[k]
        hr[k] != hc[k] && (H[hc[k], hr[k]] += hv[k])
    end
    h2 = 1.0e-5
    Hfd = zeros(n, n)
    for j in 1:n
        xp = copy(x0); xp[j] += h2
        xm = copy(x0); xm[j] -= h2
        Hfd[:, j] .= (_gradL(m, xp, y) .- _gradL(m, xm, y)) ./ (2h2)
    end
    @test isapprox(H, Hfd; rtol = 1.0e-5, atol = 1.0e-6)

    # products against the assembled (validated) matrices
    v = collect(range(-0.5, 0.5; length = n))
    Jv = zeros(m.meta.ncon)
    NLPModels.jprod_nln!(m, x0, v, Jv)
    @test isapprox(Jv, J * v; rtol = 1.0e-12)
    w = collect(range(0.1, 1.0; length = m.meta.ncon))
    Jtw = zeros(n)
    NLPModels.jtprod_nln!(m, x0, w, Jtw)
    @test isapprox(Jtw, J' * w; rtol = 1.0e-12)
    Hv = zeros(n)
    NLPModels.hprod!(m, x0, y, v, Hv)
    return @test isapprox(Hv, H * v; rtol = 1.0e-12)
end

function _test_backend_matches_host(backend)
    mb = ExaModel(_build(backend))
    # Float32 backends (Metal) carry ~1e-7 roundoff per operation
    tight, loose = eltype(mb.meta.x0) == Float32 ? (1.0e-4, 1.0e-4) : (1.0e-12, 1.0e-10)
    rb = _evalall(mb)
    rh = _evalall(ExaModel(_build(nothing)))
    @test rb[1] ≈ rh[1] rtol = tight
    for k in (2, 3, 6, 9)                       # grad, cons, jac vals, hess vals
        @test sum(rb[k]) ≈ sum(rh[k]) rtol = loose
        @test maximum(abs, rb[k]) ≈ maximum(abs, rh[k]) rtol = loose
    end
    for k in (4, 5, 7, 8)                       # sparsity coordinates
        @test sort(rb[k]) == sort(rh[k])
    end
    return
end

function runtests()
    return @testset "Long block tuples" begin
        @testset "host loop walk against finite differences" begin
            _test_host()
        end
        @testset "matches host result (backend = $b)" for b in LT_BACKENDS
            b === nothing || _test_backend_matches_host(b)
        end
    end
end

end # module
