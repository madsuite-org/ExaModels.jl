module MergeTest

using Test, ExaModels
import NLPModels

# Backend list comes from the suite driver when available (so GPU legs
# exercise merging too); standalone runs cover the plain CPU path.
const MERGE_BACKENDS = isdefined(Main, :BACKENDS) ? Main.BACKENDS : [nothing]

# A model exercising every merge path: a range family over two variable
# blocks plus a scalar-coefficient variant (lazy segmented path), a data-tuple
# family (lazy path over arrays), a same-type augmentation pair (array path on
# host, unmerged on device), a named block, and a non-mergeable singleton.
function _build(backend)
    n = 64
    c = ExaCore(; backend)
    c, x = add_var(c, n; start = 0.3)
    c, z = add_var(c, n; start = 0.8)
    c, _ = add_con(c, sin(x[i]) * x[i+1] - 0.1 for i in 1:(n-1); lcon = -2.0, ucon = 2.0)
    c, _ = add_con(c, sin(z[i]) * z[i+1] - 0.1 for i in 1:(n-1); lcon = -2.0, ucon = 2.0)
    c, _ = add_con(c, sin(x[i]) * x[i+1] - 0.7 for i in 1:(n-1); lcon = -2.0, ucon = 2.0)
    data1 = [(i = i, a = 0.5 + 0.001i, b = 0.1) for i in 1:(n-1)]
    data2 = [(i = i, a = 0.25 + 0.002i, b = 0.2) for i in 1:(n-1)]
    c, _ = add_con(c, d.a * exp(0.01 * x[d.i]) + d.b * z[d.i+1] for d in data1; lcon = -5.0, ucon = 5.0)
    c, _ = add_con(c, d.a * exp(0.01 * x[d.i]) + d.b * z[d.i+1] for d in data2; lcon = -5.0, ucon = 5.0)
    c, g = add_con(c, x[i] + z[i] for i in 1:8; lcon = -1.0, ucon = 1.0, name = Val(:gnamed))
    c, _ = add_con!(c, g, i => sin(x[i + 1]) for i in 1:8)
    c, _ = add_con!(c, g, i => sin(z[i + 1]) for i in 1:8)
    c, _ = add_con(c, tanh(x[i]) * exp(z[i]) for i in 1:4; lcon = -3.0, ucon = 3.0)
    c, _ = add_obj(c, abs2(x[i] - z[i]) for i in 1:n)
    # loop-written objectives: one scalar add per iteration, two variable
    # blocks, collapsing into a single objective family
    for k in 1:4
        c, _ = add_obj(c, 0.25 * abs2(x[k]) - 0.1 * z[k+1])
    end
    for k in 1:4
        c, _ = add_obj(c, 0.25 * abs2(z[k]) - 0.1 * x[k+1])
    end
    return c
end

function _evalall(m)
    x0 = copy(m.meta.x0) .+ 0.01
    T = eltype(x0)
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

# Merging has no off switch, so the reference is independent of the merge
# machinery: constraint and objective VALUES are checked against the host
# model (and, on the host itself, derivative values against central finite
# differences of those values); device backends are checked against the host
# result on every callback.
function _fd_jac(m, x0)
    n = length(x0)
    J = zeros(m.meta.ncon, n)
    h = 1e-6
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

function _test_host_derivatives()
    m = ExaModel(_build(nothing))
    # the fixture's 10 constraint adds collapse to: one range family, one
    # data-tuple family, the named block, two augmentations (plain), and a
    # singleton; its 9 objective adds collapse to the generator objective
    # plus one merged family for the 8 loop-written ones
    @test length(m.cons) == 6
    @test length(m.objs) == 2
    x0 = copy(m.meta.x0) .+ 0.01
    # Jacobian values against central finite differences of cons!
    jr = zeros(Int, m.meta.nnzj); jc = zeros(Int, m.meta.nnzj)
    jv = zeros(m.meta.nnzj)
    NLPModels.jac_structure!(m, jr, jc)
    NLPModels.jac_coord!(m, x0, jv)
    J = zeros(m.meta.ncon, length(x0))
    for k in 1:m.meta.nnzj
        J[jr[k], jc[k]] += jv[k]
    end
    @test isapprox(J, _fd_jac(m, x0); rtol = 1e-6, atol = 1e-8)
    # Lagrangian Hessian values against central finite differences of
    # grad(obj) + J(x)'y (the Jacobian is validated just above)
    y = fill!(zeros(m.meta.ncon), 1.0)
    hr = zeros(Int, m.meta.nnzh); hc = zeros(Int, m.meta.nnzh)
    hv = zeros(m.meta.nnzh)
    NLPModels.hess_structure!(m, hr, hc)
    NLPModels.hess_coord!(m, x0, y, hv)
    H = zeros(length(x0), length(x0))
    for k in 1:m.meta.nnzh
        H[hr[k], hc[k]] += hv[k]
        hr[k] != hc[k] && (H[hc[k], hr[k]] += hv[k])
    end
    h = 1e-5
    Hfd = zeros(length(x0), length(x0))
    for j in 1:length(x0)
        xp = copy(x0); xp[j] += h
        xm = copy(x0); xm[j] -= h
        Hfd[:, j] .= (_gradL(m, xp, y) .- _gradL(m, xm, y)) ./ (2h)
    end
    @test isapprox(H, Hfd; rtol = 1e-4, atol = 1e-6)
end

function _test_backend_matches_host(backend)
    rb = _evalall(ExaModel(_build(backend)))
    rh = _evalall(ExaModel(_build(nothing)))
    @test rb[1] ≈ rh[1] rtol = 1e-12
    for k in (2, 3, 6, 9)                       # grad, cons, jac vals, hess vals
        @test sum(rb[k]) ≈ sum(rh[k]) rtol = 1e-10
        @test maximum(abs, rb[k]) ≈ maximum(abs, rh[k]) rtol = 1e-10
    end
    for k in (4, 5, 7, 8)                       # sparsity coordinates
        @test sort(rb[k]) == sort(rh[k])
    end
end

function runtests()
    @testset "Family merge" begin
        @testset "host derivatives against finite differences" begin
            _test_host_derivatives()
        end

        @testset "matches host result (backend = $b)" for b in MERGE_BACKENDS
            b === nothing || _test_backend_matches_host(b)
        end

        @testset "cross-variable and coefficient hoisting" begin
            c = ExaCore()
            c, x = add_var(c, 10; start = 0.3)
            c, y = add_var(c, 10; start = 0.8)
            c, _ = add_con(c, sin(x[i]) * x[i+1] - 0.1 for i in 1:9; lcon = -2.0, ucon = 2.0)
            c, _ = add_con(c, sin(y[i]) * y[i+1] - 0.1 for i in 1:9; lcon = -2.0, ucon = 2.0)
            c, _ = add_con(c, sin(x[i]) * x[i+1] - 0.7 for i in 1:9; lcon = -2.0, ucon = 2.0)
            c, _ = add_obj(c, abs2(x[i] - y[i]) for i in 1:10)
            m = ExaModel(c)
            @test length(m.cons) == 1
            x0 = copy(m.meta.x0)
            cv = zeros(m.meta.ncon)
            NLPModels.cons!(m, x0, cv)
            @test cv[1:9] ≈ fill(sin(0.3) * 0.3 - 0.1, 9) atol = 1e-15
            @test cv[10:18] ≈ fill(sin(0.8) * 0.8 - 0.1, 9) atol = 1e-15
            @test cv[19:27] ≈ fill(sin(0.3) * 0.3 - 0.7, 9) atol = 1e-15
            jr = zeros(Int, m.meta.nnzj); jc = zeros(Int, m.meta.nnzj)
            jv = zeros(m.meta.nnzj)
            NLPModels.jac_structure!(m, jr, jc)
            NLPModels.jac_coord!(m, x0, jv)
            # second block's rows must reference the second variable block's columns
            @test all(col -> 11 <= col <= 20, jc[(jr .>= 10) .& (jr .<= 18)])
            @test all(col -> 1 <= col <= 10, jc[jr .<= 9])
        end

        @testset "lazy segmented iterator for range families" begin
            c = ExaCore()
            c, x = add_var(c, 40; start = 0.5)
            c, _ = add_con(c, sin(x[i]) - 0.1 for i in 1:10; lcon = -2.0, ucon = 2.0)
            c, _ = add_con(c, sin(x[i]) - 0.1 for i in 21:30; lcon = -2.0, ucon = 2.0)
            c, _ = add_obj(c, x[1])
            m = ExaModel(c)
            @test length(m.cons) == 1
            itr = m.cons[1].itr
            @test itr isa ExaModels.SegmentedItr
            @test length(itr) == 20
            # rows carry correct offsets and original data
            @test itr[1].o0 == 1 && itr[11].o0 == 11
            @test itr[11].d == 21
        end

        @testset "type stability across replication" begin
            function build(U)
                c = ExaCore()
                c, x = add_var(c, 10U; start = 0.5)
                for u in 1:U
                    lo = 10 * (u - 1)
                    c, _ = add_con(c, sin(x[i]) * x[i+1] for i in (lo+1):(lo+9); lcon = -2.0, ucon = 2.0)
                    d = [(i = lo + i, a = 0.1u + 0.01i) for i in 1:9]
                    c, _ = add_con(c, e.a * exp(0.01 * x[e.i]) for e in d; lcon = -2.0, ucon = 2.0)
                end
                c, _ = add_obj(c, x[1])
                ExaModel(c)
            end
            m2 = build(2)
            m4 = build(4)
            @test typeof(m2) == typeof(m4)
            @test typeof(m2.cons) == typeof(m4.cons)
        end

        @testset "loop-written objectives merge" begin
            N = 16
            c = ExaCore()
            c, x = add_var(c, N+1; start = 0.5)
            c, y = add_var(c, N+1; start = 0.25)
            for i in 1:N
                c, _ = add_obj(c, 2.0 * abs2(x[i]) - 0.5 * x[i+1])
            end
            for i in 1:N                     # same pattern, other variables
                c, _ = add_obj(c, 2.0 * abs2(y[i]) - 0.5 * y[i+1])
            end
            c, _ = add_con(c, x[i] + y[i] - 1.0 for i in 1:N; lcon=-1.0, ucon=1.0)
            m = ExaModel(c)
            @test length(m.objs) == 1
            x0 = copy(m.meta.x0)
            # closed form at the starting point
            want = N * (2 * 0.5^2 - 0.5 * 0.5) + N * (2 * 0.25^2 - 0.5 * 0.25)
            @test NLPModels.obj(m, x0) ≈ want rtol = 1e-14
            g = zeros(length(x0))
            NLPModels.grad!(m, x0, g)
            # d/dx[i] = 4x[i] for i <= N, and -0.5 lands on x[i+1]
            @test g[1] ≈ 4 * 0.5 rtol = 1e-14
            @test g[N+1] ≈ -0.5 rtol = 1e-14
            @test g[N+2] ≈ 4 * 0.25 rtol = 1e-14
            @test g[2N+2] ≈ -0.5 rtol = 1e-14
            hr = zeros(Int, m.meta.nnzh); hc = zeros(Int, m.meta.nnzh)
            hv = zeros(m.meta.nnzh)
            NLPModels.hess_structure!(m, hr, hc)
            NLPModels.hess_coord!(m, x0, ones(m.meta.ncon), hv)
            @test sum(hv) ≈ 4.0 * 2N rtol = 1e-14
            # every Hessian entry is a diagonal of one of the two blocks
            @test all(hr .== hc)
        end

        @testset "merging is unconditional" begin
            c = ExaCore()
            c, x = add_var(c, 10; start = 0.5)
            c, _ = add_con(c, sin(x[i]) for i in 1:9; lcon = -2.0, ucon = 2.0)
            c, _ = add_con(c, sin(x[i]) for i in 1:9; lcon = -2.0, ucon = 2.0)
            c, _ = add_obj(c, x[1])
            @test length(ExaModel(c).cons) == 1
        end

        @testset "add_expr lift = true" begin
            function buildl(mode)
                c = ExaCore()
                c, x = add_var(c, 10; start = 1.0)
                c, e1 = add_expr(c, (sin(x[j]) + cos(x[j]) for j in 1:10); lift = mode)
                c, e2 = add_expr(c, (sin(e1[j]) + cos(e1[j]) for j in 1:10); lift = mode)
                c, _ = add_con(c, (exp(e2[j]) + abs2(e2[j+1]) for j in 1:9); lcon = 0.0, ucon = 0.0)
                c, _ = add_obj(c, x[1])
                ExaModel(c)
            end
            mS = buildl(false)
            mL = buildl(true)
            @test mL.meta.nvar == mS.meta.nvar + 20       # two lifted layers
            @test mL.meta.ncon == mS.meta.ncon + 20       # two defining blocks
            x0 = fill(0.7, 10)
            e1v = sin.(x0) .+ cos.(x0)
            e2v = sin.(e1v) .+ cos.(e1v)
            cS = NLPModels.cons(mS, x0)
            cL = NLPModels.cons(mL, vcat(x0, e1v, e2v))
            @test maximum(abs, cL[1:20]) < 1e-14          # defining rows at consistent point
            @test cL[21:end] ≈ cS atol = 1e-14
            # each lifted layer is pinned by its OWN equation
            cL2 = NLPModels.cons(mL, vcat(x0, e1v, e2v .+ 0.5))
            @test all(abs.(cL2[11:20] .- 0.5) .< 1e-14)
            @test maximum(abs, cL2[1:10]) < 1e-14
        end

        @testset "non-concrete refs storage" begin
            c = ExaCore()
            c, x = add_var(c, 10; start = 0.5, name = Val(:x))
            c, g1 = add_con(c, sin(x[i]) - 0.1 for i in 1:9; lcon = -1.0, ucon = 1.0, name = Val(:g1))
            c, _ = add_obj(c, abs2(x[i] - 1.0) for i in 1:10)
            @test getfield(c, :refs) isa Vector{Pair{Symbol, Any}}
            @test c.g1 isa ExaModels.Constraint
            @test get_cons(c, :g1) === c.g1
            m = ExaModel(c)
            @test m.g1 isa ExaModels.Constraint
            # concrete mode keeps the NamedTuple end to end
            cc = ExaCore(concrete = Val(true))
            cc, y = add_var(cc, 4; name = Val(:y))
            @test getfield(cc, :refs) isa NamedTuple
            @test cc.y === y
        end

        @testset "concrete mode merges by default" begin
            # merging is the default in both storage modes; the walk and the
            # decision are static, so concrete builders stay trim-compilable
            c = ExaCore(concrete = Val(true))
            c, x = add_var(c, 10; start = 0.5)
            c, _ = add_con(c, sin(x[i]) - 0.1 for i in 1:9; lcon = -2.0, ucon = 2.0)
            c, _ = add_con(c, sin(x[i]) - 0.4 for i in 1:9; lcon = -2.0, ucon = 2.0)
            c, _ = add_obj(c, x[1])
            m = ExaModel(c)
            @test length(m.cons) == 1
            cv = zeros(m.meta.ncon)
            NLPModels.cons!(m, copy(m.meta.x0), cv)
            @test cv[1:9] ≈ fill(sin(0.5) - 0.1, 9) atol = 1e-15
            @test cv[10:18] ≈ fill(sin(0.5) - 0.4, 9) atol = 1e-15
        end
    end
end

end # module MergeTest
