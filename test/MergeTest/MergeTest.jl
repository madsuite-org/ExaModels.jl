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
function _build(backend; merge = true)
    n = 64
    c = ExaCore(; backend, merge)
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

function _test_equivalence(backend)
    m_merged = ExaModel(_build(backend))
    m_plain = ExaModel(_build(backend; merge = false))
    @test length(m_merged.cons) < length(m_plain.cons)
    rm = _evalall(m_merged)
    rp = _evalall(m_plain)
    @test rm[1] ≈ rp[1] rtol = 1e-12
    for k in (2, 3, 6, 9)                       # grad, cons, jac vals, hess vals
        @test sum(rm[k]) ≈ sum(rp[k]) rtol = 1e-10
        @test maximum(abs, rm[k]) ≈ maximum(abs, rp[k]) rtol = 1e-10
    end
    for k in (4, 5, 7, 8)                       # sparsity coordinates
        @test sort(rm[k]) == sort(rp[k])
    end
end

function runtests()
    @testset "Family merge" begin
        @testset "merged == unmerged (backend = $(b === nothing ? "CPU" : b))" for b in MERGE_BACKENDS
            _test_equivalence(b)
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

        @testset "merge = false escape" begin
            function build(merge)
                c = ExaCore(; merge)
                c, x = add_var(c, 10; start = 0.5)
                c, _ = add_con(c, sin(x[i]) for i in 1:9; lcon = -2.0, ucon = 2.0)
                c, _ = add_con(c, sin(x[i]) for i in 1:9; lcon = -2.0, ucon = 2.0)
                c, _ = add_obj(c, x[1])
                ExaModel(c)
            end
            @test length(build(true).cons) == 1
            @test length(build(false).cons) == 2
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

        @testset "concrete mode merges when asked" begin
            # concrete defaults to merge = false so that juliac-compiled
            # builders stay statically prunable; dynamic sessions opt in
            c = ExaCore(concrete = Val(true), merge = true)
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
