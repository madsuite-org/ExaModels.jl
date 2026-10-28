module MergeTest

using Test, ExaModels
import SparseArrays
import NLPModels

# Backend list comes from the suite driver when available (so GPU legs
# exercise merging too); standalone runs cover the plain CPU path.
const MERGE_BACKENDS = isdefined(Main, :BACKENDS) ? Main.BACKENDS : [nothing]

# A model exercising every merge path: a range family over two variable
# blocks plus a scalar-coefficient variant (lazy segmented path), a data-tuple
# family (lazy path over arrays), a same-type augmentation pair (stored plain
# on every backend — augmentations never merge), a named block, and a
# non-mergeable singleton.
function _build(backend)
    n = 64
    c = ExaCore(; backend)
    c, x = add_var(c, n; start = 0.3)
    c, z = add_var(c, n; start = 0.8)
    c, _ = add_con(c, sin(x[i]) * x[i + 1] - 0.1 for i in 1:(n - 1); lcon = -2.0, ucon = 2.0)
    c, _ = add_con(c, sin(z[i]) * z[i + 1] - 0.1 for i in 1:(n - 1); lcon = -2.0, ucon = 2.0)
    c, _ = add_con(c, sin(x[i]) * x[i + 1] - 0.7 for i in 1:(n - 1); lcon = -2.0, ucon = 2.0)
    data1 = [(i = i, a = 0.5 + 0.001i, b = 0.1) for i in 1:(n - 1)]
    data2 = [(i = i, a = 0.25 + 0.002i, b = 0.2) for i in 1:(n - 1)]
    c, _ = add_con(c, d.a * exp(0.01 * x[d.i]) + d.b * z[d.i + 1] for d in data1; lcon = -5.0, ucon = 5.0)
    c, _ = add_con(c, d.a * exp(0.01 * x[d.i]) + d.b * z[d.i + 1] for d in data2; lcon = -5.0, ucon = 5.0)
    c, g = add_con(c, x[i] + z[i] for i in 1:8; lcon = -1.0, ucon = 1.0, name = Val(:gnamed))
    c, _ = add_con!(c, g, i => sin(x[i + 1]) for i in 1:8)
    c, _ = add_con!(c, g, i => sin(z[i + 1]) for i in 1:8)
    c, _ = add_con(c, tanh(x[i]) * exp(z[i]) for i in 1:4; lcon = -3.0, ucon = 3.0)
    # an `exa_sum` reduction (SumNode) is statically non-mergeable: stored
    # plain (a generator `sum` folds into `+` nodes and would merge)
    c, _ = add_con(c, exa_sum(k -> x[k] * z[k], Val(1:3)) + z[i] for i in 1:4; lcon = -3.0, ucon = 3.0)
    c, _ = add_obj(c, abs2(x[i] - z[i]) for i in 1:n)
    # loop-written objectives: one scalar add per iteration, two variable
    # blocks, collapsing into a single objective family
    for k in 1:4
        c, _ = add_obj(c, 0.25 * abs2(x[k]) - 0.1 * z[k + 1])
    end
    for k in 1:4
        c, _ = add_obj(c, 0.25 * abs2(z[k]) - 0.1 * x[k + 1])
    end
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

# Merging has no off switch, so the reference is independent of the merge
# machinery: constraint and objective VALUES are checked against the host
# model (and, on the host itself, derivative values against central finite
# differences of those values); device backends are checked against the host
# result on every callback.
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

function _test_host_derivatives()
    m = ExaModel(_build(nothing))
    # the fixture's 10 constraint adds collapse to: one range family, one
    # data-tuple family, the named block and the tanh block (one-segment
    # families: no sibling), two augmentations (plain) and one statically
    # non-mergeable plain block (the `sum` reduction); its 9 objective adds
    # collapse to the generator objective plus one merged family for the 8
    # loop-written ones
    @test length(m.cons) == 7
    @test count(b -> b.itr isa ExaModels._MergedItr, m.cons) == 4   # four families
    @test count(b -> b isa ExaModels.ConstraintAugmentation, m.cons) == 2
    @test length(m.objs) == 2
    return _check_fd_derivatives(m)
end

# Jacobian and Lagrangian Hessian of a host model against central finite
# differences (of cons!, and of grad(obj) + J(x)'y respectively)
function _check_fd_derivatives(m)
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
    @test isapprox(J, _fd_jac(m, x0); rtol = 1.0e-6, atol = 1.0e-8)
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
    h = 1.0e-5
    Hfd = zeros(length(x0), length(x0))
    for j in 1:length(x0)
        xp = copy(x0); xp[j] += h
        xm = copy(x0); xm[j] -= h
        Hfd[:, j] .= (_gradL(m, xp, y) .- _gradL(m, xm, y)) ./ (2h)
    end
    return @test isapprox(H, Hfd; rtol = 1.0e-4, atol = 1.0e-6)
end

function _test_backend_matches_host(backend)
    mb = ExaModel(_build(backend))
    # every merged block is in the per-segment device layout: hoisted leaves
    # once per segment, one element per row
    for b in mb.cons
        b.itr isa ExaModels._MergedItr && @test false   # no host-side merged iterator survives finalization
        if b.itr isa ExaModels.DeviceSegItr
            @test length(b.itr.d) == length(b.itr)
            @test length(b.itr.seg_iv) == length(b.itr.offs)
            @test length(b.itr.offs) > 1                 # single segments take the field layout
        elseif b.itr isa ExaModels.DeviceSeg1Itr
            @test length(b.itr.d) == length(b.itr)
        end
    end
    # four families: the range and data-tuple families have 2 segments, the
    # named block and the tanh block are single-segment
    @test count(b -> b.itr isa ExaModels.DeviceSegItr, mb.cons) == 2
    @test count(b -> b.itr isa ExaModels.DeviceSeg1Itr, mb.cons) == 2
    # Float32 backends (Metal) carry ~1e-7 roundoff per operation
    tight, loose = eltype(mb.meta.x0) == Float32 ? (1.0e-4, 1.0e-4) : (1.0e-12, 1.0e-10)
    rb = _evalall(mb)
    rh = _evalall(ExaModel(_build(nothing)))
    @test rb[1] ≈ rh[1] rtol = tight
    @test rb[2] ≈ rh[2] rtol = loose                 # grad
    @test rb[3] ≈ rh[3] rtol = loose                 # cons
    # Jacobian and Hessian: the coordinate lists may legitimately differ in
    # order between host and device, so compare the assembled matrices
    # (duplicate coordinates sum), not sums or norms of the value vectors
    nvar, ncon = length(rh[2]), length(rh[3])
    @test SparseArrays.sparse(rb[4], rb[5], rb[6], ncon, nvar) ≈
        SparseArrays.sparse(rh[4], rh[5], rh[6], ncon, nvar) rtol = loose
    @test SparseArrays.sparse(rb[7], rb[8], rb[9], nvar, nvar) ≈
        SparseArrays.sparse(rh[7], rh[8], rh[9], nvar, nvar) rtol = loose
    return
end

# The core API is persistent: a later add on a core must not change models
# already built from it. Merged families append by copy
# (`_merged_append_itr`), which this testset locks in for both the lazy
# (SegmentedItr) and the materialized (MergedRows, pair-headed) iterator.
# Branching two cores off one base is NOT covered: the bounds vectors
# (y0/lcon/ucon) are grown in place by `_append_slot` on every add_con, so
# sibling cores share bounds state. That predates merging (#294) and holds
# for plain blocks too.
function _test_core_persistence()
    return @testset "appending to a family preserves earlier models" begin
        c0 = ExaCore()
        c0, x = add_var(c0, 10; start = 0.5)
        c1, _ = add_con(c0, sin(x[i]) for i in 1:5; lcon = -2.0, ucon = 2.0)
        c1, _ = add_obj(c1, x[1])
        m1 = ExaModel(c1)
        @test m1.meta.ncon == 5 && length(m1.cons[1].itr) == 5
        # a later same-family add on the core must not grow m1's block
        c2, _ = add_con(c1, sin(x[i]) for i in 6:10; lcon = -2.0, ucon = 2.0)
        @test length(m1.cons[1].itr) == 5
        g = zeros(m1.meta.ncon)
        NLPModels.cons!(m1, fill(0.5, 10), g)
        @test g ≈ fill(sin(0.5), 5)
        m2 = ExaModel(c2)
        @test m2.meta.ncon == 10 && length(m2.cons[1].itr) == 10
        g2 = zeros(10)
        NLPModels.cons!(m2, fill(0.5, 10), g2)
        @test g2 ≈ fill(sin(0.5), 10)

        # materialized path: pair-headed plain blocks (data-driven row targets)
        # merge into MergedRows on the host. Augmentations (add_con!) never
        # merge; they are stored plain on every backend.
        d0 = ExaCore()
        d0, z = add_var(d0, 10; start = 0.5)
        d1, _ = add_con(d0, (i => sin(z[i]) for i in 1:4); lcon = -2.0, ucon = 2.0)
        d1, _ = add_obj(d1, z[1])
        n1 = ExaModel(d1)
        @test n1.cons[1].itr isa ExaModels.MergedRows
        @test n1.meta.ncon == 4 && length(n1.cons[1].itr) == 4
        # the same generator again is the same family (row targets are
        # per-block: the second block's rows land at its own o0 base)
        d2, _ = add_con(d1, (i => sin(z[i]) for i in 1:4); lcon = -2.0, ucon = 2.0)
        @test length(n1.cons[1].itr) == 4
        gn = zeros(n1.meta.ncon)
        NLPModels.cons!(n1, fill(0.5, 10), gn)
        @test gn ≈ fill(sin(0.5), 4)
        n2 = ExaModel(d2)
        @test n2.meta.ncon == 8 && length(n2.cons[1].itr) == 8
        gn2 = zeros(8)
        NLPModels.cons!(n2, fill(0.5, 10), gn2)
        @test gn2 ≈ fill(sin(0.5), 8)
    end
end

# Backends without Float64 (Metal) upload merged rows through
# `replace_float_64_by_32`. The generic struct rebuild cannot reconstruct a
# `MergedRow` (an empty Float-slot tuple leaves its `T` undetermined), so the
# row has a dedicated method; host-only check that it rewrites `d` and keeps
# the row's layout.
function _test_merged_row_float32_rewrite()
    c = ExaCore()
    c, x = add_var(c, 4)
    data = [(i, 0.5 * i) for i in 1:4]
    c, _ = add_con(c, x[d[1]] * d[2] + x[1] for d in data)
    m = ExaModel(c)
    itr = m.cons[1].itr
    @test itr isa ExaModels.SegmentedItr
    rows = [itr[i] for i in 1:length(itr)]
    @test ExaModels._contains_float64(eltype(rows))          # the fixture does carry a Float64
    K, S = eltype(rows).parameters[1], eltype(rows).parameters[2]
    r2 = ExaModels.replace_float_64_by_32.(rows)
    @test !ExaModels._contains_float64(eltype(r2))
    @test eltype(r2) <: ExaModels.MergedRow{K, S, Float32}   # K, S kept; T rewritten, not Union{}
    @test r2[1].d === (1, 0.5f0)
    @test r2[2].d === (2, 1.0f0)
    return @test r2[1].iv == rows[1].iv && r2[1].o0 == rows[1].o0 && r2[1].o1 == rows[1].o1 && r2[1].o2 == rows[1].o2
end

# Integer-leaf slot sharing: equal-valued leaves of a block read one slot, so
# two sites that reference the same variable compare === in the sparsity probe
# and share a Jacobian slot (and Hessian pairs), as the plain block did. The
# sharing pattern is part of the family key.
function _test_slot_sharing()
    # LuksanVlcek: x[i+1] appears 5 times, x[i+2] 3 times, x[i] 2 times per
    # row; the plain-block footprint is 3 Jacobian entries per row and 6
    # Hessian pairs per constraint row plus 3 per objective row
    N = 50
    c = ExaCore()
    c, x = add_var(c, N; start = [mod(i, 2) == 1 ? -1.2 : 1.0 for i in 1:N])
    c, _ = add_con(c, 3x[i + 1]^3 + 2 * x[i + 2] - 5 + sin(x[i + 1] - x[i + 2])sin(x[i + 1] + x[i + 2]) + 4x[i + 1] - x[i]exp(x[i] - x[i + 1]) - 3 for i in 1:(N - 2))
    c, _ = add_obj(c, 100 * (x[i - 1]^2 - x[i])^2 + (x[i - 1] - 1)^2 for i in 2:N)
    m = ExaModel(c)
    @test m.meta.nnzj == 3 * (N - 2)
    @test m.meta.nnzh == 6 * (N - 2) + 3 * (N - 1)
    _check_fd_derivatives(m)

    # the compiler test's recipe fixture (`h * x[i] + x[i]`): one entry per row
    c = ExaCore()
    c, x = add_var(c, 10)
    c, _ = add_con(c, 0.5 * x[i] + x[i] for i in 1:10)
    c, _ = add_obj(c, x[i]^2 for i in 1:10)
    m = ExaModel(c)
    @test m.meta.nnzj == 10
    @test m.meta.nnzh == 10

    # same tree type, different sharing pattern: two families, each with its
    # own footprint; same pattern with other leaf values: one family
    c = ExaCore()
    c, x = add_var(c, 12; start = 0.3)
    c, _ = add_con(c, x[i + 1] * x[i + 1] for i in 1:8)     # pattern (a, b, a, b)
    c, _ = add_con(c, x[i + 1] * x[i + 2] for i in 1:8)     # pattern (a, b, c, b)
    c, _ = add_con(c, x[i + 2] * x[i + 2] for i in 1:8)     # (a, b, a, b) again: joins the first
    c, _ = add_obj(c, x[i]^2 for i in 1:12)
    m = ExaModel(c)
    @test length(m.cons) == 2
    @test sort([length(b.itr) for b in m.cons]) == [8, 16]
    # slot references print as ⟨i k⟩ / ⟨f k⟩, with and without the type header
    @test occursin("⟨i", sprint(show, m.cons[1].f.f))
    @test sprint(show, MIME"text/plain"(), ExaModels.IvRef(1)) isa String
    @test m.meta.nnzj == 16 * 1 + 8 * 2
    @test m.meta.ncon == 24
    _check_fd_derivatives(m)

    # concrete storage keeps one slot per leaf position (the family decision
    # must stay a function of types), so the same two blocks form ONE family
    # there and each row carries two Jacobian entries
    cc = ExaCore(concrete = Val(true))
    cc, xx = add_var(cc, 12; start = 0.3)
    cc, _ = add_con(cc, xx[i + 1] * xx[i + 1] for i in 1:8)
    cc, _ = add_con(cc, xx[i + 1] * xx[i + 2] for i in 1:8)
    cc, _ = add_obj(cc, xx[i]^2 for i in 1:12)
    mc = ExaModel(cc)
    @test length(mc.cons) == 1 && length(mc.cons[1].itr) == 16
    @test mc.meta.nnzj == 16 * 2
    _check_fd_derivatives(mc)
    return nothing
end

function runtests()
    return @testset "Family merge" begin
        @testset "Integer-leaf slot sharing" begin
            _test_slot_sharing()
        end
        @testset "merged rows rewrite Float64 to Float32" begin
            _test_merged_row_float32_rewrite()
        end
        @testset "host derivatives against finite differences" begin
            _test_host_derivatives()
        end

        @testset "matches host result (backend = $b)" for b in MERGE_BACKENDS
            b === nothing || _test_backend_matches_host(b)
        end

        _test_core_persistence()

        @testset "cross-variable and coefficient hoisting" begin
            c = ExaCore()
            c, x = add_var(c, 10; start = 0.3)
            c, y = add_var(c, 10; start = 0.8)
            c, _ = add_con(c, sin(x[i]) * x[i + 1] - 0.1 for i in 1:9; lcon = -2.0, ucon = 2.0)
            c, _ = add_con(c, sin(y[i]) * y[i + 1] - 0.1 for i in 1:9; lcon = -2.0, ucon = 2.0)
            c, _ = add_con(c, sin(x[i]) * x[i + 1] - 0.7 for i in 1:9; lcon = -2.0, ucon = 2.0)
            c, _ = add_obj(c, abs2(x[i] - y[i]) for i in 1:10)
            m = ExaModel(c)
            @test length(m.cons) == 1
            x0 = copy(m.meta.x0)
            cv = zeros(m.meta.ncon)
            NLPModels.cons!(m, x0, cv)
            @test cv[1:9] ≈ fill(sin(0.3) * 0.3 - 0.1, 9) atol = 1.0e-15
            @test cv[10:18] ≈ fill(sin(0.8) * 0.8 - 0.1, 9) atol = 1.0e-15
            @test cv[19:27] ≈ fill(sin(0.3) * 0.3 - 0.7, 9) atol = 1.0e-15
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
                    c, _ = add_con(c, sin(x[i]) * x[i + 1] for i in (lo + 1):(lo + 9); lcon = -2.0, ucon = 2.0)
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
            c, x = add_var(c, N + 1; start = 0.5)
            c, y = add_var(c, N + 1; start = 0.25)
            for i in 1:N
                c, _ = add_obj(c, 2.0 * abs2(x[i]) - 0.5 * x[i + 1])
            end
            for i in 1:N                     # same pattern, other variables
                c, _ = add_obj(c, 2.0 * abs2(y[i]) - 0.5 * y[i + 1])
            end
            c, _ = add_con(c, x[i] + y[i] - 1.0 for i in 1:N; lcon = -1.0, ucon = 1.0)
            m = ExaModel(c)
            @test length(m.objs) == 1
            x0 = copy(m.meta.x0)
            # closed form at the starting point
            want = N * (2 * 0.5^2 - 0.5 * 0.5) + N * (2 * 0.25^2 - 0.5 * 0.25)
            @test NLPModels.obj(m, x0) ≈ want rtol = 1.0e-14
            g = zeros(length(x0))
            NLPModels.grad!(m, x0, g)
            # d/dx[i] = 4x[i] for i <= N, and -0.5 lands on x[i+1]
            @test g[1] ≈ 4 * 0.5 rtol = 1.0e-14
            @test g[N + 1] ≈ -0.5 rtol = 1.0e-14
            @test g[N + 2] ≈ 4 * 0.25 rtol = 1.0e-14
            @test g[2N + 2] ≈ -0.5 rtol = 1.0e-14
            hr = zeros(Int, m.meta.nnzh); hc = zeros(Int, m.meta.nnzh)
            hv = zeros(m.meta.nnzh)
            NLPModels.hess_structure!(m, hr, hc)
            NLPModels.hess_coord!(m, x0, ones(m.meta.ncon), hv)
            @test sum(hv) ≈ 4.0 * 2N rtol = 1.0e-14
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
                c, _ = add_con(c, (exp(e2[j]) + abs2(e2[j + 1]) for j in 1:9); lcon = 0.0, ucon = 0.0)
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
            @test maximum(abs, cL[1:20]) < 1.0e-14          # defining rows at consistent point
            @test cL[21:end] ≈ cS atol = 1.0e-14
            # each lifted layer is pinned by its OWN equation
            cL2 = NLPModels.cons(mL, vcat(x0, e1v, e2v .+ 0.5))
            @test all(abs.(cL2[11:20] .- 0.5) .< 1.0e-14)
            @test maximum(abs, cL2[1:10]) < 1.0e-14
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
            @test cv[1:9] ≈ fill(sin(0.5) - 0.1, 9) atol = 1.0e-15
            @test cv[10:18] ≈ fill(sin(0.5) - 0.4, 9) atol = 1.0e-15
        end
    end
end

end # module MergeTest
