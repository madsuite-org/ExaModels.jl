module ReductionTest

# Callbacks that reduce many terms into few outputs: constraint rows with long and
# uneven augmentations (several blocks feeding the same rows, rows without any), a
# large objective sum (value and gradient), and the products J*v, J'*v and H*v. Each
# backend is compared with the host model (`backend = nothing`), which reduces in a
# plain loop.

using Test, ExaModels, NLPModels, SparseArrays
import ..BACKENDS

# in the backend's default element type (Float32 on Metal), data included
function model(backend)
    T = ExaModels.default_T(backend)
    c = ExaCore(T; backend, concrete = Val(true))
    c, x = add_var(c, 50; start = T[0.1 * i for i in 1:50])
    c, g = add_con(c, x[i] * x[i + 1] for i in 1:12)
    # rows 1..10 get uneven shares of 20000 terms (row 1 about a third); rows 11, 12 none
    long = [(1 + (k % 3 == 0 ? 0 : k % 10), 1 + k % 50, 1 + (7k) % 50, T(0.5 + sin(k))) for k in 1:20000]
    c, _ = add_con!(c, g, p[1] => p[4] * x[p[2]] * x[p[3]] for p in long)
    c, _ = add_con!(c, g, p[1] => sin(x[p[2]]) for p in [(1, 1), (1, 3), (2, 2), (5, 5), (5, 6), (5, 7)])
    c, _ = add_obj(
        c, (x[q[1]] - 0.3)^2 * x[q[2]] * q[3]
            for q in [(1 + k % 50, 1 + (k + 1) % 50, T(cos(k))) for k in 1:30000]
    )
    return ExaModel(c; prod = true)
end

# one row long enough that a workgroup sums more terms than it has work items
function long_row(backend)
    T = ExaModels.default_T(backend)
    c = ExaCore(T; backend, concrete = Val(true))
    c, x = add_var(c, 50; start = T[0.1 * i for i in 1:50])
    c, g = add_con(c, x[i] for i in 1:2)
    c, _ = add_con!(c, g, 1 => p[2] * x[p[1]] for p in [(1 + k % 50, T(sin(k))) for k in 1:400000])
    return ExaModel(c; prod = true)
end

# a block whose pairs share rows and leave others without a term
function shared_rows(backend)
    T = ExaModels.default_T(backend)
    c = ExaCore(T; backend, concrete = Val(true))
    c, x = add_var(c, 50; start = T[0.1 * i for i in 1:50])
    c, g = add_con(c, p[1] => sin(x[p[2]]) * x[p[2] + 1] for p in [(1 + i % 3, 1 + i % 40) for i in 1:400])
    c, h = add_con(c, x[i]^2 for i in 1:2)
    c, _ = add_con!(c, h, 2 => x[i] for i in 1:5)
    return ExaModel(c; prod = true)
end

rel(a, b) = maximum(abs.(a .- b) ./ (1 .+ abs.(b)); init = 0.0)

# Jacobian and Hessian as sparse matrices
idx(m, n) = similar(m.meta.x0, Int, n)
function structures(m)
    j = jac_structure!(m, idx(m, m.meta.nnzj), idx(m, m.meta.nnzj))
    h = hess_structure!(m, idx(m, m.meta.nnzh), idx(m, m.meta.nnzh))
    return Array.((j..., h...))
end
function matrices(m, x, y)
    idx(n) = similar(m.meta.x0, Int, n)
    ji, jj = jac_structure!(m, idx(m.meta.nnzj), idx(m.meta.nnzj))
    hi, hj = hess_structure!(m, idx(m.meta.nnzh), idx(m.meta.nnzh))
    J = sparse(Array(ji), Array(jj), Array(jac_coord(m, x)), m.meta.ncon, m.meta.nvar)
    H = sparse(Array(hi), Array(hj), Array(hess_coord(m, x, y)), m.meta.nvar, m.meta.nvar)
    return J, H
end

function runtests()
    ref = model(nothing)
    x = ref.meta.x0 .+ 0.01 .* sin.(1:ref.meta.nvar)
    y = cos.(1:ref.meta.ncon)
    v = sin.(1:ref.meta.nvar) .+ 0.2
    w = cos.(1:ref.meta.ncon)
    expected = (
        cons(ref, x), grad(ref, x), jprod(ref, x, v), jtprod(ref, x, w),
        hprod(ref, x, y, v; obj_weight = 0.7), [obj(ref, x)],
    )
    return @testset "reductions on $(something(backend, "host"))" for backend in BACKENDS
        backend === nothing && continue
        m = model(backend)
        T = eltype(m.meta.x0)
        dev(a) = ExaModels.convert_array(T.(a), backend)
        # Float32 backends (Metal) carry roundoff over sums of thousands of terms
        tol = T == Float32 ? 1.0e-3 : 1.0e-12
        got = (
            cons(m, dev(x)), grad(m, dev(x)), jprod(m, dev(x), dev(v)), jtprod(m, dev(x), dev(w)),
            hprod(m, dev(x), dev(y), dev(v); obj_weight = 0.7), [obj(m, dev(x))],
        )
        for (name, a, b) in zip((:cons, :grad, :jprod, :jtprod, :hprod, :obj), got, expected)
            @test rel(Array(a), b) < tol
        end
        # augmentation data is not reordered on any backend, so coordinates come in the same order
        @test structures(m) == structures(ref)
        Jm, Hm = matrices(m, dev(x), dev(y))
        Jr, Hr = matrices(ref, x, y)
        @test Jm.colptr == Jr.colptr && Jm.rowval == Jr.rowval && rel(Jm.nzval, Jr.nzval) < tol
        @test Hm.colptr == Hr.colptr && Hm.rowval == Hr.rowval && rel(Hm.nzval, Hr.nzval) < tol
        # CompressedNLPModel: duplicate coordinates (here many) summed on the device
        cm, cr = ExaModels.CompressedNLPModel(m), ExaModels.CompressedNLPModel(ref)
        @test cm.meta.nnzj == cr.meta.nnzj && cm.meta.nnzh == cr.meta.nnzh
        Jc, Hc = matrices(cm, dev(x), dev(y))
        Jcr, Hcr = matrices(cr, x, y)
        @test Jc.colptr == Jcr.colptr && Jc.rowval == Jcr.rowval && rel(Jc.nzval, Jcr.nzval) < tol
        @test Hc.colptr == Hcr.colptr && Hc.rowval == Hcr.rowval && rel(Hc.nzval, Hcr.nzval) < tol
        # rows without augmentation keep their own value
        @test Array(got[1])[11:12] ≈ expected[1][11:12] rtol = tol
        lm, lr = long_row(backend), long_row(nothing)
        lx, lv = lr.meta.x0, sin.(1:lr.meta.nvar)
        @test rel(Array(cons(lm, dev(lx))), cons(lr, lx)) < tol
        @test rel(Array(jprod(lm, dev(lx), dev(lv))), jprod(lr, lx, lv)) < tol
        sm, sr = shared_rows(backend), shared_rows(nothing)
        sw = cos.(1:sr.meta.ncon)
        @test rel(Array(cons(sm, dev(lx))), cons(sr, lx)) < tol
        @test rel(Array(jprod(sm, dev(lx), dev(lv))), jprod(sr, lx, lv)) < tol
        @test rel(Array(jtprod(sm, dev(lx), dev(sw))), jtprod(sr, lx, sw)) < tol
    end
end

end # module ReductionTest
