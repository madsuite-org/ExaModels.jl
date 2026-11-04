module ReductionTest

# Callbacks that reduce many terms into few outputs: constraint rows with long and
# uneven augmentations (several blocks feeding the same rows, rows without any), a
# large objective sum, and the products J*v, J'*v and H*v. Each backend is compared
# with the host model (`backend = nothing`), which reduces in a plain loop.

using Test, ExaModels, NLPModels
import ..BACKENDS

function model(backend)
    c = ExaCore(Float64; backend, concrete = Val(true))
    c, x = add_var(c, 50; start = (0.1 * i for i in 1:50))
    c, g = add_con(c, x[i] * x[i + 1] for i in 1:12)
    # rows 1..10 get uneven shares of 20000 terms (row 1 about a third); rows 11, 12 none
    long = [(1 + (k % 3 == 0 ? 0 : k % 10), 1 + k % 50, 1 + (7k) % 50, 0.5 + sin(k)) for k in 1:20000]
    c, _ = add_con!(c, g, p[1] => p[4] * x[p[2]] * x[p[3]] for p in long)
    c, _ = add_con!(c, g, p[1] => sin(x[p[2]]) for p in [(1, 1), (1, 3), (2, 2), (5, 5), (5, 6), (5, 7)])
    c, _ = add_obj(c, (x[q[1]] - 0.3)^2 * x[q[2]] * q[3]
        for q in [(1 + k % 50, 1 + (k + 1) % 50, cos(k)) for k in 1:30000])
    return ExaModel(c; prod = true)
end

rel(a, b) = maximum(abs.(a .- b) ./ (1 .+ abs.(b)); init = 0.0)

function runtests()
    ref = model(nothing)
    x = ref.meta.x0 .+ 0.01 .* sin.(1:ref.meta.nvar)
    y = cos.(1:ref.meta.ncon)
    v = sin.(1:ref.meta.nvar) .+ 0.2
    w = cos.(1:ref.meta.ncon)
    expected = (
        cons(ref, x), grad(ref, x), jprod(ref, x, v), jtprod(ref, x, w),
        hprod(ref, x, y, v; obj_weight = 0.7),
    )
    @testset "reductions on $(something(backend, "host"))" for backend in BACKENDS
        backend === nothing && continue
        m = model(backend)
        dev(a) = ExaModels.convert_array(a, backend)
        got = (
            cons(m, dev(x)), grad(m, dev(x)), jprod(m, dev(x), dev(v)), jtprod(m, dev(x), dev(w)),
            hprod(m, dev(x), dev(y), dev(v); obj_weight = 0.7),
        )
        for (name, a, b) in zip((:cons, :grad, :jprod, :jtprod, :hprod), got, expected)
            @test rel(Array(a), b) < 1.0e-12
        end
        # rows without augmentation keep their own value
        @test Array(got[1])[11:12] ≈ expected[1][11:12]
    end
end

end # module ReductionTest
