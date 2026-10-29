using ExaModels, NLPModels, SparseArrays, LinearAlgebra, Random

_dense(rows, cols, vals, m, n) = Matrix(sparse(rows, cols, vals, m, n))

function _jac(m, x)
    r, c = jac_structure(m)
    return _dense(r, c, jac_coord(m, x), m.meta.ncon, m.meta.nvar)
end
function _hess(m, x, y)
    r, c = hess_structure(m); H = _dense(r, c, hess_coord(m, x, y), m.meta.nvar, m.meta.nvar)
    return Symmetric(H + H' - Diagonal(diag(H)))  # lower triangle stored; diagonal counted once
end

"Write `m`, read it back, and compare everything NLPModels can see at a few random points."
function roundtrip_check(m; seed = 1, path = tempname() * ".h5", kwargs...)
    ExaModelsIO.write_model(path, m)
    m2 = ExaModelsIO.read_model(path; kwargs...)
    a, b = m.meta, m2.meta
    @test (a.nvar, a.ncon, a.minimize) == (b.nvar, b.ncon, b.minimize)
    for k in (:x0, :lvar, :uvar, :y0, :lcon, :ucon)
        @test getproperty(a, k) == getproperty(b, k)
    end
    @test m.θ == m2.θ
    sizes(mm) = [b.size for b in ExaModelsIO.constraint_blocks(mm)[1]]  # source blocks: merged families report only a row count
    @test isequal(sizes(m), sizes(m2))
    rng = MersenneTwister(seed)
    for _ in 1:3
        x = 0.1 .+ abs.(a.x0 .+ randn(rng, a.nvar)); y = randn(rng, a.ncon)  # positive, so log is defined
        @test obj(m, x) ≈ obj(m2, x)
        @test grad(m, x) ≈ grad(m2, x)
        @test cons(m, x) ≈ cons(m2, x)
        @test _jac(m, x) ≈ _jac(m2, x)
        @test _hess(m, x, y) ≈ _hess(m2, x, y)
    end
    return m2
end
