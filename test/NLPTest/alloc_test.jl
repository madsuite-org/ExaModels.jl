function test_callbacks_non_allocating(m)
    x = copy(m.meta.x0)
    y = zeros(eltype(x), m.meta.ncon)
    v = ones(eltype(x), m.meta.nvar)
    f = similar(x, m.meta.nvar)
    g = similar(x, m.meta.ncon)
    jac = similar(x, m.meta.nnzj)
    hess = similar(x, m.meta.nnzh)
    Hv = similar(x, m.meta.nvar)

    # warm up
    obj(m, x)
    grad!(m, x, f)
    cons_nln!(m, x, g)
    jac_coord!(m, x, jac)
    hess_coord!(m, x, y, hess)
    hprod!(m, x, y, v, Hv)

    return @testset "non-allocating callbacks" begin
        @test @allocated(obj(m, x)) == 0
        @test @allocated(grad!(m, x, f)) == 0
        @test @allocated(cons_nln!(m, x, g)) == 0
        @test @allocated(jac_coord!(m, x, jac)) == 0
        @test @allocated(hess_coord!(m, x, y, hess)) == 0
        @test @allocated(hprod!(m, x, y, v, Hv)) == 0
    end
end
