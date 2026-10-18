module ZeroNnzTest

# Models with no Jacobian or no Hessian nonzeros: every product and the
# compressed wrapper must return exact zeros (or the plain result) on each
# backend, never launch over an empty range.

using Test, ExaModels
import NLPModels
import ..BACKENDS

function _linear_model(backend)
    c = backend === nothing ? ExaCore() : ExaCore(backend = backend)
    c, x = add_var(c, 4; start = 0.5)
    c, _ = add_obj(c, 2 * x[i] for i in 1:4)
    c, _ = add_con(c, x[i] + x[i+1] for i in 1:3)
    return ExaModel(c; prod = true)
end

function _unconstrained_model(backend)
    c = backend === nothing ? ExaCore() : ExaCore(backend = backend)
    c, x = add_var(c, 4; start = 0.5)
    c, _ = add_obj(c, x[i]^2 for i in 1:4)
    return ExaModel(c; prod = true)
end

_dev(m, v) = copyto!(similar(m.meta.x0, length(v)), v)

function runtests()
    @testset "Zero-nonzero models" begin
        for backend in BACKENDS
            @testset "backend=$backend" begin
                m = _linear_model(backend)
                T = eltype(m.meta.x0)
                @test m.meta.nnzh == 0
                x = _dev(m, T[0.3, 0.4, 0.5, 0.6])
                v = _dev(m, T[1, -2, 3, -4])
                y = _dev(m, T[1, 2, 3])
                Hv = fill!(similar(m.meta.x0, 4), T(NaN))
                NLPModels.hprod!(m, x, y, v, Hv)
                @test all(iszero, Array(Hv))
                fill!(Hv, T(NaN))
                NLPModels.hprod!(m, x, v, Hv)
                @test all(iszero, Array(Hv))

                # The host `getptr` maps an empty sparsity to one segment, so
                # the compressed wrapper is checked on device backends only.
                if backend !== nothing
                    cm = ExaModels.CompressedNLPModel(m)
                    @test cm.meta.nnzh == 0
                    J = fill!(similar(m.meta.x0, cm.meta.nnzj), 0)
                    NLPModels.jac_coord!(cm, x, J)
                    @test sum(Array(J)) ≈ 6
                    NLPModels.hess_coord!(cm, x, y, similar(m.meta.x0, 0))
                end

                m = _unconstrained_model(backend)
                @test m.meta.nnzj == 0
                x = _dev(m, T[0.3, 0.4, 0.5, 0.6])
                v = _dev(m, T[1, -2, 3, -4])
                Jv = similar(m.meta.x0, 0)
                NLPModels.jprod_nln!(m, x, v, Jv)
                Jtv = fill!(similar(m.meta.x0, 4), T(NaN))
                NLPModels.jtprod_nln!(m, x, Jv, Jtv)
                @test all(iszero, Array(Jtv))

                if backend !== nothing
                    cm = ExaModels.CompressedNLPModel(m)
                    @test cm.meta.nnzj == 0
                    NLPModels.jac_coord!(cm, x, similar(m.meta.x0, 0))
                    r = similar(m.meta.x0, Int, 0)
                    NLPModels.jac_structure!(cm, r, similar(r))
                    H = fill!(similar(m.meta.x0, cm.meta.nnzh), 0)
                    NLPModels.hess_coord!(cm, x, similar(m.meta.x0, 0), H)
                    @test Array(H) ≈ fill(2, 4)
                end
            end
        end
    end
end

end # module
