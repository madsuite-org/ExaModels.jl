using ExaModels
using Test
using SparseArrays

function luksan_vlcek_obj_param(x, θ, i, j)
    # θ[1] = 100, θ[2] = 1
    return θ[1] * (x[i-1, j]^2 - x[i, j])^2 + (x[i-1, j] - θ[2])^2
end

function luksan_vlcek_con1_param(x, θ, i, j)
    # θ[3] = 3, θ[4] = 2, θ[5] = 5
    return θ[3] * x[i+1, j]^3 + θ[4] * x[i+2, j] - θ[5]
end

function luksan_vlcek_con2_param(x, θ, i, j)
    # θ[6] = 4, θ[7] = 3
    return sin(x[i+1, j] - x[i+2, j])sin(x[i+1, j] + x[i+2, j]) + θ[6] * x[i+1, j] -
           x[i, j]exp(x[i, j] - x[i+1, j]) - θ[7]
end

function exa_luksan_vlcek_parametric(
    backend,
    N;
    M = 1,
    use_parameters = true,
    param_values = nothing,
)
    c = ExaCore(backend = backend, concrete = Val(true))
    @add_var(c, x, N, M; start = [luksan_vlcek_x0(i) for i = 1:N, j = 1:M])

    if use_parameters
        @add_par(c, θ, zeros(7))
        if !isnothing(param_values)
            set_value!(c, θ, param_values)
        else
            set_value!(c, θ, [100.0, 1.0, 3.0, 2.0, 5.0, 4.0, 3.0])
        end

        @add_con(c, s, luksan_vlcek_con1_param(x, θ, i, j) for i = 1:(N-2), j = 1:M)
        @add_con!(
            c,
            s,
            (i, j) => luksan_vlcek_con2_param(x, θ, i, j) for i = 1:(N-2), j = 1:M
        )
        @add_obj(c, luksan_vlcek_obj_param(x, θ, i, j) for i = 2:N, j = 1:M)

        return ExaModel(c; prod = true), c, (x, θ), (s,)
    else
        if isnothing(param_values)
            param_values = [100.0, 1.0, 3.0, 2.0, 5.0, 4.0, 3.0]
        end

        p1 = param_values[1]
        p2 = param_values[2]
        p3 = param_values[3]
        p4 = param_values[4]
        p5 = param_values[5]
        p6 = param_values[6]
        p7 = param_values[7]

        obj_func = (x, i, j) -> p1 * (x[i-1, j]^2 - x[i, j])^2 + (x[i-1, j] - p2)^2
        con1_func = (x, i, j) -> p3 * x[i+1, j]^3 + p4 * x[i+2, j] - p5
        con2_func =
            (x, i, j) ->
                sin(x[i+1, j] - x[i+2, j])sin(x[i+1, j] + x[i+2, j]) + p6 * x[i+1, j] -
                x[i, j]exp(x[i, j] - x[i+1, j]) - p7

        @add_con(c, s, con1_func(x, i, j) for i = 1:(N-2), j = 1:M)
        @add_con!(c, s, (i, j) => con2_func(x, i, j) for i = 1:(N-2), j = 1:M)
        @add_obj(c, obj_func(x, i, j) for i = 2:N, j = 1:M)

        return ExaModel(c; prod = true), c, (x,), (s,)
    end
end

function exa_ac_power_model_parametric(backend, filename; use_parameters = true)

    data = parse_ac_power_data(filename, backend)


    w = ExaModels.ExaCore(backend = backend, concrete = Val(true))

    @add_par(w, pd, map(b->b.pd, data.bus))
    @add_par(w, qd, map(b->b.qd, data.bus))

    @add_var(w, va, length(data.bus);)

    @add_var(
        w,
        vm,
        length(data.bus);
        start = fill!(similar(data.bus, eltype(w.x0)), one(eltype(w.x0))),
        lvar = data.vmin,
        uvar = data.vmax,
    )
    @add_var(w, pg, length(data.gen); lvar = data.pmin, uvar = data.pmax)

    @add_var(w, qg, length(data.gen); lvar = data.qmin, uvar = data.qmax)

    @add_var(w, p, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    @add_var(w, q, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    @add_par(w, cost2, map(g->g.cost2, data.gen))

    o = @add_obj(
        w,
        g.cost1 * pg[g.i]^2 + cost2[g.i] * pg[g.i] + g.cost3 for g in data.gen
    )

    @add_con(w, c1, va[i] for i in data.ref_buses)

    @add_con(
        w,
        c2,
        p[b.f_idx] - b.c5 * vm[b.f_bus]^2 -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch
    )

    @add_con(
        w,
        c3,
        q[b.f_idx] +
        b.c6 * vm[b.f_bus]^2 +
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch
    )

    @add_con(
        w,
        c4,
        p[b.t_idx] - b.c7 * vm[b.t_bus]^2 -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch
    )

    @add_con(
        w,
        c5,
        q[b.t_idx] +
        b.c8 * vm[b.t_bus]^2 +
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch
    )

    @add_con(
        w,
        c6,
        va[b.f_bus] - va[b.t_bus] for b in data.branch;
        lcon = data.angmin,
        ucon = data.angmax,
    )
    @add_con(
        w,
        c7,
        p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a_sq for b in data.branch;
        lcon = fill!(similar(data.branch, eltype(w.x0), length(data.branch)), eltype(w.x0)(-Inf)),
    )
    @add_con(
        w,
        c8,
        p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a_sq for b in data.branch;
        lcon = fill!(similar(data.branch, eltype(w.x0), length(data.branch)), eltype(w.x0)(-Inf)),
    )

    @add_par(w, bs, map(b->b.bs, data.bus))
    @add_par(w, gs, map(b->b.gs, data.bus))

    @add_con(w, c9, pd[b.i] + gs[b.i] * vm[b.i]^2 for b in data.bus)

    @add_con(w, c10, qd[b.i] - bs[b.i] * vm[b.i]^2 for b in data.bus)

    c11 = @add_con!(w, c9, a.bus => p[a.i] for a in data.arc)
    c12 = @add_con!(w, c10, a.bus => q[a.i] for a in data.arc)

    c13 = @add_con!(w, c9, g.bus => -pg[g.i] for g in data.gen)
    c14 = @add_con!(w, c10, g.bus => -qg[g.i] for g in data.gen)

    return ExaModels.ExaModel(w; prod = true),
    w,
    (va, vm, pg, qg, p, q),
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10)

end

function test_function_evaluations(model1, core1, model2; tol = sol_tolerance(eltype(model1.meta.x0),eltype(model2.meta.x0)))
    x_test = ExaModels.convert_array(randn(core1.nvar), core1.backend)

    model1 = WrapperNLPModel(model1)
    model2 = WrapperNLPModel(model2)

    obj1 = NLPModels.obj(model1, x_test)
    obj2 = NLPModels.obj(model2, x_test)
    @test obj1 ≈ obj2 atol=tol rtol=tol

    if core1.ncon > 0
        con1 = NLPModels.cons(model1, x_test)
        con2 = NLPModels.cons(model2, x_test)
        @test con1 ≈ con2 atol=tol rtol=tol
    end

    grad1 = NLPModels.grad(model1, x_test)
    grad2 = NLPModels.grad(model2, x_test)
    @test grad1 ≈ grad2 atol=tol rtol=tol

    u = ExaModels.convert_array(randn(core1.nvar), core1.backend)
    v = ExaModels.convert_array(randn(core1.ncon), core1.backend)
    jprod_param = NLPModels.jprod(model2, x_test, u)
    jprod_orig = NLPModels.jprod(model1, x_test, u)
    @test jprod_param ≈ jprod_orig atol=tol rtol=tol

    jtprod_param = NLPModels.jtprod(model2, x_test, v)
    jtprod_orig = NLPModels.jtprod(model1, x_test, v)
    @test jtprod_param ≈ jtprod_orig atol=tol rtol=tol

    y_test = ExaModels.convert_array(randn(core1.ncon), core1.backend)
    hprod_param = NLPModels.hprod(model2, x_test, y_test, u)
    hprod_orig = NLPModels.hprod(model1, x_test, y_test, u)
    @test hprod_param ≈ hprod_orig atol=tol rtol=tol
end

function test_real_only()
    c = ExaModels.ExaCore(concrete = Val(true))
    @add_var(c, x, 10)

    @add_con(c, c1, 1.0 for i = 1:2)
    o = @add_obj(c, 1.0 for i = 1:2)
    @add_con!(c, c1, j => -(x[i]-1)^2 for i = 1:10, j = 1:2)
    em = ExaModels.ExaModel(c)

    xval = rand(10)
    yval = randn(2)

    @test NLPModels.obj(em, xval) ≈ 2.0
    @test NLPModels.cons(em, xval) ≈ ones(2) .- sum((xval .- 1) .^ 2)
    @test NLPModels.jac(em, xval) ≈ [(-2 .* (xval .- 1))'; (-2 .* (xval .- 1))']
    @test NLPModels.hess(em, xval, yval) ≈ spdiagm(0=>fill(-2 * sum(yval), (10,)))
end

function test_param_only()
    c = ExaModels.ExaCore(concrete = Val(true))
    @add_var(c, x, 10)
    θval = rand(2)
    @add_par(c, θ, θval)

    @add_con(c, c1, θ[i] for i = 1:2)
    o = @add_obj(c, θ[i] for i = 1:2)

    @add_con!(c, c1, j => -(x[i]-1)^2 for i = 1:10, j = 1:2)
    em = ExaModels.ExaModel(c)

    xval = rand(10)
    yval = randn(2)

    @test NLPModels.obj(em, xval) ≈ sum(θval)
    @test NLPModels.cons(em, xval) ≈ θval .- sum((xval .- 1) .^ 2)
    @test NLPModels.jac(em, xval) ≈ [(-2 .* (xval .- 1))'; (-2 .* (xval .- 1))']
    @test NLPModels.hess(em, xval, yval) ≈ spdiagm(0=>fill(-2 * sum(yval), (10,)))
end

function test_parametric_vs_nonparametric(backend)
    @testset "Basic parametric" begin
        test_real_only()
        test_param_only()
    end
    @testset "Parametric luksan" begin
        @testset "Metadata" begin
            m_param, c_param, _, _ =
                exa_luksan_vlcek_parametric(backend, 4, M = 3, use_parameters = true)
            m_nonparam, c_nonparam, _, _ =
                exa_luksan_vlcek_parametric(backend, 4, M = 3, use_parameters = false)
            @test c_param.nvar == c_nonparam.nvar
            @test c_param.ncon == c_nonparam.ncon
            @test ExaModels.length(c_param.obj) == ExaModels.length(c_nonparam.obj)
            @test ExaModels.length(c_param.cons) == ExaModels.length(c_nonparam.cons)
            @test c_param.npar == 7
            @test c_nonparam.npar == 0
        end

        @testset "Default" begin
            m_param, c_param, _, _ =
                exa_luksan_vlcek_parametric(backend, 3, M = 2, use_parameters = true)
            m_nonparam, _, _, _ =
                exa_luksan_vlcek_parametric(backend, 3, M = 2, use_parameters = false)
            test_function_evaluations(m_param, c_param, m_nonparam)
        end
        @testset "Modified Parameters objective" begin
            modified_params1 = [200.0, 2.0, 3.0, 2.0, 5.0, 4.0, 3.0]
            m_param, c_param, _, _ = exa_luksan_vlcek_parametric(
                backend,
                3,
                M = 2,
                use_parameters = true,
                param_values = modified_params1,
            )
            m_nonparam, _, _, _ = exa_luksan_vlcek_parametric(
                backend,
                3,
                M = 2,
                use_parameters = false,
                param_values = modified_params1,
            )
            test_function_evaluations(m_param, c_param, m_nonparam)
        end

        @testset "Modified Parameters constraints" begin
            modified_params2 = [100.0, 1.0, 6.0, 4.0, 10.0, -8.0, 6.0]
            m_param, c_param, _, _ = exa_luksan_vlcek_parametric(
                backend,
                3,
                M = 2,
                use_parameters = true,
                param_values = modified_params2,
            )
            m_nonparam, _, _, _ = exa_luksan_vlcek_parametric(
                backend,
                3,
                M = 2,
                use_parameters = false,
                param_values = modified_params2,
            )
            test_function_evaluations(m_param, c_param, m_nonparam)
        end
        @testset "Modified Parameters all" begin
            modified_params3 = [150.0, 0.5, 2.5, 1.5, 7.5, 3.5, 4.5]
            m_param, c_param, _, _ = exa_luksan_vlcek_parametric(
                backend,
                3,
                M = 2,
                use_parameters = true,
                param_values = modified_params3,
            )
            m_nonparam, _, _, _ = exa_luksan_vlcek_parametric(
                backend,
                3,
                M = 2,
                use_parameters = false,
                param_values = modified_params3,
            )
            test_function_evaluations(m_param, c_param, m_nonparam)
        end

        @testset "Modify after build" begin
            m_param, c_param, (_, θ_param), _ =
                exa_luksan_vlcek_parametric(backend, 3, M = 2, use_parameters = true)
            new_params = [75.0, 1.5, 4.0, 3.0, 6.0, 5.0, 2.0]
            # After the model is built, its parameters are ITS state: a model
            # copies the core's θ when it is built, so setting values on the
            # core would leave this model on the values it was built with.
            set_value!(m_param, θ_param, new_params)
            m_nonparam, _, _, _ = exa_luksan_vlcek_parametric(
                backend,
                3,
                M = 2,
                use_parameters = false,
                param_values = new_params,
            )
            test_function_evaluations(m_param, c_param, m_nonparam)
        end
    end

    @testset "Parametric power" begin
        m_param, c_param, _, _ = exa_ac_power_model_parametric(
            backend,
            "pglib_opf_case14_ieee.m",
            use_parameters = true,
        )
        m_nonparam, _, _ = _exa_ac_power_model(backend, "pglib_opf_case14_ieee.m")
        test_function_evaluations(m_param, c_param, m_nonparam)
    end
end
