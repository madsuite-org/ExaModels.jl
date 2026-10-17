module JuMPTest

using Test

import ExaModels
import Ipopt
import JuMP
using JuMP: MOI
import NLPModels
import NLPModelsIpopt
import NLPModelsJuMP
import PowerModels

import ..NLPTest
import ..BACKENDS
import ..sol_tolerance
import ..solver_tolerance

function runtests()
    is_test(name) = startswith("$name", "test_")
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_moi_tests()
    model = MOI.instantiate(
        () -> ExaModels.Optimizer(NLPModelsIpopt.ipopt);
        with_bridge_type = Float64,
        with_cache_type = Float64,
    )
    MOI.set(model, MOI.RawOptimizerAttribute("print_level"), 0)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(;
            atol = 1e-4,
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.DualObjectiveValue,
                MOI.ObjectiveBound,
                MOI.SolverVersion,
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
            ],
        ),
        exclude = [
            # NLPModels doesn't detect unboundedness
            r"^test_linear_DUAL_INFEASIBLE$",
            r"^test_linear_DUAL_INFEASIBLE_2$",
            r"^test_solve_TerminationStatus_DUAL_INFEASIBLE$",
            # Returns INVALID_MODEL becuase of the empty row
            r"^test_linear_VectorAffineFunction_empty_row$",
            # Ipopt fails because of co-linear constraint and objective and
            # redundant constraint.
            r"^test_linear_transform$",
        ],
    )
    return
end

function test_nlp_legacy()
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:10])
    JuMP.@NLobjective(jm, Min, sum(x[i] for i=1:10))
    @test_throws ErrorException ExaModels.ExaModel(jm)
    jm = JuMP.Model(() -> ExaModels.Optimizer(NLPModelsIpopt.ipopt))
    @test_throws ErrorException JuMP.optimize!(jm)
    return
end

function test_fixed_variable_e2etest()
    N = 5
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:N])
    JuMP.fix(x[1], 1.0)
    JuMP.@constraint(jm, sum(x) == 1.0)
    JuMP.@objective(jm, Min, sum(2*x[i]^2 for i = 1:N))
    em = ExaModels.ExaModel(jm)
    @test only(em.meta.lcon) == only(em.meta.ucon) == 1.0
    @test em.cons[1] isa ExaModels.ConstraintAugmentation
    # family merging folds the Pair row target into per-row data, so the
    # lowering is checked by value: at x = (1, 2, ..., N) the single row
    # accumulates sum(x), with x[1] fixed contributing its bound value
    xt = collect(1.0:N)
    cv = NLPModels.cons(em, xt)
    @test only(cv) ≈ sum(xt)
    @test length(em.objs) == 1
    @test ExaModels.obj(em, xt) ≈ sum(2 * xt[i]^2 for i = 1:N)
    return
end

function test_parameter_e2etest()
    N = 5
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:N])
    JuMP.@variable(jm, p in JuMP.Parameter(1.0))
    JuMP.@constraint(jm, sum(x) == p)
    JuMP.@objective(jm, Min, sum(x))
    em = ExaModels.ExaModel(jm)
    @test only(em.meta.lcon) == only(em.meta.ucon) == 0.0
    @test only(em.θ) == 1.0
    @test em.cons[1] isa ExaModels.ConstraintAugmentation
    @test em.cons[2] isa ExaModels.ConstraintAugmentation
    # lowering checked by value: the row accumulates sum(x) - p
    xt = fill(0.5, N)
    cv = NLPModels.cons(em, xt)
    @test only(cv) ≈ sum(xt) - 1.0

    jm = JuMP.Model()
    JuMP.@variable(jm, x)
    @test ExaModels.ExaModel(jm) isa ExaModels.ExaModel
    return
end

function test_no_constraints_e2etest()
    N = 5
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:N])
    JuMP.@objective(jm, Max, sum(sin(x[i]) for i = 1:N))
    em = ExaModels.ExaModel(jm)
    @test isempty(em.cons)
    @test length(em.objs) == 1
    @test em.objs[1].f.f isa
        ExaModels.Node1{typeof(sin),<:ExaModels.Var{<:ExaModels.DataIndexed}}
    return
end

function test_no_constraints_simd_failure()
    N = 5
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:N])
    JuMP.@objective(jm, Max, sin(sum(x[i] for i = 1:N)))
    em = ExaModels.ExaModel(jm)
    @test isempty(em.cons)
    @test length(em.objs) == 1
    # broken since ExaMOI fails to detect SIMD in this case
    @test_broken em.objs[1].f.f isa ExaModels.Node1{typeof(sin),<:ExaModels.Var}
    return
end

function test_generic_e2etest()
    N = 5
    jm = JuMP.GenericModel{Float32}()
    JuMP.@variable(jm, x[1:N])
    JuMP.@constraint(jm, sum(x) == 1.0f0)
    JuMP.@objective(jm, Min, sum(x[i]^2 for i = 1:N))
    em = ExaModels.ExaModel(jm)
    @test typeof(em) <: ExaModels.ExaModel{Float32}
    # the MOI lowering assembles its blocks directly (not through add_con),
    # so they are stored plain regardless of the merge default
    @test eltype(em.cons[1].itr) <: Tuple{Int,Float32,Int}
    return
end

function _test_jump_interface(modelfunction, case)
    jm = modelfunction(case)
    JuMP.set_optimizer(jm, Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(jm, "print_level", 0)
    JuMP.optimize!(jm)
    sol = JuMP.value.(JuMP.all_variables(jm))
    dsol = JuMP.dual.(JuMP.all_constraints(jm, include_variable_in_set_constraints = true))
    JuMP.set_optimizer(jm, () -> ExaModels.Optimizer(NLPModelsIpopt.ipopt))
    JuMP.set_optimizer_attribute(jm, "print_level", 0)
    JuMP.optimize!(jm)
    sol2 = JuMP.value.(JuMP.all_variables(jm))
    dsol2 = JuMP.dual.(JuMP.all_constraints(jm, include_variable_in_set_constraints = true))
    @test sol ≈ sol2 atol = sol_tolerance(eltype(sol), eltype(sol2))
    @test dsol ≈ dsol2 atol = sol_tolerance(eltype(sol), eltype(sol2))
    @testset "$backend" for backend in BACKENDS
        m = ExaModels.WrapperNLPModel(ExaModels.ExaModel(jm; backend))
        result = NLPModelsIpopt.ipopt(
            m;
            print_level = 0,
            tol = solver_tolerance(eltype(m.inner.meta.x0)),
        )
        @test sol ≈ result.solution atol = sol_tolerance(eltype(m.inner.meta.x0))
    end
    return
end

function jump_luksan_vlcek_model(N)
    jm = JuMP.Model()
    JuMP.@variable(jm, x[i=1:N], start = mod(i, 2) == 1 ? -1.2 : 1.0)
    JuMP.@constraint(
        jm,
        s[i=1:(N-2)],
        3x[i+1]^3 + 2x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
        x[i]exp(x[i] - x[i+1]) - 3 == 0.0
    )
    JuMP.@objective(jm, Min, sum(100(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N))
    return jm
end

function test_jump_luksan_vlcek()
    @testset "$N" for N in [3, 10]
        _test_jump_interface(jump_luksan_vlcek_model, N)
    end
    return
end

function jump_ac_power_model(filename::String)
    ref = NLPTest.get_power_data_ref(filename)
    model = JuMP.Model()
    JuMP.@variable(model, va[i in keys(ref[:bus])])
    JuMP.@variable(model, will_delete)
    JuMP.@variable(
        model,
        ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"],
        start = 1.0
    )
    JuMP.@variable(
        model,
        ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"]
    )
    JuMP.@variable(
        model,
        ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"]
    )
    JuMP.@variable(
        model,
        -ref[:branch][l]["rate_a"] <=
        p[(l, i, j) in ref[:arcs]] <=
        ref[:branch][l]["rate_a"]
    )
    JuMP.@variable(
        model,
        -ref[:branch][l]["rate_a"] <=
        q[(l, i, j) in ref[:arcs]] <=
        ref[:branch][l]["rate_a"]
    )
    JuMP.@objective(
        model,
        Min,
        sum(
            gen["cost"][1] * pg[i]^2 + gen["cost"][2] * pg[i] + gen["cost"][3] for
            (i, gen) in ref[:gen]
        )
    )
    for (i, bus) in ref[:ref_buses]
        JuMP.@constraint(model, va[i] == 0)
    end
    for (i, bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]
        JuMP.@constraint(
            model,
            sum(p[a] for a in ref[:bus_arcs][i]) ==
            sum(pg[g] for g in ref[:bus_gens][i]) - sum(load["pd"] for load in bus_loads) -
            sum(shunt["gs"] for shunt in bus_shunts) * vm[i]^2
        )
        JuMP.@constraint(
            model,
            sum(q[a] for a in ref[:bus_arcs][i]) ==
            sum(qg[g] for g in ref[:bus_gens][i]) - sum(load["qd"] for load in bus_loads) +
            sum(shunt["bs"] for shunt in bus_shunts) * vm[i]^2
        )
    end
    # Branch power flow physics and limit constraints
    for (i, branch) in ref[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        t_idx = (i, branch["t_bus"], branch["f_bus"])
        p_fr = p[f_idx]
        q_fr = q[f_idx]
        p_to = p[t_idx]
        q_to = q[t_idx]
        vm_fr = vm[branch["f_bus"]]
        vm_to = vm[branch["t_bus"]]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]
        b_fr = branch["b_fr"]
        g_to = branch["g_to"]
        b_to = branch["b_to"]
        # From side of the branch flow
        JuMP.@constraint(
            model,
            p_fr ==
            (g + g_fr) / ttm * vm_fr^2 +
            (-g * tr + b * ti) / ttm * (vm_fr * vm_to * cos(va_fr - va_to)) +
            (-b * tr - g * ti) / ttm * (vm_fr * vm_to * sin(va_fr - va_to))
        )
        JuMP.@constraint(
            model,
            q_fr ==
            -(b + b_fr) / ttm * vm_fr^2 -
            (-b * tr - g * ti) / ttm * (vm_fr * vm_to * cos(va_fr - va_to)) +
            (-g * tr + b * ti) / ttm * (vm_fr * vm_to * sin(va_fr - va_to))
        )
        # To side of the branch flow
        JuMP.@constraint(
            model,
            p_to ==
            (g + g_to) * vm_to^2 +
            (-g * tr - b * ti) / ttm * (vm_to * vm_fr * cos(va_to - va_fr)) +
            (-b * tr + g * ti) / ttm * (vm_to * vm_fr * sin(va_to - va_fr))
        )
        JuMP.@constraint(
            model,
            q_to ==
            -(b + b_to) * vm_to^2 -
            (-b * tr + g * ti) / ttm * (vm_to * vm_fr * cos(va_to - va_fr)) +
            (-g * tr - b * ti) / ttm * (vm_to * vm_fr * sin(va_to - va_fr))
        )
        # Voltage angle difference limit
        JuMP.@constraint(model, branch["angmin"] <= va_fr - va_to <= branch["angmax"])
        # Apparent power limit, from side and to side
        JuMP.@constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        JuMP.@constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
    end
    JuMP.delete(model, will_delete)
    return model
end

function test_jump_ac_power_model()
    @testset "$file" for file in ["pglib_opf_case3_lmbd.m", "pglib_opf_case14_ieee.m"]
        _test_jump_interface(jump_ac_power_model, file)
    end
    return
end

function _jacobian_matrix(model, x)
    rows = zeros(Int, model.meta.nnzj)
    cols = zeros(Int, model.meta.nnzj)
    values = zeros(eltype(x), model.meta.nnzj)
    NLPModels.jac_structure!(model, rows, cols)
    NLPModels.jac_coord!(model, x, values)
    jacobian = zeros(eltype(x), model.meta.ncon, model.meta.nvar)
    for k in eachindex(values)
        jacobian[rows[k], cols[k]] += values[k]
    end
    return jacobian
end

function _hessian_matrix(model, x, y; obj_weight)
    rows = zeros(Int, model.meta.nnzh)
    cols = zeros(Int, model.meta.nnzh)
    values = zeros(eltype(x), model.meta.nnzh)
    NLPModels.hess_structure!(model, rows, cols)
    NLPModels.hess_coord!(model, x, y, values; obj_weight = obj_weight)
    hessian = zeros(eltype(x), model.meta.nvar, model.meta.nvar)
    for k in eachindex(values)
        hessian[rows[k], cols[k]] += values[k]
        if rows[k] != cols[k]
            hessian[cols[k], rows[k]] += values[k]
        end
    end
    return hessian
end

function _test_callback_equivalence(
    model::JuMP.GenericModel{T},
    points::Vector{Vector{T}},
) where {T}
    model_exa = ExaModels.ExaModel(model)
    model_nlp = NLPModelsJuMP.MathOptNLPModel(model)
    # Constraint rows may be ordered differently by the two adapters. Restrict
    # this helper to models with at most one constraint, where direct callback
    # comparison is unambiguous.
    @assert model_exa.meta.ncon <= 1
    y = T(0.37) .* collect(T, 1:model_exa.meta.ncon)
    obj_weight = T(0.61)
    for x in points
        @test NLPModels.obj(model_exa, x) ≈ NLPModels.obj(model_nlp, x)
        @test NLPModels.cons(model_exa, x) ≈ NLPModels.cons(model_nlp, x)
        @test NLPModels.grad(model_exa, x) ≈ NLPModels.grad(model_nlp, x)
        @test _jacobian_matrix(model_exa, x) ≈ _jacobian_matrix(model_nlp, x)
        @test _hessian_matrix(model_exa, x, y; obj_weight) ≈
              _hessian_matrix(model_nlp, x, y; obj_weight)
    end
    return
end

function test_nonlinear_constraint_derivative_sparsity()
    model = JuMP.Model()
    JuMP.@variable(model, p)
    JuMP.@variable(model, vmf)
    JuMP.@variable(model, vmt)
    JuMP.@variable(model, vaf)
    JuMP.@variable(model, vat)
    JuMP.@constraint(
        model,
        p - 1.2vmf^2 - 0.7vmf * vmt * cos(vaf - vat) - 0.3vmf * vmt * sin(vaf - vat) == 0.0,
    )
    JuMP.@objective(model, Min, p)
    model_exa = ExaModels.ExaModel(model)
    @test model_exa.meta.nnzj == 10
    @test model_exa.meta.nnzh == 21
    jacobian_rows = zeros(Int, model_exa.meta.nnzj)
    jacobian_cols = zeros(Int, model_exa.meta.nnzj)
    NLPModels.jac_structure!(model_exa, jacobian_rows, jacobian_cols)
    @test length(unique(zip(jacobian_rows, jacobian_cols))) == 5
    # Hessian coordinates are unique here because this model has one
    # constraint row. Different rows may legitimately repeat coordinates.
    hessian_rows = zeros(Int, model_exa.meta.nnzh)
    hessian_cols = zeros(Int, model_exa.meta.nnzh)
    NLPModels.hess_structure!(model_exa, hessian_rows, hessian_cols)
    @test length(unique(zip(hessian_rows, hessian_cols))) == 10
    _test_callback_equivalence(
        model,
        [[0.2, 1.0, 0.9, 0.1, -0.2], [-0.4, 1.1, 1.05, -0.3, 0.25]],
    )
    return
end

function test_nonlinear_constraint_repeated_variable()
    model = JuMP.Model()
    JuMP.@variable(model, x)
    JuMP.@variable(model, y)
    JuMP.@constraint(model, sin(x) + x^2 + cos(x - y) == 0.0)
    JuMP.@objective(model, Min, x + y)
    model_exa = ExaModels.ExaModel(model)
    @test model_exa.meta.nnzj == 4
    @test model_exa.meta.nnzh == 5
    _test_callback_equivalence(model, [[0.3, -0.7], [1.2, 0.4]])
    return
end

function test_nonlinear_objective_derivative_sparsity()
    model = JuMP.Model()
    JuMP.@variable(model, x)
    JuMP.@variable(model, y)
    JuMP.@objective(model, Min, sin(x) + x^2 + cos(x - y) + 2.5)
    model_exa = ExaModels.ExaModel(model)
    @test model_exa.meta.nnzo == 2
    @test model_exa.meta.nnzh == 5
    _test_callback_equivalence(model, [[0.3, -0.7], [1.2, 0.4]])
    return
end

function test_nonlinear_objective_nested_expr()
    model = JuMP.Model()
    JuMP.@variable(model, x)
    JuMP.@variable(model, y)
    JuMP.@objective(model, Min, exp(sin(x) + x^2 + cos(x - y)))
    model_exa = ExaModels.ExaModel(model)
    @test model_exa.meta.nnzo == 2
    @test model_exa.meta.nnzh == 3
    _test_callback_equivalence(model, [[0.3, -0.7], [1.2, 0.4]])
    return
end

function test_nonlinear_objective_parameter()
    model = JuMP.Model()
    JuMP.@variable(model, x[1:8])
    JuMP.@variable(model, p in JuMP.Parameter(0.4))
    JuMP.@objective(model, Min, sum(sin(p * x[i]) for i = 1:8))
    model_exa = ExaModels.ExaModel(model)
    @test length(model_exa.objs) == 1
    @test model_exa.meta.nnzo == 8
    @test model_exa.meta.nnzh == 8
    parameter_point = collect(range(-0.7, 0.7; length = 8))
    @test NLPModels.obj(model_exa, parameter_point) ≈ sum(sin.(0.4 .* parameter_point))
    @test NLPModels.grad(model_exa, parameter_point) ≈ 0.4 .* cos.(0.4 .* parameter_point)
    return
end

function test_nonlinear_objective_coupled()
    N = 20
    model = JuMP.Model()
    JuMP.@variable(model, x[1:N])
    JuMP.@objective(model, Min, sum(100(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N))
    model_exa = ExaModels.ExaModel(model)
    _test_callback_equivalence(model, [collect(range(-0.5, 0.5; length = N))])
    return
end

function test_nonlinear_aliasing_and_batching()
    model = JuMP.Model()
    JuMP.@variable(model, z[1:4])
    JuMP.@constraint(model, sin(z[1] * z[2]) == 0.0)
    JuMP.@constraint(model, sin(z[3] * z[3]) == 0.0)
    JuMP.@constraint(model, sin(z[3] * z[4]) == 0.0)
    JuMP.@objective(model, Min, sum(z))
    model_exa = ExaModels.ExaModel(model)
    @test model_exa.meta.nnzj == 5
    @test model_exa.meta.nnzh == 7
    p = [0.2, -0.4, 0.7, 1.1]
    @test NLPModels.cons(model_exa, p) ≈ [sin(p[1] * p[2]), sin(p[3]^2), sin(p[3] * p[4])]
    return
end

function test_nonlinear_batching()
    K = 4
    model = JuMP.Model()
    JuMP.@variable(model, p[1:K])
    JuMP.@variable(model, vmf[1:K])
    JuMP.@variable(model, vmt[1:K])
    JuMP.@variable(model, vaf[1:K])
    JuMP.@variable(model, vat[1:K])
    JuMP.@constraint(
        model,
        [i = 1:K],
        p[i] -
        1.2vmf[i]^2 -
        0.7vmf[i] * vmt[i] * cos(vaf[i] - vat[i]) -
        0.3vmf[i] * vmt[i] * sin(vaf[i] - vat[i]) == 0.0,
    )
    JuMP.@objective(model, Min, sum(p))
    model_exa = ExaModels.ExaModel(model)
    @test length(model_exa.cons) == 5
    @test model_exa.meta.nnzj == 40
    @test model_exa.meta.nnzh == 84
    batched_point = vcat(
        collect(0.1:0.1:0.4),
        fill(1.0, K),
        fill(0.9, K),
        collect(0.05:0.05:0.2),
        collect(-0.2:0.05:-0.05),
    )
    @test NLPModels.cons(model_exa, batched_point) ≈
            [
        batched_point[i] -
        1.2batched_point[K+i]^2 -
        0.7batched_point[K+i] *
        batched_point[2K+i] *
        cos(batched_point[3K+i] - batched_point[4K+i]) -
        0.3batched_point[K+i] *
        batched_point[2K+i] *
        sin(batched_point[3K+i] - batched_point[4K+i]) for i = 1:K
    ]
    return
end

function test_nonlinear_parameters_and_nested_expressions()
    model = JuMP.Model()
    JuMP.@variable(model, x)
    JuMP.@variable(model, p in JuMP.Parameter(0.4))
    JuMP.@constraint(model, sin(p * x) + cos(p * x) + p == 0.0)
    JuMP.@objective(model, Min, x)
    model_exa = ExaModels.ExaModel(model)
    @test model_exa.meta.nnzj == 2
    @test NLPModels.cons(model_exa, [0.7]) ≈ [sin(0.4 * 0.7) + cos(0.4 * 0.7) + 0.4]
    return
end

function test_nonlinear_prameter_only()
    model = JuMP.Model()
    JuMP.@variable(model, x)
    JuMP.@variable(model, p in JuMP.Parameter(0.4))
    JuMP.@constraint(model, sin(p) + p^2 == 0.0)
    JuMP.@objective(model, Min, x)
    model_exa = ExaModels.ExaModel(model)
    @test model_exa.meta.nnzj == 0
    @test NLPModels.cons(model_exa, [0.7]) ≈ [sin(0.4) + 0.4^2]
    return
end

function test_nonlinear_nested_affine()
    model = JuMP.Model()
    JuMP.@variable(model, x)
    JuMP.@variable(model, y)
    JuMP.@constraint(model, sin(2x + 3y) + x == 0.0)
    JuMP.@objective(model, Min, x + y)
    model_exa = ExaModels.ExaModel(model)
    @test model_exa.meta.nnzj == 3
    @test model_exa.meta.nnzh == 3
    _test_callback_equivalence(model, [[0.3, -0.7], [1.2, 0.4]])
    return
end

function test_nonlinear_nested_quadratic()
    model = JuMP.Model()
    JuMP.@variable(model, x)
    JuMP.@variable(model, y)
    JuMP.@constraint(model, sin(x^2 + x * y) + x == 0.0)
    JuMP.@objective(model, Min, x + y)
    model_exa = ExaModels.ExaModel(model)
    @test model_exa.meta.nnzj == 3
    @test model_exa.meta.nnzh == 3
    _test_callback_equivalence(model, [[0.3, -0.7], [1.2, 0.4]])
    return
end

function test_nonlinear_shapes()
    model = JuMP.Model()
    JuMP.@variable(model, x)
    JuMP.@variable(model, y)
    JuMP.@variable(model, z)
    JuMP.@constraint(model, sin(x) == 0.0)
    JuMP.@constraint(model, exp(y + z) - 1.0 == 0.0)
    JuMP.@constraint(model, cos(x * y) + z == 0.0)
    JuMP.@objective(model, Min, x + y + z)
    model_exa = ExaModels.ExaModel(model)
    @test length(model_exa.cons) == 6
    @test model_exa.meta.nnzj == 6
    shapes_point = [0.2, -0.4, 0.7]
    @test NLPModels.cons(model_exa, shapes_point) ≈
            [
        sin(shapes_point[1]),
        exp(shapes_point[2] + shapes_point[3]) - 1.0,
        cos(shapes_point[1] * shapes_point[2]) + shapes_point[3],
    ]
    return
end

function test_nonlinear_float32()
    model = JuMP.GenericModel{Float32}()
    JuMP.@variable(model, x)
    JuMP.@variable(model, y)
    JuMP.@constraint(model, sin(x) + x^2 + cos(x - y) == 0.0f0)
    JuMP.@objective(model, Min, x + y)
    model_exa = ExaModels.ExaModel(model)
    @test typeof(model_exa) <: ExaModels.ExaModel{Float32}
    @test model_exa.meta.nnzj == 4
    @test model_exa.meta.nnzh == 5
    float_point = Float32[0.3, -0.7]
    @test eltype(NLPModels.cons(model_exa, float_point)) == Float32
    difference = float_point[1] - float_point[2]
    @test NLPModels.cons(model_exa, float_point) ≈
          Float32[sin(float_point[1]) + float_point[1]^2 + cos(difference)]
    @test NLPModels.grad(model_exa, float_point) ≈ ones(Float32, 2)
    @test _jacobian_matrix(model_exa, float_point) ≈
          Float32[(cos(float_point[1]) + 2float_point[1] - sin(difference)) sin(difference)]
    expected_hessian = Float32[
        -sin(float_point[1])+2-cos(difference) cos(difference)
        cos(difference) -cos(difference)
    ]
    @test _hessian_matrix(
        model_exa,
        float_point,
        Float32[0.37];
        obj_weight = 0.61f0,
    ) ≈ 0.37f0 .* expected_hessian
    return
end

function test_nonlinear_constraint_sum()
    N = 20
    model = JuMP.Model()
    JuMP.@variable(model, x[1:N])
    JuMP.@constraint(model, 0 <= sum(100(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N) <= 1)
    model_exa = ExaModels.ExaModel(model)
    @test length(model_exa.cons) == 5
    return
end

function test_sum_objective_decomposition()
    model = JuMP.Model()
    JuMP.@variable(model, x[1:3])
    JuMP.@expression(model, a, sum(x))
    JuMP.@expression(model, b, sum(x.^2))
    JuMP.@expression(model, c, sum(exp.(x)))
    JuMP.@objective(model, Min, a + b + c)
    model_exa = ExaModels.ExaModel(model)
    # The five cons are the Constraint +
    # i[2] * x[i[1]], i[2] * x[i[1]]^2, and exp(i[1])
    @test length(model_exa.objs) == 3
    return
end

function test_sum_constraint_decomposition()
    model = JuMP.Model()
    JuMP.@variable(model, x[1:3])
    JuMP.@expression(model, a, sum(x))
    JuMP.@expression(model, b, sum(x.^2))
    JuMP.@expression(model, c, sum(exp.(x)))
    JuMP.@constraint(model, a + b + c == 0)
    model_exa = ExaModels.ExaModel(model)
    # The four cons are the Constraint +
    # i[2] * x[i[1]], i[2] * x[i[1]]^2, and exp(i[1])
    @test length(model_exa.cons) == 4
    return
end

function test_sum_constraint_decomposition_multiple_rhs_terms()
    model = JuMP.Model()
    JuMP.@variable(model, x[1:3])
    JuMP.@expression(model, a, sum(x))
    JuMP.@expression(model, b, sum(x.^2))
    JuMP.@expression(model, c, sum(exp.(x)))
    JuMP.@constraint(model, a == b + c - 2)
    model_exa = ExaModels.ExaModel(model)
    # TODO(odow): we'd like this one to be the same as
    # test_sum_constraint_decomposition, but it requires fixing how we handle
    # :(-(arg)) terms.
    @test_broken length(model_exa.cons) == 4
    return
end

end # module
