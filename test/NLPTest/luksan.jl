function luksan_vlcek_obj(x, i, j)
    return 100 * (x[i - 1, j]^2 - x[i, j])^2 + (x[i - 1, j] - 1)^2
end

function luksan_vlcek_con1(x, i, j)
    return 3x[i + 1, j]^3 + 2 * x[i + 2, j] - 5
end
function luksan_vlcek_con2(x, i, j)
    return sin(x[i + 1, j] - x[i + 2, j])sin(x[i + 1, j] + x[i + 2, j]) + 4x[i + 1, j] -
        x[i, j]exp(x[i, j] - x[i + 1, j]) - 3
end

function luksan_vlcek_x0(i)
    return mod(i, 2) == 1 ? -1.2 : 1.0
end

function _exa_luksan_vlcek_model(backend, N; M = 1)

    c = ExaCore(backend = backend, concrete = Val(true))
    @add_var(c, x, N, M; start = [luksan_vlcek_x0(i) for i in 1:N, j in 1:M])
    @add_con(c, s, luksan_vlcek_con1(x, i, j) for i in 1:(N - 2), j in 1:M)
    @add_con!(c, s, (i, j) => luksan_vlcek_con2(x, i, j) for i in 1:(N - 2), j in 1:M)
    @add_obj(c, luksan_vlcek_obj(x, i, j) for i in 2:N, j in 1:M)

    return ExaModel(c; prod = true), (x,), (s,)
end

function exa_luksan_vlcek_model(backend, N; M = 1)
    m, vars, cons = _exa_luksan_vlcek_model(backend, N; M = M)
    return m
end

function _jump_luksan_vlcek_model(backend, N; M = 1)
    jm = JuMP.Model()

    JuMP.@variable(jm, x[i = 1:N, j = 1:M], start = mod(i, 2) == 1 ? -1.2 : 1.0)
    JuMP.@NLconstraint(
        jm,
        s[i = 1:(N - 2), j = 1:M],
        3x[i + 1, j]^3 + 2x[i + 2, j] - 5 +
            sin(x[i + 1, j] - x[i + 2, j])sin(x[i + 1, j] + x[i + 2, j]) +
            4x[i + 1, j] - x[i, j]exp(x[i, j] - x[i + 1, j]) - 3 == 0.0
    )
    JuMP.@NLobjective(
        jm,
        Min,
        sum(100(x[i - 1, j]^2 - x[i, j])^2 + (x[i - 1, j] - 1)^2 for i in 2:N, j in 1:M)
    )

    return jm, (x,), (s,)
end

function jump_luksan_vlcek_model(backend, N; M = 1)
    jm, vars, cons = _jump_luksan_vlcek_model(backend, N; M = M)
    return MathOptNLPModel(jm)
end
