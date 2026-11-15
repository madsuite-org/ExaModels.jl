struct I2
    i::Tuple{Int, Int}
end

struct I1
    i::I2
end

function _exa_luksan_struct_model(backend, N; M = 1)

    c = ExaCore(backend = backend, concrete = Val(true))
    data = [I1(I2((i, j))) for i in 1:N, j in 1:M]
    @add_var(c, x, N, M; start = [luksan_vlcek_x0(i.i.i[1]) for i in data])
    @add_con(c, s, luksan_vlcek_con1(x, i.i.i[1], i.i.i[2]) for i in data[1:(end - 2), :])
    @add_con!(c, s, (i.i.i[1], i.i.i[2]) => luksan_vlcek_con2(x, i.i.i[1], i.i.i[2]) for i in data[1:(end - 2), :])
    @add_obj(c, luksan_vlcek_obj(x, i, j) for i in 2:N, j in 1:M)

    return ExaModel(c; prod = true), (x,), (s,)
end

_jump_luksan_struct_model(backend, N; M = 1) = _jump_luksan_vlcek_model(backend, N; M)
