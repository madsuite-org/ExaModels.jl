# Shared by PrecompileApp and PrecompileAppKA. `other_model` has different expressions from
# `stored_model`, so its code is not in either image.

function stored_model(n; backend = nothing)
    c = ExaCore(; backend)
    c, p = add_par(c, ones(n))
    c, x = add_var(c, n; start = ones(n), lvar = -2.0, uvar = 2.0)
    c, s = add_expr(c, x[i] * p[i] for i in 1:n)
    c, _ = add_obj(c, (s[i] - 1)^2 + sin(x[i]) for i in 1:n)
    data = [(i = i, a = 0.5 * i) for i in 1:(n - 1)]
    c, g = add_con(c, x[d.i] * x[d.i + 1] + d.a * exp(x[d.i]) for d in data; ucon = 5.0)
    c, _ = add_con!(c, g, d.i => x[d.i + 1]^2 for d in data)
    return ExaModel(c)
end

# Buffered subexpressions go through their own Jacobian and Hessian assembly.
function buffered_model(n; backend = nothing)
    c = ExaCore(; backend)
    c, x = add_var(c, n; start = ones(n))
    c, b = add_expr(c, (sin(x[i]) * x[i] for i in 1:n); buffered = true)
    c, _ = add_obj(c, b[i]^2 for i in 1:n)
    c, _ = add_con(c, b[i] + x[i] for i in 1:n)
    return ExaModel(c)
end

function other_model(n; backend = nothing)
    c = ExaCore(; backend)
    c, x = add_var(c, n; start = 0.5)
    c, _ = add_obj(c, cos(x[i]) + x[i]^4 for i in 1:n)
    c, _ = add_con(c, x[i + 1] - tanh(x[i]) for i in 1:(n - 1))
    return ExaModel(c)
end

# The first `n` entries of a longer buffer, as MadNLP passes to `jac_coord!` and `hess_coord!`.
padded_view(x, n) = view(similar(x, n + 1), 1:n)

# Through `invokelatest`, as from the top level, so that the keyword entry point is
# compiled as a standalone method.
function workload(backend)
    m = Base.invokelatest(stored_model, 4; backend)
    Base.invokelatest(padded_view, m.meta.x0, 4)
    precompile_callbacks(m)
    return precompile_callbacks(Base.invokelatest(buffered_model, 4; backend))
end
