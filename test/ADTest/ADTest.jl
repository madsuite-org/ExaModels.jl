module ADTest

using ExaModels
using Test, ForwardDiff, SpecialFunctions
import NLPModels

const FUNCTIONS = [
    ("basic-functions-:+", x -> +(x[1])),
    ("basic-functions-:-", x -> -(x[1])),
    ("basic-functions-inv", x -> inv(x[1])),
    ("basic-functions-abs", x -> abs(x[1])),
    ("basic-functions-sqrt", x -> sqrt(x[1])),
    ("basic-functions-cbrt", x -> cbrt(x[1])),
    ("basic-functions-abs2", x -> abs2(x[1])),
    ("basic-functions-exp", x -> exp(x[1])),
    ("basic-functions-exp2", x -> exp2(x[1])),
    ("basic-functions-exp10", x -> exp10(x[1])),
    ("basic-functions-log", x -> log(x[1])),
    ("basic-functions-log2", x -> log2(x[1])),
    ("basic-functions-log1p", x -> log1p(x[1])),
    ("basic-functions-log10", x -> log10(x[1])),
    ("basic-functions-sin", x -> sin(x[1])),
    ("basic-functions-cos", x -> cos(x[1])),
    ("basic-functions-tan", x -> tan(x[1])),
    ("basic-functions-asin", x -> asin(x[1])),
    ("basic-functions-acos", x -> acos(x[1])),
    ("basic-functions-csc", x -> csc(x[1])),
    ("basic-functions-sec", x -> sec(x[1])),
    ("basic-functions-cot", x -> cot(x[1])),
    ("basic-functions-atan", x -> atan(x[1])),
    ("basic-functions-acot", x -> acot(x[1])),
    # ("basic-functions-sind", x-> sind(x[1])), # cannot extend function 
    # ("basic-functions-cosd", x-> cosd(x[1])), # cannot extend function 
    # ("basic-functions-tand", x-> tand(x[1])), # cannot extend function 
    ("basic-functions-cscd", x -> cscd(x[1])),
    ("basic-functions-secd", x -> secd(x[1])),
    ("basic-functions-cotd", x -> cotd(x[1])),
    # ("basic-functions-atand", x-> atand(x[1])), # cannot extend function 
    # ("basic-functions-acotd", x-> acotd(x[1])), # cannot extend function 
    ("basic-functions-sinh", x -> sinh(x[1])),
    ("basic-functions-asinh", x -> asinh(x[1])),
    ("basic-functions-cosh", x -> cosh(x[1])),
    ("basic-functions-acosh", x -> acosh(x[1] + 1)),
    ("basic-functions-tanh", x -> tanh(x[1])),
    ("basic-functions-csch", x -> csch(x[1])),
    ("basic-functions-sech", x -> sech(x[1])),
    ("basic-functions-coth", x -> coth(x[1])),
    ("basic-functions-atanh", x -> atanh(x[1])),
    # ("basic-functions-acoth", x-> acoth(x[1])), # range issue
    ("basic-functions-:+", x -> +(x[1], x[2])),
    ("basic-functions-:-", x -> -(x[1], x[2])),
    ("basic-functions-:*", x -> *(x[1], x[2])),
    ("basic-functions-:^", x -> ^(x[1], x[2])),
    ("basic-functions-:/", x -> /(x[1], x[2])),
    # ("basic-functions-:<=", x-> <=(x[1], x[2])), # not implemented 
    # ("basic-functions-:>=", x-> >=(x[1], x[2])), # not implemented 
    # ("basic-functions-:(==),", x-> (==)(x[1], x[2])), # not implemented 
    # ("basic-functions-:<", x-> <(x[1], x[2])), # not implemented 
    # ("basic-functions-:>", x-> >(x[1], x[2])), # not implemented 
    ("special-functions-erfi", x -> erfi(x[1])),
    ("special-functions-erfcinv", x -> erfcinv(x[1])),
    ("special-functions-erfcx", x -> erfcx(x[1])),
    ("special-functions-invdigamma", x -> invdigamma(x[1])),
    ("special-functions-bessely1", x -> bessely1(x[1])),
    ("special-functions-besselj1", x -> besselj1(x[1])),
    ("special-functions-dawson", x -> dawson(x[1])),
    ("special-functions-airyaiprime", x -> airyaiprime(x[1])),
    ("special-functions-erf", x -> erf(x[1])),
    ("special-functions-trigamma", x -> trigamma(x[1])),
    ("special-functions-gamma", x -> gamma(x[1])),
    ("special-functions-airyaiprime", x -> airyaiprime(x[1])),
    ("special-functions-airybiprime", x -> airybiprime(x[1])),
    ("special-functions-erfinv", x -> erfinv(x[1])),
    ("special-functions-bessely0", x -> bessely0(x[1])),
    ("special-functions-erfc", x -> erfc(x[1])),
    ("special-functions-trigamma", x -> trigamma(x[1])),
    ("special-functions-airybiprime", x -> airybiprime(x[1])),
    ("special-functions-besselj0", x -> besselj0(x[1])),
    ("special-functions-beta", x -> beta(x[1], x[2])),
    ("special-functions-logbeta", x -> logbeta(x[1], x[2])),
    (
        "composite-functions-1-1",
        x -> beta(erf(x[1] / x[2] / 3.0) + 3.0 * x[2], erf(x[9])^2),
    ),
    ("composite-functions-1-2", x -> 0 * x[1]),
    (
        "composite-functions-1-3",
        x -> beta(cos(log(abs2(inv(inv(x[1]))) + 1.0)), erfc(tanh(0 * x[1]))),
    ),
    ("composite-functions-1-4", x -> (0 * x[1]^x[3]^1.0 + x[1]) / x[9] / x[10]),
    (
        "composite-functions-1-5",
        x -> exp(x[1] + 1.0)^x[2] * log(abs2(x[3]) + 3) / tanh(x[2]),
    ),
    ("composite-functions-1-6", x -> beta(2 * logbeta(x[1], x[5]), beta(x[2], x[3]))),
    ("composite-functions-1-7", x -> besselj0(exp(erf(-x[1])))),
    ("composite-functions-1-8", x -> erfc(abs2(x[1]^2 / x[2])^x[9] / x[10])),
    ("composite-functions-1-9", x -> erfc(x[1])^erf(2.5x[2])),
    ("composite-functions-1-10", x -> sin(1 / x[1])),
    ("composite-functions-1-11", x -> exp(x[2]) / cos(x[1])^2 + sin(x[1]^2)),
    ("composite-functions-1-12", x -> sin(x[9]inv(x[1]) - x[8]inv(x[2]))),
    ("composite-functions-1-13", x -> x[1] / log(x[2]^2 + 9.0)),
    (
        "composite-functions-1-14",
        x -> beta(beta(tan(beta(x[1], 1) + 2.0), cos(sin(x[2]))), x[3]),
    ),
]

const PARAMETER_FUNCTIONS = [
    ("parameter-basic-1", (x, θ) -> x[1] + θ[1]),
    ("parameter-basic-2", (x, θ) -> x[1] * θ[1]),
    ("parameter-basic-3", (x, θ) -> x[1]^2 + θ[1] * x[2]),
    ("parameter-basic-4", (x, θ) -> sin(x[1]) + cos(θ[1])),
    ("parameter-basic-5", (x, θ) -> exp(x[1] + θ[1])),
    ("parameter-basic-6", (x, θ) -> log(x[1]^2 + θ[1]^2)),
    ("parameter-basic-7", (x, θ) -> x[1] / (1 + θ[1])),
    ("parameter-basic-8", (x, θ) -> θ[1] * sin(x[1]) + θ[2] * cos(x[2])),
    ("parameter-composite-1", (x, θ) -> exp(x[1] * θ[1]) + sin(x[2] + θ[2])),
    ("parameter-composite-2", (x, θ) -> beta(x[1] + θ[1], x[2] + θ[2])),
    ("parameter-composite-3", (x, θ) -> sqrt(x[1]^2 + θ[1]^2) * log(x[2] + θ[2] + 1)),
    ("parameter-composite-4", (x, θ) -> gamma(x[1] + 1) * θ[1] + erf(x[2] * θ[2])),
]

function gradient(f, x)
    T = eltype(x)
    y = fill!(similar(x), zero(T))
    ExaModels.gradient!(y, (p, x, θ) -> f(x), x, nothing, nothing, one(T))
    return y
end

function gradient_with_params(f, x, θ)
    T = eltype(x)
    y = fill!(similar(x), zero(T))
    ExaModels.gradient!(y, (p, x, θ) -> f(x, θ), x, θ, nothing, one(T))
    return y
end

function sgradient(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y1, a1 = ExaModels.grpass(d, nothing, nothing, ExaModels.NaNSource{T}(), ((),()), T(NaN))

    comp = ExaModels.Compressor(y1)

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Tuple{Int,Int}, n)

    ExaModels.sgradient!(buffer_I, ff, nothing, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), comp, 0, T(NaN))
    ExaModels.sgradient!(buffer, ff, nothing, x, nothing, comp, 0, one(T))

    y = zeros(length(x))
    y[collect(i for (i, j) in buffer_I)] += buffer

    return y
end

function sgradient_with_params(f, x, θ)
    T = eltype(x)

    ff = f(ExaModels.VarSource(), ExaModels.ParameterSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y1, a1 = ExaModels.grpass(d, nothing, nothing, ExaModels.NaNSource{T}(), ((),()), T(NaN))

    comp = ExaModels.Compressor(y1)

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Tuple{Int,Int}, n)

    ExaModels.sgradient!(buffer_I, ff, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), θ, comp, 0, T(NaN))
    ExaModels.sgradient!(buffer, ff, nothing, x, θ, comp, 0, one(T))

    y = zeros(length(x))
    y[collect(i for (i, j) in buffer_I)] += buffer

    return y
end

function sjacobian(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y1, a1 = ExaModels.grpass(d, nothing, nothing, ExaModels.NaNSource{T}(), ((),()), T(NaN))

    comp = ExaModels.Compressor(y1)

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.sjacobian!(buffer_I, buffer_J, ff, nothing, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), comp, 0, 0, T(NaN))
    ExaModels.sjacobian!(buffer, nothing, ff, nothing, x, nothing, comp, 0, 0, one(T))

    y = zeros(length(x))
    y[buffer_J] += buffer

    return y
end

function sjacobian_with_params(f, x, θ)
    T = eltype(x)

    ff = f(ExaModels.VarSource(), ExaModels.ParameterSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y1, a1 = ExaModels.grpass(d, nothing, nothing, nothing, ((),()), T(NaN))

    comp = ExaModels.Compressor(y1)

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.sjacobian!(buffer_I, buffer_J, ff, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), θ, comp, 0, 0, T(NaN))
    ExaModels.sjacobian!(buffer, nothing, ff, nothing, x, θ, comp, 0, 0, one(T))

    y = zeros(length(x))
    y[buffer_J] += buffer

    return y
end

function shessian(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    t = ff(ExaModels.Identity(), ExaModels.SecondAdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y2, a2 = ExaModels.hrpass0(t, nothing, nothing, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), ((),()), T(NaN), T(NaN))

    comp = ExaModels.Compressor(y2)

    n = length(a2)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.shessian!(
        buffer_I,
        buffer_J,
        ff,
        nothing,
        ExaModels.NaNSource{T}(),
        ExaModels.NaNSource{T}(),
        comp,
        0,
        T(NaN),
        T(NaN)
    )
    ExaModels.shessian!(buffer, nothing, ff, nothing, x, nothing, comp, 0, one(T), zero(T))

    y = zeros(length(x), length(x))
    for (k, (i, j)) in enumerate(zip(buffer_I, buffer_J))
        if i == j
            y[i, j] += buffer[k]
        else
            y[i, j] += buffer[k]
            y[j, i] += buffer[k]
        end
    end
    return y
end

function shessian_with_params(f, x, θ)
    T = eltype(x)

    ff = f(ExaModels.VarSource(), ExaModels.ParameterSource())
    t = ff(ExaModels.Identity(), ExaModels.SecondAdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y2, a2 = ExaModels.hrpass0(t, nothing, nothing, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), ((),()), T(NaN), T(NaN))

    comp = ExaModels.Compressor(y2)

    n = length(a2)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.shessian!(buffer_I, buffer_J, ff, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), θ, comp, 0, T(NaN), T(NaN))
    ExaModels.shessian!(buffer, nothing, ff, nothing, x, θ, comp, 0, one(T), zero(T))

    y = zeros(length(x), length(x))
    for (k, (i, j)) in enumerate(zip(buffer_I, buffer_J))
        if i == j
            y[i, j] += buffer[k]
        else
            y[i, j] += buffer[k]
            y[j, i] += buffer[k]
        end
    end
    return y
end
# Verify derivative tables against finite differences.
# Step sizes: h1 for first derivatives (central diff truncation O(h^2)),
# h2 for second derivatives (optimal h ~ eps^(1/4) ≈ 1e-4 to balance truncation vs rounding).
function verify_derivative_tables()
    h1 = 1.0e-7
    h2 = 1.0e-4
    @testset "Univariate derivative tables" begin
        # Special test points for domain-sensitive functions
        special_x = Dict(
            :acosh => 1.7, :acoth => 1.7, :atanh => 0.3,
            # Degree-based trig: use degrees as test points
            :sind => 15.0, :cosd => 15.0, :tand => 15.0,
            :cscd => 15.0, :secd => 15.0, :cotd => 15.0,
            :atand => 15.0, :acotd => 15.0,
        )
        for (fname, df, ddf) in ExaModels._UNIVARIATES
            f = getfield(Base, fname)
            test_x = get(special_x, fname, 0.7)
            @testset "$fname" begin
                fd_df = (f(test_x + h1) - f(test_x - h1)) / (2h1)
                @test df(test_x) ≈ fd_df atol = 1.0e-6 rtol = 1.0e-5
                ddf_val = ddf(test_x)
                if !isnan(ddf_val)
                    fd_ddf = (f(test_x + h2) - 2 * f(test_x) + f(test_x - h2)) / h2^2
                    @test ddf_val ≈ fd_ddf atol = 1.0e-6 rtol = 1.0e-4
                end
            end
        end
    end
    @testset "bselect" begin
        @test ExaModels.bselect(0.0, Inf) == 0.0
        @test ExaModels.bselect(0.0, NaN) == 0.0
        @test ExaModels.bselect(1.0, 3.5) == 3.5
        # Blend a live branch with an inactive branch whose value is -Inf at
        # the evaluation point: the blend must stay finite and carry the live
        # branch's derivatives. The 0/1 step factor is built from sign, whose
        # registered derivative is zero.
        step(t) = (sign(sign(t) + 0.5) + 1) * 0.5   # 1 iff t >= 0 (ties exact)
        c = ExaModels.ExaCore(Float64)
        c, x = ExaModels.add_var(c, 2; start = [2.0, 0.0])
        c, _ = ExaModels.add_obj(
            c,
            ExaModels.bselect(step(-x[d.i+1]), x[d.i]^2) +
            ExaModels.bselect(1.0 - step(-x[d.i+1]), log(x[d.i+1])) for
            d in [(i = 1,)]
        )
        m = ExaModels.ExaModel(c)
        x0 = [2.0, 0.0]
        @test NLPModels.obj(m, x0) == 4.0
        g = NLPModels.grad(m, x0)
        @test g[1] == 4.0 && g[2] == 0.0
    end

    return @testset "Bivariate derivative tables" begin
        for (fname, df1, df2, ddf11, ddf12, ddf22) in ExaModels._BIVARIATES
            f = getfield(Base, fname)
            x1, x2 = 0.7, 1.3
            @testset "$fname" begin
                fd_df1 = (f(x1 + h1, x2) - f(x1 - h1, x2)) / (2h1)
                fd_df2 = (f(x1, x2 + h1) - f(x1, x2 - h1)) / (2h1)
                @test df1(x1, x2) ≈ fd_df1 rtol = 1.0e-5
                @test df2(x1, x2) ≈ fd_df2 rtol = 1.0e-5
                fd_ddf11 = (f(x1 + h2, x2) - 2 * f(x1, x2) + f(x1 - h2, x2)) / h2^2
                fd_ddf22 = (f(x1, x2 + h2) - 2 * f(x1, x2) + f(x1, x2 - h2)) / h2^2
                fd_ddf12 = (f(x1 + h2, x2 + h2) - f(x1 + h2, x2 - h2) - f(x1 - h2, x2 + h2) + f(x1 - h2, x2 - h2)) / (4 * h2^2)
                @test ddf11(x1, x2) ≈ fd_ddf11 atol = 1.0e-6 rtol = 1.0e-4
                @test ddf12(x1, x2) ≈ fd_ddf12 atol = 1.0e-6 rtol = 1.0e-4
                @test ddf22(x1, x2) ≈ fd_ddf22 atol = 1.0e-6 rtol = 1.0e-4
            end
        end
    end
end

function runtests()
    @testset "AD test" begin
        verify_derivative_tables()
        for (name, f) in FUNCTIONS
            x0 = rand(10)
            @testset "$name" begin
                g = ForwardDiff.gradient(f, x0)
                h = ForwardDiff.hessian(f, x0)
                @test gradient(f, x0) ≈ g atol = 1e-6
                @test sgradient(f, x0) ≈ g atol = 1e-6
                @test sjacobian(f, x0) ≈ g atol = 1e-6
                @test shessian(f, x0) ≈ h atol = 1e-6
            end
        end
        @testset "Parameter tests" begin
            for (name, f) in PARAMETER_FUNCTIONS
                x0 = rand(10)
                θ0 = rand(5)
                @testset "$name" begin
                    f_fixed_θ = x -> f(x, θ0)
                    g = ForwardDiff.gradient(f_fixed_θ, x0)
                    h = ForwardDiff.hessian(f_fixed_θ, x0)
                    @test gradient_with_params(f, x0, θ0) ≈ g atol = 1e-6
                    @test sgradient_with_params(f, x0, θ0) ≈ g atol = 1e-6
                    @test sjacobian_with_params(f, x0, θ0) ≈ g atol = 1e-6
                    @test shessian_with_params(f, x0, θ0) ≈ h atol = 1e-6
                end
            end
        end
    end
end

end #module
