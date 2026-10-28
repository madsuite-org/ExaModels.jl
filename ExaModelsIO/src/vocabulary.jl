# The operator vocabulary: the names a file uses for functions. SPEC.md lists
# the same names; a name added here must be added there, and none may be removed
# or change meaning.

# Functions of Base that ExaModels differentiates (src/functionlist.jl).
const BASE_OPERATORS = (
    # unary
    :inv, :sqrt, :cbrt, :abs, :abs2, :sign, :exp, :exp2, :exp10, :expm1,
    :log, :log2, :log1p, :log10, :sin, :cos, :tan, :asin, :acos, :atan, :acot,
    :csc, :sec, :cot, :sinh, :cosh, :tanh, :asinh, :acosh, :csch, :sech, :coth,
    :sind, :cosd, :tand, :cscd, :secd, :cotd, :atand, :acotd, :sinpi, :cospi,
    :sinc, :deg2rad, :rad2deg, :signbit, :floor, :ceil, :atanh, :acoth,
    # unary or binary
    :+, :-,
    # binary
    :*, :/, :^, :hypot, :max, :min, :mod,
)

# Functions of SpecialFunctions.jl that ExaModels differentiates
# (ext/functionlist.jl). Reading a file that uses them needs SpecialFunctions loaded.
const SPECIAL_OPERATORS = (
    :airyai, :airyaiprime, :airybi, :airybiprime, :besselj0, :besselj1,
    :bessely0, :bessely1, :beta, :dawson, :digamma, :erf, :erfc, :erfcinv,
    :erfcx, :erfi, :erfinv, :gamma, :invdigamma, :logbeta, :trigamma,
)

const SPECIALFUNCTIONS_UUID = Base.UUID("276daf66-3868-5448-9aa4-cd146d93841b")

# Functions outside the vocabulary (registered by the user with
# @register_univariate / @register_bivariate) are stored as "ext:<Module.name>"
# and resolved only through the `functions` argument of `read_model`.
const EXT_PREFIX = "ext:"

function _specialfunctions()
    for (id, m) in Base.loaded_modules
        id.uuid == SPECIALFUNCTIONS_UUID && return m
    end
    return nothing
end

# Vocabulary name of `f`, or an "ext:" name if `f` is not in the vocabulary.
function operator_name(f)
    for s in BASE_OPERATORS
        getfield(Base, s) === f && return string(s)
    end
    isdefined(ExaModels, :bselect) && f === ExaModels.bselect && return "bselect"
    sf = _specialfunctions()
    if sf !== nothing
        for s in SPECIAL_OPERATORS
            getfield(sf, s) === f && return string(s)
        end
    end
    return EXT_PREFIX * join((string.(fullname(parentmodule(f)))..., string(nameof(f))), ".")
end

# The function a vocabulary or "ext:" name stands for.
function operator_function(name::AbstractString, functions)
    if startswith(name, EXT_PREFIX)
        key = name[length(EXT_PREFIX)+1:end]
        haskey(functions, key) && return functions[key]
        throw(ArgumentError("the file uses the user function `$key`; register it with " *
            "ExaModels and pass it as `functions = Dict(\"$key\" => f)`"))
    end
    s = Symbol(name)
    s in BASE_OPERATORS && return getfield(Base, s)
    name == "bselect" && return ExaModels.bselect
    if s in SPECIAL_OPERATORS
        sf = _specialfunctions()
        sf === nothing && throw(ArgumentError(
            "the file uses `$name` from SpecialFunctions; load SpecialFunctions first"))
        return getfield(sf, s)
    end
    throw(ArgumentError("unknown operator `$name`"))
end
