module PrecompileTest

using Test, ExaModels
import Pkg
import ExaModels: NLPModels

# Backend list comes from the suite driver when available; standalone runs
# cover the plain CPU path.
const PC_BACKENDS = isdefined(Main, :BACKENDS) ? Main.BACKENDS : [nothing]
const BACKEND_ONLY = isdefined(Main, :BACKEND_TESTS_ONLY) && Main.BACKEND_TESTS_ONLY

const TEST_DIR = abspath(joinpath(@__DIR__, ".."))
const PROBE = joinpath(@__DIR__, "probe.jl")

function _constrained(backend)
    c = ExaCore(; backend)
    c, x = add_var(c, 6; start = 0.5)
    c, _ = add_obj(c, (x[i] - 1)^2 for i in 1:6)
    c, _ = add_con(c, x[i] * x[i + 1] for i in 1:5)
    return ExaModel(c)
end

function _unconstrained(backend)
    c = ExaCore(; backend)
    c, x = add_var(c, 6; start = 0.5)
    c, _ = add_obj(c, sin(x[i]) + x[i]^2 for i in 1:6)
    return ExaModel(c)
end

function _oracle(backend, calls = zeros(Int, 3))
    c = ExaCore(; backend)
    c, x = add_var(c, 4; start = 0.5)
    c, _ = add_obj(c, x[i]^2 for i in 1:4)
    oracle = VectorNonlinearOracle(
        nvar = 4, ncon = 1, nnzj = 2, nnzh = 2,
        jac_rows = [1, 1], jac_cols = [3, 4],
        hess_rows = [3, 4], hess_cols = [3, 4],
        lcon = [-Inf], ucon = [Inf],
        adapt = Val(true),
        f! = (cv, xv) -> (calls[1] += 1; cv[1] = xv[3]^2 + xv[4]^2; nothing),
        jac! = (vv, xv) -> (calls[2] += 1; vv[1] = 2 * xv[3]; vv[2] = 2 * xv[4]; nothing),
        hess! = (hv, xv, yv) -> (calls[3] += 1; hv[1] = 2 * yv[1]; hv[2] = 2 * yv[1]; nothing),
    )
    c = constraint(c, oracle)
    return ExaModel(c)
end

function _test_callbacks(build, backend)
    m = build(backend)
    x0 = Array(m.meta.x0)
    @test precompile_callbacks(m) === nothing
    @test Array(m.meta.x0) == x0
    return nothing
end

# An oracle counts its calls, so this shows that the callbacks run: `cons!` once, and
# `jac_coord!` and `hess_coord!` with multipliers once per buffer form. `hess_coord!` without
# multipliers does not reach the oracle's Hessian.
function _test_oracle_calls(backend)
    calls = zeros(Int, 3)
    m = _oracle(backend, calls)
    precompile_callbacks(m)
    @test calls[1] >= 1
    @test calls[2] >= 2
    @test calls[3] >= 2
    return nothing
end

function _test_prod(backend)
    m = _constrained(backend)
    if backend === nothing
        @test precompile_callbacks(m; prod = true) === nothing
    else
        # A device model has no product helper unless built with `prod = true`.
        @test_throws ErrorException precompile_callbacks(m; prod = true)
    end
    return nothing
end

# A new environment holding the fixture and this ExaModels, for a process without the flags
# of the test process (`--check-bounds`, `--code-coverage`): they would disable package
# images.
# The path separator of `JULIA_LOAD_PATH` differs on Windows.
const LOAD_PATH_ENV = join(("@", "@stdlib"), Sys.iswindows() ? ';' : ':')

function _environment(app)
    env = mktempdir()
    julia = first(Base.julia_cmd().exec)
    setup = """
    import Pkg
    Pkg.develop([
        Pkg.PackageSpec(path = $(repr(pkgdir(ExaModels)))),
        Pkg.PackageSpec(path = $(repr(joinpath(TEST_DIR, app * ".jl")))),
    ])
    Pkg.add("NLPModels")
    using $app
    """
    cmd = addenv(
        `$julia --startup-file=no --project=$env -e $setup`,
        "JULIA_LOAD_PATH" => LOAD_PATH_ENV,
    )
    run(cmd)
    return julia, env
end

# Run probe.jl; returns name => (trace lines, compile ns, kernels) for each region.
function _probe(julia, env, args...)
    out = IOBuffer()
    cmd = addenv(
        `$julia --startup-file=no --project=$env --trace-compile=stderr $PROBE $args`,
        "JULIA_LOAD_PATH" => LOAD_PATH_ENV,
    )
    run(pipeline(cmd; stdout = devnull, stderr = out))
    regions = Dict{String, NamedTuple}()
    trace = String[]
    active = false
    for line in eachline(seekstart(out))
        if startswith(line, "BEGIN ")
            empty!(trace)
            active = true
        elseif startswith(line, "END ")
            active = false
        elseif startswith(line, "RESULT ")
            _, name, ns, k = split(line)
            regions[name] = (trace = copy(trace), ns = parse(UInt64, ns), kernels = parse(Int, k))
        elseif active
            push!(trace, line)
        end
    end
    return regions
end

function _test_stored(r)
    @test r["empty"].trace == String[]
    @test r["empty"].ns == 0
    if VERSION >= v"1.11"
        @test r["stored"].trace == String[]
        @test r["stored"].ns == 0
    else
        # The `Constraint` constructor, which `add_con` calls by runtime dispatch, is not in
        # the image on Julia 1.10.
        @test_broken r["stored"].trace == String[]
        @test_broken r["stored"].ns == 0
        @test any(contains("ExaModels.Constraint{"), r["stored"].trace)
    end
    # The timer and the trace both work, so the two above are not vacuous.
    @test r["other"].ns > 0
    @test !isempty(r["other"].trace)
    return nothing
end

# pocl's kernels do not persist: GPUCompiler bakes session addresses into them, so each
# session generates them again, and the host code of that compiler runs in code instances
# owned by images loaded before the fixture. Nothing of the fixture's models may compile on
# the host.
function _test_stored_ka(r)
    # GPUCompiler keeps its cache in the session on Julia 1.10, so nothing is stored there.
    if VERSION >= v"1.11"
        @test !any(contains("ExaModels"), r["stored_ka"].trace)
        @test !any(contains("PrecompileAppKA"), r["stored_ka"].trace)
    end
    @test_broken r["stored_ka"].trace == String[]
    @test_broken r["stored_ka"].ns == 0
    @test_broken r["stored_ka"].kernels == 0
    @test r["other_ka"].ns > 0
    @test r["other_ka"].kernels > 0
    return nothing
end

function runtests()
    @testset "precompile_callbacks" begin
        for backend in unique(Any[nothing; PC_BACKENDS])
            @testset "backend = $backend" begin
                _test_callbacks(_constrained, backend)
                _test_callbacks(_unconstrained, backend)
                _test_callbacks(_oracle, backend)
                _test_oracle_calls(backend)
                _test_prod(backend)
                if nameof(typeof(backend)) in (:POCLBackend, :CUDABackend, :ROCBackend)
                    @test precompile_backend(backend) === nothing
                end
            end
        end
    end

    @testset "ExaModel(c; precompile = true)" begin
        # A distinct expression for each model, so that neither reuses the other's code.
        build(f; kw...) = begin
            c = ExaCore()
            c, x = add_var(c, 3; start = 0.5)
            c, _ = add_obj(c, f(x[i]) for i in 1:3)
            c, _ = add_con(c, f(x[i]) * x[i + 1] for i in 1:2)
            ExaModel(c; kw...)
        end
        # `@nospecialize`: specialized on `m`, this function would compile the callbacks
        # itself on entry, before the timer starts.
        first_call_ns(@nospecialize(m)) = begin
            x = m.meta.x0
            g, c = similar(x), similar(x, m.meta.ncon)
            jv, hv = similar(x, m.meta.nnzj), similar(x, m.meta.nnzh)
            y = fill!(similar(x, m.meta.ncon), 1.0)
            Base.cumulative_compile_timing(true)
            t0 = sum(Base.cumulative_compile_time_ns())
            NLPModels.obj(m, x)
            NLPModels.grad!(m, x, g)
            NLPModels.cons!(m, x, c)
            NLPModels.jac_coord!(m, x, jv)
            NLPModels.hess_coord!(m, x, y, hv)
            t1 = sum(Base.cumulative_compile_time_ns())
            Base.cumulative_compile_timing(false)
            t1 - t0
        end
        @test first_call_ns(build(v -> sinh(v) + 3v; precompile = true)) == 0
        @test first_call_ns(build(v -> cosh(v) + 5v)) > 0
        # Nothing is evaluated on the CPU: `log` throws at this starting point.
        @test build(v -> log(v - 1); precompile = true) isa ExaModels.ExaModel
    end

    return @testset "no compilation for a stored model" begin
        if !BACKEND_ONLY
            julia, env = _environment("PrecompileApp")
            _test_stored(_probe(julia, env))
        end
        if any(b -> nameof(typeof(b)) === :POCLBackend, PC_BACKENDS)
            julia, env = _environment("PrecompileAppKA")
            r = _probe(julia, env, "ka")
            _test_stored(r)
            _test_stored_ka(r)
        end
    end
end

end
