# Run by PrecompileTest in a new process with `--trace-compile=stderr`, so that the
# compiler's trace lines land between the markers written here. After each region a
# `RESULT` line gives the compile time in ns and the device kernels compiled.
#
# evaluate.jl leaves out top-level `copy`, `fill!`, `length` and `view` on a
# `Vector{Float64}`. The system image has them inferred without native code, and a package
# image cannot add native code to a code instance that an image loaded before it owns, so
# they compile on first dynamic dispatch whatever a package stores.

import NLPModels

const WITH_KA = "ka" in ARGS
if WITH_KA
    using PrecompileAppKA
    const App = PrecompileAppKA
    const KernelAbstractions = App.KernelAbstractions
else
    using PrecompileApp
    const App = PrecompileApp
end
kernels() = WITH_KA ? KernelAbstractions.POCL.compilations[] : 0

# The timer stays at zero unless it is enabled.
Base.cumulative_compile_timing(true)
compile_ns() = sum(Base.cumulative_compile_time_ns())

report(line) = (println(stderr, line); flush(stderr))

# KernelAbstractions 0.10.6 and later launch POCL kernels through ThreadTeams, whose
# opaque closure (`new_job_closure`) cannot be verified when a package image loads, so
# every cached caller up to the fixture's model build is compiled again (see
# PrecompileTest.jl, `_test_stored_ka`). The line is built before it is written, so that
# no compiler trace line lands inside it.
report("THREADTEAMS " * string(any(m -> nameof(m) === :ThreadTeams, values(Base.loaded_modules))))

function begin_region(name)
    report("BEGIN $name")
    return compile_ns(), kernels()
end
function end_region(name, start)
    ns, k = compile_ns() - start[1], kernels() - start[2]
    report("END $name")
    return report("RESULT $name $ns $k")
end

# The first region compiles the helpers above. The second shows that from then on the
# probe itself compiles nothing.
end_region("warmup", begin_region("warmup"))
end_region("empty", begin_region("empty"))

const evaluate = joinpath(@__DIR__, "evaluate.jl")

start = begin_region("stored")
builder, backend = App.stored_model, nothing
Base.include(Main, evaluate)
end_region("stored", start)

start = begin_region("stored_buffered")
builder, backend = App.buffered_model, nothing
Base.include(Main, evaluate)
end_region("stored_buffered", start)

start = begin_region("other")
builder, backend = App.other_model, nothing
Base.include(Main, evaluate)
end_region("other", start)

if WITH_KA
    start = begin_region("stored_ka")
    builder, backend = App.stored_model, KernelAbstractions.CPU()
    Base.include(Main, evaluate)
    end_region("stored_ka", start)

    start = begin_region("other_ka")
    builder, backend = App.other_model, KernelAbstractions.CPU()
    Base.include(Main, evaluate)
    end_region("other_ka", start)
end
