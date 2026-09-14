import Pkg

include("backends.jl")

using Test, ExaModels
using Random
Random.seed!(0)

ad_tolerance(m1,m2) = max(ad_tolerance(m1), ad_tolerance(m2))
sol_tolerance(m1,m2) = max(sol_tolerance(m1), sol_tolerance(m2))
ad_tolerance(::Type{Float64}) = 1e-8
sol_tolerance(::Type{Float64}) = 1e-4
solver_tolerance(::Type{Float64}) = 1e-8
ad_tolerance(::Type{Float32}) = 1e-4
sol_tolerance(::Type{Float32}) = 1e-1
solver_tolerance(::Type{Float32}) = 1e-4

include("ArgumentTest/ArgumentTest.jl")
include("NLPTest/NLPTest.jl")
include("ADTest/ADTest.jl")
include("JuMPTest/JuMPTest.jl")
include("UtilsTest/UtilsTest.jl")
include("JuliaCTest/JuliaCTest.jl")
include("TwoStageTest/TwoStageTest.jl")
include("GetterSetterTest/GetterSetterTest.jl")
include("PrettyPrintTest.jl")
include("ConcreteModeTest.jl")
# include("OptimalControlTest/OptimalControlTest.jl")
include("OracleTest/OracleTest.jl")
include("SIMDModeTest/SIMDModeTest.jl")

@testset verbose = true "ExaModels test" begin
    @info "Running SIMDMode Test"
    SIMDModeTest.runtests()

    @info "Running Argument Test"
    ArgumentTest.runtests()

    @info "Running AD Test"
    ADTest.runtests()

    @info "Running NLP Test"
    NLPTest.runtests()

    @info "Running JuMP Test"
    JuMPTest.runtests()

    @info "Running Utils Test"
    UtilsTest.runtests()

    @info "Running JuliaC AOT Test"
    JuliaCTest.runtests()

    @info "Running TwoStage Test"
    TwoStageTest.runtests()

    @info "Running Getter/Setter Test"
    GetterSetterTest.runtests()

    @info "Running PrettyPrint Test"
    PrettyPrintTest.runtests()

    @info "Running Concrete Mode Test"
    ConcreteModeTest.runtests()

    # @info "Running OptimalControl Test"
    # OptimalControlTest.runtests()

    @info "Running Oracle Test"
    OracleTest.runtests()
end

# Force full GC before Julia exits so that OpenCL/PoCL objects are finalized
# in the correct order, preventing segfaults during Julia's atexit teardown.
GC.gc(true)
GC.gc(true)
