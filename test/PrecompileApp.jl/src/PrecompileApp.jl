# A package whose image holds the code of `stored_model`, for PrecompileTest.
module PrecompileApp

using ExaModels
import PrecompileTools

include("models.jl")

PrecompileTools.@compile_workload workload(nothing)

end
