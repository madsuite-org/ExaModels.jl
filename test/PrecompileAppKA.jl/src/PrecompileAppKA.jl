# PrecompileApp for a process with KernelAbstractions: this image is built with it loaded, as
# loading it after an image was built invalidates code in that image. The image holds
# `stored_model` on the CPU and on the KernelAbstractions CPU backend.
module PrecompileAppKA

using ExaModels
import KernelAbstractions
import PrecompileTools

include(joinpath(@__DIR__, "..", "..", "PrecompileApp.jl", "src", "models.jl"))

PrecompileTools.@compile_workload begin
    workload(nothing)
    workload(KernelAbstractions.CPU())
end

end
