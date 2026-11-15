module ExaModelsJuMP

import ExaModels
import JuMP

function ExaModels.ExaModel(jm::JuMP.GenericModel{T}; options...) where {T}
    if JuMP.nonlinear_model(jm) != nothing
        error("The legacy nonlinear model interface is not supported. Please use the new MOI-based interface.")
    end

    return ExaModels.ExaModel(JuMP.backend(jm); T = T, options...)
end

end # module ExaModelsJuMP
