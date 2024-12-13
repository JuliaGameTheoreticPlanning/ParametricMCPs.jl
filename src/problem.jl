struct ParametricMCP{T1<:Function,T2<:AbstractVector}
    "A callable of the form f(z, θ) with length(z) = number_of_decision_variables and length(θ) = number_of_parameters."
    f::T1
    "A vector of constant upport bounds."
    upper_bounds::T2
    "A vector of constant lower bounds."
    lower_bounds::T2
    "The number of parameters."
    number_of_parameters::Int
end

function ParametricMCP(f, upper_bounds, lower_bounds; number_of_parameters = 0)
    ParametricMCP(f, upper_bounds, lower_bounds, number_of_parameters)
end

const SymbolicNumber = Union{SymbolicTracingUtils.Symbolics.Num,SymbolicTracingUtils.FD.Node}

struct SymbolicParametricMCP{T<:AbstractVector{<:SymbolicNumber}}
    f::T
    z::T
    θ::T
    upper_bounds::T
    lower_bounds::T
end
