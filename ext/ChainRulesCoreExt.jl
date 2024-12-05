module ChainRulesCoreExt

using ParametricMCPs: ParametricMCPs
using ChainRulesCore: ChainRulesCore

function ChainRulesCore.rrule(::typeof(ParametricMCPs.solve), problem, θ; kwargs...)
    solution = ParametricMCPs.solve(problem, θ; kwargs...)
    project_to_θ = ChainRulesCore.ProjectTo(θ)

    function solve_pullback(∂solution)
        no_grad_args = (; ∂self = ChainRulesCore.NoTangent(), ∂problem = ChainRulesCore.NoTangent())

        ∂θ = ChainRulesCore.@thunk let
            ∂z∂θ = ParametricMCPs.InternalAutoDiffUtils.solve_jacobian_θ(problem, solution, θ)
            ∂l∂z = ∂solution.z
            project_to_θ(∂z∂θ' * ∂l∂z)
        end

        no_grad_args..., ∂θ
    end

    solution, solve_pullback
end

end
