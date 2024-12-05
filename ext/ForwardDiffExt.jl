module ForwardDiffExt

using ParametricMCPs: ParametricMCPs
using ForwardDiff: ForwardDiff

function ParametricMCPs.solve(
    problem::ParametricMCPs.ParametricMCP,
    θ::AbstractVector{<:ForwardDiff.Dual{T}};
    kwargs...,
) where {T}
    # strip off the duals:
    θ_v = ForwardDiff.value.(θ)
    θ_p = ForwardDiff.partials.(θ)
    # forward pass
    solution = ParametricMCPs.solve(problem, θ_v; kwargs...)
    # backward pass
    ∂z∂θ = ParametricMCPs.InternalAutoDiffUtils.solve_jacobian_θ(problem, solution, θ_v)
    # downstream gradient
    z_p = ∂z∂θ * θ_p
    # glue forward and backward pass together into dual number types
    z_d = ForwardDiff.Dual{T}.(solution.z, z_p)

    (; z = z_d, solution.status, solution.info)
end

end
