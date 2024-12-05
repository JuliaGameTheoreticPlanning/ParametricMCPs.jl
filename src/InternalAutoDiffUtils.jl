module InternalAutoDiffUtils

using ..ParametricMCPs: get_problem_size, get_result_buffer, get_parameter_dimension
using SparseArrays: SparseArrays
using LinearAlgebra: LinearAlgebra

function solve_jacobian_θ(problem, solution, θ; active_tolerance = 1e-3)
    (; jacobian_z!, jacobian_θ!, lower_bounds, upper_bounds) = problem
    z_star = solution.z

    !isnothing(jacobian_θ!) || throw(
        ArgumentError(
            "Missing sensitivities. Set `compute_sensitivities = true` when constructing the ParametricMCP.",
        ),
    )

    inactive_indices = let
        lower_inactive = z_star .>= (lower_bounds .+ active_tolerance)
        upper_inactive = z_star .<= (upper_bounds .- active_tolerance)
        findall(lower_inactive .& upper_inactive)
    end

    ∂z∂θ = SparseArrays.spzeros(get_problem_size(problem), get_parameter_dimension(problem))
    if isempty(inactive_indices)
        return ∂z∂θ
    end

    ∂f_reduce∂θ = let
        ∂f∂θ = get_result_buffer(jacobian_θ!)
        jacobian_θ!(∂f∂θ, z_star, θ)
        ∂f∂θ[inactive_indices, :]
    end

    ∂f_reduced∂z_reduced = let
        ∂f∂z = get_result_buffer(jacobian_z!)
        jacobian_z!(∂f∂z, z_star, θ)
        ∂f∂z[inactive_indices, inactive_indices]
    end

    ∂z∂θ[inactive_indices, :] =
        LinearAlgebra.qr(-collect(∂f_reduced∂z_reduced), LinearAlgebra.ColumnNorm()) \
        collect(∂f_reduce∂θ)
    ∂z∂θ
end

end
