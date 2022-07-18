function _solve_jacobian_θ(problem, solution, θ; active_tolerance = 1e-3)
    (; jacobian_z!, jacobian_θ!, lower_bounds, upper_bounds) = problem.fields
    z_star = solution.z

    active_indices = let
        lower_active = z_star .>= (lower_bounds .+ active_tolerance)
        upper_active = z_star .<= (upper_bounds .- active_tolerance)
        findall(lower_active .& upper_active)
    end

    ∂z∂θ = SparseArrays.spzeros(get_problem_size(problem), get_parameter_dimension(problem))
    if isempty(active_indices)
        return ∂z∂θ
    end

    ∂f_reduce∂θ = let
        ∂f∂θ = get_result_buffer(jacobian_θ!)
        jacobian_θ!(∂f∂θ, z_star, θ)
        ∂f∂θ[active_indices, :]
    end

    ∂f_reduced∂z_reduced = let
        ∂f∂z = get_result_buffer(jacobian_z!)
        jacobian_z!(∂f∂z, z_star, θ)
        ∂f∂z[active_indices, active_indices]
    end

    ∂z∂θ[active_indices, :] = LinearAlgebra.qr(-∂f_reduced∂z_reduced) \ collect(∂f_reduce∂θ)
    (; ∂z∂θ, active_indices)
end

function ChainRulesCore.rrule(::typeof(solve), problem, θ; kwargs...)
    solution = solve(problem, θ; kwargs...)
    project_to_θ = ProjectTo(θ)

    function solve_pullback(∂solution)
        no_grad_args = (; ∂self = NoTangent(), ∂problem = NoTangent())

        ∂θ = ChainRulesCore.@thunk let
            ∂z∂θ = _solve_jacobian_θ(problem, solution, θ)
            ∂l∂z = solution.z
            project_to_θ(∂z∂θ' * ∂l∂z)
        end

        no_grad_args..., ∂θ
    end

    res, solve_pullback
end
