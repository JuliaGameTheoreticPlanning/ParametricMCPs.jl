function _solve_jacobian_θ(problem, solution, θ; active_tolerance = 1e-3)
    (; f_jacobian_z!, f_jacobian_θ!, lower_bounds, upper_bounds) = problem.fields
    z_star = solution.z

    active_indices = let
        # if zᵢ is unbouded on both sides, Fᵢ(z, θ) = 0 is always active
        unbounded_indices = findall(isinf.(lower_bounds) .& isinf.(upper_bounds))
        # for the remainig indices, we have to check the solution
        bounded_indices = setdiff(1:get_problem_size(problem), unbounded_indices)

        @assert all(>=(2 * active_tolerance), upper_bounds - lower_bounds)
        strongly_active_indices = filter(bounded_indices) do ii
            z_star[ii] >= lower_bounds[ii] + active_tolerance &&
                z_star[ii] <= upper_bounds[ii] - active_tolerance
        end

        [unbounded_indices; strongly_active_indices]
    end

    ∂z∂θ = SparseArrays.spzeros(get_problem_size(problem), get_parameter_dimension(problem))
    if isempty(active_indices)
        return ∂z∂θ
    end

    ∂f_reduce∂θ = let
        ∂f∂θ = get_result_buffer(f_jacobian_θ!)
        f_jacobian_θ!(∂f∂θ, z_star, θ)
        ∂f∂θ[active_indices, :]
    end

    ∂f_reduced∂z_reduced = let
        ∂f∂z = get_result_buffer(f_jacobian_z!)
        f_jacobian_z!(∂f∂z, z_star, θ)
        ∂f∂z[active_indices, active_indices]
    end

    ∂z∂θ[active_indices, active_indices] = LinearAlgebra.qr(-∂f_reduced∂z_reduced) \ ∂f_reduce∂θ
    ∂z∂θ
end
