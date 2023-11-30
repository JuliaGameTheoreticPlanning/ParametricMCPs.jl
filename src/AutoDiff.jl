module AutoDiff

using ..ParametricMCPs: ParametricMCPs, get_problem_size, get_result_buffer, get_parameter_dimension
using ChainRulesCore: ChainRulesCore
using EnzymeCore: EnzymeCore, EnzymeRules
using ForwardDiff: ForwardDiff
using SparseArrays: SparseArrays
using LinearAlgebra: LinearAlgebra

function _solve_jacobian_θ(problem, solution, θ; active_tolerance = 1e-3)
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

function EnzymeRules.forward(
    func::EnzymeCore.Const{typeof(ParametricMCPs.solve)},
    ::Type{ReturnType},
    problem::EnzymeCore.Annotation{<:ParametricMCPs.ParametricMCP},
    θ::EnzymeCore.Annotation;
    kwargs...,
) where {ReturnType}
    # forward pass
    solution_val = func.val(problem.val, θ.val; kwargs...)

    if ReturnType isa EnzymeCore.Const
        return solution_val
    end

    # backward pass
    ∂z∂θ_thunk = _solve_jacobian_θ(problem.val, solution_val, θ.val)

    if ReturnType <: EnzymeCore.BatchDuplicated
        solution_dval = map(θ.dval) do θdval
            (; solution_val..., z = ∂z∂θ_thunk * θdval)
        end
        return EnzymeCore.BatchDuplicated(solution_val, solution_dval)
    end

    if ReturnType <: EnzymeCore.BatchDuplicatedNoNeed
        error("Not implemented. Please file an issue with ParametricMCPs.jl")
    end

    # downstream gradient
    dz = ∂z∂θ_thunk * θ.dval

    solution_dval = (; solution_val..., z = dz)

    if ReturnType <: EnzymeCore.DuplicatedNoNeed
        solution_dval = (; solution_val..., z = dz)
    end

    @assert ReturnType <: EnzymeCore.Duplicated

    EnzymeCore.Duplicated(solution_val, solution_dval)
end

function ChainRulesCore.rrule(::typeof(ParametricMCPs.solve), problem, θ; kwargs...)
    solution = ParametricMCPs.solve(problem, θ; kwargs...)
    project_to_θ = ChainRulesCore.ProjectTo(θ)

    function solve_pullback(∂solution)
        no_grad_args = (; ∂self = ChainRulesCore.NoTangent(), ∂problem = ChainRulesCore.NoTangent())

        ∂θ = ChainRulesCore.@thunk let
            ∂z∂θ = _solve_jacobian_θ(problem, solution, θ)
            ∂l∂z = ∂solution.z
            project_to_θ(∂z∂θ' * ∂l∂z)
        end

        no_grad_args..., ∂θ
    end

    solution, solve_pullback
end

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
    ∂z∂θ = _solve_jacobian_θ(problem, solution, θ_v)
    # downstream gradient
    z_p = ∂z∂θ * θ_p
    # glue forward and backward pass together into dual number types
    z_d = ForwardDiff.Dual{T}.(solution.z, z_p)

    (; z = z_d, solution.status, solution.info)
end

end
