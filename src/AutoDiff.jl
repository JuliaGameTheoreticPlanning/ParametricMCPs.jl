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

const EnzymeBatchedAnnotation = Union{EnzymeCore.BatchDuplicated,EnzymeCore.BatchDuplicatedNoNeed}
const EnzymeNoneedAnnotation = Union{EnzymeCore.DuplicatedNoNeed,EnzymeCore.BatchDuplicatedNoNeed}

function EnzymeRules.forward(
    func::EnzymeCore.Const{typeof(ParametricMCPs.solve)},
    ::Type{ReturnType},
    problem::EnzymeCore.Annotation{<:ParametricMCPs.ParametricMCP},
    θ::EnzymeCore.Annotation;
    kwargs...,
) where {ReturnType}
    # TODO: Enzyme sometimes passes us the problem as non-const (why?). For now, skip this check.
    if !(problem isa EnzymeCore.Const)
        throw(ArgumentError("""
                            `problem` must be annotated `Enzyme.Const`.
                            If you did not pass the non-const problem annotation yourself,
                            consider filing an issue with ParametricMCPs.jl.
                            """))
    end

    if θ isa EnzymeCore.Const
        throw(
            ArgumentError(
                """
                `θ` was annotated `Enzyme.Const` which defeats the purpose of running AD.
                If you did not pass the const θ annotation yourself,
                consider filing an issue with ParametricMCPs.jl.
                """,
            ),
        )
    end

    # forward pass
    solution_val = func.val(problem.val, θ.val; kwargs...)

    if ReturnType <: EnzymeCore.Const
        return solution_val
    end

    # backward pass
    ∂z∂θ = _solve_jacobian_θ(problem.val, solution_val, θ.val)

    if ReturnType <: EnzymeBatchedAnnotation
        solution_dval = map(θ.dval) do θdval
            _dval = deepcopy(solution_val)
            _dval.z .= ∂z∂θ * θdval
            _dval
        end
    else
        # downstream gradient
        dz = ∂z∂θ * θ.dval
        solution_dval = deepcopy(solution_val)
        solution_dval.z .= dz
    end

    if ReturnType <: EnzymeNoneedAnnotation
        return solution_dval
    end

    if ReturnType <: EnzymeCore.Duplicated
        return EnzymeCore.Duplicated(solution_val, solution_dval)
    end

    if ReturnType <: EnzymeCore.BatchDuplicated
        return EnzymeCore.BatchDuplicated(solution_val, solution_dval)
    end

    throw(ArgumentError("""
                        Forward rule for ReturnType with annotation $(ReturnType) not implemented.
                        Please file an issue with ParametricMCPs.jl.
                        """))
end

function EnzymeRules.augmented_primal(
    config,
    func::EnzymeCore.Const{typeof(ParametricMCPs.solve)},
    ::Type{ReturnType},
    problem,
    θ;
    kwargs...,
) where {ReturnType}
    if !(problem isa EnzymeCore.Const)
        throw(ArgumentError("""
                            `problem` must be annotated `Enzyme.Const`.
                            If you did not pass the non-const problem annotation yourself,
                            consider filing an issue with ParametricMCPs.jl.
                            """))
    end

    if ReturnType <: EnzymeBatchedAnnotation
        throw(ArgumentError("""
                            Return type `$(ReturnType)` currently not supported.
                            Please file an issue with ParametricMCPs.jl.
                            """))
    end

    needs_jacobian = !(ReturnType <: EnzymeCore.Const) && !(θ isa EnzymeCore.Const)

    # compute primal result if needed for forward or reverse pass
    if needs_jacobian || EnzymeRules.needs_primal(config)
        res = func.val(problem.val, θ.val; kwargs...)
    else
        res = nothing
    end

    # forward primal result if needed
    if EnzymeRules.needs_primal(config)
        primal = res
    else
        primal = nothing
    end

    # compute Jacobian if needed
    if needs_jacobian
        ∂z∂θ = _solve_jacobian_θ(problem.val, res, θ.val)
    else
        ∂z∂θ = nothing
    end

    # set up shadow if needed
    if EnzymeRules.needs_shadow(config)
        dres = deepcopy(res)
        dres.z .= 0.0
    else
        dres = nothing
    end

    tape = (; ∂z∂θ, dres)

    EnzymeRules.AugmentedReturn(res, dres, tape)
end

function EnzymeRules.reverse(
    config,
    func::EnzymeCore.Const{typeof(ParametricMCPs.solve)},
    ::Type{ReturnType},
    tape,
    problem,
    θ;
    kwargs...,
) where {ReturnType}
    if θ isa EnzymeCore.Duplicated && !(ReturnType <: EnzymeCore.Const)
        θ.dval .+= tape.∂z∂θ' * tape.dres.z
    else
        # all other cases should have been caught the checks in `augmented_primal`.
        @assert θ isa EnzymeCore.Const
    end

    (nothing, nothing)
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
