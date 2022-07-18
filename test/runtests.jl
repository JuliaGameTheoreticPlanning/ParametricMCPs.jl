using DifferentiableMCPs
using Test: @testset, @test
using Random: Random
using LinearAlgebra: norm
using Zygote: Zygote
using FiniteDiff: FiniteDiff

using Infiltrator

@testset "DifferentiableMCPs.jl" begin
    rng = Random.MersenneTwister(1)
    parameter_dimension = 2
    # setup a dummy mcp:
    #
    # This is a simple QP with
    # - cost: sum((z[1:2] - θ).^2)
    # - constaints: z[1:2] >= 0
    f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]
    lower_bounds = [-Inf, -Inf, 0, 0]
    upper_bounds = [Inf, Inf, Inf, Inf]
    problem = DifferentiableMCPs.ParametricMCP(f, lower_bounds, upper_bounds, parameter_dimension)

    feasible_parameters = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [rand(rng, 2) for _ in 1:10]...]
    infeasible_parameters = -feasible_parameters

    @testset "forward pass" begin
        for θ in feasible_parameters
            solution = DifferentiableMCPs.solve(problem, θ)
            @test solution.z[1:2] ≈ θ
        end

        for θ in feasible_parameters
            solution = DifferentiableMCPs.solve(problem, θ)
            @test norm(solution.z[1:2] - θ) <= norm(θ)
        end
    end

    @testset "backward pass" begin
        function dummy_pipeline(θ)
            solution = DifferentiableMCPs.solve(problem, θ)
            sum(solution.z .^ 2)
        end

        for θ in [feasible_parameters; infeasible_parameters]
            ∇_autodiff = only(Zygote.gradient(dummy_pipeline, θ))
            ∇_finitediff = FiniteDiff.finite_difference_gradient(dummy_pipeline, θ)
            @test isapprox(∇_autodiff, ∇_finitediff; atol = 1e-4)
        end
    end
end
