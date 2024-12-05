using ParametricMCPs
using Test: @testset, @test, @test_throws
using Random: Random
using LinearAlgebra: norm
using Zygote: Zygote
using FiniteDiff: FiniteDiff

@testset "ParametricMCPs.jl" begin
    for backend in [
        ParametricMCPs.SymbolicUtils.SymbolicsBackend(),
        ParametricMCPs.SymbolicUtils.FastDifferentiationBackend(),
    ]
        rng = Random.MersenneTwister(1)
        parameter_dimension = 2
        # setup a dummy MCP which represents a QP with:
        # - cost: sum((z[1:2] - θ).^2)
        # - constaints: z[1:2] >= 0
        f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]
        lower_bounds = [-Inf, -Inf, 0, 0]
        upper_bounds = [Inf, Inf, Inf, Inf]
        problem = ParametricMCPs.ParametricMCP(
            f,
            lower_bounds,
            upper_bounds,
            parameter_dimension;
            backend,
        )
        problem_no_jacobian = ParametricMCPs.ParametricMCP(
            f,
            lower_bounds,
            upper_bounds,
            parameter_dimension;
            compute_sensitivities = false,
            backend,
        )

        feasible_parameters = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [rand(rng, 2) for _ in 1:10]...]
        infeasible_parameters = -feasible_parameters

        @testset "forward pass" begin
            for θ in feasible_parameters
                solution = ParametricMCPs.solve(problem, θ)
                @test solution.z[1:2] ≈ θ
            end

            for θ in infeasible_parameters
                solution = ParametricMCPs.solve(problem, θ)
                @test norm(solution.z[1:2] - θ) <= norm(θ)
            end
        end

        @testset "backward pass" begin
            function dummy_pipeline(θ)
                solution = ParametricMCPs.solve(problem, θ)
                sum(solution.z .^ 2)
            end

            for θ in [feasible_parameters; infeasible_parameters]
                ∇_autodiff_reverse = only(Zygote.gradient(dummy_pipeline, θ))
                ∇_autodiff_forward =
                    only(Zygote.gradient(θ -> Zygote.forwarddiff(dummy_pipeline, θ), θ))
                ∇_finitediff = FiniteDiff.finite_difference_gradient(dummy_pipeline, θ)
                @test isapprox(∇_autodiff_reverse, ∇_finitediff; atol = 1e-4)
                @test isapprox(∇_autodiff_reverse, ∇_autodiff_forward; atol = 1e-4)
            end
        end

        @testset "missing jacobian" begin
            function dummy_pipeline(θ, problem)
                solution = ParametricMCPs.solve(problem, θ)
                sum(solution.z .^ 2)
            end

            @test_throws ArgumentError Zygote.gradient(
                θ -> dummy_pipeline(θ, problem_no_jacobian),
                feasible_parameters[1],
            )
        end
    end
end
