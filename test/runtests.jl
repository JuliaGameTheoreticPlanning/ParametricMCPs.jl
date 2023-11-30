using ParametricMCPs
using Test: @testset, @test, @test_throws
using Random: Random
using LinearAlgebra: norm
using Zygote: Zygote
using FiniteDiff: FiniteDiff
using Enzyme: Enzyme

@testset "ParametricMCPs.jl" begin
    rng = Random.MersenneTwister(1)
    parameter_dimension = 2
    # setup a dummy MCP which represents a QP with:
    # - cost: sum((z[1:2] - θ).^2)
    # - constaints: z[1:2] >= 0
    f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]
    lower_bounds = [-Inf, -Inf, 0, 0]
    upper_bounds = [Inf, Inf, Inf, Inf]
    problem = ParametricMCPs.ParametricMCP(f, lower_bounds, upper_bounds, parameter_dimension)
    problem_no_jacobian = ParametricMCPs.ParametricMCP(
        f,
        lower_bounds,
        upper_bounds,
        parameter_dimension;
        compute_sensitivities=false,
    )

    feasible_parameters = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [rand(rng, 2) for _ in 1:10]...]
    infeasible_parameters = -feasible_parameters

    @testset "forward pass" begin
        for θ in feasible_parameters
            solution = ParametricMCPs.solve(problem, θ)
            @test solution.z[1:2] ≈ θ
        end

        for θ in feasible_parameters
            solution = ParametricMCPs.solve(problem, θ)
            @test norm(solution.z[1:2] - θ) <= norm(θ)
        end
    end

    function get_pipeline(problem)
        function dummy_pipeline(θ)
            solution = ParametricMCPs.solve(problem, θ)
            sum(solution.z .^ 2)
        end
    end

    @testset "backward pass" begin
        dummy_pipeline = get_pipeline(problem)
        for θ in [feasible_parameters; infeasible_parameters]
            ∇_zygote_reverse = only(Zygote.gradient(dummy_pipeline, θ))
            ∇_zygote_forward =
                only(Zygote.gradient(θ -> Zygote.forwarddiff(dummy_pipeline, θ), θ))
            #Enzyme.jacobian(Enzyme.Reverse, dummy_pipeline, Enzyme.Duplicated([1.0, 1.0], [0.0, 0.0]))
            ∇_enzyme_forward = Enzyme.jacobian(Enzyme.Forward, dummy_pipeline, θ) |> vec
            ∇_finitediff = FiniteDiff.finite_difference_gradient(dummy_pipeline, θ)
            @test isapprox(∇_zygote_reverse, ∇_finitediff; atol=1e-4)
            @test isapprox(∇_zygote_forward, ∇_finitediff; atol=1e-4)
            @test isapprox(∇_enzyme_forward, ∇_finitediff; atol=1e-4)
        end
    end

    @testset "missing jacobian" begin
        dummy_pipeline = get_pipeline(problem_no_jacobian)
        @test_throws ArgumentError Zygote.gradient(dummy_pipeline, feasible_parameters[1],)
    end
end
