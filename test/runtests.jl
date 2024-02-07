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
        compute_sensitivities = false,
    )

    feasible_parameters = [[0.0, 0.0], [rand(rng, 2) for _ in 1:4]...]
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

    function dummy_pipeline(problem, θ)
        solution = ParametricMCPs.solve(problem, θ)
        sum(solution.z .^ 2)
    end

    @testset "backward pass" begin
        for θ in [feasible_parameters; infeasible_parameters]
            ∇_finitediff = FiniteDiff.finite_difference_gradient(θ -> dummy_pipeline(problem, θ), θ)

            @testset "Zygote Reverse" begin
                ∇_zygote_reverse = Zygote.gradient(θ) do θ
                    dummy_pipeline(problem, θ)
                end |> only
                @test isapprox(∇_zygote_reverse, ∇_finitediff; atol = 1e-4)
            end

            @testset "Zygote Forward" begin
                ∇_zygote_forward = Zygote.gradient(θ) do θ
                    Zygote.forwarddiff(θ) do θ
                        dummy_pipeline(problem, θ)
                    end
                end |> only
                @test isapprox(∇_zygote_forward, ∇_finitediff; atol = 1e-4)
            end

            @testset "Enzyme Forward" begin
                ∇_enzyme_forward =
                    Enzyme.autodiff(
                        Enzyme.Forward,
                        dummy_pipeline,
                        problem,
                        Enzyme.BatchDuplicated(θ, Enzyme.onehot(θ)),
                    ) |>
                    only |>
                    collect
                @test isapprox(∇_enzyme_forward, ∇_finitediff; atol = 1e-4)
            end

            @testset "Enzyme Reverse" begin
                ∇_enzyme_reverse = zero(θ)
                Enzyme.autodiff(
                    Enzyme.Reverse,
                    dummy_pipeline,
                    problem,
                    Enzyme.Duplicated(θ, ∇_enzyme_reverse),
                )
                @test isapprox(∇_enzyme_reverse, ∇_finitediff; atol = 1e-4)
            end
        end
    end

    @testset "missing jacobian" begin
        @test_throws ArgumentError Zygote.gradient(feasible_parameters[1]) do θ
            dummy_pipeline(problem_no_jacobian, θ)
        end
    end
end
