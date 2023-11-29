function solve(
    problem::ParametricMCP,
    θ;
    approximate_linear = false,
    initial_guess = zeros(get_problem_size(problem)),
    verbose = false,
    warn_on_convergence_failure = true,
    enable_presolve = false,
    jacobian_data_contiguous = true,
    options...,
)
    length(initial_guess) == get_problem_size(problem) ||
        throw(ArgumentError("initial guess must have length $(get_problem_size(problem))"))
    length(θ) == get_parameter_dimension(problem) ||
        throw(ArgumentError("θ must have length $(get_parameter_dimension(problem)), got $(θ)"))

    (; f!, jacobian_z!, lower_bounds, upper_bounds) = problem

    # println("start solve")

    lb = lower_bounds(θ)
    ub = upper_bounds(θ)

    # TODO: should be done with dispatch or separate function to avoid runtime overhead
    if approximate_linear
        M = jacobian_z!.result_buffer
        jacobian_z!(M, initial_guess, θ)

        q = zeros(length(initial_guess))
        f!(q, initial_guess, θ)

        F = function (n, z, f)
            # println("f-eval")
            f .= M * z .+ q
            Cint(0)
        end

        J = function (n, nnz, z, col, len, row, data)
            # println("j-eval")
            _coo_from_sparse!(col, len, row, data, M)
            Cint(0)
        end
        jacobian_linear_elements = collect(1:SparseArrays.nnz(M))
        jacobian_data_contiguous = true

        # correct bounds according to linearization point
        lb -= initial_guess
        ub -= initial_guess
    else
        F = function (n, z, f)
            f!(f, z, θ)
            Cint(0)
        end

        J = function (n, nnz, z, col, len, row, data)
            M = jacobian_z!.result_buffer
            jacobian_z!(M, z, θ)
            _coo_from_sparse!(col, len, row, data, M)
            Cint(0)
        end
        jacobian_linear_elements =
            enable_presolve ? jacobian_z!.constant_entries : empty(jacobian_z!.constant_entries)
    end

    status, z, info = PATHSolver.solve_mcp(
        F,
        J,
        lb,
        ub,
        approximate_linear ? zero(initial_guess) : initial_guess;
        silent = !verbose,
        nnz = SparseArrays.nnz(jacobian_z!),
        jacobian_structure_constant = true,
        jacobian_data_contiguous = jacobian_data_contiguous,
        jacobian_linear_elements = jacobian_linear_elements,
        options...,
    )

    if warn_on_convergence_failure && status != PATHSolver.MCP_Solved
        @warn "MCP not converged: PATH solver status is $(status)."
    end

    if approximate_linear
        z += initial_guess
    end

    (; z, status, info)
end
