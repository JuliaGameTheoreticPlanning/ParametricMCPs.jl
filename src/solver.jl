function solve(
    problem::ParametricMCP,
    θ;
    initial_guess = zeros(get_problem_size(problem)),
    verbose = false,
    solver_options = (;),
)
    (; f!, jacobian_z!, lower_bounds, upper_bounds) = problem

    problem_size = get_problem_size(problem)

    function F(n, z, f)
        f!(f, z, θ)
        Cint(0)
    end

    function J(n, nnz, z, col, len, row, data)
        result = get_result_buffer(jacobian_z!)
        jacobian_z!(result, z, θ)
        _coo_from_sparse!(col, len, row, data, result)
        Cint(0)
    end

    silent = !verbose
    status, z, info = PATHSolver.solve_mcp(
        F,
        J,
        lower_bounds,
        upper_bounds,
        initial_guess;
        silent = !verbose,
        # TODO: not sure why nnz above is too large
        nnz = SparseArrays.nnz(jacobian_z!) - 1,
        solver_options...,
    )

    if status != PATHSolver.MCP_Solved
        @warn "MCP not converged: PATH solver status is $(status)."
    end

    (; z, status, info)
end
