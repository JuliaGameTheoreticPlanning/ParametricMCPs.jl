function solve_mcp(
    problem::ParametricMCP,
    θ;
    initial_guess = zeros(get_problem_size(problem)),
    verbose = false,
)
    (; f!, f_jacobian_z!, lower_bounds, upper_bounds) = problem.fields

    problem_size = get_problem_size(problem)

    function F(n, z, f)
        f!(f, z, θ)
        Cint(0)
    end

    function J(n, nnz, z, col, len, row, data)
        result = SparseArrays.sparse(f_jacobian_z!.rows, f_jacobian_z!.cols, zero(data))
        f_jacobian_z!(result, z, θ)
        _coo_from_sparse!(col, len, row, data, result)
        Cint(0)
    end

    silent = !verbose
    status, solution, info = PATHSolver.solve_mcp(
        F,
        J,
        lower_bounds,
        upper_bounds,
        initial_guess;
        silent = !verbose,
        # TODO: not sure why nnz above is too large
        nnz = SparseArrays.nnz(f_jacobian_z!) - 1,
    )

    (; solution, status, info)
end
