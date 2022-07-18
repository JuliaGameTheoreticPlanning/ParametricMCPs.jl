struct ParametricMCP{T}
    """
    (;
        f!(result, z, θ),
        f_jacobian_z!(result, z, θ),
        f_jacobian_θ!(result, z, θ),
        lower_bounds,
        upper_bounds,
    )
    """
    fields::T
end

get_problem_size(problem::ParametricMCP) = length(problem.fields.lower_bounds)

function ParametricMCP(f, lower_bounds, upper_bounds, parameter_dimension)
    length(lower_bounds) == length(upper_bounds) ||
        throw(ArgumentError("lower_bounds and upper_bounds have inconsistent lenghts."))
    problem_size = length(lower_bounds)

    # setup the problem symblically
    z_symbolic, θ_symbolic =
        Symbolics.@variables(z_symbolic[1:problem_size], θ_symbolic[1:parameter_dimension]) .|>
        Symbolics.scalarize

    f_symbolic = f(z_symbolic, θ_symbolic)
    length(f_symbolic) == problem_size || throw(
        ArgumentError(
            "The output lenght of `f` is inconsistent with `lower_bounds` and `upper_bounds`.",
        ),
    )
    f_jacobian_z_symbolic = Symbolics.sparsejacobian(f_symbolic, z_symbolic)
    f_jacobian_θ_symbolic = Symbolics.sparsejacobian(f_symbolic, θ_symbolic)

    # compile all the symbolic expressions into callable julia code
    f! = let
        _f! = Symbolics.build_function(f_symbolic, [z_symbolic; θ_symbolic]; expression = Val{false})[2]
        (result, z, θ) -> _f!(result, [z; θ])
    end

    f_jacobian_z! = let
        _f! = Symbolics.build_function(
            f_jacobian_z_symbolic,
            [z_symbolic; θ_symbolic];
            expression = Val{false},
        )[2]
        rows, cols, _ = SparseArrays.findnz(f_jacobian_z_symbolic)
        SparseFunction(rows, cols) do result, z, θ
            _f!(result, [z; θ])
        end
    end

    f_jacobian_θ! = let
        _f! = Symbolics.build_function(
            f_jacobian_θ_symbolic,
            [z_symbolic; θ_symbolic];
            expression = Val{false},
        )[2]
        rows, cols, _ = SparseArrays.findnz(f_jacobian_θ_symbolic)
        SparseFunction(rows, cols) do result, z, θ
            _f!(result, [z; θ])
        end
    end

    ParametricMCP((; f!, f_jacobian_z!, f_jacobian_θ!, lower_bounds, upper_bounds))
end
