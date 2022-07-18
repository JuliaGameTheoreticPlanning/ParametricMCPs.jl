struct ParametricMCP{T1,T2,T3}
    f!::T1
    jacobian_z!::T2
    jacobian_θ!::T3
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    parameter_dimension::Int
end

get_problem_size(problem::ParametricMCP) = length(problem.lower_bounds)
get_parameter_dimension(problem::ParametricMCP) = problem.parameter_dimension

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
    jacobian_z_symbolic = Symbolics.sparsejacobian(f_symbolic, z_symbolic)
    jacobian_θ_symbolic = Symbolics.sparsejacobian(f_symbolic, θ_symbolic)

    # compile all the symbolic expressions into callable julia code
    f! = let
        _f! = Symbolics.build_function(f_symbolic, [z_symbolic; θ_symbolic]; expression = Val{false})[2]
        (result, z, θ) -> _f!(result, [z; θ])
    end

    jacobian_z! = let
        _f! = Symbolics.build_function(
            jacobian_z_symbolic,
            [z_symbolic; θ_symbolic];
            expression = Val{false},
        )[2]
        rows, cols, _ = SparseArrays.findnz(jacobian_z_symbolic)
        SparseFunction(rows, cols, size(jacobian_z_symbolic)) do result, z, θ
            _f!(result, [z; θ])
        end
    end

    jacobian_θ! = let
        _f! = Symbolics.build_function(
            jacobian_θ_symbolic,
            [z_symbolic; θ_symbolic];
            expression = Val{false},
        )[2]
        rows, cols, _ = SparseArrays.findnz(jacobian_θ_symbolic)
        SparseFunction(rows, cols, size(jacobian_θ_symbolic)) do result, z, θ
            _f!(result, [z; θ])
        end
    end

    ParametricMCP(f!, jacobian_z!, jacobian_θ!, lower_bounds, upper_bounds, parameter_dimension)
end
