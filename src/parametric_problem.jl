"""
ParametricMCP represents a mixed complementarity problem (MCP) parameterized by some vector `θ`.

Solutions of this problem solve f(z, θ) ⟂ lb ≤ z ≤ ub.
"""
struct ParametricMCP{T1,T2,T3}
    "A callable `f!(result, z, θ)` which stores the result of `f(z, θ)` in `result`"
    f!::T1
    "A callable `j_z!(result, z, θ)` which store the jacobian w.r.t. z at `(z, θ)` in `result`."
    jacobian_z!::T2
    "A callable `j_z!(result, z, θ)` which store the jacobian w.r.t. θ at `(z, θ)` in `result`."
    jacobian_θ!::T3
    "A vector of lower bounds on `z`."
    lower_bounds::Vector{Float64}
    "A vector of upper bounds on `z`."
    upper_bounds::Vector{Float64}
    "The number of 'runtime'-parameters of the problem."
    parameter_dimension::Int
end

"Returns the number of decision variables for this problem."
get_problem_size(problem::ParametricMCP) = length(problem.lower_bounds)

"Returns the number of decision variables for this problem."
get_parameter_dimension(problem::ParametricMCP) = problem.parameter_dimension

"""
The main constructor for compiling a `ParametricMCP` from

- `f`: callabale as `f(z, θ)` that maps a lenght `n` vector of decision variables `z` and a \
parameter vector `θ` of size `parameter_dimension` to an lenght `n` vector output.
- `lower_bounds`: A lenght `n` vector of element-wise lower bounds on the decision variables `z`.
- `upper_bounds`: A length `n` vector of element-wise upper bounds on the decision variables `z`.
- `parameter_dimension`: the size of the parameter vector `θ` in `f`.

Note, this constructor uses `Symbolics.jl` to compile the relevant low-level functions. Therefore,
`f` must be implemented in a sufficiently generic way that supports symbolic evaluation. In cases
where that is strictly infeasible, you can still use the low-level constructor to generate a
`ParametricMCP`. In general, however, the use of this convenience construtor is advised.
"""
function ParametricMCP(
    f,
    lower_bounds,
    upper_bounds,
    parameter_dimension;
    kwargs...,
)
    # TODO
    length(lower_bounds) == length(upper_bounds) ||
        throw(ArgumentError("lower_bounds and upper_bounds have inconsistent lenghts."))
    problem_size = length(lower_bounds)

    # setup the problem symblically
    z_symbolic, θ_symbolic =
        Symbolics.@variables(z_symbolic[1:problem_size], θ_symbolic[1:parameter_dimension]) .|>
        Symbolics.scalarize

    if isempty(θ_symbolic)
        θ_symbolic = Symbolics.Num[]
    end

    f_symbolic = f(z_symbolic, θ_symbolic)

    ParametricMCP(
        f_symbolic,
        z_symbolic,
        θ_symbolic,
        lower_bounds,
        upper_bounds;
        kwargs...,
    )
end

"""
Symbolic version of the ParmetricMCP constructor. If you have `f` and `z` already in terms of symbolic variables, use this.
"""
function ParametricMCP(
    f_symbolic::Vector{Symbolics.Num},
    z_symbolic::Vector{Symbolics.Num},
    θ_symbolic::Vector{Symbolics.Num},
    lower_bounds,
    upper_bounds;
    compute_sensitivities = true,
    parallel = nothing,
)
    length(lower_bounds) == length(upper_bounds) ||
        throw(ArgumentError("lower_bounds and upper_bounds have inconsistent lenghts."))
    problem_size = length(lower_bounds)
    length(f_symbolic) == problem_size || throw(
        ArgumentError(
            "The output lenght of `f` is inconsistent with `lower_bounds` and `upper_bounds`.",
        ),
    )
    jacobian_z_symbolic = Symbolics.sparsejacobian(f_symbolic, z_symbolic)
    jacobian_θ_symbolic = Symbolics.sparsejacobian(f_symbolic, θ_symbolic)

    # compile all the symbolic expressions into callable julia code
    f! = let
        _f! = Symbolics.build_function(
            f_symbolic,
            [z_symbolic; θ_symbolic];
            expression = Val{false},
            parallel,
        )[2]
        (result, z, θ) -> _f!(result, [z; θ])
    end

    jacobian_z! = let
        _f! = Symbolics.build_function(
            jacobian_z_symbolic,
            [z_symbolic; θ_symbolic];
            expression = Val{false},
            parallel,
        )[2]
        rows, cols, _ = SparseArrays.findnz(jacobian_z_symbolic)

        constant_entries = get_constant_entries(jacobian_z_symbolic, z_symbolic)
        SparseFunction(rows, cols, size(jacobian_z_symbolic), constant_entries) do result, z, θ
            _f!(result, [z; θ])
        end
    end

    if compute_sensitivities
        jacobian_θ! = let
            _f! = Symbolics.build_function(
                jacobian_θ_symbolic,
                [z_symbolic; θ_symbolic];
                expression = Val{false},
                parallel,
            )[2]
            rows, cols, _ = SparseArrays.findnz(jacobian_θ_symbolic)
            SparseFunction(rows, cols, size(jacobian_θ_symbolic)) do result, z, θ
                _f!(result, [z; θ])
            end
        end
    else
        jacobian_θ! = nothing
    end

    parameter_dimension = length(θ_symbolic)
    ParametricMCP(f!, jacobian_z!, jacobian_θ!, lower_bounds, upper_bounds, parameter_dimension)
end
