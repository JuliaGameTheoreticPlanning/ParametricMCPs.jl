"""
ParametricMCP represents a mixed complementarity problem (MCP) parameterized by some vector `θ`.

Solutions of this problem solve f(z, θ) ⟂ lb ≤ z ≤ ub.

--------------------------------------------------------------------------------

The main constructor for compiling a `ParametricMCP` from

Positional arguments:
- `f`: callable as `f(z, θ)` that maps a length `n` vector of decision variables `z` and a \
parameter vector `θ` of size `parameter_dimension` to an length `n` vector output.
- `lower_bounds`: A length `n` vector of element-wise lower bounds on the decision variables `z`.
- `upper_bounds`: A length `n` vector of element-wise upper bounds on the decision variables `z`.
- `parameter_dimension`: the size of the parameter vector `θ` in `f`.

Keyword arguments:
- `[backend]`: the backend to be used for compiling callbacks for `f` and its Jacobians needed by PATH. `SymbolicsBackend` (default) is slightly more flexible. `FastDifferentationBackend` has reduced compilation times and reduced runtime in some cases.
- `compute_sensitivities`: whether to compile the callbacks needed for sensitivity computation.
- `[problem_size]`: the number of decision variables. If not provided and `lower_bounds` or `upper_bounds` are vectors, the problem size is inferred from the length of these vectors.

Note, this constructor uses `Symbolics.jl` to compile the relevant low-level functions. Therefore,
`f` must be implemented in a sufficiently generic way that supports symbolic evaluation. In cases
where that is strictly infeasible, you can still use the low-level constructor to generate a
`ParametricMCP`. In general, however, the use of this convenience constructor is advised.

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
    "The number of decision variables of the problem."
    problem_size::Int
end

"Returns the number of decision variables for this problem."
get_problem_size(problem::ParametricMCP) = problem.problem_size

"Returns the number of decision variables for this problem."
get_parameter_dimension(problem::ParametricMCP) = problem.parameter_dimension

struct SymbolicsBackend end
struct FastDifferentationBackend end

# This method allows the user to specify the backend via a keyword argument which defaults to using Symbolics.jl
function ParametricMCP(
    f,
    lower_bounds,
    upper_bounds,
    parameter_dimension;
    backend = SymbolicsBackend(),
    kwargs...,
)
    ParametricMCP(f, backend, lower_bounds, upper_bounds, parameter_dimension; kwargs...)
end

# Dispatch for using the Symbolics.jl backend
function ParametricMCP(
    f,
    ::SymbolicsBackend,
    lower_bounds,
    upper_bounds,
    parameter_dimension;
    problem_size = Internals.infer_problem_size(lower_bounds, upper_bounds),
    kwargs...,
)
    problem_size = Internals.check_dimensions(lower_bounds, upper_bounds, problem_size)

    # setup the problem symbolically
    z_symbolic, θ_symbolic =
        Symbolics.@variables(z_symbolic[1:problem_size], θ_symbolic[1:parameter_dimension]) .|>
        Symbolics.scalarize

    if isempty(θ_symbolic)
        θ_symbolic = Symbolics.Num[]
    end

    f_symbolic = f(z_symbolic, θ_symbolic)

    ParametricMCP(f_symbolic, z_symbolic, θ_symbolic, lower_bounds, upper_bounds; kwargs...)
end

function ParametricMCP(
    f,
    ::FastDifferentationBackend,
    lower_bounds,
    upper_bounds,
    parameter_dimension;
    problem_size = Internals.infer_problem_size(lower_bounds, upper_bounds),
)
    Internals.check_dimensions(lower_bounds, upper_bounds, problem_size)

    # setup the problem symbolically
    z_node = FD.make_variables(:z, problem_size)
    θ_node = FD.make_variables(:θ, parameter_dimension)
    f_node = f(z_node, θ_node)

    ParametricMCP(f_node, z_node, θ_node, lower_bounds, upper_bounds; compute_sensitivities)
end

"""
FastDifferentation.jl version of the ParmetricMCP constructor. If you have `f` and `z` already in terms of `FastDifferentation.Node`, use this.

Dev notes:
- This may become the new default back-end since it promises to be faster than Symbolics.jl. (both in terms of compilation/code-gen time and execution time).
- We may be able to convert the Symbolics.jl representation in the future via `FDConversion.jl`
"""
function ParametricMCP(
    f::Vector{T},
    z::Vector{T},
    θ::Vector{T},
    lower_bounds::Vector,
    upper_bounds::Vector;
    compute_sensitivities = true,
) where {T<:FD.Node}
    problem_size = Internals.check_dimensions(f, z, lower_bounds, upper_bounds)

    # compile all the symbolic expressions into callable julia code
    f! = let
        # The multi-arg version of `make_function` is broken so we concatenate to a single arg here
        _f! = FD.make_function(f, [z; θ]; in_place = true)
        (result, z, θ) -> _f!(result, [z; θ])
    end

    # same as above but for the Jacobian in z
    jacobian_z! = let
        jacobian_z = FD.sparse_jacobian(f, z)
        _jacobian_z! = FD.make_function(jacobian_z, [z; θ]; in_place = true)
        rows, cols, _ = SparseArrays.findnz(jacobian_z)
        # TODO: constant entry detection
        constant_entries = get_constant_entries(jacobian_z, z)
        SparseFunction(rows, cols, size(jacobian_z), constant_entries) do result, z, θ
            _jacobian_z!(result, [z; θ])
        end
    end

    if compute_sensitivities
        jacobian_θ! = let
            jacobian_θ = FD.sparse_jacobian(f, θ)
            _jacobian_θ! = FD.make_function(jacobian_θ, [z; θ]; in_place = true)
            rows, cols, _ = SparseArrays.findnz(jacobian_θ)
            # TODO: constant entry detection
            constant_entries = get_constant_entries(jacobian_θ, θ)
            SparseFunction(rows, cols, size(jacobian_θ), constant_entries) do result, z, θ
                _jacobian_θ!(result, [z; θ])
            end
        end
    else
        jacobian_θ! = nothing
    end

    parameter_dimension = length(θ)
    ParametricMCP(
        f!,
        jacobian_z!,
        jacobian_θ!,
        lower_bounds,
        upper_bounds,
        parameter_dimension,
        problem_size,
    )
end

"""
Symbolics.jl version of the ParmetricMCP constructor. If you have `f` and `z` already in terms of symbolic variables, use this.
"""
function ParametricMCP(
    f_symbolic::Vector{<:Symbolics.Num},
    z_symbolic::Vector{<:Symbolics.Num},
    θ_symbolic::Vector{<:Symbolics.Num},
    lower_bounds,
    upper_bounds;
    compute_sensitivities = true,
    parallel = Symbolics.ShardedForm(),
)
    problem_size = Internals.check_dimensions(f_symbolic, z_symbolic, lower_bounds, upper_bounds)

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
            constant_entries = get_constant_entries(jacobian_θ_symbolic, θ_symbolic)
            SparseFunction(rows, cols, size(jacobian_θ_symbolic), constant_entries) do result, z, θ
                _f!(result, [z; θ])
            end
        end
    else
        jacobian_θ! = nothing
    end

    parameter_dimension = length(θ_symbolic)
    ParametricMCP(
        f!,
        jacobian_z!,
        jacobian_θ!,
        lower_bounds,
        upper_bounds,
        parameter_dimension,
        problem_size,
    )
end

"""
Call all functions with a dummy input to trigger JIT compilation.
"""
function compile_callbacks(mcp::ParametricMCP)
    # TODO: if all callbacks are type stable, we could also use `precompile` here
    θ = zeros(get_parameter_dimension(mcp))
    z = zeros(get_problem_size(mcp))
    mcp.f!(z, z, θ)
    jacz = ParametricMCPs.get_result_buffer(mcp.jacobian_z!)
    mcp.jacobian_z!(jacz, z, θ)
    if !isnothing(mcp.jacobian_θ!)
        jacθ = ParametricMCPs.get_result_buffer(mcp.jacobian_θ!)
        mcp.jacobian_θ!(jacθ, z, θ)
    end
    nothing
end
