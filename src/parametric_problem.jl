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
    "The number of decision variables of the problem."
    problem_size::Int
end

"Returns the number of decision variables for this problem."
get_problem_size(problem::ParametricMCP) = problem.problem_size

"Returns the number of decision variables for this problem."
get_parameter_dimension(problem::ParametricMCP) = problem.parameter_dimension

# This method allows the user to specify the backend via a keyword argument which defaults to using Symbolics.jl

"""
--------------------------------------------------------------------------------

The main constructor for compiling a `ParametricMCP` from

Positional arguments:
- `f`: callable as `f(z, θ)` that maps a length `n` vector of decision variables `z` and a \
parameter vector `θ` of size `parameter_dimension` to an length `n` vector output.
- `lower_bounds`: A length `n` vector of element-wise lower bounds on the decision variables `z`.
- `upper_bounds`: A length `n` vector of element-wise upper bounds on the decision variables `z`.
- `parameter_dimension`: the size of the parameter vector `θ` in `f`.

Keyword arguments:
- `[backend]`: the backend (from ADTypes.jl) to be used for compiling callbacks for `f` and its Jacobians needed by PATH. `AutoSymbolics()` (default) is slightly more flexible. `AutoFastDifferentiation()` has reduced compilation times and reduced runtime in some cases.
- `compute_sensitivities`: whether to compile the callbacks needed for sensitivity computation.
- `[problem_size]`: the number of decision variables. If not provided and `lower_bounds` or `upper_bounds` are vectors, the problem size is inferred from the length of these vectors.

Note, this constructor uses symbolic tracing to compile the relevant low-level functions. Therefore,
`f` must be implemented in a sufficiently generic way that supports symbolic evaluation. In cases
where that is infeasible or impractical, you can still use the low-level constructor to generate a
`ParametricMCP`. In general, however, the use of this convenience constructor is advised.
"""
function ParametricMCP(
    f,
    lower_bounds,
    upper_bounds,
    parameter_dimension;
    backend = ADTypes.AutoSymbolics(),
    problem_size = Internals.infer_problem_size(lower_bounds, upper_bounds),
    kwargs...,
)
    problem_size = Internals.check_dimensions(lower_bounds, upper_bounds, problem_size)

    z_symbolic = SymbolicUtils.make_variables(backend, :z, problem_size)
    θ_symbolic = SymbolicUtils.make_variables(backend, :θ, parameter_dimension)
    f_symbolic = f(z_symbolic, θ_symbolic)

    ParametricMCP(f_symbolic, z_symbolic, θ_symbolic, lower_bounds, upper_bounds; kwargs...)
end

"""
Symbolic version of the ParmetricMCP constructor. If you have `f` and `z` already in terms of symbolic variables, use this.
"""
function ParametricMCP(
    f_symbolic::Vector{T},
    z_symbolic::Vector{T},
    θ_symbolic::Vector{T},
    lower_bounds::Vector,
    upper_bounds::Vector;
    compute_sensitivities = true,
    warm_up_callbacks = true,
    parallel = nothing,
    backend_options = (;),
) where {T<:Union{FD.Node,Symbolics.Num}}
    problem_size = Internals.check_dimensions(f_symbolic, z_symbolic, lower_bounds, upper_bounds)

    if !isnothing(parallel)
        Base.depwarn(
            "The `parallel` keyword argument in the constructor of `ParametricMCP` is deprecated and will be removed in a future release. Use `backend_options` instead.",
            :ParametricMCPs;
            force = true,
        )
        backend_options = merge(backend_options, (; parallel))
    end

    # compile all the symbolic expressions into callable julia code
    f! = let
        # The multi-arg version of `make_function` is broken so we concatenate to a single arg here
        _f! = SymbolicUtils.build_function(
            f_symbolic,
            [z_symbolic; θ_symbolic];
            in_place = true,
            backend_options,
        )
        (result, z, θ) -> _f!(result, [z; θ])
    end

    # same as above but for the Jacobian in z
    jacobian_z! = let
        jacobian_z = SymbolicUtils.sparse_jacobian(f_symbolic, z_symbolic)
        _jacobian_z! = SymbolicUtils.build_function(
            jacobian_z,
            [z_symbolic; θ_symbolic];
            in_place = true,
            backend_options,
        )
        rows, cols, _ = SparseArrays.findnz(jacobian_z)
        constant_entries = get_constant_entries(jacobian_z, z_symbolic)
        SparseFunction(rows, cols, size(jacobian_z), constant_entries) do result, z, θ
            _jacobian_z!(result, [z; θ])
        end
    end

    if compute_sensitivities
        jacobian_θ! = let
            jacobian_θ = SymbolicUtils.sparse_jacobian(f_symbolic, θ_symbolic)
            _jacobian_θ! = SymbolicUtils.build_function(
                jacobian_θ,
                [z_symbolic; θ_symbolic];
                in_place = true,
            )
            rows, cols, _ = SparseArrays.findnz(jacobian_θ)
            constant_entries = get_constant_entries(jacobian_θ, θ_symbolic)
            SparseFunction(rows, cols, size(jacobian_θ), constant_entries) do result, z, θ
                _jacobian_θ!(result, [z; θ])
            end
        end
    else
        jacobian_θ! = nothing
    end

    parameter_dimension = length(θ_symbolic)
    mcp = ParametricMCP(
        f!,
        jacobian_z!,
        jacobian_θ!,
        lower_bounds,
        upper_bounds,
        parameter_dimension,
        problem_size,
    )

    if warm_up_callbacks
        _warm_up_callbacks(mcp)
    end

    mcp
end

"""
Call all functions with a dummy input to trigger JIT compilation.
"""
function _warm_up_callbacks(mcp::ParametricMCP)
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
