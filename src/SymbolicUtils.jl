"""
Minimal abstraction on top of `Symbolics.jl` and `FastDifferentiation.jl` to make switching between the two easier.
"""
module SymbolicUtils

using Symbolics: Symbolics
using FastDifferentiation: FastDifferentiation as FD

using ADTypes: ADTypes

export SymbolicsBackend, FastDifferentiationBackend, make_variables, build_function

function SymbolicsBackend()
    Base.depwarn(
        "The `SymbolicsBackend` type is deprecated and will be removed in a future release. Use `ADTypes.AutoSymbolics` instead.",
        :SymbolicUtils,
        force = true,
    )
    ADTypes.AutoSymbolics()
end

function FastDifferentiationBackend()
    Base.depwarn(
        "The `FastDifferentiationBackend` type is deprecated and will be removed in a future release. Use `ADTypes.AutoFastDifferentiation` instead.",
        :SymbolicUtils,
        force = true,
    )
    ADTypes.AutoFastDifferentiation()
end

"""
    make_variables(backend, name, dimension)

Creates a vector of `dimension` where each element is a scalar symbolic variable from `backend` with the given `name`.
"""
function make_variables end

function make_variables(::ADTypes.AutoSymbolics, name::Symbol, dimension::Int)
    vars = Symbolics.@variables($name[1:dimension]) |> only |> Symbolics.scalarize

    if isempty(vars)
        vars = Symbolics.Num[]
    end

    vars
end

function make_variables(::ADTypes.AutoFastDifferentiation, name::Symbol, dimension::Int)
    FD.make_variables(name, dimension)
end

"""
    build_function(backend, f_symbolic, args_symbolic...; in_place, options)

Builds a callable function from a symbolic expression `f_symbolic` with the given `args_symbolic` as arguments.

Depending on the `in_place` flag, the function will be built as in-place `f!(result, args...)` or out-of-place variant `restult = f(args...)`.

`backend_options` will be forwarded to the backend specific function and differ between backends.
"""
function build_function end

function build_function(
    f_symbolic::AbstractArray{T},
    args_symbolic...;
    in_place,
    backend_options = (;),
) where {T<:Symbolics.Num}
    f_callable, f_callable! = Symbolics.build_function(
        f_symbolic,
        args_symbolic...;
        expression = Val{false},
        # slightly saner defaults...
        (; parallel = Symbolics.ShardedForm(), backend_options...)...,
    )
    in_place ? f_callable! : f_callable
end

function build_function(
    f_symbolic::AbstractArray{T},
    args_symbolic...;
    in_place,
    backend_options = (;),
) where {T<:FD.Node}
    FD.make_function(f_symbolic, args_symbolic...; in_place, backend_options...)
end

"""
    gradient(f_symbolic, x_symbolic)

Computes the symbolic gradient of `f_symbolic` with respect to `x_symbolic`.
"""
function gradient end

function gradient(f_symbolic::T, x_symbolic::Vector{T}) where {T<:Symbolics.Num}
    Symbolics.gradient(f_symbolic, x_symbolic)
end

function gradient(f_symbolic::T, x_symbolic::Vector{T}) where {T<:FD.Node}
    # FD does not have a gradient utility so we just flatten the jacobian here
    vec(FD.jacobian([f_symbolic], x_symbolic))
end

"""
    jacobian(f_symbolic, x_symbolic)

Computes the symbolic Jacobian of `f_symbolic` with respect to `x_symbolic`.
"""
function jacobian end

function jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:Symbolics.Num}
    Symbolics.jacobian(f_symbolic, x_symbolic)
end

function jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:FD.Node}
    FD.jacobian([f_symbolic], x_symbolic)
end

"""
    sparse_jacobian(f_symbolic, x_symbolic)

Computes the symbolic Jacobian of `f_symbolic` with respect to `x_symbolic` in a sparse format.
"""
function sparse_jacobian end

function sparse_jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:Symbolics.Num}
    Symbolics.sparsejacobian(f_symbolic, x_symbolic)
end

function sparse_jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:FD.Node}
    FD.sparse_jacobian(f_symbolic, x_symbolic)
end

end
