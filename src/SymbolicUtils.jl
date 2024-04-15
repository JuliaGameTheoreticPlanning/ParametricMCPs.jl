"""
Minimal abstraction on top of `Symbolics.jl` and `FastDifferentiation.jl` to make switching between the two easier.
"""
module SymbolicUtils

using Symbolics: Symbolics
using FastDifferentiation: FastDifferentiation as FD

export SymbolicsBackend, FastDifferentationBackend, make_variables, build_function

struct SymbolicsBackend end
struct FastDifferentationBackend end

"""
    make_variables(backend, name, dimension)

Creates a vector of `dimension` where each element is a scalar symbolic variable from `backend` with the given `name`.
"""
function make_variables end

function make_variables(::SymbolicsBackend, name::Symbol, dimension::Int)
    Symbolics.@variables($name[1:dimension]) .|> only |> Symbolics.scalarize
end

function make_variables(::FastDifferentationBackend, name::Symbol, dimension::Int)
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
    ::SymbolicsBackend,
    f_symbolic,
    args_symbolic...;
    in_place,
    backend_options = (; parallel = Symbolics.ShardedForm(),),
)
    f_callable, f_callable! = Symbolics.build_function(
        f_symbolic,
        args_symbolic...;
        expression = Val{false},
        backend_options...,
    )
    in_place ? f_callable! : f_callable
end

function build_function(
    ::FastDifferentiation,
    f_symbolic,
    args_symbolic...;
    in_place,
    backend_options = (;),
)
    FD.make_function(f_symbolic, args_symbolic...; in_place, backend_options...)
end

function gradient end

function gradient(::SymbolicsBackend, f_symbolic, x_symbolic)
    Symbolics.gradient(f_symbolic, x_symbolic)
end

function gradient(::FastDifferentationBackend, f, x)
    # FD does not have a gradient utility so we just flatten the jacobian here
    vec(FD.jacobian([f], x))
end

function jacobian end

function jacobian(::SymbolicsBackend, f_symbolic, x_symbolic)
    Symbolics.jacobian(f_symbolic, x_symbolic)
end

function jacobian(::FastDifferentationBackend, f, x)
    FD.jacobian([f], x)
end

function sparse_jacobian end

function sparse_jacobian(::SymbolicsBackend, f_symbolic, x_symbolic)
    Symbolics.sparsejacobian(f_symbolic, x_symbolic)
end

function sparse_jacobian(::FastDifferentationBackend, f, x)
    FD.sparse_jacobian(f, x)
end

end
