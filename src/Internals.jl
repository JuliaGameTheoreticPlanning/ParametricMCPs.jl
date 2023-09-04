module Internals

using FastDifferentiation: FastDifferentiation as FD

function parameterize(f::Vector, θ)
    f
end

function parameterize(f, θ)
    f(θ)
end

function make_callable(f::AbstractVector{<:FD.Node}, x, output_size)
    result_buffer = zeros(output_size)
    _f! = FD.make_function(f, x; in_place = true)
    function (x)
        _f!(result_buffer, x)
        result_buffer
    end
end

function make_callable(f::AbstractVector{<:Real}, x, output_size)
    @assert output_size == length(f)
    Returns(f)
end

function infer_problem_size(lower_bounds, upper_bounds)
    if lower_bounds isa AbstractVector
        length(lower_bounds)
    elseif upper_bounds isa AbstractVector
        length(upper_bounds)
    else
        throw(
            ArgumentError(
                "Cannot infer problem size from lower_bounds and upper_bounds. Provide it explicitly.",
            ),
        )
    end
end

end
