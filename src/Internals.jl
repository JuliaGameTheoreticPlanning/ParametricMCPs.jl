module Internals

using FastDifferentiation: FastDifferentiation as FD

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

function check_dimensions(args...)
    function to_dimension(arg)
        if arg isa AbstractVector
            length(arg)
        elseif arg isa Int
            arg
        else
            throw(ArgumentError("Expected an AbstractVector or an Int."))
        end
    end

    the_unique_dimension = (to_dimension(arg) for arg in args) |> unique |> only
    the_unique_dimension
end

end
