module Internals

using SymbolicTracingUtils: FD
using SparseArrays: SparseArrays

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

"""
Convert a Julia sparse array `M` into the \
[COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) format required by PATH.
This implementation has been extracted from \
[here](https://github.com/chkwon/PATHSolver.jl/blob/8e63723e51833cdbab58c39b6646f8cdf79d74a2/src/C_API.jl#L646)
"""
function _coo_from_sparse!(col, len, row, data, M)
    @assert length(col) == length(len) == size(M, 1)
    @assert length(row) == length(data) == SparseArrays.nnz(M)
    n = length(col)
    @inbounds begin
        col .= @view M.colptr[1:n]
        len .= diff(M.colptr)
        row .= SparseArrays.rowvals(M)
        data .= SparseArrays.nonzeros(M)
    end
end

end
