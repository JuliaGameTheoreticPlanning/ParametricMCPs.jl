struct SparseFunction{T}
    _f::T
    rows::Vector{Int}
    cols::Vector{Int}
    size::Tuple{Int,Int}
end
(f::SparseFunction)(args...) = f._f(args...)
SparseArrays.nnz(f::SparseFunction) = length(f.rows)

function get_result_buffer(f::SparseFunction)
    data = zeros(SparseArrays.nnz(f))
    SparseArrays.sparse(f.rows, f.cols, data, f.size...)
end

"""
Convert a Julia sparse array `M` into the \
[COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) format required by PATH.
This implementation has been extracted from \
[here](https://github.com/chkwon/PATHSolver.jl/blob/8e63723e51833cdbab58c39b6646f8cdf79d74a2/src/C_API.jl#L646)
"""
function _coo_from_sparse!(col, len, row, data, M)
    @assert length(col) == length(len) == size(M, 1)
    @assert length(row) == length(data)
    n = length(col)
    for i in 1:n
        col[i] = M.colptr[i]
        len[i] = M.colptr[i + 1] - M.colptr[i]
    end
    for (i, v) in enumerate(SparseArrays.rowvals(M))
        row[i] = v
    end
    for (i, v) in enumerate(SparseArrays.nonzeros(M))
        data[i] = v
    end
end
