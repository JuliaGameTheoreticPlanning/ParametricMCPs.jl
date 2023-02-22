struct SparseFunction{T}
    _f::T
    rows::Vector{Int}
    cols::Vector{Int}
    size::Tuple{Int,Int}
    constant_entries::Vector{Int}
    function SparseFunction(_f::T, rows, cols, size, constant_entries = Int[]) where {T}
        length(constant_entries) <= length(rows) ||
            throw(ArgumentError("More constant entries than non-zero entries."))
        new{T}(_f, rows, cols, size, constant_entries)
    end
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

"Get the (sparse) linear indices of all entries that are constant in the symbolic matrix M w.r.t. symbolic vector z."
function get_constant_entries(M_symbolic, z_symbolic)
    _z_syms = Symbolics.tosymbol.(z_symbolic)
    findall(SparseArrays.nonzeros(M_symbolic)) do v
        _vars_syms = Symbolics.tosymbol.(Symbolics.get_variables(v))
        isempty(intersect(_vars_syms, _z_syms))
    end
end
