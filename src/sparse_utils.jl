struct SparseFunction{T1,T2}
    _f::T1
    result_buffer::T2
    rows::Vector{Int}
    cols::Vector{Int}
    size::Tuple{Int,Int}
    constant_entries::Vector{Int}
    function SparseFunction(_f::T1, rows, cols, size, constant_entries = Int[]) where {T1}
        length(constant_entries) <= length(rows) ||
            throw(ArgumentError("More constant entries than non-zero entries."))
        result_buffer = get_result_buffer(rows, cols, size)
        new{T1,typeof(result_buffer)}(_f, result_buffer, rows, cols, size, constant_entries)
    end
end

(f::SparseFunction)(args...) = f._f(args...)
SparseArrays.nnz(f::SparseFunction) = length(f.rows)

function get_result_buffer(rows::Vector{Int}, cols::Vector{Int}, size::Tuple{Int,Int})
    data = zeros(length(rows))
    SparseArrays.sparse(rows, cols, data, size...)
end

function get_result_buffer(f::SparseFunction)
    get_result_buffer(f.rows, f.cols, f.size)
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

"Get the (sparse) linear indices of all entries that are constant in the symbolic matrix M w.r.t. symbolic vector z."
function get_constant_entries(
    M_symbolic::AbstractMatrix{<:Symbolics.Num},
    z_symbolic::AbstractVector{<:Symbolics.Num},
)
    _z_syms = Symbolics.tosymbol.(z_symbolic)
    findall(SparseArrays.nonzeros(M_symbolic)) do v
        _vars_syms = Symbolics.tosymbol.(Symbolics.get_variables(v))
        isempty(intersect(_vars_syms, _z_syms))
    end
end

function get_constant_entries(
    M_symbolic::AbstractMatrix{<:FD.Node},
    z_symbolic::AbstractVector{<:FD.Node},
)
    _z_syms = [zs.node_value for zs in FD.variables(z_symbolic)]
    # find all entries that are not a function of any of the symbols in z
    findall(SparseArrays.nonzeros(M_symbolic)) do v
        _vars_syms = [vs.node_value for vs in FD.variables(v)]
        isempty(intersect(_vars_syms, _z_syms))
    end
end
