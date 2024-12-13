module ParametricMCPs

using PATHSolver: PATHSolver
using SparseArrays: SparseArrays
using FastDifferentiation: FastDifferentiation as FD
using Symbolics: Symbolics
using SymblicTracingUtils: SymbolicTracingUtils

abstract type AbstractParametricMCPSolver end

include("problem.jl")

# include("Internals.jl")
# include("sparse_utils.jl")
# include("parametric_problem.jl")
# export ParametricMCP, get_parameter_dimension, get_problem_size
# include("path_solver_backend.jl")
# export solve
# include("InternalAutoDiffUtils.jl")
end
