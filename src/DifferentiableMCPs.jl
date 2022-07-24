module DifferentiableMCPs

using ChainRulesCore: ChainRulesCore
using LinearAlgebra: LinearAlgebra
using PATHSolver: PATHSolver
using SparseArrays: SparseArrays
using Symbolics: Symbolics
using LazyArrays: LazyArrays

include("sparse_utils.jl")
include("parametric_problem.jl")
export ParametricMCP, get_parameter_dimension, get_problem_size
include("solver.jl")
export solve
include("autodiff.jl")

end
