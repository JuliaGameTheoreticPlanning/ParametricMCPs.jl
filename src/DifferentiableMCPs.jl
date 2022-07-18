module DifferentiableMCPs

using ChainRulesCore: ChainRulesCore
using LinearAlgebra: LinearAlgebra
using PATHSolver: PATHSolver
using SparseArrays: SparseArrays
using Symbolics: Symbolics

include("sparse_utils.jl")
include("parametric_problem.jl")
include("solver.jl")
include("autodiff.jl")

end
