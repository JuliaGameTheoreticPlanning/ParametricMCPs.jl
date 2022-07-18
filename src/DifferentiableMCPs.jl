module DifferentiableMCPs

using Symbolics: Symbolics
using SparseArrays: SparseArrays
using PATHSolver: PATHSolver
using LinearAlgebra: LinearAlgebra
using ChainRulesCore: ChainRulesCore

include("sparse_utils.jl")
include("parametric_problem.jl")
include("solver.jl")
include("autodiff.jl")

end
