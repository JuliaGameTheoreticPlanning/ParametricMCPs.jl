module DifferentiableMCPs

using Symbolics: Symbolics
using SparseArrays: SparseArrays
using PATHSolver: PATHSolver

include("sparse_utils.jl")
include("parametric_problem.jl")
include("solver.jl")

end
