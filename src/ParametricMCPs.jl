module ParametricMCPs

using LinearAlgebra: LinearAlgebra
using PATHSolver: PATHSolver
using SparseArrays: SparseArrays
using Symbolics: Symbolics

include("sparse_utils.jl")

include("parametric_problem.jl")
export ParametricMCP, get_parameter_dimension, get_problem_size
include("solver.jl")
export solve

include("autodiff.jl")

if !isdefined(Base, :get_extension)
    include("../ext/ChainRulesCoreExt.jl")
    include("../ext/ForwardDiffExt.jl")
end

end
