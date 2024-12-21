module ParametricMCPs

using PATHSolver: PATHSolver
using SparseArrays: SparseArrays
using SymbolicTracingUtils: SymbolicTracingUtils

const SymbolicUtils = SymbolicTracingUtils

# re-exporting symbolic tracing utilities for backward compatibility
export SymbolicTracingUtils, SymbolicUtils

include("Internals.jl")

include("parametric_problem.jl")
export ParametricMCP, get_parameter_dimension, get_problem_size
include("solver.jl")
export solve

include("InternalAutoDiffUtils.jl")
end
