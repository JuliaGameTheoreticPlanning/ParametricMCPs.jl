# ParametricMCPs

[![CI](https://github.com/JuliaGameTheoreticPlanning/ParametricMCPs.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaGameTheoreticPlanning/ParametricMCPs.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JuliaGameTheoreticPlanning/ParametricMCPs.jl/graph/badge.svg?token=2YL0BRh6HV)](https://codecov.io/gh/JuliaGameTheoreticPlanning/ParametricMCPs.jl)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

This package provides a generic, differentiable mathematical programming layer by compiling mixed complementarity problems (MCPs) parameterized by a "runtime"-parameter vector. The resulting `ParametricMCP` can be solved for different parameter instantiations using `solve(problem, parameters)` and the `solve` routine is made differentiable via [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) and [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl).

## Installation

ParametricMCPs is a registered package.
Thus, installation is as simple as:

```julia
] add ParametricMCPs
```

This package uses the proprietary PATH solver under the hood (via [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl)).
Therefore, you will need a license key to solve larger problems.
However, by courtesy of Steven Dirkse, Michael Ferris, and Tudd Munson,
[temporary licenses are available free of charge](https://pages.cs.wisc.edu/~ferris/path.html).
Please consult the documentation of [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl) to learn about loading the license key.

## Quickstart by Example

Simple forward computation:

```julia
using ParametricMCPs

# setup a simple MCP which represents a QP with
# - cost: sum((z[1:2] - θ).^2)
# - constaints: z[1:2] >= 0

f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]
lower_bounds = [-Inf, -Inf, 0, 0]
upper_bounds = [Inf, Inf, Inf, Inf]
parameter_dimension = 2
problem = ParametricMCP(f, lower_bounds, upper_bounds, parameter_dimension)

some_parameter = [1.0, 2.0]
solution = solve(problem, some_parameter)

# You can also warm-start the solver with an initial guess.
# For example, say that we want to solve the problem at a slightly perturbed parameter value, `some_other_parameter = some_parameter .+ 0.01`.
# Here, we can warm-start the solver by passing in the old solution as an intial guess.
# This is particularly handy for online optimization as in receding-horizon applications.
some_other_parameter = some_parameter .+ 0.01
other_solution = solve(problem, some_other_parameter; initial_guess = solution.z)
```

Since we provide differentiation rules via `ChainRulesCore`, the solver can be
differentiated using your favourite ad-framework, e.g., Zygote:

```julia
using Zygote

function dummy_pipeline(θ)
    solution = ParametricMCPs.solve(problem, θ)
    sum(solution.z .^ 2)
end

Zygote.gradient(dummy_pipeline, some_parameter)
```

## Acknowledgements

This package is effectively a thin wrapper around the great work of other people.
Special thanks goes to the maintainers of the following packages:

- [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl)
- [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)

## Related Packages

For some specialized, closely related applications, you may want to consider the following packages (all of which also provide differentiation rules):

- [TensorGames.jl](https://github.com/forrestlaine/TensorGames.jl) solves finite N-player normal-form games.
- [DifferentiableTrajectoryOptimization.jl](https://github.com/lassepe/DifferentiableTrajectoryOptimization.jl) solves parametric (single-player) trajectory optimization problems. The interface is very similar to ParametricMCPs.jl. Beyond the PATH solver, this package also supports backends like IPOPT and OSQP.
