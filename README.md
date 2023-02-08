# ParametricMCPs

[![CI](https://github.com/lassepe/ParametricMCPs.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/lassepe/ParametricMCPs.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lassepe/ParametricMCPs.jl/branch/main/graph/badge.svg?token=knLJ9hVfeO)](https://codecov.io/gh/lassepe/ParametricMCPs.jl)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

This packages provide a generic, differentiable mathematical programming layer by compiling mixed complementarity problems (MCPs) parameterized by a "runtime"-parameter vector. The resulting `ParametricMCP` can solved for different parameter instantiations using `solve(problem, paramters)` and the `solve` routine is made differentiable via [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) and [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl).

## Installation

This package is not yet registered. For now, simply install it as git dependency (and don't forget to check-in you `Manifest.toml` if you do).

```julia
] add https://github.com/lassepe/ParametricMCPs.jl
```

This package uses the proprietary PATH solver under the heed (via [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl)).
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
solve(problem, some_parameter)
```

Since we provide differentiation rules via `ChainRulesCore`, the solver can be
using your favourite ad-framework, e.g., Zygote:

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
