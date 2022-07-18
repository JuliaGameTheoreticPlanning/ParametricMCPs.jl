# DifferentiableMCPs

[![CI](https://github.com/lassepe/DifferentiableMCPs.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/lassepe/DifferentiableMCPs.jl/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

This packages provide a generic, differentiable mathematical programming layer by compiling mixed complementarity problems (MCPs) parameterized by a "runtime"-parameter vector. The resulting `ParametricMCP` can solved for different parameter instantiations using `solve(problem, paramters)` and the `solve` routine is made differentiable via `ChainRulesCore`.

## Installation

This package is not yet registered. For now, simply install it as git dependency (and don't forget to check-in you `Manifest.toml` if you do).

```julia
] add https://github.com/lassepe/DifferentiableMCPs.jl
```

## Quickstart by Example

Simple forward computation:

```julia
using DifferentiableMCPs

# setup a simple MCP which represents a QP with
# - cost: sum((z[1:2] - θ).^2)
# - constaints: z[1:2] >= 0

f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]
lower_bounds = [-Inf, -Inf, 0, 0]
upper_bounds = [Inf, Inf, Inf, Inf]
problem = ParametricMCP(f, lower_bounds, upper_bounds, parameter_dimension)

some_parameter = [1.0, 2.0]
solve(problem, some_parameter)
```

Since we provide differentiation rules via `ChainRulesCore`, the solver can be
using your favourite ad-framework, e.g., Zygote:

```julia
using Zygote

function dummy_pipeline(θ)
    solution = DifferentiableMCPs.solve(problem, θ)
    sum(solution.z .^ 2)
end

Zygote.gradient(dummy_pipeline, some_parameter)
```
