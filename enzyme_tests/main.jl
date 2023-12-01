using Enzyme, EnzymeTestUtils, ParametricMCPs

parameter_dimension = 2
f(z, θ) = [2z[1:2] - z[3:4] - 2θ; z[1:2]]
lower_bounds = [-Inf, -Inf, 0, 0]
upper_bounds = [Inf, Inf, Inf, Inf]
problem = ParametricMCPs.ParametricMCP(f, lower_bounds, upper_bounds, parameter_dimension)

@info "I can can trigger the rule manually just fine:"
dsol = Enzyme.autodiff(
    Forward,
    ParametricMCPs.solve,
    Const(problem),
    Duplicated([1.0, 2.0], [1.0, 0.0]),
)
@show dsol

@info """
Now testing with EnzymeTestUtils.

this fails because `FiniteDifferences.jl` cannot flatten the output struct with to_vec`
"""
try
    test_forward(solve, Duplicated, (problem, Const), ([1.0, 2.0], Duplicated))
catch e
    display(e)
end

@info """
To circumvent the issue above, now we unpack the relevant fields of the output struct for differentiation
this fails because Enzyme here thinks that the activities don't match:

```
Enzyme execution failed.
Mismatched activity for:   store {} addrspace(10)* %.fca.0.0.0.0.extract, {} addrspace(10)** %.fca.0.0.0.0.gep, align 8, !tbaa !113, !alias.scope !117, !noalias !120 const val:   %.fca.0.0.0.0.extract = extractvalue { [1 x [1 x {} addrspace(10)*]], { [1 x [1 x {} addrspace(10)*]], { i64, i64, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)* }, {} addrspace(10)*, {} addrspace(10)*, [2 x i64], {} addrspace(10)* }, { [1 x [1 x {} addrspace(10)*]], { i64, i64, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)* }, {} addrspace(10)*, {} addrspace(10)*, [2 x i64], {} addrspace(10)* }, {} addrspace(10)*, {} addrspace(10)*, i64 } %0, 0, 0, 0
```


(although they should be exactly as our manual autodiff test above?)
"""
try
    test_forward(Duplicated, (problem, Const), ([1.0, 2.0], Duplicated)) do problem, θ
        @inline
        solve(problem, θ).z
    end
catch e
    display(e)
end
