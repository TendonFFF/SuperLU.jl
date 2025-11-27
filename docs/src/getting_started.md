# Getting Started

This guide will help you get started with SuperLU.jl for solving sparse linear systems.

## Basic Usage

### Direct API

The simplest way to use SuperLU.jl is through its direct API:

```julia
using SuperLU
using SparseArrays

# Create a sparse complex matrix
A = sparse([1.0+1.0im 2.0+0im 0.0; 
            3.0-1.0im 4.0+2.0im 1.0+0im; 
            0.0 1.0+1.0im 5.0-1.0im])
b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]

# Create factorization and solve
F = SuperLUFactorize(A)
factorize!(F)
x = copy(b)
superlu_solve!(F, x)
```

### LinearSolve.jl Integration

For a more convenient interface, SuperLU.jl integrates with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl). This integration is provided via a package extension that loads automatically when both packages are loaded:

```julia
using SuperLU
using LinearSolve
using SparseArrays

A = sparse([1.0+1.0im 2.0+0im 0.0; 
            3.0-1.0im 4.0+2.0im 1.0+0im; 
            0.0 1.0+1.0im 5.0-1.0im])
b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]

prob = LinearProblem(A, b)
sol = solve(prob, SuperLUFactorization())
```

## Reusing Factorization

When solving multiple systems with the same sparsity pattern but different values, you can reuse the symbolic factorization for better performance:

```julia
using SuperLU
using LinearSolve
using SparseArrays

A1 = sparse([4.0+1.0im 1.0+0im 0.0; 
             1.0-1.0im 4.0+2.0im 1.0+0im; 
             0.0 1.0+1.0im 4.0-1.0im])
b1 = [1.0+0im, 2.0+1.0im, 3.0-1.0im]

# First solve with reuse_symbolic=true (default)
prob1 = LinearProblem(A1, b1)
cache = init(prob1, SuperLUFactorization(reuse_symbolic=true))
sol1 = solve!(cache)

# Update matrix values (same pattern) and solve again
A2 = sparse([5.0+2.0im 2.0+0im 0.0; 
             2.0-1.0im 5.0+1.0im 2.0+0im; 
             0.0 2.0+1.0im 5.0-2.0im])
b2 = [2.0+1.0im, 3.0-1.0im, 4.0+0im]

cache.A = A2
cache.b = b2
cache.isfresh = true
sol2 = solve!(cache)
```
