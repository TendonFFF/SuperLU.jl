# SuperLU.jl

A Julia interface for [SuperLU](https://portal.nersc.gov/project/sparse/superlu/), a library for direct solution of large, sparse, non-symmetric systems of linear equations.

## Features

- Complex double precision (ComplexF64) sparse matrix solver
- Integration with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) via package extension
- Support for reusing factorization objects with updated matrix values
- **GPU acceleration** via CUDA.jl extension (experimental)

## Installation

The package can be installed using Julia's package manager:

```julia
using Pkg
Pkg.add("SuperLU")
```

## Quick Start

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

See the [Getting Started](@ref) guide for more detailed examples.

## Contents

```@contents
Pages = ["getting_started.md", "options.md", "api.md"]
Depth = 2
```
