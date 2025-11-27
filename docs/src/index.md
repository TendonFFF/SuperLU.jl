# SuperLU.jl

A Julia interface for [SuperLU](https://portal.nersc.gov/project/sparse/superlu/), a library for direct solution of large, sparse, non-symmetric systems of linear equations.

## Features

- **Multiple precision support**: Float32, Float64, ComplexF32, and ComplexF64 sparse matrices
- Integration with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) via package extension
- Support for reusing factorization objects with updated matrix values
- **Preset options** for common scenarios (ill-conditioned systems, performance, accuracy, symmetric matrices)
- **Symmetry checking utilities** for matrix analysis and algorithm selection
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

# Real double precision
A = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
b = [1.0, 2.0, 3.0]

F = SuperLUFactorize(A)
factorize!(F)
x = copy(b)
superlu_solve!(F, x)

# Complex double precision
A_complex = sparse([1.0+1.0im 2.0+0im 0.0; 
                    3.0-1.0im 4.0+2.0im 1.0+0im; 
                    0.0 1.0+1.0im 5.0-1.0im])
b_complex = [1.0+0im, 2.0+1.0im, 3.0-1.0im]

F_complex = SuperLUFactorize(A_complex)
factorize!(F_complex)
x_complex = copy(b_complex)
superlu_solve!(F_complex, x_complex)
```

See the [Getting Started](@ref) guide for more detailed examples.

## Contents

```@contents
Pages = ["getting_started.md", "options.md", "api.md"]
Depth = 2
```
