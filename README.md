# SuperLU.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://TendonFFF.github.io/SuperLU.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://TendonFFF.github.io/SuperLU.jl/dev)

A Julia interface for [SuperLU](https://portal.nersc.gov/project/sparse/superlu/), a library for direct solution of large, sparse, non-symmetric systems of linear equations.

## Features

- **Multiple precision support**: Float32, Float64, ComplexF32, and ComplexF64 sparse matrices
- Integration with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) via package extension
- Support for reusing factorization objects with updated matrix values
- **Preset options** for common scenarios (ill-conditioned systems, performance, accuracy, symmetric matrices)
- **Symmetry checking utilities** for matrix analysis and algorithm selection

## Installation

```julia
using Pkg
Pkg.add("SuperLU")
```

## Usage

### Direct API

```julia
using SuperLU
using SparseArrays

# Real double precision matrix
A = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
b = [1.0, 2.0, 3.0]

F = SuperLUFactorize(A)
factorize!(F)
x = copy(b)
superlu_solve!(F, x)

# Complex double precision matrix
A_complex = sparse([1.0+1.0im 2.0+0im 0.0; 
                    3.0-1.0im 4.0+2.0im 1.0+0im; 
                    0.0 1.0+1.0im 5.0-1.0im])
b_complex = [1.0+0im, 2.0+1.0im, 3.0-1.0im]

F_complex = SuperLUFactorize(A_complex)
factorize!(F_complex)
x_complex = copy(b_complex)
superlu_solve!(F_complex, x_complex)
```

### LinearSolve.jl Integration

The LinearSolve.jl integration is provided via a package extension that loads automatically when both `SuperLU` and `LinearSolve` are loaded:

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

### Reusing Factorization with Updated Matrix Values

The `SuperLUFactorization` type accepts a `reuse_symbolic` parameter that controls whether to reuse the factorization object when solving with updated matrix values (same sparsity pattern):

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

## Preset Options

SuperLU.jl provides pre-configured option objects for common use cases:

```julia
using SuperLU

# For ill-conditioned systems
F = SuperLUFactorize(A; options=ILL_CONDITIONED_OPTIONS)

# For maximum performance (well-conditioned systems only)
F = SuperLUFactorize(A; options=PERFORMANCE_OPTIONS)

# For maximum accuracy
F = SuperLUFactorize(A; options=ACCURACY_OPTIONS)

# For symmetric or nearly symmetric matrices
F = SuperLUFactorize(A; options=SYMMETRIC_OPTIONS)
```

## Matrix Symmetry Checking

SuperLU.jl provides utilities to analyze matrix symmetry:

```julia
using SuperLU
using SparseArrays

A = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])

# Check sparsity pattern symmetry
issymmetric_structure(A)  # true

# Check value symmetry
issymmetric_approx(A)  # true

# Get suggested options based on matrix analysis
opts = suggest_options(A)
```

## API Reference

### Types

- `SuperLUFactorization(; reuse_symbolic::Bool = true)`: LinearSolve.jl compatible factorization algorithm (requires LinearSolve.jl to be loaded)
- `SuperLUFactorize{Tv}`: Internal factorization object (Tv can be Float32, Float64, ComplexF32, or ComplexF64)

### Preset Options

- `ILL_CONDITIONED_OPTIONS`: Optimized for ill-conditioned systems
- `PERFORMANCE_OPTIONS`: Optimized for maximum performance
- `ACCURACY_OPTIONS`: Optimized for maximum accuracy
- `SYMMETRIC_OPTIONS`: Optimized for symmetric matrices

### Functions

- `factorize!(F::SuperLUFactorize)`: Perform LU factorization
- `superlu_solve!(F::SuperLUFactorize, b::Vector)`: Solve the system, overwriting b with the solution
- `superlu_solve(F::SuperLUFactorize, b::Vector)`: Solve the system, returning a new solution vector
- `update_matrix!(F::SuperLUFactorize, A::SparseMatrixCSC)`: Update matrix values (same pattern required)
- `issymmetric_structure(A)`: Check if matrix has symmetric sparsity pattern
- `issymmetric_approx(A)`: Check if real matrix is approximately symmetric
- `ishermitian_approx(A)`: Check if complex matrix is approximately Hermitian
- `suggest_options(A)`: Analyze matrix and suggest appropriate solver options

## GPU and Distributed Computing

For GPU acceleration and distributed memory parallelism, consider using [SuperLUDIST.jl](https://github.com/JuliaSparse/SuperLUDIST.jl), which wraps the SuperLU_DIST library:

```julia
using Pkg
Pkg.add("SuperLUDIST")
```

SuperLU_DIST is particularly useful for:
- Very large sparse systems (N > 50,000)
- Multi-GPU workstations (e.g., servers with NVIDIA L40S GPUs)
- HPC cluster deployments

See the [GPU and Distributed Computing documentation](https://TendonFFF.github.io/SuperLU.jl/stable/gpu_distributed/) for details.

## Dependencies

This package uses [SuperLU_jll](https://github.com/JuliaBinaryWrappers/SuperLU_jll.jl) for pre-built SuperLU binaries.

The LinearSolve.jl integration is loaded automatically when both packages are available (via Julia's package extension mechanism).

## License

This package is licensed under the MIT License.