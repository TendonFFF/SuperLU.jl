# SuperLU.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://TendonFFF.github.io/SuperLU.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://TendonFFF.github.io/SuperLU.jl/dev)

A Julia interface for [SuperLU_MT](https://portal.nersc.gov/project/sparse/superlu/), a library for direct solution of large, sparse, non-symmetric systems of linear equations using **multi-threaded** LU factorization.

## Features

- **Multi-threaded factorization**: Parallel LU factorization using multiple threads via SuperLU_MT
- **Multiple precision support**: Float32, Float64, ComplexF32, and ComplexF64 sparse matrices
- Integration with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
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

# Real double precision matrix with 4 threads
A = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
b = [1.0, 2.0, 3.0]

F = SuperLUFactorize(A; nthreads=4)
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

# Use 4 threads for factorization
sol = solve(prob, SuperLUFactorization(nthreads=4))
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

## Multi-threading

SuperLU_MT supports parallel LU factorization using multiple threads. The number of threads can be specified via the `nthreads` parameter:

```julia
# Direct API
F = SuperLUFactorize(A; nthreads=4)

# LinearSolve.jl
sol = solve(prob, SuperLUFactorization(nthreads=4))
```

By default, `nthreads=1` is used. For large sparse matrices, using multiple threads can significantly improve factorization performance.

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

## Custom Options

You can customize the solver behavior using `SuperLUOptions`:

```julia
using SuperLU

opts = SuperLUOptions(
    col_perm = MMD_AT_PLUS_A,     # Column permutation strategy
    diag_pivot_thresh = 1.0,      # Diagonal pivot threshold (0.0 to 1.0)
    symmetric_mode = true,        # Enable symmetric mode
    print_stats = false           # Print statistics
)

F = SuperLUFactorize(A; options=opts, nthreads=4)
```

### Column Permutation Strategies

- `NATURAL`: Natural ordering (no permutation)
- `MMD_ATA`: Minimum degree ordering on A'A
- `MMD_AT_PLUS_A`: Minimum degree ordering on A'+A
- `COLAMD`: Column approximate minimum degree (default)

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

- `SuperLUFactorization(; reuse_symbolic::Bool = true, nthreads::Int = 1)`: LinearSolve.jl compatible factorization algorithm
- `SuperLUFactorize{Tv}`: Internal factorization object (Tv can be Float32, Float64, ComplexF32, or ComplexF64)
- `SuperLUOptions`: Solver options configuration

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

## Dependencies

This package uses [SuperLU_MT_jll](https://github.com/JuliaBinaryWrappers/SuperLU_MT_jll.jl) for pre-built SuperLU_MT binaries with multi-threading support.

The LinearSolve.jl integration is loaded automatically when both packages are available.

## License

This package is licensed under the MIT License.