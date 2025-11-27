# SuperLU.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://TendonFFF.github.io/SuperLU.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://TendonFFF.github.io/SuperLU.jl/dev)

A Julia interface for [SuperLU](https://portal.nersc.gov/project/sparse/superlu/), a library for direct solution of large, sparse, non-symmetric systems of linear equations.

## Features

- Complex double precision (ComplexF64) sparse matrix solver
- Integration with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) via package extension
- Support for reusing factorization objects with updated matrix values
- **GPU acceleration** via CUDA.jl extension (experimental)

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

## GPU Support (Experimental)

SuperLU.jl provides experimental GPU acceleration through the CUDA.jl package extension.
When CUDA.jl is loaded and a compatible GPU is available, GPU-accelerated operations
can be used for improved performance on large sparse systems.

### Enabling GPU Support

```julia
using SuperLU
using CUDA  # Load CUDA.jl to enable GPU acceleration

# Check if GPU acceleration is available
if is_gpu_available()
    println("GPU acceleration is enabled!")
end
```

### GPU Factorization

```julia
using SuperLU
using CUDA
using SparseArrays

A = sparse([1.0+1.0im 2.0+0im 0.0; 
            3.0-1.0im 4.0+2.0im 1.0+0im; 
            0.0 1.0+1.0im 5.0-1.0im])
b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]

# Create GPU-accelerated factorization
F = SuperLUGPUFactorize(A)
factorize!(F)
x = copy(b)
superlu_solve!(F, x)
```

### GPU Support with LinearSolve.jl

```julia
using SuperLU
using CUDA
using LinearSolve
using SparseArrays

A = sparse([1.0+1.0im 2.0+0im 0.0; 
            3.0-1.0im 4.0+2.0im 1.0+0im; 
            0.0 1.0+1.0im 5.0-1.0im])
b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]

prob = LinearProblem(A, b)
sol = solve(prob, SuperLUGPUFactorization())
```

### GPU Support Status

The current GPU implementation provides:

- **CUDA-accelerated BLAS operations**: Dense matrix operations during factorization
  benefit from GPU acceleration through CUDA's cuBLAS library.
- **Hybrid CPU-GPU execution**: Symbolic factorization is performed on the CPU,
  while numerical computations can leverage GPU acceleration.

**Planned Features:**
- Full GPU-native sparse factorization (pending GPU-enabled SuperLU_DIST binaries)
- Support for matrices stored directly on GPU memory
- Multi-GPU support for very large matrices

## API Reference

### Types

- `SuperLUFactorization(; reuse_symbolic::Bool = true)`: LinearSolve.jl compatible factorization algorithm (requires LinearSolve.jl to be loaded)
- `SuperLUGPUFactorization(; reuse_symbolic::Bool = true)`: GPU-accelerated factorization (requires CUDA.jl to be loaded)
- `SuperLUFactorize{Tv}`: Internal factorization object
- `SuperLUGPUFactorize{Tv}`: GPU-accelerated internal factorization object (requires CUDA.jl)

### Functions

- `factorize!(F::SuperLUFactorize)`: Perform LU factorization
- `superlu_solve!(F::SuperLUFactorize, b::Vector)`: Solve the system, overwriting b with the solution
- `superlu_solve(F::SuperLUFactorize, b::Vector)`: Solve the system, returning a new solution vector
- `update_matrix!(F::SuperLUFactorize, A::SparseMatrixCSC)`: Update matrix values (same pattern required)
- `is_gpu_available()`: Check if GPU acceleration is available

## Dependencies

This package uses [SuperLU_jll](https://github.com/JuliaBinaryWrappers/SuperLU_jll.jl) for pre-built SuperLU binaries.

The LinearSolve.jl integration is loaded automatically when both packages are available (via Julia's package extension mechanism).

GPU acceleration requires [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and is loaded automatically when CUDA.jl is available.

## License

This package is licensed under the MIT License.