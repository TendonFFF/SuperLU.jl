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

## GPU Support (Experimental)

SuperLU.jl provides experimental GPU acceleration through the CUDA.jl package extension. When CUDA.jl is loaded and a compatible GPU is available, GPU-accelerated operations can be used for improved performance on large sparse systems.

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

- **CUDA-accelerated BLAS operations**: Dense matrix operations during factorization benefit from GPU acceleration through CUDA's cuBLAS library.
- **Hybrid CPU-GPU execution**: Symbolic factorization is performed on the CPU, while numerical computations can leverage GPU acceleration.

**Planned Features:**
- Full GPU-native sparse factorization (pending GPU-enabled SuperLU_DIST binaries)
- Support for matrices stored directly on GPU memory
- Multi-GPU support for very large matrices
