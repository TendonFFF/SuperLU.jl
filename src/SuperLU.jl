module SuperLU

using LinearAlgebra
using SparseArrays
using SuperLU_jll

# Include submodules
include("types.jl")
include("wrappers.jl")
include("interface.jl")

# Define SuperLUFactorization type here (implementation in extension)
# This allows the type to be exported and used even without LinearSolve loaded
"""
    SuperLUFactorization(; reuse_symbolic::Bool = true)

A LinearSolve.jl compatible factorization algorithm using SuperLU for sparse matrices.
Supports complex double precision (ComplexF64) matrices.

Requires loading LinearSolve.jl to use with LinearSolve's solve interface.

## Arguments
- `reuse_symbolic::Bool = true`: If `true`, the symbolic factorization from a 
  previous solve will be reused when solving with a new matrix that has the same 
  sparsity pattern. If `false`, a complete factorization is performed each time.

## Example
```julia
using SuperLU, LinearSolve, SparseArrays

A = sparse([1.0+0im 2.0; 3.0 4.0])
b = [1.0+0im, 2.0]
prob = LinearProblem(A, b)
sol = solve(prob, SuperLUFactorization())
```
"""
struct SuperLUFactorization
    reuse_symbolic::Bool
end

SuperLUFactorization(; reuse_symbolic::Bool = true) = SuperLUFactorization(reuse_symbolic)

"""
    SuperLUGPUFactorization(; reuse_symbolic::Bool = true)

A GPU-accelerated variant of SuperLU factorization for sparse matrices.
This requires a CUDA-capable GPU and the CUDA.jl package to be loaded.

!!! note "GPU Support Status"
    GPU support is currently experimental and requires CUDA.jl. When CUDA.jl is 
    loaded, the GPU-accelerated kernels will be used for the numerical factorization
    phase, while symbolic analysis is performed on the CPU.

!!! warning "Requirements"
    - CUDA.jl package must be loaded
    - NVIDIA GPU with CUDA support
    - The matrix must be on the CPU (data transfer is handled automatically)

## Arguments
- `reuse_symbolic::Bool = true`: If `true`, the symbolic factorization from a 
  previous solve will be reused when solving with a new matrix that has the same 
  sparsity pattern.

## Example
```julia
using SuperLU, CUDA
using LinearSolve, SparseArrays

A = sparse([1.0+0im 2.0; 3.0 4.0])
b = [1.0+0im, 2.0]
prob = LinearProblem(A, b)
sol = solve(prob, SuperLUGPUFactorization())  # Uses GPU acceleration
```

## Current Limitations
- GPU acceleration uses CUDA for dense operations during factorization
- The sparse matrix remains on the CPU; only intermediate dense computations benefit from GPU
- For very large matrices, the GPU speedup can be significant
"""
struct SuperLUGPUFactorization
    reuse_symbolic::Bool
end

SuperLUGPUFactorization(; reuse_symbolic::Bool = true) = SuperLUGPUFactorization(reuse_symbolic)

# GPU availability flag (set by CUDA extension)
const GPU_AVAILABLE = Ref{Bool}(false)

"""
    is_gpu_available()

Check if GPU acceleration is available for SuperLU computations.
Returns `true` if CUDA.jl is loaded and a compatible GPU is detected.
"""
is_gpu_available() = GPU_AVAILABLE[]

# Export main types and functions
export SuperLUFactorization, SuperLUGPUFactorization
export SuperLUFactorize, factorize!, superlu_solve!, update_matrix!
export is_gpu_available

end # module
