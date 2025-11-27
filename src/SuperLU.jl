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
    SuperLUFactorization(; reuse_symbolic::Bool = true, options::SuperLUOptions = SuperLUOptions())

A LinearSolve.jl compatible factorization algorithm using SuperLU for sparse matrices.
Supports complex double precision (ComplexF64) matrices.

Requires loading LinearSolve.jl to use with LinearSolve's solve interface.

## Arguments
- `reuse_symbolic::Bool = true`: If `true`, the symbolic factorization from a 
  previous solve will be reused when solving with a new matrix that has the same 
  sparsity pattern. If `false`, a complete factorization is performed each time.
- `options::SuperLUOptions = SuperLUOptions()`: Solver configuration options.
  See [`SuperLUOptions`](@ref) for available settings.

## Example
```julia
using SuperLU, LinearSolve, SparseArrays

A = sparse([1.0+0im 2.0; 3.0 4.0])
b = [1.0+0im, 2.0]
prob = LinearProblem(A, b)
sol = solve(prob, SuperLUFactorization())

# With custom options
opts = SuperLUOptions(col_perm = METIS_AT_PLUS_A, equilibrate = true)
sol = solve(prob, SuperLUFactorization(options = opts))
```

See also: [`SuperLUOptions`](@ref), [`SuperLUFactorize`](@ref)
"""
struct SuperLUFactorization
    reuse_symbolic::Bool
    options::SuperLUOptions
end

SuperLUFactorization(; reuse_symbolic::Bool = true, options::SuperLUOptions = SuperLUOptions()) = 
    SuperLUFactorization(reuse_symbolic, options)

"""
    SuperLUGPUFactorization(; reuse_symbolic::Bool = true, options::SuperLUOptions = SuperLUOptions())

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
- `options::SuperLUOptions = SuperLUOptions()`: Solver configuration options.

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
    options::SuperLUOptions
end

SuperLUGPUFactorization(; reuse_symbolic::Bool = true, options::SuperLUOptions = SuperLUOptions()) = 
    SuperLUGPUFactorization(reuse_symbolic, options)

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
export SuperLUFactorize, factorize!, superlu_solve!, superlu_solve, update_matrix!
export is_gpu_available

# Export options
export SuperLUOptions

# Export trans_t enum for use in extensions
export trans_t, NOTRANS, TRANS, CONJ

# Export column permutation strategies
export colperm_t, NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD, METIS_AT_PLUS_A

# Export row permutation strategies
export rowperm_t, NOROWPERM, LargeDiag_MC64, LargeDiag_HWPM

# Export iterative refinement options
export IterRefine_t, NOREFINE, SLU_SINGLE, SLU_DOUBLE, SLU_EXTRA

# Export yes/no enum for checking options
export yes_no_t, YES, NO

end # module
