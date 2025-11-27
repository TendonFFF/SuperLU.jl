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

# Export main types and functions
export SuperLUFactorization
export SuperLUFactorize, factorize!, superlu_solve!, update_matrix!

end # module
