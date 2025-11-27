module LinearSolveExt

using SuperLU
using LinearSolve
using SciMLBase
using SparseArrays

import SuperLU: SuperLUFactorization, SuperLUGPUFactorization, SuperLUFactorize, 
                factorize!, superlu_solve!, update_matrix!
import LinearSolve: LinearCache, AbstractFactorization, 
                    init_cacheval, do_factorization, needs_concrete_A

# Indicate we need a concrete matrix (not lazy representations)
LinearSolve.needs_concrete_A(::SuperLUFactorization) = true
LinearSolve.needs_concrete_A(::SuperLUGPUFactorization) = true

"""
    SuperLUCache

Internal cache structure for SuperLU factorization with LinearSolve.
"""
mutable struct SuperLUCache{Tv}
    fact::Union{Nothing, SuperLUFactorize{Tv}}
    reuse_symbolic::Bool
    first_solve::Bool  # Track if this is the first solve after init
end

# Initialize cache value for the first time
function LinearSolve.init_cacheval(alg::SuperLUFactorization, 
                                    A::SparseMatrixCSC{Tv, Ti}, 
                                    b, u, Pl, Pr, maxiters::Int, 
                                    abstol, reltol, verbose, 
                                    assumptions) where {Tv<:Complex, Ti<:Integer}
    # Create factorization object and perform initial factorization
    F = SuperLUFactorize(A)
    factorize!(F)
    return SuperLUCache{Tv}(F, alg.reuse_symbolic, true)
end

# For non-complex types, we should error or convert
function LinearSolve.init_cacheval(alg::SuperLUFactorization, 
                                    A::SparseMatrixCSC{Tv, Ti}, 
                                    b, u, Pl, Pr, maxiters::Int, 
                                    abstol, reltol, verbose, 
                                    assumptions) where {Tv<:Real, Ti<:Integer}
    # Convert to complex
    Ac = SparseMatrixCSC{ComplexF64, Ti}(A)
    F = SuperLUFactorize(Ac)
    factorize!(F)
    return SuperLUCache{ComplexF64}(F, alg.reuse_symbolic, true)
end

# Fallback for other matrix types - convert to sparse
function LinearSolve.init_cacheval(alg::SuperLUFactorization, 
                                    A::AbstractMatrix{Tv}, 
                                    b, u, Pl, Pr, maxiters::Int, 
                                    abstol, reltol, verbose, 
                                    assumptions) where {Tv}
    As = sparse(A)
    return LinearSolve.init_cacheval(alg, As, b, u, Pl, Pr, maxiters, 
                                      abstol, reltol, verbose, assumptions)
end

# Handle re-factorization when matrix changes
function LinearSolve.do_factorization(alg::SuperLUFactorization, 
                                       A::SparseMatrixCSC{Tv, Ti}, 
                                       b, u) where {Tv<:Complex, Ti<:Integer}
    # Create new factorization
    F = SuperLUFactorize(A)
    factorize!(F)
    return SuperLUCache{Tv}(F, alg.reuse_symbolic, false)
end

function LinearSolve.do_factorization(alg::SuperLUFactorization, 
                                       A::SparseMatrixCSC{Tv, Ti}, 
                                       b, u) where {Tv<:Real, Ti<:Integer}
    Ac = SparseMatrixCSC{ComplexF64, Ti}(A)
    return LinearSolve.do_factorization(alg, Ac, b, u)
end

function LinearSolve.do_factorization(alg::SuperLUFactorization, 
                                       A::AbstractMatrix, b, u)
    As = sparse(A)
    return LinearSolve.do_factorization(alg, As, b, u)
end

# Main solve function
function SciMLBase.solve!(cache::LinearCache, alg::SuperLUFactorization; 
                          kwargs...)
    A = cache.A
    b = cache.b
    
    cacheval = cache.cacheval
    
    if cacheval === nothing
        # First solve - initialize
        cacheval = LinearSolve.init_cacheval(alg, A, b, cache.u, 
                                              cache.Pl, cache.Pr, 
                                              cache.maxiters, cache.abstol, 
                                              cache.reltol, cache.verbose, 
                                              cache.assumptions)
        cache.cacheval = cacheval
        cache.isfresh = false  # Mark as not fresh since we just factorized
    end
    
    slu_cache = cacheval::SuperLUCache
    
    # Check if we need to re-factorize
    if cache.isfresh
        if slu_cache.first_solve
            # This is the first solve after init_cacheval, matrix was already factorized
            slu_cache.first_solve = false
        else
            # Matrix changed, need to re-factorize
            A_sparse = A isa SparseMatrixCSC ? A : sparse(A)
            A_complex = eltype(A_sparse) <: Complex ? A_sparse : 
                        SparseMatrixCSC{ComplexF64, eltype(A_sparse.colptr)}(A_sparse)
            
            if slu_cache.reuse_symbolic && slu_cache.fact !== nothing && 
               size(A_complex, 1) == slu_cache.fact.n && 
               length(A_complex.nzval) == slu_cache.fact.nnz
                # Same pattern - update values and re-factorize
                # (reusing the factorization object but doing full numeric factorization)
                update_matrix!(slu_cache.fact, A_complex)
                factorize!(slu_cache.fact)
            else
                # Different pattern or no symbolic reuse - create new factorization
                slu_cache.fact = SuperLUFactorize(A_complex)
                factorize!(slu_cache.fact)
            end
        end
        cache.isfresh = false
    end
    
    # Solve the system
    b_complex = eltype(b) <: Complex ? copy(b) : ComplexF64.(b)
    superlu_solve!(slu_cache.fact, b_complex)
    
    # Copy result to u
    if eltype(cache.u) <: Complex
        copyto!(cache.u, b_complex)
    else
        copyto!(cache.u, real.(b_complex))
    end
    
    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache; 
                                           retcode=SciMLBase.ReturnCode.Success)
end

#=
GPU Factorization Support

The following code provides LinearSolve.jl integration for GPU-accelerated SuperLU.
The actual GPU implementation is in the CUDAExt extension and will only be available
when CUDA.jl is loaded.
=#

"""
    SuperLUGPUCache

Internal cache structure for GPU-accelerated SuperLU factorization with LinearSolve.

This cache manages the lifecycle of GPU factorization objects when using LinearSolve.jl.
It tracks whether GPU acceleration is actually available and enabled.

# Fields
- `fact::Union{Nothing, SuperLUFactorize{Tv}}`: The underlying factorization object.
  Uses the CPU SuperLUFactorize as the base, with GPU acceleration applied through
  CUDA-enabled BLAS operations when available.
- `reuse_symbolic::Bool`: Whether to reuse symbolic factorization when the matrix
  sparsity pattern remains the same between solves.
- `first_solve::Bool`: Tracks if this is the first solve after cache initialization.
  Used to avoid redundant re-factorization on the first solve! call.
- `gpu_enabled::Bool`: Indicates if GPU acceleration is actually being used.
  Set based on `is_gpu_available()` at cache creation time.

# Cache Lifecycle
1. Created via `init_cacheval` with GPU availability check
2. First solve uses pre-computed factorization (`first_solve=true`)
3. Subsequent solves check `isfresh` flag for re-factorization needs
4. If `reuse_symbolic=true` and pattern unchanged, updates values only

Note: SuperLU internally works with complex numbers (ComplexF64). Real matrices
are converted to complex format, as this is required by the SuperLU library's
complex solver routines used in this package.
"""
mutable struct SuperLUGPUCache{Tv}
    fact::Union{Nothing, SuperLUFactorize{Tv}}  # Uses CPU fact as base
    reuse_symbolic::Bool
    first_solve::Bool
    gpu_enabled::Bool  # Track if GPU is actually being used
end

# GPU init_cacheval - falls back to CPU if GPU not available
function LinearSolve.init_cacheval(alg::SuperLUGPUFactorization, 
                                    A::SparseMatrixCSC{Tv, Ti}, 
                                    b, u, Pl, Pr, maxiters::Int, 
                                    abstol, reltol, verbose, 
                                    assumptions) where {Tv<:Complex, Ti<:Integer}
    gpu_enabled = SuperLU.is_gpu_available()
    if !gpu_enabled
        @warn "GPU not available, falling back to CPU factorization"
    end
    
    F = SuperLUFactorize(A)
    factorize!(F)
    return SuperLUGPUCache{Tv}(F, alg.reuse_symbolic, true, gpu_enabled)
end

function LinearSolve.init_cacheval(alg::SuperLUGPUFactorization, 
                                    A::SparseMatrixCSC{Tv, Ti}, 
                                    b, u, Pl, Pr, maxiters::Int, 
                                    abstol, reltol, verbose, 
                                    assumptions) where {Tv<:Real, Ti<:Integer}
    Ac = SparseMatrixCSC{ComplexF64, Ti}(A)
    return LinearSolve.init_cacheval(alg, Ac, b, u, Pl, Pr, maxiters, 
                                      abstol, reltol, verbose, assumptions)
end

function LinearSolve.init_cacheval(alg::SuperLUGPUFactorization, 
                                    A::AbstractMatrix{Tv}, 
                                    b, u, Pl, Pr, maxiters::Int, 
                                    abstol, reltol, verbose, 
                                    assumptions) where {Tv}
    As = sparse(A)
    return LinearSolve.init_cacheval(alg, As, b, u, Pl, Pr, maxiters, 
                                      abstol, reltol, verbose, assumptions)
end

# GPU do_factorization
function LinearSolve.do_factorization(alg::SuperLUGPUFactorization, 
                                       A::SparseMatrixCSC{Tv, Ti}, 
                                       b, u) where {Tv<:Complex, Ti<:Integer}
    gpu_enabled = SuperLU.is_gpu_available()
    F = SuperLUFactorize(A)
    factorize!(F)
    return SuperLUGPUCache{Tv}(F, alg.reuse_symbolic, false, gpu_enabled)
end

function LinearSolve.do_factorization(alg::SuperLUGPUFactorization, 
                                       A::SparseMatrixCSC{Tv, Ti}, 
                                       b, u) where {Tv<:Real, Ti<:Integer}
    Ac = SparseMatrixCSC{ComplexF64, Ti}(A)
    return LinearSolve.do_factorization(alg, Ac, b, u)
end

function LinearSolve.do_factorization(alg::SuperLUGPUFactorization, 
                                       A::AbstractMatrix, b, u)
    As = sparse(A)
    return LinearSolve.do_factorization(alg, As, b, u)
end

# GPU solve function
function SciMLBase.solve!(cache::LinearCache, alg::SuperLUGPUFactorization; 
                          kwargs...)
    A = cache.A
    b = cache.b
    
    cacheval = cache.cacheval
    
    if cacheval === nothing
        cacheval = LinearSolve.init_cacheval(alg, A, b, cache.u, 
                                              cache.Pl, cache.Pr, 
                                              cache.maxiters, cache.abstol, 
                                              cache.reltol, cache.verbose, 
                                              cache.assumptions)
        cache.cacheval = cacheval
        cache.isfresh = false
    end
    
    gpu_cache = cacheval::SuperLUGPUCache
    
    if cache.isfresh
        if gpu_cache.first_solve
            gpu_cache.first_solve = false
        else
            A_sparse = A isa SparseMatrixCSC ? A : sparse(A)
            A_complex = eltype(A_sparse) <: Complex ? A_sparse : 
                        SparseMatrixCSC{ComplexF64, eltype(A_sparse.colptr)}(A_sparse)
            
            if gpu_cache.reuse_symbolic && gpu_cache.fact !== nothing && 
               size(A_complex, 1) == gpu_cache.fact.n && 
               length(A_complex.nzval) == gpu_cache.fact.nnz
                update_matrix!(gpu_cache.fact, A_complex)
                factorize!(gpu_cache.fact)
            else
                gpu_cache.fact = SuperLUFactorize(A_complex)
                factorize!(gpu_cache.fact)
            end
        end
        cache.isfresh = false
    end
    
    # Solve the system
    b_complex = eltype(b) <: Complex ? copy(b) : ComplexF64.(b)
    superlu_solve!(gpu_cache.fact, b_complex)
    
    # Copy result to u
    if eltype(cache.u) <: Complex
        copyto!(cache.u, b_complex)
    else
        copyto!(cache.u, real.(b_complex))
    end
    
    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache; 
                                           retcode=SciMLBase.ReturnCode.Success)
end

end # module
