module LinearSolveExt

using SuperLU
using LinearSolve
using SciMLBase
using SparseArrays

import SuperLU: SuperLUFactorize, SuperLUTypes,
                factorize!, superlu_solve!, update_matrix!, SuperLUOptions
import LinearSolve: LinearCache, AbstractFactorization, 
                    init_cacheval, do_factorization, needs_concrete_A,
                    LinearVerbosity, OperatorAssumptions

# Define SuperLUFactorization as a proper subtype of AbstractFactorization
@doc raw"""
    SuperLUFactorization(; reuse_symbolic::Bool = true, options::SuperLUOptions = SuperLUOptions())

A LinearSolve.jl compatible factorization algorithm using SuperLU for sparse matrices.
Supports Float32, Float64, ComplexF32, and ComplexF64 matrices.

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

# With preset options for ill-conditioned systems
sol = solve(prob, SuperLUFactorization(options = ILL_CONDITIONED_OPTIONS))
```

See also: [`SuperLUOptions`](@ref), [`SuperLUFactorize`](@ref)
"""
struct SuperLUFactorization <: AbstractFactorization
    reuse_symbolic::Bool
    options::SuperLUOptions
end

SuperLUFactorization(; reuse_symbolic::Bool = true, options::SuperLUOptions = SuperLUOptions()) = 
    SuperLUFactorization(reuse_symbolic, options)

# Indicate we need a concrete matrix (not lazy representations)
LinearSolve.needs_concrete_A(::SuperLUFactorization) = true

"""
    SuperLUCache

Internal cache structure for SuperLU factorization with LinearSolve.
"""
mutable struct SuperLUCache{Tv}
    fact::Union{Nothing, SuperLUFactorize{Tv}}
    reuse_symbolic::Bool
    options::SuperLUOptions
    first_solve::Bool  # Track if this is the first solve after init
end

# Initialize cache value for all supported SuperLU types
function LinearSolve.init_cacheval(alg::SuperLUFactorization, 
                                    A::SparseMatrixCSC{Tv, Ti}, 
                                    b, u, Pl, Pr, maxiters::Int, 
                                    abstol, reltol, 
                                    verbose::Union{Bool, LinearVerbosity},
                                    assumptions::OperatorAssumptions) where {Tv<:SuperLUTypes, Ti<:Integer}
    # Create factorization object and perform initial factorization
    F = SuperLUFactorize(A; options=alg.options)
    factorize!(F)
    return SuperLUCache{Tv}(F, alg.reuse_symbolic, alg.options, true)
end

# Fallback for other matrix types - convert to sparse
function LinearSolve.init_cacheval(alg::SuperLUFactorization, 
                                    A::AbstractMatrix{Tv}, 
                                    b, u, Pl, Pr, maxiters::Int, 
                                    abstol, reltol, 
                                    verbose::Union{Bool, LinearVerbosity},
                                    assumptions::OperatorAssumptions) where {Tv}
    As = sparse(A)
    return LinearSolve.init_cacheval(alg, As, b, u, Pl, Pr, maxiters, 
                                      abstol, reltol, verbose, assumptions)
end

# Handle re-factorization when matrix changes for all supported types
function LinearSolve.do_factorization(alg::SuperLUFactorization, 
                                       A::SparseMatrixCSC{Tv, Ti}, 
                                       b, u) where {Tv<:SuperLUTypes, Ti<:Integer}
    # Create new factorization
    F = SuperLUFactorize(A; options=alg.options)
    factorize!(F)
    return SuperLUCache{Tv}(F, alg.reuse_symbolic, alg.options, false)
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
    Tv = eltype(slu_cache.fact.nzval_ref)
    
    # Check if we need to re-factorize
    if cache.isfresh
        if slu_cache.first_solve
            # This is the first solve after init_cacheval, matrix was already factorized
            slu_cache.first_solve = false
        else
            # Matrix changed, need to re-factorize
            A_sparse = A isa SparseMatrixCSC ? A : sparse(A)
            # Convert to the same type as our factorization
            A_typed = eltype(A_sparse) == Tv ? A_sparse : 
                      SparseMatrixCSC{Tv, eltype(A_sparse.colptr)}(A_sparse)
            
            if slu_cache.reuse_symbolic && slu_cache.fact !== nothing && 
               size(A_typed, 1) == slu_cache.fact.n && 
               length(A_typed.nzval) == slu_cache.fact.nnz
                # Same pattern - update values and re-factorize
                update_matrix!(slu_cache.fact, A_typed)
                factorize!(slu_cache.fact)
            else
                # Different pattern or no symbolic reuse - create new factorization
                slu_cache.fact = SuperLUFactorize(A_typed; options=slu_cache.options)
                factorize!(slu_cache.fact)
            end
        end
        cache.isfresh = false
    end
    
    # Solve the system - convert b to the factorization type
    b_typed = eltype(b) == Tv ? copy(b) : Tv.(b)
    superlu_solve!(slu_cache.fact, b_typed)
    
    # Copy result to u, converting back to the output type
    if eltype(cache.u) == Tv
        copyto!(cache.u, b_typed)
    elseif eltype(cache.u) <: Real && Tv <: Complex
        copyto!(cache.u, real.(b_typed))
    else
        copyto!(cache.u, convert.(eltype(cache.u), b_typed))
    end
    
    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache; 
                                           retcode=SciMLBase.ReturnCode.Success)
end

# Make SuperLUFactorization available in the SuperLU module's namespace
# This allows users to use SuperLU.SuperLUFactorization() after loading LinearSolve
function __init__()
    @eval SuperLU begin
        const SuperLUFactorization = $SuperLUFactorization
        export SuperLUFactorization
    end
end

end # module
