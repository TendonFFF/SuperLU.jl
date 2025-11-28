# High-level Julia interface for SuperLU

import LinearAlgebra.BLAS

"""
    with_single_threaded_blas(f, nthreads::Int)

Execute function `f` with BLAS set to single-threaded mode if `nthreads > 1`.
SuperLU_MT requires single-threaded BLAS to avoid thread conflicts when using
multiple threads for factorization.

This function saves the current BLAS thread count, sets it to 1 if needed,
executes `f`, and restores the original thread count.
"""
function with_single_threaded_blas(f, nthreads::Int)
    if nthreads > 1
        old_blas_threads = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            return f()
        finally
            BLAS.set_num_threads(old_blas_threads)
        end
    else
        return f()
    end
end

"""
    SuperLUFactorize

A mutable struct that holds the LU factorization of a sparse matrix using SuperLU.
This includes the L and U factors, permutation vectors, and other data needed
for solving linear systems.

# Supported element types
- `Float32` (single precision real)
- `Float64` (double precision real)
- `ComplexF32` (single precision complex)
- `ComplexF64` (double precision complex)

# Fields
- `n::Int`: Matrix dimension
- `nnz::Int`: Number of non-zeros
- `nthreads::Int`: Number of threads (reserved for future multi-threading support)
- `factorized::Bool`: Whether factorization has been performed
- `symbolic_done::Bool`: Whether symbolic factorization is complete

# Constructor
    SuperLUFactorize(A::SparseMatrixCSC{T}; options::SuperLUOptions=SuperLUOptions(), nthreads::Int=1)

Create a new factorization object for the sparse matrix `A`.

# Arguments
- `A`: Sparse matrix to factorize (must be square)
- `options`: Solver options (see [`SuperLUOptions`](@ref))
- `nthreads`: Number of threads for factorization (default: 1).
  Currently, only sequential factorization is available. When `nthreads > 1`,
  BLAS is set to single-threaded mode during operations, but the factorization
  itself still uses sequential SuperLU. Full parallel factorization requires
  SuperLU_MT integration which uses a different API (pdgssv/pdgssvx).

# Example
```julia
using SuperLU, SparseArrays

# Complex double precision
A = sparse([1.0+1.0im 2.0+0im; 3.0-1.0im 4.0+2.0im])
F = SuperLUFactorize(A)
factorize!(F)
x = superlu_solve(F, [1.0+0im, 2.0+1.0im])

# Real double precision
A_real = sparse([4.0 1.0; 1.0 3.0])
F_real = SuperLUFactorize(A_real)
factorize!(F_real)
x_real = superlu_solve(F_real, [1.0, 2.0])
```

See also: [`SuperLUOptions`](@ref), [`factorize!`](@ref), [`superlu_solve!`](@ref)
"""
mutable struct SuperLUFactorize{Tv<:SuperLUTypes}
    # Original matrix info
    n::Int
    nnz::Int
    
    # Threading configuration (reserved for future MT support)
    nthreads::Int
    
    # SuperLU matrices (mutable structs for C interop)
    A::SuperMatrix
    L::SuperMatrix
    U::SuperMatrix
    
    # Permutation arrays
    perm_c::Vector{Cint}
    perm_r::Vector{Cint}
    etree::Vector{Cint}
    
    # Scaling factors
    R::Vector{Cdouble}
    C::Vector{Cdouble}
    equed::Vector{Cchar}
    
    # Options
    options::superlu_options_t
    
    # User options (for reference)
    user_options::SuperLUOptions
    
    # Global LU data
    Glu::GlobalLU_t
    
    # Keep references to prevent GC
    nzval_ref::Vector{Tv}
    rowind_ref::Vector{Cint}
    colptr_ref::Vector{Cint}
    
    # Flag to track if factorization is complete
    factorized::Bool
    symbolic_done::Bool
    
    function SuperLUFactorize{Tv}(A::SparseMatrixCSC{Tv, Ti}; 
                                   options::SuperLUOptions=SuperLUOptions(),
                                   nthreads::Int=1) where {Tv<:SuperLUTypes, Ti<:Integer}
        n = size(A, 1)
        m = size(A, 2)
        n != m && throw(DimensionMismatch("Matrix must be square"))
        nthreads < 1 && throw(ArgumentError("nthreads must be at least 1, got $nthreads"))
        
        # Warn about multi-threading limitations
        # Note: SuperLU_MT (the multi-threaded library) has a different API than SuperLU.
        # Currently we only use SuperLU_jll which provides sequential SuperLU.
        # To fully support multi-threading, we would need to use SuperLU_MT_jll and
        # implement wrappers for pdgssv/pdgssvx which take nprocs as first argument.
        if nthreads > 1
            @warn "Multi-threaded factorization (nthreads=$nthreads) is not yet implemented. The current implementation uses sequential SuperLU. BLAS threads will be set to 1 during operations to prepare for future SuperLU_MT integration. Full SuperLU_MT support requires implementing wrappers for the parallel pdgssv/pdgssvx functions." maxlog=1
        end
        
        # Convert to 0-based indexing for C
        nzval = copy(A.nzval)
        rowind = Vector{Cint}(A.rowval .- 1)
        colptr = Vector{Cint}(A.colptr .- 1)
        nnz = length(nzval)
        
        # Create SuperMatrix A using type-specific wrapper
        A_mat = SuperMatrix()
        dtype = slu_dtype(Tv)
        Create_CompCol_Matrix!(A_mat, Cint(m), Cint(n), Cint(nnz),
                               pointer(nzval), pointer(rowind), pointer(colptr),
                               SLU_NC, dtype, SLU_GE)
        
        # Initialize L and U
        L_mat = SuperMatrix()
        U_mat = SuperMatrix()
        
        # Allocate permutation arrays
        perm_c = zeros(Cint, n)
        perm_r = zeros(Cint, n)
        etree = zeros(Cint, n)
        
        # Scaling factors
        R = ones(Cdouble, m)
        C = ones(Cdouble, n)
        equed = Cchar['N']
        
        # Set default options and apply user options
        c_options = set_default_options()
        apply_options!(c_options, options)
        
        # Global LU
        Glu = GlobalLU_t()
        
        new{Tv}(n, nnz, nthreads, A_mat, L_mat, U_mat, 
                perm_c, perm_r, etree, R, C, equed,
                c_options, options, Glu,
                nzval, rowind, colptr,
                false, false)
    end
end

# Constructors for all supported types
SuperLUFactorize(A::SparseMatrixCSC{Float32, Ti}; options::SuperLUOptions=SuperLUOptions(), nthreads::Int=1) where Ti = 
    SuperLUFactorize{Float32}(A; options=options, nthreads=nthreads)

SuperLUFactorize(A::SparseMatrixCSC{Float64, Ti}; options::SuperLUOptions=SuperLUOptions(), nthreads::Int=1) where Ti = 
    SuperLUFactorize{Float64}(A; options=options, nthreads=nthreads)

SuperLUFactorize(A::SparseMatrixCSC{ComplexF32, Ti}; options::SuperLUOptions=SuperLUOptions(), nthreads::Int=1) where Ti = 
    SuperLUFactorize{ComplexF32}(A; options=options, nthreads=nthreads)

SuperLUFactorize(A::SparseMatrixCSC{ComplexF64, Ti}; options::SuperLUOptions=SuperLUOptions(), nthreads::Int=1) where Ti = 
    SuperLUFactorize{ComplexF64}(A; options=options, nthreads=nthreads)

"""
    factorize!(F::SuperLUFactorize)

Perform LU factorization using SuperLU.

When using multi-threaded factorization (`nthreads > 1`), BLAS is temporarily
set to single-threaded mode to avoid thread conflicts with SuperLU_MT.
"""
function factorize!(F::SuperLUFactorize{Tv}) where Tv
    # Use single-threaded BLAS when nthreads > 1 to avoid thread conflicts
    with_single_threaded_blas(F.nthreads) do
        # Initialize statistics
        stat = SuperLUStat_t()
        StatInit!(stat)
        
        # Set options - always do full factorization with simple driver
        F.options.Fact = DOFACT
        F.options.PrintStat = NO  # Disable printing
        
        # Create dummy B matrix for gssv (we only want factorization)
        x = zeros(Tv, F.n)
        B_mat = SuperMatrix()
        dtype = slu_dtype(Tv)
        Create_Dense_Matrix!(B_mat, Cint(F.n), Cint(1), pointer(x), Cint(F.n),
                             SLU_DN, dtype, SLU_GE)
        
        # Perform factorization using type-generic wrapper
        info = Ref{Cint}(0)
        try
            gssv!(F.options, F.A, pointer(F.perm_c), pointer(F.perm_r),
                  F.L, F.U, B_mat, stat, info, Tv)
        finally
            # Clean up temporary B matrix
            Destroy_SuperMatrix_Store!(B_mat)
            # Clean up statistics
            StatFree!(stat)
        end
        
        if info[] != 0
            if info[] < 0
                error("SuperLU: illegal argument at position $(abs(info[]))")
            else
                error("SuperLU: singular matrix, U($(info[]),$(info[])) is exactly zero")
            end
        end
        
        F.factorized = true
        F.symbolic_done = true
        
        return F
    end
end

"""
    superlu_solve!(F::SuperLUFactorize, b::AbstractVector; trans::trans_t=NOTRANS)

Solve the linear system using a previously computed LU factorization.
The solution overwrites `b`.

When using multi-threaded factorization (`nthreads > 1`), BLAS is temporarily
set to single-threaded mode to avoid thread conflicts with SuperLU_MT.
"""
function superlu_solve!(F::SuperLUFactorize{Tv}, b::AbstractVector{Tv}; 
                        trans::trans_t=NOTRANS) where Tv
    !F.factorized && error("Matrix not factorized. Call factorize! first.")
    length(b) != F.n && throw(DimensionMismatch("RHS vector length mismatch"))
    
    # Use single-threaded BLAS when nthreads > 1 to avoid thread conflicts
    with_single_threaded_blas(F.nthreads) do
        # Create B matrix using type-generic wrapper
        B_mat = SuperMatrix()
        dtype = slu_dtype(Tv)
        Create_Dense_Matrix!(B_mat, Cint(F.n), Cint(1), pointer(b), Cint(F.n),
                             SLU_DN, dtype, SLU_GE)
        
        # Initialize statistics
        stat = SuperLUStat_t()
        StatInit!(stat)
        
        # Solve using type-generic wrapper
        info = Ref{Cint}(0)
        try
            gstrs!(trans, F.L, F.U, pointer(F.perm_c), pointer(F.perm_r),
                   B_mat, stat, info, Tv)
        finally
            # Clean up temporary B matrix
            Destroy_SuperMatrix_Store!(B_mat)
            # Clean up statistics
            StatFree!(stat)
        end
        
        if info[] != 0
            error("SuperLU solve failed with info = $(info[])")
        end
        
        return b
    end
end

"""
    superlu_solve(F::SuperLUFactorize, b::AbstractVector; trans::trans_t=NOTRANS)

Solve the linear system using a previously computed LU factorization.
Returns a new vector with the solution.
"""
function superlu_solve(F::SuperLUFactorize{Tv}, b::AbstractVector{Tv}; 
                       trans::trans_t=NOTRANS) where Tv
    x = copy(b)
    superlu_solve!(F, x; trans=trans)
    return x
end

"""
    update_matrix!(F::SuperLUFactorize, A::SparseMatrixCSC)

Update the matrix in an existing factorization object with new values.
The sparsity pattern must remain the same. After calling this, you must
call `factorize!` again before solving.
"""
function update_matrix!(F::SuperLUFactorize{Tv}, A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    size(A, 1) != F.n && throw(DimensionMismatch("Matrix dimension changed"))
    length(A.nzval) != F.nnz && throw(DimensionMismatch("Sparsity pattern changed (different nnz)"))
    
    # Convert to 0-based indexing for C
    nzval = copy(A.nzval)
    rowind = Vector{Cint}(A.rowval .- 1)
    colptr = Vector{Cint}(A.colptr .- 1)
    
    # Store new references
    F.nzval_ref = nzval
    F.rowind_ref = rowind
    F.colptr_ref = colptr
    
    # Create new SuperMatrix A with updated values using type-generic wrapper
    F.A = SuperMatrix()
    dtype = slu_dtype(Tv)
    Create_CompCol_Matrix!(F.A, Cint(F.n), Cint(F.n), Cint(F.nnz),
                           pointer(nzval), pointer(rowind), pointer(colptr),
                           SLU_NC, dtype, SLU_GE)
    
    # Reset L and U for new factorization
    F.L = SuperMatrix()
    F.U = SuperMatrix()
    
    # Reset permutation arrays
    fill!(F.perm_c, 0)
    fill!(F.perm_r, 0)
    fill!(F.etree, 0)
    
    # Mark as needing re-factorization
    F.factorized = false
    
    return F
end

# ============================================================================
# Matrix symmetry checking utilities
# ============================================================================

# Helper function to check if a row index exists in a column using binary search
# Returns the index if found, or 0 if not found
function _find_row_in_col(A::SparseMatrixCSC, col::Int, target_row::Int)
    start_idx = A.colptr[col]
    end_idx = A.colptr[col+1] - 1
    
    # Binary search for target_row in the sorted rowval array
    while start_idx <= end_idx
        mid = (start_idx + end_idx) ÷ 2
        mid_row = A.rowval[mid]
        if mid_row == target_row
            return mid
        elseif mid_row < target_row
            start_idx = mid + 1
        else
            end_idx = mid - 1
        end
    end
    return 0
end

"""
    issymmetric_structure(A::SparseMatrixCSC) -> Bool

Check if a sparse matrix has symmetric sparsity structure (A[i,j] ≠ 0 ⟺ A[j,i] ≠ 0).
This is useful for selecting appropriate column permutation strategies.

Note: This only checks the sparsity pattern, not the values.

Complexity: O(nnz * log(max_column_size)) due to binary search.
"""
function issymmetric_structure(A::SparseMatrixCSC)
    m, n = size(A)
    m != n && return false
    
    # For each non-zero entry (row, col), check if (col, row) also exists
    # Use binary search since row indices are sorted within each column
    for col in 1:n
        for idx in A.colptr[col]:(A.colptr[col+1]-1)
            row = A.rowval[idx]
            if row != col  # Skip diagonal
                # Check if (col, row) exists using binary search
                if _find_row_in_col(A, row, col) == 0
                    return false
                end
            end
        end
    end
    return true
end

"""
    ishermitian_approx(A::SparseMatrixCSC; rtol::Real=1e-10) -> Bool

Check if a sparse matrix is approximately Hermitian (A ≈ A').
For real matrices, this is equivalent to checking symmetry.

Returns `true` if for all (i,j): |A[i,j] - conj(A[j,i])| ≤ rtol * max(|A[i,j]|, |A[j,i]|)

Complexity: O(nnz * log(max_column_size)) due to binary search.
"""
function ishermitian_approx(A::SparseMatrixCSC{Tv}; rtol::Real=1e-10) where Tv
    m, n = size(A)
    m != n && return false
    
    for col in 1:n
        for idx in A.colptr[col]:(A.colptr[col+1]-1)
            row = A.rowval[idx]
            val = A.nzval[idx]
            
            # Find the corresponding (col, row) entry using binary search
            idx2 = _find_row_in_col(A, row, col)
            val_transpose = idx2 > 0 ? A.nzval[idx2] : zero(Tv)
            
            # Check Hermitian condition
            diff = abs(val - conj(val_transpose))
            max_val = max(abs(val), abs(val_transpose))
            if max_val > 0 && diff > rtol * max_val
                return false
            end
        end
    end
    return true
end

"""
    issymmetric_approx(A::SparseMatrixCSC{<:Real}; rtol::Real=1e-10) -> Bool

Check if a real sparse matrix is approximately symmetric.
"""
issymmetric_approx(A::SparseMatrixCSC{Tv}; rtol::Real=1e-10) where Tv<:Real = 
    ishermitian_approx(A; rtol=rtol)

"""
    suggest_options(A::SparseMatrixCSC) -> SuperLUOptions

Analyze the matrix and suggest appropriate solver options.
This function examines the matrix structure and returns options that may
provide better performance or accuracy.

# Returns
A `SuperLUOptions` object with settings tuned for the given matrix.

# Example
```julia
A = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
opts = suggest_options(A)
F = SuperLUFactorize(A; options=opts)
```
"""
function suggest_options(A::SparseMatrixCSC)
    opts = SuperLUOptions()
    
    # Check for symmetry to use more efficient ordering
    if issymmetric_structure(A)
        # Use ordering that exploits symmetric structure
        opts = SuperLUOptions(
            col_perm = MMD_AT_PLUS_A,
            symmetric_mode = true
        )
    end
    
    return opts
end
