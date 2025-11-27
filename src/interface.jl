# High-level Julia interface for SuperLU

"""
    SuperLUFactorize

A mutable struct that holds the LU factorization of a sparse matrix using SuperLU.
This includes the L and U factors, permutation vectors, and other data needed
for solving linear systems.
"""
mutable struct SuperLUFactorize{Tv<:Complex}
    # Original matrix info
    n::Int
    nnz::Int
    
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
    
    # Global LU data
    Glu::GlobalLU_t
    
    # Keep references to prevent GC
    nzval_ref::Vector{Tv}
    rowind_ref::Vector{Cint}
    colptr_ref::Vector{Cint}
    
    # Flag to track if factorization is complete
    factorized::Bool
    symbolic_done::Bool
    
    function SuperLUFactorize{Tv}(A::SparseMatrixCSC{Tv, Ti}) where {Tv<:Complex, Ti<:Integer}
        n = size(A, 1)
        m = size(A, 2)
        n != m && throw(DimensionMismatch("Matrix must be square"))
        
        # Convert to 0-based indexing for C
        nzval = copy(A.nzval)
        rowind = Vector{Cint}(A.rowval .- 1)
        colptr = Vector{Cint}(A.colptr .- 1)
        nnz = length(nzval)
        
        # Create SuperMatrix A
        A_mat = SuperMatrix()
        zCreate_CompCol_Matrix!(A_mat, Cint(m), Cint(n), Cint(nnz),
                                pointer(nzval), pointer(rowind), pointer(colptr),
                                SLU_NC, SLU_Z, SLU_GE)
        
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
        
        # Set default options
        options = set_default_options()
        
        # Global LU
        Glu = GlobalLU_t()
        
        new{Tv}(n, nnz, A_mat, L_mat, U_mat, 
                perm_c, perm_r, etree, R, C, equed,
                options, Glu,
                nzval, rowind, colptr,
                false, false)
    end
end

# Constructor for ComplexF64 (default)
SuperLUFactorize(A::SparseMatrixCSC{ComplexF64, Ti}) where Ti = SuperLUFactorize{ComplexF64}(A)

"""
    factorize!(F::SuperLUFactorize)

Perform LU factorization using SuperLU.
"""
function factorize!(F::SuperLUFactorize{Tv}) where Tv
    # Initialize statistics
    stat = SuperLUStat_t()
    StatInit!(stat)
    
    # Set options - always do full factorization with simple driver
    F.options.Fact = DOFACT
    F.options.PrintStat = NO  # Disable printing
    
    # Create dummy B matrix for zgssv (we only want factorization)
    x = zeros(Tv, F.n)
    B_mat = SuperMatrix()
    zCreate_Dense_Matrix!(B_mat, Cint(F.n), Cint(1), pointer(x), Cint(F.n),
                          SLU_DN, SLU_Z, SLU_GE)
    
    # Perform factorization
    info = Ref{Cint}(0)
    zgssv!(F.options, F.A, pointer(F.perm_c), pointer(F.perm_r),
           F.L, F.U, B_mat, stat, info)
    
    # Clean up statistics
    StatFree!(stat)
    
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

"""
    superlu_solve!(F::SuperLUFactorize, b::AbstractVector; trans::trans_t=NOTRANS)

Solve the linear system using a previously computed LU factorization.
The solution overwrites `b`.
"""
function superlu_solve!(F::SuperLUFactorize{Tv}, b::AbstractVector{Tv}; 
                        trans::trans_t=NOTRANS) where Tv
    !F.factorized && error("Matrix not factorized. Call factorize! first.")
    length(b) != F.n && throw(DimensionMismatch("RHS vector length mismatch"))
    
    # Create B matrix
    B_mat = SuperMatrix()
    zCreate_Dense_Matrix!(B_mat, Cint(F.n), Cint(1), pointer(b), Cint(F.n),
                          SLU_DN, SLU_Z, SLU_GE)
    
    # Initialize statistics
    stat = SuperLUStat_t()
    StatInit!(stat)
    
    # Solve
    info = Ref{Cint}(0)
    zgstrs!(trans, F.L, F.U, pointer(F.perm_c), pointer(F.perm_r),
            B_mat, stat, info)
    
    # Clean up
    StatFree!(stat)
    
    if info[] != 0
        error("SuperLU solve failed with info = $(info[])")
    end
    
    return b
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
    
    # Create new SuperMatrix A with updated values
    F.A = SuperMatrix()
    zCreate_CompCol_Matrix!(F.A, Cint(F.n), Cint(F.n), Cint(F.nnz),
                            pointer(nzval), pointer(rowind), pointer(colptr),
                            SLU_NC, SLU_Z, SLU_GE)
    
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
