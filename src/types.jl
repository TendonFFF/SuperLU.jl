# SuperLU type definitions based on slu_zdefs.h and slu_util.h

# SuperLU data types for complex double precision
const doublecomplex = ComplexF64

# ============================================================================
# SuperLU Enumerations
# ============================================================================

"""
    Stype_t

Storage type enumeration for SuperLU matrices.

# Values
- `SLU_NC`: Column-wise, no supernode (compressed column format)
- `SLU_NCP`: Column-wise, column-permuted, no supernode
- `SLU_NR`: Row-wise, no supernode (compressed row format)
- `SLU_SC`: Column-wise, supernode
- `SLU_SCP`: Supernode, column-permuted
- `SLU_SR`: Row-wise, supernode
- `SLU_DN`: Fortran style column-wise storage for dense matrix
- `SLU_NR_loc`: Distributed compressed row format
"""
@enum Stype_t::Cint begin
    SLU_NC = 0       # column-wise, no supernode
    SLU_NCP = 1      # column-wise, column-permuted, no supernode
    SLU_NR = 2       # row-wise, no supernode
    SLU_SC = 3       # column-wise, supernode
    SLU_SCP = 4      # supernode, column-permuted
    SLU_SR = 5       # row-wise, supernode
    SLU_DN = 6       # Fortran style column-wise storage for dense matrix
    SLU_NR_loc = 7   # distributed compressed row format
end

"""
    Dtype_t

Data type enumeration for SuperLU matrices.

# Values
- `SLU_S`: Single precision real
- `SLU_D`: Double precision real
- `SLU_C`: Single precision complex
- `SLU_Z`: Double precision complex
"""
@enum Dtype_t::Cint begin
    SLU_S = 0        # single
    SLU_D = 1        # double
    SLU_C = 2        # single complex
    SLU_Z = 3        # double complex
end

"""
    Mtype_t

Matrix type enumeration for SuperLU.

# Values
- `SLU_GE`: General matrix
- `SLU_TRLU`: Lower triangular, unit diagonal
- `SLU_TRUU`: Upper triangular, unit diagonal
- `SLU_TRL`: Lower triangular
- `SLU_TRU`: Upper triangular
- `SLU_SYL`: Symmetric, store lower half
- `SLU_SYU`: Symmetric, store upper half
- `SLU_HEL`: Hermitian, store lower half
- `SLU_HEU`: Hermitian, store upper half
"""
@enum Mtype_t::Cint begin
    SLU_GE = 0       # general
    SLU_TRLU = 1     # lower triangular, unit diagonal
    SLU_TRUU = 2     # upper triangular, unit diagonal
    SLU_TRL = 3      # lower triangular
    SLU_TRU = 4      # upper triangular
    SLU_SYL = 5      # symmetric, store lower half
    SLU_SYU = 6      # symmetric, store upper half
    SLU_HEL = 7      # Hermitian, store lower half
    SLU_HEU = 8      # Hermitian, store upper half
end

"""
    fact_t

Factorization type enumeration for SuperLU.

# Values
- `DOFACT`: Perform fresh factorization
- `SamePattern`: Reuse column permutation from previous factorization
- `SamePattern_SameRowPerm`: Reuse both column and row permutations
- `FACTORED`: Matrix is already factored, use existing L and U
"""
@enum fact_t::Cint begin
    DOFACT = 0
    SamePattern = 1
    SamePattern_SameRowPerm = 2
    FACTORED = 3
end

"""
    yes_no_t

Boolean enumeration for SuperLU options.

# Values
- `NO`: Disable option (0)
- `YES`: Enable option (1)
"""
@enum yes_no_t::Cint begin
    NO = 0
    YES = 1
end

"""
    trans_t

Transpose type enumeration for solve operations.

# Values
- `NOTRANS`: No transpose (solve Ax = b)
- `TRANS`: Transpose (solve Aᵀx = b)
- `CONJ`: Conjugate transpose (solve Aᴴx = b)
"""
@enum trans_t::Cint begin
    NOTRANS = 0
    TRANS = 1
    CONJ = 2
end

"""
    colperm_t

Column permutation strategy enumeration for SuperLU.

Column permutation is used to reduce fill-in during factorization and improve
performance. Different strategies trade off between quality and computation time.

# Values
- `NATURAL`: Natural ordering (no permutation)
- `MMD_ATA`: Minimum degree ordering on AᵀA (for non-symmetric matrices)
- `MMD_AT_PLUS_A`: Minimum degree ordering on Aᵀ+A (for nearly symmetric matrices)
- `COLAMD`: Column approximate minimum degree (recommended for general matrices)
- `METIS_AT_PLUS_A`: METIS nested dissection on Aᵀ+A
- `PARMETIS`: ParMETIS parallel ordering (for distributed matrices)
- `ZOLTAN`: Zoltan hypergraph partitioning
- `MY_PERMC`: User-supplied column permutation

# Recommendations
- For general matrices: `COLAMD` (default in SuperLU)
- For symmetric or nearly symmetric: `MMD_AT_PLUS_A`
- For large matrices where quality matters: `METIS_AT_PLUS_A`
"""
@enum colperm_t::Cint begin
    NATURAL = 0
    MMD_ATA = 1
    MMD_AT_PLUS_A = 2
    COLAMD = 3
    METIS_AT_PLUS_A = 4
    PARMETIS = 5
    ZOLTAN = 6
    MY_PERMC = 7
end

"""
    rowperm_t

Row permutation strategy enumeration for SuperLU.

Row permutation is used for numerical stability during factorization.

# Values
- `NOROWPERM`: No row permutation (faster but may be less stable)
- `LargeDiag_MC64`: Move largest entries to diagonal using MC64 algorithm
- `LargeDiag_HWPM`: Hungarian algorithm for weighted perfect matching
- `MY_PERMR`: User-supplied row permutation

# Recommendations
- For well-conditioned matrices: `NOROWPERM`
- For general matrices: `LargeDiag_MC64` (recommended for stability)
"""
@enum rowperm_t::Cint begin
    NOROWPERM = 0
    LargeDiag_MC64 = 1
    LargeDiag_HWPM = 2
    MY_PERMR = 3
end

"""
    IterRefine_t

Iterative refinement strategy enumeration for SuperLU.

Iterative refinement improves the accuracy of the solution at the cost of
additional computation. It is particularly useful for ill-conditioned systems.

# Values
- `NOREFINE`: No iterative refinement (fastest)
- `SLU_SINGLE`: Single precision iterative refinement
- `SLU_DOUBLE`: Double precision iterative refinement
- `SLU_EXTRA`: Extra precision iterative refinement (most accurate)

# Recommendations
- For well-conditioned systems: `NOREFINE`
- For improved accuracy: `SLU_DOUBLE`
"""
@enum IterRefine_t::Cint begin
    NOREFINE = 0
    SLU_SINGLE = 1
    SLU_DOUBLE = 2
    SLU_EXTRA = 3
end

"""
    norm_t

Norm type enumeration for condition number estimation.

# Values
- `ONE_NORM`: 1-norm (max column sum)
- `TWO_NORM`: 2-norm (spectral norm)
- `INF_NORM`: ∞-norm (max row sum)
"""
@enum norm_t::Cint begin
    ONE_NORM = 0
    TWO_NORM = 1
    INF_NORM = 2
end

"""
    milu_t

Modified ILU type enumeration for incomplete factorization.

# Values
- `SILU`: Standard ILU
- `SMILU_1`: Modified ILU variant 1
- `SMILU_2`: Modified ILU variant 2
- `SMILU_3`: Modified ILU variant 3
"""
@enum milu_t::Cint begin
    SILU = 0
    SMILU_1 = 1
    SMILU_2 = 2
    SMILU_3 = 3
end

# ============================================================================
# User-Friendly Options Structure
# ============================================================================

"""
    SuperLUOptions

User-friendly options structure for configuring SuperLU solver behavior.

This structure provides a convenient way to configure SuperLU's solver options
without needing to understand the low-level C interface. Options are organized
into logical categories for ease of use.

# Constructor
    SuperLUOptions(;
        # Permutation options
        col_perm::colperm_t = COLAMD,
        row_perm::rowperm_t = LargeDiag_MC64,
        
        # Equilibration and scaling
        equilibrate::Bool = true,
        
        # Pivoting options
        diag_pivot_thresh::Float64 = 1.0,
        symmetric_mode::Bool = false,
        
        # Refinement options
        iterative_refinement::IterRefine_t = NOREFINE,
        
        # Diagnostic options
        pivot_growth::Bool = false,
        condition_number::Bool = false,
        print_stats::Bool = false,
        
        # Advanced options
        replace_tiny_pivot::Bool = false
    )

# Fields

## Permutation Options

- `col_perm::colperm_t`: Column permutation strategy (default: `COLAMD`)
  - `NATURAL`: No permutation
  - `MMD_ATA`: Minimum degree on AᵀA
  - `MMD_AT_PLUS_A`: Minimum degree on Aᵀ+A  
  - `COLAMD`: Column approximate minimum degree (recommended)
  - `METIS_AT_PLUS_A`: METIS nested dissection

- `row_perm::rowperm_t`: Row permutation strategy (default: `LargeDiag_MC64`)
  - `NOROWPERM`: No row permutation
  - `LargeDiag_MC64`: MC64 algorithm for weighted matching
  - `LargeDiag_HWPM`: Hungarian weighted perfect matching

## Equilibration and Scaling

- `equilibrate::Bool`: Whether to equilibrate (scale) the matrix (default: `true`)
  Row and column scaling can improve numerical stability.

## Pivoting Options

- `diag_pivot_thresh::Float64`: Diagonal pivot threshold (default: `1.0`)
  Controls when off-diagonal elements are preferred over diagonal elements.
  Range: [0.0, 1.0]. Value of 1.0 means always prefer diagonal (partial pivoting).
  Value of 0.0 means never prefer diagonal (complete pivoting).

- `symmetric_mode::Bool`: Symmetric mode (default: `false`)
  When `true`, uses symmetric storage and factorization patterns.
  Only applies to structurally symmetric matrices.

## Refinement Options

- `iterative_refinement::IterRefine_t`: Iterative refinement strategy (default: `NOREFINE`)
  - `NOREFINE`: No refinement (fastest)
  - `SLU_SINGLE`: Single precision refinement
  - `SLU_DOUBLE`: Double precision refinement  
  - `SLU_EXTRA`: Extra precision refinement (most accurate)

## Diagnostic Options

- `pivot_growth::Bool`: Compute reciprocal pivot growth factor (default: `false`)
- `condition_number::Bool`: Estimate condition number (default: `false`)
- `print_stats::Bool`: Print solver statistics (default: `false`)

## Advanced Options

- `replace_tiny_pivot::Bool`: Replace tiny pivots (default: `false`)
  When `true`, tiny pivots are replaced to improve stability.

# Example

```julia
# Create options with custom column permutation and iterative refinement
opts = SuperLUOptions(
    col_perm = METIS_AT_PLUS_A,
    iterative_refinement = SLU_DOUBLE,
    equilibrate = true
)

# Create factorization with these options
F = SuperLUFactorize(A; options=opts)
```

See also: [`colperm_t`](@ref), [`rowperm_t`](@ref), [`IterRefine_t`](@ref)
"""
struct SuperLUOptions
    # Permutation options
    col_perm::colperm_t
    row_perm::rowperm_t
    
    # Equilibration
    equilibrate::Bool
    
    # Pivoting
    diag_pivot_thresh::Float64
    symmetric_mode::Bool
    
    # Refinement
    iterative_refinement::IterRefine_t
    
    # Diagnostics
    pivot_growth::Bool
    condition_number::Bool
    print_stats::Bool
    
    # Advanced
    replace_tiny_pivot::Bool
end

function SuperLUOptions(;
    col_perm::colperm_t = COLAMD,
    row_perm::rowperm_t = LargeDiag_MC64,
    equilibrate::Bool = true,
    diag_pivot_thresh::Float64 = 1.0,
    symmetric_mode::Bool = false,
    iterative_refinement::IterRefine_t = NOREFINE,
    pivot_growth::Bool = false,
    condition_number::Bool = false,
    print_stats::Bool = false,
    replace_tiny_pivot::Bool = false
)
    # Validate diag_pivot_thresh
    if !(0.0 <= diag_pivot_thresh <= 1.0)
        throw(ArgumentError("diag_pivot_thresh must be in [0.0, 1.0], got $diag_pivot_thresh"))
    end
    
    SuperLUOptions(
        col_perm, row_perm, equilibrate, diag_pivot_thresh, symmetric_mode,
        iterative_refinement, pivot_growth, condition_number, print_stats,
        replace_tiny_pivot
    )
end

# apply_options! function is defined after superlu_options_t struct below

# ============================================================================
# Internal SuperLU Structures
# ============================================================================

# SuperMatrix structure - mutable for C interop
mutable struct SuperMatrix
    Stype::Stype_t      # Storage type
    Dtype::Dtype_t      # Data type
    Mtype::Mtype_t      # Matrix type
    nrow::Cint          # Number of rows
    ncol::Cint          # Number of columns
    Store::Ptr{Cvoid}   # Pointer to the actual storage
    
    function SuperMatrix()
        new(SLU_NC, SLU_Z, SLU_GE, 0, 0, C_NULL)
    end
end

# NCformat for compressed column storage
struct NCformat
    nnz::Cint           # Number of nonzeros
    nzval::Ptr{Cvoid}   # Pointer to nonzero values
    rowind::Ptr{Cint}   # Pointer to row indices
    colptr::Ptr{Cint}   # Pointer to column pointers
end

# DNformat for dense matrix storage
struct DNformat
    lda::Cint           # Leading dimension
    nzval::Ptr{Cvoid}   # Pointer to values
end

# SCformat for supernodal column storage (L and U factors)
struct SCformat
    nnz::Cint           # Number of nonzeros
    nsuper::Cint        # Number of supernodes
    nzval::Ptr{Cvoid}   # Pointer to nonzero values
    nzval_colptr::Ptr{Cint}  # Column pointers for nzval
    rowind::Ptr{Cint}   # Row indices
    rowind_colptr::Ptr{Cint} # Column pointers for rowind
    col_to_sup::Ptr{Cint}    # Column to supernode mapping
    sup_to_col::Ptr{Cint}    # Supernode to column mapping
end

# SuperLU options structure - mutable for C interop
mutable struct superlu_options_t
    Fact::fact_t
    Equil::yes_no_t
    ColPerm::colperm_t
    Trans::trans_t
    IterRefine::IterRefine_t
    DiagPivotThresh::Cdouble
    SymmetricMode::yes_no_t
    PivotGrowth::yes_no_t
    ConditionNumber::yes_no_t
    PrintStat::yes_no_t
    RowPerm::rowperm_t
    ILU_DropRule::Cint
    ILU_DropTol::Cdouble
    ILU_FillFactor::Cdouble
    ILU_Norm::norm_t
    ILU_FillTol::Cdouble
    ILU_MILU::milu_t
    ILU_MILU_Dim::Cdouble
    ParSymbFact::yes_no_t
    ReplaceTinyPivot::yes_no_t
    SolveInitialized::yes_no_t
    RefineInitialized::yes_no_t
    num_lookaheads::Cint
    lookahead_etree::yes_no_t
    SymPattern::yes_no_t
    
    # Default constructor creates zeroed struct
    function superlu_options_t()
        new(DOFACT, NO, NATURAL, NOTRANS, NOREFINE, 
            0.0, NO, NO, NO, NO, NOROWPERM, 
            0, 0.0, 0.0, ONE_NORM, 0.0, SILU, 0.0,
            NO, NO, NO, NO, 0, NO, NO)
    end
end

"""
    apply_options!(c_options::superlu_options_t, options::SuperLUOptions)

Apply user-friendly options to the internal SuperLU C options structure.
"""
function apply_options!(c_options::superlu_options_t, options::SuperLUOptions)
    c_options.ColPerm = options.col_perm
    c_options.RowPerm = options.row_perm
    c_options.Equil = options.equilibrate ? YES : NO
    c_options.DiagPivotThresh = options.diag_pivot_thresh
    c_options.SymmetricMode = options.symmetric_mode ? YES : NO
    c_options.IterRefine = options.iterative_refinement
    c_options.PivotGrowth = options.pivot_growth ? YES : NO
    c_options.ConditionNumber = options.condition_number ? YES : NO
    c_options.PrintStat = options.print_stats ? YES : NO
    c_options.ReplaceTinyPivot = options.replace_tiny_pivot ? YES : NO
    return c_options
end

# SuperLU statistics structure - mutable for C interop
mutable struct SuperLUStat_t
    panel_histo::Ptr{Cint}
    utime::Ptr{Cdouble}
    ops::Ptr{Cdouble}
    TinyPivots::Cint
    RefineSteps::Cint
    expansions::Cint
    
    function SuperLUStat_t()
        new(C_NULL, C_NULL, C_NULL, 0, 0, 0)
    end
end

# Memory usage structure
mutable struct mem_usage_t
    for_lu::Cfloat
    total_needed::Cfloat
    
    function mem_usage_t()
        new(0.0f0, 0.0f0)
    end
end

# GlobalLU_t structure for storing LU factorization - mutable for C interop
mutable struct GlobalLU_t
    xsup::Ptr{Cint}
    supno::Ptr{Cint}
    lsub::Ptr{Cint}
    xlsub::Ptr{Cint}
    lusup::Ptr{Cvoid}
    xlusup::Ptr{Cint}
    ucol::Ptr{Cvoid}
    xucol::Ptr{Cint}
    usub::Ptr{Cint}
    xusub::Ptr{Cint}
    nzlmax::Cint
    nzumax::Cint
    nzlumax::Cint
    n::Cint
    LUnum_tempv::Cint
    MemModel::Cint
    num_expansions::Cint
    expanders::Ptr{Cvoid}
    stack::Ptr{Cvoid}
    
    function GlobalLU_t()
        new(C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, 
            C_NULL, C_NULL, C_NULL, C_NULL, 0, 0, 0, 0, 0, 0, 0, 
            C_NULL, C_NULL)
    end
end

# Mutable versions for holding allocated data
mutable struct SuperMatrixMut
    mat::SuperMatrix
    store_ptr::Ptr{Cvoid}
    data_refs::Vector{Any}  # Keep references to prevent GC
    
    function SuperMatrixMut()
        new(SuperMatrix(SLU_NC, SLU_Z, SLU_GE, 0, 0, C_NULL), C_NULL, Any[])
    end
end

Base.cconvert(::Type{Ref{SuperMatrix}}, m::SuperMatrixMut) = Ref(m.mat)
Base.unsafe_convert(::Type{Ptr{SuperMatrix}}, m::SuperMatrixMut) = pointer_from_objref(Ref(m.mat))

# Helper to get pointer to SuperMatrix
function get_ptr(m::SuperMatrixMut)
    pointer_from_objref(Ref(m.mat))
end
