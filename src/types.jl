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

See also: [`colperm_t`](@ref)
"""
struct SuperLUOptions
    # Permutation options
    col_perm::colperm_t
    
    # Pivoting
    diag_pivot_thresh::Float64
    symmetric_mode::Bool
    
    # Diagnostics
    print_stats::Bool
end

function SuperLUOptions(;
    col_perm::colperm_t = COLAMD,
    diag_pivot_thresh::Float64 = 1.0,
    symmetric_mode::Bool = false,
    print_stats::Bool = false
)
    # Validate diag_pivot_thresh
    if !(0.0 <= diag_pivot_thresh <= 1.0)
        throw(ArgumentError("diag_pivot_thresh must be in [0.0, 1.0], got $diag_pivot_thresh"))
    end
    
    SuperLUOptions(col_perm, diag_pivot_thresh, symmetric_mode, print_stats)
end

# ============================================================================
# Preset Options Objects
# ============================================================================

"""
    ILL_CONDITIONED_OPTIONS

Pre-configured options optimized for solving ill-conditioned systems.
These settings prioritize numerical stability and accuracy over speed.

Note: SuperLU_MT has fewer options than the sequential SuperLU.
This preset uses the most stable column ordering.

Equivalent to:
```julia
SuperLUOptions(
    col_perm = MMD_AT_PLUS_A,
    diag_pivot_thresh = 1.0
)
```

See also: [`SuperLUOptions`](@ref), [`PERFORMANCE_OPTIONS`](@ref), [`ACCURACY_OPTIONS`](@ref)
"""
const ILL_CONDITIONED_OPTIONS = SuperLUOptions(
    col_perm = MMD_AT_PLUS_A,
    diag_pivot_thresh = 1.0
)

"""
    PERFORMANCE_OPTIONS

Pre-configured options optimized for maximum performance.
These settings prioritize speed over accuracy - use only for well-conditioned systems.

Equivalent to:
```julia
SuperLUOptions(
    col_perm = COLAMD,
    diag_pivot_thresh = 0.0  # No partial pivoting
)
```

See also: [`SuperLUOptions`](@ref), [`ILL_CONDITIONED_OPTIONS`](@ref), [`ACCURACY_OPTIONS`](@ref)
"""
const PERFORMANCE_OPTIONS = SuperLUOptions(
    col_perm = COLAMD,
    diag_pivot_thresh = 0.0
)

"""
    ACCURACY_OPTIONS

Pre-configured options optimized for maximum accuracy.
These settings prioritize accuracy over speed.

Equivalent to:
```julia
SuperLUOptions(
    col_perm = MMD_AT_PLUS_A,
    diag_pivot_thresh = 1.0
)
```

See also: [`SuperLUOptions`](@ref), [`ILL_CONDITIONED_OPTIONS`](@ref), [`PERFORMANCE_OPTIONS`](@ref)
"""
const ACCURACY_OPTIONS = SuperLUOptions(
    col_perm = MMD_AT_PLUS_A,
    diag_pivot_thresh = 1.0
)

"""
    SYMMETRIC_OPTIONS

Pre-configured options optimized for symmetric or nearly symmetric matrices.
These settings use ordering strategies that exploit symmetric structure.

Equivalent to:
```julia
SuperLUOptions(
    col_perm = MMD_AT_PLUS_A,
    symmetric_mode = true
)
```

See also: [`SuperLUOptions`](@ref), [`issymmetric_structure`](@ref)
"""
const SYMMETRIC_OPTIONS = SuperLUOptions(
    col_perm = MMD_AT_PLUS_A,
    symmetric_mode = true
)

# apply_options! function is defined after superlumt_options_t struct below

# ============================================================================
# Internal SuperLU_MT Structures
# ============================================================================

# SuperMatrix structure - mutable for C interop
# Note: SuperLU_MT uses int_t which defaults to int (Cint)
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

# SCPformat for supernodal column storage with permutation (L and U factors in SuperLU_MT)
struct SCPformat
    nnz::Cint             # Number of nonzeros
    nsuper::Cint          # Number of supernodes
    nzval::Ptr{Cvoid}     # Pointer to nonzero values
    nzval_colbeg::Ptr{Cint}   # Beginning of columns in nzval
    nzval_colend::Ptr{Cint}   # End of columns in nzval
    rowind::Ptr{Cint}     # Row indices
    rowind_colbeg::Ptr{Cint}  # Beginning of columns in rowind
    rowind_colend::Ptr{Cint}  # End of columns in rowind
    col_to_sup::Ptr{Cint}     # Column to supernode mapping
    sup_to_colbeg::Ptr{Cint}  # Supernode to column beginning mapping
    sup_to_colend::Ptr{Cint}  # Supernode to column ending mapping
end

# NCPformat for compressed column storage with permutation
struct NCPformat
    nnz::Cint           # Number of nonzeros
    nzval::Ptr{Cvoid}   # Pointer to nonzero values
    rowind::Ptr{Cint}   # Pointer to row indices
    colbeg::Ptr{Cint}   # Beginning of columns
    colend::Ptr{Cint}   # End of columns
end

# equed_t enumeration for equilibration status
@enum equed_t::Cint begin
    NOEQUIL = 0     # No equilibration
    ROW_EQUIL = 1   # Row equilibration
    COL_EQUIL = 2   # Column equilibration
    BOTH_EQUIL = 3  # Both row and column equilibration
end

# SuperLU_MT options structure - mutable for C interop
# Based on superlumt_options_t from slu_mt_util.h
mutable struct superlumt_options_t
    nprocs::Cint              # Number of processors/threads
    fact::fact_t              # Factorization type
    trans::trans_t            # Transpose option
    refact::yes_no_t          # Is this a refactorization?
    panel_size::Cint          # Panel size
    relax::Cint               # Relaxation parameter
    diag_pivot_thresh::Cdouble  # Diagonal pivot threshold
    drop_tol::Cdouble         # Drop tolerance (not implemented)
    ColPerm::colperm_t        # Column permutation strategy
    usepr::yes_no_t           # Use user-provided row permutation
    SymmetricMode::yes_no_t   # Symmetric mode
    PrintStat::yes_no_t       # Print statistics
    
    # Pointers to permutation arrays (managed by factorization object)
    perm_c::Ptr{Cint}
    perm_r::Ptr{Cint}
    work::Ptr{Cvoid}
    lwork::Cint
    
    # Pointers to structural arrays (computed by sp_colorder)
    etree::Ptr{Cint}
    colcnt_h::Ptr{Cint}
    part_super_h::Ptr{Cint}
    
    function superlumt_options_t()
        new(1, DOFACT, NOTRANS, NO, 
            8, 4, 1.0, 0.0, COLAMD, NO, NO, NO,
            C_NULL, C_NULL, C_NULL, 0,
            C_NULL, C_NULL, C_NULL)
    end
end

"""
    apply_options!(c_options::superlumt_options_t, options::SuperLUOptions, nprocs::Int)

Apply user-friendly options to the internal SuperLU_MT C options structure.
"""
function apply_options!(c_options::superlumt_options_t, options::SuperLUOptions, nprocs::Int)
    c_options.nprocs = Cint(nprocs)
    c_options.ColPerm = options.col_perm
    c_options.diag_pivot_thresh = options.diag_pivot_thresh
    c_options.SymmetricMode = options.symmetric_mode ? YES : NO
    c_options.PrintStat = options.print_stats ? YES : NO
    return c_options
end

# Memory usage structure for SuperLU_MT
mutable struct superlu_memusage_t
    for_lu::Cfloat
    total_needed::Cfloat
    expansions::Cint
    
    function superlu_memusage_t()
        new(0.0f0, 0.0f0, 0)
    end
end

# GlobalLU_t structure for storing LU factorization in SuperLU_MT
# This is different from the sequential SuperLU version
mutable struct GlobalLU_t
    xsup::Ptr{Cint}           # Supernode and column mapping
    xsup_end::Ptr{Cint}
    supno::Ptr{Cint}
    lsub::Ptr{Cint}           # Compressed L subscripts
    xlsub::Ptr{Cint}
    xlsub_end::Ptr{Cint}
    lusup::Ptr{Cvoid}         # L supernodes (type depends on precision)
    xlusup::Ptr{Cint}
    xlusup_end::Ptr{Cint}
    ucol::Ptr{Cvoid}          # U columns (type depends on precision)
    usub::Ptr{Cint}
    xusub::Ptr{Cint}
    xusub_end::Ptr{Cint}
    nsuper::Cint              # Current supernode number
    nextl::Cint               # Next position in lsub
    nextu::Cint               # Next position in usub/ucol
    nextlu::Cint              # Next position in lusup
    nzlmax::Cint              # Current max size of lsub
    nzumax::Cint              # Current max size of ucol
    nzlumax::Cint             # Current max size of lusup
    map_in_sup::Ptr{Cint}     # Memory management for L supernodes
    dynamic_snode_bound::Cint
    
    function GlobalLU_t()
        new(C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
            C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
            0, 0, 0, 0, 0, 0, 0, C_NULL, 0)
    end
end

# Gstat_t structure for SuperLU_MT statistics
# This is a simplified version - full structure has more fields for profiling
mutable struct Gstat_t
    panel_histo::Ptr{Cint}
    utime::Ptr{Cdouble}
    ops::Ptr{Cvoid}           # flops_t*
    procstat::Ptr{Cvoid}      # procstat_t*
    panstat::Ptr{Cvoid}       # panstat_t*
    num_panels::Cint
    dom_flopcnt::Cfloat
    flops_last_P_panels::Cfloat
    stat_relax::Ptr{Cvoid}    # stat_relax_t*
    stat_col::Ptr{Cvoid}      # stat_col_t*
    stat_snode::Ptr{Cvoid}    # stat_snode_t*
    panhows::Ptr{Cint}
    cp_panel::Ptr{Cvoid}      # cp_panel_t*
    desc_eft::Ptr{Cvoid}      # desc_eft_t*
    cp_firstkid::Ptr{Cint}
    cp_nextkid::Ptr{Cint}
    height::Ptr{Cint}
    flops_by_height::Ptr{Cfloat}
    
    function Gstat_t()
        new(C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, 0, 0.0f0, 0.0f0,
            C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
            C_NULL, C_NULL, C_NULL, C_NULL)
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
