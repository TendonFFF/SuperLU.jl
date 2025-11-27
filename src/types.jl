# SuperLU type definitions based on slu_zdefs.h and slu_util.h

# SuperLU data types for complex double precision
const doublecomplex = ComplexF64

# Enums for SuperLU options
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

@enum Dtype_t::Cint begin
    SLU_S = 0        # single
    SLU_D = 1        # double
    SLU_C = 2        # single complex
    SLU_Z = 3        # double complex
end

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

@enum fact_t::Cint begin
    DOFACT = 0
    SamePattern = 1
    SamePattern_SameRowPerm = 2
    FACTORED = 3
end

@enum yes_no_t::Cint begin
    NO = 0
    YES = 1
end

@enum trans_t::Cint begin
    NOTRANS = 0
    TRANS = 1
    CONJ = 2
end

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

@enum rowperm_t::Cint begin
    NOROWPERM = 0
    LargeDiag_MC64 = 1
    LargeDiag_HWPM = 2
    MY_PERMR = 3
end

@enum IterRefine_t::Cint begin
    NOREFINE = 0
    SLU_SINGLE = 1
    SLU_DOUBLE = 2
    SLU_EXTRA = 3
end

@enum norm_t::Cint begin
    ONE_NORM = 0
    TWO_NORM = 1
    INF_NORM = 2
end

@enum milu_t::Cint begin
    SILU = 0
    SMILU_1 = 1
    SMILU_2 = 2
    SMILU_3 = 3
end

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
