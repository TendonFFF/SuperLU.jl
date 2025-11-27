# Low-level C wrappers for SuperLU functions

# Reference to the SuperLU library
const libsuperlu = SuperLU_jll.libsuperlu

# ============================================================================
# Supported element types for SuperLU
# ============================================================================

"""
    SuperLUTypes

Union of all element types supported by SuperLU: Float32, Float64, ComplexF32, ComplexF64.

## Supported Types

- `Float32`: Single precision real. Fastest, least accurate. Use for large problems 
  where single precision is sufficient.
- `Float64`: Double precision real. Good balance of speed and accuracy. 
  Recommended for most real-valued problems.
- `ComplexF32`: Single precision complex. Use for complex problems where single 
  precision is sufficient.
- `ComplexF64`: Double precision complex. Recommended for most complex-valued problems.

## Performance Considerations

- Single precision types (`Float32`, `ComplexF32`) use half the memory and may be 
  faster on some hardware, but have reduced numerical precision.
- Double precision types (`Float64`, `ComplexF64`) offer approximately 15-16 significant 
  digits vs 6-7 for single precision.
- The choice of precision affects both the factorization and solve phases.

## Example

```julia
using SuperLU, SparseArrays

# Float64 (double precision real)
A_d = sparse([4.0 1.0; 1.0 4.0])
F_d = SuperLUFactorize(A_d)

# Float32 (single precision real)  
A_s = sparse(Float32[4.0 1.0; 1.0 4.0])
F_s = SuperLUFactorize(A_s)

# ComplexF64 (double precision complex)
A_z = sparse([4.0+1.0im 1.0; 1.0 4.0+1.0im])
F_z = SuperLUFactorize(A_z)
```
"""
const SuperLUTypes = Union{Float32, Float64, ComplexF32, ComplexF64}

"""
    slu_dtype(::Type{T}) -> Dtype_t

Return the SuperLU data type enum for Julia type T.
"""
slu_dtype(::Type{Float32}) = SLU_S
slu_dtype(::Type{Float64}) = SLU_D
slu_dtype(::Type{ComplexF32}) = SLU_C
slu_dtype(::Type{ComplexF64}) = SLU_Z

# Set default options - pass the mutable struct directly
function set_default_options!(options::superlu_options_t)
    ccall((:set_default_options, libsuperlu), Cvoid,
          (Ref{superlu_options_t},), options)
end

function set_default_options()
    options = superlu_options_t()
    set_default_options!(options)
    return options
end

# Statistics functions
function StatInit!(stat::SuperLUStat_t)
    ccall((:StatInit, libsuperlu), Cvoid,
          (Ref{SuperLUStat_t},), stat)
end

function StatFree!(stat::SuperLUStat_t)
    ccall((:StatFree, libsuperlu), Cvoid,
          (Ref{SuperLUStat_t},), stat)
end

# ============================================================================
# Float32 (single precision real) wrappers - prefix 's'
# ============================================================================

function sCreate_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                                  nzval::Ptr{Float32}, rowind::Ptr{Cint}, 
                                  colptr::Ptr{Cint}, stype::Stype_t, 
                                  dtype::Dtype_t, mtype::Mtype_t)
    ccall((:sCreate_CompCol_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Cint, Ptr{Float32}, Ptr{Cint}, 
           Ptr{Cint}, Stype_t, Dtype_t, Mtype_t),
          A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)
end

function sCreate_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint,
                                nzval::Ptr{Float32}, lda::Cint,
                                stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t)
    ccall((:sCreate_Dense_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Ptr{Float32}, Cint, 
           Stype_t, Dtype_t, Mtype_t),
          B, m, n, nzval, lda, stype, dtype, mtype)
end

function sgssv!(options::superlu_options_t, A::SuperMatrix, 
                perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                L::SuperMatrix, U::SuperMatrix,
                B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint})
    ccall((:sgssv, libsuperlu), Cvoid,
          (Ref{superlu_options_t}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{SuperMatrix}, 
           Ref{SuperLUStat_t}, Ref{Cint}),
          options, A, perm_c, perm_r, L, U, B, stat, info)
end

function sgstrs!(trans::trans_t, L::SuperMatrix, U::SuperMatrix,
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                 B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint})
    ccall((:sgstrs, libsuperlu), Cvoid,
          (trans_t, Ref{SuperMatrix}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperLUStat_t}, Ref{Cint}),
          trans, L, U, perm_c, perm_r, B, stat, info)
end

# ============================================================================
# Float64 (double precision real) wrappers - prefix 'd'
# ============================================================================

function dCreate_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                                  nzval::Ptr{Float64}, rowind::Ptr{Cint}, 
                                  colptr::Ptr{Cint}, stype::Stype_t, 
                                  dtype::Dtype_t, mtype::Mtype_t)
    ccall((:dCreate_CompCol_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Cint, Ptr{Float64}, Ptr{Cint}, 
           Ptr{Cint}, Stype_t, Dtype_t, Mtype_t),
          A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)
end

function dCreate_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint,
                                nzval::Ptr{Float64}, lda::Cint,
                                stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t)
    ccall((:dCreate_Dense_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Ptr{Float64}, Cint, 
           Stype_t, Dtype_t, Mtype_t),
          B, m, n, nzval, lda, stype, dtype, mtype)
end

function dgssv!(options::superlu_options_t, A::SuperMatrix, 
                perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                L::SuperMatrix, U::SuperMatrix,
                B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint})
    ccall((:dgssv, libsuperlu), Cvoid,
          (Ref{superlu_options_t}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{SuperMatrix}, 
           Ref{SuperLUStat_t}, Ref{Cint}),
          options, A, perm_c, perm_r, L, U, B, stat, info)
end

function dgstrs!(trans::trans_t, L::SuperMatrix, U::SuperMatrix,
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                 B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint})
    ccall((:dgstrs, libsuperlu), Cvoid,
          (trans_t, Ref{SuperMatrix}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperLUStat_t}, Ref{Cint}),
          trans, L, U, perm_c, perm_r, B, stat, info)
end

# ============================================================================
# ComplexF32 (single precision complex) wrappers - prefix 'c'
# ============================================================================

function cCreate_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                                  nzval::Ptr{ComplexF32}, rowind::Ptr{Cint}, 
                                  colptr::Ptr{Cint}, stype::Stype_t, 
                                  dtype::Dtype_t, mtype::Mtype_t)
    ccall((:cCreate_CompCol_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Cint, Ptr{ComplexF32}, Ptr{Cint}, 
           Ptr{Cint}, Stype_t, Dtype_t, Mtype_t),
          A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)
end

function cCreate_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint,
                                nzval::Ptr{ComplexF32}, lda::Cint,
                                stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t)
    ccall((:cCreate_Dense_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Ptr{ComplexF32}, Cint, 
           Stype_t, Dtype_t, Mtype_t),
          B, m, n, nzval, lda, stype, dtype, mtype)
end

function cgssv!(options::superlu_options_t, A::SuperMatrix, 
                perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                L::SuperMatrix, U::SuperMatrix,
                B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint})
    ccall((:cgssv, libsuperlu), Cvoid,
          (Ref{superlu_options_t}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{SuperMatrix}, 
           Ref{SuperLUStat_t}, Ref{Cint}),
          options, A, perm_c, perm_r, L, U, B, stat, info)
end

function cgstrs!(trans::trans_t, L::SuperMatrix, U::SuperMatrix,
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                 B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint})
    ccall((:cgstrs, libsuperlu), Cvoid,
          (trans_t, Ref{SuperMatrix}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperLUStat_t}, Ref{Cint}),
          trans, L, U, perm_c, perm_r, B, stat, info)
end

# ============================================================================
# ComplexF64 (double precision complex) wrappers - prefix 'z'
# ============================================================================

function zCreate_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                                  nzval::Ptr{ComplexF64}, rowind::Ptr{Cint}, 
                                  colptr::Ptr{Cint}, stype::Stype_t, 
                                  dtype::Dtype_t, mtype::Mtype_t)
    ccall((:zCreate_CompCol_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Cint, Ptr{ComplexF64}, Ptr{Cint}, 
           Ptr{Cint}, Stype_t, Dtype_t, Mtype_t),
          A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)
end

function zCreate_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint,
                                nzval::Ptr{ComplexF64}, lda::Cint,
                                stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t)
    ccall((:zCreate_Dense_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Ptr{ComplexF64}, Cint, 
           Stype_t, Dtype_t, Mtype_t),
          B, m, n, nzval, lda, stype, dtype, mtype)
end

# ============================================================================
# Common destroy functions
# ============================================================================

# Destroy matrix store
function Destroy_SuperMatrix_Store!(A::SuperMatrix)
    ccall((:Destroy_SuperMatrix_Store, libsuperlu), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_CompCol_Matrix!(A::SuperMatrix)
    ccall((:Destroy_CompCol_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_CompCol_Permuted!(A::SuperMatrix)
    ccall((:Destroy_CompCol_Permuted, libsuperlu), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_SuperNode_Matrix!(L::SuperMatrix)
    ccall((:Destroy_SuperNode_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix},), L)
end

function Destroy_Dense_Matrix!(B::SuperMatrix)
    ccall((:Destroy_Dense_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix},), B)
end

# Get column permutation
function get_perm_c!(ispec::Cint, A::SuperMatrix, perm_c::Ptr{Cint})
    ccall((:get_perm_c, libsuperlu), Cvoid,
          (Cint, Ref{SuperMatrix}, Ptr{Cint}),
          ispec, A, perm_c)
end

# ============================================================================
# Simple drivers (gssv) for all types
# ============================================================================

# Simple driver for complex double (zgssv) - keep for backward compatibility
function zgssv!(options::superlu_options_t, A::SuperMatrix, 
                perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                L::SuperMatrix, U::SuperMatrix,
                B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint})
    ccall((:zgssv, libsuperlu), Cvoid,
          (Ref{superlu_options_t}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{SuperMatrix}, 
           Ref{SuperLUStat_t}, Ref{Cint}),
          options, A, perm_c, perm_r, L, U, B, stat, info)
end

# ============================================================================
# Expert drivers (gssvx) for all types
# ============================================================================

# Expert driver for complex double (zgssvx)
function zgssvx!(options::superlu_options_t, A::SuperMatrix,
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint}, etree::Ptr{Cint},
                 equed::Ptr{Cchar}, R::Ptr{Cdouble}, C::Ptr{Cdouble},
                 L::SuperMatrix, U::SuperMatrix,
                 work::Ptr{Cvoid}, lwork::Cint,
                 B::SuperMatrix, X::SuperMatrix,
                 recip_pivot_growth::Ref{Cdouble}, rcond::Ref{Cdouble},
                 ferr::Ptr{Cdouble}, berr::Ptr{Cdouble},
                 Glu::GlobalLU_t,
                 mem_usage::mem_usage_t, stat::SuperLUStat_t, 
                 info::Ref{Cint})
    ccall((:zgssvx, libsuperlu), Cvoid,
          (Ref{superlu_options_t}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ptr{Cint}, Ptr{Cchar}, Ptr{Cdouble}, Ptr{Cdouble},
           Ref{SuperMatrix}, Ref{SuperMatrix}, Ptr{Cvoid}, Cint,
           Ref{SuperMatrix}, Ref{SuperMatrix}, 
           Ref{Cdouble}, Ref{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ref{GlobalLU_t}, Ref{mem_usage_t}, Ref{SuperLUStat_t}, Ref{Cint}),
          options, A, perm_c, perm_r, etree, equed, R, C,
          L, U, work, lwork, B, X, 
          recip_pivot_growth, rcond, ferr, berr, Glu, mem_usage, stat, info)
end

# ============================================================================
# Symbolic factorization (gstrf) for all types
# ============================================================================

function zgstrf!(options::superlu_options_t, A::SuperMatrix,
                 relax::Cint, panel_size::Cint, 
                 etree::Ptr{Cint}, work::Ptr{Cvoid}, lwork::Cint,
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                 L::SuperMatrix, U::SuperMatrix,
                 Glu::GlobalLU_t,
                 stat::SuperLUStat_t, info::Ref{Cint})
    ccall((:zgstrf, libsuperlu), Cvoid,
          (Ref{superlu_options_t}, Ref{SuperMatrix}, Cint, Cint,
           Ptr{Cint}, Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{GlobalLU_t},
           Ref{SuperLUStat_t}, Ref{Cint}),
          options, A, relax, panel_size, etree, work, lwork,
          perm_c, perm_r, L, U, Glu, stat, info)
end

# ============================================================================
# Triangular solves (gstrs) for all types
# ============================================================================

# Triangular solve for complex double (zgstrs) - keep for backward compatibility
function zgstrs!(trans::trans_t, L::SuperMatrix, U::SuperMatrix,
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                 B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint})
    ccall((:zgstrs, libsuperlu), Cvoid,
          (trans_t, Ref{SuperMatrix}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperLUStat_t}, Ref{Cint}),
          trans, L, U, perm_c, perm_r, B, stat, info)
end

# ============================================================================
# Type-generic wrappers using dispatch
# ============================================================================

"""
    Create_CompCol_Matrix!(A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype, ::Type{T})

Create a compressed column matrix for element type T.
"""
Create_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                       nzval::Ptr{Float32}, rowind::Ptr{Cint}, colptr::Ptr{Cint},
                       stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t) =
    sCreate_CompCol_Matrix!(A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)

Create_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                       nzval::Ptr{Float64}, rowind::Ptr{Cint}, colptr::Ptr{Cint},
                       stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t) =
    dCreate_CompCol_Matrix!(A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)

Create_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                       nzval::Ptr{ComplexF32}, rowind::Ptr{Cint}, colptr::Ptr{Cint},
                       stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t) =
    cCreate_CompCol_Matrix!(A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)

Create_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                       nzval::Ptr{ComplexF64}, rowind::Ptr{Cint}, colptr::Ptr{Cint},
                       stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t) =
    zCreate_CompCol_Matrix!(A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)

"""
    Create_Dense_Matrix!(B, m, n, nzval, lda, stype, dtype, mtype, ::Type{T})

Create a dense matrix for element type T.
"""
Create_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint, nzval::Ptr{Float32}, lda::Cint,
                     stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t) =
    sCreate_Dense_Matrix!(B, m, n, nzval, lda, stype, dtype, mtype)

Create_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint, nzval::Ptr{Float64}, lda::Cint,
                     stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t) =
    dCreate_Dense_Matrix!(B, m, n, nzval, lda, stype, dtype, mtype)

Create_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint, nzval::Ptr{ComplexF32}, lda::Cint,
                     stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t) =
    cCreate_Dense_Matrix!(B, m, n, nzval, lda, stype, dtype, mtype)

Create_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint, nzval::Ptr{ComplexF64}, lda::Cint,
                     stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t) =
    zCreate_Dense_Matrix!(B, m, n, nzval, lda, stype, dtype, mtype)

"""
    gssv!(options, A, perm_c, perm_r, L, U, B, stat, info, ::Type{T})

Simple driver for LU factorization and solve for element type T.
"""
gssv!(options::superlu_options_t, A::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
      L::SuperMatrix, U::SuperMatrix, B::SuperMatrix, stat::SuperLUStat_t, 
      info::Ref{Cint}, ::Type{Float32}) =
    sgssv!(options, A, perm_c, perm_r, L, U, B, stat, info)

gssv!(options::superlu_options_t, A::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
      L::SuperMatrix, U::SuperMatrix, B::SuperMatrix, stat::SuperLUStat_t, 
      info::Ref{Cint}, ::Type{Float64}) =
    dgssv!(options, A, perm_c, perm_r, L, U, B, stat, info)

gssv!(options::superlu_options_t, A::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
      L::SuperMatrix, U::SuperMatrix, B::SuperMatrix, stat::SuperLUStat_t, 
      info::Ref{Cint}, ::Type{ComplexF32}) =
    cgssv!(options, A, perm_c, perm_r, L, U, B, stat, info)

gssv!(options::superlu_options_t, A::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
      L::SuperMatrix, U::SuperMatrix, B::SuperMatrix, stat::SuperLUStat_t, 
      info::Ref{Cint}, ::Type{ComplexF64}) =
    zgssv!(options, A, perm_c, perm_r, L, U, B, stat, info)

"""
    gstrs!(trans, L, U, perm_c, perm_r, B, stat, info, ::Type{T})

Triangular solve for element type T.
"""
gstrs!(trans::trans_t, L::SuperMatrix, U::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
       B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint}, ::Type{Float32}) =
    sgstrs!(trans, L, U, perm_c, perm_r, B, stat, info)

gstrs!(trans::trans_t, L::SuperMatrix, U::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
       B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint}, ::Type{Float64}) =
    dgstrs!(trans, L, U, perm_c, perm_r, B, stat, info)

gstrs!(trans::trans_t, L::SuperMatrix, U::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
       B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint}, ::Type{ComplexF32}) =
    cgstrs!(trans, L, U, perm_c, perm_r, B, stat, info)

gstrs!(trans::trans_t, L::SuperMatrix, U::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
       B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint}, ::Type{ComplexF64}) =
    zgstrs!(trans, L, U, perm_c, perm_r, B, stat, info)

# ============================================================================
# Permutation for preprocessing
# ============================================================================

function sp_preorder!(options::superlu_options_t, A::SuperMatrix,
                      perm_c::Ptr{Cint}, etree::Ptr{Cint}, AC::SuperMatrix)
    ccall((:sp_preorder, libsuperlu), Cvoid,
          (Ref{superlu_options_t}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint}, Ref{SuperMatrix}),
          options, A, perm_c, etree, AC)
end

# Get default relax and panel_size
function sp_ienv(ispec::Cint)
    ccall((:sp_ienv, libsuperlu), Cint, (Cint,), ispec)
end
