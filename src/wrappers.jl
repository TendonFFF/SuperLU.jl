# Low-level C wrappers for SuperLU_MT functions

# References to the SuperLU_MT libraries (one per data type)
const libsuperlumts = SuperLU_MT_jll.libsuperlumts  # Float32 (single precision real)
const libsuperlumtd = SuperLU_MT_jll.libsuperlumtd  # Float64 (double precision real)
const libsuperlumtc = SuperLU_MT_jll.libsuperlumtc  # ComplexF32 (single precision complex)
const libsuperlumtz = SuperLU_MT_jll.libsuperlumtz  # ComplexF64 (double precision complex)

# ============================================================================
# Supported element types for SuperLU_MT
# ============================================================================

"""
    SuperLUTypes

Union of all element types supported by SuperLU_MT: Float32, Float64, ComplexF32, ComplexF64.

## Supported Types

- `Float32`: Single precision real. Fastest, least accurate. Use for large problems 
  where single precision is sufficient.
- `Float64`: Double precision real. Good balance of speed and accuracy. 
  Recommended for most real-valued problems.
- `ComplexF32`: Single precision complex. Use for complex problems where single 
  precision is sufficient.
- `ComplexF64`: Double precision complex. Recommended for most complex-valued problems.

## Multi-threading Support

SuperLU_MT supports parallel LU factorization using multiple threads. The number of 
threads can be specified when creating a factorization object via the `nthreads` parameter.

## Example

```julia
using SuperLU, SparseArrays

# Float64 (double precision real) with 4 threads
A_d = sparse([4.0 1.0; 1.0 4.0])
F_d = SuperLUFactorize(A_d; nthreads=4)

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

"""
    libsuperlumt(::Type{T})

Return the appropriate SuperLU_MT library for element type T.
"""
libsuperlumt(::Type{Float32}) = libsuperlumts
libsuperlumt(::Type{Float64}) = libsuperlumtd
libsuperlumt(::Type{ComplexF32}) = libsuperlumtc
libsuperlumt(::Type{ComplexF64}) = libsuperlumtz

# ============================================================================
# Float32 (single precision real) wrappers - prefix 's'
# ============================================================================

function sCreate_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                                  nzval::Ptr{Float32}, rowind::Ptr{Cint}, 
                                  colptr::Ptr{Cint}, stype::Stype_t, 
                                  dtype::Dtype_t, mtype::Mtype_t)
    ccall((:sCreate_CompCol_Matrix, libsuperlumts), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Cint, Ptr{Float32}, Ptr{Cint}, 
           Ptr{Cint}, Stype_t, Dtype_t, Mtype_t),
          A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)
end

function sCreate_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint,
                                nzval::Ptr{Float32}, lda::Cint,
                                stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t)
    ccall((:sCreate_Dense_Matrix, libsuperlumts), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Ptr{Float32}, Cint, 
           Stype_t, Dtype_t, Mtype_t),
          B, m, n, nzval, lda, stype, dtype, mtype)
end

# psgssv - Parallel simple driver for Float32
# Signature: void psgssv(int_t nprocs, SuperMatrix *A, int_t *perm_c, int_t *perm_r,
#                        SuperMatrix *L, SuperMatrix *U, SuperMatrix *B, int_t *info)
function psgssv!(nprocs::Cint, A::SuperMatrix, 
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                 L::SuperMatrix, U::SuperMatrix,
                 B::SuperMatrix, info::Ref{Cint})
    ccall((:psgssv, libsuperlumts), Cvoid,
          (Cint, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{Cint}),
          nprocs, A, perm_c, perm_r, L, U, B, info)
end

# ============================================================================
# Float64 (double precision real) wrappers - prefix 'd'
# ============================================================================

function dCreate_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                                  nzval::Ptr{Float64}, rowind::Ptr{Cint}, 
                                  colptr::Ptr{Cint}, stype::Stype_t, 
                                  dtype::Dtype_t, mtype::Mtype_t)
    ccall((:dCreate_CompCol_Matrix, libsuperlumtd), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Cint, Ptr{Float64}, Ptr{Cint}, 
           Ptr{Cint}, Stype_t, Dtype_t, Mtype_t),
          A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)
end

function dCreate_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint,
                                nzval::Ptr{Float64}, lda::Cint,
                                stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t)
    ccall((:dCreate_Dense_Matrix, libsuperlumtd), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Ptr{Float64}, Cint, 
           Stype_t, Dtype_t, Mtype_t),
          B, m, n, nzval, lda, stype, dtype, mtype)
end

# pdgssv - Parallel simple driver for Float64
function pdgssv!(nprocs::Cint, A::SuperMatrix, 
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                 L::SuperMatrix, U::SuperMatrix,
                 B::SuperMatrix, info::Ref{Cint})
    ccall((:pdgssv, libsuperlumtd), Cvoid,
          (Cint, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{Cint}),
          nprocs, A, perm_c, perm_r, L, U, B, info)
end

# ============================================================================
# ComplexF32 (single precision complex) wrappers - prefix 'c'
# ============================================================================

function cCreate_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                                  nzval::Ptr{ComplexF32}, rowind::Ptr{Cint}, 
                                  colptr::Ptr{Cint}, stype::Stype_t, 
                                  dtype::Dtype_t, mtype::Mtype_t)
    ccall((:cCreate_CompCol_Matrix, libsuperlumtc), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Cint, Ptr{ComplexF32}, Ptr{Cint}, 
           Ptr{Cint}, Stype_t, Dtype_t, Mtype_t),
          A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)
end

function cCreate_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint,
                                nzval::Ptr{ComplexF32}, lda::Cint,
                                stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t)
    ccall((:cCreate_Dense_Matrix, libsuperlumtc), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Ptr{ComplexF32}, Cint, 
           Stype_t, Dtype_t, Mtype_t),
          B, m, n, nzval, lda, stype, dtype, mtype)
end

# pcgssv - Parallel simple driver for ComplexF32
function pcgssv!(nprocs::Cint, A::SuperMatrix, 
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                 L::SuperMatrix, U::SuperMatrix,
                 B::SuperMatrix, info::Ref{Cint})
    ccall((:pcgssv, libsuperlumtc), Cvoid,
          (Cint, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{Cint}),
          nprocs, A, perm_c, perm_r, L, U, B, info)
end

# ============================================================================
# ComplexF64 (double precision complex) wrappers - prefix 'z'
# ============================================================================

function zCreate_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                                  nzval::Ptr{ComplexF64}, rowind::Ptr{Cint}, 
                                  colptr::Ptr{Cint}, stype::Stype_t, 
                                  dtype::Dtype_t, mtype::Mtype_t)
    ccall((:zCreate_CompCol_Matrix, libsuperlumtz), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Cint, Ptr{ComplexF64}, Ptr{Cint}, 
           Ptr{Cint}, Stype_t, Dtype_t, Mtype_t),
          A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)
end

function zCreate_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint,
                                nzval::Ptr{ComplexF64}, lda::Cint,
                                stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t)
    ccall((:zCreate_Dense_Matrix, libsuperlumtz), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Ptr{ComplexF64}, Cint, 
           Stype_t, Dtype_t, Mtype_t),
          B, m, n, nzval, lda, stype, dtype, mtype)
end

# pzgssv - Parallel simple driver for ComplexF64
function pzgssv!(nprocs::Cint, A::SuperMatrix, 
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                 L::SuperMatrix, U::SuperMatrix,
                 B::SuperMatrix, info::Ref{Cint})
    ccall((:pzgssv, libsuperlumtz), Cvoid,
          (Cint, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{SuperMatrix}, Ref{Cint}),
          nprocs, A, perm_c, perm_r, L, U, B, info)
end

# ============================================================================
# Common destroy functions - type-specific versions
# ============================================================================

# Float32 versions
function Destroy_SuperMatrix_Store_s!(A::SuperMatrix)
    ccall((:Destroy_SuperMatrix_Store, libsuperlumts), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_CompCol_Matrix_s!(A::SuperMatrix)
    ccall((:Destroy_CompCol_Matrix, libsuperlumts), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_CompCol_Permuted_s!(A::SuperMatrix)
    ccall((:Destroy_CompCol_Permuted, libsuperlumts), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_SuperNode_Matrix_s!(L::SuperMatrix)
    ccall((:Destroy_SuperNode_Matrix, libsuperlumts), Cvoid,
          (Ref{SuperMatrix},), L)
end

# Float64 versions
function Destroy_SuperMatrix_Store_d!(A::SuperMatrix)
    ccall((:Destroy_SuperMatrix_Store, libsuperlumtd), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_CompCol_Matrix_d!(A::SuperMatrix)
    ccall((:Destroy_CompCol_Matrix, libsuperlumtd), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_CompCol_Permuted_d!(A::SuperMatrix)
    ccall((:Destroy_CompCol_Permuted, libsuperlumtd), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_SuperNode_Matrix_d!(L::SuperMatrix)
    ccall((:Destroy_SuperNode_Matrix, libsuperlumtd), Cvoid,
          (Ref{SuperMatrix},), L)
end

# ComplexF32 versions
function Destroy_SuperMatrix_Store_c!(A::SuperMatrix)
    ccall((:Destroy_SuperMatrix_Store, libsuperlumtc), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_CompCol_Matrix_c!(A::SuperMatrix)
    ccall((:Destroy_CompCol_Matrix, libsuperlumtc), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_CompCol_Permuted_c!(A::SuperMatrix)
    ccall((:Destroy_CompCol_Permuted, libsuperlumtc), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_SuperNode_Matrix_c!(L::SuperMatrix)
    ccall((:Destroy_SuperNode_Matrix, libsuperlumtc), Cvoid,
          (Ref{SuperMatrix},), L)
end

# ComplexF64 versions
function Destroy_SuperMatrix_Store_z!(A::SuperMatrix)
    ccall((:Destroy_SuperMatrix_Store, libsuperlumtz), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_CompCol_Matrix_z!(A::SuperMatrix)
    ccall((:Destroy_CompCol_Matrix, libsuperlumtz), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_CompCol_Permuted_z!(A::SuperMatrix)
    ccall((:Destroy_CompCol_Permuted, libsuperlumtz), Cvoid,
          (Ref{SuperMatrix},), A)
end

function Destroy_SuperNode_Matrix_z!(L::SuperMatrix)
    ccall((:Destroy_SuperNode_Matrix, libsuperlumtz), Cvoid,
          (Ref{SuperMatrix},), L)
end

# Type-dispatch wrappers
Destroy_SuperMatrix_Store!(A::SuperMatrix, ::Type{Float32}) = Destroy_SuperMatrix_Store_s!(A)
Destroy_SuperMatrix_Store!(A::SuperMatrix, ::Type{Float64}) = Destroy_SuperMatrix_Store_d!(A)
Destroy_SuperMatrix_Store!(A::SuperMatrix, ::Type{ComplexF32}) = Destroy_SuperMatrix_Store_c!(A)
Destroy_SuperMatrix_Store!(A::SuperMatrix, ::Type{ComplexF64}) = Destroy_SuperMatrix_Store_z!(A)

# Default versions using Float64 library
Destroy_SuperMatrix_Store!(A::SuperMatrix) = Destroy_SuperMatrix_Store_d!(A)
Destroy_CompCol_Matrix!(A::SuperMatrix) = Destroy_CompCol_Matrix_d!(A)
Destroy_CompCol_Permuted!(A::SuperMatrix) = Destroy_CompCol_Permuted_d!(A)
Destroy_SuperNode_Matrix!(L::SuperMatrix) = Destroy_SuperNode_Matrix_d!(L)

# Get column permutation - type-specific versions
function get_perm_c_s!(ispec::Cint, A::SuperMatrix, perm_c::Ptr{Cint})
    ccall((:get_perm_c, libsuperlumts), Cvoid,
          (Cint, Ref{SuperMatrix}, Ptr{Cint}),
          ispec, A, perm_c)
end

function get_perm_c_d!(ispec::Cint, A::SuperMatrix, perm_c::Ptr{Cint})
    ccall((:get_perm_c, libsuperlumtd), Cvoid,
          (Cint, Ref{SuperMatrix}, Ptr{Cint}),
          ispec, A, perm_c)
end

function get_perm_c_c!(ispec::Cint, A::SuperMatrix, perm_c::Ptr{Cint})
    ccall((:get_perm_c, libsuperlumtc), Cvoid,
          (Cint, Ref{SuperMatrix}, Ptr{Cint}),
          ispec, A, perm_c)
end

function get_perm_c_z!(ispec::Cint, A::SuperMatrix, perm_c::Ptr{Cint})
    ccall((:get_perm_c, libsuperlumtz), Cvoid,
          (Cint, Ref{SuperMatrix}, Ptr{Cint}),
          ispec, A, perm_c)
end

# Type-dispatch wrapper
get_perm_c!(ispec::Cint, A::SuperMatrix, perm_c::Ptr{Cint}, ::Type{Float32}) = get_perm_c_s!(ispec, A, perm_c)
get_perm_c!(ispec::Cint, A::SuperMatrix, perm_c::Ptr{Cint}, ::Type{Float64}) = get_perm_c_d!(ispec, A, perm_c)
get_perm_c!(ispec::Cint, A::SuperMatrix, perm_c::Ptr{Cint}, ::Type{ComplexF32}) = get_perm_c_c!(ispec, A, perm_c)
get_perm_c!(ispec::Cint, A::SuperMatrix, perm_c::Ptr{Cint}, ::Type{ComplexF64}) = get_perm_c_z!(ispec, A, perm_c)

# ============================================================================
# Type-generic wrappers using dispatch
# ============================================================================

"""
    Create_CompCol_Matrix!(A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)

Create a compressed column matrix. Dispatches to the correct library based on nzval pointer type.
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
    Create_Dense_Matrix!(B, m, n, nzval, lda, stype, dtype, mtype)

Create a dense matrix. Dispatches to the correct library based on nzval pointer type.
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
    pgssv!(nprocs, A, perm_c, perm_r, L, U, B, info, ::Type{T})

Parallel simple driver for LU factorization and solve for element type T.
"""
pgssv!(nprocs::Cint, A::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
       L::SuperMatrix, U::SuperMatrix, B::SuperMatrix, 
       info::Ref{Cint}, ::Type{Float32}) =
    psgssv!(nprocs, A, perm_c, perm_r, L, U, B, info)

pgssv!(nprocs::Cint, A::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
       L::SuperMatrix, U::SuperMatrix, B::SuperMatrix, 
       info::Ref{Cint}, ::Type{Float64}) =
    pdgssv!(nprocs, A, perm_c, perm_r, L, U, B, info)

pgssv!(nprocs::Cint, A::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
       L::SuperMatrix, U::SuperMatrix, B::SuperMatrix, 
       info::Ref{Cint}, ::Type{ComplexF32}) =
    pcgssv!(nprocs, A, perm_c, perm_r, L, U, B, info)

pgssv!(nprocs::Cint, A::SuperMatrix, perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
       L::SuperMatrix, U::SuperMatrix, B::SuperMatrix, 
       info::Ref{Cint}, ::Type{ComplexF64}) =
    pzgssv!(nprocs, A, perm_c, perm_r, L, U, B, info)

# Get default relax and panel_size - type-specific versions
sp_ienv_s(ispec::Cint) = ccall((:sp_ienv, libsuperlumts), Cint, (Cint,), ispec)
sp_ienv_d(ispec::Cint) = ccall((:sp_ienv, libsuperlumtd), Cint, (Cint,), ispec)
sp_ienv_c(ispec::Cint) = ccall((:sp_ienv, libsuperlumtc), Cint, (Cint,), ispec)
sp_ienv_z(ispec::Cint) = ccall((:sp_ienv, libsuperlumtz), Cint, (Cint,), ispec)

# Default version using Float64 library
sp_ienv(ispec::Cint) = sp_ienv_d(ispec)
