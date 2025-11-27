# Low-level C wrappers for SuperLU functions

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

# Create compressed column matrix (for complex double)
function zCreate_CompCol_Matrix!(A::SuperMatrix, m::Cint, n::Cint, nnz::Cint,
                                  nzval::Ptr{ComplexF64}, rowind::Ptr{Cint}, 
                                  colptr::Ptr{Cint}, stype::Stype_t, 
                                  dtype::Dtype_t, mtype::Mtype_t)
    ccall((:zCreate_CompCol_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Cint, Ptr{ComplexF64}, Ptr{Cint}, 
           Ptr{Cint}, Stype_t, Dtype_t, Mtype_t),
          A, m, n, nnz, nzval, rowind, colptr, stype, dtype, mtype)
end

# Create dense matrix (for complex double)
function zCreate_Dense_Matrix!(B::SuperMatrix, m::Cint, n::Cint,
                                nzval::Ptr{ComplexF64}, lda::Cint,
                                stype::Stype_t, dtype::Dtype_t, mtype::Mtype_t)
    ccall((:zCreate_Dense_Matrix, libsuperlu), Cvoid,
          (Ref{SuperMatrix}, Cint, Cint, Ptr{ComplexF64}, Cint, 
           Stype_t, Dtype_t, Mtype_t),
          B, m, n, nzval, lda, stype, dtype, mtype)
end

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

# Simple driver for complex double (zgssv)
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

# Symbolic factorization
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

# Triangular solve
function zgstrs!(trans::trans_t, L::SuperMatrix, U::SuperMatrix,
                 perm_c::Ptr{Cint}, perm_r::Ptr{Cint},
                 B::SuperMatrix, stat::SuperLUStat_t, info::Ref{Cint})
    ccall((:zgstrs, libsuperlu), Cvoid,
          (trans_t, Ref{SuperMatrix}, Ref{SuperMatrix}, Ptr{Cint}, Ptr{Cint},
           Ref{SuperMatrix}, Ref{SuperLUStat_t}, Ref{Cint}),
          trans, L, U, perm_c, perm_r, B, stat, info)
end

# Permutation for preprocessing
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
