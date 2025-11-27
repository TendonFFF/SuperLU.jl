module SuperLU

using LinearAlgebra
using SparseArrays
using SuperLU_jll

# Include submodules
include("types.jl")
include("wrappers.jl")
include("interface.jl")

# Export main types and functions
export SuperLUFactorize, factorize!, superlu_solve!, superlu_solve, update_matrix!

# Export options
export SuperLUOptions

# Export preset options
export ILL_CONDITIONED_OPTIONS, PERFORMANCE_OPTIONS, ACCURACY_OPTIONS, SYMMETRIC_OPTIONS

# Export symmetry checking utilities
export issymmetric_structure, ishermitian_approx, issymmetric_approx, suggest_options

# Export supported types alias
export SuperLUTypes

# Export trans_t enum for use in extensions
export trans_t, NOTRANS, TRANS, CONJ

# Export column permutation strategies
export colperm_t, NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD, METIS_AT_PLUS_A

# Export row permutation strategies
export rowperm_t, NOROWPERM, LargeDiag_MC64, LargeDiag_HWPM

# Export iterative refinement options
export IterRefine_t, NOREFINE, SLU_SINGLE, SLU_DOUBLE, SLU_EXTRA

# Export yes/no enum for checking options
export yes_no_t, YES, NO

end # module
