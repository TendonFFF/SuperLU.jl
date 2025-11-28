module SuperLU

using LinearAlgebra
using SparseArrays
using SuperLU_MT_jll

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
export colperm_t, NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD

# Export yes/no enum for checking options
export yes_no_t, YES, NO

# Include LinearSolve.jl integration
include("linearsolve.jl")

# Export LinearSolve-compatible type
export SuperLUFactorization

end # module
