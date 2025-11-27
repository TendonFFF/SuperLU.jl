module SuperLU

using LinearAlgebra
using SparseArrays
using SuperLU_jll

# Include submodules
include("types.jl")
include("wrappers.jl")
include("interface.jl")
include("linearsolve.jl")

# Export main types and functions
export SuperLUFactorization
export SuperLUFactorize, superlu_solve!

end # module
