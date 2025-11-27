# Solver Options

SuperLU.jl provides configurable options for customizing solver behavior. These options control various aspects of the factorization and solve process, including permutation strategies, pivoting, and iterative refinement.

## SuperLUOptions

The [`SuperLUOptions`](@ref) struct provides a user-friendly interface for configuring the SuperLU solver. All options have sensible defaults for general use cases.

```julia
using SuperLU

# Create options with all defaults
opts = SuperLUOptions()

# Create options with custom settings
opts = SuperLUOptions(
    col_perm = METIS_AT_PLUS_A,      # Use METIS for column ordering
    equilibrate = true,               # Enable matrix equilibration
    iterative_refinement = SLU_DOUBLE # Enable double precision refinement
)

# Use options with factorization
F = SuperLUFactorize(A; options=opts)
factorize!(F)
```

## Permutation Strategies

### Column Permutation (`col_perm`)

Column permutation reorders the columns of the matrix to reduce fill-in during LU factorization. The choice of strategy can significantly impact both performance and memory usage.

| Strategy | Description | Best For |
|----------|-------------|----------|
| `NATURAL` | No permutation | Already well-ordered matrices |
| `MMD_ATA` | Minimum degree on AᵀA | General non-symmetric matrices |
| `MMD_AT_PLUS_A` | Minimum degree on Aᵀ+A | Nearly symmetric matrices |
| `COLAMD` | Column approximate minimum degree | General matrices (default) |
| `METIS_AT_PLUS_A` | METIS nested dissection | Large matrices where quality matters |

**Example:**
```julia
# For a symmetric or nearly symmetric matrix
opts = SuperLUOptions(col_perm = MMD_AT_PLUS_A)

# For large matrices where fill-in reduction is critical
opts = SuperLUOptions(col_perm = METIS_AT_PLUS_A)
```

### Row Permutation (`row_perm`)

Row permutation is used for numerical stability during factorization. It helps ensure that the diagonal elements are sufficiently large.

| Strategy | Description | Best For |
|----------|-------------|----------|
| `NOROWPERM` | No row permutation | Well-conditioned matrices |
| `LargeDiag_MC64` | MC64 weighted matching | General matrices (recommended) |
| `LargeDiag_HWPM` | Hungarian algorithm | Alternative to MC64 |

**Example:**
```julia
# For well-conditioned matrices (faster)
opts = SuperLUOptions(row_perm = NOROWPERM)

# For general matrices (more stable)
opts = SuperLUOptions(row_perm = LargeDiag_MC64)
```

## Numerical Options

### Equilibration (`equilibrate`)

Matrix equilibration (scaling) can improve numerical stability by making the matrix rows and columns have similar norms.

```julia
# Enable equilibration (default: true)
opts = SuperLUOptions(equilibrate = true)

# Disable equilibration for already well-scaled matrices
opts = SuperLUOptions(equilibrate = false)
```

### Diagonal Pivot Threshold (`diag_pivot_thresh`)

Controls the preference for diagonal elements during pivoting. The value should be between 0.0 and 1.0.

- `1.0`: Always prefer diagonal elements (partial pivoting, default)
- `0.0`: Never prefer diagonal elements (complete pivoting)
- Values in between: Threshold for when to consider off-diagonal elements

```julia
# Default: partial pivoting
opts = SuperLUOptions(diag_pivot_thresh = 1.0)

# More aggressive pivoting for ill-conditioned matrices
opts = SuperLUOptions(diag_pivot_thresh = 0.1)
```

### Symmetric Mode (`symmetric_mode`)

When enabled, uses symmetric storage and factorization patterns. Only applicable to structurally symmetric matrices.

```julia
# For symmetric matrices
opts = SuperLUOptions(symmetric_mode = true)
```

## Iterative Refinement

Iterative refinement improves solution accuracy at the cost of additional computation. It is particularly useful for ill-conditioned systems.

| Strategy | Description | Performance Impact |
|----------|-------------|-------------------|
| `NOREFINE` | No refinement | Fastest |
| `SLU_SINGLE` | Single precision refinement | Minor overhead |
| `SLU_DOUBLE` | Double precision refinement | Moderate overhead |
| `SLU_EXTRA` | Extra precision refinement | Most accurate but slowest |

**Example:**
```julia
# For maximum accuracy
opts = SuperLUOptions(iterative_refinement = SLU_DOUBLE)

# For speed (no refinement)
opts = SuperLUOptions(iterative_refinement = NOREFINE)
```

## Diagnostic Options

### Pivot Growth (`pivot_growth`)

Computes the reciprocal pivot growth factor, which can be used to assess numerical stability.

```julia
opts = SuperLUOptions(pivot_growth = true)
```

### Condition Number (`condition_number`)

Estimates the condition number of the matrix. Useful for detecting ill-conditioned systems.

```julia
opts = SuperLUOptions(condition_number = true)
```

### Print Statistics (`print_stats`)

Enables printing of solver statistics for debugging and performance analysis.

```julia
opts = SuperLUOptions(print_stats = true)
```

## Advanced Options

### Replace Tiny Pivots (`replace_tiny_pivot`)

When enabled, very small pivot elements are replaced to improve stability. This can help with nearly singular matrices.

```julia
opts = SuperLUOptions(replace_tiny_pivot = true)
```

## Using Options with LinearSolve.jl

Options can also be passed when using the LinearSolve.jl interface:

```julia
using SuperLU
using LinearSolve
using SparseArrays

A = sparse([4.0+1.0im 1.0+0im; 1.0-1.0im 4.0+2.0im])
b = [1.0+0im, 2.0+1.0im]

# Create options
opts = SuperLUOptions(
    col_perm = METIS_AT_PLUS_A,
    iterative_refinement = SLU_DOUBLE
)

# Use with LinearSolve
prob = LinearProblem(A, b)
sol = solve(prob, SuperLUFactorization(options = opts))
```

## Preset Options

SuperLU.jl provides pre-configured option objects for common use cases, so you don't need to manually configure each option.

### ILL_CONDITIONED_OPTIONS

Optimized for solving ill-conditioned systems:

```julia
using SuperLU
using SparseArrays

A = sparse([4.0 1.0; 1.0 4.0])
b = [1.0, 2.0]

F = SuperLUFactorize(A; options=ILL_CONDITIONED_OPTIONS)
factorize!(F)
x = superlu_solve(F, b)
```

### PERFORMANCE_OPTIONS

Optimized for maximum performance (use only for well-conditioned systems):

```julia
F = SuperLUFactorize(A; options=PERFORMANCE_OPTIONS)
```

### ACCURACY_OPTIONS

Optimized for maximum accuracy:

```julia
F = SuperLUFactorize(A; options=ACCURACY_OPTIONS)
```

### SYMMETRIC_OPTIONS

Optimized for symmetric or nearly symmetric matrices:

```julia
# Check if matrix has symmetric structure
if issymmetric_structure(A)
    F = SuperLUFactorize(A; options=SYMMETRIC_OPTIONS)
end
```

## Matrix Symmetry Checking

SuperLU.jl provides utility functions to analyze matrix symmetry and suggest appropriate options.

### Checking Sparsity Pattern Symmetry

```julia
using SuperLU
using SparseArrays

A_sym = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
issymmetric_structure(A_sym)  # true

A_asym = sparse([4.0 1.0 0.0; 0.0 4.0 1.0; 0.0 1.0 4.0])
issymmetric_structure(A_asym)  # false (no entry at (2,1))
```

### Checking Value Symmetry

```julia
# Check if values are approximately symmetric (for real matrices)
issymmetric_approx(A_sym)  # true

# Check if values are approximately Hermitian (for complex matrices)
ishermitian_approx(A_complex)
```

### Automatic Option Suggestion

```julia
# Analyze matrix and get suggested options
opts = suggest_options(A)
F = SuperLUFactorize(A; options=opts)
```

## Recommended Settings

### General Purpose
```julia
opts = SuperLUOptions()  # All defaults are good for most cases
```

### Maximum Performance
```julia
opts = SuperLUOptions(
    col_perm = COLAMD,
    row_perm = NOROWPERM,
    equilibrate = false,
    iterative_refinement = NOREFINE
)
# Or use the preset:
opts = PERFORMANCE_OPTIONS
```

### Maximum Accuracy
```julia
opts = SuperLUOptions(
    col_perm = MMD_AT_PLUS_A,
    row_perm = LargeDiag_MC64,
    equilibrate = true,
    iterative_refinement = SLU_DOUBLE,
    diag_pivot_thresh = 1.0
)
# Or use the preset:
opts = ACCURACY_OPTIONS
```

### Ill-Conditioned Matrices
```julia
opts = SuperLUOptions(
    equilibrate = true,
    iterative_refinement = SLU_EXTRA,
    replace_tiny_pivot = true,
    condition_number = true  # To monitor the condition
)
# Or use the preset:
opts = ILL_CONDITIONED_OPTIONS
```
