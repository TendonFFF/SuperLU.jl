using TestItemRunner

@run_package_tests

@testitem "Basic complex solve" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    
    # Create a simple sparse complex matrix
    A = sparse([1.0+1.0im 2.0+0im 0.0; 
                3.0-1.0im 4.0+2.0im 1.0+0im; 
                0.0 1.0+1.0im 5.0-1.0im])
    b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]
    
    # Test direct interface
    F = SuperLU.SuperLUFactorize(A)
    SuperLU.factorize!(F)
    x = copy(b)
    SuperLU.superlu_solve!(F, x)
    
    # Check solution
    @test norm(A * x - b) < 1e-10
end

@testitem "LinearSolve integration" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    using LinearSolve
    
    # Create a sparse complex matrix
    A = sparse([1.0+1.0im 2.0+0im 0.0; 
                3.0-1.0im 4.0+2.0im 1.0+0im; 
                0.0 1.0+1.0im 5.0-1.0im])
    b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]
    
    # Test with LinearSolve
    prob = LinearProblem(A, b)
    sol = solve(prob, SuperLUFactorization())
    
    @test norm(A * sol.u - b) < 1e-10
end

@testitem "Reuse symbolic factorization" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    using LinearSolve
    
    # Create a sparse matrix
    A1 = sparse([4.0+1.0im 1.0+0im 0.0; 
                 1.0-1.0im 4.0+2.0im 1.0+0im; 
                 0.0 1.0+1.0im 4.0-1.0im])
    b1 = [1.0+0im, 2.0+1.0im, 3.0-1.0im]
    
    # First solve with reuse_symbolic=true
    prob1 = LinearProblem(A1, b1)
    cache = init(prob1, SuperLUFactorization(reuse_symbolic=true))
    sol1 = solve!(cache)
    @test norm(A1 * sol1.u - b1) < 1e-10
    
    # Now change values (same pattern) and solve again
    A2 = sparse([5.0+2.0im 2.0+0im 0.0; 
                 2.0-1.0im 5.0+1.0im 2.0+0im; 
                 0.0 2.0+1.0im 5.0-2.0im])
    b2 = [2.0+1.0im, 3.0-1.0im, 4.0+0im]
    
    cache.A = A2
    cache.b = b2
    cache.isfresh = true
    sol2 = solve!(cache)
    
    @test norm(A2 * sol2.u - b2) < 1e-10
end

@testitem "No symbolic reuse" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    using LinearSolve
    
    A = sparse([4.0+1.0im 1.0+0im 0.0; 
                1.0-1.0im 4.0+2.0im 1.0+0im; 
                0.0 1.0+1.0im 4.0-1.0im])
    b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]
    
    # Test with reuse_symbolic=false
    prob = LinearProblem(A, b)
    sol = solve(prob, SuperLUFactorization(reuse_symbolic=false))
    
    @test norm(A * sol.u - b) < 1e-10
end

@testitem "Real matrix conversion" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    using LinearSolve
    
    # SuperLU should handle real matrices by converting to complex
    A = sparse([4.0 1.0 0.0; 
                1.0 4.0 1.0; 
                0.0 1.0 4.0])
    b = [1.0, 2.0, 3.0]
    
    prob = LinearProblem(A, b)
    sol = solve(prob, SuperLUFactorization())
    
    @test norm(A * sol.u - b) < 1e-10
end

@testitem "Larger system" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    using LinearSolve
    
    n = 100
    # Create a sparse diagonally dominant matrix
    A = spdiagm(-1 => fill(-1.0+0.5im, n-1), 
                 0 => fill(4.0+1.0im, n), 
                 1 => fill(-1.0-0.5im, n-1))
    b = randn(ComplexF64, n)
    
    prob = LinearProblem(A, b)
    sol = solve(prob, SuperLUFactorization())
    
    @test norm(A * sol.u - b) < 1e-8
end

@testitem "GPU availability check" begin
    using SuperLU
    
    # Test that is_gpu_available function works
    @test is_gpu_available() isa Bool
    
    # Without CUDA.jl loaded, GPU should not be available
    @test is_gpu_available() == false
end

@testitem "GPU factorization type" begin
    using SuperLU
    
    # Test that SuperLUGPUFactorization can be constructed
    gpu_fact = SuperLUGPUFactorization()
    @test gpu_fact.reuse_symbolic == true
    
    gpu_fact_no_reuse = SuperLUGPUFactorization(reuse_symbolic=false)
    @test gpu_fact_no_reuse.reuse_symbolic == false
end

@testitem "GPU LinearSolve integration (fallback to CPU)" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    using LinearSolve
    
    # Test GPU factorization which should fall back to CPU when no GPU available
    A = sparse([4.0+1.0im 1.0+0im 0.0; 
                1.0-1.0im 4.0+2.0im 1.0+0im; 
                0.0 1.0+1.0im 4.0-1.0im])
    b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]
    
    prob = LinearProblem(A, b)
    # This should work but emit a warning about GPU not being available
    sol = solve(prob, SuperLUGPUFactorization())
    
    @test norm(A * sol.u - b) < 1e-10
end

@testitem "SuperLUOptions construction" begin
    using SuperLU
    
    # Test default options
    opts = SuperLUOptions()
    @test opts.col_perm == COLAMD
    @test opts.row_perm == LargeDiag_MC64
    @test opts.equilibrate == true
    @test opts.diag_pivot_thresh == 1.0
    @test opts.symmetric_mode == false
    @test opts.iterative_refinement == NOREFINE
    @test opts.pivot_growth == false
    @test opts.condition_number == false
    @test opts.print_stats == false
    @test opts.replace_tiny_pivot == false
    
    # Test custom options
    opts2 = SuperLUOptions(
        col_perm = METIS_AT_PLUS_A,
        row_perm = NOROWPERM,
        equilibrate = false,
        diag_pivot_thresh = 0.5,
        symmetric_mode = true,
        iterative_refinement = SLU_DOUBLE
    )
    @test opts2.col_perm == METIS_AT_PLUS_A
    @test opts2.row_perm == NOROWPERM
    @test opts2.equilibrate == false
    @test opts2.diag_pivot_thresh == 0.5
    @test opts2.symmetric_mode == true
    @test opts2.iterative_refinement == SLU_DOUBLE
    
    # Test invalid diag_pivot_thresh
    @test_throws ArgumentError SuperLUOptions(diag_pivot_thresh = -0.1)
    @test_throws ArgumentError SuperLUOptions(diag_pivot_thresh = 1.5)
end

@testitem "Options are applied to C struct" begin
    using SuperLU
    using SparseArrays
    
    A = sparse([4.0+1.0im 1.0+0im 0.0; 
                1.0-1.0im 4.0+2.0im 1.0+0im; 
                0.0 1.0+1.0im 4.0-1.0im])
    
    # Test with various column permutation options
    for col_perm in [NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD]
        opts = SuperLUOptions(col_perm = col_perm)
        F = SuperLU.SuperLUFactorize(A; options=opts)
        
        # Verify user_options is stored
        @test F.user_options.col_perm == col_perm
        
        # Verify the C options struct has the correct value
        @test F.options.ColPerm == col_perm
    end
    
    # Test with various row permutation options
    for row_perm in [NOROWPERM, LargeDiag_MC64]
        opts = SuperLUOptions(row_perm = row_perm)
        F = SuperLU.SuperLUFactorize(A; options=opts)
        @test F.user_options.row_perm == row_perm
        @test F.options.RowPerm == row_perm
    end
    
    # Test equilibrate option
    opts_eq = SuperLUOptions(equilibrate = true)
    F_eq = SuperLU.SuperLUFactorize(A; options=opts_eq)
    @test F_eq.options.Equil == SuperLU.YES
    
    opts_no_eq = SuperLUOptions(equilibrate = false)
    F_no_eq = SuperLU.SuperLUFactorize(A; options=opts_no_eq)
    @test F_no_eq.options.Equil == SuperLU.NO
    
    # Test iterative refinement options
    for iter_ref in [NOREFINE, SLU_SINGLE, SLU_DOUBLE, SLU_EXTRA]
        opts = SuperLUOptions(iterative_refinement = iter_ref)
        F = SuperLU.SuperLUFactorize(A; options=opts)
        @test F.options.IterRefine == iter_ref
    end
    
    # Test diag_pivot_thresh
    opts_thresh = SuperLUOptions(diag_pivot_thresh = 0.7)
    F_thresh = SuperLU.SuperLUFactorize(A; options=opts_thresh)
    @test F_thresh.options.DiagPivotThresh ≈ 0.7
    
    # Test symmetric_mode
    opts_sym = SuperLUOptions(symmetric_mode = true)
    F_sym = SuperLU.SuperLUFactorize(A; options=opts_sym)
    @test F_sym.options.SymmetricMode == SuperLU.YES
    
    # Test diagnostic options
    opts_diag = SuperLUOptions(
        pivot_growth = true,
        condition_number = true,
        print_stats = false,
        replace_tiny_pivot = true
    )
    F_diag = SuperLU.SuperLUFactorize(A; options=opts_diag)
    @test F_diag.options.PivotGrowth == SuperLU.YES
    @test F_diag.options.ConditionNumber == SuperLU.YES
    @test F_diag.options.PrintStat == SuperLU.NO
    @test F_diag.options.ReplaceTinyPivot == SuperLU.YES
end

@testitem "Solve with custom options" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    
    A = sparse([4.0+1.0im 1.0+0im 0.0; 
                1.0-1.0im 4.0+2.0im 1.0+0im; 
                0.0 1.0+1.0im 4.0-1.0im])
    b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]
    
    # Test with custom options
    opts = SuperLUOptions(
        col_perm = MMD_AT_PLUS_A,
        equilibrate = true
    )
    
    F = SuperLU.SuperLUFactorize(A; options=opts)
    SuperLU.factorize!(F)
    x = copy(b)
    SuperLU.superlu_solve!(F, x)
    
    @test norm(A * x - b) < 1e-10
end

@testitem "LinearSolve with custom options" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    using LinearSolve
    
    A = sparse([4.0+1.0im 1.0+0im 0.0; 
                1.0-1.0im 4.0+2.0im 1.0+0im; 
                0.0 1.0+1.0im 4.0-1.0im])
    b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]
    
    # Test with custom options via LinearSolve
    opts = SuperLUOptions(
        col_perm = MMD_AT_PLUS_A,
        iterative_refinement = SLU_DOUBLE
    )
    
    prob = LinearProblem(A, b)
    sol = solve(prob, SuperLUFactorization(options=opts))
    
    @test norm(A * sol.u - b) < 1e-10
end
