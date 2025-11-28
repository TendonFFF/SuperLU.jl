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

    A = sprand(ComplexF64, 10000, 10000, 0.0001) + I
    b = randn(ComplexF64, 10000)
    prob = LinearProblem(A, b)
    sol = solve(prob, SuperLUFactorization())
    @test norm(A * sol.u - b) < 1e-8
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

@testitem "SuperLUOptions construction" begin
    using SuperLU
    
    # Test default options
    opts = SuperLUOptions()
    @test opts.col_perm == COLAMD
    @test opts.diag_pivot_thresh == 1.0
    @test opts.symmetric_mode == false
    @test opts.print_stats == false
    
    # Test custom options
    opts2 = SuperLUOptions(
        col_perm = MMD_AT_PLUS_A,
        diag_pivot_thresh = 0.5,
        symmetric_mode = true,
        print_stats = true
    )
    @test opts2.col_perm == MMD_AT_PLUS_A
    @test opts2.diag_pivot_thresh == 0.5
    @test opts2.symmetric_mode == true
    @test opts2.print_stats == true
    
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
    
    # Test diag_pivot_thresh
    opts_thresh = SuperLUOptions(diag_pivot_thresh = 0.7)
    F_thresh = SuperLU.SuperLUFactorize(A; options=opts_thresh)
    @test F_thresh.options.diag_pivot_thresh ≈ 0.7
    
    # Test symmetric_mode
    opts_sym = SuperLUOptions(symmetric_mode = true)
    F_sym = SuperLU.SuperLUFactorize(A; options=opts_sym)
    @test F_sym.options.SymmetricMode == SuperLU.YES
    
    # Test print_stats option
    opts_print = SuperLUOptions(print_stats = true)
    F_print = SuperLU.SuperLUFactorize(A; options=opts_print)
    @test F_print.options.PrintStat == SuperLU.YES
    
    opts_no_print = SuperLUOptions(print_stats = false)
    F_no_print = SuperLU.SuperLUFactorize(A; options=opts_no_print)
    @test F_no_print.options.PrintStat == SuperLU.NO
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
    opts = SuperLUOptions(col_perm = MMD_AT_PLUS_A)
    
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
    opts = SuperLUOptions(col_perm = MMD_AT_PLUS_A)
    
    prob = LinearProblem(A, b)
    sol = solve(prob, SuperLUFactorization(options=opts))
    
    @test norm(A * sol.u - b) < 1e-10
end

@testitem "Float64 direct solve" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    
    # Create a real double precision matrix
    A = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    b = [1.0, 2.0, 3.0]
    
    # Test direct interface with Float64
    F = SuperLU.SuperLUFactorize(A)
    SuperLU.factorize!(F)
    x = copy(b)
    SuperLU.superlu_solve!(F, x)
    
    @test norm(A * x - b) < 1e-10
end

@testitem "Float32 direct solve" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    
    # Create a real single precision matrix
    A = sparse(Float32[4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    b = Float32[1.0, 2.0, 3.0]
    
    # Test direct interface with Float32
    F = SuperLU.SuperLUFactorize(A)
    SuperLU.factorize!(F)
    x = copy(b)
    SuperLU.superlu_solve!(F, x)
    
    @test norm(A * x - b) < 1e-5
end

@testitem "ComplexF32 direct solve" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    
    # Create a complex single precision matrix
    A = sparse(ComplexF32[4.0+1.0im 1.0+0im 0.0; 
                          1.0-1.0im 4.0+2.0im 1.0+0im; 
                          0.0 1.0+1.0im 4.0-1.0im])
    b = ComplexF32[1.0+0im, 2.0+1.0im, 3.0-1.0im]
    
    # Test direct interface with ComplexF32
    F = SuperLU.SuperLUFactorize(A)
    SuperLU.factorize!(F)
    x = copy(b)
    SuperLU.superlu_solve!(F, x)
    
    @test norm(A * x - b) < 1e-5
end

@testitem "Float64 LinearSolve integration" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    using LinearSolve
    
    # Create a real double precision matrix
    A = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    b = [1.0, 2.0, 3.0]
    
    # Test with LinearSolve - Float64 should work directly now
    prob = LinearProblem(A, b)
    sol = solve(prob, SuperLUFactorization())
    
    @test norm(A * sol.u - b) < 1e-10
end

@testitem "Preset options" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    
    A = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    b = [1.0, 2.0, 3.0]
    
    # Test ILL_CONDITIONED_OPTIONS
    @test ILL_CONDITIONED_OPTIONS.col_perm == MMD_AT_PLUS_A
    @test ILL_CONDITIONED_OPTIONS.diag_pivot_thresh == 1.0
    
    F1 = SuperLU.SuperLUFactorize(A; options=ILL_CONDITIONED_OPTIONS)
    SuperLU.factorize!(F1)
    x1 = copy(b)
    SuperLU.superlu_solve!(F1, x1)
    @test norm(A * x1 - b) < 1e-10
    
    # Test PERFORMANCE_OPTIONS
    @test PERFORMANCE_OPTIONS.col_perm == COLAMD
    @test PERFORMANCE_OPTIONS.diag_pivot_thresh == 1.0
    
    F2 = SuperLU.SuperLUFactorize(A; options=PERFORMANCE_OPTIONS)
    SuperLU.factorize!(F2)
    x2 = copy(b)
    SuperLU.superlu_solve!(F2, x2)
    @test norm(A * x2 - b) < 1e-10
    
    # Test ACCURACY_OPTIONS
    @test ACCURACY_OPTIONS.col_perm == MMD_AT_PLUS_A
    @test ACCURACY_OPTIONS.diag_pivot_thresh == 1.0
    
    F3 = SuperLU.SuperLUFactorize(A; options=ACCURACY_OPTIONS)
    SuperLU.factorize!(F3)
    x3 = copy(b)
    SuperLU.superlu_solve!(F3, x3)
    @test norm(A * x3 - b) < 1e-10
    
    # Test SYMMETRIC_OPTIONS
    @test SYMMETRIC_OPTIONS.symmetric_mode == true
    @test SYMMETRIC_OPTIONS.col_perm == MMD_AT_PLUS_A
    
    F4 = SuperLU.SuperLUFactorize(A; options=SYMMETRIC_OPTIONS)
    SuperLU.factorize!(F4)
    x4 = copy(b)
    SuperLU.superlu_solve!(F4, x4)
    @test norm(A * x4 - b) < 1e-10
end

@testitem "Symmetry checking" begin
    using SuperLU
    using SparseArrays
    
    # Symmetric matrix (both structure and values)
    A_sym = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    @test issymmetric_structure(A_sym) == true
    @test issymmetric_approx(A_sym) == true
    
    # Asymmetric structure
    A_asym_struct = sparse([4.0 1.0 0.0; 0.0 4.0 1.0; 0.0 1.0 4.0])
    @test issymmetric_structure(A_asym_struct) == false
    @test issymmetric_approx(A_asym_struct) == false
    
    # Symmetric structure but asymmetric values
    A_sym_struct = sparse([4.0 1.0 0.0; 2.0 4.0 1.0; 0.0 1.0 4.0])
    @test issymmetric_structure(A_sym_struct) == true
    @test issymmetric_approx(A_sym_struct) == false
    
    # Non-square matrix
    A_rect = sparse([4.0 1.0; 1.0 4.0; 0.0 1.0])
    @test issymmetric_structure(A_rect) == false
    
    # Complex Hermitian matrix
    A_herm = sparse([4.0+0im 1.0+1.0im 0.0; 
                     1.0-1.0im 4.0+0im 1.0+1.0im; 
                     0.0 1.0-1.0im 4.0+0im])
    @test issymmetric_structure(A_herm) == true
    @test ishermitian_approx(A_herm) == true
    
    # Test suggest_options
    opts = suggest_options(A_sym)
    @test opts.col_perm == MMD_AT_PLUS_A
    @test opts.symmetric_mode == true
end

@testitem "Larger real system" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    using LinearSolve
    
    n = 100
    # Create a sparse diagonally dominant real matrix
    A = spdiagm(-1 => fill(-1.0, n-1), 
                 0 => fill(4.0, n), 
                 1 => fill(-1.0, n-1))
    b = randn(n)
    
    prob = LinearProblem(A, b)
    sol = solve(prob, SuperLUFactorization())
    
    @test norm(A * sol.u - b) < 1e-8
end

@testitem "nthreads setting" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    
    # Create a sparse matrix
    A = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    b = [1.0, 2.0, 3.0]
    
    # Test with single thread (default)
    F1 = SuperLU.SuperLUFactorize(A)
    @test F1.nthreads == 1
    SuperLU.factorize!(F1)
    x1 = copy(b)
    SuperLU.superlu_solve!(F1, x1)
    @test norm(A * x1 - b) < 1e-10
    
    # Test with multiple threads
    F2 = SuperLU.SuperLUFactorize(A; nthreads=2)
    @test F2.nthreads == 2
    SuperLU.factorize!(F2)
    x2 = copy(b)
    SuperLU.superlu_solve!(F2, x2)
    @test norm(A * x2 - b) < 1e-10
    
    F4 = SuperLU.SuperLUFactorize(A; nthreads=4)
    @test F4.nthreads == 4
    SuperLU.factorize!(F4)
    x4 = copy(b)
    SuperLU.superlu_solve!(F4, x4)
    @test norm(A * x4 - b) < 1e-10
    
    # Test invalid nthreads
    @test_throws ArgumentError SuperLU.SuperLUFactorize(A; nthreads=0)
    @test_throws ArgumentError SuperLU.SuperLUFactorize(A; nthreads=-1)
end

@testitem "Multithreaded solve on larger system" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    
    # Create a larger sparse system where threading may be beneficial
    n = 2000
    # Create a sparse diagonally dominant matrix with multiple diagonals
    A = spdiagm(-2 => fill(-0.5, n-2),
                -1 => fill(-1.0, n-1), 
                 0 => fill(6.0, n), 
                 1 => fill(-1.0, n-1),
                 2 => fill(-0.5, n-2))
    b = randn(n)
    
    # Solve with 1 thread
    F1 = SuperLU.SuperLUFactorize(A; nthreads=1)
    SuperLU.factorize!(F1)
    x1 = copy(b)
    SuperLU.superlu_solve!(F1, x1)
    @test norm(A * x1 - b) < 1e-8
    
    # Solve with 4 threads - should produce the same result
    F4 = SuperLU.SuperLUFactorize(A; nthreads=4)
    SuperLU.factorize!(F4)
    x4 = copy(b)
    SuperLU.superlu_solve!(F4, x4)
    @test norm(A * x4 - b) < 1e-8
    
    # Solutions should be identical (up to numerical precision)
    @test norm(x1 - x4) < 1e-10
end

@testitem "Multithreaded performance comparison" begin
    using SuperLU
    using SparseArrays
    using LinearAlgebra
    
    # Create a larger sparse system for timing comparison
    n = 20000
    # Create a sparse banded matrix
    A = sprand(n, n, 1/n) + I
    # A = spdiagm(-5 => fill(-0.2, n-5),
    #             -2 => fill(-0.5, n-2),
    #             -1 => fill(-1.0, n-1), 
    #              0 => fill(8.0, n), 
    #              1 => fill(-1.0, n-1),
    #              2 => fill(-0.5, n-2),
    #              5 => fill(-0.2, n-5))
    b = randn(n)
    
    # Warm-up run
    F_warmup = SuperLU.SuperLUFactorize(A; nthreads=1)
    SuperLU.factorize!(F_warmup)
    
    # Time with 1 thread
    t1 = @elapsed begin
        for _ in 1:3
            F = SuperLU.SuperLUFactorize(A; nthreads=1)
            SuperLU.factorize!(F)
            x = copy(b)
            SuperLU.superlu_solve!(F, x)
        end
    end
    
    # Time with multiple threads (use available threads, at least 2)
    nthreads = 4
    t_mt = @elapsed begin
        for _ in 1:3
            F = SuperLU.SuperLUFactorize(A; nthreads=nthreads)
            SuperLU.factorize!(F)
            x = copy(b)
            SuperLU.superlu_solve!(F, x)
        end
    end
    
    # Both should produce correct solutions
    F_final = SuperLU.SuperLUFactorize(A; nthreads=nthreads)
    SuperLU.factorize!(F_final)
    x_final = copy(b)
    SuperLU.superlu_solve!(F_final, x_final)
    @test norm(A * x_final - b) < 1e-8
    
    # Print timing info (not a strict test since threading benefit depends on hardware)
    @info "Timing comparison" single_thread_time=t1 multi_thread_time=t_mt nthreads=nthreads speedup=t1/t_mt
end
