using Test
using SuperLU
using SparseArrays
using LinearAlgebra
using LinearSolve

@testset "SuperLU.jl" begin
    @testset "Basic complex solve" begin
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
    
    @testset "LinearSolve integration" begin
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
    
    @testset "Reuse symbolic factorization" begin
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
    
    @testset "No symbolic reuse" begin
        A = sparse([4.0+1.0im 1.0+0im 0.0; 
                    1.0-1.0im 4.0+2.0im 1.0+0im; 
                    0.0 1.0+1.0im 4.0-1.0im])
        b = [1.0+0im, 2.0+1.0im, 3.0-1.0im]
        
        # Test with reuse_symbolic=false
        prob = LinearProblem(A, b)
        sol = solve(prob, SuperLUFactorization(reuse_symbolic=false))
        
        @test norm(A * sol.u - b) < 1e-10
    end
    
    @testset "Real matrix conversion" begin
        # SuperLU should handle real matrices by converting to complex
        A = sparse([4.0 1.0 0.0; 
                    1.0 4.0 1.0; 
                    0.0 1.0 4.0])
        b = [1.0, 2.0, 3.0]
        
        prob = LinearProblem(A, b)
        sol = solve(prob, SuperLUFactorization())
        
        @test norm(A * sol.u - b) < 1e-10
    end
    
    @testset "Larger system" begin
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
end
