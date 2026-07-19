# Optimization.jl (SciML) extension smoke tests; run when OptimizationBase is present
# being resolvable in the test environment (see runtests.jl).
using OptimizationBase
using OptimizationBase: SciMLBase

@testset "optimization extension" begin
    sphere(u, p) = sum(abs2, u)
    u0 = fill(2.0, 4)

    @testset "solve + Success retcode + bounds" begin
        prob = OptimizationProblem(sphere, u0; lb = fill(-3.0, 4), ub = fill(3.0, 4))
        sol = SciMLBase.solve(prob, PolyStepOptimizer(); maxiters = 60,
            batched = X -> vec(sum(abs2, X; dims = 1)), rng = Xoshiro(0))
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test sol.objective < sphere(u0, nothing)
        @test all(-3.0 .<= sol.u .<= 3.0)
        @test sol.stats.fevals == 60 * 2 * 4
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "callback halt -> Terminated" begin
        prob = OptimizationProblem(sphere, u0)
        ncalls = Ref(0)
        cb = (state, obj) -> (ncalls[] += 1; ncalls[] >= 5)
        sol = SciMLBase.solve(prob, PolyStepOptimizer(); maxiters = 100, callback = cb,
            batched = X -> vec(sum(abs2, X; dims = 1)), rng = Xoshiro(1))
        @test sol.retcode == SciMLBase.ReturnCode.Terminated
        @test sol.stats.iterations == 5
    end

    @testset "scalar objective path (warns once, still works)" begin
        prob = OptimizationProblem(sphere, u0)
        sol = SciMLBase.solve(prob, PolyStepOptimizer(); maxiters = 20, rng = Xoshiro(2))
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test isfinite(sol.objective)
    end
end
