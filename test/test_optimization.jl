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
        @test sol.stats.fevals == 60 * 2 * 4 + 1          # + the final iterate
        @test SciMLBase.successful_retcode(sol)
        # scalar bounds pass through unchanged
        probs = OptimizationProblem(sphere, u0; lb = -1.0, ub = 1.0)
        sols = SciMLBase.solve(probs, PolyStepOptimizer(); maxiters = 5,
            batched = X -> vec(sum(abs2, X; dims = 1)))
        @test all(-1.0 .<= sols.u .<= 1.0)
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
        sol = @test_logs (:warn, r"one candidate at a time") match_mode = :any SciMLBase.solve(
            prob, PolyStepOptimizer(); maxiters = 20, rng = Xoshiro(2))
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test isfinite(sol.objective)
    end

    @testset "MaxSense on the batched path" begin
        prob = OptimizationProblem(OptimizationFunction((x, p) -> -sum(abs2, x .- 1)),
            zeros(2); sense = MaxSense)
        sol = SciMLBase.solve(prob, PolyStepOptimizer(step_radius = 0.1); maxiters = 200,
            batched = X -> -vec(sum(abs2, X .- 1; dims = 1)), rng = Xoshiro(3))
        @test norm(sol.u .- 1) < 0.3                  # maximized, not minimized
        @test -0.1 < sol.objective <= 0               # reported in the user's sign
    end

    @testset "maxtime -> MaxTime; abstol/reltol warn" begin
        prob = OptimizationProblem(sphere, u0)
        slow = X -> (sleep(0.002); vec(sum(abs2, X; dims = 1)))
        sol = SciMLBase.solve(prob, PolyStepOptimizer(); maxiters = 10_000, maxtime = 0.05,
            batched = slow, rng = Xoshiro(4))
        @test sol.retcode == SciMLBase.ReturnCode.MaxTime
        @test sol.stats.iterations < 10_000
        @test_logs (:warn, r"abstol/reltol") match_mode = :any SciMLBase.solve(prob,
            PolyStepOptimizer(); maxiters = 2, abstol = 1e-3,
            batched = X -> vec(sum(abs2, X; dims = 1)))
    end

    @testset "no finite objective -> Failure at u0" begin
        prob = OptimizationProblem(sphere, u0)
        sol = SciMLBase.solve(prob, PolyStepOptimizer(); maxiters = 5,
            batched = X -> fill(NaN, size(X, 2)))
        @test sol.retcode == SciMLBase.ReturnCode.Failure
        @test !SciMLBase.successful_retcode(sol)
        @test sol.u == u0
    end

    @testset "Matrix u0 keeps its shape" begin
        target = [2.0 1.0; 1.0 2.0]
        fm(u, p) = sum(abs2, u * u' - target)          # needs u as a 2x2 matrix
        U0 = [1.0 0.0; 0.0 1.0]
        prob = OptimizationProblem(fm, U0; lb = fill(-3.0, 2, 2), ub = fill(3.0, 2, 2))
        sol = @test_logs min_level = Logging.Error SciMLBase.solve(prob, PolyStepOptimizer();
            maxiters = 50, rng = Xoshiro(5))
        @test sol.u isa Matrix{Float64} && size(sol.u) == size(U0)
        @test sol.objective < fm(U0, nothing)
    end
end
