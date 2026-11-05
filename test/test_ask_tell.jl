using PolyStep
using Statistics: mean

sphere_b(X) = vec(sum(abs2, X; dims = 1))

@testset "ask/tell" begin
    @testset "construction + validation" begin
        es = PolyStepES(4; num_particles = 3)
        @test popsize(es) == 3 * 2 * 4
        @test size(es.X) == (4, 3)
        @test all(iszero, es.X)
        @test_throws ArgumentError PolyStepES(0)
        @test_throws ArgumentError PolyStepES(3; num_particles = 0)
        @test_throws ArgumentError PolyStepES(3; epsilon = 0.0)
        @test_throws ArgumentError PolyStepES(3; step_radius = -1.0)
        @test_throws ArgumentError PolyStepES(3; step_radius = Inf)
        @test_throws ArgumentError PolyStepES(3; x0 = zeros(2))
        @test_throws ArgumentError PolyStepES(3; lb = 0.0)
        # non-finite x0 would freeze every particle
        @test_throws ArgumentError PolyStepES(2; x0 = [0.0, NaN])
        @test_throws ArgumentError PolyStepES(2; num_particles = 2, x0 = [0.0 Inf; 0.0 0.0])
        @test_throws ArgumentError PolyStepES(3; scale_cost = :meen)
        @test_throws ArgumentError PolyStepES(3; scale_cost = -1.0)
        @test_throws ArgumentError PolyStepES(3; scale_cost = 0.0)
        @test PolyStepES(3; scale_cost = :max_cost) isa PolyStepES
        @test PolyStepES(3; scale_cost = 2.0) isa PolyStepES
        @test PolyStepES(3; scale_cost = nothing) isa PolyStepES
        # x0 vector broadcasts to all particles
        es2 = PolyStepES(3; num_particles = 4, x0 = [1.0, 2.0, 3.0])
        @test es2.X == repeat([1.0 2.0 3.0]', 1, 4)
        X0 = randn(Xoshiro(1), 3, 2)
        es3 = PolyStepES(3; num_particles = 2, x0 = X0)
        @test es3.X == X0
        # Sinkhorn + single particle warns
        @test_logs (:warn, r"uniform transport plan") PolyStepES(3; solver = SinkhornSolver())
    end

    @testset "ask/tell state machine" begin
        es = PolyStepES(3; rng = Xoshiro(2))
        X = ask!(es)
        @test size(X) == (3, popsize(es))
        @test_throws ErrorException ask!(es)
        tell!(es, sphere_b(X))
        @test es.pending === nothing
        @test_throws ErrorException tell!(es, zeros(popsize(es)))
        X2 = ask!(es)
        @test_throws DimensionMismatch tell!(es, zeros(popsize(es) + 1))
        tell!(es, sphere_b(X2))
        @test es.evals == 2 * popsize(es)
    end

    @testset "candidate geometry" begin
        es = PolyStepES(4; num_particles = 2, step_radius = 0.7,
            x0 = [1.0, -1.0, 0.5, 2.0], rng = Xoshiro(3))
        X = ask!(es)
        d = 4
        for p in 1:2, i in 1:d

            plus = X[:, (p - 1) * 2d + i]
            minus = X[:, (p - 1) * 2d + d + i]
            @test isapprox((plus .+ minus) ./ 2, es.X[:, p]; atol = 1e-12)
            @test isapprox(norm(plus .- es.X[:, p]), 0.7; atol = 1e-10)
        end
    end

    @testset "best tracking + finite guard" begin
        es = PolyStepES(3; rng = Xoshiro(4))
        X = ask!(es)
        fit = sphere_b(X)
        tell!(es, fit)
        @test es.best_f == minimum(fit)
        @test es.best_x == X[:, argmin(fit)]
        Xheld = copy(es.X)
        ask!(es)
        tell!(es, fill(NaN, popsize(es)))
        @test es.X == Xheld
        @test es.best_f == minimum(fit)          # NaN never becomes the incumbent
    end

    @testset "ask! reuses its buffer; R doubles as rotation scratch" begin
        es = PolyStepES(3; rng = Xoshiro(6))
        A1 = ask!(es)
        tell!(es, sphere_b(A1))
        @test ask!(es) === A1
        for d in (5, 12)                          # static and LAPACK rotation paths
            es = PolyStepES(d; num_particles = 3, rng = Xoshiro(5))
            ask!(es)
            @test haar_rotations!(zeros(d, d, 3), zeros(d, d, 3), Xoshiro(5)) == es.R
        end
    end

    @testset "minimize scores the final iterate" begin
        # zero steps: only x0 is scored, so best_f is f(x0)
        es0 = minimize(sphere_b, 3; steps = 0, x0 = fill(2.0, 3))
        @test es0.best_f == 12.0 && es0.best_x == fill(2.0, 3) && es0.evals == 1
        # best_f includes the scored final iterate
        es = minimize(sphere_b, 10; steps = 300, x0 = fill(0.5, 10), rng = Xoshiro(0))
        @test es.best_f <= sphere_b(es.X)[1]
        @test es.evals == 300 * popsize(es) + 1
        rnd(X) = (X .= round.(X); X)
        esr = minimize(sphere_b, 3; steps = 0, repair = rnd, x0 = fill(0.4, 3))
        @test esr.best_x == zeros(3) && esr.best_f == 0.0
    end

    @testset "seeded determinism" begin
        run(seed) = begin
            es = PolyStepES(4; num_particles = 2, rng = Xoshiro(seed), x0 = ones(4))
            for _ in 1:5
                tell!(es, sphere_b(ask!(es)))
            end
            es.X
        end
        @test run(7) == run(7)
        @test run(7) != run(8)
    end

    @testset "translation equivariance" begin
        c = [0.3, -0.7, 1.1]
        s = [10.0, -5.0, 2.0]
        f1(X) = vec(sum(abs2, X .- c; dims = 1))
        f2(X) = vec(sum(abs2, X .- (c .+ s); dims = 1))
        run(f, x0, seed) = begin
            es = PolyStepES(3; x0, rng = Xoshiro(seed), epsilon = 0.1, step_radius = 0.3)
            for _ in 1:10
                tell!(es, f(ask!(es)))
            end
            es.X
        end
        X1 = run(f1, zeros(3), 11)
        X2 = run(f2, s, 11)
        @test isapprox(X2, X1 .+ s; rtol = 1e-9)
    end

    @testset "minimize on sphere + rastrigin" begin
        es = minimize(sphere_b, 5; steps = 150, epsilon = 0.05, step_radius = 0.3,
            x0 = fill(2.0, 5), rng = Xoshiro(21))
        @test es.best_f < 1.0                     # from ||x||^2 = 20
        @test norm(mean(es)) < 2.0
        rast(X) = vec(sum(X .^ 2 .- 10 .* cos.(2 * pi .* X) .+ 10; dims = 1))
        esr = minimize(rast, 4; steps = 200, epsilon = 0.05, step_radius = 0.2,
            x0 = fill(1.5, 4), rng = Xoshiro(22))
        @test esr.best_f < rast(reshape(fill(1.5, 4), 4, 1))[1]
    end

    @testset "bounds + repair keep everything feasible" begin
        lo, hi = -0.5, 0.5
        es = PolyStepES(3; step_radius = 2.0, lb = lo, ub = hi, x0 = fill(0.4, 3),
            rng = Xoshiro(31))
        for _ in 1:5
            X = ask!(es)
            @test all(x -> lo <= x <= hi, X)
            tell!(es, sphere_b(X))
            @test all(x -> lo <= x <= hi, es.X)   # barycenter of box points stays in box
        end
        @test all(x -> lo <= x <= hi, es.best_x)
        rnd(X) = (X .= round.(X); X)
        esi = PolyStepES(3; step_radius = 1.4, repair = rnd, x0 = fill(2.6, 3),
            rng = Xoshiro(32))
        for _ in 1:5
            tell!(esi, sphere_b(ask!(esi)))
        end
        @test all(x -> x == round(x), esi.best_x)
    end

    @testset "minimize callback" begin
        n = Ref(0)
        minimize(sphere_b, 3; steps = 50, rng = Xoshiro(41),
            callback = es -> (n[] += 1; n[] >= 4))
        @test n[] == 4
    end

    @testset "radius decay, eltype, promotion, warm duals, validation" begin
        es = PolyStepES(3; step_radius = 1.0, radius_decay = 0.5, rng = Xoshiro(60))
        for _ in 1:3
            tell!(es, sphere_b(ask!(es)))
        end
        @test es.step_radius == 0.125
        @test_throws ArgumentError PolyStepES(3; radius_decay = 0.0)
        @test_throws ArgumentError PolyStepES(3; radius_decay = 1.5)
        @test PolyStepES(2; x0 = Float32[1, 2]) isa PolyStepES{Float32}
        @test_throws ArgumentError PolyStepES(2; x0 = 1.0)
        @test_throws ArgumentError PolyStepES(2; lb = (0, 0), ub = (1, 1))
        @test_throws ArgumentError PolyStepES(2; num_particles = 2, solver = SinkhornSolver(),
            epsilon = Inf)
        e32 = PolyStepES(2; x0 = [1.0, 1.0], step_radius = 0.1, T = Float32, rng = Xoshiro(5))
        X = ask!(e32)
        tell!(e32, 1e8 .+ sphere_b(Float64.(X)))
        @test norm(e32.X .- 1) > 1e-3
        ek = PolyStepES(4; num_particles = 3, solver = SinkhornSolver(), rng = Xoshiro(61))
        tell!(ek, sphere_b(ask!(ek)))
        @test ek.f isa AbstractVector && ek.g isa AbstractVector
        er = PolyStepES(2; x0 = [0.3, 0.7], repair = X -> round.(X), rng = Xoshiro(62))
        @test all(isinteger, ask!(er))
        ev = PolyStepES(2; x0 = [0.3, 0.7], repair = X -> (X[1, :] .= round.(X[1, :])), rng = Xoshiro(63))
        @test all(isinteger, ask!(ev)[1, :])
        @test_throws DimensionMismatch ask!(PolyStepES(2; repair = X -> round.(X)[1:1, :]))
        ek2 = PolyStepES(4; num_particles = 3, solver = SinkhornSolver(), rng = Xoshiro(61))
        fit = sphere_b(ask!(ek2))
        tell!(ek2, fit)
        f1, g1 = copy(ek2.f), copy(ek2.g)
        ask!(ek2)
        tell!(ek2, fit)
        @test ek2.f == PolyStep.solve(SinkhornSolver(), reshape(fit, 8, 3), ek2.epsilon;
            scale_cost = ek2.scale_cost, f0 = f1, g0 = g1).f
        ea = PolyStepES(2; x0 = Any[0.1, 0.2], rng = Xoshiro(64))
        @test ea isa PolyStepES{Float64}
        tell!(ea, Any[sphere_b(ask!(ea))...])
        @test ea.evals == popsize(ea)
        eb = PolyStepES(2; x0 = [1.0, 1.0], rng = Xoshiro(65))
        Xb = ask!(eb)
        tell!(eb, big"1e400" .* BigFloat.(sphere_b(Xb)))
        @test mean(eb) != [1.0, 1.0] && eb.best_f == Inf
        @test_throws DimensionMismatch minimize(X -> [3.0], 2; num_particles = 2, steps = 0)
    end

    @testset "Float32 mode" begin
        es = PolyStepES(4; T = Float32, x0 = ones(Float32, 4), rng = Xoshiro(51))
        X = ask!(es)
        @test eltype(X) === Float32
        tell!(es, sphere_b(X))
        @test eltype(es.X) === Float32 && all(isfinite, es.X)
    end
end
