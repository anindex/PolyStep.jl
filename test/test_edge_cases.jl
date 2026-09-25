using PolyStep: _best_finite, solve, SinkhornSolver, KLSoftmaxSolver,
                 TopKMeanSolver, SoftmaxSolver, OTResult, AbstractOTSolver,
                 normalize_particle_masses!, softmax_cols!, newton_refinement!, _diverged

struct ZeroPlanSolver <: AbstractOTSolver end
function PolyStep.solve(::ZeroPlanSolver, C::AbstractMatrix{T}, eps::Real;
        kwargs...) where {T}
    OTResult(zeros(T, size(C)), 0.0, nothing, nothing, true, 1, nothing)
end

@testset "edge cases and regressions" begin
    @testset "NaN-safe incumbent scan (_best_finite)" begin
        @test _best_finite([3.0, NaN, 1.0, 5.0]) == (1.0, 3)
        @test _best_finite([NaN, NaN]) == (Inf, 0)
        @test _best_finite([-Inf, 2.0, Inf]) == (2.0, 2)   # non-finite never wins
        @test _best_finite(Float32[4, 2, 3]) == (2.0, 2)
    end

    @testset "mixed NaN+finite batch updates the incumbent (core)" begin
        ps = PolyStepConfig(dim = 3)
        st = init_state(ps, randn(Xoshiro(11), 3, 4))
        fn(X) = [isodd(j) ? NaN : sum(abs2, view(X, :, j)) for j in 1:size(X, 2)]
        for _ in 1:3
            PolyStep.step!(fn, ps, st; rng = Xoshiro(12))
        end
        @test isfinite(st.best_f)
        @test all(isfinite, st.best_x)
    end

    @testset "ProgressiveEpsilon: repeat runs identical, config stays inert" begin
        f(X) = vec(sum(abs2, X; dims = 1))
        sched = ProgressiveEpsilon(init = 0.8, target = 0.05)
        ps = PolyStepConfig(dim = 3, solver = SinkhornSolver(max_iterations = 40),
            epsilon = sched)
        function go()
            st = init_state(ps, randn(Xoshiro(7), 3, 4))
            for _ in 1:8
                PolyStep.step!(f, ps, st; rng = Xoshiro(8))
            end
            return st.X
        end
        X1 = go()
        X2 = go()
        @test X1 == X2                       # bit-identical repeat runs
        @test sched.current == 0.8           # shared config never mutated
        @test sched.smoothed == 0.8
        @test_throws ArgumentError init_state(
            PolyStepConfig(dim = 3, solver = SinkhornSolver(),
                epsilon = ProgressiveEpsilon(), ent_epsilon = 0.3),
            zeros(3, 2))
    end

    @testset "unconverged/overrelaxed Sinkhorn plan stays usable" begin
        rng = Xoshiro(42)
        C = 50.0 .* randn(rng, 8, 16)
        res = solve(SinkhornSolver(threshold = 0.0, max_iterations = 5, omega = 1.9),
            C, 1e-3)
        @test all(isfinite, res.plan)
        Wn = similar(res.plan)
        normalize_particle_masses!(Wn, res.plan)
        @test all(isfinite, Wn)
        @test any(>(0), sum(res.plan; dims = 1))
    end

    @testset "KL overflow repair keeps mass on the BEST vertices" begin
        C = [0.0 1e4; 1e4 0.0; 5e3 5e3]
        res = solve(KLSoftmaxSolver(lam = Inf, max_iterations = 50), C, 1e-3)
        @test all(isfinite, res.plan)
        for p in 1:2
            best_v = argmin(view(C, :, p))
            @test argmax(view(res.plan, :, p)) == best_v
        end
        Cx = [0.0 1e300; 1e300 0.0; 5e299 1.0]
        for lam in (0.0, 1.0, Inf)
            r = @test_logs solve(KLSoftmaxSolver(lam = lam, max_iterations = 50), Cx, 1e-10)
            @test isapprox(vec(sum(r.plan; dims = 1)), [0.5, 0.5])
            @test r.plan[1, 1] == 0.5 && all(isfinite, r.g)
        end
    end

    @testset "KL warm-start NaN duals reset instead of persisting" begin
        C = randn(Xoshiro(5), 4, 6)
        res = solve(KLSoftmaxSolver(lam = 2.0, max_iterations = 100), C, 0.3;
            f0 = fill(NaN, 6), g0 = fill(NaN, 4))
        @test all(isfinite, res.plan)
        @test all(isfinite, res.f)
        @test all(isfinite, res.g)
    end

    @testset "validation gates" begin
        @test_throws ArgumentError TopKMeanSolver(k = 0)
        @test_throws ArgumentError KLSoftmaxSolver(threshold = 0.0)   # strict test, no fixed mode
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 2,
            ent_epsilon = ProgressiveEpsilon(), solver = KLSoftmaxSolver(lam = 0.0)), zeros(2, 2))
        @test_throws ArgumentError solve(SinkhornSolver(max_iterations = 5),
            ones(3, 4), 0.1; a = [1.0, 1.0, 1.0, 1.0], b = [0.1, 0.1, 0.1])
    end

    @testset "solve! nested inside user threading (d>8 LAPACK path)" begin
        f(X) = vec(sum(abs2, X; dims = 1))
        ok = fill(false, 2)
        Threads.@threads for i in 1:2
            ps = PolyStepConfig(dim = 10, max_iterations = 3, min_iterations = 1)
            st = init_state(ps, randn(Xoshiro(i), 10, 4))
            PolyStep.solve!(f, ps, st; rng = Xoshiro(100 + i))
            ok[i] = all(isfinite, st.X)
        end
        @test all(ok)
        # columnwise(:threads) objective inside a threaded region
        g = columnwise(x -> sum(abs2, x); parallel = :threads)
        res = fill(false, 2)
        Threads.@threads for i in 1:2
            res[i] = length(g(randn(Xoshiro(i), 3, 5))) == 5
        end
        @test all(res)
    end

    @testset "zero-mass plan column holds position (ES)" begin
        es = PolyStepES(2; solver = ZeroPlanSolver(), x0 = [1.0, 2.0], rng = Xoshiro(1))
        X0 = copy(es.X)
        X = ask!(es)
        tell!(es, ones(size(X, 2)))
        @test es.X == X0        # held, not moved to the origin
    end

    @testset "two-sided P=1 warning covers KLSoftmax lam=Inf" begin
        @test_logs (:warn, r"Two-sided OT") PolyStepES(3;
            solver = KLSoftmaxSolver(lam = Inf), rng = Xoshiro(0))
        @test_logs (:warn, r"Two-sided OT") match_mode=:any init_state(
            PolyStepConfig(dim = 3, solver = SinkhornSolver()), zeros(3, 1))
    end

    @testset "convergence needs a small step, not merely a steady one" begin
        lin(X) = vec(sum(X; dims = 1))          # unbounded linear: steady descent
        ps = PolyStepConfig(dim = 4, epsilon = 0.1, max_iterations = 60, min_iterations = 3)
        st = init_state(ps, zeros(4, 16))
        solve!(lin, ps, st; rng = Xoshiro(1))
        @test st.iteration == 60                # a steady descent never "converges"
        @test st.best_f < 0                     # and it kept descending
    end

    @testset "PolyStep config validation: signs, bounds, jitter" begin
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, epsilon = 0.0), zeros(3, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, epsilon = -0.5), zeros(3, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, ent_epsilon = -0.1), zeros(3, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, step_radius = -1.0), zeros(3, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, probe_radius = -2.0), zeros(3, 2))
        @test_throws DimensionMismatch init_state(
            PolyStepConfig(dim = 3, lb = [-1.0], ub = [1.0]), zeros(3, 4))
        @test_throws DimensionMismatch PolyStepES(3; lb = [-1.0], ub = [1.0, 1.0, 1.0])
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, probe_radius_jitter = 1.0), zeros(3, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, probe_radius_jitter = -0.1), zeros(3, 2))
        # invalid bounds and ProgressiveEpsilon radii rejected
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 2, lb = [1.0, 1.0], ub = [0.0, 2.0]), zeros(2, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 2, lb = NaN, ub = 1.0), zeros(2, 2))
        @test_throws ArgumentError PolyStepES(2; lb = [1.0, 1.0], ub = [0.0, 2.0])
        @test_throws ArgumentError PolyStepES(1; lb = Inf, ub = Inf)
        @test_throws ArgumentError PolyStepES(1; lb = -Inf, ub = -Inf)
        @test PolyStepES(2; lb = [0.0, -Inf], ub = [Inf, Inf]) isa PolyStepES
        sq1(X) = vec(sum(abs2, X .+ 1; dims = 1))      # optimum (-1, -1) outside the box
        psh = PolyStepConfig(dim = 2, lb = [0.0, -Inf], ub = Inf, max_iterations = 30)
        sth = init_state(psh, ones(2, 3))
        solve!(sq1, psh, sth; rng = Xoshiro(3))
        @test all(>=(0), sth.X[1, :]) && sth.best_x[1] >= 0 && isfinite(sth.best_f)
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 2, step_radius = ProgressiveEpsilon()), zeros(2, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 2, probe_radius = ProgressiveEpsilon()), zeros(2, 2))
        st = init_state(PolyStepConfig(dim = 3, lb = fill(-1.0, 3), ub = fill(1.0, 3),
            probe_radius_jitter = 0.5), zeros(3, 4))
        @test st.iteration == 0
    end

    @testset "state-local solver isolates runs from one config" begin
        sq(X) = vec(sum(abs2, X; dims = 1))
        ps = PolyStepConfig(dim = 4, solver = SinkhornSolver(max_iterations = 40),
            max_iterations = 5, min_iterations = 1)
        st1 = init_state(ps, randn(Xoshiro(1), 4, 8))
        st2 = init_state(ps, randn(Xoshiro(2), 4, 8))
        @test st1.solver !== ps.solver
        @test st1.solver !== st2.solver
        ok = fill(false, 2)
        Threads.@threads for i in 1:2
            st = i == 1 ? st1 : st2
            solve!(sq, ps, st; rng = Xoshiro(100 + i))
            ok[i] = all(isfinite, st.X)
        end
        @test all(ok)
    end

    @testset "biased rotation survives zero descent on a flat objective" begin
        flat(X) = fill(1.0, size(X, 2))
        ps = PolyStepConfig(dim = 3, biased_rotation = true)
        st = init_state(ps, randn(Xoshiro(1), 3, 6))
        for _ in 1:5
            step!(flat, ps, st; rng = Xoshiro(2))
        end
        @test all(isfinite, st.X)
        for p in 1:6
            @test maximum(abs, st.R[:, :, p]' * st.R[:, :, p] - I(3)) < 1e-5
        end
    end

    @testset "biased rotation stays per particle when some probes return Inf" begin
        g = columnwise(x -> x[1] > 1.0 ? Inf : sum(abs2, x .- 0.5))
        ps = PolyStepConfig(dim = 3, num_probe = 2, use_quadratic_model = true,
            biased_rotation = true)
        st = init_state(ps, [-1.0 1.0; 0.0 0.0; 0.0 0.0])   # particle 2 straddles the Inf wall
        step!(g, ps, st; rng = Xoshiro(1))
        pd = copy(st.prev_descent)
        @test all(isfinite, pd)
        step!(g, ps, st; rng = Xoshiro(2))
        # particle 1 is still biased toward its FD descent
        @test isapprox(st.R[:, 1, 1], pd[:, 1] ./ norm(pd[:, 1]); atol = 1e-10)
        @test maximum(abs, st.R[:, :, 2]' * st.R[:, :, 2] - I(3)) < 1e-10
    end

    @testset "KLSoftmax(lam=Inf) guards unequal marginal mass" begin
        C = zeros(3, 2)
        @test_throws ArgumentError solve(KLSoftmaxSolver(lam = Inf), C, 0.5;
            a = [0.5, 0.5], b = [0.5, 0.5, 0.5])
        res = solve(KLSoftmaxSolver(lam = 1.0), C, 0.5; a = [0.5, 0.5], b = [0.5, 0.5, 0.5])
        @test all(isfinite, res.plan)
    end

    @testset "empty / zero-dimension inputs are rejected" begin
        @test_throws ArgumentError solve(SoftmaxSolver(), zeros(0, 3), 0.5)
        @test_throws ArgumentError solve(SoftmaxSolver(), zeros(3, 0), 0.5)
        @test_throws ArgumentError solve(TopKMeanSolver(), zeros(0, 2), 0.5)
        @test_throws ArgumentError solve(SinkhornSolver(), zeros(0, 2), 0.5)
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3), zeros(3, 0))
    end

    @testset "integer cost matrices are promoted, not InexactError" begin
        Ci = [1 2; 3 4; 5 0]
        resf = solve(SoftmaxSolver(), Float64.(Ci), 0.5)
        res = solve(SoftmaxSolver(), Ci, 0.5)
        @test eltype(res.plan) <: AbstractFloat
        @test isapprox(res.plan, resf.plan)
        @test all(isfinite, solve(SinkhornSolver(max_iterations = 30), Ci, 0.5).plan)
        @test all(isfinite, solve(TopKMeanSolver(k = 2), Ci, 0.5).plan)
        @test all(isfinite, solve(KLSoftmaxSolver(lam = 2.0), Ci, 0.5).plan)
    end

    @testset "Sinkhorn generic path (non-Matrix input) stays finite" begin
        Cv = view(randn(Xoshiro(3), 6, 8), :, :)   # SubArray -> generic scratch path
        res = solve(SinkhornSolver(max_iterations = 30), Cv, 0.3)
        @test all(isfinite, res.plan)
    end

    @testset "softmax fast path stays finite at extreme cost/eps" begin
        C = fill(1e308, 3, 2)
        W = similar(C)
        softmax_cols!(W, C, 1e-6)
        @test all(isfinite, W)
        @test all(isapprox(1 / 3), W)                          # equal costs -> uniform
        C2 = [0.0 1e300; 1e308 0.0; 1e307 1e307]
        W2 = similar(C2)
        softmax_cols!(W2, C2, 1e-3)
        @test all(isfinite, W2)
        @test all(isapprox(1), sum(W2; dims = 1))
        for p in 1:2
            @test argmax(view(W2, :, p)) == argmin(view(C2, :, p))
        end
        @test all(isfinite, solve(SoftmaxSolver(), fill(1e300, 4, 3), 1e-4).plan)
        Wi = similar(C); softmax_cols!(Wi, C, Inf)                # fast path
        Wig = similar(C); softmax_cols!(Wig, view(C, :, :), Inf)  # generic path
        @test all(isapprox(1 / 3), Wi) && all(isapprox(1 / 3), Wig)
        @test all(isfinite, solve(SoftmaxSolver(), C, Inf).plan)
    end

    @testset "divergence detector fires on a fully non-finite objective" begin
        allinf(X) = fill(Inf, size(X, 2))
        ps = PolyStepConfig(dim = 3, max_iterations = 50, min_iterations = 2)
        st = init_state(ps, randn(Xoshiro(1), 3, 4))
        solve!(allinf, ps, st; rng = Xoshiro(2))
        @test st.last_all_nonfinite
        @test _diverged(st)              # early-stop predicate fires
        @test st.iteration == 1
        @test !isfinite(st.best_f)       # no finite candidate was ever seen
    end

    @testset "PolyStepES deep-copies its solver (no shared workspace)" begin
        shared = SinkhornSolver(max_iterations = 30)
        es1 = PolyStepES(4; num_particles = 2, solver = shared, rng = Xoshiro(1))
        es2 = PolyStepES(4; num_particles = 2, solver = shared, rng = Xoshiro(2))
        @test es1.solver !== shared
        @test es1.solver !== es2.solver
        sq(X) = vec(sum(abs2, X; dims = 1))
        ok = fill(false, 2)
        Threads.@threads for i in 1:2
            es = i == 1 ? es1 : es2
            for _ in 1:5
                tell!(es, sq(ask!(es)))
            end
            ok[i] = all(isfinite, es.X)
        end
        @test all(ok)
    end

    @testset "epsilon schedule constructors validate parameters" begin
        @test_throws ArgumentError LinearEpsilon(init = -1.0)
        @test_throws ArgumentError LinearEpsilon(target = 0.0)
        @test_throws ArgumentError LinearEpsilon(decay = -0.1)
        @test_throws ArgumentError CosineEpsilon(init = -1.0)
        @test_throws ArgumentError CosineEpsilon(restart_mult = 0.5)
        @test_throws ArgumentError ProgressiveEpsilon(init = -1.0)
        @test_throws ArgumentError ProgressiveEpsilon(target = 0.0)
        @test_throws ArgumentError ProgressiveEpsilon(target = 10.0, max_epsilon = 5.0)
        @test LinearEpsilon(init = 1.0, target = 0.01, decay = 0.01) isa LinearEpsilon
    end

    @testset "CosineEpsilon horizon (ceil) and SGDR period growth" begin
        @test epsilon_at(CosineEpsilon(), 99) > 1e-3
        @test epsilon_at(CosineEpsilon(), 100) == 1e-3
        s = CosineEpsilon(total_steps = 4, restart_mult = 1.2)
        @test epsilon_at(s, 4) == 1.0 && epsilon_at(s, 9) == 1.0
        @test epsilon_at(s, 1000) > 0.1
        # zero decay: the inferred horizon must not overflow Int
        @test isapprox(epsilon_at(CosineEpsilon(init = 1e7, target = 1.0, decay = 0.0), 5), 1e7)
    end

    @testset "simplex / cube polytope step! path descends" begin
        sq(X) = vec(sum(abs2, X; dims = 1))
        for poly in (:simplex, :cube)
            X0 = randn(Xoshiro(1), 3, 6)
            ps = PolyStepConfig(dim = 3, polytope = poly, epsilon = 0.2,
                step_radius = 1.0, probe_radius = 1.5)
            st = init_state(ps, copy(X0))
            for _ in 1:30
                step!(sq, ps, st; rng = Xoshiro(2))
            end
            @test st.dirs !== nothing                     # used the general (matmul) path
            @test all(isfinite, st.X)
            @test mean(sq(st.X)) < mean(sq(X0))
        end
    end

    @testset "quadratic model with bounds masks clamped probes" begin
        sq(X) = vec(sum(abs2, X; dims = 1))
        ps = PolyStepConfig(dim = 3, num_probe = 2, use_quadratic_model = true,
            newton_refinement = true, epsilon = 0.3, step_radius = 1.0,
            probe_radius = 1.0, lb = -0.5, ub = 0.5)
        st = init_state(ps, fill(0.5, 3, 4))     # on the boundary -> probes clamp
        @test st.clampflag !== nothing
        for _ in 1:10
            step!(sq, ps, st; rng = Xoshiro(3))
        end
        @test all(isfinite, st.X)
        @test all(x -> -0.5 <= x <= 0.5, st.X)   # stays feasible
    end

    @testset "newton_refinement! descent gate and mask" begin
        d, P, K = 3, 2, 3
        rng = Xoshiro(1)
        R = zeros(d, d, P)
        haar_rotations!(R, zeros(d, d, P), rng)
        D = [1.0, 2.0, 0.5]
        c = randn(rng, d)
        fq(x) = 0.5 * sum(D .* (x .- c) .^ 2)
        Xcur = randn(rng, d, P)
        pr = 0.05
        scales = PolyStep.probe_scales(Float64, K)
        losses3 = zeros(K, 2d, P)
        for p in 1:P, i in 1:d, (sgn, off) in ((1.0, 0), (-1.0, d)), k in 1:K
            losses3[k, i + off, p] = fq(Xcur[:, p] .+ (pr * scales[k] * sgn) .* R[:, i, p])
        end
        Xbary = Xcur .+ 0.1 .* (c .- Xcur)       # a small move toward the minimum
        G = zeros(d, P); H = zeros(d, P)
        N = zeros(d, P); refrot = zeros(d, P)
        Xout = similar(Xbary)
        newton_refinement!(Xout, Xbary, G, H, N, refrot, losses3, scales, pr, R, Xcur;
            alpha = 0.5, max_step_norm = 10.0)
        @test all(isfinite, Xout)
        # masked particle keeps X_bary exactly
        Xout2 = similar(Xbary)
        newton_refinement!(Xout2, Xbary, G, H, N, refrot, losses3, scales, pr, R, Xcur;
            alpha = 0.5, max_step_norm = 10.0, mask = [true, false])
        @test Xout2[:, 1] == Xbary[:, 1]
    end

    @testset "Sinkhorn never leaks NaN duals when max_iterations < check_every" begin
        res = solve(SinkhornSolver(max_iterations = 5, check_every = 10),
            [-1e308 0.0; 0.0 0.0], 1e-6)
        @test all(isfinite, res.f)
        @test all(isfinite, res.g)
        @test all(isfinite, res.plan)
    end

    @testset "iterative solvers reject non-finite epsilon" begin
        C = [1.0 2.0; 3.0 4.0]
        @test_throws ArgumentError solve(SinkhornSolver(max_iterations = 5), C, Inf)
        @test_throws ArgumentError solve(KLSoftmaxSolver(lam = 0.0), C, Inf)
        @test_throws ArgumentError solve(KLSoftmaxSolver(lam = 2.0), C, Inf)
        @test all(isapprox(0.25), solve(SoftmaxSolver(), C, Inf).plan)
    end

    @testset "Sinkhorn/KL are numerically shift-invariant at extreme cost magnitudes" begin
        C = fill(1e308, 2, 2)
        sh = solve(SinkhornSolver(max_iterations = 200), C, 1e-6)
        @test all(isfinite, sh.plan)
        @test all(isapprox(0.25), sh.plan)                         # a (x) b, uniform
        kli = solve(KLSoftmaxSolver(lam = Inf, max_iterations = 200), C, 1e-6)
        @test all(isapprox(0.25), kli.plan)
        kl0 = solve(KLSoftmaxSolver(lam = 0.0), C, 1e-6)
        sm = solve(SoftmaxSolver(), C, 1e-6)
        @test isapprox(kl0.plan, sm.plan; rtol = 1e-10)
        C2 = [0.0 1e308; 1e308 0.0]
        sh2 = solve(SinkhornSolver(max_iterations = 500), C2, 1e-3)
        @test all(isfinite, sh2.plan)
        @test sh2.plan[1, 1] > sh2.plan[2, 1]               # cheaper vertex favored
        @test sh2.plan[2, 2] > sh2.plan[1, 2]
    end

    @testset "ProgressiveEpsilon decreasing setpoint (total_steps > 0)" begin
        # total_steps = 0: iteration ignored (reactive governor)
        ep0 = ProgressiveEpsilon(init = 1.0, target = 0.01)
        @test epsilon_at(ep0, 5) == epsilon_at(ep0, 500) == 1.0
        ep = ProgressiveEpsilon(init = 1.0, target = 0.01, total_steps = 10)
        @test isapprox(epsilon_at(ep, 0), 1.0)
        @test isapprox(epsilon_at(ep, 10), 0.01; atol = 1e-9)
        @test epsilon_at(ep, 3) > epsilon_at(ep, 7)          # sharpens over the run
        base5 = epsilon_at(ep, 5)
        update!(ep; n_iters = 100, max_iterations = 100, converged = false)
        @test epsilon_at(ep, 5) > base5
        @test_throws ArgumentError ProgressiveEpsilon(total_steps = -1)
    end

    @testset "trust-region mean proxy stays bounded and descends anisotropic quadratic" begin
        D = [1.0, 10.0, 0.5, 4.0]
        aniso(X) = vec(sum(D .* X .^ 2; dims = 1))
        ps = PolyStepConfig(dim = 4, epsilon = 0.3, num_probe = 3, use_quadratic_model = true,
            trust_region = true, step_radius = 1.0, probe_radius = 2.0, scale_cost = :mean)
        X0 = 2.0 .* randn(Xoshiro(7), 4, 8)
        st = init_state(ps, copy(X0))
        for _ in 1:12
            step!(aniso, ps, st; rng = Xoshiro(8))
        end
        @test all(m -> 0.1 <= m <= 3.0, st.trust_multipliers)
        @test mean(aniso(st.X)) < mean(aniso(X0))
    end

    @testset "all-non-finite batch holds the iterate (bounds active)" begin
        es = PolyStepES(1; x0 = [0.4], lb = -0.5, ub = 0.5, step_radius = 2.0)
        tell!(es, fill(NaN, length(ask!(es))))
        @test isapprox(es.X[1], 0.4) && es.best_f == Inf && es.evals == popsize(es)
        psc = PolyStepConfig(dim = 2, lb = [-0.5, -0.5], ub = [0.5, 0.5], step_radius = 2.0)
        st = init_state(psc, fill(0.4, 2, 3))
        step!(X -> fill(NaN, size(X, 2)), psc, st; rng = Xoshiro(0))
        @test isapprox(st.X, fill(0.4, 2, 3))
    end
end
