using PolyStep
using PolyStep: _resolve_radii, _batched_mul!, _batched_matvec!, normalize_particle_masses!,
                 fd_gradient!, fd_hessian_diag!, newton_step!, predicted_improvement,
                 predicted_improvement_mean, update_trust_region,
                 momentum_coefficient, update_adaptive_radius, solve

sphere(X) = vec(sum(abs2, X; dims = 1))

@testset "core" begin
    @testset "dynamics units" begin
        @test isapprox(momentum_coefficient(0, 11), 0.5)
        @test isapprox(momentum_coefficient(10, 11), 0.95)
        @test isapprox(momentum_coefficient(999, 11), 0.95)
        # first step (prev=Inf): no adaptation
        @test update_adaptive_radius(1.0, Inf, 0, 1.0) == (1.0, 0, 1.0)
        # NaN loss: state unchanged
        rm_nan, sc_nan, pl_nan = update_adaptive_radius(NaN, 1.0, 2, 1.3)
        @test rm_nan == 1.3 && sc_nan == 2 && isnan(pl_nan)
        # stagnation accumulates, boosts at patience, then resets
        rm, sc, _ = update_adaptive_radius(1.0, 1.0, 9, 1.0; stagnation_patience = 10)
        @test isapprox(rm, 1.5) && sc == 0
        # improvement decays
        rm, sc, _ = update_adaptive_radius(0.5, 1.0, 0, 1.0)
        @test isapprox(rm, 0.9) && sc == 0
        @test isapprox(update_trust_region(-1.0, -0.9, 1.0), 1.5)  # accurate improvement: expand
        @test isapprox(update_trust_region(-1.0, 1.0, 1.0), 0.25)  # opposite sign: large shrink
        @test isapprox(update_trust_region(1.0, 1.0, 1.0), 0.5)  # predicted worsening: shrink
        @test isapprox(update_trust_region(1.0, -1.0, 1.0), 0.5)
        @test isapprox(update_trust_region(NaN, -1.0, 1.0), 0.5)
        @test isapprox(update_trust_region(0.0, -1.0, 1.0), 1.0)  # tiny prediction: unchanged
    end

    @testset "FD extractors exact on quadratics" begin
        rng = Xoshiro(31)
        d, P, K = 4, 3, 3
        R = zeros(d, d, P)
        haar_rotations!(R, zeros(d, d, P), rng)
        D = [1.0, 2.5, 0.5, 4.0]                       # curvature diag (original frame)
        c = randn(rng, d)
        fq(x) = 0.5 * sum(D .* (x .- c) .^ 2)
        X = randn(rng, d, P)
        scales = PolyStep.probe_scales(Float64, K)
        G = zeros(d, P);
        H = zeros(d, P)
        for pr in (0.05, 2e-3)
            losses3 = zeros(K, 2d, P)
            for p in 1:P, i in 1:d, (sgn, off) in ((1.0, 0), (-1.0, d)), k in 1:K
                losses3[k, i + off, p] = fq(X[:, p] .+ (pr * scales[k] * sgn) .* R[:, i, p])
            end
            fd_gradient!(G, losses3, scales, pr)
            fd_hessian_diag!(H, losses3, scales, pr)
            for p in 1:P
                grad_true = D .* (X[:, p] .- c)
                @test isapprox(G[:, p], R[:, :, p]' * grad_true; atol = 1e-8)  # rotated frame
                Hrot = R[:, :, p]' * Diagonal(D) * R[:, :, p]
                @test isapprox(H[:, p], diag(Hrot); atol = 1e-8)
            end
        end
        # newton step norm clipped
        N = zeros(d, P)
        newton_step!(N, G, H; max_step_norm = 0.01)
        @test all(p -> norm(N[:, p]) <= 0.01 + 1e-12, 1:P)
        @test all(<=(0), predicted_improvement(G, H, N))            # descent step predicts improvement
        N2 = zeros(2, 1)
        newton_step!(N2, reshape([1e3, 1.0], 2, 1), reshape([1e-8, 1.0], 2, 1); max_step_norm = 1.0)
        @test isapprox(vec(N2), [-1, -1] ./ sqrt(2))
        @test predicted_improvement(zeros(1, 1), fill(-5.0, 1, 1), ones(1, 1))[1] == 0.5e-4
        X1 = X .+ 0.1 .* randn(rng, d, P)
        S = reduce(hcat, [R[:, :, p]' * (X1[:, p] - X[:, p]) for p in 1:P])
        @test isapprox(predicted_improvement_mean(G, H, R, X1, X),
            mean(predicted_improvement(G, H, S)))
    end

    @testset "probe point layout (k fastest, then v, then p)" begin
        d, K = 3, 2
        V = 2d
        rng = Xoshiro(77)
        scales = PolyStep.probe_scales(Float64, K)
        pr = 0.6
        for P in (4, PolyStep._KERNEL_BATCH_MIN)
            X = randn(rng, d, P)
            R = zeros(d, d, P)
            haar_rotations!(R, zeros(d, d, P), rng)
            Xp = zeros(d, K * V * P)
            PolyStep._probe_points_orthoplex!(Xp, X, R, pr, scales)
            Xref = similar(Xp)
            for p in 1:P, i in 1:d, k in 1:K
                cplus = (p - 1) * V * K + (i - 1) * K + k
                cminus = (p - 1) * V * K + (d + i - 1) * K + k
                Xref[:, cplus] = X[:, p] .+ pr * scales[k] .* R[:, i, p]
                Xref[:, cminus] = X[:, p] .- pr * scales[k] .* R[:, i, p]
            end
            @test isapprox(Xp, Xref; atol = 1e-12)
            Xg = zeros(d, K * V * P)
            dirs = zeros(d, V, P)
            _batched_mul!(dirs, R, orthoplex_vertices(d))
            PolyStep._probe_points_general!(Xg, X, dirs, pr, scales)
            @test isapprox(Xg, Xref; atol = 1e-12)
        end
    end

    @testset "step! decreases quadratic" begin
        ps = PolyStepConfig(dim = 5, epsilon = 0.05, step_radius = 4.0, probe_radius = 4.0,
            scale_cost = :mean)
        st = init_state(ps, 2.0 .* randn(Xoshiro(1), 5, 16))
        rng = Xoshiro(2)
        first_cost = step!(sphere, ps, st; rng)
        last_cost = first_cost
        for _ in 1:60
            last_cost = step!(sphere, ps, st; rng)
        end
        @test last_cost < 0.2 * first_cost
        @test st.iteration == 61
        @test st.evals == 61 * 1 * 10 * 16
        @test isfinite(st.best_f) && st.best_f <= first_cost
    end

    @testset "chunked eval identical" begin
        mk() = init_state(PolyStepConfig(dim = 3, epsilon = 0.3), randn(Xoshiro(3), 3, 8))
        ps0 = PolyStepConfig(dim = 3, epsilon = 0.3)
        psc = PolyStepConfig(dim = 3, epsilon = 0.3, eval_chunk = 7)
        st0, stc = mk(), mk()
        for (ps, st, seed) in ((ps0, st0, 4), (psc, stc, 4))
            rng = Xoshiro(seed)
            for _ in 1:3
                step!(sphere, ps, st; rng)
            end
        end
        @test st0.X == stc.X
    end

    @testset "objective must return one cost per column" begin
        ps = PolyStepConfig(dim = 3)
        @test_throws DimensionMismatch step!(X -> sum(X), ps, init_state(ps, zeros(3, 2)))  # forgot columnwise
        psc = PolyStepConfig(dim = 3, eval_chunk = 5)
        @test_throws DimensionMismatch step!(X -> [1.0], psc, init_state(psc, zeros(3, 2)))
    end

    @testset "jitter=0 consumes no rng" begin
        ps0 = PolyStepConfig(dim = 3, probe_radius_jitter = 0.0)
        psj = PolyStepConfig(dim = 3, probe_radius_jitter = 0.1)
        st = init_state(ps0, zeros(3, 2))
        rng = Xoshiro(5)
        _resolve_radii(ps0, st, rng)
        @test rand(rng) == rand(Xoshiro(5))
        rngj = Xoshiro(5)
        _resolve_radii(psj, st, rngj)
        @test rand(rngj) != rand(Xoshiro(5))
    end

    @testset "scale_cost does not leak into adaptive state" begin
        ps = PolyStepConfig(dim = 3, epsilon = 0.5, scale_cost = 1e6, use_adaptive_radius = true)
        st = init_state(ps, fill(2.0, 3, 4))
        cost = step!(sphere, ps, st; rng = Xoshiro(6))
        @test st.prev_loss == mean(minimum, eachcol(st.Craw))
        @test cost > 1.0
    end

    @testset "trust region + quadratic model" begin
        ps = PolyStepConfig(dim = 4, epsilon = 0.3, num_probe = 3, use_quadratic_model = true,
            trust_region = true, biased_rotation = true)
        st = init_state(ps, randn(Xoshiro(7), 4, 8))
        rng = Xoshiro(8)
        for _ in 1:6
            step!(sphere, ps, st; rng)
        end
        @test !isempty(st.trust_multipliers)
        @test all(isfinite, st.trust_multipliers)
        # the ratio update moved the radius off its initial 1.0
        @test any(!=(1.0), st.trust_multipliers)
        @test st.prev_descent !== nothing && all(isfinite, st.prev_descent)
    end

    @testset "momentum + Newton + trust region score the realized move" begin
        ps = PolyStepConfig(dim = 4, epsilon = 0.3, num_probe = 3, use_quadratic_model = true,
            newton_refinement = true, newton_alpha = 0.5, trust_region = true,
            use_momentum = true, velocity_lr = 0.8)
        st = init_state(ps, randn(Xoshiro(25), 4, 6))
        rng = Xoshiro(26)
        for _ in 1:8
            step!(sphere, ps, st; rng)
            @test isapprox(st.X - st.Xprev, ps.velocity_lr .* st.velocity; atol = 1e-12)
            # the trust-region prediction is for the move actually taken
            @test st.prev_predicted ==
                  predicted_improvement_mean(st.G, st.H, st.R, st.X, st.Xprev)
        end
    end

    @testset "fixed-iteration Sinkhorn gives ProgressiveEpsilon no feedback" begin
        ps = PolyStepConfig(dim = 3, ent_epsilon = ProgressiveEpsilon(init = 0.1, target = 0.01),
            solver = SinkhornSolver(threshold = 0.0, max_iterations = 20))
        st = init_state(ps, randn(Xoshiro(27), 3, 4))
        for _ in 1:5
            step!(sphere, ps, st; rng = Xoshiro(28))
        end
        @test st.prog_ent.smoothed == 0.1   # not inflated
    end

    @testset "trust-region proxy is the center value under a shrinking probe radius" begin
        A = [3.0 1.0 0.0; 1.0 2.0 0.5; 0.0 0.5 1.0]
        quad(X) = vec(sum(X .* (A * X); dims = 1)) ./ 2
        ps = PolyStepConfig(dim = 3, epsilon = LinearEpsilon(init = 0.5, target = 0.01, decay = 0.1),
            num_probe = 3, use_quadratic_model = true, trust_region = true)
        st = init_state(ps, randn(Xoshiro(41), 3, 5))
        rng = Xoshiro(42)
        for _ in 1:4
            Xc = copy(st.X)
            step!(quad, ps, st; rng)
            @test isapprox(st.prev_pre_step_loss, mean(quad(Xc)); rtol = 1e-10)
        end
    end

    @testset "non-finite probes leave a finite quadratic model" begin
        nanfar(X) = [x[1] > 1.2 ? NaN : sum(abs2, x) for x in eachcol(X)]
        ps = PolyStepConfig(dim = 3, epsilon = 0.5, num_probe = 2, use_quadratic_model = true,
            newton_refinement = true, trust_region = true)
        st = init_state(ps, fill(1.0, 3, 4))
        step!(nanfar, ps, st; rng = Xoshiro(43))
        @test all(isfinite, st.G) && all(isfinite, st.H)
        @test isfinite(st.prev_predicted)
    end

    @testset "newton refinement invalidates duals" begin
        ps = PolyStepConfig(dim = 3, epsilon = 0.5, num_probe = 2, use_quadratic_model = true,
            newton_refinement = true, solver = SinkhornSolver(max_iterations = 200))
        st = init_state(ps, randn(Xoshiro(9), 3, 6))
        step!(sphere, ps, st; rng = Xoshiro(10))
        @test st.f === nothing && st.g === nothing
        # without refinement, Sinkhorn duals persist for warm start
        ps2 = PolyStepConfig(dim = 3, epsilon = 0.5, solver = SinkhornSolver(max_iterations = 200))
        st2 = init_state(ps2, randn(Xoshiro(9), 3, 6))
        step!(sphere, ps2, st2; rng = Xoshiro(10))
        @test st2.f !== nothing
    end

    @testset "NaN revert" begin
        ps = PolyStepConfig(dim = 3, epsilon = 0.5, step_radius = Inf, biased_rotation = true)
        st = init_state(ps, ones(3, 4))
        step!(sphere, ps, st; rng = Xoshiro(11))
        @test st.X == ones(3, 4)                 # reverted
        @test st.f === nothing && st.g === nothing
        @test st.prev_descent === nothing
    end

    @testset "momentum" begin
        mk(; kw...) = PolyStepConfig(; dim = 4, epsilon = 0.4, kw...)
        X0 = randn(Xoshiro(12), 4, 8)
        stm = init_state(mk(use_momentum = true), copy(X0))
        stp = init_state(mk(), copy(X0))
        for (ps, st) in ((mk(use_momentum = true), stm), (mk(), stp))
            rng = Xoshiro(13)
            for _ in 1:5
                step!(sphere, ps, st; rng)
            end
        end
        @test stm.velocity !== nothing && any(!iszero, stm.velocity)
        @test stm.X != stp.X
    end

    @testset "adaptive radius boosts on flat objective" begin
        flat(X) = fill(1.0, size(X, 2))
        ps = PolyStepConfig(dim = 3, use_adaptive_radius = true, stagnation_patience = 3)
        st = init_state(ps, randn(Xoshiro(14), 3, 4))
        rng = Xoshiro(15)
        for _ in 1:5
            step!(flat, ps, st; rng)
        end
        @test isapprox(st.radius_multiplier, 1.5)  # one boost fired, counter reset
    end

    @testset "bounds + repair" begin
        lo, hi = -0.25, 0.25
        ps = PolyStepConfig(dim = 3, epsilon = 0.5, step_radius = 2.0, probe_radius = 2.0,
            lb = lo, ub = hi)
        st = init_state(ps, fill(0.25, 3, 4))
        rng = Xoshiro(16)
        for _ in 1:5
            step!(sphere, ps, st; rng)
        end
        @test all(x -> lo <= x <= hi, st.X)
        @test all(x -> lo <= x <= hi, st.best_x)
        snap(X) = (X .= round.(X .* 2) ./ 2; X)
        ps_r = PolyStepConfig(dim = 3, epsilon = 0.5, repair = snap)
        st_r = init_state(ps_r, randn(Xoshiro(17), 3, 4))
        step!(sphere, ps_r, st_r; rng = Xoshiro(18))
        @test all(x -> isapprox(x, round(x * 2) / 2; atol = 1e-12), st_r.best_x)
        st_s = init_state(ps_r, randn(Xoshiro(17), 3, 4))
        solve!(sphere, ps_r, st_s; rng = Xoshiro(18))
        @test all(x -> isapprox(x, round(x * 2) / 2; atol = 1e-12), st_s.best_x)
        @test isapprox(st_s.best_f, sum(abs2, st_s.best_x))
    end

    @testset "projection identity on unconverged plans" begin
        rng = Xoshiro(19)
        d, P = 3, 5
        V = 2d
        C = randn(rng, V, P)
        R = zeros(d, d, P)
        haar_rotations!(R, zeros(d, d, P), rng)
        X = randn(rng, d, P)
        sr = 0.7
        res = solve(SinkhornSolver(threshold = 0.0, max_iterations = 2), C, 0.3)  # unconverged
        Wn = similar(res.plan)
        normalize_particle_masses!(Wn, res.plan)
        Vt = orthoplex_vertices(d)
        dirs = zeros(d, V, P)
        _batched_mul!(dirs, R, Vt)
        X_exp = zeros(d, P)
        for p in 1:P, v in 1:V

            X_exp[:, p] .+= Wn[v, p] .* (X[:, p] .+ sr .* dirs[:, v, p])
        end
        cent = Vt * Wn
        rotc = zeros(d, P)
        _batched_matvec!(rotc, R, cent)
        X_fused = X .+ sr .* rotc
        @test isapprox(X_fused, X_exp; rtol = 1e-10)
        shift = randn(rng, d)
        X_exp2 = zeros(d, P)
        for p in 1:P, v in 1:V

            X_exp2[:, p] .+= Wn[v, p] .* (X[:, p] .+ shift .+ sr .* dirs[:, v, p])
        end
        @test isapprox(X_exp2, X_exp .+ shift; rtol = 1e-10)
    end

    @testset "solve! loop + callback + counters" begin
        ps = PolyStepConfig(dim = 3, epsilon = 0.4, max_iterations = 30, min_iterations = 3)
        st = init_state(ps, randn(Xoshiro(20), 3, 8))
        solve!(sphere, ps, st; rng = Xoshiro(21))
        @test 1 <= st.iteration <= 30
        @test length(st.costs) == st.iteration == length(st.disp_sqnorms)
        @test st.evals == st.iteration * 6 * 8 + 8
        @test st.best_f <= minimum(sphere(st.X))
        @test st.cent == orthoplex_vertices(3) * st.Wn
        stc = init_state(ps, randn(Xoshiro(20), 3, 8))
        solve!(sphere, ps, stc; rng = Xoshiro(21), callback = s -> s.iteration >= 2)
        @test stc.iteration == 2
        # convergence is only checked after min_iterations steps
        flat(X) = fill(1.0, size(X, 2))
        psf = PolyStepConfig(dim = 3, min_iterations = 5)
        stf = init_state(psf, randn(Xoshiro(20), 3, 4))
        solve!(flat, psf, stf; rng = Xoshiro(21))
        @test stf.iteration == 5
    end

    @testset "columnwise adapter" begin
        f(x) = sum(abs2, x)
        X = randn(Xoshiro(22), 4, 9)
        @test isapprox(columnwise(f)(X), sphere(X))
        @test isapprox(columnwise(f; parallel = :threads)(X), sphere(X))
        @test_throws ArgumentError columnwise(f; parallel = :processes)
    end

    @testset "config validation" begin
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 7, polytope = :cube), zeros(7, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, newton_refinement = true), zeros(3, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, use_quadratic_model = true), zeros(3, 2))  # K=1
        @test_throws ArgumentError init_state(
            PolyStepConfig(dim = 3, use_quadratic_model = true,
                num_probe = 2, polytope = :simplex),
            zeros(3, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, ent_epsilon = ProgressiveEpsilon()), zeros(3, 2))
        @test_throws DimensionMismatch init_state(PolyStepConfig(dim = 4), zeros(3, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 3, lb = -1.0), zeros(3, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 2), [0.0 NaN; 0.0 0.0])
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 2, scale_cost = :meen), zeros(2, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 2, scale_cost = -1.0), zeros(2, 2))
        @test_throws ArgumentError init_state(PolyStepConfig(dim = 2, epsilon = Inf), zeros(2, 2))
        @test init_state(PolyStepConfig(dim = 2, epsilon = Inf, step_radius = LinearEpsilon(),
            probe_radius = LinearEpsilon()), zeros(2, 2)).iteration == 0
        # ProgressiveEpsilon works with Sinkhorn
        ps = PolyStepConfig(dim = 3, ent_epsilon = ProgressiveEpsilon(),
            solver = SinkhornSolver(max_iterations = 100))
        st = init_state(ps, randn(Xoshiro(23), 3, 4))
        step!(sphere, ps, st; rng = Xoshiro(24))
        @test st.iteration == 1
    end
end
