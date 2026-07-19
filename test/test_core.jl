using PolyStep
using PolyStep: _resolve_radii, _batched_mul!, _batched_matvec!, normalize_particle_masses!,
                 fd_gradient!, fd_hessian_diag!, newton_step!, predicted_improvement,
                 update_trust_region,
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
        @test isapprox(update_trust_region(1.0, 1.0, 1.0), 1.0)  # predicted worsening: no expand
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
        pr = 0.05
        scales = PolyStep.probe_scales(Float64, K)
        losses3 = zeros(K, 2d, P)
        for p in 1:P, i in 1:d, (sgn, off) in ((1.0, 0), (-1.0, d)), k in 1:K
            losses3[k, i + off, p] = fq(X[:, p] .+ (pr * scales[k] * sgn) .* R[:, i, p])
        end
        G = zeros(d, P);
        H = zeros(d, P)
        fd_gradient!(G, losses3, scales, pr)
        fd_hessian_diag!(H, losses3, scales, pr)
        for p in 1:P
            grad_true = D .* (X[:, p] .- c)
            @test isapprox(G[:, p], R[:, :, p]' * grad_true; atol = 1e-8)  # rotated frame
            Hrot = R[:, :, p]' * Diagonal(D) * R[:, :, p]
            @test isapprox(H[:, p], diag(Hrot); atol = 1e-8)
        end
        # newton step norm clipped
        N = zeros(d, P)
        newton_step!(N, G, H; max_step_norm = 0.01)
        @test all(p -> norm(N[:, p]) <= 0.01 + 1e-12, 1:P)
        @test all(<=(0), predicted_improvement(G, H, N))            # descent step predicts improvement
    end

    @testset "probe point layout (k fastest, then v, then p)" begin
        d, P, K = 3, 4, 2
        V = 2d
        rng = Xoshiro(77)
        X = randn(rng, d, P)
        R = zeros(d, d, P)
        haar_rotations!(R, zeros(d, d, P), rng)
        scales = PolyStep.probe_scales(Float64, K)
        pr = 0.6
        Xp = zeros(d, K * V * P)
        PolyStep._probe_points_orthoplex!(Xp, X, R, pr, scales)
        # column c = (p-1)*V*K + (v-1)*K + k; v=i is +e_i, v=d+i is -e_i
        for p in 1:P, i in 1:d, k in 1:K
            cplus = (p - 1) * V * K + (i - 1) * K + k
            cminus = (p - 1) * V * K + (d + i - 1) * K + k
            @test isapprox(Xp[:, cplus], X[:, p] .+ pr * scales[k] .* R[:, i, p]; atol = 1e-12)
            @test isapprox(Xp[:, cminus], X[:, p] .- pr * scales[k] .* R[:, i, p]; atol = 1e-12)
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
        # huge divisor: solver input is scaled but prev_loss must stay raw-scale
        ps = PolyStepConfig(dim = 3, epsilon = 0.5, scale_cost = 1e6, use_adaptive_radius = true)
        st = init_state(ps, fill(2.0, 3, 4))
        cost = step!(sphere, ps, st; rng = Xoshiro(6))
        @test st.prev_loss == cost
        @test cost > 1.0            # raw sphere costs near ||x||^2 = 12, not 1e-5
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
        # the ratio update actually moved the radius off its initial 1.0 (the
        # [0.1, 3.0] clamp range is trivially satisfied, so it is not a real check)
        @test any(!=(1.0), st.trust_multipliers)
        @test st.prev_descent !== nothing && all(isfinite, st.prev_descent)
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
        # repair: snap evaluated candidates to a 0.5 grid; incumbent obeys it
        snap(X) = (X .= round.(X .* 2) ./ 2; X)
        ps_r = PolyStepConfig(dim = 3, epsilon = 0.5, repair = snap)
        st_r = init_state(ps_r, randn(Xoshiro(17), 3, 4))
        step!(sphere, ps_r, st_r; rng = Xoshiro(18))
        @test all(x -> isapprox(x, round(x * 2) / 2; atol = 1e-12), st_r.best_x)
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
        # translation invariance: shifting X shifts the barycenter identically
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
        stc = init_state(ps, randn(Xoshiro(20), 3, 8))
        solve!(sphere, ps, stc; rng = Xoshiro(21), callback = s -> s.iteration >= 2)
        @test stc.iteration == 2
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
        # ProgressiveEpsilon works with Sinkhorn
        ps = PolyStepConfig(dim = 3, ent_epsilon = ProgressiveEpsilon(),
            solver = SinkhornSolver(max_iterations = 100))
        st = init_state(ps, randn(Xoshiro(23), 3, 4))
        step!(sphere, ps, st; rng = Xoshiro(24))
        @test st.iteration == 1
    end
end
