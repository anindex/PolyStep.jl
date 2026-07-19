using PolyStep: solve, OTResult, SoftmaxSolver, TemperedSoftmaxSolver, KLSoftmaxSolver,
                 SinkhornSolver, MinCostGreedySolver, TopKMeanSolver,
                 LinearEpsilon, CosineEpsilon, ProgressiveEpsilon, epsilon_at, update!, is_scheduled

@testset "schedules" begin
    lin = LinearEpsilon(target = 0.1, init = 1.0, decay = 0.05)
    @test epsilon_at(lin, nothing) == 1.0
    @test epsilon_at(lin, 0) == 1.0
    @test isapprox(epsilon_at(lin, 4), 0.8)
    @test epsilon_at(lin, 1000) == 0.1
    @test epsilon_at(0.25, 7) == 0.25
    @test !is_scheduled(0.25) && is_scheduled(lin)

    cosd = CosineEpsilon(target = 0.1, init = 1.0, total_steps = 10)
    @test epsilon_at(cosd, nothing) == 1.0
    @test isapprox(epsilon_at(cosd, 0), 1.0)
    @test isapprox(epsilon_at(cosd, 10), 0.1; atol = 1e-12)
    @test isapprox(epsilon_at(cosd, 5), 0.55)
    @test isapprox(epsilon_at(cosd, 100), 0.1; atol = 1e-12)  # clamped past T
    # decay-inferred horizon: T = (1 - 0.1)/0.09 = 10
    cos2 = CosineEpsilon(target = 0.1, init = 1.0, decay = 0.09)
    @test isapprox(epsilon_at(cos2, 5), 0.55)
    # warm restarts: resets toward init at period boundary
    cosr = CosineEpsilon(target = 0.1, init = 1.0, total_steps = 10, restart_mult = 2.0)
    @test isapprox(epsilon_at(cosr, 10), 1.0)  # t=10 -> new period, t_local=0
    @test 0.1 <= epsilon_at(cosr, 25) <= 1.0

    prog = ProgressiveEpsilon(init = 1.0, target = 0.01, ema_alpha = 0.0)
    @test epsilon_at(prog, nothing) == 1.0
    @test epsilon_at(prog, 999) == 1.0            # iteration ignored
    update!(prog; n_iters = 100, max_iterations = 100, converged = false)   # struggling
    @test isapprox(epsilon_at(prog, 0), 1.2)
    update!(prog; n_iters = 1, max_iterations = 100, converged = true)      # fast
    @test isapprox(epsilon_at(prog, 0), 1.2 * 0.95)
end

@testset "softmax solver" begin
    rng = Xoshiro(10)
    V, P = 6, 9
    C = randn(rng, V, P)
    s = SoftmaxSolver()
    res = solve(s, C, 0.5)
    @test res.plan isa Matrix{Float64}
    @test isapprox(vec(sum(res.plan; dims = 1)), fill(1 / P, P); rtol = 1e-12)  # column sums = a
    @test res.converged && res.iters == 1 && res.f === nothing
    # custom marginal a
    a = rand(rng, P);
    a ./= sum(a)
    res_a = solve(s, C, 0.5; a)
    @test isapprox(vec(sum(res_a.plan; dims = 1)), a; rtol = 1e-12)
    # eps -> 0: one-hot on argmin; eps -> Inf-ish: uniform
    res_sharp = solve(s, C, 1e-6)
    for p in 1:P
        @test isapprox(res_sharp.plan[argmin(C[:, p]), p], 1 / P; rtol = 1e-6)
    end
    res_flat = solve(s, C, 1e9)
    @test all(x -> isapprox(x, 1 / (V * P); rtol = 1e-3), res_flat.plan)
    # cost-translation invariance: C + const -> same plan
    res_t = solve(s, C .+ 42.0, 0.5)
    @test isapprox(res_t.plan, res.plan; rtol = 1e-9)
    # scale_cost float divides
    res_sc = solve(s, C, 0.5; scale_cost = 2.0)
    res_eq = solve(s, C ./ 2, 0.5)
    @test isapprox(res_sc.plan, res_eq.plan; rtol = 1e-12)
    # sanitize: Inf cost -> vertex never picked, finite plan
    Cinf = copy(C);
    Cinf[3, 1] = Inf
    res_inf = solve(s, Cinf, 0.5)
    @test all(isfinite, res_inf.plan)
    @test res_inf.plan[3, 1] < 1e-10
    # input not mutated
    Ccopy = copy(Cinf)
    solve(s, Cinf, 0.5)
    @test Cinf == Ccopy
    @test_throws ArgumentError solve(s, C, 0.0)
    @test_throws ArgumentError solve(s, C, 0.5; a = fill(-1.0, P))
end

@testset "tempered softmax ignores eps" begin
    rng = Xoshiro(11)
    C = randn(rng, 4, 5)
    ts = TemperedSoftmaxSolver(tau = 0.7)
    r1 = solve(ts, C, 0.001)
    r2 = solve(ts, C, 100.0)
    @test r1.plan == r2.plan                     # eps argument must not matter
    ref = solve(SoftmaxSolver(), C, 0.7)
    @test isapprox(r1.plan, ref.plan; rtol = 1e-12)  # equals softmax at eps=tau
    @test_throws ArgumentError solve(TemperedSoftmaxSolver(tau = 0.0), C, 1.0)
end

@testset "greedy solvers" begin
    C = [1.0 5.0; 3.0 2.0; 2.0 2.0]   # V=3, P=2; column 2 has tie at rows 2,3
    g = solve(MinCostGreedySolver(), C, 0.1)
    @test g.plan[1, 1] == 0.5 && sum(g.plan[:, 1]) == 0.5
    @test g.plan[2, 2] == 0.5          # tie -> first occurrence
    tk = solve(TopKMeanSolver(k = 2), C, 0.1)
    @test isapprox(tk.plan[:, 1], [0.25, 0.0, 0.25])
    @test isapprox(tk.plan[:, 2], [0.0, 0.25, 0.25])
    # k > V falls back to all vertices
    tk_all = solve(TopKMeanSolver(k = 99), C, 0.1)
    @test all(isapprox(0.5 / 3), tk_all.plan[:, 1])
end

@testset "sinkhorn" begin
    rng = Xoshiro(12)
    V, P = 8, 5                       # non-square: catches swapped f/g orientation
    C = randn(rng, V, P)
    s = SinkhornSolver(threshold = 1e-9, max_iterations = 5000)
    res = solve(s, C, 0.2)
    @test res.converged
    @test isapprox(vec(sum(res.plan; dims = 1)), fill(1 / P, P); atol = 1e-7)  # P1 = a
    @test isapprox(vec(sum(res.plan; dims = 2)), fill(1 / V, V); atol = 1e-7)  # P'1 = b
    # nonuniform marginals
    a = rand(rng, P);
    a ./= sum(a)
    b = rand(rng, V);
    b ./= sum(b)
    res_ab = solve(s, C, 0.2; a, b)
    @test isapprox(vec(sum(res_ab.plan; dims = 1)), a; atol = 1e-7)
    @test isapprox(vec(sum(res_ab.plan; dims = 2)), b; atol = 1e-7)
    # ent_reg_cost = <f,a> + <g,b> - eps*sum(a)
    @test isapprox(res_ab.ent_cost, dot(res_ab.f, a) + dot(res_ab.g, b) - 0.2 * sum(a); rtol = 1e-10)
    # warm start converges faster
    cold = solve(s, C, 0.2; a, b)
    warm = solve(s, C, 0.2; a, b, f0 = cold.f, g0 = cold.g)
    @test warm.iters <= cold.iters
    @test isapprox(warm.plan, cold.plan; atol = 1e-6)
    # gauge invariance: f+c, g-c yields the same plan
    shifted = solve(s, C, 0.2; a, b, f0 = cold.f .+ 5.0, g0 = cold.g .- 5.0)
    @test isapprox(shifted.plan, cold.plan; atol = 1e-6)
    # eps-rescale heuristic path runs and still converges
    res_rs = solve(s, C, 0.1; a, b, f0 = cold.f, g0 = cold.g, last_eps = 0.2)
    @test res_rs.converged
    # non-finite warm start falls back to zeros (still converges)
    res_nf = solve(s, C, 0.2; f0 = fill(NaN, P), g0 = zeros(V))
    @test res_nf.converged
    # shape-mismatched warm start ignored with a warning
    res_mm = @test_logs (:warn, r"expected") solve(s, C, 0.2; f0 = zeros(P + 1))
    @test res_mm.converged
    # overrelaxation converges to the same plan
    res_sor = solve(
        SinkhornSolver(threshold = 1e-9, max_iterations = 5000, omega = 1.5), C, 0.2; a, b)
    @test isapprox(res_sor.plan, res_ab.plan; atol = 1e-6)
    # adaptive omega + anderson paths run and converge
    res_ad = solve(SinkhornSolver(threshold = 1e-9, max_iterations = 5000, adaptive_omega = true), C, 0.2)
    @test res_ad.converged
    res_aa = solve(SinkhornSolver(threshold = 1e-9, max_iterations = 5000, anderson_depth = 3), C, 0.2)
    @test res_aa.converged
    @test isapprox(res_aa.plan, res.plan; atol = 1e-6)
    # fixed-iteration mode: finite result reports converged (ProgressiveEpsilon contract)
    res_fx = solve(SinkhornSolver(threshold = 0.0, max_iterations = 50), C, 0.2)
    @test res_fx.converged && res_fx.iters == 50
    # numerical stress: huge, tiny, equal, Inf costs stay finite
    for Cx in (fill(1e8, 4, 3), fill(1e-12, 4, 3), zeros(4, 3),
        [1.0 2 3; 4 5 6; 7 8 9; Inf 1 2])
        r = solve(SinkhornSolver(max_iterations = 500), Cx, 0.5)
        @test all(isfinite, r.plan)
    end
    # ctor validation
    @test_throws ArgumentError SinkhornSolver(omega = 2.5)
    @test_throws ArgumentError SinkhornSolver(check_every = 0)
    @test_throws ArgumentError SinkhornSolver(max_iterations = 0)
    @test_throws ArgumentError solve(s, zeros(0, 0), 0.5)
end

@testset "kl softmax" begin
    rng = Xoshiro(13)
    V, P = 6, 4
    C = randn(rng, V, P)
    # lam = 0 == softmax
    kl0 = solve(KLSoftmaxSolver(lam = 0.0), C, 0.3)
    sm = solve(SoftmaxSolver(), C, 0.3)
    @test isapprox(kl0.plan, sm.plan; rtol = 1e-10)
    @test kl0.iters == 1
    # lam = Inf == sinkhorn (marginals within tolerance)
    klinf = solve(KLSoftmaxSolver(lam = Inf, threshold = 1e-10, max_iterations = 5000), C, 0.3)
    @test isapprox(vec(sum(klinf.plan; dims = 1)), fill(1 / P, P); atol = 1e-6)
    @test isapprox(vec(sum(klinf.plan; dims = 2)), fill(1 / V, V); atol = 1e-5)
    # intermediate lam: row marginal exact, column violation between the extremes
    klmid = solve(KLSoftmaxSolver(lam = 0.3, threshold = 1e-10, max_iterations = 5000), C, 0.3)
    @test isapprox(vec(sum(klmid.plan; dims = 1)), fill(1 / P, P); atol = 1e-6)
    s0 = KLSoftmaxSolver(lam = 0.0)
    smid = KLSoftmaxSolver(lam = 0.3, threshold = 1e-10, max_iterations = 5000)
    sinf = KLSoftmaxSolver(lam = Inf, threshold = 1e-10, max_iterations = 5000)
    for s in (s0, smid, sinf)
        solve(s, C, 0.3)
    end
    @test sinf.last_marginal_violation < smid.last_marginal_violation < s0.last_marginal_violation
    @test_throws ArgumentError KLSoftmaxSolver(lam = -1.0)
end
