using PolyStep: softmax_cols!, lse_cols!, lse_rows!, sanitize_cost!, scale_cost!,
                 normalize_particle_masses!, _cost_from_losses!, _probe_points_orthoplex!,
                 haar_rotations!, fd_gradient!, fd_hessian_diag!, newton_step!

@testset "zero steady-state allocation" begin
    T = Float64
    d, P, K = 6, 64, 2
    V = 2d
    rng = Xoshiro(99)
    C = randn(rng, T, V, P)
    W = similar(C)
    out_p = zeros(T, P)
    out_v = zeros(T, V)
    addv = randn(rng, T, V)
    addp = randn(rng, T, P)
    accm = zeros(T, V)
    accs = zeros(T, V)
    losses = randn(rng, T, K * V * P)
    Cm = zeros(T, V, P)
    X = randn(rng, T, d, P)
    R = zeros(T, d, d, P)
    Z = zeros(T, d, d, P)
    Xp = zeros(T, d, K * V * P)
    scales = PolyStep.probe_scales(T, K)
    L3 = randn(rng, T, K, V, P)
    G = zeros(T, d, P)
    H = zeros(T, d, P)
    N = zeros(T, d, P)

    checks = [
        () -> softmax_cols!(W, C, 0.3),
        () -> lse_cols!(out_p, C, addv),
        () -> lse_rows!(out_v, C, addp, accm, accs),
        () -> sanitize_cost!(C),                       # all-finite fast path
        () -> scale_cost!(Cm, C, :mean),
        () -> normalize_particle_masses!(W, abs.(C)),
        () -> _cost_from_losses!(Cm, losses, K),
        () -> _probe_points_orthoplex!(Xp, X, R, T(0.5), scales),
        () -> haar_rotations!(R, Z, rng),
        () -> fd_gradient!(G, L3, scales, 0.5),
        () -> fd_hessian_diag!(H, L3, scales, 0.5),
        () -> newton_step!(N, G, H; max_step_norm = 1.0)
    ]
    for f in checks
        f()
        f()
    end
    absC = abs.(C)
    @test @allocated(softmax_cols!(W, C, 0.3)) == 0
    lse_bound = VERSION < v"1.11" ? 2048 : 0
    @test @allocated(lse_cols!(out_p, C, addv)) <= lse_bound
    @test @allocated(lse_rows!(out_v, C, addp, accm, accs)) <= lse_bound
    @test @allocated(sanitize_cost!(C)) == 0
    @test @allocated(scale_cost!(Cm, C, :mean)) == 0
    @test @allocated(normalize_particle_masses!(W, absC)) == 0
    @test @allocated(_cost_from_losses!(Cm, losses, K)) == 0
    @test @allocated(fd_gradient!(G, L3, scales, 0.5)) == 0
    @test @allocated(fd_hessian_diag!(H, L3, scales, 0.5)) == 0
    # Julia 1.10 boxes the keyword NamedTuple (0 B on 1.11+)
    @test @allocated(newton_step!(N, G, H; max_step_norm = 1.0)) <= 512
    @test @allocated(_probe_points_orthoplex!(Xp, X, R, T(0.5), scales)) <= 512
    @test @allocated(haar_rotations!(R, Z, rng)) <= 512

    es = PolyStepES(8; num_particles = 2, rng = Xoshiro(7))
    ask!(es)
    tell!(es, zeros(popsize(es)))
    @test @allocated(ask!(es)) <= (VERSION < v"1.11" ? 512 : 0)
    tell!(es, zeros(popsize(es)))

    # whole-step budgets: no per-step (V,P) allocations
    fobj(Xm) = vec(sum(abs2, Xm; dims = 1))
    ps_soft = PolyStepConfig(dim = 8)
    st_soft = init_state(ps_soft, randn(rng, 8, 64))
    ps_sink = PolyStepConfig(dim = 8, solver = SinkhornSolver(max_iterations = 50))
    st_sink = init_state(ps_sink, randn(rng, 8, 64))
    for _ in 1:2
        PolyStep.step!(fobj, ps_soft, st_soft; rng)
        PolyStep.step!(fobj, ps_sink, st_sink; rng)
    end
    @test @allocated(PolyStep.step!(fobj, ps_soft, st_soft; rng)) < 32_000
    @test @allocated(PolyStep.step!(fobj, ps_sink, st_sink; rng)) < 32_000
    s_fx = SinkhornSolver(threshold = 0.0, max_iterations = 1000)
    C_fx = rand(rng, 16, 8)
    PolyStep.solve(s_fx, C_fx, 0.1)
    @test @allocated(PolyStep.solve(s_fx, C_fx, 0.1)) < 4_000

    ps_quad = PolyStepConfig(dim = 8, num_probe = 2, use_quadratic_model = true,
        newton_refinement = true, trust_region = true, biased_rotation = true)
    st_quad = init_state(ps_quad, randn(rng, 8, 64))
    for _ in 1:2
        PolyStep.step!(fobj, ps_quad, st_quad; rng)
    end
    @test @allocated(PolyStep.step!(fobj, ps_quad, st_quad; rng)) < 32_000
end
