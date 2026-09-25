# Results must be bit-identical for any thread count; P >= 1024 and d >= 256 hit the
# threaded kernels. Pkg.test forces --check-bounds=yes, which changes @simd reductions
# in threaded closures, so the child uses the default.
@testset "thread-count invariance" begin
    code = """
    using PolyStep, Random
    f(X) = vec(sum(abs2, X; dims = 1))
    ps = PolyStepConfig(dim = 12, solver = SinkhornSolver(), max_iterations = 30)
    st = init_state(ps, ones(12, 1200))
    solve!(f, ps, st; rng = Xoshiro(1))
    es = minimize(f, 300; num_particles = 4, steps = 3, x0 = ones(300))
    print(hash((st.X, es.X)))
    """
    run_with(t) = read(`$(Base.julia_cmd()) --check-bounds=auto --startup-file=no --project=$(Base.active_project()) -t $t -e $code`, String)
    @test run_with(1) == run_with(4)
end
