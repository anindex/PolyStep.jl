using CUDA
using PolyStep: softmax_cols!, sanitize_cost!, normalize_particle_masses!, cuda_objective

@testset "cuda extension" begin
    @assert CUDA.functional() "JPOLYSTEP_TEST_CUDA=1 but CUDA is not functional"
    rng = Xoshiro(2024)
    V, P = 8, 16

    # generic solver kernels on device match CPU
    C = randn(rng, Float32, V, P)
    W = similar(C)
    softmax_cols!(W, C, 0.3f0)
    Wd = CuArray(similar(C))
    softmax_cols!(Wd, CuArray(C), 0.3f0)
    @test isapprox(Array(Wd), W; rtol = 1e-5)
    Cinf = copy(C)
    Cinf[2, 3] = Inf32
    Cid = CuArray(Cinf)
    sanitize_cost!(Cid)
    sanitize_cost!(Cinf)
    @test isapprox(Array(Cid), Cinf; rtol = 1e-6)
    Wn = similar(W)
    normalize_particle_masses!(Wn, W)
    Wnd = CuArray(similar(W))
    normalize_particle_masses!(Wnd, CuArray(W))
    @test isapprox(Array(Wnd), Wn; rtol = 1e-5)

    f_gpu(Xd) = vec(sum(abs2, Xd; dims = 1))
    es = minimize(cuda_objective(f_gpu), 6; steps = 60, epsilon = 0.05,
        step_radius = 0.3, x0 = fill(2.0, 6), rng = Xoshiro(5))
    @test es.best_f < 4.0     # from ||x||^2 = 24
end
