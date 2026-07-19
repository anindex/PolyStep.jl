# GPU tests: run with JPOLYSTEP_TEST_CUDA=1 on a CUDA-capable machine
# (needs a clean library environment: a system CUDA on LD_LIBRARY_PATH can
# clash with CUDA.jl artifacts).
using CUDA
using PolyStep: softmax_cols!, sanitize_cost!, normalize_particle_masses!,
                 _batched_mul!, _batched_matvec!, cuda_objective

@testset "cuda extension" begin
    @assert CUDA.functional() "JPOLYSTEP_TEST_CUDA=1 but CUDA is not functional"
    rng = Xoshiro(2024)
    d, V, P = 4, 8, 16
    A = randn(rng, Float32, d, d, P)
    B = randn(rng, Float32, d, V)
    Yc = zeros(Float32, d, V, P)
    _batched_mul!(Yc, A, B)
    Yd = CUDA.zeros(Float32, d, V, P)
    _batched_mul!(Yd, CuArray(A), CuArray(B))
    @test isapprox(Array(Yd), Yc; rtol = 1e-5)

    X = randn(rng, Float32, d, P)
    Ycv = zeros(Float32, d, P)
    _batched_matvec!(Ycv, A, X)
    Ydv = CUDA.zeros(Float32, d, P)
    _batched_matvec!(Ydv, CuArray(A), CuArray(X))
    @test isapprox(Array(Ydv), Ycv; rtol = 1e-5)

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

    # end-to-end: GPU-evaluated objective through the CPU step loop
    f_gpu(Xd) = vec(sum(abs2, Xd; dims = 1))
    es = minimize(cuda_objective(f_gpu), 6; steps = 60, epsilon = 0.05,
        step_radius = 0.3, x0 = fill(2.0, 6), rng = Xoshiro(5))
    @test es.best_f < 4.0     # from ||x||^2 = 24
end
