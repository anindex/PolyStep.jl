using PolyStep: softmax_cols!, lse_cols!, lse_rows!, sanitize_cost!, scale_cost!,
                 normalize_particle_masses!, _batched_mul!, _batched_matvec!, splitmix64

# reference implementations (naive, max-subtracted)
ref_softmax(C, eps) = begin
    X = -C ./ eps
    E = exp.(X .- maximum(X; dims = 1))
    E ./ sum(E; dims = 1)
end
ref_lse(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))

@testset "numeric" begin
    @testset "splitmix64" begin
        # canonical SplitMix64 first output for seed 0
        @test splitmix64(UInt64(0)) == 0xe220a8397b1dcdaf
        @test splitmix64(0, 1, 2) isa UInt64
        @test splitmix64(0, 1, 2) == splitmix64(0, 1, 2)
        @test splitmix64(0, 1, 2) != splitmix64(0, 2, 1)
        @test splitmix64(42) != splitmix64(43)
        @test splitmix64(splitmix64(42, 1), 2) isa UInt64
        @test splitmix64(-5, 7) == splitmix64(xor(splitmix64(reinterpret(UInt64, -5)), UInt64(7)))
    end

    @testset "softmax_cols! T=$T" for T in (Float64, Float32)
        rng = Xoshiro(1)
        C = randn(rng, T, 6, 17)
        W = similar(C)
        eps = T(0.3)
        softmax_cols!(W, C, eps)
        @test isapprox(W, ref_softmax(C, eps); rtol = (T === Float64 ? 1e-12 : 1e-5))
        @test all(isapprox(1), sum(W; dims = 1))
        # huge costs stay finite thanks to max subtraction
        Ch = T.([1e30 2e30; 3e30 1e30])
        Wh = similar(Ch)
        softmax_cols!(Wh, Ch, T(1))
        @test all(isfinite, Wh)
        @test all(isapprox(1), sum(Wh; dims = 1))
        # generic (non-Matrix) method agrees with the fast path
        Wg = similar(C)
        softmax_cols!(Wg, view(C, :, :), eps)
        @test isapprox(Wg, W; rtol = (T === Float64 ? 1e-12 : 1e-5))
        for e in (1e-46, floatmin(Float64) / 4)
            softmax_cols!(W, C, e)
            @test W[:, 1] == (1:6 .== argmin(C[:, 1]))
            softmax_cols!(Wg, view(C, :, :), e)
            @test Wg[:, 1] == (1:6 .== argmin(C[:, 1]))
        end
    end

    @testset "lse kernels" begin
        rng = Xoshiro(2)
        A = randn(rng, 5, 11)
        addv = randn(rng, 5)
        addp = randn(rng, 11)
        out_p = zeros(11)
        lse_cols!(out_p, A, addv)
        @test isapprox(out_p, [ref_lse(A[:, p] .+ addv) for p in 1:11]; rtol = 1e-12)
        out_v = zeros(5)
        accm = zeros(5)
        accs = zeros(5)
        lse_rows!(out_v, A, addp, accm, accs)
        @test isapprox(out_v, [ref_lse(A[v, :] .+ addp) for v in 1:5]; rtol = 1e-12)
        # generic methods
        out_p2 = zeros(11)
        lse_cols!(out_p2, view(A, :, :), addv)
        @test isapprox(out_p2, out_p; rtol = 1e-12)
        out_v2 = zeros(5)
        lse_rows!(out_v2, view(A, :, :), addp, nothing, nothing)
        @test isapprox(out_v2, out_v; rtol = 1e-12)
        B = [-Inf 0.0 -Inf; 0.0 -Inf -Inf]
        for Bx in (B, view(B, :, :))
            @test lse_rows!(zeros(2), Bx, zeros(3), zeros(2), zeros(2)) == [0.0, 0.0]
            @test lse_cols!(zeros(3), Bx, zeros(2)) == [0.0, 0.0, -Inf]
        end
    end

    @testset "sanitize_cost!" begin
        C = [1.0 Inf; -2.0 NaN]
        sanitize_cost!(C)
        @test C == [1.0 5.0; -2.0 5.0]  # 2*max|finite|+1, no absolute floor
        C2 = [1e7 -Inf]
        sanitize_cost!(C2)
        @test C2 == [1e7 (2e7 + 1)]
        C3 = [1.0 2.0; 3.0 4.0]
        @test sanitize_cost!(copy(C3)) == C3
        Csat = [1e308, Inf, -3.0]
        sanitize_cost!(Csat)
        @test all(isfinite, Csat)
        C32 = Float32[3.0f37, Inf32]
        sanitize_cost!(C32)
        @test all(isfinite, C32)
        # generic method
        C4 = view([1.0 Inf; -2.0 NaN], :, :)
        sanitize_cost!(C4)
        @test C4 == [1.0 5.0; -2.0 5.0]
    end

    @testset "scale_cost!" begin
        C = [1.0 -2.0; 3.0 -4.0]
        Cs = similar(C)
        @test scale_cost!(Cs, C, nothing) == C
        # :mean/:max recenter by min(C) first
        scale_cost!(Cs, C, :mean)
        @test isapprox(Cs, (C .- minimum(C)) ./ mean(C .- minimum(C)))
        scale_cost!(Cs, C, :max)
        @test isapprox(Cs, (C .- minimum(C)) ./ 7.0)
        # shift-invariant: f and f + c get the same temperature
        for spec in (:mean, :max)
            @test isapprox(scale_cost!(similar(C), C .+ 100, spec), scale_cost!(Cs, C, spec))
        end
        scale_cost!(Cs, C, 2.0)
        @test isapprox(Cs, C ./ 2.0)
        @test_throws ArgumentError scale_cost!(Cs, C, -1.0)
        @test_throws ArgumentError scale_cost!(Cs, C, 0.0)
        @test_throws ArgumentError scale_cost!(Cs, C, Inf)
        @test_throws ArgumentError scale_cost!(Cs, C, :bogus)
        # tiny-cost clamp floor 1e-10
        Z = zeros(2, 2)
        Zs = similar(Z)
        scale_cost!(Zs, Z, :mean)
        @test all(iszero, Zs)
        @test all(iszero, scale_cost!(similar(Z, Float16), fill(Float16(3), 2, 2), :mean))
        Cbig = fill(1e308, 2, 2)
        Csb = similar(Cbig)
        scale_cost!(Csb, Cbig, :mean)
        @test all(isfinite, Csb)
        Cpm = [-1e308 1e308; 0.0 0.0]
        @test scale_cost!(similar(Cpm), Cpm, :mean) == [0.0 1.0; 0.5 0.5]
        @test scale_cost!(similar(Cpm), Cpm, :max) == [0.0 1.0; 0.5 0.5]
    end

    @testset "normalize_particle_masses!" begin
        rng = Xoshiro(3)
        plan = rand(rng, 4, 7) .* 0.3
        Wn = similar(plan)
        normalize_particle_masses!(Wn, plan)
        @test all(isapprox(1), sum(Wn; dims = 1))
        @test isapprox(Wn, plan ./ sum(plan; dims = 1); rtol = 1e-12)
        plan[:, 3] .= 0
        plan[:, 2] .= 1e-15
        normalize_particle_masses!(Wn, plan)
        @test all(iszero, Wn[:, 3])
        @test all(iszero, Wn[:, 2])
        @test all(isfinite, Wn)
        # generic
        Wg = similar(plan)
        normalize_particle_masses!(Wg, view(plan, :, :))
        @test isapprox(Wg, Wn; rtol = 1e-12)
    end

    @testset "batched mul" begin
        rng = Xoshiro(4)
        A = randn(rng, 3, 3, 5)
        B = randn(rng, 3, 6)
        Y = zeros(3, 6, 5)
        _batched_mul!(Y, A, B)
        for p in 1:5
            @test isapprox(Y[:, :, p], A[:, :, p] * B; rtol = 1e-12)
        end
        X = randn(rng, 3, 5)
        Yv = zeros(3, 5)
        _batched_matvec!(Yv, A, X)
        for p in 1:5
            @test isapprox(Yv[:, p], A[:, :, p] * X[:, p]; rtol = 1e-12)
        end
    end
end
