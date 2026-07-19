using PolyStep: orthoplex_vertices, simplex_vertices, cube_vertices, polytope_vertices,
                 num_vertices, probe_scales, haar_rotations!, biased_rotation!,
                 _qr_slices_static!, _qr_slices_lapack!, _mezzadri_phase

isrotation(Q; atol = 1e-8) = isapprox(Q' * Q, I; atol) && isapprox(det(Q), 1; atol)

@testset "geometry" begin
    @testset "orthoplex template" begin
        d = 5
        V = orthoplex_vertices(d)
        @test size(V) == (d, 2d)
        # vertex order contract: cols 1..d = +e_i, d+1..2d = -e_i
        for i in 1:d
            e = zeros(d);
            e[i] = 1
            @test V[:, i] == e
            @test V[:, d + i] == -e
        end
        @test orthoplex_vertices(Float32, 3; radius = 2)[1, 1] === 2.0f0
    end

    @testset "simplex template" begin
        for d in (2, 3, 8)
            V = simplex_vertices(d)
            @test size(V) == (d, d + 1)
            @test maximum(abs, mean(V; dims = 2)) < 1e-12          # centered
            norms = [norm(V[:, v]) for v in 1:(d + 1)]
            @test all(n -> isapprox(n, norms[1]; rtol = 1e-10), norms)  # regular
            dots = [dot(V[:, 1], V[:, v]) for v in 2:(d + 1)]
            @test all(x -> isapprox(x, dots[1]; rtol = 1e-8), dots)
            @test isapprox(simplex_vertices(d; radius = 3), 3 .* V; rtol = 1e-12)
        end
    end

    @testset "cube template" begin
        d = 4
        V = cube_vertices(d)
        @test size(V) == (d, 2^d)
        @test all(x -> isapprox(abs(x), 1 / sqrt(d); rtol = 1e-12), V)
        @test isapprox(V[:, 1], fill(1 / sqrt(d), d))  # index 0: all bits 0 -> +
        v2 = fill(1 / sqrt(d), d);
        v2[1] = -1 / sqrt(d)
        @test isapprox(V[:, 2], v2)  # index 1: bit 0 set -> coord 1 negative
        @test length(unique(eachcol(V))) == 2^d
    end

    @testset "dispatch + counts" begin
        @test polytope_vertices(:orthoplex, 3) == orthoplex_vertices(3)
        @test polytope_vertices(:simplex, 3) == simplex_vertices(3)
        @test polytope_vertices(:cube, 3) == cube_vertices(3)
        @test_throws ArgumentError polytope_vertices(:bogus, 3)
        @test num_vertices(:orthoplex, 7) == 14
        @test num_vertices(:simplex, 7) == 8
        @test num_vertices(:cube, 7) == 128
    end

    @testset "probe scales" begin
        @test probe_scales(Float64, 1) == [0.5]
        @test probe_scales(Float64, 3) == [0.25, 0.5, 0.75]
        @test isapprox(probe_scales(Float32, 2), [1 / 3, 2 / 3]; rtol = 1e-6)
    end

    @testset "mezzadri phase" begin
        @test _mezzadri_phase(0.0) === 1.0      # sign(0) := +1 (avoids nulling a column)
        @test _mezzadri_phase(-0.0) === 1.0
        @test _mezzadri_phase(3.0) === 1.0
        @test _mezzadri_phase(-3.0) === -1.0
    end

    @testset "haar rotations d=$d" for d in (2, 3, 5, 8, 12)
        P = 32
        rng = Xoshiro(7)
        R = zeros(d, d, P)
        Z = zeros(d, d, P)
        haar_rotations!(R, Z, rng)
        for p in 1:P
            @test isrotation(R[:, :, p])
        end
        # determinism: same seed -> identical rotations
        R2 = zeros(d, d, P)
        haar_rotations!(R2, zeros(d, d, P), Xoshiro(7))
        @test R2 == R
        if d == 2
            for p in 1:P
                @test isapprox(R[1, 1, p], R[2, 2, p])
                @test isapprox(R[2, 1, p], -R[1, 2, p])
            end
        end
    end

    @testset "haar distribution smoke" begin
        d, P = 4, 4000
        R = zeros(d, d, P)
        haar_rotations!(R, zeros(d, d, P), Xoshiro(11))
        @test abs(mean(R[1, 1, :])) < 0.05   # Haar => entries mean 0
        @test abs(mean(R[3, 2, :])) < 0.05
    end

    @testset "cross-impl QR agreement d=$d" for d in 3:8
        # StaticArrays path vs LAPACK path on identical full-rank Gaussian Z
        P = 16
        Z = randn(Xoshiro(100 + d), d, d, P)
        R1 = zeros(d, d, P)
        R2 = zeros(d, d, P)
        _qr_slices_static!(R1, copy(Z), Val(d))
        _qr_slices_lapack!(R2, copy(Z))       # mutates its Z copy
        @test isapprox(R1, R2; rtol = 1e-10)
    end

    @testset "biased rotation d=$d" for d in (2, 3, 6, 12)
        P = 24
        rng = Xoshiro(21)
        R = zeros(d, d, P)
        haar_rotations!(R, zeros(d, d, P), rng)
        bias = randn(rng, d, P)
        for p in 1:P
            bias[:, p] ./= norm(bias[:, p])
        end
        biased_rotation!(R, bias)
        for p in 1:P
            @test isrotation(R[:, :, p])
            # column 1 must equal the bias itself, not its negation
            @test isapprox(R[:, 1, p], bias[:, p]; atol = 1e-8)
        end
    end

    @testset "biased rotation sign-ambiguous case" begin
        # R = I, bias = -e1: realign must keep column 1 = bias (not flip to +e1)
        d = 4
        R = zeros(d, d, 1)
        R[:, :, 1] = Matrix{Float64}(I, d, d)
        bias = zeros(d, 1)
        bias[1, 1] = -1.0
        biased_rotation!(R, bias)
        @test isapprox(R[:, 1, 1], bias[:, 1]; atol = 1e-12)
        @test isrotation(R[:, :, 1])
    end
end
