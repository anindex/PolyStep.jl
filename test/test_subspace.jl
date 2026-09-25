const DN_GNN_SHAPES = [(64, 7), (64, 7), (64,), (64, 64), (64, 64), (64,), (64, 64), (64, 64),
                       (64,), (64, 64), (64,), (64, 64), (64,), (1, 64), (1,)]
const BIAS_ONLY_SHAPES = [(10,), (3,)]
const FULL_WIDTH_SHAPES = [(1, 64), (5, 5), (7,)]
const MIXED_SHAPES = [(128, 32), (128,), (10, 128), (10,)]

const PY_REFERENCE = Dict(
    "districtnet_gnn" => Dict((1, 0) => 1295, (1, 256) => 393, (1, 512) => 512, (1, 1024) => 1024,
                              (1, 2048) => 1295, (2, 0) => 2205, (2, 256) => 393, (2, 512) => 512,
                              (2, 1024) => 1024, (2, 2048) => 2048, (4, 0) => 4025, (4, 256) => 393,
                              (4, 512) => 512, (4, 1024) => 1024, (4, 2048) => 2048, (8, 0) => 7425,
                              (8, 256) => 1287, (8, 512) => 1287, (8, 1024) => 1287,
                              (8, 2048) => 2048, (32, 0) => 25857, (32, 512) => 25857),
    "bias_only" => Dict((1, 0) => 13, (1, 256) => 13, (2, 0) => 13, (4, 512) => 13, (8, 0) => 13,
                        (32, 2048) => 13),
    "full_width" => Dict((1, 0) => 81, (1, 512) => 81, (2, 0) => 91, (2, 1024) => 91, (4, 0) => 96,
                         (8, 0) => 96, (8, 256) => 96, (32, 0) => 96),
    "mixed" => Dict((1, 0) => 436, (1, 256) => 256, (1, 512) => 436, (2, 0) => 734,
                    (2, 256) => 256, (2, 512) => 512, (2, 1024) => 734, (4, 0) => 1330,
                    (4, 512) => 512, (4, 1024) => 1024, (4, 2048) => 1330, (8, 0) => 2522,
                    (8, 256) => 256, (8, 2048) => 2048, (32, 0) => 5514, (32, 1024) => 5514))

const PY_LAYOUTS = Dict("districtnet_gnn" => DN_GNN_SHAPES, "bias_only" => BIAS_ONLY_SHAPES,
                        "full_width" => FULL_WIDTH_SHAPES, "mixed" => MIXED_SHAPES)

const PY_DN_COORDS_R8_CAP512 = [448, 448, 64, 1, 1, 64, 1, 1, 64, 1, 64, 1, 64, 64, 1]
const PY_DN_COORDS_R1_CAP512 = [10, 10, 64, 18, 18, 64, 18, 18, 64, 18, 64, 17, 64, 64, 1]
const PY_MIXED_COORDS_R4_CAP512 = [201, 128, 173, 10]
const PY_MIXED_COORDS_R8_CAP256 = [63, 128, 55, 10]

@testset "subspace" begin
    @testset "ParamLayout" begin
        l = ParamLayout(["a" => (3, 4), "b" => (5,)])
        @test l.total_params == 17
        @test [e.offset for e in l.entries] == [0, 12]
        @test [e.numel for e in l.entries] == [12, 5]
        @test [e.key for e in l.entries] == ["a", "b"]
        @test length(l) == 2
        la = ParamLayout([zeros(3, 4), zeros(5)])
        @test [e.shape for e in la.entries] == [(3, 4), (5,)]
        @test [e.key for e in la.entries] == ["p1", "p2"]
    end

    @testset "dimension arithmetic against python from_layout" begin
        for (name, shapes) in PY_LAYOUTS
            layout = ParamLayout([("p$(i)" => s) for (i, s) in enumerate(shapes)])
            for ((rank, cap), dim) in PY_REFERENCE[name]
                s = @test_logs min_level = Logging.Error HybridSubspace(layout; rank = rank,
                    seed = 0, max_subspace_dim = cap == 0 ? nothing : cap)
                @test subspace_dim(s) == dim
                @test sum(sp.ncoords for sp in s.specs) == dim
                @test s.total_params == sum(prod(sh; init = 1) for sh in shapes)
            end
        end
    end

    @testset "per-layer coordinate counts" begin
        dn = ParamLayout([("p$(i)" => s) for (i, s) in enumerate(DN_GNN_SHAPES)])
        s8 = @test_logs min_level = Logging.Error HybridSubspace(dn; rank = 8, max_subspace_dim = 512)
        @test [sp.ncoords for sp in s8.specs] == PY_DN_COORDS_R8_CAP512
        @test [sp.projected for sp in s8.specs] ==
              [c < n for (c, n) in zip(PY_DN_COORDS_R8_CAP512, [sp.numel for sp in s8.specs])]
        s1 = @test_logs min_level = Logging.Error HybridSubspace(dn; rank = 1, max_subspace_dim = 512)
        @test [sp.ncoords for sp in s1.specs] == PY_DN_COORDS_R1_CAP512
        mixed = ParamLayout([("p$(i)" => s) for (i, s) in enumerate(MIXED_SHAPES)])
        @test [sp.ncoords for sp in HybridSubspace(mixed; rank = 4, max_subspace_dim = 512).specs] ==
              PY_MIXED_COORDS_R4_CAP512
        @test [sp.ncoords for sp in HybridSubspace(mixed; rank = 8, max_subspace_dim = 256).specs] ==
              PY_MIXED_COORDS_R8_CAP256
        @test isapprox(compression_ratio(s1), 512 / 25857)
        conv = HybridSubspace(ParamLayout(["c" => (3, 3, 16, 32)]); rank = 4)
        @test conv.specs[1].ncoords == 704
        @test conv.specs[1].projected
    end

    @testset "unprojected parameters" begin
        layout = ParamLayout([("p$(i)" => s) for (i, s) in enumerate(FULL_WIDTH_SHAPES)])
        s = HybridSubspace(layout; rank = 4)
        @test all(!sp.projected for sp in s.specs)
        @test subspace_dim(s) == s.total_params
        z = randn(Xoshiro(3), subspace_dim(s))
        @test isapprox(expand(s, z), z)
        bias = ParamLayout([("p$(i)" => sh) for (i, sh) in enumerate(BIAS_ONLY_SHAPES)])
        sb = HybridSubspace(bias; rank = 8)
        @test all(!sp.projected for sp in sb.specs)
        @test subspace_dim(sb) == 13
    end

    @testset "orthonormal per-layer factors" begin
        mixed = ParamLayout([("p$(i)" => s) for (i, s) in enumerate(MIXED_SHAPES)])
        s = HybridSubspace(mixed; rank = 2, seed = 7)
        for (P, sp) in zip(s.projections, s.specs)
            sp.projected || continue
            @test size(P) == (sp.numel, sp.ncoords)
            @test isapprox(transpose(P) * P, I(sp.ncoords); atol = 1e-10)
        end
    end

    @testset "round trip on the span" begin
        mixed = ParamLayout([("p$(i)" => s) for (i, s) in enumerate(MIXED_SHAPES)])
        s = HybridSubspace(mixed; rank = 3, seed = 11)
        z = randn(Xoshiro(5), subspace_dim(s))
        x = expand(s, z)
        @test length(x) == s.total_params
        @test isapprox(project(s, x), z; atol = 1e-10)
        @test isapprox(expand(s, project(s, x)), x; atol = 1e-10)
        zz = similar(z)
        @test isapprox(project!(zz, s, x), z; atol = 1e-10)
        xx = similar(x)
        @test isapprox(expand!(xx, s, z), x)
        @test_throws DimensionMismatch expand(s, zeros(subspace_dim(s) + 1))
        @test_throws DimensionMismatch project(s, zeros(s.total_params + 1))
    end

    @testset "seed determinism" begin
        layout = ParamLayout([("p$(i)" => s) for (i, s) in enumerate(MIXED_SHAPES)])
        a = HybridSubspace(layout; rank = 2, seed = 42)
        b = HybridSubspace(layout; rank = 2, seed = 42)
        c = HybridSubspace(layout; rank = 2, seed = 43)
        @test all(a.projections[k] == b.projections[k] for k in eachindex(a.projections))
        @test any(a.specs[k].projected && a.projections[k] != c.projections[k]
                  for k in eachindex(a.projections))
    end

    @testset "cap semantics" begin
        mixed = ParamLayout([("p$(i)" => s) for (i, s) in enumerate(MIXED_SHAPES)])
        @test subspace_dim(HybridSubspace(mixed; rank = 8, max_subspace_dim = 256)) == 256
        @test subspace_dim(HybridSubspace(mixed; rank = 1, max_subspace_dim = 10_000)) == 436
        dn = ParamLayout([("p$(i)" => s) for (i, s) in enumerate(DN_GNN_SHAPES)])
        s = (@test_logs (:warn,) match_mode = :any HybridSubspace(dn; rank = 8,
                                                                  max_subspace_dim = 512))
        @test subspace_dim(s) == 1287
        s0 = (@test_logs (:warn,) HybridSubspace(mixed; rank = 4, max_subspace_dim = 0))
        @test subspace_dim(s0) == 140
        @test_throws ArgumentError HybridSubspace(dn; rank = 0)
    end

    @testset "reconstruct_batch and subspace_objective" begin
        layout = ParamLayout(["w" => (6, 4), "b" => (6,)])
        s = HybridSubspace(layout; rank = 2, seed = 1)
        base = randn(Xoshiro(2), s.total_params)
        Z = randn(Xoshiro(4), subspace_dim(s), 5)
        X = reconstruct_batch(s, base, Z)
        @test size(X) == (s.total_params, 5)
        for j in 1:5
            @test isapprox(X[:, j], base .+ expand(s, Z[:, j]))
        end
        g = subspace_objective(columnwise(x -> sum(abs2, x)), s, base)
        @test isapprox(g(Z), [sum(abs2, X[:, j]) for j in 1:5])
        Xv = reconstruct_batch(s, base, view(Z, :, 2:4))
        for j in 1:3
            @test isapprox(Xv[:, j], base .+ expand(s, Z[:, j + 1]))
        end
        s32 = HybridSubspace(layout; rank = 2, seed = 1, T = Float32)
        X32 = reconstruct_batch(s32, base, Z)
        @test eltype(X32) == Float32
        for j in 1:5
            @test isapprox(X32[:, j], base .+ expand(s32, Z[:, j]))
        end
        @test sprint(show, s) == "HybridSubspace(2 layers, dim=26/30, rank=2)"
    end

    @testset "quadratic optimized in the subspace" begin
        layout = ParamLayout(["w" => (6, 4), "b" => (6,)])
        s = HybridSubspace(layout; rank = 1, seed = 3)
        d = subspace_dim(s)
        @test d == 16
        base = zeros(s.total_params)
        target = expand(s, randn(Xoshiro(9), d) ./ sqrt(d))
        f = columnwise(x -> sum(abs2, x .- target))
        g = subspace_objective(f, s, base)
        f0 = f(reshape(base, :, 1))[1]
        cfg = PolyStepConfig(dim = d, polytope = :orthoplex, solver = SoftmaxSolver(),
                             epsilon = CosineEpsilon(init = 1.0, target = 0.01,
                                                     total_steps = 199),
                             scale_cost = :mean, step_radius = 1.0, probe_radius = 1.0,
                             num_probe = 2, max_iterations = 200, min_iterations = 200)
        st = init_state(cfg, zeros(d, 1))
        solve!(g, cfg, st; rng = Xoshiro(17))
        best = base .+ expand(s, st.best_x)
        @test isapprox(sum(abs2, best .- target), st.best_f; atol = 1e-8)
        @test st.best_f < 1e-3 * f0
        @test isapprox(project(s, best .- base), st.best_x; atol = 1e-10)
    end
end
