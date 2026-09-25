using LoopVectorization
using PolyStep: _TURBO_ACTIVE, _lse_cols_base!, _lse_rows_base!, lse_cols!, lse_rows!

@testset "loopvectorization extension" begin
    @test _TURBO_ACTIVE[]
    for (V, P) in ((3, 7), (8, 33), (16, 512), (1, 5), (12, 1))
        rng = Xoshiro(1000 + V + P)
        A = randn(rng, V, P)
        addv = randn(rng, V)
        addp = randn(rng, P)
        o1 = zeros(P)
        o2 = zeros(P)
        _lse_cols_base!(o1, A, addv)
        lse_cols!(o2, A, addv)                   # dispatches to turbo
        @test isapprox(o1, o2; rtol = 1e-12)
        r1 = zeros(V)
        r2 = zeros(V)
        _lse_rows_base!(r1, A, addp, zeros(V), zeros(V))
        lse_rows!(r2, A, addp, zeros(V), zeros(V))
        @test isapprox(r1, r2; rtol = 1e-12)
    end
    B = [-Inf 0.0 -Inf; 0.0 -Inf -Inf]
    @test lse_rows!(zeros(2), B, zeros(3), zeros(2), zeros(2)) == [0.0, 0.0]
    @test lse_cols!(zeros(3), B, zeros(2)) == [0.0, 0.0, -Inf]
    A32 = randn(Xoshiro(7), Float32, 8, 21)
    o1 = zeros(Float32, 21)
    o2 = zeros(Float32, 21)
    _lse_cols_base!(o1, A32, randn(Xoshiro(8), Float32, 8))
    lse_cols!(o2, A32, randn(Xoshiro(8), Float32, 8))
    @test isapprox(o1, o2; rtol = 1e-6)
end
