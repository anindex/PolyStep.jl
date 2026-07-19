using Aqua
@testset "aqua" begin
    Aqua.test_all(PolyStep; ambiguities = false)
    Aqua.test_ambiguities(PolyStep)
end
