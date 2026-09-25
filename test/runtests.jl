using Test
using PolyStep
using LinearAlgebra
using Random
using Statistics

@testset "PolyStep" begin
    include("test_numeric.jl")
    include("test_geometry.jl")
    include("test_solvers.jl")
    include("test_core.jl")
    include("test_ask_tell.jl")
    include("test_edge_cases.jl")
    include("test_subspace.jl")
    isempty(VERSION.prerelease) && include("test_threads.jl")
    # optional suites run only when the package resolves (the CI without-lv lane strips LV)
    Base.find_package("LoopVectorization") === nothing || include("test_lv.jl")
    Base.find_package("OptimizationBase") === nothing || include("test_optimization.jl")
    include("test_zero_alloc.jl")
    include("test_aqua.jl")
    # CUDA is not in the test target; add it to the environment to opt in
    if get(ENV, "JPOLYSTEP_TEST_CUDA", "0") == "1"
        if Base.find_package("CUDA") === nothing
            @warn "JPOLYSTEP_TEST_CUDA=1 but CUDA is not resolvable in this environment; add CUDA to run the GPU suite. Skipping test_cuda.jl."
        else
            include("test_cuda.jl")
        end
    end
end
