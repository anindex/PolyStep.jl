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
    # optional-dependency suites: run only when the package resolves in the
    # test env (the CI without-lv lane strips LoopVectorization)
    Base.find_package("LoopVectorization") === nothing || include("test_lv.jl")
    Base.find_package("OptimizationBase") === nothing || include("test_optimization.jl")
    include("test_zero_alloc.jl")
    include("test_aqua.jl")
    # CUDA is a weakdep, not in the test target (it would force a heavy install
    # on every run); GPU users add it to the active environment before opting in.
    if get(ENV, "JPOLYSTEP_TEST_CUDA", "0") == "1"
        if Base.find_package("CUDA") === nothing
            @warn "JPOLYSTEP_TEST_CUDA=1 but CUDA is not resolvable in this environment; add CUDA to run the GPU suite. Skipping test_cuda.jl."
        else
            include("test_cuda.jl")
        end
    end
end
