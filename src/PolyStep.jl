"""
    PolyStep

Gradient-free polytope direct search with optimal-transport weighting, for
piecewise-constant and combinatorial objectives. See [`minimize`](@ref),
[`PolyStepES`](@ref) and [`PolyStepConfig`](@ref).
"""
module PolyStep

using LinearAlgebra
using Random
using Statistics
using StaticArrays
using Polyester: @batch
using PrecompileTools: @setup_workload, @compile_workload
import SLEEFPirates

include("numeric.jl")
include("geometry.jl")
include("schedules.jl")
include("solvers.jl")
include("sinkhorn.jl")
include("dynamics.jl")
include("quadratic.jl")
include("core.jl")
include("es.jl")
include("subspace.jl")

export orthoplex_vertices, simplex_vertices, cube_vertices, polytope_vertices, num_vertices
export haar_rotations!, biased_rotation!, probe_scales
export splitmix64
export LinearEpsilon, CosineEpsilon, ProgressiveEpsilon, epsilon_at, update!
export SoftmaxSolver, TemperedSoftmaxSolver, KLSoftmaxSolver, SinkhornSolver,
       MinCostGreedySolver, TopKMeanSolver, OTResult
export PolyStepConfig, PolyStepState, init_state, step!, solve!, columnwise
export PolyStepES, ask!, tell!, minimize, popsize, PolyStepOptimizer
export ParamEntry, ParamLayout, LayerSpec, HybridSubspace
export subspace_dim, compression_ratio, expand, expand!, project, project!,
       reconstruct_batch, subspace_objective
# `solve` is not exported (CommonSolve/SciML name clash)

@setup_workload begin
    sphere(X) = vec(sum(abs2, X; dims = 1))
    @compile_workload begin
        rng = Xoshiro(0)
        for d in 3:9
            R = zeros(d, d, 2)
            haar_rotations!(R, R, rng)
            b = zeros(d, 2)
            b[1, :] .= 1
            biased_rotation!(R, b)
        end
        minimize(sphere, 3; steps = 2)
        ps = PolyStepConfig(dim = 3, max_iterations = 2)
        solve!(sphere, ps, init_state(ps, zeros(3, 1)); rng)
        solve!(sphere, ps, init_state(ps, zeros(3, 1)))   # default rng
    end
end

end
