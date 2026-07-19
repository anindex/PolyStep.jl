module PolyStep

using LinearAlgebra
using Random
using Statistics
using StaticArrays
using Polyester: @batch
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

export orthoplex_vertices, simplex_vertices, cube_vertices, polytope_vertices, num_vertices
export haar_rotations!, biased_rotation!, probe_scales
export splitmix64
export LinearEpsilon, CosineEpsilon, ProgressiveEpsilon, epsilon_at, update!
export SoftmaxSolver, TemperedSoftmaxSolver, KLSoftmaxSolver, SinkhornSolver,
       MinCostGreedySolver, TopKMeanSolver, OTResult
export PolyStepConfig, PolyStepState, init_state, step!, solve!, columnwise
export PolyStepES, ask!, tell!, minimize, popsize, PolyStepOptimizer
# `solve` is not exported (avoids the CommonSolve/SciML name clash). Call
# PolyStep.solve or use step!/ask!/tell!.

end
