# Optimization.jl (SciML) adapter: current cache-based API (OptimizationBase
# v5 `__solve(cache::OptimizationCache)`), mirroring the structure of
# OptimizationCMAEvolutionStrategy.
module PolyStepOptimizationExt

using PolyStep
using Random
using OptimizationBase
using OptimizationBase: SciMLBase

SciMLBase.allowscallback(::PolyStepOptimizer) = true
SciMLBase.allowsbounds(::PolyStepOptimizer) = true
SciMLBase.has_init(::PolyStepOptimizer) = true
SciMLBase.requiresgradient(::PolyStepOptimizer) = false
SciMLBase.requireshessian(::PolyStepOptimizer) = false
SciMLBase.requiresconsjac(::PolyStepOptimizer) = false
SciMLBase.requiresconshess(::PolyStepOptimizer) = false

function SciMLBase.__solve(cache::OptimizationBase.OptimizationCache{O}) where {O <:
                                                                                PolyStepOptimizer}
    opt = cache.opt
    u0 = cache.u0
    T = float(eltype(u0))
    dim = length(u0)
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    rounds = maxiters === nothing ? 1000 : maxiters
    batched = get(cache.solver_args, :batched, nothing)
    rng = get(cache.solver_args, :rng, Random.Xoshiro(0))
    if batched === nothing
        @warn "PolyStepOptimizer is evaluating Optimization.jl's scalar objective one " *
              "candidate at a time, forfeiting PolyStep's batched evaluation. Pass " *
              "`batched = f` (f(X::AbstractMatrix)::Vector) through solve kwargs, or " *
              "use PolyStep.minimize directly." maxlog = 1
    end

    es = PolyStepES(dim; num_particles = opt.num_particles, epsilon = opt.epsilon,
        step_radius = opt.step_radius, solver = opt.solver,
        scale_cost = opt.scale_cost, x0 = Vector{T}(u0),
        lb = cache.lb, ub = cache.ub, rng = rng, T = T)

    t0 = time()
    done = 0
    halted = false
    for round in 1:rounds
        X = ask!(es)
        fitness = if batched === nothing
            # candidates handed to the user objective as plain Vector{eltype(u0)}
            [Float64(first(cache.f(X[:, j], cache.p))) for j in 1:size(X, 2)]
        else
            Vector{Float64}(batched(X))
        end
        tell!(es, fitness)
        done = round
        state = OptimizationBase.OptimizationState(; iter = round, u = es.best_x,
            p = cache.p, objective = es.best_f,
            original = es)
        halt = cache.callback(state, es.best_f)
        halt isa Bool ||
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        if halt
            halted = true
            break
        end
    end
    t1 = time()

    stats = OptimizationBase.OptimizationStats(; iterations = done, time = t1 - t0,
        fevals = es.evals)
    # budget-only optimizer: exhausting the budget is the normal, successful
    # exit; a callback halt reports Terminated so callers can distinguish it
    retcode = halted ? SciMLBase.ReturnCode.Terminated : SciMLBase.ReturnCode.Success
    return SciMLBase.build_solution(cache, opt, es.best_x, es.best_f;
        original = es, retcode = retcode,
        stats = stats)
end

end
