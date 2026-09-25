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
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)
    rounds = maxiters !== nothing ? maxiters : maxtime !== nothing ? typemax(Int) : 1000
    batched = get(cache.solver_args, :batched, nothing)
    rng = get(cache.solver_args, :rng, Random.Xoshiro(0))
    if batched === nothing
        @warn "PolyStepOptimizer is evaluating Optimization.jl's scalar objective one " *
              "candidate at a time, forfeiting PolyStep's batched evaluation. Pass " *
              "`batched = f` (f(X::AbstractMatrix)::Vector) through solve kwargs, or " *
              "use PolyStep.minimize directly." maxlog = 1
    end
    (cache.solver_args.abstol === nothing && cache.solver_args.reltol === nothing) ||
        @warn "abstol/reltol are not used by PolyStepOptimizer (budget-only)" maxlog = 1

    shaped(x) = copyto!(similar(u0, T), x)
    flat(b) = b isa AbstractArray ? vec(b) : b
    sgn = cache.sense === SciMLBase.MaxSense ? -1.0 : 1.0
    fitness(X) = batched === nothing ?
                 [Float64(first(cache.f(shaped(view(X, :, j)), cache.p))) for j in 1:size(X, 2)] :
                 sgn .* Vector{Float64}(batched(X))

    es = PolyStepES(dim; num_particles = opt.num_particles, epsilon = opt.epsilon,
        step_radius = opt.step_radius, solver = opt.solver,
        scale_cost = opt.scale_cost, x0 = Vector{T}(vec(u0)),
        lb = flat(cache.lb), ub = flat(cache.ub), rng = rng, T = T)

    t0 = time()
    done = 0
    halted = false
    timedout = false
    for round in 1:rounds
        tell!(es, fitness(ask!(es)))
        done = round
        state = OptimizationBase.OptimizationState(; iter = round, u = shaped(es.best_x),
            p = cache.p, objective = es.best_f,
            original = es)
        halt = cache.callback(state, es.best_f)
        halt isa Bool ||
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        if halt
            halted = true
            break
        end
        if maxtime !== nothing && time() - t0 >= maxtime
            timedout = true
            break
        end
    end
    PolyStep._score_iterate!(es, fitness)
    t1 = time()

    stats = OptimizationBase.OptimizationStats(; iterations = done, time = t1 - t0,
        fevals = es.evals)
    found = isfinite(es.best_f)
    retcode = !found ? SciMLBase.ReturnCode.Failure :
              halted ? SciMLBase.ReturnCode.Terminated :
              timedout ? SciMLBase.ReturnCode.MaxTime : SciMLBase.ReturnCode.Success
    return SciMLBase.build_solution(cache, opt, found ? shaped(es.best_x) : copy(u0),
        es.best_f; original = es, retcode = retcode, stats = stats)
end

end
