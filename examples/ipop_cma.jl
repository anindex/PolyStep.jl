# IPOP-CMA-ES (Auger and Hansen, CEC 2005). CMA-ES stops early on flat objectives,
# so restart from the incumbent with a doubled population; evals are counted so the
# budget is never exceeded.
import CMAEvolutionStrategy as CMA

function ipop_cma(f, x0::AbstractVector, s0::Real, budget::Integer;
        seed::Integer = 0, lower = nothing, upper = nothing, max_popsize::Integer = 256)
    pop = 4 + floor(Int, 3 * log(length(x0)))
    evals = Ref(0)
    fc = x -> (evals[] += 1; f(x))
    xb, fb = copy(x0), Inf
    restart = 0
    while budget - evals[] >= pop
        # maxfevals stops after the first generation past it, so leave one generation
        o = CMA.minimize(fc, xb, s0; popsize = pop, maxfevals = budget - evals[] - pop,
            lower = lower, upper = upper, seed = seed + restart, verbosity = 0)
        if CMA.fbest(o) < fb
            xb, fb = copy(CMA.xbest(o)), CMA.fbest(o)
        end
        pop = min(2pop, max_popsize)
        restart += 1
    end
    return xb, fb, evals[]
end
