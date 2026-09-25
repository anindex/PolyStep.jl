# (s, S) inventory policy by simulation optimization (Fu and Healy style
# periodic review): K=100, c=1, h=1, p=100, Poisson(20) demand, zero lead time,
# T=100 periods, R=100 replications. Common random numbers: optimized on a train
# demand stream, reported on train and a held-out test stream. Integer demand
# makes the cost piecewise constant in s, so the best integer policy on the
# train stream is an exact reference and gaps are >= 0.
#
# Compared: PolyStep, IPOP-CMA-ES, SPSA, Nelder-Mead (Optim.jl), all from
# x0 = (30, 30) in [0, 150]^2, seeds 1:5. Budgets: 500 and 20000 evaluations.
#
# Tuning: 9 configs per method and budget, best mean train cost on seeds 101:105
# (picks at 500 | 20000):
#   PolyStep     radius {5,20,60} x epsilon {0.05,0.2,0.5}  -> 20, 0.05 | 5, 0.05
#   IPOP-CMA-ES  sigma0 {1,2,5,10,15,20,30,45,60}           -> 30 | 15
#   SPSA         a {100,300,1000} x c {2,5,10}               -> 100, 5 | 1000, 2
#   Nelder-Mead  step {1,2,5,10,15,20,30,45,60}              -> 60 | 60
# First steps of 5 or less stall near x0, where the policy orders every period.
#
# Result: PolyStep, IPOP-CMA-ES and Nelder-Mead tie at the grid optimum at both
# budgets; SPSA ties at 20000 but trails at 500.
#
# Run:  julia --project=examples examples/02_inventory_sS.jl
#       ECHELONS=5 julia --project=examples ...   # 10-dim serial system, untuned, no reference

using PolyStep
using Random
using Statistics
using Printf
using Distributions
using Optim
include("ipop_cma.jl")

const K_ORDER = 100.0
const C_UNIT = 1.0
const H_HOLD = 1.0
const P_BACK = 100.0
const T_PERIODS = 100
const R_REPS = 100
const SEEDS = 1:5
const BOX = (0.0, 150.0)
const ECHELONS = parse(Int, get(ENV, "ECHELONS", "1"))

# selected configs per budget (see header)
const CONFIGS = Dict(
    500 => (ps = (radius = 20.0, epsilon = 0.05), cma = 30.0,
        spsa = (a = 100.0, c = 5.0), nm = 60.0),
    20_000 => (ps = (radius = 5.0, epsilon = 0.05), cma = 15.0,
        spsa = (a = 1000.0, c = 2.0), nm = 60.0),
)

demand_tensor(seed) = rand(Xoshiro(seed), Poisson(20), T_PERIODS, R_REPS)

# Serial system: echelon j > 1 faces the orders of echelon j-1, the top one
# draws from an infinite source. Backorder cost only at echelon 1.
function sim_cost(x::AbstractVector, D::Matrix{Int})
    n = ECHELONS
    s = ntuple(j -> x[2j - 1], n)
    S = ntuple(j -> max(x[2j], x[2j - 1]), n)      # feasibility clamp: S >= s
    total = 0.0
    @inbounds for r in 1:R_REPS
        inv = collect(float.(S))
        cost = 0.0
        for t in 1:T_PERIODS
            dem = float(D[t, r])                 # echelon-1 demand
            for j in 1:n
                if inv[j] <= s[j]
                    q = S[j] - inv[j]
                    cost += K_ORDER + C_UNIT * q
                    inv[j] = float(S[j])
                    dem_up = q                   # becomes upstream demand
                else
                    dem_up = 0.0
                end
                inv[j] -= dem                    # backorders carry as negative inventory
                if j == 1
                    cost += H_HOLD * max(inv[j], 0.0) + P_BACK * max(-inv[j], 0.0)
                else
                    cost += H_HOLD * max(inv[j], 0.0)
                end
                dem = dem_up
            end
        end
        total += cost / T_PERIODS
    end
    return total / R_REPS
end

boxed(D) = x -> sim_cost(clamp.(x, BOX...), D)

function grid_optimum(D)                 # exact reference on the train stream (1 echelon)
    best = (Inf, 0, 0)
    for s in 0:60, S in s:120
        c = sim_cost([float(s), float(S)], D)
        c < best[1] && (best = (c, s, S))
    end
    return best
end

# Each runner returns (policy, train cost, evals used).

function run_polystep(D, x0, seed, budget; radius, epsilon)
    dim = length(x0)
    f = X -> [sim_cost(view(X, :, j), D) for j in 1:size(X, 2)]
    steps = div(budget - 1, 2dim)       # 2*dim evals per step, one to score the final iterate
    shrink = 1e-3^(1 / steps)           # radius shrinks geometrically to radius/1000
    es = minimize(f, dim; steps = steps, epsilon = epsilon, step_radius = radius,
        x0 = x0, lb = BOX[1], ub = BOX[2], rng = Xoshiro(seed),
        callback = es -> (es.step_radius *= shrink; false))
    return es.best_x, es.best_f, es.evals
end

function run_cma(D, x0, seed, budget; s0)
    dim = length(x0)
    x, fx, ev = ipop_cma(boxed(D), x0, s0, budget; seed = seed,
        lower = fill(BOX[1], dim), upper = fill(BOX[2], dim))
    return clamp.(x, BOX...), fx, ev
end

# SPSA with Spall's gains; returns the best evaluated point
function run_spsa(D, x0, seed, budget; a, c)
    f = boxed(D)
    dim = length(x0)
    rng = Xoshiro(seed)
    iters = div(budget - 1, 2)
    A = 0.1 * iters
    theta = copy(x0)
    bestx, bestf = copy(x0), Inf
    for k in 0:(iters - 1)
        ck = c / (k + 1)^0.101
        delta = float.(rand(rng, (-1, 1), dim))
        xp = clamp.(theta .+ ck .* delta, BOX...)
        xm = clamp.(theta .- ck .* delta, BOX...)
        fp, fm = f(xp), f(xm)
        fp < bestf && ((bestf, bestx) = (fp, xp))
        fm < bestf && ((bestf, bestx) = (fm, xm))
        theta .-= a / (k + 1 + A)^0.602 .* (fp - fm) ./ (2ck .* delta)
        clamp!(theta, BOX...)
    end
    ftheta = f(theta)
    ftheta < bestf && ((bestf, bestx) = (ftheta, copy(theta)))
    return bestx, bestf, 2iters + 1
end

# initial simplex x0 + step * e_j, stops at its own tolerance
function run_nm(D, x0, seed, budget; step)
    nm = NelderMead(initial_simplex = Optim.AffineSimplexer(step, 0.0))
    o = Optim.optimize(boxed(D), x0, nm, Optim.Options(f_calls_limit = budget, iterations = 10^6))
    return clamp.(Optim.minimizer(o), BOX...), Optim.minimum(o), Optim.f_calls(o)
end

function main()
    dim = 2 * ECHELONS
    Dtrain = demand_tensor(42)
    Dtest = demand_tensor(43)
    x0 = fill(30.0, dim)
    names = ("PolyStep", "IPOP-CMA-ES", "SPSA", "Nelder-Mead")
    gopt, gs, gS = ECHELONS == 1 ? grid_optimum(Dtrain) : (NaN, 0, 0)

    println("(s,S) inventory simulation optimization, $(ECHELONS) echelon(s), dim=$dim")
    @printf("  %d replications per eval, CRN train/test streams, seeds %s\n", R_REPS, SEEDS)
    ECHELONS == 1 && @printf("  integer-grid optimum: train %.4f at (s=%d, S=%d), test %.4f\n",
        gopt, gs, gS, sim_cost([float(gs), float(gS)], Dtest))
    @printf("  %-12s %7s %6s %10s %10s %10s %10s\n", "method", "budget", "evals", "train",
        "test", "gap% mean", "gap% max")
    println("  " * "-"^71)
    psgap = 0.0
    for budget in sort(collect(keys(CONFIGS)))
        cfg = CONFIGS[budget]
        runs = (
            s -> run_polystep(Dtrain, x0, s, budget; cfg.ps...),
            s -> run_cma(Dtrain, x0, s, budget; s0 = cfg.cma),
            s -> run_spsa(Dtrain, x0, s, budget; cfg.spsa...),
            s -> run_nm(Dtrain, x0, s, budget; step = cfg.nm),     # deterministic
        )
        for (name, run) in zip(names, runs)
            rs = [run(s) for s in SEEDS]
            train = [r[2] for r in rs]
            test = [sim_cost(r[1], Dtest) for r in rs]
            g = max.(100 .* (train .- gopt) ./ gopt, 0.0)          # clip float roundoff
            name == "PolyStep" && (psgap = max(psgap, maximum(g)))
            @printf("  %-12s %7d %6d %10.4f %10.4f %10.4f %10.4f\n", name, budget,
                maximum(r[3] for r in rs), mean(train), mean(test), mean(g), maximum(g))
        end
    end
    if ECHELONS == 1
        @assert psgap < 0.01 "PolyStep train gap $(psgap)% above 0.01%"
        println("  Gate passed: PolyStep within 0.01% of the grid optimum on every run")
    end
end

main()
