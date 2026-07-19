# (s, S) inventory policy by simulation optimization, the classic OR testbed
# (Fu & Healy-style periodic review). Costs: fixed order K=100, unit c=1,
# holding h=1, backorder p=100; Poisson(20) demand; T=100 periods, R=100
# replications. Integer demand makes the cost surface piecewise-constant in
# the policy; Nelder-Mead style simplex methods sit on plateaus while
# PolyStep's finite-radius probes step across them.
#
# Common random numbers done properly: one demand tensor per stream, shared by
# every candidate and every method (paired comparisons); optimized on the
# train stream, reported on train and a held-out test stream.
#
# Run:  julia --project=examples examples/02_inventory_sS.jl
#       ECHELONS=5 julia --project=examples ...   # 5-echelon serial system (10-dim)

using PolyStep
using Random
using Statistics
using Printf
using Distributions
using CMAEvolutionStrategy: minimize as cma_minimize, xbest, fbest
using Optim

const K_ORDER = 100.0
const C_UNIT = 1.0
const H_HOLD = 1.0
const P_BACK = 100.0
const T_PERIODS = 100
const R_REPS = 100
const BUDGET = 20_000
const ECHELONS = parse(Int, get(ENV, "ECHELONS", "1"))

demand_tensor(seed) = rand(Xoshiro(seed), Poisson(20), T_PERIODS, R_REPS)

# Serial system: echelon 1 faces customer demand; echelon j > 1 faces the
# orders of echelon j-1; the top echelon draws from an infinite source.
# Zero lead time, order-up-to (s_j, S_j) at every echelon; backorder cost only
# at echelon 1, holding cost everywhere.
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

batched(D) = X -> [sim_cost(view(X, :, j), D) for j in 1:size(X, 2)]

function grid_optimum(D)                 # exact reference on the train stream (1 echelon)
    best = (Inf, 0, 0)
    for s in 0:60, S in s:120

        c = sim_cost([float(s), float(S)], D)
        c < best[1] && (best = (c, s, S))
    end
    return best
end

function main()
    dim = 2 * ECHELONS
    Dtrain = demand_tensor(42)
    Dtest = demand_tensor(43)
    ftrain = batched(Dtrain)
    x0 = fill(30.0, dim)
    box = (0.0, 150.0)
    rows = Vector{Tuple{String, Vector{Float64}, Float64, Float64}}()

    # PolyStep (budget = rounds * popsize)
    es = minimize(ftrain, dim; steps = BUDGET ÷ (2dim), epsilon = 0.05,
        step_radius = 40.0, x0 = x0, lb = box[1], ub = box[2],
        rng = Xoshiro(0))
    push!(rows, ("PolyStep", es.best_x, es.best_f, sim_cost(es.best_x, Dtest)))

    # CMA-ES under the same budget/box/objective (BlackBoxOptim.jl is broken on
    # Julia 1.12, every method dies on the removed Base.warn, so the
    # registered baselines are CMA-ES + Nelder-Mead, plus an inline SPSA)
    res = cma_minimize(x -> sim_cost(clamp.(x, box[1], box[2]), Dtrain), x0, 20.0;
        lower = fill(box[1], dim), upper = fill(box[2], dim),
        maxfevals = BUDGET, verbosity = 0, seed = UInt(1))
    xc = clamp.(xbest(res), box[1], box[2])
    push!(rows, ("CMA-ES", xc, fbest(res), sim_cost(xc, Dtest)))

    # inline SPSA (Spall 1992), 2 evals per iteration
    let θ = copy(x0), rng = Xoshiro(2), bestf = Inf, bestx = copy(x0)
        for k in 0:(BUDGET ÷ 2 - 1)
            ck = 2.0 / (k + 1)^0.101
            Δ = float.(rand(rng, (-1, 1), dim))
            fp = sim_cost(clamp.(θ .+ ck .* Δ, box[1], box[2]), Dtrain)
            fm = sim_cost(clamp.(θ .- ck .* Δ, box[1], box[2]), Dtrain)
            fmin = min(fp, fm)
            if fmin < bestf
                bestf = fmin
                bestx = clamp.(fp < fm ? θ .+ ck .* Δ : θ .- ck .* Δ, box[1], box[2])
            end
            ak = 2.0 / (k + 11)^0.602
            θ .-= ak .* (fp - fm) ./ (2ck .* Δ)
            clamp!(θ, box[1], box[2])
        end
        push!(rows, ("SPSA", bestx, bestf, sim_cost(bestx, Dtest)))
    end

    # Nelder-Mead (Optim.jl), same eval budget via f_calls_limit
    onm = Optim.optimize(x -> sim_cost(clamp.(x, box[1], box[2]), Dtrain), x0,
        NelderMead(), Optim.Options(f_calls_limit = BUDGET, iterations = 10^6))
    xnm = clamp.(Optim.minimizer(onm), box[1], box[2])
    push!(rows, ("Nelder-Mead", xnm, Optim.minimum(onm), sim_cost(xnm, Dtest)))

    println("(s,S) inventory simulation optimization, $(ECHELONS) echelon(s), dim=$dim")
    @printf("  budget=%d evals x %d replications, CRN train/test streams\n", BUDGET, R_REPS)
    if ECHELONS == 1
        gopt, gs, gS = grid_optimum(Dtrain)
        @printf("  integer-grid optimum (train): %.2f at (s=%d, S=%d)\n", gopt, gs, gS)
        @printf("  %-14s %-22s %12s %12s %9s\n", "method", "policy", "train", "test", "gap%")
        println("  " * "-"^72)
        for (name, x, tr, te) in rows
            pol = "(" * join([@sprintf("%.1f", v) for v in x], ", ") * ")"
            @printf("  %-14s %-22s %12.2f %12.2f %8.2f%%\n", name, pol, tr, te,
                100 * (tr - gopt) / gopt)
        end
        ps_gap = 100 * (rows[1][3] - gopt) / gopt
        @assert ps_gap < 2.0 "PolyStep train gap $(ps_gap)% above 2% of grid optimum"
        println("  Gate passed: PolyStep within 2% of the grid optimum")
    else
        @printf("  %-14s %12s %12s\n", "method", "train", "test")
        for (name, _, tr, te) in rows
            @printf("  %-14s %12.2f %12.2f\n", name, tr, te)
        end
    end
end

main()
