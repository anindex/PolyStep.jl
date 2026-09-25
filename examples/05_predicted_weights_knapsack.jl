# Decision-focused learning with the prediction in the constraints. A linear
# model predicts 0-1 knapsack item weights from features; the decision is greedy
# value/weight on the rounded predicted weights, scored under the true weights
# with hard feasibility (overweight = 0). Regret is against greedy on the true
# weights, not the optimum, so it can be slightly negative. Pathwise gradients
# through round() and the sort are zero a.e.; SPO+ and FY need the prediction in
# the objective, so they do not apply.
#
# Methods, BUDGET train-set evals each (PolyStep pays 2d evals per step):
#   PolyStep   minimize(), step_radius shrinking to 10% (fixed radius was worse)
#   SFGE       score-function gradient (arXiv:2307.05213), baseline, Adam, 4 samples
#   OpenAI-ES  antithetic parameter-space smoothing, Adam, 8 pairs/step
#   CMA-ES     IPOP-CMA-ES (ipop_cma.jl)
#   random     (1+1)-ES with the 1/5th success rule
#
# Tuning: 12-point grids, selected on median train regret (selected in brackets):
#   PolyStep   epsilon {0.1, [0.3], 1} x step_radius {0.5, 1, [2], 4}
#   SFGE       sigma {[0.35], 0.5, 0.75} x lr {0.01, 0.015, [0.02], 0.03}
#   OpenAI-ES  sigma {0.05, [0.1], 0.2} x lr {0.02, [0.035], 0.05, 0.075}
#   CMA-ES     s0 {0.12, 0.14, ..., [0.2], ..., 0.3, 0.35, 0.4}
#   random     sigma0 {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, [5]}
#
# Result: PolyStep reaches the lowest train regret, but with 30 instances for 30
# parameters the train/test gap dominates and the test ranking is seed noise.
#
# Run:  julia --project=examples examples/05_predicted_weights_knapsack.jl

using PolyStep
using Random
using Statistics
using Printf
import LinearAlgebra: dot
include(joinpath(@__DIR__, "ipop_cma.jl"))

# ---------------------------------------------------------------------------
# Problem: contextual 0-1 knapsack, weights predicted from features
# ---------------------------------------------------------------------------
const N_FEAT = 30           # model dimension d = N_FEAT (linear weight model)
const N_ITEMS = 15
const N_TRAIN = 30
const N_TEST = 60
const CAP_FRAC = 0.15       # tight capacity: few items fit
const BUDGET = 12_000       # train-set evaluations per method
const SEEDS = 1:10

softplus(x) = x > 20 ? x : log1p(exp(x))

struct Instance
    Phi::Matrix{Float64}    # (N_FEAT, N_ITEMS) item features
    w_true::Vector{Float64} # true weights
    v::Vector{Float64}      # item values
    cap::Float64
    ref_value::Float64      # greedy value under the true weights (not the optimum)
end

# greedy 0-1 knapsack by value/weight ratio
function greedy_value(v, w, cap)
    order = sortperm(v ./ w; rev = true)
    total_w = 0.0
    total_v = 0.0
    @inbounds for i in order
        if total_w + w[i] <= cap
            total_w += w[i]
            total_v += v[i]
        end
    end
    return total_v
end

# greedy on predicted weights, scored on true weights; infeasible scores 0 (no repair)
function decision_value(v, w_pred, w_true, cap)
    order = sortperm(v ./ w_pred; rev = true)
    pred_w = 0.0
    true_w = 0.0
    total_v = 0.0
    @inbounds for i in order
        if pred_w + w_pred[i] <= cap
            pred_w += w_pred[i]
            true_w += w_true[i]
            total_v += v[i]
        end
    end
    return true_w <= cap ? total_v : 0.0
end

function make_instances(rng, n, theta_star)
    [begin
         Phi = randn(rng, N_FEAT, N_ITEMS)
         # integer weights (e.g. pallets or slots)
         w = [max(1.0, round(4 * softplus(2 * dot(view(Phi, :, i), theta_star) /
                                          sqrt(N_FEAT) + 0.05 * randn(rng))))
              for i in 1:N_ITEMS]
         v = exp.(0.8 .* randn(rng, N_ITEMS))   # dispersed values: ratio order matters
         cap = max(1.0, round(CAP_FRAC * sum(w)))
         Instance(Phi, w, v, cap, greedy_value(v, w, cap))
     end
     for _ in 1:n]
end

# continuous model output before rounding
function predict_latent(inst, theta)
    [4 * softplus(2 * dot(view(inst.Phi, :, i), theta) / sqrt(N_FEAT))
     for i in 1:N_ITEMS]
end
# the decision uses integer weights; round() has zero derivative a.e.
predict_weights(inst, theta) = max.(1.0, round.(predict_latent(inst, theta)))

function regret(theta, data)
    mean(data) do inst
        w_hat = predict_weights(inst, theta)
        inst.ref_value - decision_value(inst.v, w_hat, inst.w_true, inst.cap)
    end
end

# ---------------------------------------------------------------------------
# Methods (each uses at most BUDGET train-set evaluations, returns best theta)
# ---------------------------------------------------------------------------
# the radius shrinks geometrically to 10% of step_radius over the run
function run_polystep(rng, train; epsilon = 0.3, step_radius = 2.0)
    f(TH) = [regret(view(TH, :, j), train) for j in 1:size(TH, 2)]
    steps = div(BUDGET - 1, 2 * N_FEAT)   # -1: minimize scores the final iterate
    k = 0.1^(1 / steps)
    es = minimize(f, N_FEAT; steps, epsilon, step_radius, rng,
        callback = es -> (es.step_radius *= k; false))
    return es.best_x
end

# Adam for the gradient-estimate baselines
mutable struct Adam
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end
Adam(d) = Adam(zeros(d), zeros(d), 0)
function step_adam!(theta, g, a::Adam; lr)
    a.t += 1
    @. a.m = 0.9 * a.m + 0.1 * g
    @. a.v = 0.999 * a.v + 0.001 * g^2
    mhat = a.m ./ (1 - 0.9^a.t)
    vhat = a.v ./ (1 - 0.999^a.t)
    @. theta -= lr * mhat / (sqrt(vhat) + 1e-8)
    return theta
end

# SFGE: REINFORCE on perturbed predicted weights, baseline = current train regret
function run_sfge(rng, train; sigma = 0.35, lr = 0.02, nsamples = 4)
    theta = zeros(N_FEAT)
    adam = Adam(N_FEAT)
    best = copy(theta)
    best_r = regret(theta, train)
    b = best_r
    evals = 1
    while evals + nsamples + 1 <= BUDGET
        g = zeros(N_FEAT)
        for _ in 1:nsamples, inst in train
            mu = predict_latent(inst, theta)
            eps_w = sigma .* randn(rng, N_ITEMS)
            w_pert = max.(1.0, round.(mu .+ eps_w))
            r = inst.ref_value - decision_value(inst.v, w_pert, inst.w_true, inst.cap)
            # d E[r] / d theta = E[(r - b) * score * d mu / d theta]
            for i in 1:N_ITEMS
                z = 2 * dot(view(inst.Phi, :, i), theta) / sqrt(N_FEAT)
                dsp = 1 / (1 + exp(-z))          # softplus'
                coef = (r - b) * eps_w[i] / sigma^2 * dsp * 8 / sqrt(N_FEAT)
                @. g += coef * @view inst.Phi[:, i]
            end
        end
        g ./= (nsamples * length(train))
        step_adam!(theta, g, adam; lr)
        b = regret(theta, train)
        evals += nsamples + 1
        b < best_r && (best_r = b; best = copy(theta))
    end
    return best
end

# OpenAI-ES: antithetic parameter-space Gaussian smoothing + Adam
function run_openai_es(rng, train; sigma = 0.1, lr = 0.035, npairs = 8)
    theta = zeros(N_FEAT)
    adam = Adam(N_FEAT)
    best = copy(theta)
    best_r = regret(theta, train)
    evals = 1
    while evals + 2npairs + 1 <= BUDGET
        g = zeros(N_FEAT)
        for _ in 1:npairs
            u = randn(rng, N_FEAT)
            rp = regret(theta .+ sigma .* u, train)
            rm = regret(theta .- sigma .* u, train)
            @. g += (rp - rm) / (2 * sigma) * u
        end
        g ./= npairs
        step_adam!(theta, g, adam; lr)
        r = regret(theta, train)
        evals += 2npairs + 1
        r < best_r && (best_r = r; best = copy(theta))
    end
    return best
end

function run_cmaes(rng, train; s0 = 0.2)
    xb, _, _ = ipop_cma(theta -> regret(theta, train), zeros(N_FEAT), s0, BUDGET;
        seed = rand(rng, 1:10^9))
    return xb
end

# (1+1)-ES, 1/5th success rule: x1.5 on success, x1.5^(-1/4) on failure.
# Ties count as success so it can drift across flat regions.
function run_random_search(rng, train; sigma0 = 5.0)
    theta = zeros(N_FEAT)
    r_theta = regret(theta, train)
    sigma = sigma0
    for _ in 2:BUDGET
        cand = theta .+ sigma .* randn(rng, N_FEAT)
        r = regret(cand, train)
        if r <= r_theta
            theta, r_theta = cand, r
            sigma *= 1.5
        else
            sigma *= 1.5^(-1 / 4)
        end
    end
    return theta
end

const METHODS = ("PolyStep" => run_polystep, "SFGE" => run_sfge,
    "OpenAI-ES" => run_openai_es, "CMA-ES" => run_cmaes, "random" => run_random_search)

function make_data(seed)
    rng = Xoshiro(splitmix64(1234, seed))
    theta_star = randn(rng, N_FEAT)
    train = make_instances(rng, N_TRAIN, theta_star)
    test = make_instances(rng, N_TEST, theta_star)
    return train, test
end

# ---------------------------------------------------------------------------
function main()
    tr = Dict(name => Float64[] for (name, _) in METHODS)
    te = Dict(name => Float64[] for (name, _) in METHODS)
    for seed in SEEDS
        train, test = make_data(seed)
        for (name, run) in METHODS
            theta = run(Xoshiro(splitmix64(seed, 7)), train)
            push!(tr[name], regret(theta, train))
            push!(te[name], regret(theta, test))
        end
    end

    n = length(SEEDS)
    println("Regret vs greedy on true weights, $n seeds, " *
            "budget $BUDGET train-set evals per method")
    @printf("  %-10s %11s %11s %16s %18s\n", "method", "train med", "test med",
        "test IQR", "PolyStep better")
    for (name, _) in METHODS
        q1, q3 = quantile(te[name], (0.25, 0.75))
        wins = name == "PolyStep" ? "" : "$(count(te["PolyStep"] .< te[name]))/$n seeds"
        @printf("  %-10s %11.3f %11.3f   [%5.2f, %5.2f] %18s\n", name,
            median(tr[name]), median(te[name]), q1, q3, wins)
    end

    # no test claim against the tuned baselines: that ranking is within seed noise
    tr_med = Dict(name => median(v) for (name, v) in tr)
    te_med = Dict(name => median(v) for (name, v) in te)
    @assert all(tr_med["PolyStep"] <= tr_med[name] for (name, _) in METHODS) "PolyStep should reach the lowest train regret"
    @assert te_med["PolyStep"] < te_med["random"] "PolyStep should beat random search on test"
    println("\nGate passed: lowest median train regret; beats random search on test.")
    return tr_med, te_med
end

main()
