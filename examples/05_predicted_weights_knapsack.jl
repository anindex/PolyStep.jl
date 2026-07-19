# Decision-focused learning with predicted parameters in the constraints:
# the case the differentiable-DFL toolbox cannot express. Predict item weights
# of a 0-1 knapsack from features; the decision is produced by a hard greedy
# maximizer under the predicted weights and evaluated under the true weights
# (with feasibility repair). The training loss "true test-value regret" is
# piecewise-constant in the model parameters θ:
#
#   * Fenchel-Young / SPO+ / perturbed-layer losses are undefined here: they
#     require the predicted parameters in the objective, not the constraints
#     (see Mandi et al., JAIR 2024 survey; zero-gradient failure of smoothed
#     layers documented in arXiv:2508.11365).
#   * The published escape is score-function REINFORCE (SFGE, arXiv:2307.05213),
#     exactly the high-variance gradient-estimate class that stalls on flat
#     piecewise-constant regions.
#   * PolyStep optimizes the hard pipeline directly: no smoothing, no
#     surrogate, no variance tuning.
#
# What this script reports: the direct-search methods (PolyStep, CMA-ES, and
# tuned parameter-space OpenAI-ES used as a plain optimizer) all beat the
# published DFL escape (SFGE) on this task, where the differentiable-DFL layers
# cannot express the constraint-side prediction. The assertion gate compares
# PolyStep against the published baseline class (SFGE) and random search; the
# ES/CMA-ES numbers are reported unfiltered for reference.
#
# Baselines (equal total candidate-evaluation budgets,
# PolyStep is charged its full 2d evaluations per step):
#   SFGE       score-function gradient on the prediction distribution + Adam
#   OpenAI-ES  parameter-space Gaussian smoothing gradient estimate + Adam
#   CMA-ES     CMAEvolutionStrategy.jl (a strong gradient-free baseline)
#   random     (1+1) random search with 1/5th-rule step adaptation
#
# Run:  julia --project=examples examples/05_predicted_weights_knapsack.jl

using PolyStep
using Random
using Statistics
using Printf
import CMAEvolutionStrategy

# ---------------------------------------------------------------------------
# Problem: contextual 0-1 knapsack, weights predicted from features
# ---------------------------------------------------------------------------
const N_FEAT = 30           # model dimension d = N_FEAT (linear weight model)
const N_ITEMS = 15
const N_TRAIN = 30
const N_TEST = 60
const CAP_FRAC = 0.15  # tight capacity: few items fit, decision sets are stable -> large flat regret plateaus
const BUDGET = 12_000       # candidate evaluations per method (1 eval = full train pass)
const SEEDS = 1:5

softplus(x) = x > 20 ? x : log1p(exp(x))

struct Instance
    Phi::Matrix{Float64}    # (N_FEAT, N_ITEMS) item features
    w_true::Vector{Float64} # true weights
    v::Vector{Float64}      # item values
    cap::Float64
    opt_value::Float64      # greedy value under true weights (reference decision)
end

# greedy 0-1 knapsack by value/weight ratio: deterministic hard maximizer
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

# decision under predicted weights, evaluated under true weights with hard
# feasibility (no recourse): the greedy solution built from predicted weights
# scores its value only if it satisfies the true capacity; an overweight
# solution is worthless. This is the standard constraint-side DFL setting;
# the regret surface is wide flat plateaus (same chosen set, feasible) broken
# by cliffs (set change or feasibility flip): smoothed/score-function
# gradients are exactly zero on the plateaus.
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
    return true_w <= cap ? total_v : 0.0    # hard feasibility, no repair
end

function make_instances(rng, n, theta_star)
    [begin
         Phi = randn(rng, N_FEAT, N_ITEMS)
         # integer resource units (pallets/slots): the standard knapsack regime
         w = [max(1.0, round(4 * softplus(2 * dot(view(Phi, :, i), theta_star) /
                                          sqrt(N_FEAT) + 0.05 * randn(rng))))
              for i in 1:N_ITEMS]
         v = exp.(0.8 .* randn(rng, N_ITEMS))   # dispersed values: ratio order is decisive
         cap = max(1.0, round(CAP_FRAC * sum(w)))
         Instance(Phi, w, v, cap, greedy_value(v, w, cap))
     end
     for _ in 1:n]
end

# continuous latent prediction (what a trained model outputs pre-integerization)
function predict_latent(inst, theta)
    [4 * softplus(2 * dot(view(inst.Phi, :, i), theta) / sqrt(N_FEAT))
     for i in 1:N_ITEMS]
end
# the decision consumes integer weights: round has zero derivative a.e., so
# any parameter-space smoothing gradient is exactly zero on wide plateaus
predict_weights(inst, theta) = max.(1.0, round.(predict_latent(inst, theta)))

# mean true-value regret of parameter vector theta over a dataset
function regret(theta, data)
    mean(data) do inst
        w_hat = predict_weights(inst, theta)
        inst.opt_value - decision_value(inst.v, w_hat, inst.w_true, inst.cap)
    end
end

import LinearAlgebra: dot

# ---------------------------------------------------------------------------
# Methods (each consumes exactly BUDGET candidate evaluations)
# ---------------------------------------------------------------------------
function run_polystep(rng, train)
    f_batched(TH) = [regret(view(TH, :, j), train) for j in 1:size(TH, 2)]
    es = PolyStepES(N_FEAT; epsilon = 0.1, step_radius = 1.5, rng = rng)
    steps = BUDGET ÷ popsize(es)
    for _ in 1:steps
        tell!(es, f_batched(ask!(es)))
    end
    return es.best_x
end

# hand-rolled Adam (shared by the gradient-estimate baselines)
mutable struct Adam
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end
Adam(d) = Adam(zeros(d), zeros(d), 0)
function step_adam!(theta, g, a::Adam; lr = 0.05)
    a.t += 1
    @. a.m = 0.9 * a.m + 0.1 * g
    @. a.v = 0.999 * a.v + 0.001 * g^2
    mhat = a.m ./ (1 - 0.9^a.t)
    vhat = a.v ./ (1 - 0.999^a.t)
    @. theta -= lr * mhat / (sqrt(vhat) + 1e-8)
    return theta
end

# SFGE-style score-function gradient: perturb the predictions, REINFORCE the
# regret back through the Gaussian score, chain through the linear model
function run_sfge(rng, train; sigma = 0.5, nsamples = 4)
    theta = zeros(N_FEAT)
    adam = Adam(N_FEAT)
    best = copy(theta)
    best_r = regret(theta, train)
    evals = 1
    while evals + nsamples <= BUDGET
        g = zeros(N_FEAT)
        base_r = 0.0
        for _ in 1:nsamples
            r_acc = 0.0
            for inst in train
                mu = predict_latent(inst, theta)
                eps_w = sigma .* randn(rng, N_ITEMS)
                w_pert = max.(1.0, round.(mu .+ eps_w))
                r = inst.opt_value -
                    decision_value(inst.v, w_pert, inst.w_true, inst.cap)
                r_acc += r
                # d regret / d theta ≈ r * score * d mu / d theta
                for i in 1:N_ITEMS
                    z = 2 * dot(view(inst.Phi, :, i), theta) / sqrt(N_FEAT)
                    dsp = 1 / (1 + exp(-z))          # softplus'
                    coef = r * eps_w[i] / sigma^2 * dsp * 8 / sqrt(N_FEAT)
                    @. g += coef * @view inst.Phi[:, i]
                end
            end
            base_r += r_acc / length(train)
        end
        g ./= (nsamples * length(train))
        step_adam!(theta, g, adam)
        evals += nsamples
        r = regret(theta, train)
        evals += 1
        r < best_r && (best_r = r; best = copy(theta))
    end
    return best
end

# OpenAI-ES: antithetic parameter-space Gaussian smoothing + Adam
function run_openai_es(rng, train; sigma = 0.2, npairs = 8)
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
        step_adam!(theta, g, adam)
        evals += 2npairs
        r = regret(theta, train)
        evals += 1
        r < best_r && (best_r = r; best = copy(theta))
    end
    return best
end

# CMA-ES reference baseline (scalar interface; charged the same budget)
function run_cmaes(rng, train)
    r = CMAEvolutionStrategy.minimize(theta -> regret(theta, train),
        zeros(N_FEAT), 0.3;
        maxfevals = BUDGET, verbosity = 0,
        seed = rand(rng, UInt))
    return CMAEvolutionStrategy.xbest(r)
end

# (1+1) random search with 1/5th-rule step-size adaptation
function run_random_search(rng, train; sigma0 = 0.3)
    theta = zeros(N_FEAT)
    best_r = regret(theta, train)
    sigma = sigma0
    for _ in 2:BUDGET
        cand = theta .+ sigma .* randn(rng, N_FEAT)
        r = regret(cand, train)
        if r < best_r
            best_r = r
            theta = cand
            sigma *= 1.5
        else
            sigma *= 0.98
        end
    end
    return theta
end

# ---------------------------------------------------------------------------
function main()
    names = ("PolyStep", "SFGE", "OpenAI-ES", "CMA-ES", "random")
    runners = (run_polystep, run_sfge, run_openai_es, run_cmaes, run_random_search)
    results = Dict(name => Float64[] for name in names)
    for seed in SEEDS
        rng = Xoshiro(splitmix64(1234, seed))
        theta_star = randn(rng, N_FEAT)
        train = make_instances(rng, N_TRAIN, theta_star)
        test = make_instances(rng, N_TEST, theta_star)
        for (name, runner) in zip(names, runners)
            theta = runner(Xoshiro(splitmix64(seed, 7)), train)
            push!(results[name], regret(theta, test))
        end
        @printf("seed %d done\n", seed)
    end

    println("\nTest regret (median over $(length(SEEDS)) seeds, budget = $BUDGET candidate evals):")
    meds = Dict(name => median(rs) for (name, rs) in results)
    for name in names
        @printf("  %-10s %.4f\n", name, meds[name])
    end

    # assertion gate: PolyStep must beat the published baseline for
    # constraint-side DFL (SFGE, score-function REINFORCE) and random search.
    # OpenAI-ES / CMA-ES are direct-search methods, reported unfiltered for
    # reference alongside PolyStep.
    @assert meds["PolyStep"] <= meds["SFGE"] "PolyStep should beat SFGE"
    @assert meds["PolyStep"] <= meds["random"] "PolyStep should beat random search"
    println("\nGate passed: PolyStep beats the published SFGE baseline and random search.")
    return meds
end

main()
