# Training a hard oblique decision tree with no gradients and no relaxation, on a
# self-contained checkerboard classification task. An oblique tree routes each
# sample with a strict w.x + b > 0 test; the 0-1 loss is piecewise-constant and
# its gradient is zero almost everywhere. OpenAI-ES and SPSA average equal
# perturbed losses to a zero gradient estimate and stall; PolyStep's
# finite-radius polytope steps across split boundaries.
#
# Run:  julia --project=examples examples/01_hard_decision_tree.jl

using PolyStep
import PolyStep: ask!, tell!    # extended below for the inline baselines
using Random
using Statistics
using Printf

const DEPTH = 4                  # 15 internal nodes, 16 leaves
const TOTAL_EVALS = 40_000
const N_INTERNAL = 2^DEPTH - 1
const N_LEAVES = 2^DEPTH
const D_IN = 2
const D = N_INTERNAL * (D_IN + 1) + N_LEAVES   # 61

function make_checkerboard(; n = 600, k = 3, noise = 0.05, seed = 0)
    rng = Xoshiro(seed)
    X = rand(rng, n, 2) .* k
    y = Float64.((floor.(Int, X[:, 1]) .+ floor.(Int, X[:, 2])) .% 2)
    X = (X .- mean(X; dims = 1)) ./ std(X; dims = 1) .+ noise .* randn(rng, n, 2)
    return Matrix(X'), y            # (2, N) columns = samples
end

# Self-contained checkerboard task (data and x0); no external fixtures.
function load_task()
    Xd, y = make_checkerboard()
    return Xd, y, 0.3 .* randn(Xoshiro(1), D), "checkerboard"
end

# 0-1 error of the hard tree for a (D, B) batch of flat parameter columns.
# Layout per column: W (n_internal x d_in, node-major), then biases, then leaf logits.
function hard_error(flat::AbstractMatrix, Xd::Matrix{Float64}, y::Vector{Float64})
    B = size(flat, 2)
    N = size(Xd, 2)
    out = Vector{Float64}(undef, B)
    Threads.@threads :static for bi in 1:B
        col = view(flat, :, bi)
        err = 0
        @inbounds for m in 1:N
            node = 0                            # 0-based global node index, root = 0
            for _ in 1:DEPTH
                w1 = col[node * D_IN + 1]
                w2 = col[node * D_IN + 2]
                bb = col[N_INTERNAL * D_IN + node + 1]
                dec = w1 * Xd[1, m] + w2 * Xd[2, m] + bb
                node = 2 * node + 1 + (dec > 0 ? 1 : 0)
            end
            leaf = node - (2^DEPTH - 1)
            pred = col[N_INTERNAL * (D_IN + 1) + leaf + 1] > 0
            err += (pred != (y[m] == 1.0)) ? 1 : 0
        end
        out[bi] = err / N
    end
    return out
end

# OpenAI-ES (Salimans et al. 2017): antithetic sampling, z-scored shaping.
mutable struct OpenAIES
    mean::Vector{Float64}
    sigma::Float64
    lr::Float64
    pop::Int
    rng::Xoshiro
    eps::Matrix{Float64}
    best::Float64
end
function OpenAIES(dim, pop, x0; sigma = 0.3, lr = 0.15, seed = 0)
    OpenAIES(copy(x0), sigma, lr, pop + pop % 2, Xoshiro(seed), zeros(dim, pop + pop % 2), Inf)
end
function ask!(es::OpenAIES)
    half = randn(es.rng, length(es.mean), es.pop ÷ 2)
    es.eps = hcat(half, -half)
    return es.mean .+ es.sigma .* es.eps
end
function tell!(es::OpenAIES, fit::Vector{Float64})
    es.best = min(es.best, minimum(fit))
    adv = (fit .- mean(fit)) ./ (std(fit) + 1e-8)
    es.mean .-= es.lr .* vec(mean(es.eps .* adv'; dims = 2)) ./ es.sigma
end

# SPSA (Spall 1992): two-point Rademacher estimate, decaying gains.
mutable struct SPSA
    theta::Vector{Float64}
    a::Float64;
    c::Float64;
    alpha::Float64;
    gamma::Float64
    k::Int
    rng::Xoshiro
    delta::Vector{Float64}
    ck::Float64
    best::Float64
end
function SPSA(dim, x0; a = 0.2, c = 0.2, alpha = 0.602, gamma = 0.101, seed = 0)
    SPSA(copy(x0), a, c, alpha, gamma, 0, Xoshiro(seed), zeros(dim), 0.0, Inf)
end
function ask!(s::SPSA)
    s.ck = s.c / (s.k + 1)^s.gamma
    s.delta = float.(rand(s.rng, (-1, 1), length(s.theta)))
    return hcat(s.theta .+ s.ck .* s.delta, s.theta .- s.ck .* s.delta)
end
function tell!(s::SPSA, fit::Vector{Float64})
    s.best = min(s.best, minimum(fit))
    ak = s.a / (s.k + 1 + 10)^s.alpha
    s.theta .-= ak .* (fit[1] - fit[2]) ./ (2.0 .* s.ck .* s.delta)
    s.k += 1
end

function run_to_budget!(opt, fit_fn, pop; budget = TOTAL_EVALS)
    used = 0
    while used < budget
        tell!(opt, fit_fn(ask!(opt)))
        used += pop
    end
    return opt
end

best_acc(o::PolyStepES) = 100 * (1 - o.best_f)
best_acc(o) = 100 * (1 - o.best)

function solve_one(Xd, y, x0, seed)
    pop = 2D
    fit(flat) = hard_error(flat, Xd, y)
    ps = PolyStepES(D; num_particles = 1, epsilon = 0.02, step_radius = 1.5,
        x0 = x0, rng = Xoshiro(seed))
    es = OpenAIES(D, pop, x0; seed)
    sp = SPSA(D, x0; seed)
    run_to_budget!(ps, fit, popsize(ps))
    run_to_budget!(es, fit, es.pop)
    run_to_budget!(sp, fit, 2)
    return best_acc(ps), best_acc(es), best_acc(sp)
end

function main()
    Xd, y, x0, src = load_task()
    println("Hard oblique decision tree: no gradients, no relaxation")
    @printf("  depth=%d  nodes=%d  leaves=%d  params=%d  budget=%d evals, 5 seeds, task=%s\n",
        DEPTH, N_INTERNAL, N_LEAVES, D, TOTAL_EVALS, src)
    accs = [solve_one(Xd, y, x0, seed) for seed in 0:4]
    ps_med = median(getindex.(accs, 1))
    es_med = median(getindex.(accs, 2))
    sp_med = median(getindex.(accs, 3))
    @printf("  %-24s%10s\n", "method", "median acc")
    println("  " * "-"^34)
    @printf("  %-24s%9.1f%%\n", "PolyStep", ps_med)
    @printf("  %-24s%9.1f%%\n", "OpenAI-ES", es_med)
    @printf("  %-24s%9.1f%%\n", "SPSA", sp_med)

    # gate: PolyStep solves the hard tree; gradient-estimate baselines stall
    @assert ps_med >= 90.0 "PolyStep median $(ps_med)% below the 90% gate"
    @assert ps_med >= es_med + 5.0 "PolyStep ($(ps_med)%) must beat OpenAI-ES ($(es_med)%) by >= 5 points"
    println("  Gate passed")
    return ps_med, es_med, sp_med
end

main()
