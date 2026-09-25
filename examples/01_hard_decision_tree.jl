# Hard oblique decision tree trained with no gradients and no relaxation on a
# checkerboard task. Each node routes with a strict w.x + b > 0 test, so the 0-1
# training loss is piecewise constant with zero gradient almost everywhere.
#
# Compared: PolyStep, OpenAI-ES, SPSA, IPOP-CMA-ES (if CMAEvolutionStrategy is
# installed). Budget: 40,000 train-loss evals per run, best evaluated point kept.
# 20 instances (seeds 0:19), each with fresh data (600 train, 600 test) and x0.
#
# Tuning: 16-point grid per method, best median train accuracy on seeds 100:119:
#   PolyStep     epsilon {0.01,0.03,0.1,0.3} x step_radius {0.75,1.5,3,6}  -> 0.03, 1.5
#   OpenAI-ES    sigma {0.1,0.3,1,3} x lr {0.1,0.3,1,3}                    -> 1.0, 3.0
#   SPSA         a {10,30,100,300} x c {0.1,0.3,1,3}                       -> 30, 0.3
#   IPOP-CMA-ES  sigma0, 16 log-spaced values in [0.1, 10]                 -> 10 (edge; up to 100 no better)
#
# Result (train/test medians): PolyStep 94.9/89.8%, IPOP-CMA-ES 87.7/83.3%,
# OpenAI-ES 85.4/80.8%, SPSA 84.6/80.4%.
#
# Run:  julia --project=examples examples/01_hard_decision_tree.jl
#       (julia --project=. runs it without the IPOP-CMA-ES row)

using PolyStep
import PolyStep: ask!, tell!, popsize    # extended below for the inline baselines
using Random
using Statistics
using Printf

const HAVE_CMA = Base.find_package("CMAEvolutionStrategy") !== nothing
if HAVE_CMA
    include("ipop_cma.jl")
end

const DEPTH = 4
const TOTAL_EVALS = 40_000
const N_INTERNAL = 2^DEPTH - 1
const N_LEAVES = 2^DEPTH
const D_IN = 2
const D = N_INTERNAL * (D_IN + 1) + N_LEAVES   # 61

# 3x3 checkerboard standardized with training statistics, plus a random x0.
function make_task(seed; n_train = 600, n_test = 600, k = 3, noise = 0.05)
    rng = Xoshiro(seed)
    n = n_train + n_test
    X = rand(rng, 2, n) .* k
    y = isodd.(floor.(Int, X[1, :]) .+ floor.(Int, X[2, :]))
    tr, te = 1:n_train, (n_train + 1):n
    mu, sd = mean(X[:, tr]; dims = 2), std(X[:, tr]; dims = 2)
    X = (X .- mu) ./ sd .+ noise .* randn(rng, 2, n)
    x0 = 0.3 .* randn(rng, D)
    return (; Xtr = X[:, tr], ytr = y[tr], Xte = X[:, te], yte = y[te], x0)
end

# 0-1 error of the hard tree for a (D, B) batch of flat parameter columns.
# Layout per column: W (n_internal x d_in, node-major), then biases, then leaf logits.
function hard_error(flat::AbstractMatrix, X::Matrix{Float64}, y::AbstractVector{Bool})
    B = size(flat, 2)
    N = size(X, 2)
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
                dec = w1 * X[1, m] + w2 * X[2, m] + bb
                node = 2 * node + 1 + (dec > 0 ? 1 : 0)
            end
            leaf = node - N_INTERNAL
            pred = col[N_INTERNAL * (D_IN + 1) + leaf + 1] > 0
            err += pred != y[m]
        end
        out[bi] = err / N
    end
    return out
end
accuracy(x::AbstractVector, X, y) = 100 * (1 - hard_error(reshape(x, :, 1), X, y)[1])

# OpenAI-ES (Salimans et al. 2017): antithetic sampling, z-scored shaping.
mutable struct OpenAIES
    mean::Vector{Float64}
    sigma::Float64
    lr::Float64
    pop::Int
    rng::Xoshiro
    eps::Matrix{Float64}
end
function OpenAIES(x0, pop; sigma, lr, seed = 0)
    pop += pop % 2
    OpenAIES(copy(x0), sigma, lr, pop, Xoshiro(seed), zeros(length(x0), pop))
end
popsize(es::OpenAIES) = es.pop
center(es::OpenAIES) = es.mean
function ask!(es::OpenAIES)
    half = randn(es.rng, length(es.mean), div(es.pop, 2))
    es.eps = hcat(half, -half)
    return es.mean .+ es.sigma .* es.eps
end
function tell!(es::OpenAIES, fit::Vector{Float64})
    adv = (fit .- mean(fit)) ./ (std(fit) + 1e-8)
    es.mean .-= es.lr .* vec(mean(es.eps .* adv'; dims = 2)) ./ es.sigma
end

# SPSA (Spall 1992): two-point Rademacher estimate, A = 10% of the iterations.
mutable struct SPSA
    theta::Vector{Float64}
    a::Float64
    c::Float64
    A::Float64
    k::Int
    rng::Xoshiro
    delta::Vector{Float64}
    ck::Float64
end
function SPSA(x0; a, c, iters, seed = 0)
    SPSA(copy(x0), a, c, 0.1 * iters, 0, Xoshiro(seed), zero(x0), 0.0)
end
popsize(::SPSA) = 2
center(s::SPSA) = s.theta
function ask!(s::SPSA)
    s.ck = s.c / (s.k + 1)^0.101
    s.delta = rand(s.rng, (-1.0, 1.0), length(s.theta))
    return hcat(s.theta .+ s.ck .* s.delta, s.theta .- s.ck .* s.delta)
end
function tell!(s::SPSA, fit::Vector{Float64})
    ak = s.a / (s.k + 1 + s.A)^0.602
    s.theta .-= ak .* (fit[1] - fit[2]) ./ (2 * s.ck) .* s.delta   # 1/delta == delta
    s.k += 1
end

# Keeps one eval to score the final iterate; returns the best evaluated point.
function run_baseline(opt, f; budget = TOTAL_EVALS)
    bx, bf = copy(center(opt)), Inf
    for _ in 1:div(budget - 1, popsize(opt))
        X = ask!(opt)
        fit = f(X)
        tell!(opt, fit)
        v, i = findmin(fit)
        v < bf && (bf = v; bx = X[:, i])
    end
    v = f(reshape(center(opt), :, 1))[1]
    v < bf && (bx = copy(center(opt)))
    return bx
end

# tuned configs from the header
function run_methods(task, seed)
    f(X) = hard_error(X, task.Xtr, task.ytr)
    ps = minimize(f, D; steps = div(TOTAL_EVALS - 1, 2D), num_particles = 1,
        epsilon = 0.03, step_radius = 1.5, x0 = task.x0, rng = Xoshiro(seed))
    es = OpenAIES(task.x0, 2D; sigma = 1.0, lr = 3.0, seed)
    spsa = SPSA(task.x0; a = 30.0, c = 0.3, iters = div(TOTAL_EVALS, 2), seed)
    res = ["PolyStep" => ps.best_x, "OpenAI-ES" => run_baseline(es, f),
        "SPSA" => run_baseline(spsa, f)]
    if HAVE_CMA
        g(x) = f(reshape(x, :, 1))[1]
        push!(res, "IPOP-CMA-ES" => ipop_cma(g, task.x0, 10.0, TOTAL_EVALS; seed)[1])
    end
    return res
end

function main(; seeds = 0:19)
    println("Hard oblique decision tree: no gradients, no relaxation")
    @printf("  depth=%d  nodes=%d  leaves=%d  params=%d  budget=%d evals per run\n",
        DEPTH, N_INTERNAL, N_LEAVES, D, TOTAL_EVALS)
    @printf("  %d task instances (fresh data, x0 and optimizer seed each)\n", length(seeds))
    names = String[]
    tr = Dict{String, Vector{Float64}}()
    te = Dict{String, Vector{Float64}}()
    for s in seeds
        task = make_task(s)
        for (name, x) in run_methods(task, s)
            name in names || push!(names, name)
            push!(get!(tr, name, Float64[]), accuracy(x, task.Xtr, task.ytr))
            push!(get!(te, name, Float64[]), accuracy(x, task.Xte, task.yte))
        end
    end
    ps = "PolyStep"
    @printf("  %-14s%11s%11s%15s%14s\n", "method", "train med", "test med", "test IQR",
        "PS ahead")
    println("  " * "-"^65)
    for n in names
        ahead = n == ps ? "-" : "$(count(te[ps] .> te[n]))/$(length(seeds))"
        @printf("  %-14s%10.1f%%%10.1f%%%9.1f-%.1f%%%14s\n", n, median(tr[n]),
            median(te[n]), quantile(te[n], 0.25), quantile(te[n], 0.75), ahead)
    end
    println("  (PS ahead = instances where PolyStep has the higher test accuracy)")
    HAVE_CMA || println("  (IPOP-CMA-ES skipped: CMAEvolutionStrategy not installed)")

    @assert median(te[ps]) >= 85.0 "PolyStep test median $(median(te[ps]))% below 85%"
    for n in names[2:end]
        @assert median(tr[ps]) >= median(tr[n]) + 2 "train median: PolyStep must lead $n by 2"
        @assert median(te[ps]) >= median(te[n]) + 2 "test median: PolyStep must lead $n by 2"
    end
    println("  Gate passed")
    return tr, te
end

main()
