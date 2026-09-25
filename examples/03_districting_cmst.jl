# Contextual districting through a hard capacitated-MST decoder, the problem
# class of DistrictNet (Ahmed et al., NeurIPS 2024, arXiv 2412.08287),
# re-created without their smoothed CMST layer.
#
# Pipeline: edge features -> MLP (57 params) -> predicted edge costs ->
# Esau-Williams CMST -> districts -> loss = sum of district TSP tours on the
# true costs. True costs carry a "river" multiplier; the encoder sees only the
# crossing flag. EW is a heuristic, so learned costs can decode cheaper than
# the true costs (gap closed above 100%).
#
# Compared: PolyStep, IPOP-CMA-ES, perturbed-optimizer gradient + Adam.
# Budget: 8000 decoder calls (EW + TSP) per method, one instance, seeds 0-4.
# Tuning: 12-point grid per method, best mean cost on seeds 100-104:
#   PolyStep        epsilon {0.01,0.1,1} x step_radius {0.3,1.2,4.8,19.2}  -> 0.01, 19.2
#   IPOP-CMA-ES     sigma0 {0.15 * sqrt(2)^k, k = 0:11}                    -> 3.4
#   Perturbed+Adam  sigma {0.3,1,3,10} x lr {0.03,0.1,0.3}                 -> 3, 0.3
#
# Result: a tie; all three beat the true-cost decode (9.674) and reach 9.549
# on most seeds, means within 0.4%.
#
# Run:  julia --project=examples examples/03_districting_cmst.jl

using PolyStep
using Random
using Statistics
using Printf
include(joinpath(@__DIR__, "ipop_cma.jl"))

const NCUST = 60
const CAP = 10
const BUDGET = 8_000            # decoder calls per method
const D_PARAM = 8 * 5 + 8 + 8 + 1   # MLP 5 -> 8(tanh) -> 1: W1(8x5), b1(8), w2(8), b2

# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------
struct Instance
    xy::Matrix{Float64}       # (2, n+1); column 1 = depot at (0.5, 0.5)
    true_cost::Matrix{Float64}    # (n+1, n+1) symmetric
    feats::Array{Float64, 3}   # (5, n+1, n+1) edge features
end

crosses_river(y1, y2) = (y1 - 0.5) * (y2 - 0.5) < 0

function make_instance(; n = NCUST, seed = 1)
    rng = Xoshiro(seed)
    xy = hcat([0.5, 0.5], rand(rng, 2, n))
    m = n + 1
    tc = zeros(m, m)
    fe = zeros(5, m, m)
    for i in 1:m, j in 1:m

        i == j && continue
        len = hypot(xy[1, i] - xy[1, j], xy[2, i] - xy[2, j])
        cr = crosses_river(xy[2, i], xy[2, j])
        tc[i, j] = len * (1 + 2 * cr)          # river multiplier
        d_i = hypot(xy[1, i] - 0.5, xy[2, i] - 0.5)
        d_j = hypot(xy[1, j] - 0.5, xy[2, j] - 0.5)
        fe[:, i, j] .= (len, float(cr), d_i, d_j, 1.0)
    end
    return Instance(xy, tc, fe)
end

# ---------------------------------------------------------------------------
# Encoder: flat 57-vector -> predicted edge cost matrix (softplus > 0)
# ---------------------------------------------------------------------------
softplus(x) = x > 20 ? x : log1p(exp(x))

function predict_costs!(pc::Matrix{Float64}, theta::AbstractVector, inst::Instance)
    m = size(pc, 1)
    W1 = reshape(view(theta, 1:40), 8, 5)
    b1 = view(theta, 41:48)
    w2 = view(theta, 49:56)
    b2 = theta[57]
    h = zeros(8)
    @inbounds for i in 1:m, j in 1:m

        if i == j
            pc[i, j] = 0.0
            continue
        end
        f = view(inst.feats, :, i, j)
        for k in 1:8
            acc = b1[k]
            for l in 1:5
                acc += W1[k, l] * f[l]
            end
            h[k] = tanh(acc)
        end
        out = b2
        for k in 1:8
            out += w2[k] * h[k]
        end
        pc[i, j] = softplus(out) + 1e-6
    end
    return pc
end

# ---------------------------------------------------------------------------
# Esau-Williams capacitated MST (1966): from the depot star, apply the best
# positive saving gate(i) - link cost within capacity. O(n^2) per merge.
# ---------------------------------------------------------------------------
function esau_williams(cost::AbstractMatrix, cap::Int)
    m = size(cost, 1)                 # node 1 = depot
    n = m - 1
    comp = collect(1:n)               # component id per customer (index by cust = node-1)
    size_c = ones(Int, n)
    gate = [cost[1, i + 1] for i in 1:n]     # current depot-link cost of i's component
    links = Tuple{Int, Int}[]               # chosen customer-customer edges (node ids)
    while true
        best_s = 1e-12
        best = (0, 0)
        for i in 1:n, j in 1:n

            ci = comp[i];
            cj = comp[j]
            ci == cj && continue
            size_c[ci] + size_c[cj] > cap && continue
            s = gate[ci] - cost[i + 1, j + 1]
            if s > best_s
                best_s = s
                best = (i, j)
            end
        end
        best == (0, 0) && break
        i, j = best
        ci, cj = comp[i], comp[j]
        push!(links, (i + 1, j + 1))
        newsize = size_c[ci] + size_c[cj]
        gate_new = gate[cj]                # merged component keeps j-side depot gate
        for k in 1:n
            comp[k] == ci && (comp[k] = cj)
        end
        size_c[cj] = newsize
        gate[cj] = gate_new
    end
    clusters = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(clusters, comp[i], Int[]), i + 1)
    end
    return collect(values(clusters)), links
end

function check_feasible(clusters, links, n, cap)
    all_nodes = sort!(reduce(vcat, clusters))
    @assert all_nodes == collect(2:(n + 1)) "every customer in exactly one district"
    @assert all(length(c) <= cap for c in clusters) "capacity respected"
    @assert length(links) == n - length(clusters) "forest structure (acyclic)"
    return true
end

# ---------------------------------------------------------------------------
# District cost: TSP tour (depot + district) on true costs, NN + 2-opt
# ---------------------------------------------------------------------------
function tour_cost(nodes::Vector{Int}, cost::Matrix{Float64})
    tour = [1; nodes]                       # start at depot
    k = length(tour)
    k <= 2 && return k == 2 ? 2 * cost[1, tour[2]] : 0.0
    # nearest neighbor
    unvis = Set(tour[2:end])
    path = [1]
    cur = 1
    while !isempty(unvis)
        nxt = argmin(Dict(v => cost[cur, v] for v in unvis))
        push!(path, nxt)
        delete!(unvis, nxt)
        cur = nxt
    end
    # 2-opt
    improved = true
    while improved
        improved = false
        for a in 1:(k - 2), b in (a + 2):k

            i1, i2 = path[a], path[a % k + 1]
            j1, j2 = path[b], path[b % k + 1]
            (i1 == j1 || i2 == j1 || j2 == i1) && continue
            delta = cost[i1, j1] + cost[i2, j2] - cost[i1, i2] - cost[j1, j2]
            if delta < -1e-12
                reverse!(path, a + 1, b)
                improved = true
            end
        end
    end
    total = 0.0
    for a in 1:k
        total += cost[path[a], path[a % k + 1]]
    end
    return total
end

function decode_cost(pc::Matrix{Float64}, inst::Instance)
    clusters, links = esau_williams(pc, CAP)
    check_feasible(clusters, links, NCUST, CAP)
    return sum(tour_cost(c, inst.true_cost) for c in clusters)
end

# the full pipeline objective (one "decoder call" per theta)
function pipeline_loss(theta::AbstractVector, inst::Instance, pc::Matrix{Float64})
    predict_costs!(pc, theta, inst)
    return decode_cost(pc, inst)
end

# ---------------------------------------------------------------------------
# Baseline: perturbed-optimizer gradient (Berthet et al. 2020) + Adam,
# M decoder calls per gradient.
# ---------------------------------------------------------------------------
function perturbed_adam(inst, theta0; sigma = 3.0, M = 16, lr = 0.3, budget = BUDGET, rng = Xoshiro(0))
    theta = copy(theta0)
    mt = zero(theta)
    vt = zero(theta)
    pc = zeros(NCUST + 1, NCUST + 1)
    best = (Inf, copy(theta))
    used = 0
    t = 0
    while used + M <= budget
        t += 1
        Z = randn(rng, length(theta), M)
        losses = [pipeline_loss(theta .+ sigma .* Z[:, k], inst, pc) for k in 1:M]
        used += M
        lmin = minimum(losses)
        lmin < best[1] && (best = (lmin, theta .+ sigma .* Z[:, argmin(losses)]))
        adv = (losses .- mean(losses)) ./ (std(losses) + 1e-8)
        g = (Z * adv) ./ (M * sigma)
        mt .= 0.9 .* mt .+ 0.1 .* g
        vt .= 0.999 .* vt .+ 0.001 .* g .^ 2
        theta .-= lr .* (mt ./ (1 - 0.9^t)) ./ (sqrt.(vt ./ (1 - 0.999^t)) .+ 1e-8)
    end
    return best[1], best[2], used
end

# ---------------------------------------------------------------------------
function main()
    inst = make_instance()
    pc = zeros(NCUST + 1, NCUST + 1)

    # small-instance sanity check of the decoder
    small = make_instance(n = 6, seed = 7)
    spc = copy(small.true_cost)
    cl, li = esau_williams(spc, 3)
    @assert all(length(c) <= 3 for c in cl)
    star = 2 * sum(small.true_cost[1, i] for i in 2:7)
    ew_cost = sum(tour_cost(c, small.true_cost) for c in cl)
    @assert ew_cost <= star + 1e-9 "EW must beat the star solution"

    references = Dict{String, Float64}()
    references["true-cost decode"] = decode_cost(copyto!(pc, inst.true_cost), inst)
    rng0 = Xoshiro(99)
    references["random-theta decode"] = mean(pipeline_loss(0.3 .* randn(rng0, D_PARAM), inst,
                                                 zeros(NCUST + 1, NCUST + 1))
    for _ in 1:10)

    c_true = references["true-cost decode"]
    c_rand = references["random-theta decode"]
    gap(c) = 100 * (c_rand - c) / (c_rand - c_true)

    println("Contextual districting via capacitated MST (DistrictNet problem class)")
    @printf("  n=%d customers, capacity=%d, encoder D=%d, budget=%d decoder calls/method\n",
        NCUST, CAP, D_PARAM, BUDGET)
    @printf("  reference: random-theta %.3f | true-cost decode %.3f (EW carries ~5-10%% CMST gap)\n",
        c_rand, c_true)

    seeds = 0:4
    results = Dict{String, Vector{Float64}}()
    evals = Dict{String, Vector{Int}}()
    record!(name, f, n) = (push!(get!(results, name, Float64[]), f);
        push!(get!(evals, name, Int[]), n))
    for seed in seeds
        theta0 = 0.3 .* randn(Xoshiro(10 + seed), D_PARAM)
        pcl = zeros(NCUST + 1, NCUST + 1)
        f_b(X) = [pipeline_loss(view(X, :, j), inst, pcl) for j in 1:size(X, 2)]

        es = minimize(f_b, D_PARAM; steps = div(BUDGET, 2 * D_PARAM), epsilon = 0.01,
            step_radius = 19.2, x0 = theta0, rng = Xoshiro(seed), scale_cost = :mean)
        record!("PolyStep", es.best_f, es.evals)

        _, fb, n = ipop_cma(x -> pipeline_loss(x, inst, pcl), theta0, 3.4, BUDGET; seed = seed + 1)
        record!("IPOP-CMA-ES", fb, n)

        bf, _, n = perturbed_adam(inst, theta0; rng = Xoshiro(seed))
        record!("Perturbed+Adam", bf, n)
    end

    @printf("  %-16s %14s %18s %8s\n", "method", "cost (mean+-sd)", "gap closed (mean)", "evals")
    println("  " * "-"^61)
    for name in ("PolyStep", "IPOP-CMA-ES", "Perturbed+Adam")
        cs = results[name]
        @printf("  %-16s %8.3f +- %.3f %16.1f%% %8d\n", name, mean(cs), std(cs), mean(gap.(cs)),
            maximum(evals[name]))
    end
    println("  (gap closed = progress from random-theta toward true-cost decoding;")
    println("   evals = most decoder calls used by any seed)")
    ps = mean(results["PolyStep"])
    @assert ps <= c_true "PolyStep did not reach the true-cost decode"
    @assert ps <= 1.01 * minimum(mean, values(results)) "PolyStep not within 1% of the best method"
    println("  Gate passed: PolyStep beats the true-cost decode and is within 1% of the best mean")
    return results
end

main()
