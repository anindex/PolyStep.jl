# Decision-focused learning through a hard maximizer: SubsetSelectionBenchmark
# from DecisionFocusedLearningBenchmarks.jl (top-5 of 25 items). A linear model
# (625 weights) scores the items; the loss is the piecewise-constant true regret.
#
# Methods, seeds 1:3 (the seed fixes data, initial model and rng):
#   PolyStep     zeroth-order on mean train regret, straight through top-k
#   IPOP-CMA-ES  same objective and budget (ipop_cma.jl)
#   InferOpt FY  PerturbedAdditive + FenchelYoungLoss + Adam on the true top-k
# Budget in maximizer calls: PolyStep and CMA-ES 100 rounds x 2*625 candidates
# x 30 samples (3.75M); FY 60 epochs x 10 perturbations (18k, converged).
#
# Tuning on seeds 101:103, train objective only (selected in brackets):
#   PolyStep  epsilon {0.1, [0.3], 1} x step_radius {6, 12, 24, [48]}
#   CMA-ES    s0 {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, [1], 2, 5, 10, 20}
#   FY        lr {0.0005, 0.001, 0.002, 0.005, 0.01, [0.02], 0.05, 0.1, 0.2, 0.5, 1, 2}
#             at scale 0.1; all lr >= 0.002 reach zero train regret, so picked by FY loss
#
# Result: PolyStep has the lowest test regret on every seed; FY fits the train
# set exactly with about 200x fewer calls.
#
# Run:  julia -t 4 --project=examples examples/04_dfl_benchmarks.jl  (about 7 min)

using PolyStep
using DecisionFocusedLearningBenchmarks
using InferOpt
using Flux
using Zygote
using Random
using Statistics
using Printf
using LinearAlgebra
include(joinpath(@__DIR__, "ipop_cma.jl"))

BLAS.set_num_threads(1)   # reproducible CMA-ES eigendecompositions

const NTRAIN = 30
const NTEST = 30
const SEEDS = 1:3
const ROUNDS = 100
const EPOCHS = 60
const NB_SAMPLES = 10

function regret(model, samples, maximizer)
    r = 0.0
    for s in samples
        y = maximizer(model(s.x))
        r += Float64(s.θ' * s.y - s.θ' * y)     # value gap under the true item values
    end
    return r / length(samples)
end

# (train regret, test regret, maximizer calls) for each method on one seed
function run_seed(seed)
    bench = SubsetSelectionBenchmark()
    train = generate_dataset(bench, NTRAIN; seed = seed)
    test = generate_dataset(bench, NTEST; seed = 1000 + seed)
    maximizer = generate_maximizer(bench)
    model0 = generate_statistical_model(bench; seed = seed)
    w0, restruct = Flux.destructure(model0)
    D = length(w0)
    score(m) = (regret(m, train, maximizer), regret(m, test, maximizer))

    # PolyStep: mean train regret, one thread per candidate
    function f_batched(X)
        out = Vector{Float64}(undef, size(X, 2))
        Threads.@threads :static for j in 1:size(X, 2)
            out[j] = regret(restruct(Float32.(view(X, :, j))), train, maximizer)
        end
        return out
    end
    es = minimize(f_batched, D; steps = ROUNDS, epsilon = 0.3, step_radius = 48.0,
        x0 = Float64.(w0), rng = Xoshiro(seed))
    ps = (score(restruct(Float32.(es.best_x)))..., es.evals * NTRAIN)

    # IPOP-CMA-ES with PolyStep's eval count
    f(x) = regret(restruct(Float32.(x)), train, maximizer)
    xc, _, ev = ipop_cma(f, Float64.(w0), 1.0, es.evals; seed = seed)
    cma = (score(restruct(Float32.(xc)))..., ev * NTRAIN)

    # InferOpt FY: imitate the true y
    fy_model = deepcopy(model0)
    layer = PerturbedAdditive(maximizer; ε = 0.1, nb_samples = NB_SAMPLES, rng = Xoshiro(seed))
    loss = FenchelYoungLoss(layer)
    opt_state = Flux.setup(Flux.Adam(0.02), fy_model)
    for _ in 1:EPOCHS, s in train
        g = Zygote.gradient(m -> loss(m(s.x), s.y), fy_model)
        Flux.update!(opt_state, fy_model, g[1])
    end
    fy = (score(fy_model)..., EPOCHS * NTRAIN * NB_SAMPLES)

    return (; untrained = (score(model0)..., 0), ps, cma, fy)
end

function main()
    runs = [run_seed(s) for s in SEEDS]
    println("Decision-focused learning: subset selection (top-5 of 25), hard maximizer")
    @printf("  linear model, D=625 weights, %d train / %d test samples, seeds %s\n",
        NTRAIN, NTEST, SEEDS)
    @printf("  %-20s %15s %12s %11s   %s\n", "method", "maximizer calls", "train regret",
        "test regret", "test per seed")
    println("  " * "-"^80)
    rows = (:untrained => "untrained model", :ps => "PolyStep", :cma => "IPOP-CMA-ES",
        :fy => "InferOpt FY + Adam")
    for (k, name) in rows
        r = [getfield(run, k) for run in runs]
        @printf("  %-20s %15d %12.4f %11.4f   %s\n", name, r[1][3], mean(first, r),
            mean(x -> x[2], r), join((@sprintf("%.3f", x[2]) for x in r), " "))
    end
    println("  (train and test regret are means over seeds; calls are per seed)")
    r0 = mean(run.untrained[2] for run in runs)
    ps = mean(run.ps[2] for run in runs)
    fy = mean(run.fy[2] for run in runs)
    cma = mean(run.cma[2] for run in runs)
    @assert ps < 0.75 * r0 "PolyStep failed to substantially reduce test regret ($ps vs untrained $r0)"
    @assert ps < min(fy, cma) "PolyStep test regret $ps is not below FY $fy and CMA-ES $cma"
    println("  Gate passed: PolyStep has the lowest mean test regret (FY uses about 200x fewer calls)")
end

main()
