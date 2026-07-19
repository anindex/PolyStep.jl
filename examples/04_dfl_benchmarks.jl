# Decision-focused learning without smoothing: PolyStep trains the
# statistical model of a DecisionFocusedLearningBenchmarks.jl problem
# zeroth-order straight through the hard maximizer (top-k subset selection):
# no perturbation smoothing, no Fenchel-Young surrogate, the true
# piecewise-constant decision regret end to end.
#
# Baseline: the intended pipeline of that ecosystem, InferOpt.jl
# PerturbedAdditive + FenchelYoungLoss + Adam (imitation of the true θ).
# Budget accounting: budgets are counted in maximizer calls; the FY gradient
# pays nb_samples perturbations per training sample per epoch.
# Result: FY wins on maximizer-call efficiency when its smoothing
# assumptions hold; PolyStep needs no smoothing hyperparameters and optimizes
# the hard pipeline directly.
#
# Run:  julia --project=examples examples/04_dfl_benchmarks.jl

using PolyStep
using DecisionFocusedLearningBenchmarks
using InferOpt
using Flux
using Zygote
using Random
using Statistics
using Printf

const NTRAIN = 30
const NTEST = 30

function regret(model, samples, maximizer)
    r = 0.0
    for s in samples
        θ̂ = model(s.x)
        y = maximizer(θ̂)
        r += Float64(s.θ' * s.y - s.θ' * y)     # value gap under the true θ (maximization)
    end
    return r / length(samples)
end

function main()
    Random.seed!(0)
    bench = SubsetSelectionBenchmark()
    train = generate_dataset(bench, NTRAIN)
    test = generate_dataset(bench, NTEST)
    maximizer = generate_maximizer(bench)

    # ---- PolyStep: flat params -> restructure -> mean train regret ----
    model0 = generate_statistical_model(bench)
    θ0, restruct = Flux.destructure(model0)
    D = length(θ0)
    calls = Ref(0)
    function f_batched(X)
        out = Vector{Float64}(undef, size(X, 2))
        Threads.@threads :static for j in 1:size(X, 2)
            m = restruct(Float32.(view(X, :, j)))
            out[j] = regret(m, train, maximizer)
        end
        calls[] += size(X, 2) * NTRAIN
        return out
    end
    rounds = 300
    es = minimize(f_batched, D; steps = rounds, epsilon = 0.01, step_radius = 1.5,
        x0 = Float64.(θ0), rng = Xoshiro(1), scale_cost = :mean)
    ps_model = restruct(Float32.(es.best_x))
    ps_calls = calls[]

    # ---- InferOpt: PerturbedAdditive + FenchelYoungLoss + Adam (imitation) ----
    fy_model = generate_statistical_model(bench)
    nb_samples = 10
    layer = PerturbedAdditive(maximizer; ε = 0.1, nb_samples = nb_samples)
    loss = FenchelYoungLoss(layer)
    opt_state = Flux.setup(Flux.Adam(0.01), fy_model)
    epochs = 60
    for _ in 1:epochs
        for s in train
            g = Zygote.gradient(fy_model) do m
                loss(m(s.x), s.y)
            end
            Flux.update!(opt_state, fy_model, g[1])
        end
    end
    fy_calls = epochs * NTRAIN * nb_samples

    println("Decision-focused learning: subset selection (top-k), hard maximizer")
    @printf("  model D=%d params, %d train / %d test samples\n", D, NTRAIN, NTEST)
    @printf("  %-22s %14s %12s %12s\n", "method", "maximizer calls", "train regret", "test regret")
    println("  " * "-"^64)
    @printf("  %-22s %14d %12.4f %12.4f\n", "PolyStep (zeroth-order)", ps_calls,
        regret(ps_model, train, maximizer), regret(ps_model, test, maximizer))
    @printf("  %-22s %14d %12.4f %12.4f\n", "InferOpt FY + Adam", fy_calls,
        regret(fy_model, train, maximizer), regret(fy_model, test, maximizer))
    r0 = regret(model0, test, maximizer)
    @printf("  (untrained model test regret: %.4f)\n", r0)
    ps_test = regret(ps_model, test, maximizer)
    @assert ps_test < 0.75 * r0 "PolyStep failed to substantially reduce regret ($(ps_test) vs untrained $(r0))"
    println("  Gate passed: PolyStep reduces test regret through the hard maximizer")
end

main()
