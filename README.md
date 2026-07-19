# PolyStep.jl

[![CI](https://github.com/anindex/PolyStep.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/anindex/PolyStep.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/anindex/PolyStep.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/anindex/PolyStep.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-%E2%89%A5%201.10-9558B2.svg)](https://julialang.org)

Gradient-free direct search for piecewise-constant objectives: simulation
optimization, decision-focused learning, and contextual combinatorial pipelines.
A Julia port of [PolyStep](https://github.com/anindex/polystep) for
operations-research problems.

Each step rotates a unit polytope (an orthoplex by default) by a Haar-random
rotation, evaluates the objective at the vertex directions in one batched call,
weights the vertices by `softmax(-C/eps)` or an entropic Sinkhorn plan, and steps
to the barycenter. The finite probe radius crosses the flat pieces of a
piecewise-constant loss, where gradient-estimate methods (OpenAI-ES, SPSA)
average equal losses into a zero gradient and stall.

## When to use it

Good fit: the objective is piecewise-constant or non-differentiable (losses
through argmax routing, combinatorial solvers, integer-demand simulations), and
evaluations are cheap and batchable (one `(d, N)` call is thousands of
vectorized simulations).

Poor fit: expensive single simulations (minutes per run); use Bayesian
optimization instead. PolyStep can spend orders of magnitude more evaluations
than ES, traded for not needing a local-gradient signal.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/anindex/PolyStep.jl")
```

## Quick start

```julia
using PolyStep, Random

# batched objective: (d, N) candidate columns -> N costs
rastrigin(X) = vec(sum(X .^ 2 .- 10 .* cos.(2 * pi .* X) .+ 10; dims = 1))

es = minimize(rastrigin, 10; steps = 500, epsilon = 0.05, step_radius = 0.3,
              x0 = fill(2.0, 10), rng = Xoshiro(0))
es.best_x, es.best_f
```

Ask/tell, with bounds and integrality repair:

```julia
es = PolyStepES(5; lb = 0.0, ub = 10.0, repair = X -> (X .= round.(X); X))
for _ in 1:200
    X = ask!(es)          # (dim, popsize) columns, clamped and repaired
    tell!(es, my_costs(X))
end
```

The lower-level `PolyStepConfig` / `init_state` / `step!` / `solve!` API exposes the
full feature set: Sinkhorn / KL-softmax / tempered / greedy solvers, epsilon
schedules, decoupled probe and step radii, momentum, adaptive radius, biased
rotations, and a finite-difference quadratic model with Newton refinement and
trust region. It also plugs into
[Optimization.jl](https://github.com/SciML/Optimization.jl) through
`PolyStepOptimizer`.

## Examples

Each example is a runnable script with a printed table and a pass/fail assertion.

| # | Problem | Result |
|---|---|---|
| [01](examples/01_hard_decision_tree.jl) | Hard oblique decision tree, argmax routing, 0-1 loss, 40k-eval budget | PolyStep 93.7% vs OpenAI-ES 85.7%, SPSA 82.5% |
| [02](examples/02_inventory_sS.jl) | (s,S) inventory policy, Poisson demand, CRN streams | PolyStep 0.02% from the grid optimum; SPSA stalls at a 52% gap |
| [03](examples/03_districting_cmst.jl) | Contextual districting through an Esau-Williams capacitated-MST decoder | PolyStep closes 100% of the random-to-true-cost gap |
| [04](examples/04_dfl_benchmarks.jl) | Decision-focused learning, top-k selection, vs InferOpt Fenchel-Young | PolyStep test regret 4.10 vs FY 4.41 |
| [05](examples/05_predicted_weights_knapsack.jl) | DFL with predicted constraint weights, integer knapsack | PolyStep 6.80 vs SFGE 9.24, random 8.93 (CMA-ES and OpenAI-ES also beat SFGE) |

## Where gradient-free is the only option

Decision-focused learning that predicts *objective* coefficients has smoothed
surrogates: SPO+ (Elmachtoub and Grigas, Management Science 2022), Fenchel-Young
losses and perturbed optimizers (Berthet et al., NeurIPS 2020), available in Julia
through [InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl).
When the prediction instead enters the *constraints*
([example 05](examples/05_predicted_weights_knapsack.jl): predicted integer knapsack
weights), those surrogates do not apply. The decision is a feasible-set solve, the
regret is piecewise-constant in the parameters, and smoothing leaves the gradient
zero, so the only gradient-based route is a high-variance score-function estimator
(SFGE, Silvestri et al. 2023, arXiv:2307.05213). PolyStep optimizes that regret
directly, with no surrogate and no smoothing hyperparameters.
[Example 01](examples/01_hard_decision_tree.jl) (a hard oblique decision tree under
0-1 loss) is the same story without the DFL framing: the gradient is zero almost
everywhere and OpenAI-ES and SPSA stall, while PolyStep's finite-radius steps cross
the split boundaries.

## Notes

- Column-major throughout: candidates are `(d, N)` columns and cost/plan
  matrices are `(V, P)`, so per-particle reductions stay contiguous. Hot kernels
  are zero-allocation in steady state, and rotations come from one serial RNG, so
  results are bit-identical for any thread count.
- Float64 by default (what OR closures expect); `T = Float32` is the
  performance/GPU mode.
- Optional extensions: LoopVectorization (`@turbo` kernels), CUDA
  (`cuda_objective` for GPU-batched objectives), and Optimization.jl.
- Two outer-loop heuristics differ from the Python reference for correctness
  (see [CHANGELOG.md](CHANGELOG.md)): convergence also requires a small step
  magnitude, and the trust-region ratio scores the transport step taken.
