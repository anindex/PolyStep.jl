# PolyStep.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://anindex.github.io/PolyStep.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://anindex.github.io/PolyStep.jl/dev/)
[![CI](https://github.com/anindex/PolyStep.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/anindex/PolyStep.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/anindex/PolyStep.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/anindex/PolyStep.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![arXiv](https://img.shields.io/badge/arXiv-2605.01928-b31b1b.svg)](https://arxiv.org/abs/2605.01928)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Gradient-free direct search for black-box objectives that are piecewise constant or
non-differentiable: simulation optimization, decision-focused learning, and pipelines
through combinatorial solvers. It is the Julia implementation of
[PolyStep](https://github.com/anindex/polystep)
([TMLR 2026](https://arxiv.org/abs/2605.01928)).

Each step places a randomly rotated polytope around every particle, evaluates all
vertices in one batched call, weights them by `softmax(-C/epsilon)` or an entropic
transport plan, and moves to the weighted barycenter. The finite radius lets a step
cross the flat pieces of a loss whose gradient is zero almost everywhere. Results are
bit-identical for any thread count.

[Paper](https://arxiv.org/abs/2605.01928) ·
[Interactive visualization](https://vietngth.github.io/polystep-visualization/) ·
[Guide](https://anindex.github.io/PolyStep.jl/stable/guide/) ·
[Benchmarks](https://anindex.github.io/PolyStep.jl/stable/benchmarks/)

<p align="center">
  <img src="https://raw.githubusercontent.com/anindex/polystep/main/docs/figures/method_diagram.png" width="760"
       alt="PolyStep step: polytope probes around each particle, a cost matrix, a soft assignment, and a barycentric update.">
</p>

## Installation

```julia
using Pkg
Pkg.add("PolyStep")   # before registration: Pkg.add(url = "https://github.com/anindex/PolyStep.jl")
```

## Quick start

```julia
using PolyStep, Random

# batched objective: (d, N) candidate columns -> N costs.
# Piecewise constant, zero gradient almost everywhere, minimum 0.
f(X) = vec(sum(floor.(abs.(X)); dims = 1))

x0 = fill(5.5, 10)
es = minimize(f, 10; steps = 100, epsilon = 0.1, step_radius = 1.0, x0,
              rng = Xoshiro(0))
println((f0 = f(reshape(x0, :, 1))[1], best_f = es.best_f, evals = es.evals))
# (f0 = 50.0, best_f = 0.0, evals = 2001)
```

The radius has to reach past the flat pieces: with `step_radius = 0.5` every vertex
costs 50, the weights are uniform, and the iterate stays at `x0`.

Ask/tell, with bounds and integrality repair:

```julia
cost(X) = vec(sum(abs2, X .- 3.4; dims = 1))   # best integer point: fill(3.0, 5)

es = PolyStepES(5; epsilon = 0.1, step_radius = 1.0, x0 = fill(9.0, 5),
                lb = 0.0, ub = 10.0, repair = X -> (X .= round.(X); X))
for _ in 1:100
    X = ask!(es)          # (dim, popsize) columns, clamped and repaired
    tell!(es, cost(X))
end
println((best_x = es.best_x, best_f = es.best_f))
# (best_x = [3.0, 3.0, 3.0, 3.0, 3.0], best_f = 0.7999999999999996)
```

`PolyStepConfig` / `step!` / `solve!` expose the full feature set (Sinkhorn and other
OT solvers, epsilon schedules, momentum, adaptive radius, biased rotations, a
finite-difference quadratic model with Newton refinement and a trust region), and
`PolyStepOptimizer` plugs into [Optimization.jl](https://github.com/SciML/Optimization.jl).
See the [guide](https://anindex.github.io/PolyStep.jl/stable/guide/) for tuning,
bounds, threading and subspaces.

## When to use it

- **Good fit:** the loss goes through argmax routing, a combinatorial solver or an
  integer-valued simulation; evaluations are cheap and batchable; runs must
  reproduce bit for bit at any thread count.
- **Use something else when:** evaluations take minutes (Bayesian optimization); the
  problem is smooth or low dimensional (CMA-ES, or Nelder-Mead, which solves example 02
  in 106 evaluations); a usable gradient or surrogate exists (in example 04 a
  Fenchel-Young loss needs about 200x fewer solver calls); you need proven convergence
  with native integer variables (NOMAD).

## Examples

Runnable scripts, each with a printed table and a pass/fail assertion. Every method gets
the same evaluation budget and an equal-size tuning grid selected on the training
objective only (grids in the script headers). CMA-ES is IPOP-CMA-ES with restarts
([ipop_cma.jl](examples/ipop_cma.jl)). Run `julia --project=examples -e 'using Pkg;
Pkg.instantiate()'` once, then `julia --project=examples examples/<script>.jl`.

| # | Problem | Result |
|---|---|---|
| [01](examples/01_hard_decision_tree.jl) | Hard oblique decision tree, 0-1 loss, 20 checkerboards, 40k evals | Median train/test accuracy: PolyStep 94.2/90.9%, IPOP-CMA-ES 87.7/83.3%, OpenAI-ES 85.4/80.8%, SPSA 84.6/80.4%. PolyStep has the higher test accuracy on 16-19 of 20 instances. |
| [02](examples/02_inventory_sS.jl) | (s,S) inventory policy, 2-D, Poisson demand, 5 seeds | Tie: PolyStep, IPOP-CMA-ES and Nelder-Mead reach the grid optimum. SPSA trails at a 500-eval budget. |
| [03](examples/03_districting_cmst.jl) | Contextual districting through a capacitated-MST decoder, 8000 decoder calls, 5 seeds | Tie: PolyStep 9.549, perturbed gradient + Adam 9.554, IPOP-CMA-ES 9.584 (true-cost decode 9.674); means within 0.4%. |
| [04](examples/04_dfl_benchmarks.jl) | Decision-focused subset selection (DecisionFocusedLearningBenchmarks), 625 weights, hard top-k, 3 seeds | Test regret: PolyStep 2.27, Fenchel-Young 3.51, IPOP-CMA-ES 4.02 (untrained 6.51), lowest on every seed, at about 200x more solver calls than Fenchel-Young. |
| [05](examples/05_predicted_weights_knapsack.jl) | DFL with predicted knapsack weights, 10 seeds, 12k evals | Lowest median train regret (0.229 vs CMA-ES 0.302); median test regret 4.74 vs CMA-ES 4.67 and SFGE 4.90, within seed noise. |

On the COCO `bbob-mixint` suite (mixed-integer, dimensions 5 and 10) PolyStep solves
about as many targets as CMA-ES with margin, leads at budgets up to `1000 * dim`, and
trails pycma at `10^4 * dim` in dimension 10
([benchmarks](https://anindex.github.io/PolyStep.jl/stable/benchmarks/)).

## Maintainers

- [An T. Le](https://github.com/anindex)
- [Viet T. Nguyen](https://vietngth.github.io/)

## Citation

```bibtex
@article{le2026training,
  title   = {Training Non-Differentiable Networks via Optimal Transport},
  author  = {Le, An T.},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.01928}
}
```

See also [CITATION.cff](CITATION.cff) and the [changelog](CHANGELOG.md).
