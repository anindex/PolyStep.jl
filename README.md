# PolyStep.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://anindex.github.io/PolyStep.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://anindex.github.io/PolyStep.jl/dev/)
[![CI](https://github.com/anindex/PolyStep.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/anindex/PolyStep.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/anindex/PolyStep.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/anindex/PolyStep.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![arXiv](https://img.shields.io/badge/arXiv-2605.01928-b31b1b.svg)](https://arxiv.org/abs/2605.01928)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PolyStep.jl minimizes black-box functions that give you no useful gradient: losses that
are piecewise constant, pass through an argmax or a combinatorial solver, or come out of
an integer-valued simulation. It is the Julia version of
[PolyStep](https://github.com/anindex/polystep) ([TMLR 2026](https://arxiv.org/abs/2605.01928)).

Each step puts a randomly rotated polytope around the current point, evaluates all of
its vertices in one batched call, and moves to a softmax-weighted average of them. The
vertices sit a finite distance away, so a step can see across flat regions where the
gradient is zero. The same seed gives bit-identical results on any number of threads.

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

For more control, `PolyStepConfig` with `step!`/`solve!` adds Sinkhorn and other OT
solvers, epsilon schedules, momentum, an adaptive radius, biased rotations, and a
quadratic model with Newton refinement and a trust region. `PolyStepOptimizer` plugs
into [Optimization.jl](https://github.com/SciML/Optimization.jl). The
[guide](https://anindex.github.io/PolyStep.jl/stable/guide/) covers tuning, bounds,
threads and subspaces.

## When to use it

Use it when the loss goes through argmax routing, a combinatorial solver or an
integer-valued simulation, and a batch of evaluations is cheap.

Reach for something else when one evaluation takes minutes (Bayesian optimization), when
the problem is smooth and small (CMA-ES is usually better, and Nelder-Mead solves
example 02 in 106 evaluations), or when a usable gradient or surrogate exists: in
example 04 a Fenchel-Young loss needs about 200x fewer solver calls. For convergence
guarantees with native integer variables, try NOMAD.

## Examples

Each script prints a results table and asserts its claim. All methods get the same
evaluation budget and a tuning grid of the same size, chosen on training data only; the
grids are in the script headers. "CMA-ES" means IPOP-CMA-ES with restarts
([ipop_cma.jl](examples/ipop_cma.jl)).

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/01_hard_decision_tree.jl
```

| # | Problem | Result |
|---|---|---|
| [01](examples/01_hard_decision_tree.jl) | Hard oblique decision tree, 0-1 loss, 20 checkerboards, 40k evals | Median train/test accuracy: PolyStep 94.2/90.9%, IPOP-CMA-ES 87.7/83.3%, OpenAI-ES 85.4/80.8%, SPSA 84.6/80.4%. PolyStep has the higher test accuracy on 16-19 of 20 instances. |
| [02](examples/02_inventory_sS.jl) | (s,S) inventory policy, 2-D, Poisson demand, 5 seeds | PolyStep, IPOP-CMA-ES and Nelder-Mead all reach the grid optimum. SPSA falls behind at a 500-eval budget. |
| [03](examples/03_districting_cmst.jl) | Contextual districting through a capacitated-MST decoder, 8000 decoder calls, 5 seeds | PolyStep 9.549, perturbed gradient + Adam 9.554, IPOP-CMA-ES 9.584; all beat the true-cost decode (9.674) and sit within 0.4% of each other. |
| [04](examples/04_dfl_benchmarks.jl) | Decision-focused subset selection (DecisionFocusedLearningBenchmarks), 625 weights, hard top-k, 3 seeds | Test regret: PolyStep 2.27, Fenchel-Young 3.51, IPOP-CMA-ES 4.02 (untrained 6.51). PolyStep is lowest on every seed but uses about 200x more solver calls than Fenchel-Young. |
| [05](examples/05_predicted_weights_knapsack.jl) | DFL with predicted knapsack weights, 10 seeds, 12k evals | PolyStep has the lowest median train regret (0.229 vs CMA-ES 0.302). On test it is 4.74 vs CMA-ES 4.67 and SFGE 4.90, which is within seed noise. |

We also ran the COCO `bbob-mixint` suite, where 80% of the variables are integers. In
dimensions 5 and 10, PolyStep keeps up with CMA-ES with margin and is ahead up to
`1000 * dim` evaluations; with the full `10^4 * dim` budget, pycma does better in
dimension 10. Details are on the
[benchmarks page](https://anindex.github.io/PolyStep.jl/stable/benchmarks/).

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
