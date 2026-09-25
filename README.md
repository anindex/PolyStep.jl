# PolyStep.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://anindex.github.io/PolyStep.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://anindex.github.io/PolyStep.jl/dev/)
[![CI](https://github.com/anindex/PolyStep.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/anindex/PolyStep.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/anindex/PolyStep.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/anindex/PolyStep.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![arXiv](https://img.shields.io/badge/arXiv-2605.01928-b31b1b.svg)](https://arxiv.org/abs/2605.01928)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-%E2%89%A5%201.10-9558B2.svg)](https://julialang.org)

Gradient-free direct search for piecewise-constant objectives: simulation
optimization, decision-focused learning, and contextual combinatorial pipelines.
A Julia port of [PolyStep](https://github.com/anindex/polystep) for
operations-research problems.

Each step rotates a unit polytope (an orthoplex by default) by a Haar-random
rotation, evaluates the objective at the vertices in one batched call, weights
the vertices by `softmax(-C/epsilon)` (by default `C` is shifted to a minimum of 0
and scaled to mean 1, so `epsilon` is relative to the cost spread) or an entropic
Sinkhorn plan, and moves to
the weighted barycenter. The probe radius is finite, so a step compares losses
across the flat pieces of a piecewise-constant loss instead of relying on a local
gradient. OpenAI-ES and SPSA follow a noisy estimate of the gradient of a
smoothed loss; PolyStep's polytope steps can cross from one piece to the next
directly.

## When to use it

Good fit: the objective is piecewise-constant or non-differentiable (losses
through argmax routing, combinatorial solvers, integer-demand simulations), and
evaluations are cheap and batchable (one `(d, N)` call is thousands of
vectorized simulations). PolyStep's strengths are batched evaluation of an exact
black-box pipeline and results that do not depend on the thread count.

Poor fit: expensive single simulations (minutes per run); use Bayesian
optimization instead. On smooth or low-dimensional problems CMA-ES is usually
better per evaluation, and on the 2-D inventory example Nelder-Mead reaches the
optimum in 106 evaluations. When a usable gradient or surrogate exists,
PolyStep can need far more evaluations: in example 04 it used about 200x more
solver calls than a Fenchel-Young loss.

## Installation

```julia
using Pkg
Pkg.add("PolyStep")
# before registration: Pkg.add(url = "https://github.com/anindex/PolyStep.jl")
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

These settings reach 0 for every one of 100 rngs tried. The radius has to reach
past the flat pieces: with `step_radius = 0.5` every vertex costs 50, the
weights are uniform, and the iterate stays at `x0`.

Ask/tell, with bounds and integrality repair:

```julia
# integer points in [0, 10]^5; the best one is fill(3.0, 5) with cost 0.8
cost(X) = vec(sum(abs2, X .- 3.4; dims = 1))

es = PolyStepES(5; epsilon = 0.1, step_radius = 1.0, x0 = fill(9.0, 5),
                lb = 0.0, ub = 10.0, repair = X -> (X .= round.(X); X))
for _ in 1:100
    X = ask!(es)          # (dim, popsize) columns, clamped and repaired
    tell!(es, cost(X))
end
println((best_x = es.best_x, best_f = es.best_f))
# (best_x = [3.0, 3.0, 3.0, 3.0, 3.0], best_f = 0.7999999999999996)
```

The lower-level `PolyStepConfig` / `init_state` / `step!` / `solve!` API exposes
the full feature set: Sinkhorn / KL-softmax / tempered / greedy solvers, epsilon
schedules, decoupled probe and step radii, momentum, adaptive radius, biased
rotations, and a finite-difference quadratic model with Newton refinement and a
trust region. It also plugs into
[Optimization.jl](https://github.com/SciML/Optimization.jl) through
`PolyStepOptimizer`.

## Examples

Each example is a runnable script with a printed table and a pass/fail
assertion. In each one every method gets at most the same evaluation budget
(Fenchel-Young in 04 uses far fewer solver calls) and a tuning grid of the same
size over its step-size knobs, selected on the training objective only; the
grids are listed in the script headers. CMA-ES is
IPOP-CMA-ES with restarts until the budget is spent
([ipop_cma.jl](examples/ipop_cma.jl)). Run them with
`julia --project=examples examples/<script>.jl`, after
`julia --project=examples -e 'using Pkg; Pkg.instantiate()'` once (04 takes about
7 minutes with `-t 4`).

PolyStep leads in 01 and 04 and ties in 02 and 03. In 05 it has the lowest
train regret, but its test regret is not the lowest.

| # | Problem | Result |
|---|---|---|
| [01](examples/01_hard_decision_tree.jl) | Hard oblique decision tree, 0-1 loss, 40k-eval budget, 20 fresh checkerboards, every method grid-tuned | Median train/test acc: PolyStep 94.9/89.8% vs IPOP-CMA-ES 87.7/83.3%, OpenAI-ES 85.4/80.8%, SPSA 84.6/80.4% (PolyStep higher test acc on 15-17 of 20 instances) |
| [02](examples/02_inventory_sS.jl) | (s,S) inventory policy, 2-D, Poisson demand, common random numbers, 5 seeds, budgets 500 and 20000 evals, 9-point grids | Tie. PolyStep, IPOP-CMA-ES and Nelder-Mead all reach the grid optimum (PolyStep's worst gap is 0.0001%; the others print 0.0000%). Nelder-Mead needs only 106 evals. Tuned SPSA is at 0.0003% at 20000 but trails at 500 (9.92% mean gap, 48.94% worst seed). |
| [03](examples/03_districting_cmst.jl) | Contextual districting through an Esau-Williams capacitated-MST decoder, 8000 decoder calls, 5 seeds, 12-point grids | Tie. All three methods beat the true-cost decode (9.674) and stop on the same few plateaus. Means: PolyStep 9.549 +- 0.000 (100.8% gap closed), perturbed gradient + Adam 9.554 +- 0.011 (100.7%), IPOP-CMA-ES 9.584 +- 0.047 (100.5%). The means differ by less than 0.4%. |
| [04](examples/04_dfl_benchmarks.jl) | Decision-focused learning, subset selection (DecisionFocusedLearningBenchmarks), 625-weight linear model, hard top-k, seeds 1:3 | Test regret: PolyStep 2.43 vs InferOpt Fenchel-Young 3.51 vs IPOP-CMA-ES 4.02 (untrained 6.51), lowest on every seed. PolyStep and CMA-ES used 3.75M maximizer calls; FY used 18k, about 200x fewer. |
| [05](examples/05_predicted_weights_knapsack.jl) | DFL with predicted constraint weights, 0-1 knapsack, 10 seeds, 12k evals | Median train regret: PolyStep 0.119, the best, against CMA-ES 0.302, OpenAI-ES 0.384, SFGE 0.392 and random 3.409. Median test regret: CMA-ES 4.670, SFGE 4.903, PolyStep 5.164, OpenAI-ES 5.391, random 7.505. PolyStep has the lower test regret against each tuned baseline on only 4/10 seeds, so the test ranking is within noise. It beats random search on 10/10 seeds. |

## Where gradient-free fits best

In decision-focused learning with predicted *objective* coefficients, SPO+,
Fenchel-Young losses and perturbed optimizers apply, for example through
[InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl)
([Dalle et al., 2022](https://arxiv.org/abs/2207.13513)). In
[example 04](examples/04_dfl_benchmarks.jl) Fenchel-Young used about 200x fewer
solver calls than PolyStep, but it needs optimal-decision labels and a smoothing
layer. With predicted *constraint*
parameters there are also CombOptNet
([Paulus et al., ICML 2021](https://arxiv.org/abs/2105.02343)), Branch & Learn
([Hu, Lee and Lee, CPAIOR 2023](https://doi.org/10.1007/978-3-031-33271-5_18)),
two-stage Predict+Optimize
([Hu, Lee and Lee, NeurIPS 2023](https://arxiv.org/abs/2311.08022)), IntOpt-C
([Hu, Lee and Lee, AAAI 2023](https://doi.org/10.1609/aaai.v37i4.25513)) and
SFGE ([Silvestri et al., JAIR 2026](https://doi.org/10.1613/jair.1.19498)). These
need LP/ILP structure or a variance-controlled gradient estimator.

PolyStep needs only a black-box decision pipeline and optimizes the regret
directly, with no surrogate loss. Its finite probe radius `step_radius` is the
smoothing scale, so it still has to be tuned. It is one option among these, not
the only one that works: in [example 05](examples/05_predicted_weights_knapsack.jl)
(predicted knapsack weights) the pathwise gradient is zero, yet tuned SFGE and
OpenAI-ES learn, and CMA-ES and SFGE have lower median test regret than PolyStep.
[Example 01](examples/01_hard_decision_tree.jl) (a hard oblique decision tree
under 0-1 loss) has the same kind of loss, piecewise constant with a zero
gradient almost everywhere, without the DFL framing.

## Related methods

As `epsilon` goes to 0 the softmax weights pick the best vertex, so a step polls
the 2d directions `+-step_radius * R[:, i]` of a Haar-random orthonormal basis
and moves to the best one. This is close to OrthoMADS polling
([Abramson et al., SIAM J. Optim. 2009](https://doi.org/10.1137/080716980)), an
instance of MADS
([Audet and Dennis, SIAM J. Optim. 2006](https://doi.org/10.1137/040603371)), and
to direct search based on probabilistic descent
([Gratton et al., SIAM J. Optim. 2015](https://doi.org/10.1137/140961602)). Unlike
those methods it moves even when no vertex improves on the current point; the
best evaluated point is tracked separately in `best_x`. At larger `epsilon` the
weights become close to linear in the costs and the mirrored vertex pairs give
central differences, so the expected step over the random rotation is a
gradient step on the objective averaged over a ball of radius `step_radius`.
`epsilon` interpolates between the two. Mirrored orthogonal sampling has also
been used in CMA-ES
([Wang, Emmerich and Baeck, Evol. Comput. 2019](https://doi.org/10.1162/evco_a_00251)).
The quadratic model, trust region and radius controllers in `PolyStepConfig` are
heuristics and carry no convergence guarantee.

## Notes

- Column-major throughout: candidates are `(d, N)` columns and cost/plan
  matrices are `(V, P)`, so per-particle reductions stay contiguous. Hot kernels
  are zero-allocation in steady state.
- Float64 by default (what OR closures expect). `T = Float32` in
  `PolyStepES`/`minimize`, or a Float32 starting matrix in `init_state`, is the
  performance/GPU mode.
- Rotations come from one serial RNG, so results are bit-identical for any
  thread count under the default `--check-bounds` setting (forcing
  `--check-bounds=yes`, as `Pkg.test` does, changes the `@simd` summation order
  in the threaded kernels). The large kernels use Polyester `@batch`, whose threads
  spin-wait. When running many optimizations under `Threads.@threads`, or on an
  oversubscribed node, disable them (add Polyester to your environment):

  ```julia
  using Polyester
  best = zeros(8)
  Polyester.disable_polyester_threads() do
      Threads.@threads for s in 1:8
          best[s] = minimize(f, 10; steps = 100, epsilon = 0.1, step_radius = 1.0,
                             x0 = fill(5.5, 10), rng = Xoshiro(s)).best_f
      end
  end
  println(best)   # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  ```

- `step_radius` is absolute in `PolyStepES`/`minimize`: candidates are
  `x +- step_radius * R[:, i]`. In `PolyStepConfig` an unscheduled radius is
  multiplied by `epsilon`, so pass `r / epsilon` or a schedule there for the
  same step. With a fixed radius the iterate keeps moving by a fixed fraction of
  `step_radius` near a minimizer (set by `epsilon` and `dim`, not by the distance
  to the minimizer); to shrink it, mutate it from a callback, e.g.
  `minimize(f, d; callback = es -> (es.step_radius *= 0.99; false))`. Examples
  02 and 05 use a geometric decay.
- With the default `scale_cost = :mean` the costs are shifted to a minimum of 0 and
  divided by their mean every round, so `epsilon` is relative to the cost spread;
  at large `epsilon` the step then shrinks about as `1/sqrt(dim)`. In high
  dimensions and in a `HybridSubspace`, lower `epsilon` if the steps stall. With a
  fixed divisor (`scale_cost = 1.0`, the `PolyStepConfig` default) the step shrinks
  as `1/dim`; there, lower the temperature through `ent_epsilon`, since `epsilon`
  also scales both radii.
- `HybridSubspace` runs PolyStep in a fixed per-layer orthonormal basis of a
  layered model's parameters (a Julia version of the Python `HybridSubspace`):

  ```julia
  layout = ParamLayout(["W1" => (32, 16), "b1" => (32,), "W2" => (3, 32)])
  s = HybridSubspace(layout; rank = 4)
  println(s)   # HybridSubspace(3 layers, dim=320/640, rank=4)
  theta0 = randn(Xoshiro(1), s.total_params)
  loss(Theta) = vec(sum(abs, Theta; dims = 1))   # (640, N) parameter columns -> N
  g = subspace_objective(loss, s, theta0)        # (320, N) coordinates -> N
  es = minimize(g, subspace_dim(s); steps = 200, epsilon = 0.1, step_radius = 1.0,
                rng = Xoshiro(0))
  theta = theta0 .+ expand(s, es.best_x)
  println((before = loss(reshape(theta0, :, 1))[1], after = es.best_f))
  # (before = 533.488788647366, after = 329.8953944084636)
  ```

- One heuristic still differs from the Python reference (see
  [CHANGELOG.md](CHANGELOG.md)): in `solve!`, convergence also needs a small
  step relative to the running peak displacement.
- Optional extensions: LoopVectorization (`@turbo` kernels), CUDA
  (`cuda_objective` wraps a GPU-batched objective), and Optimization.jl
  (`PolyStepOptimizer`; pass `batched = f_batch` to `solve` to keep vectorized
  evaluation).

## Citation

If you use PolyStep.jl, please cite the paper (see also
[CITATION.cff](CITATION.cff)):

```bibtex
@article{le2026training,
  title   = {Training Non-Differentiable Networks via Optimal Transport},
  author  = {Le, An T.},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.01928}
}
```
