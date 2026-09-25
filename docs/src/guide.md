# Guide

## How a step works

Each particle `x` (a column of `X`) goes through four stages per step:

1. Draw a random rotation `R` and put an orthoplex around `x`. The candidates are
   `x +- h * R[:, i]` for `i = 1..d`, so `2d` probes in mirrored orthogonal pairs.
2. Evaluate every candidate of every particle in one call `f(::Matrix)::Vector`. A scalar
   objective can be wrapped with `columnwise(f)`, or `columnwise(f; parallel = :threads)`.
3. Turn the costs into weights, `w = softmax(-C / epsilon)` after `scale_cost`, or use an
   entropic transport plan (see [Solvers](@ref)).
4. Move `x` to the weighted average of its candidates.

`epsilon` slides between two familiar methods. Near 0 the weights pick the best vertex,
so a step polls `2d` orthogonal directions and jumps to the best one. That is close to
OrthoMADS ([Abramson et al., 2009](https://doi.org/10.1137/080716980)) and to direct
search with probabilistic descent ([Gratton et al., 2015](https://doi.org/10.1137/140961602)),
except that PolyStep moves even when no vertex improves; the best point seen is kept in
`best_x`. For large `epsilon` the weights become almost linear in the costs, the mirrored
pairs act as central differences, and the expected step is a gradient step on `f`
smoothed over a ball of radius `h`. That radius is the smoothing scale, and it has to
reach past the flat pieces of the loss.

In `PolyStepConfig`, when the cost differences are small next to the temperature, a step
is gradient descent with rate `sr * (pr/2) / (dim * eps_ot * s)`, where `sr` is the step
radius, `pr/2` the default probe distance and `s` the `scale_cost` divisor.

## Two APIs

| | `minimize` / `PolyStepES` | `PolyStepConfig` / `step!` / `solve!` |
|---|---|---|
| Radius | `step_radius`, absolute | `step_radius` and `probe_radius`, multiplied by `epsilon` unless scheduled |
| Default `scale_cost` | `:mean` | `1.0` |
| Candidates | the evaluated points are the averaged points (clamped and repaired first) | probes are clamped and repaired; the iterate averages the raw template |
| Extras | any solver, bounds, repair, callbacks | schedules, momentum, adaptive radius, biased rotations, quadratic model, Newton refinement, trust region |

Both keep the best evaluated point in `best_x`/`best_f` and score the final iterate once
at the end, which costs `num_particles` extra evaluations.

## Choosing epsilon and radii

Pick the radius first. If `step_radius` is smaller than the flat pieces of the loss,
every vertex costs the same, the weights are uniform, and nothing moves.

A fixed radius means a fixed amount of motion: near a minimizer the iterate keeps
jumping by a fraction of `step_radius`. Shrink it from a callback,
`minimize(f, d; callback = es -> (es.step_radius *= 0.99; false))`, or use a schedule in
`PolyStepConfig`.

With `scale_cost = :mean` the costs are shifted to a minimum of 0 and divided by their
mean every round. `epsilon` then measures the spread of the costs, and a constant offset
in `f` changes nothing. At large `epsilon` the step shrinks roughly like `1/sqrt(dim)`
(like `1/dim` with a fixed divisor), so in high dimension lower the temperature:
`epsilon` in `PolyStepES`, `ent_epsilon` in `PolyStepConfig`. There, `epsilon` also
scales both radii, and lowering it would shrink the step like `1/dim^2`.

## Bounds, repair and integers

`lb` and `ub` (scalars or `(dim,)` vectors, `+-Inf` allowed) clamp every evaluated
candidate. `repair(X)` then edits the candidates in place, for example
`X -> (X .= round.(X); X)` for integer variables. Its output is not clamped again, so it
has to respect the bounds itself. In `PolyStepES` the evaluated points and the averaged
points are the same, so the iterate stays in the box and `best_x` is always feasible. In
`PolyStepConfig` the iterate is projected onto the box after each step but not repaired.

## Reproducibility and threads

All random numbers come from one serial RNG and the threaded kernels work on independent
slices, so a seeded run gives the same bits on any number of threads. One exception:
`--check-bounds=yes`, which `Pkg.test` uses, changes the `@simd` summation order.
Rotations with `d >= 192` use LAPACK with BLAS pinned to one thread; smaller ones run in
pure Julia.

The large kernels use Polyester `@batch`, whose threads spin while they wait. If you run
many optimizations under `Threads.@threads`, turn them off (Polyester has to be in your
environment):

```julia
using Polyester, PolyStep, Random
f(X) = vec(sum(floor.(abs.(X)); dims = 1))
best = zeros(8)
Polyester.disable_polyester_threads() do
    Threads.@threads for s in 1:8
        best[s] = minimize(f, 10; steps = 100, epsilon = 0.1, step_radius = 1.0,
                           x0 = fill(5.5, 10), rng = Xoshiro(s)).best_f
    end
end
```

`splitmix64(seed, i...)` gives stable per-run seeds.

## Solvers

| Solver | Weights |
|---|---|
| `SoftmaxSolver()` (default) | `softmax(-C/eps)` per particle |
| `TemperedSoftmaxSolver(tau)` | softmax at a fixed temperature, ignoring `eps` |
| `SinkhornSolver()` | entropic OT with uniform marginals over particles and vertices |
| `KLSoftmaxSolver(lam)` | relaxed column marginal: `lam = 0` is softmax, `lam = Inf` is Sinkhorn |
| `MinCostGreedySolver()`, `TopKMeanSolver(k)` | ablations: the best vertex, or the mean of the `k` best |

The two-sided solvers need more than one particle; with a single particle the plan is
uniform. Each particle also has its own rotation, so vertex `v` points in a different
direction for each particle. The Sinkhorn column marginal therefore spreads mass over
vertex slots, not over directions.

## Subspaces

`HybridSubspace` runs PolyStep in a fixed orthonormal basis per layer of a model, like
the Python `HybridSubspace`:

```julia
using PolyStep, Random
layout = ParamLayout(["W1" => (32, 16), "b1" => (32,), "W2" => (3, 32)])
s = HybridSubspace(layout; rank = 4)          # HybridSubspace(3 layers, dim=320/640, rank=4)
theta0 = randn(Xoshiro(1), s.total_params)
loss(Theta) = vec(sum(abs, Theta; dims = 1))  # (640, N) parameter columns -> N
g = subspace_objective(loss, s, theta0)       # (320, N) coordinates -> N
es = minimize(g, subspace_dim(s); steps = 200, epsilon = 0.1, step_radius = 1.0,
              rng = Xoshiro(0))
theta = theta0 .+ expand(s, es.best_x)
```

The bases are dense (`numel * ncoords` numbers per layer). `T = Float32` halves the memory.

## Extensions

Loading LoopVectorization switches the Sinkhorn/KL log-sum-exp kernels to `@turbo`.
Loading CUDA adds `cuda_objective(f_gpu)` for GPU-batched objectives and CUBLAS batched
rotation products. Loading Optimization.jl (OptimizationBase) enables
`PolyStepOptimizer()`; pass `batched = f_batch` to `solve`, otherwise candidates are
evaluated one at a time.

## Decision-focused learning

When the model predicts *objective* coefficients, SPO+, Fenchel-Young losses and
perturbed optimizers work well, for example through
[InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl)
([Dalle et al., 2022](https://arxiv.org/abs/2207.13513)). In example 04, Fenchel-Young
used about 200x fewer solver calls than PolyStep. When it predicts *constraint*
parameters, the options are CombOptNet ([Paulus et al., 2021](https://arxiv.org/abs/2105.02343)),
Branch & Learn ([Hu, Lee and Lee, 2023](https://doi.org/10.1007/978-3-031-33271-5_18)),
two-stage Predict+Optimize ([Hu, Lee and Lee, 2023](https://arxiv.org/abs/2311.08022)) and
SFGE ([Silvestri et al., 2026](https://doi.org/10.1613/jair.1.19498)). These need LP/ILP
structure or a low-variance gradient estimator. PolyStep only needs the black-box
pipeline and minimizes the regret directly, but its radius still has to be tuned. In
example 05 the pathwise gradient is zero, and tuned SFGE and OpenAI-ES still learn.

## Related methods

- Direct search: MADS ([Audet and Dennis, 2006](https://doi.org/10.1137/040603371)),
  OrthoMADS and probabilistic descent (above).
- Orthogonal and mirrored sampling in evolution strategies:
  [Choromanski et al., ICML 2018](https://arxiv.org/abs/1804.02395) and
  [Wang, Emmerich and Baeck, 2019](https://doi.org/10.1162/evco_a_00251).
- Consensus-based optimization also weights points by `exp(-f/eps)`, but pulls every
  particle toward one global weighted mean
  ([Pinnau et al., 2017](https://doi.org/10.1142/S0218202517400061);
  [ConsensusBasedX.jl](https://arxiv.org/abs/2403.14470)). PolyStep weights each
  particle's own probes.
- Rotations for `d > 8` use Stewart's reflector construction
  ([Stewart, 1980](https://doi.org/10.1137/0717034)) with Mezzadri's sign fix.
- The OT step generalizes the Sinkhorn Step of
  [Le et al., NeurIPS 2023](https://arxiv.org/abs/2309.15970).

The quadratic model, trust region and radius controllers in `PolyStepConfig` are
heuristics and come with no convergence guarantee.

## Differences from the Python version

- `solve!` also requires the last step to be small compared with the largest step so
  far; Python compares with the first step.
- A step counts as diverged only when every probe is non-finite. Python stops at the
  first non-finite probe, which ends runs whose objective returns `Inf` for infeasible
  points.
- The quadratic model needs the orthoplex and `num_probe >= 2`, and the trust region
  uses the model's center value instead of an extra evaluation of `f(X)`.
- `HybridSubspace` bases are always dense.
