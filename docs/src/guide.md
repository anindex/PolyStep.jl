# Guide

## How a step works

For each particle `x` (a column of `X`), one step:

1. draws a Haar-random rotation `R` and places an orthoplex around `x`, so the
   candidates are `x +- h * R[:, i]` for `i = 1..d` (`2d` mirrored orthogonal probes);
2. evaluates all candidates of all particles in one batched call `f(::Matrix)::Vector`
   (wrap a scalar objective with `columnwise(f)`, or `columnwise(f; parallel = :threads)`);
3. turns the costs into weights `w = softmax(-C / epsilon)` after `scale_cost`
   (or an entropic transport plan, see [Solvers](@ref));
4. moves `x` to the weighted barycenter of the candidates.

`epsilon` interpolates between two known methods. As `epsilon -> 0` the weights pick the
best vertex, so a step polls `2d` orthogonal directions and moves to the best one, like
OrthoMADS polling ([Abramson et al., 2009](https://doi.org/10.1137/080716980)) or direct
search based on probabilistic descent
([Gratton et al., 2015](https://doi.org/10.1137/140961602)). Unlike those, the iterate
moves even when no vertex improves; the best evaluated point is kept in `best_x`. At
large `epsilon` the weights are close to linear in the costs and the mirrored pairs give
central differences, so the expected step is a gradient step on `f` averaged over a ball
of radius `h`. The radius is the smoothing scale: it has to reach past the flat pieces of
a piecewise-constant loss.

In `PolyStepConfig`, when vertex cost differences are small next to the temperature, one
step is gradient descent with rate `sr * (pr/2) / (dim * eps_ot * s)` (`s` is the
`scale_cost` divisor, `pr/2` the default probe distance, `sr` the step radius).

## Two APIs

| | `minimize` / `PolyStepES` | `PolyStepConfig` / `step!` / `solve!` |
|---|---|---|
| Radius | `step_radius`, absolute | `step_radius` and `probe_radius`, multiplied by `epsilon` unless scheduled |
| Default `scale_cost` | `:mean` | `1.0` |
| Candidates | evaluated points are the barycenter points (clamped and repaired before caching) | probes are clamped/repaired; the iterate is a barycenter of the raw template |
| Features | softmax or any solver, bounds, repair, restarts via callbacks | schedules, momentum, adaptive radius, biased rotations, quadratic model, Newton refinement, trust region |

Both track `best_x`/`best_f` over every evaluated point and score the final iterate
once at the end (`num_particles` extra evaluations).

## Choosing epsilon and radii

- **Radius first.** With `step_radius` below the width of the flat pieces every vertex
  costs the same, the weights are uniform, and the iterate does not move.
- **Fixed radius, fixed motion.** With a fixed radius the iterate keeps moving by a
  fixed fraction of `step_radius` near a minimizer. Shrink it from a callback,
  `minimize(f, d; callback = es -> (es.step_radius *= 0.99; false))`, or use a schedule
  in `PolyStepConfig`.
- **Cost scaling.** `scale_cost = :mean` shifts the costs to a minimum of 0 and divides
  by their mean each round, so `epsilon` is relative to the cost spread and invariant to
  a constant offset. At large `epsilon` the step then shrinks about as `1/sqrt(dim)`; with
  a fixed divisor it shrinks as `1/dim`. In high dimension lower the temperature: `epsilon`
  in `PolyStepES`, `ent_epsilon` in `PolyStepConfig` (its `epsilon` also scales both
  radii, so the step would shrink as `1/dim^2`).

## Bounds, repair and integrality

`lb`/`ub` (scalar or `(dim,)`, `+-Inf` allowed) clamp every evaluated candidate.
`repair(X)` post-processes candidates in place after clamping, for example
`X -> (X .= round.(X); X)` for integer variables; it must preserve the bounds itself,
since its output is not re-clamped. In `PolyStepES` the evaluated and cached candidates
are the same points, so the barycenter stays in the box and `best_x` is always feasible.
In `PolyStepConfig` the iterate is box-projected after each step but not repaired.

## Reproducibility and threads

Rotations come from one serial RNG and every threaded kernel works on independent
slices, so seeded results are bit-identical for any thread count under the default
`--check-bounds` setting (`--check-bounds=yes`, as `Pkg.test` uses, changes `@simd`
summation order). Rotations for `d >= 192` use LAPACK with BLAS pinned to one thread;
smaller ones run in pure Julia.

The large kernels use Polyester `@batch`, whose threads spin-wait. When running many
optimizations under `Threads.@threads`, disable them (add Polyester to your environment):

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

`splitmix64(seed, i...)` derives stable per-run seeds.

## Solvers

| Solver | Weights |
|---|---|
| `SoftmaxSolver()` (default) | `softmax(-C/eps)` per particle |
| `TemperedSoftmaxSolver(tau)` | softmax at a fixed temperature, ignoring `eps` |
| `SinkhornSolver()` | entropic OT with uniform marginals over particles and vertices (log-domain, SOR, Anderson, warm starts) |
| `KLSoftmaxSolver(lam)` | KL-relaxed column marginal: `lam = 0` is softmax, `lam = Inf` is Sinkhorn |
| `MinCostGreedySolver()`, `TopKMeanSolver(k)` | ablations: best vertex, mean of the `k` best |

Two-sided solvers need several particles (with one, the plan is uniform). Each particle
has its own rotation, so a vertex index is not a common direction across particles: the
Sinkhorn column marginal spreads mass over vertex slots, not over directions.

## Subspaces

`HybridSubspace` runs PolyStep in a fixed per-layer orthonormal basis of a layered
model's parameters, as the Python `HybridSubspace` does:

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

Bases are dense (`numel * ncoords` entries per layer); pass `T = Float32` to halve memory.

## Extensions

- **LoopVectorization**: `@turbo` log-sum-exp kernels for the Sinkhorn/KL solvers.
- **CUDA**: `cuda_objective(f_gpu)` wraps a GPU-batched objective; CUBLAS batched
  rotation products.
- **Optimization.jl** (via OptimizationBase): `PolyStepOptimizer()`. Pass
  `batched = f_batch` to `solve` to keep vectorized evaluation; otherwise candidates are
  evaluated one at a time.

## Decision-focused learning

With predicted *objective* coefficients, SPO+, Fenchel-Young losses and perturbed
optimizers apply, for example through
[InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl)
([Dalle et al., 2022](https://arxiv.org/abs/2207.13513)); in example 04 Fenchel-Young
used about 200x fewer solver calls than PolyStep. With predicted *constraint* parameters
there are CombOptNet ([Paulus et al., 2021](https://arxiv.org/abs/2105.02343)), Branch &
Learn ([Hu, Lee and Lee, 2023](https://doi.org/10.1007/978-3-031-33271-5_18)), two-stage
Predict+Optimize ([Hu, Lee and Lee, 2023](https://arxiv.org/abs/2311.08022)) and SFGE
([Silvestri et al., 2026](https://doi.org/10.1613/jair.1.19498)), which need LP/ILP
structure or a variance-controlled gradient estimator. PolyStep needs only the black-box
pipeline and optimizes the regret directly; its radius is the smoothing scale and still
has to be tuned. In example 05 the pathwise gradient is zero, yet tuned SFGE and
OpenAI-ES learn too.

## Related methods

- Direct search: MADS ([Audet and Dennis, 2006](https://doi.org/10.1137/040603371)),
  OrthoMADS, probabilistic descent (see above).
- Orthogonal and mirrored sampling in evolution strategies:
  [Choromanski et al., ICML 2018](https://arxiv.org/abs/1804.02395);
  [Wang, Emmerich and Baeck, 2019](https://doi.org/10.1162/evco_a_00251).
- Consensus-based optimization weights sampled points by `exp(-f/eps)` too, but pulls
  all particles toward one global weighted mean
  ([Pinnau et al., 2017](https://doi.org/10.1142/S0218202517400061);
  [ConsensusBasedX.jl](https://arxiv.org/abs/2403.14470)). PolyStep applies the
  weighting locally, to each particle's own probes.
- Haar rotations for `d > 8` follow Stewart's reflector construction
  ([Stewart, 1980](https://doi.org/10.1137/0717034)) with Mezzadri's sign fix.
- The OT step generalizes the Sinkhorn Step of
  [Le et al., NeurIPS 2023](https://arxiv.org/abs/2309.15970).

The quadratic model, trust region and radius controllers in `PolyStepConfig` are
heuristics without a convergence guarantee.

## Differences from the Python reference

- Convergence in `solve!` also needs the last displacement to be small relative to the
  running peak (Python compares with the first displacement).
- Divergence means every probe in a step is non-finite; Python stops on any non-finite
  probe, which ends runs whose objective returns `Inf` for infeasible points.
- The quadratic model requires the orthoplex and `num_probe >= 2`; the trust region
  scores the model's center value instead of an extra evaluation of `f(X)`.
- `HybridSubspace` bases are dense; there is no sparse fallback.
