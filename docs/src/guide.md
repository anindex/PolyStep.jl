# Guide

## How a step works

A step does four operations on each particle `x` (a column of `X`):

1. It draws a random rotation `R` and puts a polytope around `x`. The default orthoplex
   gives the `2d` candidates `x +- h * R[:, i]`, which are mirrored orthogonal pairs.
   In `PolyStepConfig`, `:simplex` gives `d + 1` candidates and `:cube` gives
   `2^d` (`d <= 6`).
2. It evaluates all candidates of all particles in one call `f(X::Matrix)::Vector`. To
   use a scalar objective, wrap it with `columnwise(f)` or
   `columnwise(f; parallel = :threads)`.
3. It converts the costs into weights. The default is `softmax(-C / epsilon)` after
   `scale_cost`. An entropic transport plan is also available (see Solvers below).
4. It moves `x` to the weighted average of its candidates.

`epsilon` sets the behavior between two known methods. Near 0, the weights select the
best vertex. A step then polls `2d` orthogonal directions and moves to the best one.
OrthoMADS ([Abramson et al., 2009](https://doi.org/10.1137/080716980)) and probabilistic
direct search ([Gratton et al., 2015](https://doi.org/10.1137/140961602)) use the same poll.
Unlike these methods, PolyStep also moves when no vertex is better; `best_x` keeps the
best point found. For a large `epsilon`, the weights are almost linear in the costs and
the mirrored pairs act as central differences. The expected step is then a gradient step
on `f` smoothed over a ball of radius `h`. This radius must be larger than the flat
regions of the loss.

## Two APIs

| | `minimize` / `PolyStepES` | `PolyStepConfig` / `step!` / `solve!` |
|:--|:--|:--|
| Radius | `step_radius`, absolute, multiplied by `radius_decay` after each round | `step_radius` and `probe_radius`, multiplied by `epsilon` unless scheduled |
| Default `scale_cost` | `:mean` | `1.0` |
| Candidates | the evaluated points are the averaged points | probes are clamped and repaired; the iterate averages the raw template |
| Options | all solvers, bounds, `repair`, callback | also schedules, momentum, adaptive radius, biased rotations, quadratic model, Newton refinement, trust region |

`minimize` and `solve!` evaluate the final iterate once at the end (`num_particles`
evaluations). A manual `ask!`/`tell!` loop does not.

## Choosing epsilon and radii

Set the radius first. If the radius is smaller than the flat regions of the loss, all
vertices have the same cost. The weights are then uniform, and the particles do not move.

With a fixed radius, the iterate continues to move by a fraction of `step_radius` near
a minimizer. To decrease the radius, set `radius_decay` (for example `0.99` per round),
change `es.step_radius` in a `minimize` callback, or use a schedule in `PolyStepConfig`:

```@example decay
using PolyStep, Random
sphere(X) = vec(sum(abs2, X; dims = 1))
fixed = minimize(sphere, 10; steps = 300, x0 = fill(3.0, 10), rng = Xoshiro(1))
decay = minimize(sphere, 10; steps = 300, x0 = fill(3.0, 10), radius_decay = 0.99,
                 rng = Xoshiro(1))
(fixed.best_f, decay.best_f)
```

With `scale_cost = :mean`, each round shifts the costs to a minimum of 0 and divides them
by their mean. Thus `epsilon` is relative to the spread of the costs, and a constant
offset in `f` has no effect. Non-finite costs, for example from an infeasible simulation,
rank last and are not included in the mean. At a large `epsilon`, the step decreases
approximately as `1/sqrt(dim)` (as `1/dim` with a fixed divisor). In high dimension,
decrease the temperature: `epsilon` in `PolyStepES`, `ent_epsilon` in `PolyStepConfig`.
In `PolyStepConfig`, `epsilon` also multiplies both radii. If `epsilon` decreases as
`1/dim`, the step decreases as `1/dim^2`.

## Bounds, repair and integers

`lb` and `ub` clamp all evaluated candidates. They are scalars or `(dim,)` vectors, and
`+-Inf` is permitted. Then `repair(X)` corrects the candidates, in place or by returning
the corrected matrix. For integer variables, use `X -> round.(X)`. The package does not
clamp the output of `repair` again, so `repair` must keep the bounds. In `PolyStepES`,
the evaluated points are also the averaged points. Thus the iterate stays in the box and
`best_x` is always feasible. In `PolyStepConfig`, each step projects the iterate onto
the box but does not repair it.

## Solvers

| Solver | Weights for each particle |
|:--|:--|
| `SoftmaxSolver()` (default) | `softmax(-C / eps)` |
| `SinkhornSolver()` | entropic optimal transport; also balances the mass of each vertex slot over the particles |
| `KLSoftmaxSolver(lam = Inf)` | relaxed vertex marginal: `lam = 0` is softmax, `lam = Inf` is Sinkhorn |
| `MinCostGreedySolver()` | all weight on the best vertex |
| `TopKMeanSolver(k = 3)` | equal weight on the `k` best vertices |

Sinkhorn stops when the relative marginal error is less than `threshold = 3e-5`. KL stops
when the dual step is less than `threshold`. Both stop early, without convergence, when
the error stops decreasing at the floating-point precision. Sinkhorn and
`KLSoftmaxSolver(lam = Inf)` need more than one particle. With one particle, the plan is
uniform. Each particle has its own
rotation, so the Sinkhorn vertex marginal balances vertex slots, not directions. The
solvers run on CPU arrays.

## PolyStepConfig

`PolyStepConfig` gives access to all parts of the step. [`PolyStepConfig`](@ref) lists
its keywords and defaults. `init_state` takes a `(dim, P)` matrix of start particles.
Its element type sets the working precision.

```@example config
using PolyStep, Random
f(X) = vec(sum(floor.(abs.(X)); dims = 1))
ps = PolyStepConfig(dim = 10, solver = SinkhornSolver(), epsilon = 0.1,
                    step_radius = 10.0, probe_radius = 20.0, max_iterations = 200)
st = init_state(ps, repeat(fill(5.5, 10), 1, 4))
solve!(f, ps, st; rng = Xoshiro(0))
(st.best_f, st.evals, st.iteration)
```

Unscheduled radii are multiplied by `epsilon`, so this run uses a step and a probe
distance of 1.0. `epsilon`, `ent_epsilon`, `step_radius` and `probe_radius` also accept
schedules: `LinearEpsilon` and `CosineEpsilon` (with optional warm restarts).
`ProgressiveEpsilon` controls the OT temperature of an iterative solver. It increases
the temperature when the solver converges slowly and decreases it when the solver
converges fast.

`biased_rotation = true` turns each frame so that one vertex points along the last
descent direction. `use_quadratic_model = true` fits a finite-difference gradient and a
diagonal Hessian from the probes. It needs the orthoplex and `num_probe >= 2`. With this
model, the descent direction is the negative model gradient. `newton_refinement` adds a
clamped Newton step to the transport target. `trust_region` scales the step by the ratio
of actual to predicted improvement. Both options need the quadratic model. The
trust region measures the actual improvement with one more evaluation of `f(X)` for each
particle in each step. These controllers are heuristics and have no convergence guarantee.

## Subspaces

`HybridSubspace` runs PolyStep in a fixed orthonormal basis for each parameter array of
a model, as the Python `HybridSubspace` does:

```@example subspace
using PolyStep, Random
layout = ParamLayout(["W1" => (32, 16), "b1" => (32,), "W2" => (3, 32)])
s = HybridSubspace(layout; rank = 4)
theta0 = randn(Xoshiro(1), s.total_params)
loss(Theta) = vec(sum(abs, Theta; dims = 1))
g = subspace_objective(loss, s, theta0)
es = minimize(g, subspace_dim(s); steps = 200, epsilon = 0.1, step_radius = 1.0,
              rng = Xoshiro(0))
theta = theta0 .+ expand(s, es.best_x)
(s, loss(reshape(theta0, :, 1))[1], es.best_f)
```

`max_subspace_dim` limits the total dimension. It decreases the coordinates of all weight
arrays. 1-D arrays keep all their coordinates. The bases are dense, with `numel * ncoords`
numbers for each array. `T = Float32` uses half the memory.

## Optimization.jl

`PolyStepOptimizer` is available when Optimization.jl (or OptimizationBase) is loaded.
Give `batched` to evaluate each round in one call:

```julia
using Optimization, PolyStep, Random
f(u, p) = sum(floor.(abs.(u)))
prob = OptimizationProblem(f, fill(5.5, 10))
sol = solve(prob, PolyStepOptimizer(epsilon = 0.1, step_radius = 1.0); maxiters = 100,
            batched = X -> vec(sum(floor.(abs.(X)); dims = 1)), rng = Xoshiro(0))
(sol.objective, sol.retcode, sol.stats.fevals)   # (0.0, Success, 2001)
```

`maxiters` counts rounds, and the optimizer checks `maxtime` after each round. `lb` and
`ub` clamp the candidates, and `MaxSense` is supported. The return code is `Success`,
`Terminated`, `MaxTime` or `Failure`.

## Reproducibility and threads

All random numbers come from one serial RNG, and the threaded kernels operate on
independent slices. Thus a seeded run gives the same bits on any number of threads. Runs
under `--check-bounds=yes`, which `Pkg.test` uses, give other bits than normal runs,
because the option changes the order of `@simd` sums. Rotations with `d >= 192` use LAPACK
with BLAS set to one thread. Smaller rotations run in pure Julia.

The large kernels use Polyester `@batch`, whose threads spin while they wait. To run
many optimizations under `Threads.@threads`, disable them (Polyester must be in your
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

`splitmix64(seed, i...)` gives stable seeds for each run.

## Extensions

- LoopVectorization: the Sinkhorn and KL log-sum-exp kernels use `@turbo`.
- CUDA: `PolyStep.cuda_objective(f_gpu; T = Float32)` wraps a GPU objective
  `f_gpu(X::CuMatrix)` for the batched protocol. The optimizer stays on the CPU.
- Optimization.jl or OptimizationBase: `PolyStepOptimizer` (see above).

## Related methods

- Direct search: MADS ([Audet and Dennis, 2006](https://doi.org/10.1137/040603371)),
  OrthoMADS and probabilistic descent (see above). For integer variables with a
  convergence guarantee, use NOMAD.
- Orthogonal and mirrored sampling in evolution strategies
  ([Choromanski et al., 2018](https://arxiv.org/abs/1804.02395);
  [Wang, Emmerich and Baeck, 2019](https://doi.org/10.1162/evco_a_00251)).
- Consensus-based optimization also weights points by `exp(-f/eps)`, but it moves all
  particles to one weighted mean ([Pinnau et al., 2017](https://doi.org/10.1142/S0218202517400061)).
- The OT step generalizes the Sinkhorn Step of [Le et al., 2023](https://arxiv.org/abs/2309.15970).
  Rotations for `d > 8` use the reflectors of [Stewart (1980)](https://doi.org/10.1137/0717034)
  with the sign correction of Mezzadri.
- Decision-focused learning: when a model predicts objective coefficients, Fenchel-Young
  losses and perturbed optimizers ([InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl))
  need far fewer solver calls (approximately 200 times fewer in example 04). PolyStep
  needs only the black-box pipeline. This is useful when the prediction is in the
  constraints ([Hu, Lee and Lee, 2023](https://arxiv.org/abs/2311.08022)).

## Differences from the Python version

The Julia package follows the Python optimizer at v1.1.1 for standalone objectives. It
has no neural-network training loop, no `FactoredSubspace` and no `AdaptiveSubspace`.
Other differences:

- The default polytope is the orthoplex and the default solver is softmax. The Python
  optimizer uses the simplex and, with more than one particle, Sinkhorn.
- `solve!` compares the last step with the largest step so far. Python compares it with
  the first step.
- The quadratic model needs the orthoplex and `num_probe >= 2`. The trust region uses
  the value `f(X)` of each particle.
- `HybridSubspace` bases are always dense, and the subspace is one polytope, not 8-D blocks.
