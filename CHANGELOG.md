# Changelog

## v0.2.0

New:

- `ParamLayout` and `HybridSubspace`: a fixed per-layer orthonormal basis for
  parameter vectors of layered models, with per-layer rank, a total-dimension
  budget (`max_subspace_dim`), `project`/`expand` between the full vector and the
  subspace coordinates, and `subspace_objective` to run any PolyStep solver in the
  subspace. The dimension arithmetic and the cap semantics mirror the Python
  `HybridSubspace` (a cap of 0 is a real cap); the package tests assert them against
  the Python reference for four layouts, five ranks and five caps. N-D arrays use the
  last axis as `d_out`, so Flux/Lux conv weights are projected. Bases are dense per
  layer (no sparse fallback). `reconstruct_batch` runs one GEMM per layer (8-14x
  faster than per-column products).

Behavior changes (these follow the Python reference v0.13; seeded results change):

- `scale_cost = :mean`/`:max` subtract the minimum cost before scaling, so the
  temperature follows the spread of the costs, not their level. This is the default
  for `PolyStepES`, `minimize` and `PolyStepOptimizer`; before, a constant offset
  on the objective changed the run, and default runs could stall far from the optimum.
- Non-finite costs get the penalty `2*max|finite| + 1` (no `1e6` floor), and
  `MinCostGreedySolver`/`TopKMeanSolver` sanitize their costs too.
- `CosineEpsilon` uses `ceil` for its default horizon, and each restart period grows
  by at least one step. `solve!` checks convergence from `min_iterations` on.
- The trust-region ratio scores the realized move, after momentum, Newton
  refinement and the bounds clamp, as in Python (v0.1.0 scored the transport step).
  The one remaining deviation: convergence also needs a small step relative to the
  running peak displacement.
- Sinkhorn `adaptive_omega` estimates omega only while `omega == 1`, and the
  Anderson history resets when omega changes (fewer iterations, no failed solves).
- `best_x`/`best_f` now include the final iterate: `minimize`, `solve!` and
  `PolyStepOptimizer` score it once at the end (`num_particles` extra evaluations).
- `KLSoftmaxSolver` checks convergence every iteration (Python checks every
  `max_iterations/20` to limit GPU syncs; this solver runs on the CPU), re-fits its
  row duals before building the plan, and reports the generalized KL marginal
  violation. Sinkhorn and KL `ent_cost` are reported in the caller's cost frame.

Fixes:

- The finite-difference Hessian no longer collapses for probe radii below about
  `6e-3`, which the default schedules reach.
- `ProgressiveEpsilon` with a fixed-iteration Sinkhorn (`threshold <= 0`) no longer
  drives epsilon to `max_epsilon`.
- Float32 rotations for `d >= 500` are proper rotations (the determinant sign test
  underflowed). `biased_rotation!` works at `d = 1` with threads and applies the
  same phase convention as the Haar path.
- `step!`/`solve!` reject an objective that returns one value for the whole batch
  (a scalar objective without `columnwise`).
- Non-finite starting points are rejected; half-bounded boxes (`+-Inf` bounds) are
  accepted; `scale_cost` is validated at construction.
- Newton refinement under momentum keeps the velocity consistent with the move.
- Softmax with a subnormal epsilon returns the one-hot limit instead of NaN;
  `splitmix64` accepts `UInt64` seeds of `2^63` and above.
- Optimization.jl extension: `MaxSense` with a `batched` objective, `maxtime`
  (`ReturnCode.MaxTime`), a warning for the unused `abstol`/`reltol`, `Failure`
  when no finite value was seen, and the shape of `u0` is kept.

Performance:

- Rotations for `d > 8` use in-place `geqrf`/`orgqr` with a parity sign (no LU):
  `step!` is 1.4-2.4x faster, and seeded runs for `d > 8` differ at roundoff.
- `step!` and `solve!` no longer recompile for each new objective (0.3 s to about
  15 ms), and a PrecompileTools workload makes the first `minimize` about 20x faster.
- Sinkhorn allocates about 10x less and runs up to 2x faster at small sizes; the
  orthoplex centroid is O(d); state memory is halved.

Removed: the small-epsilon warning (it could not trigger). `SoftmaxSolver` and
`TemperedSoftmaxSolver` are now immutable.

## v0.1.0

Initial release.

Gradient-free polytope direct search for piecewise-constant and combinatorial
objectives:

- Batched objective protocol `(d, N) -> N`, column-major throughout, zero-allocation
  hot kernels, and serial-RNG results that are bit-identical for any thread count.
- OT weighting solvers: softmax (default), tempered softmax, min-cost greedy,
  top-k mean, log-domain Sinkhorn (SOR, Anderson, adaptive omega), and KL-softmax.
- Epsilon schedules (linear, cosine with SGDR restarts, progressive), decoupled probe
  and step radii, momentum, adaptive radius, biased rotations, and a finite-difference
  quadratic model with diagonal-Newton refinement and a trust region.
- Ask/tell interface (`PolyStepES`), `minimize`, and an Optimization.jl algorithm
  (`PolyStepOptimizer`).
- Optional extensions: LoopVectorization (`@turbo` kernels) and CUDA (`cuda_objective`).
- Two outer-loop heuristics differ from the Python reference for correctness:
  convergence also requires a small step magnitude, and the trust-region ratio scores
  the transport step taken.
