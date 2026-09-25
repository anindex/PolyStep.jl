# Changelog

## v0.2.0 (2026-09-25)

Seeded results change for `d > 8` and wherever behavior changed below.

### Added

- `ParamLayout` and `HybridSubspace`: per-layer orthonormal subspaces for layered models,
  matching the Python `HybridSubspace` dimensions.
- Guide and Benchmarks docs pages; a COCO `bbob-mixint` harness in `benchmark/`.

### Changed

- `scale_cost = :mean`/`:max` recenter costs before scaling; non-finite costs get
  `2*max|finite| + 1`; `best_x`/`best_f` include the final iterate.
- Trust region scores the quadratic model's center value and shrinks on a predicted rise;
  curvature is floored in all predictions; Newton steps are clamped per coordinate.
- Adaptive radius follows each particle's best vertex. Greedy solvers hold on flat costs.
- Sinkhorn/KL: warm starts are no longer rescaled by epsilon (`last_eps` removed), the KL
  plan is a column softmax, and `data_dependent_init` is one plain sweep.

### Fixed

- NaN from log-sum-exp kernels on rows starting with `-Inf`.
- FD Hessian collapse at small probe radii; Float32 rotations for `d >= 500`.
- `ProgressiveEpsilon` drift with fixed-iteration Sinkhorn; one-pass solvers are rejected.
- Optimization.jl extension: `MaxSense`, `maxtime`, `Failure` retcode, `u0` shape.

### Performance

- Stewart rotation sampler, pure Julia below `d = 192`: `step!` 5.6x faster at `d = 9`,
  2.4x at `d = 100`, 1.8x at `d = 1000`; parallel runs no longer share a BLAS lock.
- Biased rotations in O(d^2); no recompilation per objective; precompiled first call.

## v0.1.0

Initial release.
