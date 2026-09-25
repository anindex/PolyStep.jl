# Changelog

## v0.2.0 (2026-09-26)

Seeded runs give different numbers than v0.1.0 for `d > 8` (new rotation sampler, same
distribution) and wherever the behavior below changed.

### Added

- `ParamLayout` and `HybridSubspace` for searching a per-layer subspace of a model's
  parameters, with the same dimensions as the Python `HybridSubspace`.
- A guide, a benchmarks page, and a COCO `bbob-mixint` harness in `benchmark/`.

### Changed

- `scale_cost = :mean`/`:max` subtract the minimum cost before scaling, so adding a
  constant to the objective no longer changes a run.
- Non-finite costs are replaced by `2*max|finite| + 1`.
- `best_x`/`best_f` also cover the final iterate.
- The trust region compares against the quadratic model's value at the center and
  shrinks when the model predicts a rise.
- The quadratic model floors its curvature everywhere, and Newton steps are clamped per
  coordinate before the norm clip.
- The adaptive radius follows each particle's best vertex.
- The greedy solvers leave a particle in place when all its costs are equal.
- Sinkhorn/KL warm starts are no longer rescaled when epsilon changes (the `last_eps`
  keyword is gone). The KL plan is built as a column softmax, and
  `data_dependent_init` runs one plain sweep.

### Fixed

- Log-sum-exp kernels returned NaN for rows starting with `-Inf`.
- The finite-difference Hessian collapsed at small probe radii.
- Float32 rotations for `d >= 500` were not always proper rotations.
- `ProgressiveEpsilon` drifted with fixed-iteration Sinkhorn, and it now rejects
  one-pass solvers.
- Optimization.jl extension: `MaxSense`, `maxtime`, a `Failure` return code, and the
  shape of `u0`.

### Performance

- Random rotations are drawn with Stewart's method and, below `d = 192`, built in pure
  Julia, so parallel runs no longer wait on a BLAS lock. `step!` is 5.6x faster at
  `d = 9`, 2.4x at `d = 100` and 1.8x at `d = 1000`.
- Biased rotations take one reflection, O(d^2).
- `step!` no longer recompiles for each new objective, and the first call is precompiled.

## v0.1.0

First release.
