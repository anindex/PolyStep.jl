# Changelog

## v0.2.0

- `ParamLayout` and `HybridSubspace`: a fixed per-layer orthonormal basis for
  parameter vectors of layered models, with per-layer rank, a total-dimension
  budget (`max_subspace_dim`), `project`/`expand` between the full vector and the
  subspace coordinates, and `subspace_objective` to run any PolyStep solver in the
  subspace. The dimension arithmetic and the cap semantics mirror the Python
  `HybridSubspace`; the package tests assert them against the Python reference
  for four layouts, five ranks and five caps.

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
