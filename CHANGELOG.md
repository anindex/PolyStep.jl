# Changelog

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
