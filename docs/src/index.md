# PolyStep.jl

Gradient-free direct search for piecewise-constant objectives: simulation
optimization, decision-focused learning, and contextual combinatorial pipelines.

Each step rotates a unit polytope (an orthoplex by default) by a Haar-random
rotation, evaluates the objective at the vertex directions in one batched call,
weights the vertices by `softmax(-C/eps)` or an entropic Sinkhorn plan, and steps to
the barycenter. The finite probe radius crosses the flat pieces of a piecewise-constant
loss, where gradient-estimate methods (OpenAI-ES, SPSA) average equal losses into a
zero gradient and stall.

## When to use it

Good fit: the objective is piecewise-constant or non-differentiable (losses through
argmax routing, combinatorial solvers, integer-demand simulations), and evaluations
are cheap and batchable (one `(d, N)` call is thousands of vectorized simulations).

Poor fit: expensive single simulations (minutes per run); use Bayesian optimization
instead.

## Quick start

```julia
using PolyStep, Random

# batched objective: (d, N) candidate columns -> N costs
rastrigin(X) = vec(sum(X .^ 2 .- 10 .* cos.(2 * pi .* X) .+ 10; dims = 1))

es = minimize(rastrigin, 10; steps = 500, epsilon = 0.05, step_radius = 0.3,
              x0 = fill(2.0, 10), rng = Xoshiro(0))
es.best_x, es.best_f
```

Ask/tell, with bounds and integrality repair:

```julia
es = PolyStepES(5; lb = 0.0, ub = 10.0, repair = X -> (X .= round.(X); X))
for _ in 1:200
    X = ask!(es)          # (dim, popsize) columns, clamped and repaired
    tell!(es, my_costs(X))
end
```

The lower-level `PolyStepConfig` / `init_state` / `step!` / `solve!` API exposes the
full feature set: Sinkhorn / KL-softmax / tempered / greedy solvers, epsilon
schedules, decoupled probe and step radii, momentum, adaptive radius, biased
rotations, and a finite-difference quadratic model with Newton refinement and trust
region. See the [API](api.md) reference and the runnable scripts in `examples/`.
