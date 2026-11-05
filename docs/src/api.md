# API reference

```@docs
PolyStep
```

## Optimizers

```@docs
minimize
PolyStepES
ask!
tell!
popsize
PolyStepOptimizer
PolyStepConfig
PolyStepState
init_state
step!(::Any, ::PolyStepConfig, ::PolyStepState)
solve!(::Any, ::PolyStepConfig, ::PolyStepState)
columnwise
```

## Solvers

```@docs
SoftmaxSolver
SinkhornSolver
KLSoftmaxSolver
MinCostGreedySolver
TopKMeanSolver
OTResult
PolyStep.solve
```

## Epsilon schedules

```@docs
LinearEpsilon
CosineEpsilon
ProgressiveEpsilon
epsilon_at
update!
```

## Geometry

```@docs
orthoplex_vertices
simplex_vertices
cube_vertices
polytope_vertices
num_vertices
probe_scales
haar_rotations!
biased_rotation!
```

## Subspaces

```@docs
ParamLayout
ParamEntry
LayerSpec
HybridSubspace
subspace_dim
compression_ratio
expand
expand!
project
project!
reconstruct_batch
subspace_objective
```

## Utilities

```@docs
splitmix64
PolyStep.cuda_objective
```
