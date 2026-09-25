# bbob-mixint benchmark

This runs PolyStep on COCO's `bbob-mixint` suite (Tušar et al., GECCO 2019) in dimensions
5 and 10, instances 1-15, and compares it with the archived COCO runs of CMA-ES with
margin (Hamano et al., GECCO 2022) and pycma (2019). Results are on the
[Benchmarks page](../docs/src/benchmarks.md).

## Running it

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/bbob_mixint.jl          # budget 1000 * dim
julia --project=benchmark benchmark/bbob_mixint.jl 10000    # budget 10^4 * dim
julia --project=benchmark benchmark/tune.jl                 # tuning grid
```

The first run installs `coco-experiment` (the `cocoex` module) and `cocopp` from PyPI
through CondaPkg. The script logs to `exdata/PolyStep_B<B>`, prints the per-group table,
and runs `cocopp` against the two archived baselines. The plots end up in
`ppdata/PolyStep_B<B>/<timestamp folder>/`; open `index.html` there, or look at
`pprldmany_05D_noiselessall.png` and `pprldmany_10D_noiselessall.png`.

To list other archived algorithms, run
`python -c "import cocopp; print(cocopp.archives.bbob_mixint)"` in the CondaPkg
environment. The embedded Python cannot find a CA bundle on its own, so the script sets
`SSL_CERT_FILE` to the CondaPkg bundle before `cocopp` downloads the archives.

## What PolyStep does here

In `bbob-mixint` the first `number_of_integer_variables` coordinates are integers with
bounds `[0,1]`, `[0,3]`, `[0,7]` or `[0,15]`; the rest are continuous in `[-5,5]`. Each
run is a `PolyStepES` with box clamping and a `repair` that rounds the integer
coordinates. The radius starts at `rfrac * mean(ub - lb)` and is multiplied by `decay`
every round. When it drops below `1e-10`, or below `1e-2` times the radius at the last
improvement, the run restarts from a random point in the box. The first run starts at
`initial_solution`, and everything stops at the budget or at the final target `1e-8`.
Problem `i` is seeded with `Xoshiro(splitmix64(1, i))`, so reruns give the same numbers.

## Tuning

The suite has only 15 instances, so tuning uses dimension 20 (not in the results),
instances 1-5, all 24 functions and a budget of `1000 * dim`. The score is the area under
each run's ECDF over `log(evals)`, pooled over the 51 targets from `10^2` to `10^-8`.

| epsilon | rfrac | decay 0.95 | decay 0.99 |
|---|---|---|---|
| 0.03 | 0.5 | 0.082 | 0.082 |
| 0.1 | 0.5 | 0.085 | 0.085 |
| 0.3 | 0.5 | 0.057 | 0.078 |
| 0.03 | 1.0 | 0.099 | 0.069 |
| 0.1 | 1.0 | 0.101 | 0.070 |
| 0.3 | 1.0 | 0.080 | 0.069 |

The best setting, `epsilon = 0.1`, `rfrac = 1.0`, `decay = 0.95`, is used for both
budgets. An earlier grid (`epsilon` 0.1 or 0.5, `rfrac` 0.25 to 1, `decay` 0.9 or 0.97)
peaked at its edge, which is why this one is shifted.

## Timing and versions

On one thread the experiment takes about 6 s per dimension at `B = 1000`, and 18 s
(d = 5) and 49 s (d = 10) at `B = 10000`; `cocopp` adds about a minute. The baselines
were run with COCO 2.3.4 (pycma) and 2.5 (CMA-ES with margin), this harness with 2.8.2;
the optimal values match on all 720 problems.
