# bbob-mixint benchmark

This harness runs PolyStep on the COCO `bbob-mixint` suite (Tušar et al., GECCO 2019) in
dimensions 5 and 10, instances 1-15. It compares PolyStep with the archived COCO runs of
CMA-ES with margin (Hamano et al., GECCO 2022) and pycma (2019). The
[Benchmarks page](../docs/src/benchmarks.md) gives the results.

## Running it

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/bbob_mixint.jl          # budget 1000 * dim
julia --project=benchmark benchmark/bbob_mixint.jl 10000    # budget 10^4 * dim
julia --project=benchmark benchmark/tune.jl                 # tuning grid
```

The first run installs `coco-experiment` (the `cocoex` module) and `cocopp` from PyPI
through CondaPkg. The script writes its logs to `exdata/PolyStep_B<B>`, prints the table
for each function group, and runs `cocopp` against the two archived baselines. The plots
are in `ppdata/PolyStep_B<B>/<timestamp folder>/`: open `index.html`, or
`pprldmany_05D_noiselessall.png` and `pprldmany_10D_noiselessall.png`.

To list other archived algorithms, run this command in the CondaPkg environment:

```sh
julia --project=benchmark -e 'using CondaPkg; CondaPkg.withenv() do
    run(`python -c "import cocopp; print(cocopp.archives.bbob_mixint)"`)
end'
```

The embedded Python does not find a CA bundle. If `SSL_CERT_FILE` is not set, the script
sets it to the CondaPkg bundle before `cocopp` downloads the archives.

## What PolyStep does here

In `bbob-mixint`, the first `number_of_integer_variables` coordinates are integers with
bounds `[0,1]`, `[0,3]`, `[0,7]` or `[0,15]`. The other coordinates are continuous in
`[-5,5]`. Each run is a `PolyStepES` with box clamping and a `repair` that rounds the
integer coordinates. The radius starts at `rfrac * mean(ub - lb)` and is multiplied by
`decay` after each round. The run restarts from a random point in the box when the radius
is less than `1e-10`. It also restarts when the radius is less than `1e-2` times the
radius at the last improvement. The first run starts at `initial_solution`. A run stops at
the budget or at the final target `1e-8`. Problem `i` uses `Xoshiro(splitmix64(1, i))`, so
a repeated run gives the same numbers.

## Tuning

The suite has only 15 instances, so the tuning uses dimension 20 (not in the results),
instances 1-5, all 24 functions and a budget of `1000 * dim`. The score is the area under
the ECDF of each run over `log(evals)`, pooled over the 51 targets from `10^2` to `10^-8`.

| epsilon | rfrac | decay 0.95 | decay 0.99 |
|---|---|---|---|
| 0.03 | 0.5 | 0.082 | 0.082 |
| 0.1 | 0.5 | 0.085 | 0.085 |
| 0.3 | 0.5 | 0.057 | 0.078 |
| 0.03 | 1.0 | 0.099 | 0.069 |
| 0.1 | 1.0 | 0.101 | 0.070 |
| 0.3 | 1.0 | 0.080 | 0.069 |

Both budgets use the best setting, `epsilon = 0.1`, `rfrac = 1.0`, `decay = 0.95`. An
earlier grid (`epsilon` 0.1 or 0.5, `rfrac` 0.25 to 1, `decay` 0.9 or 0.97) had its best
value at its edge, so this grid is shifted.

## Timing and versions

On one thread, the experiment takes approximately 6 s per dimension at `B = 1000`. At `B =
10000`, it takes 18 s (d = 5) and 49 s (d = 10). `cocopp` adds approximately one minute.
The baselines used COCO 2.3.4 (pycma) and 2.5 (CMA-ES with margin), and this harness uses
2.8.2. The optimal values agree on all 720 problems.
