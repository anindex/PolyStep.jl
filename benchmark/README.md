# bbob-mixint benchmark

PolyStep on the COCO `bbob-mixint` suite (Tusar et al., GECCO 2019), dimensions
5 and 10, instances 1-15, one run per instance, compared with the archived COCO
data of CMA-ES with margin (Hamano et al., GECCO 2022) and CMA-ES pycma (2019).

## Run

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/bbob_mixint.jl          # B = 1000
julia --project=benchmark benchmark/bbob_mixint.jl 10000    # B = 10000
julia --project=benchmark benchmark/tune.jl                 # tuning grid
```

The first `using PythonCall` installs `coco-experiment` (module `cocoex`) and
`cocopp` from PyPI through CondaPkg. `bbob_mixint.jl` runs each problem with a
budget of `B * dim` evaluations, logs to `exdata/PolyStep_B<B>`, prints the
table below, and calls `cocopp.main` against the two archived entries. Figures
(PNG and SVG) land in
`ppdata/PolyStep_B<B>/mixint_PolyS_CMA-E_CMA-E_<timestamp>/`, for example
`pprldmany_05D_noiselessall.png`, `pprldmany_10D_noiselessall.png` and
`pprldmany_{05,10}D_{separ,lcond,hcond,multi,mult2}.png`; open `index.html` there
for all of them.

To post-process by hand with the CondaPkg Python:

```sh
julia --project=benchmark -e 'using CondaPkg; cd("benchmark"); CondaPkg.withenv() do
    run(`python -m cocopp exdata/PolyStep_B1000 bbob-mixint/2022/CMA-ESwM bbob-mixint/2019/CMA-ES-pycma`)
end'
```

`python -c "import cocopp; print(cocopp.archives.bbob_mixint)"` lists the archive
(`2019/CMA-ES-pycma.tgz`, `2019/DE-scipy.tgz`, `2019/RANDOMSEARCH.tgz`,
`2019/TPE-hyperopt.tgz`, `2022/CMA-ESwM_Hamano.zip`, three `2024/DE-*_Tanabe`
entries); any unique substring works as a name. Inside Julia the embedded Python
uses Julia's OpenSSL, which cannot find a CA bundle, so the script points
`SSL_CERT_FILE` at the CondaPkg bundle when it is unset.

## Solver

In `bbob-mixint` the first `number_of_integer_variables` coordinates are
integers with integer bounds (`[0,1]`, `[0,3]`, `[0,7]`, `[0,15]`), the rest are
continuous in `[-5,5]`; the problem itself rounds integer inputs. Each run is a
`PolyStepES` with box clamping and a `repair` that rounds the integer
coordinates (bounds-preserving). The radius starts at `rfrac * mean(ub - lb)`
and is multiplied by `decay` every round. A run restarts from a uniform point in
the box, with the radius reset, when the radius drops below `1e-10` or below
`1e-2` times the radius at the run's last improvement. The first run starts at
`initial_solution`. Runs stop when the budget or the final target (`1e-8`) is
reached. The seed of problem `i` is `Xoshiro(splitmix64(1, i))`.

## Tuning

`bbob-mixint` has only 15 instances; `instance_indices: 16-20` is clipped to
1-15 with a COCO warning. Tuning therefore uses dimension 20 (not in the
benchmark set), instances 1-5, all 24 functions, `B = 1000`. The score is the
area under the per-run ECDF over `log(evals)` in `[1, B * dim]`, pooled over
the 51 targets `10^2 ... 10^-8`.

A first grid (`epsilon` 0.1/0.5, `rfrac` 0.25/0.5/1.0, `decay` 0.9/0.97) peaked
at its edge (best 0.094 at 0.1/0.5/0.97), so the grid in `tune.jl` was shifted.
24 configurations were tried in total.

| epsilon | rfrac | decay 0.95 | decay 0.99 |
|---|---|---|---|
| 0.03 | 0.5 | 0.091 | 0.084 |
| 0.1 | 0.5 | 0.087 | 0.087 |
| 0.3 | 0.5 | 0.057 | 0.079 |
| 0.03 | 1.0 | 0.097 | 0.069 |
| **0.1** | **1.0** | **0.100** | 0.069 |
| 0.3 | 1.0 | 0.077 | 0.066 |

Chosen: `epsilon = 0.1`, `rfrac = 1.0`, `decay = 0.95`, used for both budgets.

## Results

See the [Benchmarks page](../docs/src/benchmarks.md). Wall-clock on one thread for the
experiment alone: about 6 s per dimension at `B = 1000`, 18 s (d=5) and 49 s (d=10) at
`B = 10000`, plus about a minute of cocopp. The baselines ran with COCO 2.3.4 (pycma)
and 2.5 (CMA-ESwM), this run with 2.8.2; `Fopt` matches on all 720 problems.
