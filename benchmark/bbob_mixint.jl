using PolyStep, PythonCall, CondaPkg, Random, Statistics

const cocoex = pyimport("cocoex")
const np = pyimport("numpy")

const B = isempty(ARGS) ? 1000 : parse(Int, ARGS[1])
const DIMS = "5,10"
const INSTANCES = "1-15"
const SEED = 1
const EPSILON = 0.1
const RFRAC = 1.0
const DECAY = 0.95
const FLOOR = 1e-10
const STALL = 1e-2
const TARGETS = 10.0 .^ (2:-0.2:-8)
const ARCHIVES = ["bbob-mixint/2022/CMA-ESwM", "bbob-mixint/2019/CMA-ES-pycma"]
const GROUPS = ["separ" => 1:5, "lcond" => 6:9, "hcond" => 10:14, "multi" => 15:19,
    "mult2" => 20:24, "all" => 1:24]

function polystep!(p, budget, rng; epsilon = EPSILON, rfrac = RFRAC, decay = DECAY)
    d = pyconvert(Int, p.dimension)
    nint = pyconvert(Int, p.number_of_integer_variables)
    lb = pyconvert(Vector{Float64}, p.lower_bounds)
    ub = pyconvert(Vector{Float64}, p.upper_bounds)
    x0 = pyconvert(Vector{Float64}, p.initial_solution)
    r0 = rfrac * mean(ub .- lb)
    repair(X) = (V = view(X, 1:nint, :); V .= round.(V); X)
    buf = zeros(d)
    pybuf = np.asarray(buf)
    fx = zeros(2d)
    evals = 0
    while evals + 2d <= budget
        es = PolyStepES(d; lb, ub, x0, repair, epsilon, step_radius = r0, rng)
        fbest, rimp = Inf, r0
        while evals + 2d <= budget && es.step_radius > max(FLOOR, STALL * rimp)
            X = ask!(es)
            for j in axes(X, 2)
                copyto!(buf, view(X, :, j))
                fx[j] = pyconvert(Float64, p(pybuf))
            end
            evals += 2d
            tell!(es, fx)
            pyconvert(Bool, p.final_target_hit) && return
            es.best_f < fbest && ((fbest, rimp) = (es.best_f, es.step_radius))
            es.step_radius *= decay
        end
        x0 = lb .+ rand(rng, d) .* (ub .- lb)
    end
end

function experiment(name, options, mult; kw...)
    rm(joinpath("exdata", name); force = true, recursive = true)
    suite = cocoex.Suite("bbob-mixint", "", options)
    observer = cocoex.Observer("bbob-mixint", "result_folder: $name algorithm_name: PolyStep")
    seconds = Dict{Int, Float64}()
    for p in suite
        p.observe_with(observer)
        d = pyconvert(Int, p.dimension)
        rng = Xoshiro(splitmix64(SEED, pyconvert(Int, p.index)))
        seconds[d] = get(seconds, d, 0.0) + @elapsed polystep!(p, mult * d, rng; kw...)
        p.free()
    end
    return joinpath("exdata", name), seconds
end

function cocopp()
    haskey(ENV, "SSL_CERT_FILE") ||
        (ENV["SSL_CERT_FILE"] = joinpath(CondaPkg.envdir(), "ssl", "cert.pem"))
    return pyimport("cocopp")
end

function runtimes(src)
    rt = Dict{Tuple{Int, Int}, Vector{Float64}}()
    for ds in cocopp().load(src)
        key = (pyconvert(Int, ds.funcId), pyconvert(Int, ds.dim))
        for e in ds.detEvals(pylist(TARGETS))
            append!(get!(rt, key, Float64[]), pyconvert(Vector{Float64}, e))
        end
    end
    return rt
end

pooled(rt, fids, d) = reduce(vcat, [v for ((f, dd), v) in rt if f in fids && dd == d]; init = Float64[])
solved(rt, fids, d, b) = mean(pooled(rt, fids, d) .<= b)
auc(rt, mult) = mean(isnan(t) ? 0.0 : max(0.0, 1 - log(t) / log(mult * d)) for ((_, d), v) in rt for t in v)

function report(srcs, dims; mults = (10, 100, 1000, 10000))
    rts = [src => runtimes(src) for src in srcs]
    println("fraction of (function, instance, target) triples solved within budget = mult * dim")
    println(rpad("dim group algorithm", 48), join(lpad.(mults, 8)))
    for d in dims, (g, fids) in GROUPS, (src, rt) in rts
        println(rpad("$d $g $(basename(src))", 48),
            join(lpad.(round.([solved(rt, fids, d, m * d) for m in mults]; digits = 3), 8)))
    end
end

function main()
    cd(@__DIR__)
    name = "PolyStep_B$B"
    dir, seconds = experiment(name, "dimensions: $DIMS instance_indices: $INSTANCES", B)
    println("wall-clock seconds per dimension: ", sort(collect(seconds)))
    report([dir; ARCHIVES], parse.(Int, split(DIMS, ",")))
    flush(stdout)
    pp = cocopp()
    pp.genericsettings.figure_file_formats = pylist(["png", "svg"])
    pp.main("-o ppdata/$name $dir " * join(ARCHIVES, " "))
end

abspath(PROGRAM_FILE) == @__FILE__() && main()
