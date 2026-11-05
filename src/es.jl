"""
    PolyStepES(dim; num_particles=1, epsilon=0.5, step_radius=0.5, radius_decay=1.0,
               solver=SoftmaxSolver(), scale_cost=:mean, x0=nothing,
               lb=nothing, ub=nothing, repair=nothing,
               rng=Xoshiro(0), T=float(eltype(x0)))

Ask/tell wrapper around the PolyStep update. Population size is
`num_particles * 2 * dim` (orthoplex). `x0` is a `(dim,)` vector (replicated
to all particles) or a `(dim, num_particles)` matrix. `lb`/`ub` (scalar or
`(dim,)`) box-clamp candidates. `repair(X_cols)` post-processes candidates after
clamping, for example by rounding for integrality. It edits `X_cols` in place or
returns the repaired matrix. It must keep the bounds, because its output is not
clamped again. `T` defaults to the
element type of `x0` (`Float64` without one).
`best_x`/`best_f` track the best evaluated point: every `tell!` candidate, plus
the final iterate that `minimize` and `PolyStepOptimizer` score once. They are
always feasible when bounds and a bounds-preserving `repair` are active.

`step_radius` is absolute: candidates are `X +/- step_radius * R`. (In
`PolyStepConfig` an unscheduled radius is multiplied by `epsilon`, so pass
`r / epsilon` or a schedule there for the same step.) With the default
`radius_decay = 1` the radius is fixed, so the mean settles in an O(`step_radius`)
neighborhood of a minimizer. With `radius_decay < 1`, every `tell!` multiplies
`step_radius` by `radius_decay`, also when the update is skipped.
`:mean` recenters and rescales the costs every round, so near a minimizer the
iterate keeps moving by a fixed fraction of `step_radius` (set by `epsilon` and
`dim`, not by the distance to the minimizer).
"""
mutable struct PolyStepES{
    T <: AbstractFloat, S <: AbstractOTSolver, SC, LB, UB, RP, RNG <: AbstractRNG}
    const dim::Int
    const num_particles::Int
    epsilon::Float64
    step_radius::Float64
    const radius_decay::Float64
    const solver::S
    const scale_cost::SC
    const lb::LB
    const ub::UB
    const repair::RP
    const rng::RNG
    X::Matrix{T}                 # (d, P)
    pending::Union{Nothing, Matrix{T}}   # points at askbuf between ask! and tell!
    const askbuf::Matrix{T}
    const R::Array{T, 3}
    const Wn::Matrix{T}          # (V, P)
    const Cbuf::Matrix{T}
    const Xnew::Matrix{T}
    best_x::Vector{T}
    best_f::Float64
    evals::Int
    f::Union{Nothing, AbstractVector}
    g::Union{Nothing, AbstractVector}
end

function PolyStepES(dim::Integer; num_particles::Integer = 1, epsilon::Real = 0.5,
        step_radius::Real = 0.5, radius_decay::Real = 1.0,
        solver::AbstractOTSolver = SoftmaxSolver(),
        scale_cost = :mean, x0 = nothing, lb = nothing, ub = nothing,
        repair = nothing, rng::AbstractRNG = Xoshiro(0),
        T::Type{TT} = _float_type(x0)) where {TT <: AbstractFloat}
    dim >= 1 || throw(ArgumentError("dim must be >= 1, got $dim"))
    num_particles >= 1 || throw(ArgumentError("num_particles must be >= 1, got $num_particles"))
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0, got $epsilon"))
    isinf(epsilon) && solver isa Union{SinkhornSolver, KLSoftmaxSolver} &&
        throw(ArgumentError("$(nameof(typeof(solver))) needs a finite epsilon"))
    (isfinite(step_radius) && step_radius >= 0) ||
        throw(ArgumentError("step_radius must be finite and >= 0, got $step_radius"))
    0 < radius_decay <= 1 ||
        throw(ArgumentError("radius_decay must be in (0, 1], got $radius_decay"))
    x0 === nothing || x0 isa AbstractArray ||
        throw(ArgumentError("x0 must be a (dim,) vector or a (dim, num_particles) matrix, got $(typeof(x0))"))
    scale_cost === nothing || scale_cost!(zeros(1, 1), zeros(1, 1), scale_cost)
    (lb === nothing) == (ub === nothing) ||
        throw(ArgumentError("provide both lb and ub or neither"))
    lb === nothing || _check_bounds(lb, ub)
    lb isa AbstractVector && length(lb) != dim &&
        throw(DimensionMismatch("lb has length $(length(lb)), expected dim=$dim"))
    ub isa AbstractVector && length(ub) != dim &&
        throw(DimensionMismatch("ub has length $(length(ub)), expected dim=$dim"))
    if num_particles == 1 && (solver isa SinkhornSolver ||
        (solver isa KLSoftmaxSolver && isinf(solver.lam)))
        @warn "Two-sided OT (Sinkhorn / KLSoftmax lam=Inf) with num_particles=1 yields " *
              "a uniform transport plan (the column marginal forces it), so steps " *
              "ignore fitness. Use the default SoftmaxSolver, or num_particles > 1."
    end
    P = Int(num_particles)
    d = Int(dim)
    X = if x0 === nothing
        zeros(TT, d, P)
    else
        if x0 isa AbstractVector && length(x0) != d
            throw(ArgumentError("x0 has length $(length(x0)), expected dim=$d"))
        end
        X0 = x0 isa AbstractVector ? repeat(reshape(TT.(x0), d, 1), 1, P) : Matrix{TT}(x0)
        size(X0) == (d, P) ||
            throw(ArgumentError("x0 has size $(size(X0)), expected ($d,) or ($d, $P)"))
        # one non-finite column would make tell! skip every update
        all(isfinite, X0) || throw(ArgumentError("x0 must be finite"))
        copy(X0)
    end
    return PolyStepES{TT, typeof(solver), typeof(scale_cost), typeof(lb), typeof(ub),
        typeof(repair), typeof(rng)}(
        d, P, Float64(epsilon), Float64(step_radius), Float64(radius_decay),
        deepcopy(solver), scale_cost, lb, ub, repair, rng,
        X, nothing, Matrix{TT}(undef, d, 2d * P),
        zeros(TT, d, d, P), zeros(TT, 2d, P),
        zeros(TT, 2d, P), zeros(TT, d, P),
        fill(TT(NaN), d), Inf, 0, nothing, nothing)
end

_float_type(x) = x isa AbstractArray && isconcretetype(eltype(x)) && eltype(x) <: Real ?
                 float(eltype(x)) : Float64

"""
    popsize(es)

Candidates per `ask!`: `num_particles * 2 * dim`.
"""
popsize(es::PolyStepES) = es.num_particles * 2 * es.dim
Statistics.mean(es::PolyStepES) = vec(mean(es.X; dims = 2))

"""
    ask!(es) -> (dim, popsize) Matrix

Candidate points as columns (column `v + (p-1)*2dim` = vertex v of particle p).
Errors when called twice without an intervening `tell!`. The returned matrix
is an internal buffer: do not mutate it (`tell!` reads it back), and `copy` it
to keep it, since the next `ask!` overwrites it in place.
"""
function ask!(es::PolyStepES{T}) where {T}
    es.pending === nothing ||
        throw(ErrorException("ask! called twice before tell!; tell! the previous population first"))
    d = es.dim
    P = es.num_particles
    haar_rotations!(es.R, es.R, es.rng)   # R doubles as the Gaussian scratch
    C = es.askbuf
    sr = T(es.step_radius)
    @inbounds for p in 1:P
        base = (p - 1) * 2d
        for i in 1:d
            @simd ivdep for a in 1:d
                C[a, base + i] = es.X[a, p] + sr * es.R[a, i, p]
                C[a, base + d + i] = es.X[a, p] - sr * es.R[a, i, p]
            end
        end
    end
    if es.lb !== nothing
        C .= clamp.(C, es.lb, es.ub)
    end
    _repair!(es.repair, C)
    es.pending = C
    return C
end

"""
    tell!(es, fitness::AbstractVector)

Update particles from the fitness of the last `ask!` (lower is better). The
barycentric projection runs over the cached candidates (identical to the
evaluated points, including clamping/repair), normalized by realized column
mass. The particles stay when no fitness is finite or the update is not finite.
Costs are weighted in the wider of `T` and `eltype(fitness)`. Iterative solvers
warm-start from the duals of the previous `tell!`. Tracks `best_x`/`best_f`/`evals`
and applies `radius_decay`.
"""
function tell!(es::PolyStepES{T}, fitness::AbstractVector) where {T}
    es.pending === nothing && throw(ErrorException("tell! called before ask!"))
    ps = popsize(es)
    length(fitness) == ps ||
        throw(DimensionMismatch("fitness has length $(length(fitness)), expected popsize $ps"))
    d = es.dim
    P = es.num_particles
    V = 2d
    S = promote_type(T, _float_type(fitness))
    C = S === T ? es.Cbuf : Matrix{S}(undef, V, P)
    copyto!(C, fitness)
    pend = es.pending
    es.pending = nothing
    es.evals += ps
    es.step_radius *= es.radius_decay
    fmin, idx = _best_finite(C)
    idx == 0 && return es
    W = S === T ? es.Wn : similar(C)
    local plan
    if es.solver isa SoftmaxSolver
        _prepare_cost!(C, C, es.scale_cost)
        softmax_cols!(W, C, es.epsilon)
        plan = nothing
    else
        res = solve(es.solver, C, es.epsilon; scale_cost = es.scale_cost, f0 = es.f, g0 = es.g)
        es.f, es.g = res.f, res.g
        normalize_particle_masses!(W, res.plan)
        plan = res.plan
    end
    W === es.Wn || (es.Wn .= W)
    X_new = es.Xnew
    fill!(X_new, zero(T))
    @inbounds for p in 1:P
        if plan !== nothing && !_has_mass(sum(view(plan, :, p)))
            for a in 1:d
                X_new[a, p] = es.X[a, p]
            end
            continue
        end
        base = (p - 1) * V
        for v in 1:V
            w = es.Wn[v, p]
            @simd ivdep for a in 1:d
                X_new[a, p] += w * pend[a, base + v]
            end
        end
    end
    if all(isfinite, X_new)
        copyto!(es.X, X_new)
    end
    if fmin < es.best_f
        es.best_f = Float64(fmin)
        copyto!(es.best_x, view(pend, :, idx))
    end
    return es
end

"""
    PolyStepOptimizer(; num_particles=1, epsilon=0.5, step_radius=0.5, radius_decay=1.0,
                      solver=SoftmaxSolver(), scale_cost=:mean)

Optimization.jl algorithm (load Optimization or OptimizationBase). It runs a
[`PolyStepES`](@ref) with these settings. `maxiters` counts ask/tell rounds,
`maxiters * num_particles * 2 * length(u0)` evaluations plus `num_particles` for
the final iterate; `maxtime` is checked after each round; with neither, 1000
rounds run. `abstol`/`reltol` are ignored. `lb`/`ub` clamp the candidates.
`MaxSense` works. Constraints and `repair` are not supported. Extra `solve`
keywords: `batched = f_batch`
evaluates the flat `(length(u0), N)` candidate matrix in one call (otherwise the
scalar objective runs once per candidate, with a warning), and `rng` (default
`Xoshiro(0)`, so repeated solves are identical). Return codes: `Success`,
`Terminated` (callback), `MaxTime`, `Failure` (no finite objective).
"""
Base.@kwdef struct PolyStepOptimizer{S <: AbstractOTSolver, SC}
    num_particles::Int = 1
    epsilon::Float64 = 0.5
    step_radius::Float64 = 0.5
    radius_decay::Float64 = 1.0
    solver::S = SoftmaxSolver()
    scale_cost::SC = :mean
end

"""
    minimize(f, dim; steps=200, callback=nothing, kwargs...) -> PolyStepES

Run `steps` ask/tell rounds of the batched objective `f(X::(dim, N))::(N,)`.
`callback(es) -> true` stops early. The final iterate (clamped and repaired) is
then scored once as a `(dim, num_particles)` matrix, costing `num_particles`
extra evaluations, so `best_f` is never worse than `f` at the point the run
ended on. `f` and `repair` must therefore accept any column count, not just
`popsize`. Result in `es.best_x`, `es.best_f` and `mean(es)` (from Statistics).
"""
function minimize(f, dim::Integer; steps::Integer = 200, callback = nothing, kwargs...)
    es = PolyStepES(dim; kwargs...)
    for _ in 1:steps
        X = ask!(es)
        tell!(es, f(X))
        callback !== nothing && callback(es) === true && break
    end
    return _score_iterate!(es, f)
end

function _score_iterate!(es::PolyStepES, f)
    Xc = copy(es.X)
    es.lb === nothing || (Xc .= clamp.(Xc, es.lb, es.ub))
    _repair!(es.repair, Xc)
    y = f(Xc)
    length(y) == size(Xc, 2) || throw(DimensionMismatch(
        "objective returned $(length(y)) costs for $(size(Xc, 2)) columns; wrap scalar objectives with columnwise"))
    fc, i = _best_finite(y)
    es.evals += size(Xc, 2)
    if i != 0 && fc < es.best_f
        es.best_f = fc
        copyto!(es.best_x, view(Xc, :, i))
    end
    return es
end
