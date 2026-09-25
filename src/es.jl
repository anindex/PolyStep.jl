# PolyStepES ask/tell. Candidates are clamped/repaired before caching, so the
# evaluated and projected points match and the barycenter stays in bounds.

"""
    PolyStepES(dim; num_particles=1, epsilon=0.5, step_radius=0.5,
               solver=SoftmaxSolver(), scale_cost=:mean, x0=nothing,
               lb=nothing, ub=nothing, repair=nothing,
               rng=Xoshiro(0), T=Float64)

Ask/tell wrapper around the PolyStep update. Population size is
`num_particles * 2 * dim` (orthoplex). `x0` is a `(dim,)` vector (replicated
to all particles) or a `(dim, num_particles)` matrix. `lb`/`ub` (scalar or
`(dim,)`) box-clamp candidates; `repair` is an in-place candidate
post-processor `repair(X_cols)` (e.g. rounding for integrality) applied after
clamping; it must itself preserve the bounds (its output is not re-clamped).
`best_x`/`best_f` track the best evaluated point: every `tell!` candidate, plus
the final iterate that `minimize` and `PolyStepOptimizer` score once. They are
always feasible when bounds and a bounds-preserving `repair` are active.

`step_radius` is absolute: candidates are `X +/- step_radius * R`. (In
`PolyStepConfig` an unscheduled radius is multiplied by `epsilon`, so pass
`r / epsilon` or a schedule there for the same step.) The radius is fixed, so
the mean settles in an O(`step_radius`) neighborhood of a minimizer. To shrink
it, mutate `es.step_radius` from a callback, e.g.
`minimize(f, d; callback = es -> (es.step_radius *= 0.99; false))`.
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
    const solver::S
    const scale_cost::SC
    const lb::LB
    const ub::UB
    const repair::RP
    const rng::RNG
    X::Matrix{T}                 # (d, P)
    pending::Union{Nothing, Matrix{T}}   # points at askbuf between ask! and tell!
    const askbuf::Matrix{T}      # (d, popsize) candidates, refilled each ask!
    const R::Array{T, 3}
    const Wn::Matrix{T}          # (V, P)
    const Cbuf::Matrix{T}        # (V, P) fitness -> cost staging, reused per tell!
    const Xnew::Matrix{T}        # (d, P) barycenter accumulator, reused per tell!
    best_x::Vector{T}
    best_f::Float64
    evals::Int
end

function PolyStepES(dim::Integer; num_particles::Integer = 1, epsilon::Real = 0.5,
        step_radius::Real = 0.5, solver::AbstractOTSolver = SoftmaxSolver(),
        scale_cost = :mean, x0 = nothing, lb = nothing, ub = nothing,
        repair = nothing, rng::AbstractRNG = Xoshiro(0),
        T::Type{<:AbstractFloat} = Float64)
    dim >= 1 || throw(ArgumentError("dim must be >= 1, got $dim"))
    num_particles >= 1 || throw(ArgumentError("num_particles must be >= 1, got $num_particles"))
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0, got $epsilon"))
    (isfinite(step_radius) && step_radius >= 0) ||
        throw(ArgumentError("step_radius must be finite and >= 0, got $step_radius"))
    # reject a bad spec now, not after the first population is evaluated
    scale_cost === nothing || scale_cost!(zeros(1, 1), zeros(1, 1), scale_cost)
    (lb === nothing) == (ub === nothing) ||
        throw(ArgumentError("provide both lb and ub or neither"))
    lb isa AbstractVector && length(lb) != dim &&
        throw(DimensionMismatch("lb has length $(length(lb)), expected dim=$dim"))
    ub isa AbstractVector && length(ub) != dim &&
        throw(DimensionMismatch("ub has length $(length(ub)), expected dim=$dim"))
    lb === nothing || _check_bounds(lb, ub)
    if num_particles == 1 && (solver isa SinkhornSolver ||
        (solver isa KLSoftmaxSolver && isinf(solver.lam)))
        @warn "Two-sided OT (Sinkhorn / KLSoftmax lam=Inf) with num_particles=1 yields " *
              "a uniform transport plan (the column marginal forces it), so steps " *
              "ignore fitness. Use the default SoftmaxSolver, or num_particles > 1."
    end
    P = Int(num_particles)
    d = Int(dim)
    X = if x0 === nothing
        zeros(T, d, P)
    else
        if x0 isa AbstractVector && length(x0) != d
            throw(ArgumentError("x0 has length $(length(x0)), expected dim=$d"))
        end
        X0 = x0 isa AbstractVector ? repeat(reshape(T.(x0), d, 1), 1, P) : Matrix{T}(x0)
        size(X0) == (d, P) ||
            throw(ArgumentError("x0 has size $(size(X0)), expected ($d,) or ($d, $P)"))
        # one non-finite column would make tell! skip every update
        all(isfinite, X0) || throw(ArgumentError("x0 must be finite"))
        copy(X0)
    end
    return PolyStepES{T, typeof(solver), typeof(scale_cost), typeof(lb), typeof(ub),
        typeof(repair), typeof(rng)}(
        d, P, Float64(epsilon), Float64(step_radius), deepcopy(solver), scale_cost,
        lb, ub, repair, rng,
        X, nothing, Matrix{T}(undef, d, 2d * P),
        zeros(T, d, d, P), zeros(T, 2d, P),
        zeros(T, 2d, P), zeros(T, d, P),
        fill(T(NaN), d), Inf, 0)
end

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
    C = es.askbuf                # reused; tell! consumes it before the next ask!
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
    es.repair === nothing || es.repair(C)
    es.pending = C
    return C
end

"""
    tell!(es, fitness::AbstractVector)

Update particles from the fitness of the last `ask!` (lower is better). The
barycentric projection runs over the cached candidates (identical to the
evaluated points, including clamping/repair), normalized by realized column
mass; a non-finite update is skipped. Tracks `best_x`/`best_f`/`evals`.
"""
function tell!(es::PolyStepES{T}, fitness::AbstractVector) where {T}
    es.pending === nothing && throw(ErrorException("tell! called before ask!"))
    ps = popsize(es)
    length(fitness) == ps ||
        throw(DimensionMismatch("fitness has length $(length(fitness)), expected popsize $ps"))
    d = es.dim
    P = es.num_particles
    V = 2d
    # no finite fitness: hold every particle (a uniform softmax would drift)
    fmin, idx = _best_finite(fitness)
    if idx == 0
        es.evals += ps
        es.pending = nothing
        return es
    end
    C = es.Cbuf
    @inbounds for p in 1:P, v in 1:V
        C[v, p] = T(fitness[(p - 1) * V + v])
    end
    local plan
    if es.solver isa SoftmaxSolver
        # fast path: softmax weights already equal the mass-normalized plan
        sanitize_cost!(C)
        es.scale_cost === nothing || scale_cost!(C, C, es.scale_cost)
        softmax_cols!(es.Wn, C, es.epsilon)
        plan = nothing
    else
        res = solve(es.solver, C, es.epsilon; scale_cost = es.scale_cost)
        normalize_particle_masses!(es.Wn, res.plan)
        plan = res.plan
    end
    pend = es.pending
    X_new = es.Xnew
    fill!(X_new, zero(T))
    @inbounds for p in 1:P
        if plan !== nothing
            # hold particles whose plan column has (near-)zero mass (Wn ~ 0)
            mass = zero(T)
            for v in 1:V
                mass += plan[v, p]
            end
            if !(mass > T(1e-12))
                for a in 1:d
                    X_new[a, p] = es.X[a, p]
                end
                continue
            end
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
    # idx != 0: the all-non-finite case returned early
    if fmin < es.best_f
        es.best_f = Float64(fmin)
        copyto!(es.best_x, view(pend, :, idx))
    end
    es.evals += ps
    es.pending = nothing
    return es
end

"""
    PolyStepOptimizer(; num_particles=1, epsilon=0.5, step_radius=0.5,
                      solver=SoftmaxSolver(), scale_cost=:mean)

Optimization.jl algorithm wrapper (activated by loading Optimization /
OptimizationBase; implemented in the package extension). `maxiters` counts
ask/tell rounds: total objective evaluations = `maxiters * num_particles *
2 * dim`, plus `num_particles` to score the final iterate. `maxtime` (seconds)
is checked after each round; `abstol`/`reltol` are ignored. With neither
`maxiters` nor `maxtime`, 1000 rounds run. Optimization.jl objectives are
scalar-per-point; the adapter loops over candidates (reshaped like `u0`) and
warns once; pass `batched = f_batch` through `solve` kwargs (or use the native
`minimize`/`PolyStepES` API) to keep vectorized evaluation. `f_batch` takes the
flat `(length(u0), N)` candidate matrix and returns the N objective values.
"""
Base.@kwdef struct PolyStepOptimizer{S <: AbstractOTSolver, SC}
    num_particles::Int = 1
    epsilon::Float64 = 0.5
    step_radius::Float64 = 0.5
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
`popsize`. Result in `es.best_x` / `es.best_f` / `mean(es)`.
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

# candidates sit step_radius off the iterate, so score the iterate itself once
function _score_iterate!(es::PolyStepES, f)
    Xc = copy(es.X)
    es.lb === nothing || (Xc .= clamp.(Xc, es.lb, es.ub))
    es.repair === nothing || es.repair(Xc)   # keeps best_x feasible
    fc, i = _best_finite(f(Xc))
    es.evals += size(Xc, 2)
    if i != 0 && fc < es.best_f
        es.best_f = fc
        copyto!(es.best_x, view(Xc, :, i))
    end
    return es
end
