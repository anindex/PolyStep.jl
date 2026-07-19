# PolyStepES: ask/tell interface. ask! returns (dim, popsize) candidate columns;
# tell! takes their fitness (lower better). Candidates are clamped/repaired before
# caching, so evaluated and projected points match and the barycenter stays in bounds.

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
`best_x`/`best_f` track the best evaluated candidate, always feasible when
bounds and a bounds-preserving `repair` are active.
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
    const Zc::Array{T, 3}
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
        copy(X0)
    end
    return PolyStepES{T, typeof(solver), typeof(scale_cost), typeof(lb), typeof(ub),
        typeof(repair), typeof(rng)}(
        d, P, Float64(epsilon), Float64(step_radius), deepcopy(solver), scale_cost,
        lb, ub, repair, rng,
        X, nothing, Matrix{T}(undef, d, 2d * P),
        zeros(T, d, d, P), zeros(T, d, d, P), zeros(T, 2d, P),
        zeros(T, 2d, P), zeros(T, d, P),
        fill(T(NaN), d), Inf, 0)
end

popsize(es::PolyStepES) = es.num_particles * 2 * es.dim
Statistics.mean(es::PolyStepES) = vec(mean(es.X; dims = 2))

"""
    ask!(es) -> (dim, popsize) Matrix

Candidate points as columns (column `v + (p-1)*2dim` = vertex v of particle p).
Errors when called twice without an intervening `tell!`. Do not mutate the
returned matrix; evaluate it and pass fitness to `tell!`.
"""
function ask!(es::PolyStepES{T}) where {T}
    es.pending === nothing ||
        throw(ErrorException("ask! called twice before tell!; tell! the previous population first"))
    d = es.dim
    P = es.num_particles
    haar_rotations!(es.R, es.Zc, es.rng)
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
    # NaN-safe scan up front. If the whole batch is non-finite, hold every
    # particle: a uniform softmax over clamped/repaired candidates would drift
    # the iterate on a round with no valid evaluation.
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
        # fast path: sanitize+scale in place on the state buffer; the softmax
        # weights are the realized-mass-normalized plan (plan = W .* a' and
        # the normalization divides a' back out), columns carry mass 1 exactly
        sanitize_cost!(C)
        es.scale_cost === nothing || scale_cost!(C, C, es.scale_cost)
        _warn_tiny_eps!(es.solver, es.epsilon, C)
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
            # a (near-)zero-mass plan column would "project" onto the origin
            # (Wn ~ 0); hold that particle's position instead
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
    # incumbent update (idx != 0 guaranteed: the all-non-finite case returned early)
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
2 * dim`. Optimization.jl objectives are scalar-per-point; the adapter loops
over candidates and warns once; pass `batched = f_batch` through `solve`
kwargs (or use the native `minimize`/`PolyStepES` API) to keep vectorized
evaluation.
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

Run `steps` ask/tell rounds of the batched objective
`f(X::(dim, popsize))::(popsize,)`. `callback(es) -> true` stops early.
Result in `es.best_x` / `es.best_f` / `mean(es)`.
"""
function minimize(f, dim::Integer; steps::Integer = 200, callback = nothing, kwargs...)
    es = PolyStepES(dim; kwargs...)
    for _ in 1:steps
        X = ask!(es)
        tell!(es, f(X))
        callback !== nothing && callback(es) === true && break
    end
    return es
end
