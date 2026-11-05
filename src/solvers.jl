abstract type AbstractOTSolver end

"""
    PolyStep.solve(solver, C, eps; a=nothing, b=nothing, f0=nothing, g0=nothing,
                   scale_cost=nothing) -> OTResult

Weight the `(V, P)` cost matrix `C` (vertices by particles) at temperature `eps`.
`a` (length `P`) and `b` (length `V`) are the particle and vertex marginals, uniform
by default. Only `SinkhornSolver` and `KLSoftmaxSolver` use `b` and `f0`/`g0` (dual
warm start); the other solvers warn on a non-uniform `b` and ignore it. `scale_cost`
(`nothing`, a positive divisor, `:mean` or `:max`) rescales `C` first. Non-finite costs
rank last.
"""
function solve end

"""
    OTResult

`plan::(V,P)` transport (column p sums to ~ a[p] for one-sided solvers),
`ent_cost` a per-solver cost proxy on the sanitized, scaled cost (Sinkhorn
reports the dual value, the other solvers the transport cost `<C,P>`; it is not
the regularized primal `<C,P> + eps*KL`), warm-startable duals `f::(P,)` /
`g::(V,)` (nothing for non-iterative solvers), `converged`, `iters`, per-check
marginal `errors` (Sinkhorn diagnostics, else nothing).
"""
struct OTResult{MT <: AbstractMatrix, FT, GT}
    plan::MT
    ent_cost::Float64
    f::FT
    g::GT
    converged::Bool
    iters::Int
    errors::Union{Nothing, Vector{Float64}}
end
function OTResult(plan, ent_cost, f, g, converged, iters)
    OTResult(plan, ent_cost, f, g, converged, iters, nothing)
end

function _align_marginal(a, n::Integer, C::AbstractMatrix{T}, name::String) where {T}
    if a === nothing
        u = similar(C, T, n)
        fill!(u, T(1) / n)
        return u
    end
    length(a) == n || throw(DimensionMismatch("$name has length $(length(a)), expected $n"))
    all(isfinite, a) || throw(ArgumentError("$name must be finite"))
    all(x -> x >= 0, a) || throw(ArgumentError("$name must be nonnegative"))
    out = similar(C, T, n)
    copyto!(out, a)
    sum(out) > 0 || throw(ArgumentError("$name must have positive total mass in $T"))
    return out
end

_prepare_cost(C::AbstractMatrix, scale_cost) = _prepare_cost!(similar(C), C, scale_cost)

function _solve_on_support(s, Cs::AbstractMatrix{T}, eps, am, bm, f0, g0) where {T}
    rows = findall(>(0), bm)
    cols = findall(>(0), am)
    sub(x, idx, n) = x === nothing || length(x) != n ? nothing : collect(x)[idx]
    r = solve(s, Cs[rows, cols], eps; a = am[cols], b = bm[rows],
        f0 = sub(f0, cols, length(am)), g0 = sub(g0, rows, length(bm)))
    plan = fill!(similar(Cs), zero(T))
    plan[rows, cols] .= r.plan
    f = fill!(similar(Cs, T, length(am)), T(-Inf))
    g = fill!(similar(Cs, T, length(bm)), T(-Inf))
    f[cols] .= r.f
    g[rows] .= r.g
    return OTResult(plan, r.ent_cost, f, g, r.converged, r.iters, r.errors)
end

_maxabs(x) = isempty(x) ? zero(eltype(x)) : maximum(abs, x)

function _check_eps(eps)
    eps > 0 ||
        throw(ArgumentError("epsilon must be > 0, got $eps (it is the temperature in softmax(-C/epsilon))"))
end

function _check_nonempty(V::Integer, P::Integer)
    (V == 0 || P == 0) &&
        throw(ArgumentError("cost matrix must be non-empty, got (V=$V, P=$P)"))
end

function _warn_nonuniform_b(b, V::Integer)
    if b !== nothing
        uniform = 1 / V
        if !all(x -> isapprox(x, uniform; atol = 1e-6), b)
            @warn "This solver ignores the target marginal `b`: it only enforces column sums " *
                  "equal to the source marginal `a`. Use SinkhornSolver for two-sided OT."
        end
    end
    return nothing
end

"""
    SoftmaxSolver()

One-sided softmax weighting: `W = softmax(-C/eps, dims=1)`,
`plan = W .* a'` (column sums equal `a`). No duals, single pass.
"""
struct SoftmaxSolver <: AbstractOTSolver end

function solve(s::SoftmaxSolver, C::AbstractMatrix{T}, eps::Real;
        a = nothing, b = nothing, f0 = nothing, g0 = nothing,
        scale_cost = nothing) where {T}
    T <: AbstractFloat || return solve(s, float.(C), eps; a, b, f0, g0, scale_cost)
    _check_eps(eps)
    V, P = size(C)
    _check_nonempty(V, P)
    _warn_nonuniform_b(b, V)
    Cs = _prepare_cost(C, scale_cost)
    am = _align_marginal(a, P, Cs, "a")
    W = similar(Cs)
    softmax_cols!(W, Cs, eps)
    plan = W .* am'
    ent = Float64(dot(Cs, plan))
    return OTResult(plan, ent, nothing, nothing, true, 1)
end

_rank(c::T) where {T} = isfinite(c) ? c : T(Inf)

"""
    MinCostGreedySolver()

All of each particle's mass on its lowest-cost vertex. Non-finite costs rank
last. A column with no finite spread (constant, or all non-finite) spreads
uniformly, so the particle holds.
"""
struct MinCostGreedySolver <: AbstractOTSolver end

function solve(s::MinCostGreedySolver, C::AbstractMatrix{T}, eps::Real = 0.1;
        a = nothing, b = nothing, f0 = nothing, g0 = nothing,
        scale_cost = nothing) where {T}
    T <: AbstractFloat || return solve(s, float.(C), eps; a, b, f0, g0, scale_cost)
    V, P = size(C)
    _check_nonempty(V, P)
    _warn_nonuniform_b(b, V)
    Cs = _prepare_cost(C, scale_cost)
    am = _align_marginal(a, P, Cs, "a")
    plan = fill!(similar(Cs), zero(T))
    for p in 1:P
        c = view(C, :, p)
        lo, hi = extrema(_rank, c)
        if hi > lo
            plan[argmin(v -> _rank(c[v]), eachindex(c)), p] = am[p]
        else
            plan[:, p] .= am[p] / V
        end
    end
    ent = Float64(dot(Cs, plan))
    return OTResult(plan, ent, nothing, nothing, true, 1)
end

"""
    TopKMeanSolver(; k=3)

Uniform mass over each particle's `k_eff = min(k, V)` lowest-cost vertices.
Non-finite costs rank last, and selected invalid vertices pass their mass to the
valid ones. A column with no finite spread spreads uniformly, so the particle holds.
"""
Base.@kwdef struct TopKMeanSolver <: AbstractOTSolver
    k::Int = 3
    function TopKMeanSolver(k)
        k >= 1 ||
            throw(ArgumentError("k must be >= 1, got $k (k = 0 would produce an all-zero plan and freeze every particle)"))
        new(k)
    end
end

function solve(s::TopKMeanSolver, C::AbstractMatrix{T}, eps::Real = 0.1;
        a = nothing, b = nothing, f0 = nothing, g0 = nothing,
        scale_cost = nothing) where {T}
    T <: AbstractFloat || return solve(s, float.(C), eps; a, b, f0, g0, scale_cost)
    V, P = size(C)
    _check_nonempty(V, P)
    _warn_nonuniform_b(b, V)
    Cs = _prepare_cost(C, scale_cost)
    am = _align_marginal(a, P, Cs, "a")
    k_eff = min(s.k, V)
    plan = fill!(similar(Cs), zero(T))
    picked = Vector{Int}(undef, k_eff)
    for p in 1:P
        c = view(C, :, p)
        lo, hi = extrema(_rank, c)
        if !(hi > lo)
            plan[:, p] .= am[p] / V
            continue
        end
        for j in 1:k_eff
            best = 0
            for v in 1:V
                v in view(picked, 1:(j - 1)) && continue
                (best == 0 || _rank(c[v]) < _rank(c[best])) && (best = v)
            end
            picked[j] = best
        end
        nvalid = count(v -> isfinite(c[v]), picked)
        mass = am[p] / (nvalid == 0 ? k_eff : nvalid)
        for v in picked
            (nvalid == 0 || isfinite(c[v])) && (plan[v, p] = mass)
        end
    end
    ent = Float64(dot(Cs, plan))
    return OTResult(plan, ent, nothing, nothing, true, 1)
end

"""
    KLSoftmaxSolver(; lam=Inf, max_iterations=2000, threshold=3e-5)

One-sided KL relaxation of `min_P <C,P> + eps*H(P) + lam*KL(P'1 || b)` with
particle marginal `a`: a log-domain fixed point with `alpha = lam/(lam+eps)`, exact
f-update and damped `g = alpha * g_target`. This is the unbalanced Sinkhorn iteration, so it converges
to the stated minimizer for every `lam`: `lam=0` is SoftmaxSolver, `lam=Inf` is
Sinkhorn, and intermediate `lam` soft-relaxes the vertex marginal toward `b`. It
stops when the dual step `max(|df|, |dg|/alpha)/eps` is at most `threshold`, or
unconverged once that step stalls at the dtype's precision. For `lam > 0`, zero-mass
entries of `a`/`b` get zero mass and `-Inf` duals. `last_marginal_violation` holds
`KL(P'1 || b)` of the last solve.
"""
Base.@kwdef mutable struct KLSoftmaxSolver <: AbstractOTSolver
    lam::Float64 = Inf
    max_iterations::Int = 2000
    threshold::Float64 = 3e-5
    last_marginal_violation::Union{Nothing, Float64} = nothing
    function KLSoftmaxSolver(lam, max_iterations, threshold, last_marginal_violation)
        lam >= 0 || throw(ArgumentError("lam must be >= 0, got $lam"))
        threshold > 0 || throw(ArgumentError("threshold must be > 0, got $threshold"))
        max_iterations >= 1 ||
            throw(ArgumentError("max_iterations must be >= 1, got $max_iterations"))
        new(lam, max_iterations, threshold, last_marginal_violation)
    end
end

kl_alpha(s::KLSoftmaxSolver, eps::Real) = 1 / (1 + eps / s.lam)

_xlogy(x, y) = iszero(x) ? zero(x) : x * log(y)

function solve(s::KLSoftmaxSolver, C::AbstractMatrix{T}, eps::Real;
        a = nothing, b = nothing, f0 = nothing, g0 = nothing,
        scale_cost = nothing) where {T}
    T <: AbstractFloat || return solve(s, float.(C), eps; a, b, f0, g0, scale_cost)
    _check_eps(eps)
    isfinite(eps) ||
        throw(ArgumentError("KLSoftmaxSolver requires a finite epsilon, got $eps (use SoftmaxSolver for the infinite-temperature/uniform limit)"))
    V, P = size(C)
    _check_nonempty(V, P)
    epsT = T(eps)
    (isfinite(epsT) && epsT > 0) || throw(ArgumentError("epsilon $eps is not representable in $T"))
    Cs = _prepare_cost(C, scale_cost)
    am = _align_marginal(a, P, Cs, "a")
    bm = _align_marginal(b, V, Cs, "b")
    alpha = T(kl_alpha(s, eps))
    alpha >= floatmin(T) && (any(iszero, am) || any(iszero, bm)) &&
        return _solve_on_support(s, Cs, eps, am, bm, f0, g0)
    shift = dot(am, vec(minimum(Cs; dims = 1)))
    center_cols!(Cs)
    if isinf(s.lam)
        ta, tb = sum(am), sum(bm)
        abs(ta - tb) <= sqrt(Base.eps(T)) * max(ta, tb) ||
            throw(ArgumentError("KLSoftmaxSolver(lam=Inf) marginals must have equal total mass; got sum(a)=$ta, sum(b)=$tb"))
    end

    log_a = log.(am)
    log_b = log.(bm)
    logK = Cs ./ (-epsT)

    fa = _align_dual(f0, P, T, "f0")
    ga = _align_dual(g0, V, T, "g0")
    f = fa === nothing ? zeros(T, P) : fa
    g = ga === nothing ? zeros(T, V) : ga
    if !(all(isfinite, f) && all(isfinite, g))
        fill!(f, zero(T))
        fill!(g, zero(T))
    end
    converged = false
    n_iters = s.max_iterations
    ftmp = zeros(T, P)
    gtmp = zeros(T, V)

    if alpha < floatmin(T)
        fill!(g, zero(T))
        converged = true
        n_iters = 1
    else
        accm = zeros(T, V)
        accs = zeros(T, V)
        f_new = similar(f)
        g_new = similar(g)
        fdiv = similar(f)
        gdiv = similar(g)
        prev_step = T(Inf)
        for it in 1:(s.max_iterations)
            @. gdiv = g / epsT
            lse_cols!(ftmp, logK, gdiv)
            @. f_new = epsT * (log_a - ftmp)
            @. fdiv = f_new / epsT
            lse_rows!(gtmp, logK, fdiv, accm, accs)
            @. g_new = alpha * epsT * (log_b - gtmp)
            df = zero(T)
            @inbounds for i in eachindex(f)
                df = max(df, abs(f_new[i] - f[i]))
            end
            dg = zero(T)
            @inbounds for i in eachindex(g)
                dg = max(dg, abs(g_new[i] - g[i]))
            end
            step = max(df, dg / alpha) / epsT
            f, f_new = f_new, f
            g, g_new = g_new, g
            n_iters = it
            if !isfinite(step)
                fill!(f, zero(T))
                fill!(g, zero(T))
                break
            end
            if step <= s.threshold
                converged = true
                break
            end
            prev_step <= step <= 2 * Base.eps(T) * max(_maxabs(f), _maxabs(g)) / epsT && break
            prev_step = step
        end
    end
    lse_cols!(ftmp, logK, g ./ epsT)
    @. f = epsT * (log_a - ftmp)
    plan = softmax_cols!(similar(Cs), Cs .- g, epsT) .* am'
    ent = Float64(dot(Cs, plan) + shift)
    q = vec(sum(plan; dims = 2))
    s.last_marginal_violation = Float64(sum(_xlogy.(q, q) .- _xlogy.(q, bm) .- q .+ bm))
    return OTResult(plan, ent, f, g, converged, n_iters)
end
