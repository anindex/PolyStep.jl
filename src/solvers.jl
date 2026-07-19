# OT weighting solvers. Layout: cost/plan are (V, P); f has length P (particles),
# g has length V (vertices). epsilon is a `solve` argument, not solver state
# (TemperedSoftmaxSolver ignores it).

abstract type AbstractOTSolver end

"""
    OTResult

`plan::(V,P)` transport (column p sums to ~ a[p] for one-sided solvers),
`ent_cost` a per-solver cost proxy (Sinkhorn reports the dual value, one-sided
solvers report the transport cost `<C,P>`; it is not the regularized primal
`<C,P> + eps*KL`), warm-startable duals `f::(P,)` / `g::(V,)` (nothing for
non-iterative solvers), `converged`, `iters`, per-check marginal `errors`
(Sinkhorn diagnostics, else nothing).
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
    sum(a) > 0 || throw(ArgumentError("$name must have positive total mass"))
    out = similar(C, T, n)
    copyto!(out, a)
    return out
end

# sanitize + scale into a fresh working copy (never mutates the caller's C);
# scale_cost!(X, X, spec) is alias-safe (pure broadcast divide).
function _prepare_cost(C::AbstractMatrix{T}, scale_cost) where {T}
    Cs = similar(C)
    copyto!(Cs, C)
    sanitize_cost!(Cs)
    scale_cost === nothing || scale_cost!(Cs, Cs, scale_cost)
    return Cs
end

function _check_eps(eps)
    eps > 0 ||
        throw(ArgumentError("epsilon must be > 0, got $eps (it is the temperature in softmax(-C/epsilon))"))
end

function _check_nonempty(V::Integer, P::Integer)
    (V == 0 || P == 0) &&
        throw(ArgumentError("cost matrix must be non-empty, got (V=$V, P=$P)"))
end

# One-shot small-epsilon warning, re-armed whenever eps drops into a new
# smaller-eps regime (per-solver min-checked state, not maxlog=1).
function _warn_tiny_eps!(s, eps::Real, Cs::AbstractMatrix)
    if eps < s.min_eps_checked
        s.min_eps_checked = Float64(eps)
        cost_max = isempty(Cs) ? 0.0 : Float64(maximum(abs, Cs))
        if cost_max > 0 && eps < TINY_EPSILON_RATIO * cost_max
            @warn "epsilon=$(eps) is very small relative to the cost scale (max |C|=$(cost_max)); " *
                  "-C/epsilon may overflow before the column-max subtraction. " *
                  "Consider rescaling the cost or raising epsilon."
        end
    end
    return nothing
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

# ---------------------------------------------------------------------------
# SoftmaxSolver (port of solvers/softmax.py), the default solver
# ---------------------------------------------------------------------------

"""
    SoftmaxSolver()

One-sided softmax weighting: `W = softmax(-C/eps, dims=1)`,
`plan = W .* a'` (column sums equal `a`). No duals, single pass.

Solver instances carry a mutable warn latch, single-run-per-instance; do not
share one across concurrent optimizations.
"""
mutable struct SoftmaxSolver <: AbstractOTSolver
    min_eps_checked::Float64
end
SoftmaxSolver() = SoftmaxSolver(Inf)

function solve(s::SoftmaxSolver, C::AbstractMatrix{T}, eps::Real;
        a = nothing, b = nothing, f0 = nothing, g0 = nothing,
        scale_cost = nothing, last_eps = nothing) where {T}
    T <: AbstractFloat || return solve(s, float.(C), eps; a, b, f0, g0, scale_cost, last_eps)
    _check_eps(eps)
    V, P = size(C)
    _check_nonempty(V, P)
    _warn_nonuniform_b(b, V)
    Cs = _prepare_cost(C, scale_cost)
    am = _align_marginal(a, P, Cs, "a")
    _warn_tiny_eps!(s, eps, Cs)
    W = similar(Cs)
    softmax_cols!(W, Cs, eps)
    plan = W .* am'
    ent = Float64(dot(Cs, plan))
    return OTResult(plan, ent, nothing, nothing, true, 1)
end

# ---------------------------------------------------------------------------
# TemperedSoftmaxSolver (port of solvers/tempered_softmax.py)
# ---------------------------------------------------------------------------

"""
    TemperedSoftmaxSolver(; tau=1.0)

Softmax at fixed temperature `tau`; the `eps` argument to `solve` is accepted
and ignored (decouples softmax sharpness from the eps schedule).
"""
Base.@kwdef mutable struct TemperedSoftmaxSolver <: AbstractOTSolver
    tau::Float64 = 1.0
    min_eps_checked::Float64 = Inf
end

function solve(s::TemperedSoftmaxSolver, C::AbstractMatrix{T}, eps::Real;
        a = nothing, b = nothing, f0 = nothing, g0 = nothing,
        scale_cost = nothing, last_eps = nothing) where {T}
    T <: AbstractFloat || return solve(s, float.(C), eps; a, b, f0, g0, scale_cost, last_eps)
    s.tau > 0 || throw(ArgumentError("tau must be > 0, got $(s.tau)"))
    V, P = size(C)
    _check_nonempty(V, P)
    _warn_nonuniform_b(b, V)
    Cs = _prepare_cost(C, scale_cost)
    am = _align_marginal(a, P, Cs, "a")
    W = similar(Cs)
    softmax_cols!(W, Cs, s.tau)
    plan = W .* am'
    ent = Float64(dot(Cs, plan))
    return OTResult(plan, ent, nothing, nothing, true, 1)
end

# ---------------------------------------------------------------------------
# Greedy ablation solvers (no sanitize, matching the reference; ties break to first)
# ---------------------------------------------------------------------------

"""
    MinCostGreedySolver()

All of each particle's mass on its single lowest-cost vertex. Costs must be
finite: this solver does not sanitize (matching the Python reference), so a NaN
cost could be selected.
"""
struct MinCostGreedySolver <: AbstractOTSolver end

function solve(s::MinCostGreedySolver, C::AbstractMatrix{T}, eps::Real = 0.1;
        a = nothing, b = nothing, f0 = nothing, g0 = nothing,
        scale_cost = nothing, last_eps = nothing) where {T}
    T <: AbstractFloat || return solve(s, float.(C), eps; a, b, f0, g0, scale_cost, last_eps)
    V, P = size(C)
    _check_nonempty(V, P)
    _warn_nonuniform_b(b, V)
    Cs = similar(C)
    copyto!(Cs, C)
    scale_cost === nothing || scale_cost!(Cs, Cs, scale_cost)
    am = _align_marginal(a, P, Cs, "a")
    plan = similar(Cs)
    fill!(plan, zero(T))
    for p in 1:P
        v = argmin(view(Cs, :, p))   # first occurrence on ties, like torch argmin
        plan[v, p] = am[p]
    end
    ent = Float64(dot(Cs, plan))
    return OTResult(plan, ent, nothing, nothing, true, 1)
end

"""
    TopKMeanSolver(; k=3)

Uniform mass `a[p]/k_eff` over each particle's `k_eff = min(k, V)` lowest-cost
vertices. Costs must be finite: this solver does not sanitize (matching the
Python reference).
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
        scale_cost = nothing, last_eps = nothing) where {T}
    T <: AbstractFloat || return solve(s, float.(C), eps; a, b, f0, g0, scale_cost, last_eps)
    V, P = size(C)
    _check_nonempty(V, P)
    _warn_nonuniform_b(b, V)
    Cs = similar(C)
    copyto!(Cs, C)
    scale_cost === nothing || scale_cost!(Cs, Cs, scale_cost)
    am = _align_marginal(a, P, Cs, "a")
    k_eff = min(s.k, V)
    plan = similar(Cs)
    fill!(plan, zero(T))
    for p in 1:P
        # stable sort -> lowest-index-among-ties, a convention we define
        # (torch.topk guarantees no tie order; exact parity is impossible)
        idx = sortperm(view(Cs, :, p); alg = Base.Sort.DEFAULT_STABLE)
        mass = am[p] / k_eff
        for j in 1:k_eff
            plan[idx[j], p] = mass
        end
    end
    ent = Float64(dot(Cs, plan))
    return OTResult(plan, ent, nothing, nothing, true, 1)
end

# ---------------------------------------------------------------------------
# KLSoftmaxSolver: iterative alpha-blended loop, softmax (lam=0) <-> Sinkhorn (lam=Inf)
# ---------------------------------------------------------------------------

"""
    KLSoftmaxSolver(; lam=Inf, max_iterations=2000, threshold=1e-6)

One-sided KL relaxation of `min_P <C,P> + eps*H(P) + lam*KL(P'1 || b)` s.t.
`P1 = a`: a log-domain interpolating fixed-point with `alpha = lam/(lam+eps)`,
exact f-update and damped `g = alpha * g_target`. This is the unbalanced (KL-relaxed)
Sinkhorn scaling iteration, so it converges to the stated minimizer for every `lam`,
not only at the endpoints: `lam=0` == SoftmaxSolver, `lam=Inf` == Sinkhorn, and
intermediate `lam` soft-relaxes the row marginal toward `b`.
"""
Base.@kwdef mutable struct KLSoftmaxSolver <: AbstractOTSolver
    lam::Float64 = Inf
    max_iterations::Int = 2000
    threshold::Float64 = 1e-6
    last_marginal_violation::Union{Nothing, Float64} = nothing
    function KLSoftmaxSolver(lam, max_iterations, threshold, last_marginal_violation)
        lam >= 0 || throw(ArgumentError("lam must be >= 0, got $lam"))
        # strict delta < threshold test (no fixed-iteration mode), so 0 never converges
        threshold > 0 || throw(ArgumentError("threshold must be > 0, got $threshold"))
        max_iterations >= 1 ||
            throw(ArgumentError("max_iterations must be >= 1, got $max_iterations"))
        new(lam, max_iterations, threshold, last_marginal_violation)
    end
end

function kl_alpha(s::KLSoftmaxSolver, eps::Real)
    s.lam == 0 ? 0.0 : (isinf(s.lam) ? 1.0 : s.lam / (s.lam + eps))
end

function solve(s::KLSoftmaxSolver, C::AbstractMatrix{T}, eps::Real;
        a = nothing, b = nothing, f0 = nothing, g0 = nothing,
        scale_cost = nothing, last_eps = nothing) where {T}
    T <: AbstractFloat || return solve(s, float.(C), eps; a, b, f0, g0, scale_cost, last_eps)
    _check_eps(eps)
    # the log-domain kernel Cs/(-eps) is undefined at eps=Inf (the alpha blend and
    # dual iteration cannot represent infinite temperature; use SoftmaxSolver for that)
    isfinite(eps) ||
        throw(ArgumentError("KLSoftmaxSolver requires a finite epsilon, got $eps (use SoftmaxSolver for the infinite-temperature/uniform limit)"))
    V, P = size(C)
    _check_nonempty(V, P)
    Cs = _prepare_cost(C, scale_cost)
    # per-column shift invariance (matches SinkhornSolver): keeps -Cs/eps and the
    # reconstructed plan finite at extreme cost magnitudes, and makes the lam=0 path
    # numerically identical to SoftmaxSolver (which centers via cmin - C)
    center_cols!(Cs)
    am = _align_marginal(a, P, Cs, "a")
    bm = _align_marginal(b, V, Cs, "b")
    epsT = T(eps)
    alpha = T(kl_alpha(s, eps))
    # alpha == 1 (lam = Inf) is exact Sinkhorn: two-sided OT is infeasible for
    # mismatched total mass. Guard it like SinkhornSolver rather than return a
    # wrong plan
    if alpha == 1
        ta, tb = sum(am), sum(bm)
        abs(ta - tb) <= sqrt(Base.eps(T)) * max(ta, tb) ||
            throw(ArgumentError("KLSoftmaxSolver(lam=Inf) marginals must have equal total mass; got sum(a)=$ta, sum(b)=$tb"))
    end

    log_a = log.(max.(am, T(1e-30)))
    log_b = log.(max.(bm, T(1e-30)))
    logK = Cs ./ (-epsT)

    # warm-start pipeline mirrors Sinkhorn: shape-align, rescale, finite-guard
    fa = _align_dual(f0, P, T, "f0")
    ga = _align_dual(g0, V, T, "g0")
    f = fa === nothing ? zeros(T, P) : fa
    g = ga === nothing ? zeros(T, V) : ga
    # KL duals scale ~O(eps): rescale warm starts when the schedule moved eps
    # (same heuristic as Sinkhorn; Python never warm-starts KL in a loop, so
    # this path is Julia-specific and required for schedule + warm-start use)
    if last_eps !== nothing && last_eps > 0 && (f0 !== nothing || g0 !== nothing) &&
       abs(last_eps - eps) / max(eps, 1e-9) > 1e-6
        sc = T(eps / last_eps)
        f .*= sc
        g .*= sc
    end
    if !(all(isfinite, f) && all(isfinite, g))
        fill!(f, zero(T))
        fill!(g, zero(T))
    end
    converged = false
    n_iters = s.max_iterations
    ftmp = zeros(T, P)
    gtmp = zeros(T, V)

    if alpha == 0
        # softmax limit: closed form, g-independent (Python ignores any warm g0);
        # only ftmp/gtmp are needed, gtmp stays zero as the lse_cols! add term
        lse_cols!(ftmp, logK, gtmp)
        @. f = epsT * (log_a - ftmp)
        fill!(g, zero(T))
        converged = true
        n_iters = 1
    else
        accm = zeros(T, V)
        accs = zeros(T, V)
        check_every = max(1, div(s.max_iterations, 20))
        # preallocate the dual buffers and ping-pong (f/g are private copies from
        # _align_dual, safe to swap) instead of binding two fresh arrays per iter
        f_new = similar(f)
        g_new = similar(g)
        fdiv = similar(f)
        gdiv = similar(g)
        for it in 1:(s.max_iterations)
            @. gdiv = g / epsT
            lse_cols!(ftmp, logK, gdiv)
            @. f_new = epsT * (log_a - ftmp)
            @. fdiv = f_new / epsT
            lse_rows!(gtmp, logK, fdiv, accm, accs)
            @. g_new = alpha * epsT * (log_b - gtmp)
            if it % check_every == 0 || it == s.max_iterations
                delta = zero(T)
                @inbounds for i in eachindex(f)
                    delta = max(delta, abs(f_new[i] - f[i]))
                end
                @inbounds for i in eachindex(g)
                    delta = max(delta, abs(g_new[i] - g[i]))
                end
                if delta < s.threshold
                    f, g = f_new, g_new
                    converged = true
                    n_iters = it
                    break
                end
            end
            f, f_new = f_new, f
            g, g_new = g_new, g
        end
    end

    plan = exp.((f' .+ g .- Cs) ./ epsT)
    if !all(isfinite, plan)
        @warn "KLSoftmaxSolver produced non-finite transport entries; consider raising epsilon or lowering lam."
        # Inf entries are the lowest-cost vertices; zeroing them would hand mass to
        # the worst vertices, so saturate at floatmax/V (column sums stay finite);
        # NaN maps to zero.
        plan .= ifelse.(isnan.(plan), zero(T), min.(plan, floatmax(T) / V))
    end
    # Cs was per-column centered above, so this is a diagnostic proxy, not raw <C,P>
    ent = Float64(dot(Cs, plan))
    # instrumentation: KL(P'1 || b) on the realized per-vertex mass
    q = max.(vec(sum(plan; dims = 2)), T(1e-30))
    bs = max.(bm, T(1e-30))
    s.last_marginal_violation = Float64(sum(q .* (log.(q) .- log.(bs))))
    return OTResult(plan, ent, f, g, converged, n_iters)
end
