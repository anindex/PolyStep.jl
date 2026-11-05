mutable struct SinkhornWorkspace{T}
    Cs::Matrix{T}
    logK::Matrix{T}
    log_a::Vector{T}
    log_b::Vector{T}
    ftmp::Vector{T}
    gtmp::Vector{T}
    fdiv::Vector{T}
    gdiv::Vector{T}
    accm::Vector{T}
    accs::Vector{T}
end
function SinkhornWorkspace{T}(V::Int, P::Int) where {T}
    SinkhornWorkspace{T}(
        Matrix{T}(undef, V, P), Matrix{T}(undef, V, P),
        Vector{T}(undef, P), Vector{T}(undef, V),
        Vector{T}(undef, P), Vector{T}(undef, V),
        Vector{T}(undef, P), Vector{T}(undef, V),
        Vector{T}(undef, V), Vector{T}(undef, V))
end

"""
    SinkhornSolver(; max_iterations=2000, threshold=3e-5, check_every=10,
                   omega=1.0, anderson_depth=0, adaptive_omega=false,
                   data_dependent_init=false)

Entropic OT on the `(V, P)` cost: `min_P <C,P> + eps*sum(P .* (log.(P) .- 1))`
with vertex marginal `b` (row sums) and particle marginal `a` (column sums),
solved by log-domain alternating dual updates with optional overrelaxation
`omega in [0.5, 1.95]`, warm-started duals, Anderson acceleration (accepted only
when it raises the dual), the residual-ratio adaptive omega (estimated while
`omega == 1`), and a divergence back-off for `omega > 1.5`. It stops when the
relative marginal error `max|expm1(log(P'1) - log(a))|` drops below `threshold`,
or unconverged once the error stalls at the dtype's precision. `threshold <= 0`
runs a fixed number of iterations and reports `converged` when the result is
finite. Zero-mass entries of `a`/`b` get zero mass and `-Inf` duals. `ent_cost` is the
dual value in the sanitized, scaled cost frame.

An instance keeps a workspace between calls: do not call `solve` on one instance
from two tasks at once (`init_state` and `PolyStepES` copy the solver). The
solve is a CPU path.
"""
Base.@kwdef mutable struct SinkhornSolver <: AbstractOTSolver
    max_iterations::Int = 2000
    threshold::Float64 = 3e-5
    check_every::Int = 10
    omega::Float64 = 1.0
    anderson_depth::Int = 0
    adaptive_omega::Bool = false
    data_dependent_init::Bool = false
    ws::Union{Nothing, SinkhornWorkspace} = nothing
    function SinkhornSolver(max_iterations, threshold, check_every, omega,
            anderson_depth, adaptive_omega, data_dependent_init, ws = nothing)
        0.5 <= omega <= 1.95 ||
            throw(ArgumentError("omega must be in [0.5, 1.95], got $omega (values < 0.5 diverge; > 1.95 are unstable)"))
        check_every >= 1 || throw(ArgumentError("check_every must be >= 1, got $check_every"))
        max_iterations >= 1 ||
            throw(ArgumentError("max_iterations must be >= 1, got $max_iterations"))
        new(max_iterations, threshold, check_every, omega,
            anderson_depth, adaptive_omega, data_dependent_init, ws)
    end
end

function _sinkhorn_ws(s::SinkhornSolver, ::Type{T}, V::Int, P::Int) where {T}
    ws = s.ws
    ws isa SinkhornWorkspace{T} && size(ws.Cs) == (V, P) && return ws
    fresh = SinkhornWorkspace{T}(V, P)
    s.ws = fresh
    return fresh
end

function _align_dual(init, n::Integer, ::Type{T}, name::String) where {T}
    init === nothing && return nothing
    if length(init) != n
        @warn "$name has length $(length(init)), expected $n; ignoring warm start."
        return nothing
    end
    return convert(Vector{T}, collect(init))
end

function _dual_objective(f::AbstractVector{T}, g::AbstractVector{T}, logK::AbstractMatrix{T},
        am, bm, epsT::T, tmpP::AbstractVector{T}, gdiv::AbstractVector{T}) where {T}
    @. gdiv = g / epsT
    lse_cols!(tmpP, logK, gdiv)
    @. tmpP += f / epsT
    m = maximum(tmpP)
    s = zero(T)
    @inbounds @simd for i in eachindex(tmpP)
        s += exp_fast(tmpP[i] - m)
    end
    mass = exp(m + log(s))
    return dot(f, am) + dot(g, bm) - epsT * mass
end

function _sor_iterate!(f, g, omegaT, epsT, logK, log_a, log_b,
        ftmp, gtmp, fdiv, gdiv, accm, accs)
    gdiv .= g ./ epsT
    lse_cols!(ftmp, logK, gdiv)
    @. f = (1 - omegaT) * f + omegaT * epsT * (log_a - ftmp)
    fdiv .= f ./ epsT
    lse_rows!(gtmp, logK, fdiv, accm, accs)
    @. g = (1 - omegaT) * g + omegaT * epsT * (log_b - gtmp)
    return nothing
end

function solve(s::SinkhornSolver, C::AbstractMatrix{T}, eps::Real;
        a = nothing, b = nothing, f0 = nothing, g0 = nothing,
        scale_cost = nothing) where {T}
    T <: AbstractFloat || return solve(s, float.(C), eps; a, b, f0, g0, scale_cost)
    _check_eps(eps)
    isfinite(eps) ||
        throw(ArgumentError("Sinkhorn requires a finite epsilon, got $eps (the log-domain dual iteration is undefined at infinite temperature)"))
    V, P = size(C)
    (V == 0 || P == 0) &&
        throw(ArgumentError("Sinkhorn received an empty cost matrix (V=$V, P=$P)"))
    epsT = T(eps)
    (isfinite(epsT) && epsT > 0) || throw(ArgumentError("epsilon $eps is not representable in $T"))
    ws = C isa Matrix{T} ? _sinkhorn_ws(s, T, V, P) : nothing
    Cs = ws === nothing ? _prepare_cost(C, scale_cost) : _prepare_cost!(ws.Cs, C, scale_cost)
    am = _align_marginal(a, P, Cs, "a")
    bm = _align_marginal(b, V, Cs, "b")
    tot_a, tot_b = sum(am), sum(bm)
    abs(tot_a - tot_b) <= sqrt(Base.eps(T)) * max(tot_a, tot_b) ||
        throw(ArgumentError("Sinkhorn marginals must have equal total mass; got sum(a)=$tot_a, sum(b)=$tot_b"))
    (any(iszero, am) || any(iszero, bm)) && return _solve_on_support(s, Cs, eps, am, bm, f0, g0)
    shift = dot(am, vec(minimum(Cs; dims = 1)))
    center_cols!(Cs)

    local logK, log_a, log_b
    if ws === nothing
        logK = Cs ./ (-epsT)
        log_a = log.(am)
        log_b = log.(bm)
    else
        logK = ws.logK
        @. logK = Cs / (-epsT)
        log_a = ws.log_a
        @. log_a = log(am)
        log_b = ws.log_b
        @. log_b = log(bm)
    end

    local ftmp, gtmp, fdiv, gdiv, accm, accs
    if ws === nothing
        ftmp = similar(Cs, T, P)
        gtmp = similar(Cs, T, V)
        fdiv = similar(Cs, T, P)
        gdiv = similar(Cs, T, V)
        accm = similar(Cs, T, V)
        accs = similar(Cs, T, V)
    else
        ftmp = ws.ftmp
        gtmp = ws.gtmp
        fdiv = ws.fdiv
        gdiv = ws.gdiv
        accm = ws.accm
        accs = ws.accs
    end

    if s.data_dependent_init && f0 === nothing && g0 === nothing
        f = fill!(similar(Cs, T, P), zero(T))
        g = fill!(similar(Cs, T, V), zero(T))
        _sor_iterate!(f, g, one(T), epsT, logK, log_a, log_b,
            ftmp, gtmp, fdiv, gdiv, accm, accs)
    else
        fa = _align_dual(f0, P, T, "f0")
        ga = _align_dual(g0, V, T, "g0")
        # same array backend as Cs
        f = fa === nothing ? fill!(similar(Cs, T, P), zero(T)) : fa
        g = ga === nothing ? fill!(similar(Cs, T, V), zero(T)) : ga
    end

    # clamp warm starts to the cost magnitude
    cost_scale = max(maximum(abs, Cs), T(1e-6))
    mad = 10 * cost_scale
    if !(all(isfinite, f) && all(isfinite, g))
        fill!(f, zero(T))
        fill!(g, zero(T))
    else
        clamp!(f, -mad, mad)
        clamp!(g, -mad, mad)
    end

    c = T(0.5) * (mean(g) - mean(f))
    f .+= c
    g .-= c

    fixed_mode = s.threshold <= 0
    if fixed_mode
        s.anderson_depth > 0 &&
            @warn "anderson_depth > 0 has no effect in fixed-iteration mode (threshold <= 0)."
        s.adaptive_omega &&
            @warn "adaptive_omega=true has no effect in fixed-iteration mode (threshold <= 0)."
    end

    converged = false
    dual_reset = false
    n_iters = 0
    errors = Float64[]
    omega = s.omega
    omegaT = T(omega)

    if fixed_mode
        for i in 1:(s.max_iterations)
            _sor_iterate!(f, g, omegaT, epsT, logK, log_a, log_b,
                ftmp, gtmp, fdiv, gdiv, accm, accs)
            n_iters = i
        end
        if !(all(isfinite, f) && all(isfinite, g))
            fill!(f, zero(T))
            fill!(g, zero(T))
            dual_reset = true
        else
            converged = true
        end
    else
        aa_x = Tuple{Vector{T}, Vector{T}}[]
        aa_r = Tuple{Vector{T}, Vector{T}}[]
        prev_err = NaN
        div_prev_norm = Inf
        div_growth = 0
        div_patience = 3
        omega_capped = false

        for i in 1:(s.max_iterations)
            use_anderson = s.anderson_depth > 0 && i % s.check_every == 0
            local f_old, g_old
            if use_anderson
                f_old = copy(f)
                g_old = copy(g)
            end
            _sor_iterate!(f, g, omegaT, epsT, logK, log_a, log_b,
                ftmp, gtmp, fdiv, gdiv, accm, accs)

            if use_anderson
                r_f = f .- f_old
                r_g = g .- g_old
                push!(aa_x, (f_old, g_old))
                push!(aa_r, (copy(r_f), copy(r_g)))
                if length(aa_x) > s.anderson_depth + 1
                    popfirst!(aa_x)
                    popfirst!(aa_r)
                end
                if length(aa_r) >= 2
                    k = length(aa_r) - 1
                    delta_r = Matrix{T}(undef, P + V, k)
                    for j in 1:k
                        delta_r[1:P, j] .= aa_r[j + 1][1] .- aa_r[j][1]
                        delta_r[(P + 1):end, j] .= aa_r[j + 1][2] .- aa_r[j][2]
                    end
                    current_r = vcat(r_f, r_g)
                    lam = max(T(1e-8) * sum(abs2, delta_r), T(1e-30))
                    augA = vcat(delta_r, sqrt(lam) * Matrix{T}(I, k, k))
                    augb = vcat(current_r, zeros(T, k))
                    alpha = try
                        augA \ augb
                    catch
                        nothing
                    end
                    if alpha !== nothing && all(isfinite, alpha) && norm(alpha) < 1e3
                        delta_x = Matrix{T}(undef, P + V, k)
                        for j in 1:k
                            delta_x[1:P, j] .= aa_x[j + 1][1] .- aa_x[j][1]
                            delta_x[(P + 1):end, j] .= aa_x[j + 1][2] .- aa_x[j][2]
                        end
                        # type-II Anderson: G(x) - dG*alpha with dG = dX + dR
                        combined = vcat(f, g) .- (delta_x .+ delta_r) * alpha
                        if all(isfinite, combined)
                            f_c = combined[1:P]
                            g_c = combined[(P + 1):end]
                            lyap_prev = _dual_objective(
                                f_old, g_old, logK, am, bm, epsT, ftmp, gdiv)
                            lyap_plain = _dual_objective(f, g, logK, am, bm, epsT, ftmp, gdiv)
                            lyap_comb = _dual_objective(
                                f_c, g_c, logK, am, bm, epsT, ftmp, gdiv)
                            if lyap_comb >= lyap_plain - 1e-6 && lyap_comb >= lyap_prev - 1e-6
                                f = f_c
                                g = g_c
                            end
                        end
                    end
                end
            end
            n_iters = i

            if i % s.check_every == 0 || i == s.max_iterations
                if !(all(isfinite, f) && all(isfinite, g))
                    fill!(f, zero(T))
                    fill!(g, zero(T))
                    dual_reset = true
                    break
                end
                gdiv .= g ./ epsT
                lse_cols!(ftmp, logK, gdiv)
                marg_a_err = zero(T)
                @inbounds for p in 1:P
                    marg_a_err = max(marg_a_err, abs(expm1(f[p] / epsT + ftmp[p] - log_a[p])))
                end
                marg_b_err = zero(T)
                if omega != 1 || s.anderson_depth > 0 || s.adaptive_omega
                    fdiv .= f ./ epsT
                    lse_rows!(gtmp, logK, fdiv, accm, accs)
                    @inbounds for v in 1:V
                        marg_b_err = max(marg_b_err, abs(expm1(g[v] / epsT + gtmp[v] - log_b[v])))
                    end
                end
                err = Float64(max(marg_a_err, marg_b_err))
                omega_old = omega

                if omega > 1.5
                    dual_norm = Float64(maximum(abs, f) + maximum(abs, g))
                    if dual_norm > div_prev_norm * 1.05
                        div_growth += 1
                        if div_growth >= div_patience
                            @warn "Sinkhorn divergence detected with omega=$(round(omega; digits=2)) " *
                                  "after $div_patience consecutive growth checks; backing omega off to 1.0."
                            omega = 1.0
                            s.omega = 1.0
                            omegaT = T(omega)
                            omega_capped = true
                            div_growth = 0
                        end
                    else
                        div_growth = 0
                    end
                    div_prev_norm = dual_norm
                end

                if s.adaptive_omega && !omega_capped && omega == 1.0
                    if !isnan(prev_err) && prev_err > 1e-12 && 0 < err < prev_err
                        r = min(err / prev_err, 0.99)
                        omega = clamp(2.0 / (1.0 + sqrt(1.0 - r^(1.0 / s.check_every))), 1.0, 1.95)
                        omegaT = T(omega)
                    end
                    prev_err = err
                end
                # Anderson history holds residuals of the old map: restart it
                if omega != omega_old
                    empty!(aa_x)
                    empty!(aa_r)
                end

                push!(errors, err)
                if err < s.threshold
                    converged = true
                    break
                end
                n = length(errors)
                if n > 3 && minimum(@view errors[(n - 2):n]) >= minimum(@view errors[1:(n - 3)])
                    dmax = Float64(_maxabs(f) + _maxabs(g))
                    stall = 2 * Base.eps(T) * ((dmax + min(dmax, Float64(cost_scale))) / epsT + 1)
                    err < stall && break
                end
            end
        end
    end

    ent_reg_cost = Float64(_dual_objective(f, g, logK, am, bm, epsT, ftmp, gdiv) + shift)
    logP = ws === nothing ? ((f' .+ g .- Cs) ./ epsT) : (@. ws.logK = (f' + g - Cs) / epsT)
    plan = exp.(logP)
    @inbounds for p in 1:P
        colfinite = true
        colmass = zero(T)
        for v in 1:V
            x = plan[v, p]
            colfinite &= isfinite(x)
            colmass += x
        end
        if dual_reset && colfinite && colmass > 0
            for v in 1:V
                plan[v, p] *= am[p] / colmass
            end
        elseif !colfinite || colmass == 0
            m = maximum(view(logP, :, p))
            if isfinite(m)
                z = zero(T)
                for v in 1:V
                    z += exp_fast(logP[v, p] - m)
                end
                for v in 1:V
                    plan[v, p] = am[p] * (exp_fast(logP[v, p] - m) / z)
                end
            else
                for v in 1:V
                    plan[v, p] = zero(T)
                end
            end
        end
    end
    if converged && !fixed_mode
        tol = max(2 * s.threshold, sqrt(Base.eps(T)))
        colerr = maximum(p -> abs(sum(view(plan, :, p)) / am[p] - 1), 1:P)
        rowerr = maximum(v -> abs(sum(view(plan, v, :)) / bm[v] - 1), 1:V)
        converged = max(colerr, rowerr) <= tol
    end
    return OTResult(plan, ent_reg_cost, f, g, converged, n_iters,
        isempty(errors) ? nothing : errors)
end
