# Core PolyStep step loop. Layout: X is (d, P); Xprobe is (d, K*V*P) with column
# c = (p-1)*V*K + (v-1)*K + k (k fastest, matches torch reshape(-1, d) order).

"""
    PolyStepConfig(; dim, kwargs...)

Gradient-free polytope direct-search optimizer configuration. The objective is
batched: `f(X::AbstractMatrix)::AbstractVector` maps `(d, N)` candidate
columns to `N` costs (lower better); wrap a scalar function with
[`columnwise`](@ref). Costs are measured at fractions `k/(num_probe+1)`,
k = 1..num_probe, of `probe_radius` (see [`probe_scales`](@ref)), so the
default single probe sits at half of it, the same distance as the default step.
The step is taken at `step_radius` (decoupled). `epsilon`/`step_radius`/
`probe_radius` accept a Real or an epsilon schedule; unscheduled radii are
multiplied by the current epsilon (scheduled ones are not).

Dimension scaling: when vertex cost differences are small next to the OT
temperature, one step is gradient descent with rate
`sr * (pr/2) / (dim * eps_ot * s)` (`s` = `scale_cost` divisor), so at a fixed
temperature the step shrinks as `1/dim`. For a larger `dim`, scale the OT
temperature as `1/dim` through `ent_epsilon` (not `epsilon`, which also scales
both radii, so the step would shrink as `1/dim^2`).

Bounds and repair (these differ from `PolyStepES`): `lb`/`ub` clamp the
evaluated probe points and box-project the iterate after each step; `repair`
applies to evaluated candidates only. The iterate is a barycenter of template
geometry and is not repaired (its `best_x` incumbent always is). `repair` runs
after clamping and must itself preserve the bounds (re-clamping its output
would corrupt integrality repairs). For cached candidates
(evaluated == projected, repair respected end to end), use `PolyStepES`.
Either bound may be `+-Inf` per coordinate (half-bounded boxes).

`best_x`/`best_f` track the best evaluated point: every probe, plus the final
iterate that [`solve!`](@ref) scores once at the end (clamped and repaired).
"""
Base.@kwdef struct PolyStepConfig{S <: AbstractOTSolver, E, EE, SR, PR, SC, LB, UB, RP}
    dim::Int
    polytope::Symbol = :orthoplex
    solver::S = SoftmaxSolver()
    epsilon::E = 0.1
    ent_epsilon::EE = nothing
    scale_cost::SC = 1.0
    step_radius::SR = 1.0
    probe_radius::PR = 2.0
    probe_radius_jitter::Float64 = 0.0
    num_probe::Int = 1
    max_iterations::Int = 50
    min_iterations::Int = 5
    threshold::Float64 = 1e-3
    eval_chunk::Int = 0                 # 0 = single objective call per step
    lb::LB = nothing                    # box bounds, scalar or (d,)
    ub::UB = nothing
    repair::RP = nothing                # candidate post-processor f!(X_cols) before eval
    biased_rotation::Bool = false
    use_momentum::Bool = false
    momentum_init::Float64 = 0.5
    momentum_final::Float64 = 0.95
    velocity_lr::Float64 = 1.0
    use_adaptive_radius::Bool = false
    stagnation_threshold::Float64 = 1e-4
    stagnation_patience::Int = 10
    radius_increase::Float64 = 1.5
    radius_decrease::Float64 = 0.9
    radius_min::Float64 = 0.5
    radius_max::Float64 = 3.0
    use_quadratic_model::Bool = false
    newton_refinement::Bool = false
    newton_alpha::Float64 = 0.3
    trust_region::Bool = false
end

function _validate(ps::PolyStepConfig)
    ps.dim >= 1 || throw(ArgumentError("dim must be >= 1, got $(ps.dim)"))
    ps.num_probe >= 1 || throw(ArgumentError("num_probe must be >= 1, got $(ps.num_probe)"))
    ps.polytope === :cube && ps.dim > 6 &&
        throw(ArgumentError("cube polytope is capped at dim <= 6 (V = 2^dim vertices breaks the V = O(dim) cost model); use :orthoplex"))
    if ps.newton_refinement || ps.trust_region
        # stricter than Python, which silently no-ops these
        ps.use_quadratic_model ||
            throw(ArgumentError("newton_refinement / trust_region require use_quadratic_model=true (they consume its FD losses)"))
    end
    if ps.use_quadratic_model
        ps.polytope === :orthoplex ||
            throw(ArgumentError("quadratic model requires polytope = :orthoplex (FD extractors assume its vertex order)"))
        ps.num_probe >= 2 ||
            throw(ArgumentError("quadratic model requires num_probe >= 2 (Hessian regression degenerates at K = 1)"))
    end
    (ps.epsilon isa ProgressiveEpsilon && ps.ent_epsilon isa ProgressiveEpsilon) &&
        throw(ArgumentError("epsilon and ent_epsilon cannot both be ProgressiveEpsilon (only the OT schedule receives update! feedback; the radius schedule would stay pinned at init)"))
    (ps.epsilon isa ProgressiveEpsilon && ps.ent_epsilon !== nothing &&
     !(ps.ent_epsilon isa ProgressiveEpsilon)) &&
        throw(ArgumentError("epsilon is ProgressiveEpsilon but ent_epsilon overrides the OT schedule: update! feedback would never reach the radius schedule (pinned at init); make ent_epsilon the ProgressiveEpsilon or use a non-Progressive epsilon"))
    if ps.use_quadratic_model && ps.repair !== nothing
        @warn "use_quadratic_model with `repair`: FD extractors assume unmodified probe offsets; repaired probes corrupt the quadratic model (bounds-clamped probes are detected and masked, repaired ones cannot be)." maxlog = 1
    end
    sched = ps.ent_epsilon === nothing ? ps.epsilon : ps.ent_epsilon
    if sched isa ProgressiveEpsilon && !hasfield(typeof(ps.solver), :max_iterations)
        throw(ArgumentError("ProgressiveEpsilon requires an iterative solver (Sinkhorn/KLSoftmax); $(typeof(ps.solver)) always converges in one pass"))
    end
    (ps.lb === nothing) == (ps.ub === nothing) ||
        throw(ArgumentError("provide both lb and ub or neither"))
    ps.lb isa AbstractVector && length(ps.lb) != ps.dim &&
        throw(DimensionMismatch("lb has length $(length(ps.lb)), expected dim=$(ps.dim)"))
    ps.ub isa AbstractVector && length(ps.ub) != ps.dim &&
        throw(DimensionMismatch("ub has length $(length(ps.ub)), expected dim=$(ps.dim)"))
    ps.lb === nothing || _check_bounds(ps.lb, ps.ub)
    ps.step_radius isa ProgressiveEpsilon &&
        throw(ArgumentError("step_radius does not accept ProgressiveEpsilon (radii get no update! feedback); use a Real, LinearEpsilon, or CosineEpsilon"))
    ps.probe_radius isa ProgressiveEpsilon &&
        throw(ArgumentError("probe_radius does not accept ProgressiveEpsilon (radii get no update! feedback); use a Real, LinearEpsilon, or CosineEpsilon"))
    # constant values only; schedules validate their own outputs
    ps.epsilon isa Real && !(ps.epsilon > 0) &&
        throw(ArgumentError("epsilon must be > 0, got $(ps.epsilon)"))
    ps.ent_epsilon isa Real && !(ps.ent_epsilon > 0) &&
        throw(ArgumentError("ent_epsilon must be > 0, got $(ps.ent_epsilon)"))
    ps.epsilon isa Real && isinf(ps.epsilon) &&
        (ps.step_radius isa Real || ps.probe_radius isa Real) &&
        throw(ArgumentError("epsilon=Inf scales unscheduled radii to Inf; schedule step_radius and probe_radius, or use ent_epsilon=Inf"))
    # fail before the first objective batch, not inside step!
    ps.scale_cost === nothing || scale_cost!(zeros(1, 1), zeros(1, 1), ps.scale_cost)
    ps.step_radius isa Real && !(ps.step_radius >= 0) &&
        throw(ArgumentError("step_radius must be >= 0, got $(ps.step_radius)"))
    ps.probe_radius isa Real && !(ps.probe_radius >= 0) &&
        throw(ArgumentError("probe_radius must be >= 0, got $(ps.probe_radius)"))
    0 <= ps.probe_radius_jitter < 1 ||
        throw(ArgumentError("probe_radius_jitter must be in [0, 1), got $(ps.probe_radius_jitter) (>= 1 can drive the probe radius negative)"))
    return ps
end

"""
    columnwise(f; parallel=:serial) -> batched objective

Adapt a scalar objective `f(x::AbstractVector)::Real` to the batched
`(d, N) -> (N,)` protocol. `parallel = :threads` evaluates columns with
`Threads.@threads`; off by default because OR simulators are frequently stateful
and not thread-safe.
"""
function columnwise(f; parallel::Symbol = :serial)
    parallel in (:serial, :threads) ||
        throw(ArgumentError("parallel must be :serial or :threads, got :$parallel"))
    if parallel === :threads
        return _columnwise_threads(f)
    end
    return _columnwise_serial(f)
end

# ::F forces specialization on f (else dynamic dispatch per column)
function _columnwise_threads(f::F) where {F}
    return function (X::AbstractMatrix)
        out = Vector{float(eltype(X))}(undef, size(X, 2))
        # not :static, which throws when nested inside user threading
        Threads.@threads for j in 1:size(X, 2)
            out[j] = f(view(X, :, j))
        end
        out
    end
end

function _columnwise_serial(f::F) where {F}
    return X::AbstractMatrix -> [f(view(X, :, j)) for j in 1:size(X, 2)]
end

"""
    PolyStepState

Mutable optimizer state from [`init_state`](@ref): particles `X` `(d, P)`, the
incumbent `best_x`/`best_f`, `evals`, `iteration`, and the reused step buffers.
"""
mutable struct PolyStepState{T <: AbstractFloat, S <: AbstractOTSolver}
    X::Matrix{T}                        # (d, P)
    a::Vector{T}                        # (P,) uniform 1/P
    iteration::Int                      # 0-based count of completed steps
    f::Union{Nothing, Vector{T}}         # OT warm-start duals
    g::Union{Nothing, Vector{T}}
    last_eps::Union{Nothing, Float64}
    velocity::Union{Nothing, Matrix{T}}
    radius_multiplier::Float64
    stagnation_count::Int
    prev_loss::Float64
    tr_multiplier::Float64
    prev_predicted::Union{Nothing, Float64}
    prev_pre_step_loss::Union{Nothing, Float64}
    prev_descent::Union{Nothing, Matrix{T}}   # (d, P)
    best_x::Vector{T}                   # incumbent: best evaluated point
    best_f::Float64
    evals::Int
    # diagnostics
    costs::Vector{Float64}
    ot_converged::Vector{Bool}
    disp_sqnorms::Vector{Float64}
    trust_multipliers::Vector{Float64}
    dmax_peak::Float64                  # running max of disp_sqnorms
    # preallocated workspace
    Vt::Union{Nothing, Matrix{T}}        # (d, V) template, general polytope only
    scales::Vector{T}                   # (K,)
    R::Array{T, 3}                       # (d, d, P), also the Gaussian scratch
    dirs::Union{Nothing, Array{T, 3}}     # (d, V, P), general polytope only
    Xprobe::Matrix{T}                   # (d, K*V*P)
    losses::Vector{T}                   # (K*V*P,)
    losses3::Union{Nothing, Array{T, 3}}  # (K, V, P) raw per-probe, quadratic model
    Craw::Matrix{T}                     # (V, P) sanitized, unscaled
    Csolve::Matrix{T}                   # (V, P) scaled solver input
    Wn::Matrix{T}                       # (V, P) realized-mass-normalized plan
    cent::Matrix{T}                     # (d, P) template-space centroid
    delta::Matrix{T}                    # (d, P) rotated step direction
    Xprev::Matrix{T}                    # (d, P)
    bias::Matrix{T}                     # (d, P) normalized descent for biased rotation
    G::Union{Nothing, Matrix{T}}         # FD gradient / Hessian (quadratic model)
    H::Union{Nothing, Matrix{T}}
    N::Union{Nothing, Matrix{T}}          # Newton step (newton_refinement scratch)
    Xref::Union{Nothing, Matrix{T}}       # newton_refinement output buffer
    refrot::Union{Nothing, Matrix{T}}     # newton_refinement rotated-step buffer
    # state-local schedule copies: repeat/concurrent runs share no mutable state
    prog_eps::Union{Nothing, ProgressiveEpsilon}
    prog_ent::Union{Nothing, ProgressiveEpsilon}
    clampflag::Union{Nothing, Vector{Bool}}   # (P,) probe clamped; quad+bounds only
    # state-local solver copy (own workspace and warn latch)
    solver::S
    # no finite loss in the last batch (divergence signal; Craw is sanitized)
    last_all_nonfinite::Bool
end

"""
    init_state(ps::PolyStepConfig, X0::AbstractMatrix) -> PolyStepState

`X0` is `(dim, P)`: columns are particles; its element type sets the working
precision (Float64 recommended for OR use; Float32 is the perf/GPU mode).
"""
function init_state(ps::PolyStepConfig, X0::AbstractMatrix{T}) where {T <: AbstractFloat}
    _validate(ps)
    d, P = size(X0)
    d == ps.dim || throw(DimensionMismatch("X0 has $(d) rows, expected dim=$(ps.dim)"))
    P >= 1 ||
        throw(ArgumentError("X0 must have at least one column (particle), got size $(size(X0))"))
    # a non-finite particle would trip the NaN revert every step
    all(isfinite, X0) || throw(ArgumentError("X0 must be finite"))
    if P == 1 && (ps.solver isa SinkhornSolver ||
        (ps.solver isa KLSoftmaxSolver && isinf(ps.solver.lam)))
        @warn "Two-sided OT (Sinkhorn / KLSoftmax lam=Inf) with a single particle yields " *
              "a uniform transport plan (the column marginal forces it), so steps ignore " *
              "the costs. Use SoftmaxSolver or more particles." maxlog = 1
    end
    V = num_vertices(ps.polytope, d)
    K = ps.num_probe
    quad = ps.use_quadratic_model
    solver = deepcopy(ps.solver)
    losses = zeros(T, K * V * P)
    return PolyStepState{T, typeof(solver)}(
        Matrix{T}(X0), fill(T(1) / P, P), 0,
        nothing, nothing, nothing,
        ps.use_momentum ? zeros(T, d, P) : nothing,
        1.0, 0, Inf, 1.0, nothing, nothing, nothing,
        fill(T(NaN), d), Inf, 0,
        Float64[], Bool[], Float64[], Float64[], 0.0,
        # orthoplex reads R columns directly, no Vt/dirs
        ps.polytope === :orthoplex ? nothing : polytope_vertices(T, ps.polytope, d),
        probe_scales(T, K),
        zeros(T, d, d, P),
        ps.polytope === :orthoplex ? nothing : zeros(T, d, V, P),
        zeros(T, d, K * V * P), losses,
        # losses3 aliases losses (reshape shares memory)
        quad ? reshape(losses, K, V, P) : nothing,
        zeros(T, V, P), zeros(T, V, P), zeros(T, V, P),
        zeros(T, d, P), zeros(T, d, P), zeros(T, d, P), zeros(T, d, P),
        quad ? zeros(T, d, P) : nothing,                    # G
        quad ? zeros(T, d, P) : nothing,                    # H
        ps.newton_refinement ? zeros(T, d, P) : nothing,    # N
        ps.newton_refinement ? zeros(T, d, P) : nothing,    # Xref
        ps.newton_refinement ? zeros(T, d, P) : nothing,    # refrot
        ps.epsilon isa ProgressiveEpsilon ? deepcopy(ps.epsilon) : nothing,
        ps.ent_epsilon isa ProgressiveEpsilon ? deepcopy(ps.ent_epsilon) : nothing,
        (quad && ps.lb !== nothing) ? fill(false, P) : nothing,
        solver, false
    )
end

# orthoplex fast path: vertex (v=i, v=d+i) = +/-R[:, i, p], no matmul or (d,V,P) buffer
function _probe_points_orthoplex!(Xp::Matrix{T}, X::Matrix{T}, R::Array{T, 3},
        pr::T, scales::Vector{T}) where {T}
    P = size(X, 2)
    if P >= _KERNEL_BATCH_MIN
        @batch for p in 1:P
            _probe_col_orthoplex!(Xp, X, R, pr, scales, p)
        end
    else
        for p in 1:P
            _probe_col_orthoplex!(Xp, X, R, pr, scales, p)
        end
    end
    return Xp
end

# abstract types: inside @batch, Polyester passes the arrays as PtrArrays
@inline function _probe_col_orthoplex!(Xp::AbstractMatrix{T}, X::AbstractMatrix{T},
        R::AbstractArray{T, 3}, pr::T, scales::AbstractVector{T}, p::Int) where {T}
    d = size(X, 1)
    K = length(scales)
    base = (p - 1) * 2d * K
    for i in 1:d
        c_plus = base + (i - 1) * K
        c_minus = base + (d + i - 1) * K
        for k in 1:K
            s = pr * scales[k]
            @inbounds @simd ivdep for a in 1:d
                Xp[a, c_plus + k] = X[a, p] + s * R[a, i, p]
                Xp[a, c_minus + k] = X[a, p] - s * R[a, i, p]
            end
        end
    end
    return nothing
end

function _probe_points_general!(Xp::Matrix{T}, X::Matrix{T}, dirs::Array{T, 3},
        pr::T, scales::Vector{T}) where {T}
    P = size(dirs, 3)
    if P >= _KERNEL_BATCH_MIN
        @batch for p in 1:P
            _probe_col_general!(Xp, X, dirs, pr, scales, p)
        end
    else
        for p in 1:P
            _probe_col_general!(Xp, X, dirs, pr, scales, p)
        end
    end
    return Xp
end

@inline function _probe_col_general!(Xp::AbstractMatrix{T}, X::AbstractMatrix{T},
        dirs::AbstractArray{T, 3}, pr::T, scales::AbstractVector{T}, p::Int) where {T}
    d, V, _ = size(dirs)
    K = length(scales)
    base = (p - 1) * V * K
    for v in 1:V, k in 1:K
        c = base + (v - 1) * K + k
        s = pr * scales[k]
        @inbounds @simd ivdep for a in 1:d
            Xp[a, c] = X[a, p] + s * dirs[a, v, p]
        end
    end
    return nothing
end

# flags particles with a clamped probe; their FD model is zeroed downstream
function _clamp_probes!(Xp::Matrix{T}, lb, ub, flag::Vector{Bool}, cols_per_p::Int) where {T}
    fill!(flag, false)
    d = size(Xp, 1)
    @inbounds for c in 1:size(Xp, 2)
        p = div(c - 1, cols_per_p) + 1
        moved = false
        for a in 1:d
            x = Xp[a, c]
            y = clamp(x, _bound(lb, a), _bound(ub, a))
            moved |= y != x
            Xp[a, c] = y
        end
        moved && (flag[p] = true)
    end
    return Xp
end
@inline _bound(b::Real, _) = b
@inline _bound(b::AbstractVector, a) = @inbounds b[a]

# +-Inf is fine (half-bounded boxes, bounds only feed clamp); shape checked by caller
function _check_bounds(lb, ub)
    all(<(Inf), lb) || throw(ArgumentError("lb must be < Inf and not NaN, got $lb"))
    all(>(-Inf), ub) || throw(ArgumentError("ub must be > -Inf and not NaN, got $ub"))
    all(lb .<= ub) ||
        throw(ArgumentError("require lb <= ub elementwise, got lb=$lb, ub=$ub"))
    return nothing
end

# mean over the contiguous K-run per (v, p)
function _cost_from_losses!(C::Matrix{T}, losses::Vector{T}, K::Int) where {T}
    V, P = size(C)
    if P >= _KERNEL_BATCH_MIN
        @batch for p in 1:P
            _cost_col!(C, losses, K, V, p)
        end
    else
        for p in 1:P
            _cost_col!(C, losses, K, V, p)
        end
    end
    return C
end

@inline function _cost_col!(
        C::AbstractMatrix{T}, losses::AbstractVector{T}, K::Int, V::Int, p::Int) where {T}
    @inbounds for v in 1:V
        base = ((p - 1) * V + (v - 1)) * K
        acc = zero(T)
        @simd for k in 1:K
            acc += losses[base + k]
        end
        C[v, p] = acc / K
    end
    return nothing
end

function _resolve_radii(ps::PolyStepConfig, st::PolyStepState, rng::AbstractRNG)
    t = st.iteration
    eps = epsilon_at(st.prog_eps === nothing ? ps.epsilon : st.prog_eps, t)
    rmult = ps.use_adaptive_radius ? st.radius_multiplier : 1.0
    sr = epsilon_at(ps.step_radius, t) * (is_scheduled(ps.step_radius) ? 1.0 : eps) * rmult
    ps.trust_region && (sr *= st.tr_multiplier)
    pr = epsilon_at(ps.probe_radius, t) * (is_scheduled(ps.probe_radius) ? 1.0 : eps) * rmult
    if ps.probe_radius_jitter > 0        # jitter == 0 consumes no rng
        pr *= 1 + ps.probe_radius_jitter * (2 * rand(rng) - 1)
    end
    return eps, sr, pr
end

# not specialized on f; the objective call goes through the _eval_losses! barrier
"""
    step!(f, ps::PolyStepConfig, st::PolyStepState; rng=Random.default_rng()) -> Float64

One PolyStep iteration; returns the mean sanitized cost (the step's loss
proxy). `f` is the batched objective `(d, N) -> (N,)` and must return one
cost per column.
"""
Base.@nospecializeinfer function step!(@nospecialize(f), ps::PolyStepConfig,
        st::PolyStepState{T}; rng::AbstractRNG = Random.default_rng()) where {T}
    d, P = size(st.X)
    V = size(st.Wn, 1)
    K = ps.num_probe
    eps, sr_f, pr_f = _resolve_radii(ps, st, rng)
    sr = T(sr_f)
    pr = T(pr_f)

    # serial randn (thread-count independent); R doubles as the Gaussian scratch
    haar_rotations!(st.R, st.R, rng)
    # a non-finite descent gives a NaN bias; biased_rotation! then keeps the Haar draw
    if ps.biased_rotation && st.prev_descent !== nothing
        @inbounds for p in 1:P
            nrm = zero(T)
            @simd for a in 1:d
                nrm += st.prev_descent[a, p]^2
            end
            inv = one(T) / max(sqrt(nrm), T(1e-10))
            @simd ivdep for a in 1:d
                st.bias[a, p] = st.prev_descent[a, p] * inv
            end
        end
        biased_rotation!(st.R, st.bias)
    end

    # probe points (at probe radius; the step uses step radius)
    if ps.polytope === :orthoplex
        _probe_points_orthoplex!(st.Xprobe, st.X, st.R, pr, st.scales)
    else
        _batched_mul!(st.dirs, st.R, st.Vt)
        _probe_points_general!(st.Xprobe, st.X, st.dirs, pr, st.scales)
    end

    # bounds / repair apply to evaluated candidates only
    if ps.lb !== nothing
        if st.clampflag === nothing
            st.Xprobe .= clamp.(st.Xprobe, ps.lb, ps.ub)
        else
            _clamp_probes!(st.Xprobe, ps.lb, ps.ub, st.clampflag, V * K)
        end
    end
    ps.repair === nothing || ps.repair(st.Xprobe)

    # objective (dominant cost)
    ncols = K * V * P
    _eval_cols!(f, st.losses, st.Xprobe, ps.eval_chunk)
    st.evals += ncols

    # incumbent; NaN-safe scan (findmin returns NaN if any loss is NaN)
    bf, bidx = _best_finite(st.losses)
    if bidx != 0 && bf < st.best_f
        st.best_f = Float64(bf)
        st.best_x .= view(st.Xprobe, :, bidx)
    end
    st.last_all_nonfinite = bidx == 0     # divergence signal (raw, pre-sanitize)

    # Craw: sanitized, unscaled; Csolve: scaled solver input (keep scale_cost out of Craw)
    _cost_from_losses!(st.Craw, st.losses, K)
    sanitize_cost!(st.Craw)
    scale_cost!(st.Csolve, st.Craw, ps.scale_cost)

    # trust region: last step's predicted vs realized change in mean probe cost
    tr_proxy = ps.trust_region ? Float64(mean(st.Craw)) : 0.0
    if ps.trust_region && st.prev_predicted !== nothing && st.prev_pre_step_loss !== nothing
        st.tr_multiplier = update_trust_region(st.prev_predicted,
            tr_proxy - st.prev_pre_step_loss,
            st.tr_multiplier)
        push!(st.trust_multipliers, st.tr_multiplier)
        st.prev_predicted = nothing
        st.prev_pre_step_loss = nothing
    end

    # OT solve on the sanitized, scaled Csolve (warm-start handling in sinkhorn.jl)
    ot_eps = ps.ent_epsilon === nothing ? eps :
             epsilon_at(st.prog_ent === nothing ? ps.ent_epsilon : st.prog_ent, st.iteration)
    local newf, newg
    local ot_conv::Bool
    if st.solver isa SoftmaxSolver
        # softmax weights are already the mass-normalized plan; skip solve's copy/sanitize
        _check_eps(ot_eps)
        softmax_cols!(st.Wn, st.Csolve, ot_eps)
        newf = nothing
        newg = nothing
        ot_conv = all(isfinite, st.Wn)
    else
        res = solve(st.solver, st.Csolve, ot_eps;
            a = st.a, f0 = st.f, g0 = st.g, last_eps = st.last_eps)
        # fixed-iteration Sinkhorn always hits max_iterations: no signal, no feedback
        sched = ps.ent_epsilon === nothing ? st.prog_eps : st.prog_ent
        if sched isa ProgressiveEpsilon &&
           !(st.solver isa SinkhornSolver && st.solver.threshold <= 0)
            update!(sched; n_iters = res.iters, max_iterations = st.solver.max_iterations,
                converged = res.converged)
        end
        # normalize by realized column mass (same as barycentric projection)
        normalize_particle_masses!(st.Wn, res.plan)
        newf = res.f
        newg = res.g
        ot_conv = res.converged
    end
    st.last_eps = ot_eps
    copyto!(st.Xprev, st.X)
    if ps.polytope === :orthoplex
        # Vt = [I -I]: the centroid is the +/- vertex weight difference
        @inbounds for p in 1:P
            @simd for i in 1:d
                st.cent[i, p] = st.Wn[i, p] - st.Wn[d + i, p]
            end
        end
    else
        mul!(st.cent, st.Vt, st.Wn)
    end
    _batched_matvec!(st.delta, st.R, st.cent)
    if ps.use_momentum
        beta = T(momentum_coefficient(st.iteration, ps.max_iterations,
            ps.momentum_init, ps.momentum_final))
        @. st.velocity = beta * st.velocity + sr * st.delta
        @. st.X += T(ps.velocity_lr) * st.velocity
    else
        @. st.X += sr * st.delta
    end

    # quadratic-model features (FD on raw losses3; orthoplex+K>=2 enforced at init)
    duals_stale = false
    quad_ready = st.losses3 !== nothing
    fd_ready = false
    if (ps.biased_rotation || ps.trust_region) && quad_ready
        fd_gradient!(st.G, st.losses3, st.scales, pr)
        fd_hessian_diag!(st.H, st.losses3, st.scales, pr)
        fd_ready = true
        if st.clampflag !== nothing
            # clamped probes break the FD offsets: zero that particle's model
            @inbounds for p in 1:P
                if st.clampflag[p]
                    for i in 1:d
                        st.G[i, p] = zero(T)
                        st.H[i, p] = zero(T)
                    end
                end
            end
        end
        if ps.biased_rotation
            st.prev_descent === nothing && (st.prev_descent = zeros(T, d, P))
            _batched_matvec!(st.prev_descent, st.R, st.G)
            st.prev_descent .*= -1        # descent = -R * fd_grad
        end
    elseif ps.biased_rotation
        # OT fallback: descent = sr * delta (scale as in Python)
        st.prev_descent === nothing && (st.prev_descent = zeros(T, d, P))
        @. st.prev_descent = sr * st.delta
    end

    # Newton refinement moves X after OT, so the duals go stale
    if ps.newton_refinement && quad_ready
        newton_refinement!(st.Xref, st.X, st.G, st.H, st.N, st.refrot,
            st.losses3, st.scales, pr, st.R, st.Xprev;
            alpha = ps.newton_alpha, max_step_norm = sr_f * 0.5,
            hessian_reg = 1e-4, mask = st.clampflag, fd_ready = fd_ready)
        if all(isfinite, st.Xref)
            # keep X - Xprev == lr*v (re-deriving v would lose sub-ulp momentum)
            if st.velocity !== nothing && ps.velocity_lr != 0
                @. st.velocity += (st.Xref - st.X) / T(ps.velocity_lr)
            end
            copyto!(st.X, st.Xref)
            duals_stale = true
        end
    end

    ps.lb === nothing || (st.X .= clamp.(st.X, ps.lb, ps.ub))

    # adaptive radius on the sanitized unscaled mean cost
    cost_mean = Float64(mean(st.Craw))
    if ps.use_adaptive_radius
        st.radius_multiplier, st.stagnation_count, st.prev_loss = update_adaptive_radius(
            cost_mean, st.prev_loss, st.stagnation_count,
            st.radius_multiplier;
            stagnation_threshold = ps.stagnation_threshold,
            stagnation_patience = ps.stagnation_patience,
            radius_increase = ps.radius_increase,
            radius_decrease = ps.radius_decrease,
            radius_min = ps.radius_min, radius_max = ps.radius_max)
    end

    # also hold on an all-non-finite batch: the uniform penalty is a no-op only for
    # symmetric unbounded probes, clamped/repaired ones would drift the particle
    nan_reverted = !all(isfinite, st.X) || st.last_all_nonfinite
    if nan_reverted
        copyto!(st.X, st.Xprev)
        st.velocity === nothing || fill!(st.velocity, zero(T))
        st.prev_descent = nothing
    end
    if nan_reverted || duals_stale
        st.f = nothing
        st.g = nothing
    else
        st.f = newf
        st.g = newg
    end

    # predict the realized move (after momentum, Newton, clamp, revert) for the next ratio
    if ps.trust_region && fd_ready
        st.prev_predicted = predicted_improvement_mean(st.G, st.H, st.R, st.X, st.Xprev)
        st.prev_pre_step_loss = tr_proxy
    end

    push!(st.costs, cost_mean)
    push!(st.ot_converged, ot_conv)
    disp = 0.0
    @inbounds @simd for i in eachindex(st.X)
        disp += Float64(abs2(st.X[i] - st.Xprev[i]))
    end
    dnorm = disp / P
    push!(st.disp_sqnorms, dnorm)
    st.dmax_peak = max(st.dmax_peak, dnorm)
    st.iteration += 1
    return cost_mean
end

# `chunk`-column views when chunk > 0; f stays unspecialized
Base.@nospecializeinfer function _eval_cols!(@nospecialize(f), dst::AbstractVector,
        X::AbstractMatrix, chunk::Int)
    chunk > 0 || return _eval_losses!(f, dst, X)
    n = size(X, 2)
    for j in 1:chunk:n
        hi = min(j + chunk - 1, n)
        _eval_losses!(f, view(dst, j:hi), view(X, :, j:hi))
    end
    return dst
end

# function barrier; the length check catches a scalar return (missing columnwise)
function _eval_losses!(f::F, dst::AbstractVector, X::AbstractMatrix) where {F}
    out = f(X)
    length(out) == size(X, 2) || throw(DimensionMismatch(
        "objective returned $(length(out)) costs for $(size(X, 2)) columns; wrap scalar objectives with columnwise"))
    dst .= out
    return dst
end

# probes sit off the iterate, so score it once (repair a copy, never st.X)
Base.@nospecializeinfer function _score_iterate!(@nospecialize(f), ps::PolyStepConfig,
        st::PolyStepState)
    Xc = copy(st.X)
    ps.lb === nothing || (Xc .= clamp.(Xc, ps.lb, ps.ub))
    ps.repair === nothing || ps.repair(Xc)
    fx = _eval_cols!(f, similar(st.a), Xc, ps.eval_chunk)
    st.evals += length(fx)
    bf, bidx = _best_finite(fx)
    if bidx != 0 && bf < st.best_f
        st.best_f = bf
        st.best_x .= view(Xc, :, bidx)
    end
    return st
end

function _converged(ps::PolyStepConfig, st::PolyStepState)
    st.iteration < 3 && return false
    d = st.disp_sqnorms
    plateau = abs(d[end] - d[end - 1]) / (abs(d[end - 1]) + 1e-10) < ps.threshold
    # also require a small step: constant-velocity descent plateaus mid-run
    small = d[end] <= ps.threshold * st.dmax_peak
    return plateau && small
end
_diverged(st::PolyStepState) = st.last_all_nonfinite

"""
    solve!(f, ps::PolyStepConfig, st::PolyStepState; rng, callback=nothing) -> PolyStepState

Outer loop: runs up to `ps.max_iterations` steps, checking the
displacement-plateau convergence criterion once `min_iterations` steps have run
(Python's `i + 1 >= min_iterations`) and divergence after every step.
`callback(st) -> true` stops early. At the end the final iterate is scored once
(`P` extra evaluations, clamped and repaired like a probe) and can become
`st.best_x`/`st.best_f`.
"""
function solve!(f, ps::PolyStepConfig, st::PolyStepState;
        rng::AbstractRNG = Random.default_rng(), callback = nothing)
    for i in 1:(ps.max_iterations)
        step!(f, ps, st; rng)
        callback !== nothing && callback(st) === true && break
        # stop on divergence immediately; only the plateau test needs the warm-up
        _diverged(st) && break
        if i >= ps.min_iterations
            _converged(ps, st) && break
        end
    end
    _score_iterate!(f, ps, st)
    return st
end
