"""
    LinearEpsilon(; target=1e-3, init=1.0, decay=0.01)

`eps(t) = max(init - decay*t, target)`.
"""
Base.@kwdef struct LinearEpsilon
    target::Float64 = 1e-3
    init::Float64 = 1.0
    decay::Float64 = 0.01
    function LinearEpsilon(target, init, decay)
        init > 0 || throw(ArgumentError("init must be > 0, got $init"))
        target > 0 || throw(ArgumentError("target must be > 0, got $target"))
        decay >= 0 || throw(ArgumentError("decay must be >= 0, got $decay"))
        new(target, init, decay)
    end
end

"""
    CosineEpsilon(; target=1e-3, init=1.0, decay=0.01, total_steps=0, restart_mult=1.0)

`eps(t) = target + 0.5*(init-target)*(1 + cos(pi*t/T))` with
`T = total_steps > 0 ? total_steps : max(1, ceil((init-target)/decay))` (so the
target lands where `LinearEpsilon`'s does), plus optional SGDR warm restarts
(`restart_mult > 1`); each restart period grows by at least one step.
"""
Base.@kwdef struct CosineEpsilon
    target::Float64 = 1e-3
    init::Float64 = 1.0
    decay::Float64 = 0.01
    total_steps::Int = 0
    restart_mult::Float64 = 1.0
    function CosineEpsilon(target, init, decay, total_steps, restart_mult)
        init > 0 || throw(ArgumentError("init must be > 0, got $init"))
        target > 0 || throw(ArgumentError("target must be > 0, got $target"))
        decay >= 0 || throw(ArgumentError("decay must be >= 0, got $decay"))
        total_steps >= 0 || throw(ArgumentError("total_steps must be >= 0, got $total_steps"))
        restart_mult >= 1 ||
            throw(ArgumentError("restart_mult must be >= 1, got $restart_mult (>1 enables SGDR restarts, 1 disables them)"))
        new(target, init, decay, total_steps, restart_mult)
    end
end

"""
    ProgressiveEpsilon(; init=1.0, target=0.01, max_epsilon=5.0, total_steps=0, ...)

Feedback-driven epsilon (ProgOT-inspired); `update!` reacts to solver convergence.
With the default `total_steps=0`, `epsilon_at` ignores the iteration and returns the
EMA-smoothed value. With `total_steps>0`, `epsilon_at(t)` returns a decreasing cosine
baseline from `init` to `target` over `total_steps`, multiplied by the bounded
feedback factor `smoothed/init`, so epsilon sharpens over the run while the feedback
still raises it when the solver struggles and lowers it when it converges fast.
Incompatible with non-iterative solvers (they always report converged in 1
iteration, which would only ever shrink epsilon).
"""
Base.@kwdef mutable struct ProgressiveEpsilon
    init::Float64 = 1.0
    target::Float64 = 0.01
    max_epsilon::Float64 = 5.0
    increase_factor::Float64 = 1.2
    decrease_factor::Float64 = 0.95
    fast_threshold::Float64 = 0.1
    slow_threshold::Float64 = 0.5
    ema_alpha::Float64 = 0.7
    total_steps::Int = 0
    current::Float64 = init
    smoothed::Float64 = init
    function ProgressiveEpsilon(init, target, max_epsilon, increase_factor,
            decrease_factor, fast_threshold, slow_threshold, ema_alpha, total_steps,
            current, smoothed)
        init > 0 || throw(ArgumentError("init must be > 0, got $init"))
        target > 0 || throw(ArgumentError("target must be > 0, got $target"))
        max_epsilon >= target ||
            throw(ArgumentError("max_epsilon must be >= target, got max_epsilon=$max_epsilon, target=$target"))
        0 <= ema_alpha <= 1 || throw(ArgumentError("ema_alpha must be in [0, 1], got $ema_alpha"))
        total_steps >= 0 || throw(ArgumentError("total_steps must be >= 0, got $total_steps"))
        new(init, target, max_epsilon, increase_factor, decrease_factor,
            fast_threshold, slow_threshold, ema_alpha, total_steps, current, smoothed)
    end
end

const EpsilonSchedule = Union{LinearEpsilon, CosineEpsilon, ProgressiveEpsilon}

is_scheduled(::Real) = false
is_scheduled(::EpsilonSchedule) = true

"""
    epsilon_at(schedule, t)

Value of a constant or a schedule at step `t` (0-based); `t = nothing` gives the
initial value.
"""
epsilon_at(x::Real, _) = Float64(x)
epsilon_at(s::LinearEpsilon, ::Nothing) = s.init
epsilon_at(s::LinearEpsilon, t::Integer) = max(s.init - s.decay * t, s.target)
epsilon_at(s::ProgressiveEpsilon, ::Nothing) = s.total_steps > 0 ? _prog_setpoint(s, 0) : s.smoothed
epsilon_at(s::ProgressiveEpsilon, t::Integer) = s.total_steps > 0 ? _prog_setpoint(s, t) : s.smoothed
function _prog_setpoint(s::ProgressiveEpsilon, t::Integer)
    tt = clamp(Int(t), 0, s.total_steps)
    base = s.target + 0.5 * (s.init - s.target) * (1.0 + cos(pi * tt / s.total_steps))
    return clamp(base * (s.smoothed / s.init), s.target, s.max_epsilon)
end
epsilon_at(s::CosineEpsilon, ::Nothing) = s.init
function epsilon_at(s::CosineEpsilon, t::Integer)
    T = s.total_steps > 0 ? s.total_steps :
        ceil(Int, clamp((s.init - s.target) / max(s.decay, 1e-12), 1.0, 2.0^62))
    if s.restart_mult > 1.0
        period = T
        tt = Int(t)
        while tt >= period
            tt -= period
            period = max(period + 1, floor(Int, min(period * s.restart_mult, 2.0^62)))
        end
        T_local = period
        t_local = tt
    else
        T_local = T
        t_local = min(Int(t), T)
    end
    # clamp keeps cos inside [0, pi] for a negative t
    t_local = clamp(t_local, 0, T_local)
    return s.target + 0.5 * (s.init - s.target) * (1.0 + cos(pi * t_local / max(T_local, 1)))
end

"""
    update!(s::ProgressiveEpsilon; n_iters, max_iterations, converged)

Struggling solve (non-converged or `n_iters/max > slow_threshold`) -> increase;
fast solve (`ratio < fast_threshold`) -> decrease; then EMA-smooth and clamp.
"""
function update!(s::ProgressiveEpsilon; n_iters::Integer, max_iterations::Integer, converged::Bool)
    ratio = n_iters / max(max_iterations, 1)
    if !converged || ratio > s.slow_threshold
        s.current = min(s.current * s.increase_factor, s.max_epsilon)
    elseif ratio < s.fast_threshold
        s.current = max(s.current * s.decrease_factor, s.target)
    end
    s.smoothed = s.ema_alpha * s.smoothed + (1.0 - s.ema_alpha) * s.current
    s.smoothed = clamp(s.smoothed, s.target, s.max_epsilon)
    return s
end
