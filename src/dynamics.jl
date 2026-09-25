"""
    momentum_coefficient(iteration0, max_iterations, init, final)

Linear warmup from `init` to `final`; `iteration0` is 0-based.
"""
function momentum_coefficient(iteration0::Integer, max_iterations::Integer,
        init::Real = 0.5, final::Real = 0.95)
    progress = min(1.0, iteration0 / max(1, max_iterations - 1))
    return init + progress * (final - init)
end

"""
    update_adaptive_radius(current_loss, prev_loss, stagnation_count, radius_multiplier; kw...)

Stagnation (relative change < threshold) accumulates; at `patience` the radius
multiplier is boosted (explore); improvement decays it (exploit). Non-finite
losses skip adaptation entirely. Returns
`(radius_multiplier, stagnation_count, prev_loss_for_next)`.
"""
function update_adaptive_radius(current_loss::Real, prev_loss::Real,
        stagnation_count::Integer, radius_multiplier::Real;
        stagnation_threshold::Real = 1e-4, stagnation_patience::Integer = 10,
        radius_increase::Real = 1.5, radius_decrease::Real = 0.9,
        radius_min::Real = 0.5, radius_max::Real = 3.0)
    isfinite(current_loss) || return (radius_multiplier, stagnation_count, current_loss)
    isfinite(prev_loss) || return (radius_multiplier, stagnation_count, current_loss)
    rel_change = abs(current_loss - prev_loss) / (abs(prev_loss) + 1e-10)
    if rel_change < stagnation_threshold
        stagnation_count += 1
    else
        stagnation_count = 0
    end
    if stagnation_count >= stagnation_patience
        radius_multiplier = min(radius_multiplier * radius_increase, radius_max)
        stagnation_count = 0
    elseif current_loss < prev_loss
        radius_multiplier = max(radius_multiplier * radius_decrease, radius_min)
    end
    return (radius_multiplier, stagnation_count, current_loss)
end
