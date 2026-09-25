# Quadratic model from orthoplex probe evaluations. Consumes raw per-probe losses
# in (K, V, P) layout (V = 2d, v=i is +e_i, v=d+i is -e_i). Requires K >= 2.

"""
    fd_gradient!(G, losses3, scales, probe_radius) -> G

Central difference averaged over probe scales:
`G[i,p] = mean_k (L[k,i,p] - L[k,d+i,p]) / max(2*scales[k]*r, 1e-10)`.
`G` is in the rotated frame, shape (d, P).
"""
function fd_gradient!(G::AbstractMatrix{T}, losses3::AbstractArray{T, 3},
        scales::AbstractVector, probe_radius::Real) where {T}
    K, V, P = size(losses3)
    d = size(G, 1)
    V == 2d || throw(DimensionMismatch("FD gradient needs orthoplex (V == 2d); got V=$V, d=$d"))
    @inbounds for p in 1:P, i in 1:d
        acc = zero(T)
        for k in 1:K
            denom = max(T(2 * scales[k] * probe_radius), T(1e-10))
            acc += (losses3[k, i, p] - losses3[k, d + i, p]) / denom
        end
        G[i, p] = acc / K
    end
    return G
end

"""
    fd_hessian_diag!(H, losses3, scales, probe_radius) -> H

Regress the symmetric sum `L(+s) + L(-s) = 2a + H*s^2` on centered unit-scale
`scales[k]^2`, then divide by `probe_radius^2`, so the floor on the regression
denominator does not depend on the radius. `H` is in the rotated frame, shape (d, P).
"""
function fd_hessian_diag!(H::AbstractMatrix{T}, losses3::AbstractArray{T, 3},
        scales::AbstractVector, probe_radius::Real) where {T}
    K, V, P = size(losses3)
    d = size(H, 1)
    V == 2d || throw(DimensionMismatch("FD Hessian needs orthoplex (V == 2d); got V=$V, d=$d"))
    # centered s^2 regression, zero-alloc
    s_mean = zero(T)
    @inbounds for k in 1:K
        s_mean += T(scales[k]^2)
    end
    s_mean /= K
    denom = zero(T)
    @inbounds for k in 1:K
        denom += (T(scales[k]^2) - s_mean)^2
    end
    # r-free floor, then convert unit s^2 to (s*r)^2
    denom = max(denom, T(1e-10)) * max(T(probe_radius)^2, floatmin(T))
    @inbounds for p in 1:P, i in 1:d
        sym_mean = zero(T)
        for k in 1:K
            sym_mean += losses3[k, i, p] + losses3[k, d + i, p]
        end
        sym_mean /= K
        num = zero(T)
        for k in 1:K
            s_c = T(scales[k]^2) - s_mean
            num += s_c * ((losses3[k, i, p] + losses3[k, d + i, p]) - sym_mean)
        end
        H[i, p] = num / denom
    end
    return H
end

"""
    newton_step!(N, G, H; max_step_norm=10.0, hessian_reg=1e-4) -> N

Diagonal Newton step `-G ./ max(H, reg)` (curvature at/below `reg` floored,
never an ascent step), per-particle norm clipped to `max_step_norm`.
"""
function newton_step!(N::AbstractMatrix{T}, G::AbstractMatrix{T}, H::AbstractMatrix{T};
        max_step_norm::Real = 10.0, hessian_reg::Real = 1e-4) where {T}
    d, P = size(G)
    reg = T(hessian_reg)
    @inbounds for p in 1:P
        nrm2 = zero(T)
        for i in 1:d
            Hs = H[i, p] > reg ? H[i, p] : reg
            delta = -G[i, p] / Hs
            N[i, p] = delta
            nrm2 += delta * delta
        end
        nrm = max(sqrt(nrm2), T(1e-10))
        sc = min(T(max_step_norm) / nrm, one(T))
        for i in 1:d
            N[i, p] *= sc
        end
    end
    return N
end

"""
    predicted_improvement(G, H, S) -> Vector (P,)

Quadratic-model loss change `g'delta + 0.5*delta'H delta` per particle
(negative = improvement).
"""
function predicted_improvement(G::AbstractMatrix{T}, H::AbstractMatrix{T},
        S::AbstractMatrix{T}) where {T}
    d, P = size(G)
    out = Vector{T}(undef, P)
    @inbounds for p in 1:P
        acc = zero(T)
        @simd for i in 1:d
            acc += G[i, p] * S[i, p] + T(0.5) * H[i, p] * S[i, p]^2
        end
        out[p] = acc
    end
    return out
end

"""
    predicted_improvement_mean(G, H, R, X, X0) -> Float64

Mean over particles of the quadratic-model loss change `g's + 0.5 s'H s` for the
realized move `X - X0`, taken in the rotated frame `s = R[:, :, p]' (X - X0)[:, p]`
where `G`/`H` live (negative = improvement). Allocation-free companion to
`predicted_improvement` for the trust-region ratio.
"""
function predicted_improvement_mean(G::AbstractMatrix{T}, H::AbstractMatrix{T},
        R::AbstractArray{T, 3}, X::AbstractMatrix{T}, X0::AbstractMatrix{T}) where {T}
    d, P = size(G)
    P == 0 && return 0.0
    acc = 0.0
    @inbounds for p in 1:P
        a = zero(T)
        for i in 1:d
            s = zero(T)
            @simd for j in 1:d      # R' rows = R columns
                s += R[j, i, p] * (X[j, p] - X0[j, p])
            end
            a += G[i, p] * s + T(0.5) * H[i, p] * s^2
        end
        acc += Float64(a)
    end
    return acc / P
end

"""
    newton_refinement!(Xout, X_bary, G, H, Nbuf, refrot, losses3, scales, probe_radius,
                       R, X_current; alpha=0.3, max_step_norm=1.0, hessian_reg=1e-4,
                       mask=nothing, fd_ready=false) -> Xout

In-place post-OT Newton correction, writing the refined iterate into the scratch
`Xout`. Newton step anchored at the probe center `X_current` (X_bary already
carries the transport move), blended `(1-alpha)*X_bary + alpha*(X_current +
R*delta_rot)`, then a per-particle descent gate keeps the blend only where the
quadratic model predicts it is no worse than the pure OT step (rotated frame from
X_current). `mask[p] == true` skips particle `p` (keeps X_bary), used for
bounds-clamped probes, whose FD model is invalid. `Nbuf`/`refrot` are (d,P)
scratch. `fd_ready=true` reuses `G`/`H` already filled by the caller; otherwise
they are computed here.
"""
function newton_refinement!(Xout::AbstractMatrix{T}, X_bary::AbstractMatrix{T},
        G::AbstractMatrix{T}, H::AbstractMatrix{T}, Nbuf::AbstractMatrix{T},
        refrot::AbstractMatrix{T}, losses3::AbstractArray{T, 3},
        scales::AbstractVector, probe_radius::Real,
        R::AbstractArray{T, 3}, X_current::AbstractMatrix{T};
        alpha::Real = 0.3, max_step_norm::Real = 1.0,
        hessian_reg::Real = 1e-4, mask = nothing, fd_ready::Bool = false) where {T}
    d, P = size(X_bary)
    if !fd_ready
        fd_gradient!(G, losses3, scales, probe_radius)
        fd_hessian_diag!(H, losses3, scales, probe_radius)
    end
    newton_step!(Nbuf, G, H; max_step_norm, hessian_reg)
    _batched_matvec!(refrot, R, Nbuf)
    a = T(alpha)
    copyto!(Xout, X_bary)
    @inbounds for p in 1:P
        (mask !== nothing && mask[p]) && continue
        # tentative blend, read back by the descent gate below
        for i in 1:d
            Xout[i, p] = (1 - a) * X_bary[i, p] + a * (X_current[i, p] + refrot[i, p])
        end
        pred_ot = zero(T)
        pred_ref = zero(T)
        for i in 1:d
            ot_i = zero(T)
            ref_i = zero(T)
            for j in 1:d      # R' rows = R columns
                ot_i += R[j, i, p] * (X_bary[j, p] - X_current[j, p])
                ref_i += R[j, i, p] * (Xout[j, p] - X_current[j, p])
            end
            pred_ot += G[i, p] * ot_i + T(0.5) * H[i, p] * ot_i^2
            pred_ref += G[i, p] * ref_i + T(0.5) * H[i, p] * ref_i^2
        end
        if !(pred_ref <= pred_ot)      # reject (NaN too): restore X_bary
            for j in 1:d
                Xout[j, p] = X_bary[j, p]
            end
        end
    end
    return Xout
end

"""
    update_trust_region(pred_mean, actual, current_radius; kw...) -> Float64

Predicted-vs-actual ratio update (both negative = improvement): ratio clamped
to [-2, 5]; negative ratio -> large shrink; expand only when the model
predicted an improvement (`pred < 0`) and reality matched.
"""
function update_trust_region(pred_mean::Real, actual::Real, current_radius::Real;
        expand_threshold::Real = 0.75, shrink_threshold::Real = 0.25,
        expand_factor::Real = 1.5, shrink_factor::Real = 0.5,
        min_radius::Real = 0.1, max_radius::Real = 3.0)
    abs(pred_mean) < 1e-10 && return Float64(current_radius)
    ratio = clamp(actual / pred_mean, -2.0, 5.0)
    if ratio < 0
        return max(current_radius * shrink_factor * 0.5, min_radius)
    elseif ratio > expand_threshold && pred_mean < 0
        return min(current_radius * expand_factor, max_radius)
    elseif ratio < shrink_threshold
        return max(current_radius * shrink_factor, min_radius)
    end
    return Float64(current_radius)
end
