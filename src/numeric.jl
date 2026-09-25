@inline exp_fast(x::Float64) = SLEEFPirates.exp(x)
@inline exp_fast(x::Float32) = SLEEFPirates.exp(x)
@inline exp_fast(x) = exp(x)

const _TURBO_ACTIVE = Ref(false)
function _lse_cols_turbo! end
function _lse_rows_turbo! end

"""
    splitmix64(seed, extras...) -> UInt64

Stable seed derivation (SplitMix64). Unlike `Base.hash`, output is documented
stable across Julia versions and processes, safe for reproducible RNG streams,
e.g. `Xoshiro(splitmix64(master_seed, iteration))`.
"""
function splitmix64(z::UInt64)
    z += 0x9e3779b97f4a7c15
    z = xor(z, z >> 30) * 0xbf58476d1ce4e5b9
    z = xor(z, z >> 27) * 0x94d049bb133111eb
    return xor(z, z >> 31)
end
function splitmix64(seed::Integer, extras::Integer...)
    # modular cast, so UInt64 seeds >= 2^63 work
    z = splitmix64(seed % UInt64)
    for e in extras
        z = splitmix64(xor(z, e % UInt64))
    end
    return z
end

function _best_finite(x::AbstractVector)
    bf = Inf
    bidx = 0
    @inbounds for j in eachindex(x)
        v = Float64(x[j])
        if isfinite(v) && v < bf
            bf = v
            bidx = j
        end
    end
    return bf, bidx
end

const _KERNEL_BATCH_MIN = 1024

"""
    softmax_cols!(W, C, eps) -> W

`W[:, p] = softmax(-C[:, p] / eps)` with column-max subtraction before `exp`
(torch.softmax subtracts internally; forgetting this overflows at small eps).
"""
function softmax_cols!(W::Matrix{T}, C::Matrix{T}, eps::Real) where {T <: AbstractFloat}
    V, P = size(C)
    size(W) == (V, P) || throw(DimensionMismatch("W $(size(W)) vs C $(size(C))"))
    if isinf(eps)
        fill!(W, one(T) / V)
        return W
    end
    invabseps = min(one(T) / T(eps), floatmax(T))
    if P >= _KERNEL_BATCH_MIN
        @batch for p in 1:P
            _softmax_col!(W, C, invabseps, V, p)
        end
    else
        for p in 1:P
            _softmax_col!(W, C, invabseps, V, p)
        end
    end
    return W
end

@inline function _softmax_col!(
        W::AbstractMatrix{T}, C::AbstractMatrix{T}, invabseps::T, V::Int, p::Int) where {T}
    @inbounds begin
        cmin = C[1, p]
        @simd for v in 2:V
            cv = C[v, p]
            cmin = ifelse(cv < cmin, cv, cmin)
        end
        s = zero(T)
        @simd for v in 1:V
            e = exp_fast((cmin - C[v, p]) * invabseps)
            W[v, p] = e
            s += e
        end
        invs = one(T) / s
        @simd ivdep for v in 1:V
            W[v, p] *= invs
        end
    end
    return nothing
end

function softmax_cols!(W::AbstractMatrix{T}, C::AbstractMatrix, eps::Real) where {T}
    # Base exp: GPU-broadcast safe (SLEEFPirates is CPU-only)
    if isinf(eps)
        W .= one(T) / size(C, 1)
        return W
    end
    cmin = minimum(C; dims = 1)
    W .= exp.((cmin .- C) .* min(one(T) / T(eps), floatmax(T)))   # see the Array method
    W ./= sum(W; dims = 1)
    return W
end

"""
    lse_cols!(out, A, add) -> out

`out[p] = log(sum_v(exp(A[v, p] + add[v])))`: contiguous column logsumexp,
the Sinkhorn f-update reduction.
"""
function lse_cols!(
        out::AbstractVector{T}, A::Matrix{T}, add::AbstractVector{T}) where {T <:
                                                                             AbstractFloat}
    _TURBO_ACTIVE[] ? _lse_cols_turbo!(out, A, add) : _lse_cols_base!(out, A, add)
end

function _lse_cols_base!(out::AbstractVector{T}, A::Matrix{T},
        add::AbstractVector{T}) where {T <: AbstractFloat}
    V, P = size(A)
    @inbounds for p in 1:P
        m = typemin(T)
        @simd for v in 1:V
            x = A[v, p] + add[v]
            m = ifelse(x > m, x, m)
        end
        s = zero(T)
        @simd for v in 1:V
            x = A[v, p] + add[v]
            s += exp_fast(ifelse(x == m, zero(T), x - m))
        end
        out[p] = m + log(s)
    end
    return out
end

function lse_cols!(out::AbstractVector, A::AbstractMatrix, add::AbstractVector)
    B = A .+ add
    m = maximum(B; dims = 1)
    out .= vec(m .+ log.(sum(exp.(ifelse.(B .== m, zero(eltype(B)), B .- m)); dims = 1)))
    return out
end

"""
    lse_rows!(out, A, add, accm, accs) -> out

`out[v] = log(sum_p(exp(A[v, p] + add[p])))`: the strided direction, computed
with an online-logsumexp accumulator pair (`accm` running max, `accs` running
sum, both length V) so every read of `A` is contiguous. This is why the (V, P)
layout serves both Sinkhorn update directions without a transposed copy.
"""
function lse_rows!(out::AbstractVector{T}, A::Matrix{T}, add::AbstractVector{T},
        accm::Vector{T}, accs::Vector{T}) where {T <: AbstractFloat}
    _TURBO_ACTIVE[] ? _lse_rows_turbo!(out, A, add, accm, accs) :
    _lse_rows_base!(out, A, add, accm, accs)
end

function _lse_rows_base!(out::AbstractVector{T}, A::Matrix{T}, add::AbstractVector{T},
        accm::Vector{T}, accs::Vector{T}) where {T <: AbstractFloat}
    V, P = size(A)
    fill!(accm, typemin(T))
    fill!(accs, zero(T))
    @inbounds for p in 1:P
        ap = add[p]
        @simd for v in 1:V
            x = A[v, p] + ap
            mo = accm[v]
            mn = ifelse(x > mo, x, mo)
            accs[v] = accs[v] * exp_fast(ifelse(mo == mn, zero(T), mo - mn)) +
                      exp_fast(ifelse(x == mn, zero(T), x - mn))
            accm[v] = mn
        end
    end
    @inbounds for v in 1:V
        out[v] = accm[v] + log(accs[v])
    end
    return out
end

function lse_rows!(out::AbstractVector, A::AbstractMatrix, add::AbstractVector, _accm, _accs)
    B = A .+ add'
    m = maximum(B; dims = 2)
    out .= vec(m .+ log.(sum(exp.(ifelse.(B .== m, zero(eltype(B)), B .- m)); dims = 2)))
    return out
end

"""
    sanitize_cost!(C) -> C

Replace non-finite entries with `2*max|finite| + 1` (clamped to `floatmax`): a
finite penalty that ranks last, modeling "never pick this vertex". No absolute
floor, as in the Python reference: a large floor would dominate the `:mean`/`:max`
scale and flatten the weights over the finite vertices.
"""
function sanitize_cost!(C::Array{T}) where {T <: AbstractFloat}
    maxabs = zero(T)
    allfinite = true
    # relies on @simd recognizing & as a reduction over allfinite
    @inbounds @simd for i in eachindex(C)
        c = C[i]
        fin = isfinite(c)
        allfinite &= fin
        a = ifelse(fin, abs(c), zero(T))
        maxabs = max(maxabs, a)   # `a` is never NaN, so max vectorizes
    end
    if !allfinite
        # 2*maxabs+1 can overflow to Inf near floatmax
        penalty = min(2 * maxabs + one(T), floatmax(T))
        @inbounds @simd ivdep for i in eachindex(C)
            c = C[i]
            C[i] = ifelse(isfinite(c), c, penalty)
        end
    end
    return C
end

function sanitize_cost!(C::AbstractArray{T}) where {T}
    allfinite = mapreduce(isfinite, &, C; init = true)
    if !allfinite
        maxabs = mapreduce(c -> ifelse(isfinite(c), abs(c), zero(T)), max, C; init = zero(T))
        penalty = min(2 * maxabs + one(T), floatmax(T))
        C .= ifelse.(isfinite.(C), C, penalty)
    end
    return C
end

"""
    scale_cost!(Cs, C, spec) -> Cs

Write the scaled solver input `Cs` from the raw (sanitized) cost `C`.
`spec`: `nothing` (copy), `:mean` (`(C - m) / mean(C - m)`), `:max`
(`(C - m) / (max(C) - m)`), with `m = minimum(C)` and the divisor floored at
`1e-10`, or a finite positive real divisor (`C / s`, no recentering).
Recentering first makes `:mean`/`:max` shift-invariant, as in the Python
reference: the temperature follows the spread of the costs, not their level.
A negative divisor would reverse the objective (rejected).
"""
scale_cost!(Cs::AbstractMatrix, C::AbstractMatrix, ::Nothing) = copyto!(Cs, C)
function scale_cost!(Cs::AbstractMatrix{T}, C::AbstractMatrix, spec::Symbol) where {T}
    # halves keep c/2 - m/2 finite even when C spans +-floatmax
    m = minimum(C) / 2
    h = if spec === :mean
        v = mean(c -> c / 2 - m, C)
        isfinite(v) ? v : maximum(c -> c / 2 - m, C)
    elseif spec === :max || spec === :max_cost   # :max_cost is the Python name
        maximum(c -> c / 2 - m, C)
    else
        throw(ArgumentError("scale_cost spec must be nothing, :mean, :max, or a positive real; got :$spec"))
    end
    Cs .= (C ./ 2 .- m) ./ max(T(h), T(5e-11), floatmin(T))
    return Cs
end
function scale_cost!(Cs::AbstractMatrix{T}, C::AbstractMatrix, s::Real) where {T}
    (isfinite(s) && s > 0) ||
        throw(ArgumentError("scale_cost divisor must be finite and > 0, got $s (negative would reverse the objective)"))
    Cs .= C ./ T(s)
    return Cs
end

"""
    center_cols!(Cs) -> Cs

Subtract each column's minimum in place. Entropic OT is invariant to a per-column
cost shift (column sums are pinned to the source marginal), so this leaves every
solver's transport plan unchanged while anchoring the log-kernel `-Cs/eps` at 0 for
the cheapest vertex. The log-domain iterations then stay finite even when the raw
cost magnitude would overflow `-Cs/eps` to `-Inf` before the running-max subtraction
can correct it (the softmax kernels already apply the same `cmin - C` shift).
"""
function center_cols!(Cs::Matrix{T}) where {T <: AbstractFloat}
    V, P = size(Cs)
    @inbounds for p in 1:P
        m = Cs[1, p]
        @simd for v in 2:V
            cv = Cs[v, p]
            m = ifelse(cv < m, cv, m)
        end
        @simd ivdep for v in 1:V
            Cs[v, p] -= m
        end
    end
    return Cs
end
center_cols!(Cs::AbstractMatrix) = (Cs .-= minimum(Cs; dims = 1); Cs)

"""
    normalize_particle_masses!(Wn, plan) -> Wn

`Wn[:, p] = plan[:, p] / sum(plan[:, p])`: normalization by the realized column
mass (never by the marginal `a`): equal for softmax columns, and the
translation-invariant choice for unconverged Sinkhorn plans. With this
normalization the vertex-free centroid step `X .+= sr .* R*(V*Wn)` is
algebraically identical to the explicit barycentric projection for every solver
whose plan columns carry mass above the `1e-12` floor. A column at or below that
floor yields `Wn = 0`: an exact per-particle no-op step, not a barycenter (the
core loop holds the particle since its update is incremental).
"""
function normalize_particle_masses!(Wn::Matrix{T}, plan::Matrix{T}) where {T <: AbstractFloat}
    V, P = size(plan)
    if P >= _KERNEL_BATCH_MIN
        @batch for p in 1:P
            _normmass_col!(Wn, plan, V, p)
        end
    else
        for p in 1:P
            _normmass_col!(Wn, plan, V, p)
        end
    end
    return Wn
end

@inline function _normmass_col!(
        Wn::AbstractMatrix{T}, plan::AbstractMatrix{T}, V::Int, p::Int) where {T}
    @inbounds begin
        s = zero(T)
        @simd for v in 1:V
            s += plan[v, p]
        end
        invs = s > T(1e-12) ? one(T) / s : zero(T)
        @simd ivdep for v in 1:V
            Wn[v, p] = plan[v, p] * invs
        end
    end
    return nothing
end

function normalize_particle_masses!(Wn::AbstractMatrix{T}, plan::AbstractMatrix) where {T}
    s = sum(plan; dims = 1)
    Wn .= ifelse.(s .> T(1e-12), plan ./ s, zero(T))
    return Wn
end

"""
    cuda_objective(f_gpu; T=Float32)

GPU-objective adapter, implemented in the CUDA extension (load CUDA.jl).
"""
function cuda_objective end

"""
    _batched_mul!(Y, A, B) -> Y

`Y[:, :, p] = A[:, :, p] * B` for a shared right factor `B`. Generic fallback
is a per-slice 5-arg `mul!`; the CUDA extension overrides with a stride-0
`gemm_strided_batched!`.
"""
function _batched_mul!(Y::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3}, B::AbstractMatrix)
    @views for p in axes(A, 3)
        mul!(Y[:, :, p], A[:, :, p], B)
    end
    return Y
end

"""
    _batched_matvec!(Y, A, X) -> Y

`Y[:, p] = A[:, :, p] * X[:, p]` (per-particle rotation of a vector).
"""
function _batched_matvec!(
        Y::AbstractMatrix{T}, A::AbstractArray{<:Any, 3}, X::AbstractMatrix) where {T}
    d, P = size(Y)
    @inbounds for p in 1:P
        @simd for i in 1:d
            Y[i, p] = zero(T)
        end
        for j in 1:d
            xj = X[j, p]
            @simd ivdep for i in 1:d
                Y[i, p] += A[i, j, p] * xj
            end
        end
    end
    return Y
end
