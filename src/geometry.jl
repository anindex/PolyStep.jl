"""
    orthoplex_vertices([T,] dim; radius=1) -> (dim, 2dim)

Cross-polytope template. **Vertex order** (FD extractors index by it):
columns `1..d` are `+e_i`, columns `d+1..2d` are `-e_i`.
"""
function orthoplex_vertices(::Type{T}, dim::Integer; radius::Real = 1) where {T <: AbstractFloat}
    dim >= 1 || throw(ArgumentError("dim must be >= 1, got $dim"))
    V = zeros(T, dim, 2dim)
    r = T(radius)
    @inbounds for i in 1:dim
        V[i, i] = r
        V[i, dim + i] = -r
    end
    return V
end

"""
    simplex_vertices([T,] dim; radius=1) -> (dim, dim+1)

Regular simplex centered at the origin, unit-norm vertices (as in the Python
`get_simplex_vertices`).
"""
function simplex_vertices(::Type{T}, dim::Integer; radius::Real = 1) where {T <: AbstractFloat}
    dim >= 1 || throw(ArgumentError("dim must be >= 1, got $dim"))
    V = zeros(T, dim, dim + 1)
    shift = T((sqrt(dim + 1) + 1) / sqrt(dim^3))
    diagval = T(sqrt(1 + 1 / dim))
    @inbounds for v in 1:dim, a in 1:dim
        V[a, v] = (a == v ? diagval : zero(T)) - shift
    end
    lastval = T(1 / sqrt(dim))
    @inbounds for a in 1:dim
        V[a, dim + 1] = lastval
    end
    V .*= T(radius)
    return V
end

"""
    cube_vertices([T,] dim; radius=1) -> (dim, 2^dim)

Hypercube template, `/sqrt(dim)` normalized. Vertex column `v` encodes the
0-based index `i = v-1`; coordinate `a` reads bit `a-1`: bit 0 -> +1, bit 1 -> -1
(exact port of geometry.py:79-111, `signs = 1 - 2*bits`).
"""
function cube_vertices(::Type{T}, dim::Integer; radius::Real = 1) where {T <: AbstractFloat}
    dim >= 1 || throw(ArgumentError("dim must be >= 1, got $dim"))
    n = 2^dim
    V = zeros(T, dim, n)
    scale = T(radius / sqrt(dim))
    @inbounds for v in 1:n
        i = v - 1
        for a in 1:dim
            bit = (i >> (a - 1)) & 1
            V[a, v] = (1 - 2 * bit) * scale
        end
    end
    return V
end

for f in (:orthoplex_vertices, :simplex_vertices, :cube_vertices)
    @eval $f(dim::Integer; kwargs...) = $f(Float64, dim; kwargs...)
end

"""
    polytope_vertices([T,] polytope::Symbol, dim; radius=1)

Dispatch on `:orthoplex | :simplex | :cube`.
"""
function polytope_vertices(
        ::Type{T}, polytope::Symbol, dim::Integer; radius::Real = 1) where {T <:
                                                                            AbstractFloat}
    polytope === :orthoplex && return orthoplex_vertices(T, dim; radius)
    polytope === :simplex && return simplex_vertices(T, dim; radius)
    polytope === :cube && return cube_vertices(T, dim; radius)
    throw(ArgumentError("unknown polytope :$polytope (expected :orthoplex, :simplex, or :cube)"))
end
function polytope_vertices(polytope::Symbol, dim::Integer; kwargs...)
    polytope_vertices(Float64, polytope, dim; kwargs...)
end

"""
    num_vertices(polytope, dim)

Vertex count: `2dim` (`:orthoplex`), `dim + 1` (`:simplex`), `2^dim` (`:cube`).
"""
function num_vertices(polytope::Symbol, dim::Integer)
    polytope === :orthoplex ? 2dim :
    polytope === :simplex ? dim + 1 :
    polytope === :cube ? 2^dim :
    throw(ArgumentError("unknown polytope :$polytope"))
end

"""
    probe_scales(T, K) -> Vector{T}

Fractional probe radii: `range(0, 1, K+2)[2:K+1]` == torch
`linspace(0,1,K+2)[1:K+1]` (0-based). K=1 -> `[0.5]`.
"""
probe_scales(::Type{T}, K::Integer) where {T} = collect(T, range(0, 1; length = K + 2))[2:(K + 1)]

# guards the global BLAS thread count
const _BLAS_PIN_LOCK = ReentrantLock()

const _ORGQR_LAPACK_MIN = 192

@inline _mezzadri_phase(r::T) where {T} = ifelse(r < zero(T), -one(T), one(T))

"""
    haar_rotations!(R, Z, rng) -> R

Fill `R::(d,d,P)` with Haar-uniform SO(d) rotations. `Z::(d,d,P)` is Gaussian
scratch for d = 3..8 and may alias `R` (each slice is read fully before it is
written). All Gaussians come serially from one rng and the per-slice work is
deterministic given them, so results are bit-identical for any thread count.

d=2: analytic rotation from theta ~ U[0,2 * pi) (matches geometry.py:173-176; consumes
one uniform per particle, no QR). d=3..8: StaticArrays QR of a Gaussian matrix.
d>8: Stewart (1980): reflector j of the Householder QR of a Gaussian matrix is
drawn directly from d-j+1 fresh Gaussians (its law under QR), then Q is
accumulated from the reflectors. d>2: Mezzadri sign fix, then a det=+1 flip of
column **1**.
"""
function haar_rotations!(
        R::Array{T, 3}, Z::Array{T, 3}, rng::AbstractRNG) where {T <:
                                                                 AbstractFloat}
    d = size(R, 1)
    P = size(R, 3)
    size(R, 2) == d || throw(DimensionMismatch("R must be (d,d,P)"))
    if d == 2
        @inbounds for p in 1:P
            theta = T(2 * pi) * rand(rng, T)
            c, s = cos(theta), sin(theta)
            R[1, 1, p] = c
            R[1, 2, p] = -s
            R[2, 1, p] = s
            R[2, 2, p] = c
        end
    elseif d <= 8
        randn!(rng, Z)
        _qr_slices_static!(R, Z, Val(d))
    else
        _haar_stewart!(R, rng)
    end
    return R
end

function _qr_slices_static!(R::Array{T, 3}, Z::Array{T, 3}, v::Val{D}) where {T, D}
    # no batch-size gate: threading pays off already at small P
    @batch for p in 1:size(R, 3)
        _haar_slice!(R, Z, p, v)
    end
    return R
end

@inline function _haar_slice!(
        R::AbstractArray{T, 3}, Z::AbstractArray{T, 3}, p::Int, ::Val{D}) where {T, D}
    # _haar_q also returns Rf so tests can check the sign fix
    Q, _ = _haar_q(SMatrix{D, D, T}(@view Z[:, :, p]))
    @inbounds for j in 1:D, i in 1:D
        R[i, j, p] = Q[i, j]
    end
    return nothing
end

@inline function _haar_q(Zp::SMatrix{D, D, T}) where {D, T}
    F = qr(Zp)
    Rf = F.R
    phases = SVector(ntuple(i -> _mezzadri_phase(Rf[i, i]), Val(D)))
    Q = F.Q * Diagonal(phases)
    nrefl = count(k -> !iszero(Rf[k, k]), 1:(D - 1))
    if isodd(nrefl + count(<(zero(T)), phases))
        Q = _flipcol(Q, Val(1))
    end
    return Q, Rf
end

@inline function _flipcol(Q::SMatrix{D, D, T}, ::Val{J}) where {D, T, J}
    return SMatrix{D, D, T}(ntuple(k -> begin
            i = (k - 1) % D + 1
            j = div(k - 1, D) + 1
            j == J ? -Q[i, j] : Q[i, j]
        end, Val(D * D)))
end

function _foreach_slice(f::F, P::Int, big::Bool) where {F}
    if big && P >= 4 && Threads.nthreads() > 1
        Threads.@threads for p in 1:P
            f(p)
        end
    else
        for p in 1:P
            f(p)
        end
    end
    return nothing
end

function _foreach_slice_pinned(f::F, P::Int, big::Bool) where {F}
    lock(_BLAS_PIN_LOCK)
    nb = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        _foreach_slice(f, P, big)
    finally
        BLAS.set_num_threads(nb)
        unlock(_BLAS_PIN_LOCK)
    end
    return nothing
end

_lapack_orgqr(::Type{T}, d::Int) where {T} = T <: LinearAlgebra.BlasReal && d >= _ORGQR_LAPACK_MIN
_orgqr!(A, tau) = _lapack_orgqr(eltype(A), size(A, 1)) ? LAPACK.orgqr!(A, tau) : _org2r!(A, tau)

function _org2r!(A::AbstractMatrix{T}, tau) where {T}
    d = size(A, 1)
    @inbounds for i in d:-1:1
        t = tau[i]
        A[i, i] = one(T)
        c = i + 1
        while c + 3 <= d
            w1 = w2 = w3 = w4 = zero(T)
            @simd for r in i:d
                v = A[r, i]
                w1 = muladd(v, A[r, c], w1)
                w2 = muladd(v, A[r, c + 1], w2)
                w3 = muladd(v, A[r, c + 2], w3)
                w4 = muladd(v, A[r, c + 3], w4)
            end
            w1, w2, w3, w4 = -t * w1, -t * w2, -t * w3, -t * w4
            @simd ivdep for r in i:d
                v = A[r, i]
                A[r, c] = muladd(w1, v, A[r, c])
                A[r, c + 1] = muladd(w2, v, A[r, c + 1])
                A[r, c + 2] = muladd(w3, v, A[r, c + 2])
                A[r, c + 3] = muladd(w4, v, A[r, c + 3])
            end
            c += 4
        end
        for k in c:d
            w = zero(T)
            @simd for r in i:d
                w = muladd(A[r, i], A[r, k], w)
            end
            w *= -t
            @simd ivdep for r in i:d
                A[r, k] = muladd(w, A[r, i], A[r, k])
            end
        end
        @simd for r in (i + 1):d
            A[r, i] *= -t
        end
        A[i, i] = one(T) - t
        for r in 1:(i - 1)
            A[r, i] = zero(T)
        end
    end
    return A
end

function _phase_fixed_q!(A::AbstractMatrix{T}, tau, ph, flip::Int) where {T}
    d = size(A, 1)
    n = 0
    @inbounds for j in 1:d
        ph[j] = _mezzadri_phase(A[j, j])
        n += (ph[j] < zero(T)) + !iszero(tau[j])
    end
    if isodd(n)
        ph[flip] = -ph[flip]
    end
    _orgqr!(A, tau)
    @inbounds for j in 1:d
        s = ph[j]
        @simd ivdep for i in 1:d
            A[i, j] *= s
        end
    end
    return A
end

function _haar_stewart!(R::Array{T, 3}, rng::AbstractRNG) where {T}
    d = size(R, 1)
    P = size(R, 3)
    for p in 1:P, j in 1:d
        randn!(rng, view(R, j:d, j, p))
    end
    tau = Matrix{T}(undef, d, P)
    ph = Matrix{T}(undef, d, P)
    f = p -> _stewart_q!(view(R, :, :, p), view(tau, :, p), view(ph, :, p))
    (_lapack_orgqr(T, d) ? _foreach_slice_pinned : _foreach_slice)(f, P, P * d^3 >= 20_000)
    return R
end

@noinline function _stewart_q!(A::AbstractMatrix{T}, tau, ph) where {T}
    d = size(A, 1)
    @inbounds for j in 1:d
        alpha = A[j, j]
        xx = zero(T)
        @simd for i in (j + 1):d
            xx += A[i, j]^2
        end
        if iszero(xx)
            tau[j] = zero(T)
            continue
        end
        beta = -copysign(sqrt(alpha^2 + xx), alpha)
        tau[j] = (beta - alpha) / beta
        s = inv(alpha - beta)
        @simd for i in (j + 1):d
            A[i, j] *= s
        end
        A[j, j] = beta
    end
    return _phase_fixed_q!(A, tau, ph, 1)
end

"""
    biased_rotation!(R, bias) -> R

Make column 1 of each rotation the direction of `bias[:, p]`: one Householder
reflection maps `R[:, 1]` to `+-bias[:, p]` (sign chosen to avoid cancellation)
and is applied to every column, then a column sign flip restores det=+1. For
Haar `R`, columns 2..d are uniform on the bias complement (the law of the
Gram-Schmidt in geometry.py `apply_biased_rotation`). A slice keeps its rotation
when its bias has squared norm < 0.25 or a non-finite entry, or its rotation is
non-finite. d=1 returns `R` unchanged (SO(1) = {1}).
"""
function biased_rotation!(R::Array{T, 3}, bias::AbstractMatrix{T}) where {T <: AbstractFloat}
    d = size(R, 1)
    P = size(R, 3)
    size(bias) == (d, P) || throw(DimensionMismatch("bias must be (d,P)"))
    d == 1 && return R
    _foreach_slice(p -> _biased_slice!(view(R, :, :, p), view(bias, :, p)), P, P * d^2 >= 200_000)
    return R
end

@noinline function _biased_slice!(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    d = size(A, 1)
    bb = sum(abs2, b)
    (bb < T(0.25) || !all(isfinite, b) || !all(isfinite, A)) && return nothing
    nb = inv(sqrt(bb))
    aa = ab = zero(T)
    @inbounds for i in 1:d
        aa += A[i, 1]^2
        ab += A[i, 1] * b[i]
    end
    ab *= nb
    s = ab > 0 ? nb : -nb
    c = 2 / (aa + 1 + 2 * abs(ab))
    @inbounds for j in 2:d
        w = zero(T)
        @simd for i in 1:d
            w += (A[i, 1] + s * b[i]) * A[i, j]
        end
        w *= c
        @simd ivdep for i in 1:d
            A[i, j] -= w * (A[i, 1] + s * b[i])
        end
    end
    @. A[:, 1] = b * nb
    s < 0 && (view(A, :, d) .*= -one(T))
    return nothing
end
