# Polytope templates, Haar/biased rotations, probe scales. Layout: templates are
# (d, V) columns = vertices; rotations are (d, d, P), applied as R[:, :, p] * v.

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

Regular simplex centered at the origin (exact port of geometry.py:45-76,
including the scalar shift and post-hoc centroid subtraction).
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
    centroid = mean(V; dims = 2)
    V .= (V .- centroid) .* T(radius)
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

# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

# The d>8 LAPACK paths pin BLAS to one thread (threaded outer loop + threaded
# BLAS oversubscribes); the pin mutates process-global state, so concurrent
# callers serialize on this lock to avoid restoring a stale thread count.
const _BLAS_PIN_LOCK = ReentrantLock()

# Mezzadri phase: sign of the R-diagonal with sign(0) := +1, so an underflowed
# zero diagonal can't null a column (geometry.py:209-213).
@inline _mezzadri_phase(r::T) where {T} = ifelse(r < zero(T), -one(T), one(T))

"""
    haar_rotations!(R, Z, rng) -> R

Fill `R::(d,d,P)` with Haar-uniform SO(d) rotations. `Z::(d,d,P)` is Gaussian
scratch. The Gaussian fill is serial from one rng (bit-identical for any thread
count); the per-slice QR is threaded (deterministic given Z).

d=2: analytic rotation from theta ~ U[0,2 * pi) (matches geometry.py:173-176; consumes
one uniform per particle, no QR). d>2: QR + Mezzadri sign fix + det=+1 flip of
column **1** (the Haar path flips the first column; the biased path flips the
last; do not unify).
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
        return R
    end
    randn!(rng, Z)
    _qr_slices!(R, Z)
    return R
end

function _qr_slices!(R::Array{T, 3}, Z::Array{T, 3}) where {T}
    d = size(R, 1)
    if d <= 8
        _qr_slices_static!(R, Z, Val(d))
    else
        _qr_slices_lapack!(R, Z)
    end
    return R
end

function _qr_slices_static!(R::Array{T, 3}, Z::Array{T, 3}, ::Val{D}) where {T, D}
    P = size(R, 3)
    @batch for p in 1:P
        Zp = SMatrix{D, D, T}(@view Z[:, :, p])
        Q, Rf = _haar_q(Zp)
        @inbounds for j in 1:D, i in 1:D
            R[i, j, p] = Q[i, j]
        end
        # Rf is unused past _haar_q; kept in the signature so the sign fix is
        # testable in isolation.
    end
    return R
end

@inline function _haar_q(Zp::SMatrix{D, D, T}) where {D, T}
    F = qr(Zp)
    Rf = F.R
    phases = SVector(ntuple(i -> _mezzadri_phase(Rf[i, i]), Val(D)))
    Q = F.Q * Diagonal(phases)
    if det(Q) < zero(T)
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

function _qr_slices_lapack!(R::Array{T, 3}, Z::Array{T, 3}) where {T}
    d = size(R, 1)
    P = size(R, 3)
    lock(_BLAS_PIN_LOCK)
    nb = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        # Not :static, the static scheduler throws when nested inside user
        # threading (e.g. @threads over seeds); slices are independent, so the
        # dynamic scheduler keeps results bit-identical for any thread count
        Threads.@threads for p in 1:P
            A = @view Z[:, :, p]
            F = qr!(A)
            # capture the Mezzadri signs from diag(R) (upper triangle of A) before
            # materializing Q. Matrix(F.Q) leaves A intact today, but reading the
            # signs first keeps the phase fix correct if Q is ever formed in place.
            phases = [_mezzadri_phase(A[j, j]) for j in 1:d]
            Qm = Matrix(F.Q)
            @inbounds for j in 1:d
                ph = phases[j]
                @simd ivdep for i in 1:d
                    Qm[i, j] *= ph
                end
            end
            if det(Qm) < zero(T)
                @inbounds @simd ivdep for i in 1:d
                    Qm[i, 1] = -Qm[i, 1]
                end
            end
            copyto!(view(R, :, :, p), Qm)
        end
    finally
        BLAS.set_num_threads(nb)
        unlock(_BLAS_PIN_LOCK)
    end
    return R
    # qr!/Matrix(Q) allocate per slice; use FastLapackInterface if the
    # d>8 path ever shows in profiles (d<=8 is the designed regime).
end

"""
    biased_rotation!(R, bias) -> R

Bias column 1 of each rotation toward the unit direction `bias[:, p]` and
re-orthonormalize (exact port of geometry.py:226-256): plain QR (no Mezzadri
phases here), non-finite Q falls back to the input rotation, then (applied to
the output regardless of fallback) column-1 sign realign against the bias
(`dot < 0` -> flip; 0 -> keep) and a det=+1 fix flipping the **last** column.
`bias` columns must be unit vectors (caller normalizes with a 1e-10 floor).
"""
function biased_rotation!(R::Array{T, 3}, bias::AbstractMatrix{T}) where {T <: AbstractFloat}
    d = size(R, 1)
    P = size(R, 3)
    size(bias) == (d, P) || throw(DimensionMismatch("bias must be (d,P)"))
    if d <= 8
        _biased_rotation_static!(R, bias, Val(d))
    else
        _biased_rotation_lapack!(R, bias)
    end
    return R
end

function _biased_rotation_static!(R::Array{T, 3}, bias::AbstractMatrix{T}, ::Val{D}) where {T, D}
    P = size(R, 3)
    @batch for p in 1:P
        b = SVector{D, T}(@view bias[:, p])
        # a unit bias has ||b||=1; ||b||~0 means the descent was below the caller's
        # normalization floor (e.g. flat objective); keep the fresh Haar
        # rotation instead of biasing toward a degenerate direction
        dot(b, b) < T(0.25) && continue
        Rp = SMatrix{D, D, T}(@view R[:, :, p])
        M = hcat(b, _dropcol1(Rp))
        F = qr(M)
        Q = SMatrix{D, D, T}(F.Q)
        out = all(isfinite, Q) ? Q : Rp
        # QR fixes column 1 only up to sign; realign toward descent (0 -> keep)
        dot0 = dot(out[:, 1], b)
        if dot0 < zero(T)
            out = _flipcol(out, Val(1))
        end
        if det(out) < zero(T)
            out = _flipcol(out, Val(D))
        end
        @inbounds for j in 1:D, i in 1:D
            R[i, j, p] = out[i, j]
        end
    end
    return R
end

@inline function _dropcol1(Rp::SMatrix{D, D, T}) where {D, T}
    return SMatrix{D, D - 1, T}(ntuple(k -> begin
            i = (k - 1) % D + 1
            j = div(k - 1, D) + 2
            Rp[i, j]
        end, Val(D * (D - 1))))
end

function _biased_rotation_lapack!(R::Array{T, 3}, bias::AbstractMatrix{T}) where {T}
    d = size(R, 1)
    P = size(R, 3)
    lock(_BLAS_PIN_LOCK)
    nb = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for p in 1:P
            bnorm2 = zero(T)
            @inbounds @simd for i in 1:d
                bnorm2 += bias[i, p]^2
            end
            # ||b||~0: descent below the caller's floor; keep the fresh Haar rotation
            bnorm2 < T(0.25) && continue
            Rp = @view R[:, :, p]
            M = Matrix{T}(undef, d, d)
            copyto!(M, Rp)
            @inbounds for i in 1:d
                M[i, 1] = bias[i, p]
            end
            F = qr!(M)
            Qm = Matrix(F.Q)
            if !all(isfinite, Qm)
                copyto!(Qm, Rp)
            end
            dot0 = zero(T)
            @inbounds @simd for i in 1:d
                dot0 += Qm[i, 1] * bias[i, p]
            end
            if dot0 < zero(T)
                @inbounds @simd ivdep for i in 1:d
                    Qm[i, 1] = -Qm[i, 1]
                end
            end
            if det(Qm) < zero(T)
                @inbounds @simd ivdep for i in 1:d
                    Qm[i, d] = -Qm[i, d]
                end
            end
            copyto!(Rp, Qm)
        end
    finally
        BLAS.set_num_threads(nb)
        unlock(_BLAS_PIN_LOCK)
    end
    return R
end
