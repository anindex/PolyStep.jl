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

# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

# BLAS pinning is global state; the lock stops callers restoring a stale count
const _BLAS_PIN_LOCK = ReentrantLock()

# Mezzadri phase with sign(0) := +1, so an underflowed zero diagonal can't null a column
@inline _mezzadri_phase(r::T) where {T} = ifelse(r < zero(T), -one(T), one(T))

"""
    haar_rotations!(R, Z, rng) -> R

Fill `R::(d,d,P)` with Haar-uniform SO(d) rotations. `Z::(d,d,P)` is Gaussian
scratch and may alias `R` (each slice is read fully before it is written). The
Gaussian fill is serial from one rng (bit-identical for any thread count); the
per-slice QR may run threaded (deterministic given Z).

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
    # det parity without LU: StaticArrays applies a reflection (det -1) iff Rf[k,k] != 0
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

# f(p) per slice with BLAS pinned to one thread (bits independent of BLAS threads);
# the slice loop threads only from d = 256, below that serial is faster
function _foreach_slice_pinned(f::F, d::Int, P::Int) where {F}
    lock(_BLAS_PIN_LOCK)
    nb = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        if Threads.nthreads() > 1 && P > 1 && d >= 256
            # not :static (throws when nested); independent slices stay bit-identical
            Threads.@threads for p in 1:P
                f(p)
            end
        else
            for p in 1:P
                f(p)
            end
        end
    finally
        BLAS.set_num_threads(nb)
        unlock(_BLAS_PIN_LOCK)
    end
    return nothing
end

# In-place Householder QR in LAPACK layout; tau[j] == 0 means reflector j is the identity.
# Generic methods cover BigFloat/Float16 (Julia's QR stores (factors, tau)).
_geqrf!(A::StridedMatrix{T}, tau) where {T <: LinearAlgebra.BlasReal} = LAPACK.geqrf!(A, tau)
_geqrf!(A, tau) = copyto!(tau, getfield(qr!(A), 2))
_orgqr!(A::StridedMatrix{T}, tau) where {T <: LinearAlgebra.BlasReal} = LAPACK.orgqr!(A, tau)
_orgqr!(A, tau) = copyto!(A, Matrix(LinearAlgebra.QRPackedQ(A, tau)))

# Factored A -> Q * Diagonal(sign(diag(R))), negating column `flip` for det = +1.
# det(Q) = (-1)^count(tau .!= 0): O(d), no LU det underflow at large d.
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

function _qr_slices_lapack!(R::Array{T, 3}, Z::Array{T, 3}) where {T}
    d = size(R, 1)
    P = size(R, 3)
    tau = Matrix{T}(undef, d, P)
    ph = Matrix{T}(undef, d, P)
    _foreach_slice_pinned(d, P) do p
        A = view(R, :, :, p)
        Z === R || copyto!(A, view(Z, :, :, p))
        _geqrf!(A, view(tau, :, p))
        _phase_fixed_q!(A, view(tau, :, p), view(ph, :, p), 1)
    end
    return R
end

"""
    biased_rotation!(R, bias) -> R

Bias column 1 of each rotation toward the unit direction `bias[:, p]` and
re-orthonormalize by QR with Mezzadri phases, which equals the Gram-Schmidt of
geometry.py `apply_biased_rotation`: column 1 is the bias itself and columns
2..d are uniform on its complement. Non-finite input keeps the input
rotation; a det=+1 fix flips the **last** column. d=1 returns `R` unchanged
(SO(1) = {1}). `bias` columns must be unit vectors (caller normalizes with a
1e-10 floor).
"""
function biased_rotation!(R::Array{T, 3}, bias::AbstractMatrix{T}) where {T <: AbstractFloat}
    d = size(R, 1)
    P = size(R, 3)
    size(bias) == (d, P) || throw(DimensionMismatch("bias must be (d,P)"))
    d == 1 && return R
    if d <= 8
        _biased_rotation_static!(R, bias, Val(d))
    else
        _biased_rotation_lapack!(R, bias)
    end
    return R
end

function _biased_rotation_static!(R::Array{T, 3}, bias::AbstractMatrix{T}, v::Val{D}) where {T, D}
    @batch for p in 1:size(R, 3)
        _biased_slice!(R, bias, p, v)
    end
    return R
end

@inline function _biased_slice!(
        R::AbstractArray{T, 3}, bias::AbstractMatrix{T}, p::Int, v::Val{D}) where {T, D}
    b = SVector{D, T}(@view bias[:, p])
    # ||b|| ~ 0: descent below the caller's normalization floor; keep the Haar rotation
    dot(b, b) < T(0.25) && return nothing
    Rp = SMatrix{D, D, T}(@view R[:, :, p])
    F = qr(hcat(b, _dropcol1(Rp)))
    # phases make column 1 = +b and remove the Householder sign bias on columns 2..D
    phases = SVector(ntuple(i -> _mezzadri_phase(F.R[i, i]), Val(D)))
    Q = F.Q * Diagonal(phases)
    all(isfinite, Q) || return nothing
    if det(Q) < zero(T)
        Q = _flipcol(Q, v)
    end
    @inbounds for j in 1:D, i in 1:D
        R[i, j, p] = Q[i, j]
    end
    return nothing
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
    tau = Matrix{T}(undef, d, P)
    ph = Matrix{T}(undef, d, P)
    _foreach_slice_pinned(d, P) do p
        b = view(bias, :, p)
        A = view(R, :, :, p)
        # ||b|| ~ 0 or non-finite input: keep the Haar rotation (screened before
        # the QR overwrites A; finite input gives a finite Q)
        (sum(abs2, b) < T(0.25) || !all(isfinite, b) ||
         !all(isfinite, view(A, :, 2:d))) && return
        copyto!(view(A, :, 1), b)
        _geqrf!(A, view(tau, :, p))
        _phase_fixed_q!(A, view(tau, :, p), view(ph, :, p), d)
    end
    return R
end
