"""
    ParamEntry(key, shape, offset, numel)

One parameter array in a [`ParamLayout`](@ref).
"""
struct ParamEntry
    key::String
    shape::Dims
    offset::Int
    numel::Int
end

"""
    ParamLayout(params; particle_dim = 2)

Flat layout of a model's parameter arrays, given as `name => shape` pairs, shapes, or arrays.
"""
struct ParamLayout
    entries::Vector{ParamEntry}
    total_params::Int
    padded_size::Int
    particle_dim::Int
end

_param_shape(x::AbstractArray) = size(x)
_param_shape(x::Tuple) = Dims(x)
_param_shape(x::Integer) = (Int(x),)

function ParamLayout(pairs::AbstractVector{<:Pair}; particle_dim::Int = 2)
    particle_dim >= 1 || throw(ArgumentError("particle_dim must be >= 1, got $particle_dim"))
    entries = ParamEntry[]
    offset = 0
    for (name, value) in pairs
        shape = _param_shape(value)
        numel = prod(shape; init = 1)
        push!(entries, ParamEntry(String(name), shape, offset, numel))
        offset += numel
    end
    pad = offset == 0 ? 0 : offset + mod(-offset, particle_dim)
    return ParamLayout(entries, offset, pad, particle_dim)
end

function ParamLayout(params::AbstractVector; particle_dim::Int = 2)
    return ParamLayout(["p$(i)" => p for (i, p) in enumerate(params)]; particle_dim = particle_dim)
end

Base.length(l::ParamLayout) = length(l.entries)

"""
    LayerSpec

Coordinate block a [`HybridSubspace`](@ref) assigns to one parameter array.
"""
struct LayerSpec
    key::String
    shape::Dims
    numel::Int
    ncoords::Int
    param_start::Int
    flat_start::Int
    projected::Bool
end

"""
    HybridSubspace(layout; rank, seed = 0, max_subspace_dim = nothing, T = Float64)

Per-layer fixed orthonormal basis: `min(d_out*r + r*d_in, numel)` coordinates per 2-D array, 1-D and
full-width arrays unprojected.
"""
struct HybridSubspace{T <: AbstractFloat}
    specs::Vector{LayerSpec}
    projections::Vector{Matrix{T}}
    subspace_dim::Int
    total_params::Int
    compression_ratio::Float64
    rank::Int
    seed::Int
    max_subspace_dim::Int
end

function _layer_coords(shape::Dims, numel::Int, rank::Int)
    length(shape) < 2 && return numel, false
    d_out = shape[1]
    d_in = prod(shape[2:end])
    er = min(rank, d_in, d_out)
    nc = min(d_out * er + er * d_in, numel)
    return nc, nc < numel
end

function _scale_to_budget(ncoords::Vector{Int}, projected::Vector{Bool}, numels::Vector{Int},
                          max_dim::Int)
    total = sum(ncoords; init = 0)
    (max_dim <= 0 || total <= max_dim) && return ncoords, projected
    unprojected = sum(ncoords[k] for k in eachindex(ncoords) if !projected[k]; init = 0)
    n_projected = count(projected)
    target = max_dim - unprojected
    if target < n_projected
        target = n_projected
        @warn "max_subspace_dim=$max_dim is unreachable: unprojected parameters take $unprojected coordinates and $n_projected projected layers need one each, so the subspace is $(unprojected + n_projected)."
    end
    projected_dim = total - unprojected
    projected_dim <= target && return ncoords, projected
    scale = target / projected_dim
    nc = copy(ncoords)
    pr = copy(projected)
    remaining = target
    left = n_projected
    for k in eachindex(nc)
        pr[k] || continue
        left -= 1
        headroom = min(numels[k], remaining - left)
        nc[k] = min(max(1, round(Int, ncoords[k] * scale)), headroom)
        remaining -= nc[k]
        pr[k] = nc[k] < numels[k]
    end
    return nc, pr
end

function _orthonormal_basis(::Type{T}, numel::Int, ncoords::Int, seed::UInt64) where {T}
    numel >= ncoords ||
        throw(ArgumentError("ncoords ($ncoords) exceeds numel ($numel): no orthonormal basis exists"))
    A = randn(Xoshiro(seed), T, numel, ncoords)
    return Matrix{T}(qr(A).Q * Matrix{T}(I, numel, ncoords))
end

function HybridSubspace(layout::ParamLayout; rank::Int, seed::Int = 0,
                        max_subspace_dim::Union{Nothing, Integer} = nothing,
                        T::Type{<:AbstractFloat} = Float64)
    rank >= 1 || throw(ArgumentError("rank must be >= 1, got $rank"))
    max_dim = max_subspace_dim === nothing ? 0 : Int(max_subspace_dim)
    numels = [e.numel for e in layout.entries]
    ncoords = Vector{Int}(undef, length(numels))
    projected = Vector{Bool}(undef, length(numels))
    for (k, e) in enumerate(layout.entries)
        ncoords[k], projected[k] = _layer_coords(e.shape, e.numel, rank)
    end
    ncoords, projected = _scale_to_budget(ncoords, projected, numels, max_dim)
    specs = Vector{LayerSpec}(undef, length(numels))
    projections = Vector{Matrix{T}}(undef, length(numels))
    flat = 0
    for (k, e) in enumerate(layout.entries)
        specs[k] = LayerSpec(e.key, e.shape, e.numel, ncoords[k], e.offset, flat, projected[k])
        projections[k] = projected[k] ?
                         _orthonormal_basis(T, e.numel, ncoords[k], splitmix64(seed, k)) :
                         Matrix{T}(undef, 0, 0)
        flat += ncoords[k]
    end
    ratio = layout.total_params > 0 ? flat / layout.total_params : 0.0
    return HybridSubspace{T}(specs, projections, flat, layout.total_params, ratio, rank, seed,
                             max_dim)
end

"""
    subspace_dim(s) -> Int

Total number of subspace coordinates.
"""
subspace_dim(s::HybridSubspace) = s.subspace_dim

"""
    compression_ratio(s) -> Float64

`subspace_dim / total_params`.
"""
compression_ratio(s::HybridSubspace) = s.compression_ratio

"""
    expand!(out, s, z) -> out

Write the full-parameter displacement of subspace coordinates `z` into `out`.
"""
function expand!(out::AbstractVector, s::HybridSubspace, z::AbstractVector)
    length(out) == s.total_params ||
        throw(DimensionMismatch("out has length $(length(out)), expected $(s.total_params)"))
    length(z) == s.subspace_dim ||
        throw(DimensionMismatch("z has length $(length(z)), expected $(s.subspace_dim)"))
    @inbounds for k in eachindex(s.specs)
        spec = s.specs[k]
        rows = (spec.param_start + 1):(spec.param_start + spec.numel)
        cols = (spec.flat_start + 1):(spec.flat_start + spec.ncoords)
        if spec.projected
            mul!(view(out, rows), s.projections[k], view(z, cols))
        else
            copyto!(view(out, rows), view(z, cols))
        end
    end
    return out
end

"""
    expand(s, z) -> Vector

Full-parameter displacement of subspace coordinates `z`.
"""
expand(s::HybridSubspace{T}, z::AbstractVector) where {T} =
    expand!(Vector{T}(undef, s.total_params), s, z)

"""
    project!(z, s, x) -> z

Write the subspace coordinates of the full-parameter displacement `x` into `z`.
"""
function project!(z::AbstractVector, s::HybridSubspace, x::AbstractVector)
    length(x) == s.total_params ||
        throw(DimensionMismatch("x has length $(length(x)), expected $(s.total_params)"))
    length(z) == s.subspace_dim ||
        throw(DimensionMismatch("z has length $(length(z)), expected $(s.subspace_dim)"))
    @inbounds for k in eachindex(s.specs)
        spec = s.specs[k]
        rows = (spec.param_start + 1):(spec.param_start + spec.numel)
        cols = (spec.flat_start + 1):(spec.flat_start + spec.ncoords)
        if spec.projected
            mul!(view(z, cols), transpose(s.projections[k]), view(x, rows))
        else
            copyto!(view(z, cols), view(x, rows))
        end
    end
    return z
end

"""
    project(s, x) -> Vector

Subspace coordinates of the full-parameter displacement `x`.
"""
project(s::HybridSubspace{T}, x::AbstractVector) where {T} =
    project!(Vector{T}(undef, s.subspace_dim), s, x)

"""
    reconstruct_batch(s, base, Z) -> Matrix

Columns `base .+ expand(s, Z[:, j])` for every column of `Z`.
"""
function reconstruct_batch(s::HybridSubspace{T}, base::AbstractVector,
                           Z::AbstractMatrix) where {T}
    length(base) == s.total_params ||
        throw(DimensionMismatch("base has length $(length(base)), expected $(s.total_params)"))
    size(Z, 1) == s.subspace_dim ||
        throw(DimensionMismatch("Z has $(size(Z, 1)) rows, expected $(s.subspace_dim)"))
    out = Matrix{T}(undef, s.total_params, size(Z, 2))
    @inbounds for j in axes(Z, 2)
        col = view(out, :, j)
        expand!(col, s, view(Z, :, j))
        col .+= base
    end
    return out
end

"""
    subspace_objective(f, s, base) -> g

Batched objective in subspace coordinates: `g(Z) = f(reconstruct_batch(s, base, Z))`.
"""
function subspace_objective(f, s::HybridSubspace, base::AbstractVector)
    return Z -> f(reconstruct_batch(s, base, Z))
end
