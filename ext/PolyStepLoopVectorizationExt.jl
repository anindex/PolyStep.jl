module PolyStepLoopVectorizationExt

using PolyStep
using LoopVectorization: @turbo

const _F = Union{Float32, Float64}

# LV #540: keep each @turbo body to one reduction shape

function PolyStep._lse_cols_turbo!(out::AbstractVector{T}, A::Matrix{T},
        add::AbstractVector{T}) where {T <: _F}
    V, P = size(A)
    @turbo for p in 1:P
        m = typemin(T)
        for v in 1:V
            m = max(m, A[v, p] + add[v])
        end
        out[p] = m
    end
    @turbo for p in 1:P
        s = zero(T)
        for v in 1:V
            x = A[v, p] + add[v]
            s += exp(ifelse(x == out[p], zero(T), x - out[p]))
        end
        out[p] += log(s)
    end
    return out
end

PolyStep._lse_cols_turbo!(out, A, add) = PolyStep._lse_cols_base!(out, A, add)

function PolyStep._lse_rows_turbo!(out::AbstractVector{T}, A::Matrix{T},
        add::AbstractVector{T}, accm::Vector{T},
        accs::Vector{T}) where {T <: _F}
    V, P = size(A)
    fill!(accm, typemin(T))
    fill!(accs, zero(T))
    @turbo for p in 1:P, v in 1:V
        accm[v] = max(accm[v], A[v, p] + add[p])
    end
    @turbo for p in 1:P, v in 1:V
        x = A[v, p] + add[p]
        accs[v] += exp(ifelse(x == accm[v], zero(T), x - accm[v]))
    end
    @turbo for v in 1:V
        out[v] = accm[v] + log(accs[v])
    end
    return out
end

function PolyStep._lse_rows_turbo!(out, A, add, accm, accs)
    PolyStep._lse_rows_base!(out, A, add, accm, accs)
end

function __init__()
    PolyStep._TURBO_ACTIVE[] = true
    return nothing
end

end
