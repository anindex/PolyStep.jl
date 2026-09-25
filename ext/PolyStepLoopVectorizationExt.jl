module PolyStepLoopVectorizationExt

using PolyStep
using LoopVectorization: @turbo

const _F = Union{Float32, Float64}

# LV #540: keep each @turbo body to one reduction shape

function PolyStep._lse_cols_turbo!(out::AbstractVector{T}, A::Matrix{T},
        add::AbstractVector{T}) where {T <: _F}
    V, P = size(A)
    lo = -floatmax(T)
    @turbo for p in 1:P
        m = lo
        for v in 1:V
            m = max(m, A[v, p] + add[v])
        end
        out[p] = m
    end
    @turbo for p in 1:P
        s = zero(T)
        for v in 1:V
            s += exp(A[v, p] + add[v] - out[p])
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
    fill!(accm, -floatmax(T))
    fill!(accs, zero(T))
    for p in 1:P
        ap = add[p]
        @turbo for v in 1:V
            x = A[v, p] + ap
            mo = accm[v]
            mn = max(x, mo)
            accs[v] = accs[v] * exp(mo - mn) + exp(x - mn)
            accm[v] = mn
        end
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
