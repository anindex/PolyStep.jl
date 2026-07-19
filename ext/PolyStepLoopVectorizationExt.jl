# Optional @turbo accelerations. LoopVectorization on Julia 1.12 is fragile
# (issue #540): multi-inner-loop @turbo bodies miscompile here, so this
# extension only provides kernels with the classic
# single-reduction-per-outer-iteration shape, each verified against the
# baseline in the test suite. The softmax stays on the SLEEFPirates baseline,
# already SIMD at ~1 ns/element.
# Loading this extension (having LoopVectorization anywhere in the active
# manifest) switches the whole session to the turbo kernels; set
# PolyStep._TURBO_ACTIVE[] = false to disable it at runtime.
module PolyStepLoopVectorizationExt

using PolyStep
using LoopVectorization: @turbo

const _F = Union{Float32, Float64}

# two single-shape passes; `out` doubles as the running-max buffer (no scratch)
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
            s += exp(A[v, p] + add[v] - out[p])
        end
        out[p] += log(s)
    end
    return out
end

# generic-eltype fallback keeps `lse_cols!` total when the flag is on
PolyStep._lse_cols_turbo!(out, A, add) = PolyStep._lse_cols_base!(out, A, add)

function PolyStep._lse_rows_turbo!(out::AbstractVector{T}, A::Matrix{T},
        add::AbstractVector{T}, accm::Vector{T},
        accs::Vector{T}) where {T <: _F}
    V, P = size(A)
    fill!(accm, typemin(T))
    fill!(accs, zero(T))
    for p in 1:P            # streaming online-LSE: outer loop stays serial
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
