module PolyStepCUDAExt

using PolyStep
using CUDA

"""
    cuda_objective(f_gpu; T=Float32) -> batched objective

Wrap a GPU objective `f_gpu(X::CuMatrix{T})::CuVector` for use with
`step!`/`minimize`/`PolyStepES`: candidates are uploaded once per call in
precision `T`, evaluated on device, and the losses downloaded.
"""
function PolyStep.cuda_objective(f_gpu; T::Type{<:AbstractFloat} = Float32)
    return function (X::AbstractMatrix)
        Xd = CuMatrix{T}(X)
        return Vector{Float64}(Array(f_gpu(Xd)))
    end
end

function PolyStep._batched_mul!(
        Y::CuArray{T, 3}, A::CuArray{T, 3}, B::CuMatrix{T}) where {T <: Union{Float32, Float64}}
    d, V, _ = size(Y)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), A, reshape(B, d, V, 1), zero(T), Y)
    return Y
end

function PolyStep._batched_matvec!(
        Y::CuMatrix{T}, A::CuArray{T, 3}, X::CuMatrix{T}) where {T <: Union{Float32, Float64}}
    d, P = size(Y)
    Y3 = reshape(Y, d, 1, P)
    X3 = reshape(X, d, 1, P)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), A, X3, zero(T), Y3)
    return Y
end

end
