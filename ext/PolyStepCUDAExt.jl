# CUDA extension. v1 scope (see README):
#   1. `cuda_objective`: adapt a GPU batched objective to the CPU-side step loop
#      (one H2D upload of the candidate matrix, one D2H download of the losses
#      per call). The objective dominates PolyStep's cost, so this is
#      where the GPU pays off at OR problem scales.
#   2. CUBLAS strided-batched methods for the per-particle rotation products
#      (stride-0 broadcast of the shared right factor; the high-level
#      batched_mul! rejects 2D x 3D).
# The numeric kernels (softmax, sanitize, mass-normalize) have AbstractMatrix
# broadcast fallbacks that run on CuArray and are covered by the GPU tests.
# SinkhornSolver/KLSoftmaxSolver.solve use scalar-indexed convergence and repair
# loops and stay CPU-side; a GPU objective reaches OT through the softmax path.
# A fully device-resident PolyStepState is deferred until profiles show the
# CPU-side kernels (us at V = 2d <= 16) actually bound a workload.
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

# Y[:,:,p] = A[:,:,p] * B: the shared B enters as a size-1 batch, which the
# CUBLAS wrapper broadcasts with stride 0 (one kernel for all particles)
function PolyStep._batched_mul!(
        Y::CuArray{T, 3}, A::CuArray{T, 3}, B::CuMatrix{T}) where {T <: Union{Float32, Float64}}
    d, V, _ = size(Y)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), A, reshape(B, d, V, 1), zero(T), Y)
    return Y
end

# Y[:,p] = A[:,:,p] * X[:,p]: per-particle matvec as (d,1,P) batched gemm
function PolyStep._batched_matvec!(
        Y::CuMatrix{T}, A::CuArray{T, 3}, X::CuMatrix{T}) where {T <: Union{Float32, Float64}}
    d, P = size(Y)
    Y3 = reshape(Y, d, 1, P)
    X3 = reshape(X, d, 1, P)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), A, X3, zero(T), Y3)
    return Y
end

end
