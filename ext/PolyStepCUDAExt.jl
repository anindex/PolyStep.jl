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

end
