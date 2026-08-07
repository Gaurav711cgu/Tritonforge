#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp reduction helper using __shfl_xor_sync
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduction helper across 4 warps (128 threads)
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[4]; // 128 / 32 = 4 warps
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// CUDA RMSNorm Kernel
__global__ void rmsnorm_cuda_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int N,
    float eps
) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_x = x + row_idx * N;
    float* row_out = out + row_idx * N;

    // 1. Accumulate sum of squares across elements mapped to this thread
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = row_x[i];
        sum_sq += val * val;
    }

    // 2. Block-wide reduction
    float total_sum_sq = block_reduce_sum(sum_sq);

    // 3. Compute scale factor (RMS)
    __shared__ float rrms;
    if (tid == 0) {
        rrms = rsqrtf(total_sum_sq / static_cast<float>(N) + eps);
    }
    __syncthreads();

    // 4. Normalize and scale output (coalesced writes)
    float scale = rrms;
    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = row_x[i] * scale * weight[i];
    }
}

// C++ Launcher for PyTorch binding
torch::Tensor rmsnorm_cuda_forward(
    torch::Tensor x,
    torch::Tensor weight,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int N = x.size(-1);
    int M = x.numel() / N;

    auto out = torch::empty_like(x);

    dim3 blocks(M);
    dim3 threads(128); // 4 warps

    rmsnorm_cuda_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        eps
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_cuda_forward, "RMSNorm CUDA forward");
}
