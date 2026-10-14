#pragma once
#include <cuda_runtime.h>

// Availability of the user-written naive implementation.
bool naive_available() noexcept;

// A[M,K], B[K,N], C[M,N], contiguous row-major; alpha=1, beta=0.
// Buffers are valid, disjoint, device-resident; dimensions are positive.
// Enqueue on the supplied stream, overwrite every C element, do not synchronize.
// Return the CUDA launch status.
cudaError_t launch_naive(const float* a, const float* b, float* c,
                         int m, int n, int k, cudaStream_t stream);

__global__ void sgemm_kernel_naive(const float* a, const float* b, float* c,
                             int m, int n, int k);
