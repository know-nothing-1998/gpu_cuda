#pragma once
#include <cuda_runtime.h>

bool register_available() noexcept;

// A[M,K], B[K,N], C[M,N]: contiguous row-major FP32; alpha=1, beta=0.
// Enqueue on stream, overwrite every C element, and return launch status.
// Keep allocation, buffer copies, and host synchronization in the harness.
cudaError_t launch_register(const float* a, const float* b, float* c,
                            int m, int n, int k, cudaStream_t stream);

__global__ void sgemm_kernel_register(const float* a, const float* b, float* c, 
                            int m, int n, int k);
