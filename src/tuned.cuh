#pragma once
#include <cuda_runtime.h>

bool tuned_available() noexcept;

// A[M,K], B[K,N], C[M,N]: contiguous row-major FP32; alpha=1, beta=0.
// Enqueue on stream, overwrite every C element, and return launch status.
// Keep allocation, buffer copies, and host synchronization in the harness.
cudaError_t launch_tuned(const float* a, const float* b, float* c,
                         int m, int n, int k, cudaStream_t stream);

template <int BM, int BN, int BK, int TM, int TN>
cudaError_t launch_tuned_config(const float* a, const float* b, float* c, int m, int n, int k,
                                cudaStream_t stream);

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_tuned(const float* a, const float* b, float* c, int m, int n, int k);
