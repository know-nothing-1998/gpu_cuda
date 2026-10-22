#pragma once

#include <cuda_runtime.h>
#include <string>

bool doublebuffer_available() noexcept;
std::string doublebuffer_configuration();

// Row-major FP32 C[M,N] = A[M,K] * B[K,N], alpha=1, beta=0.
// Enqueue on stream; no allocation or host synchronization in the launch path.
cudaError_t launch_doublebuffer(const float* a, const float* b, float* c,
                               int m, int n, int k, cudaStream_t stream);

// Thread count and block shape are derived from these eight tile parameters.
template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
cudaError_t launch_doublebuffer_config(const float* a, const float* b, float* c,
                                      int m, int n, int k, cudaStream_t stream);

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
__global__ void sgemm_doublebuffer(const float* a, const float* b, float* c,
                                 int m, int n, int k);
