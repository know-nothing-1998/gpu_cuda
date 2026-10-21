#pragma once
#include <cuda_runtime.h>
#include <vector>

struct WarptiledConfig {
    int bm, bn, bk, wm, wn, tm, tn, wniter;
};

struct WarptiledVariant {
    WarptiledConfig config;
    cudaError_t (*launch)(const float*, const float*, float*, int, int, int, cudaStream_t);
};

// Every entry is a precompiled specialization of the same user-written kernel.
const std::vector<WarptiledVariant>& warptiled_variants();
const WarptiledVariant* find_warptiled_variant(const WarptiledConfig& config);
WarptiledConfig default_warptiled_config() noexcept;

bool warptiled_available() noexcept;

// A[M,K], B[K,N], C[M,N]: contiguous row-major FP32; alpha=1, beta=0.
// Enqueue on stream, overwrite C, and return launch status. No allocations or sync.
cudaError_t launch_warptiled(const float* a, const float* b, float* c,
                           int m, int n, int k, cudaStream_t stream);

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
cudaError_t launch_warptiled_config(const float* a, const float* b, float* c, int m, int n, int k,
                                   cudaStream_t stream);

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
__global__ void sgemm_warptiled(const float* a, const float* b, float* c, int m, int n, int k);
