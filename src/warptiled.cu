#include "warptiled.cuh"
#include "warptiled_config.h"

// TODO: Set true after implementing the kernel and its launch below.
bool warptiled_available() noexcept {
    return true;
}

cudaError_t launch_warptiled(const float* a, const float* b, float* c, int m, int n, int k,
                             cudaStream_t stream) {
    using namespace gemm_warp_config;
    return launch_warptiled_config<BM, BN, BK, WM, WN, TM, TN, WNITER>(a, b, c, m, n, k, stream);
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
cudaError_t launch_warptiled_config(const float* a, const float* b, float* c, int m, int n, int k,
                                    cudaStream_t stream) {
    static_assert(BM > 0 && BN > 0 && BK > 0 && WM > 0 && WN > 0);
    static_assert(TM > 0 && TN > 0 && WNITER > 0);
    static_assert(BM % WM == 0 && BN % WN == 0);
    static_assert(WM % TM == 0 && WN % TN == 0);
    // verify for 4-vectorization
    static_assert(BK % 4 == 0 && BN % 4 == 0);
    constexpr int subTileArea = 32 * TM * TN * WNITER;
    static_assert((WM * WN) % subTileArea == 0);
    constexpr int WMITER = (WM * WN) / subTileArea;
    static_assert(WMITER > 0);
    static_assert(WM % WMITER == 0 && WN % WNITER == 0);
    static_assert((WM / WMITER) % TM == 0 && (WN / WNITER) % TN == 0);
    static_assert(32 * (BM / WM) * (BN / WN) <= 1024);
    static_assert(4 * BK * (BM + BN) <= 49152);

    // Each thread owns TM*WMITER rows and TN*WNITER columns across subtiles.
    // The kernel derives warp/lane IDs from the flattened 2D thread index.
    dim3 blockDim(BN / (TN * WNITER), BM / (TM * WMITER));
    dim3 gridDim((n + BN - 1) / BN, (m + BM - 1) / BM);

    sgemm_warptiled<BM, BN, BK, WM, WN, TM, TN, WNITER>
        <<<gridDim, blockDim, 0, stream>>>(a, b, c, m, n, k);

    return cudaGetLastError();
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
__global__ void sgemm_warptiled(const float* a, const float* b, float* c, int m, int n, int k) {

    __shared__ __align__(16) float As[BK][BM]; //transposed As
    __shared__ __align__(16) float Bs[BK][BN];

    // Used in fixed-size local arrays below; this is not a launch parameter.
    constexpr int WMITER = WM * WN / (32 * TM * TN * WNITER);

    int WSUBM = WM / WMITER;
    int WSUBN = WN / WNITER;

    float threadReslts[TM * WMITER][TN * WNITER]{0.0f};

    float atmp[TM * WMITER];
    float btmp[TN * WNITER];

    int block_start_row = blockIdx.y * BM;
    int block_start_col = blockIdx.x * BN;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32 /*warp size*/;

    int warp_local_tid = tid % 32;

    int local_start_row = TM * (warp_local_tid / (WSUBN / TN));
    int local_start_col = TN * (warp_local_tid % (WSUBN / TN));

    int warp_start_row = WM * (warp_id / (BN / WN));
    int warp_start_col = WN * (warp_id % (BN / WN));

    int nThreadsPerBlock = blockDim.x * blockDim.y;

    for (int p = 0; p < k; p += BK) {

        //load shared memory by 4-vectorization

        // load As
        for (int t = tid * 4; t < BM * BK; t += 4 * nThreadsPerBlock) {

            int local_row = t / BK;
            int local_col = t % BK;

            int arow = block_start_row + local_row;
            int acol = local_col + p;

            float4 av = make_float4({0.0f}, {0.0f}, {0.0f}, {0.0f});

            if (arow < m) {
                size_t offset = static_cast<size_t>(arow * k + acol);
                if (acol + 3 < k && offset % 4 == 0) {
                    av = *reinterpret_cast<const float4*>(a + offset);
                } else {
                    av.x = acol < k ? a[offset] : 0.0f;
                    av.y = acol + 1 < k ? a[offset + 1] : 0.0f;
                    av.z = acol + 2 < k ? a[offset + 2] : 0.0f;
                    av.w = acol + 3 < k ? a[offset + 3] : 0.0f;
                }
            }

            //since we transpose As, we write As one-by-one.
            As[local_col][local_row] = av.x;
            As[local_col + 1][local_row] = av.y;
            As[local_col + 2][local_row] = av.z;
            As[local_col + 3][local_row] = av.w;
        }

        // load Bs
        for (int t = tid * 4; t < BN * BK; t += 4 * nThreadsPerBlock) {

            int local_row = t / BN;
            int local_col = t % BN;

            int brow = local_row + p;
            int bcol = block_start_col + local_col;

            float4 bv = make_float4({0.0f}, {0.0f}, {0.0f}, {0.0f});

            if (brow < k) {
                size_t offset = static_cast<size_t>(brow * n + bcol);
                if (bcol + 3 < n && offset % 4 == 0) {
                    bv = *reinterpret_cast<const float4*>(b + offset);
                } else {
                    bv.x = bcol < n ? b[offset] : 0.0f;
                    bv.y = bcol + 1 < n ? b[offset + 1] : 0.0f;
                    bv.z = bcol + 2 < n ? b[offset + 2] : 0.0f;
                    bv.w = bcol + 3 < n ? b[offset + 3] : 0.0f;
                }
            }

            *reinterpret_cast<float4*>(&Bs[local_row][local_col]) = bv;
        }

        __syncthreads();

        // outer product in warp-tilting
        for (int t = 0; t < BK; t++) {

            for (int wm = 0; wm < WMITER; wm++) {
                int arow = local_start_row + WSUBM * wm;
                for (int i = 0; i < TM; i++) {
                    atmp[wm * TM + i] = As[t][warp_start_row + arow + i];
                }
            }
            for (int wn = 0; wn < WNITER; wn++) {
                int bcol = local_start_col + WSUBN * wn;
                for (int i = 0; i < TN; i++) {
                    btmp[wn * TN + i] = Bs[t][warp_start_col + bcol + i];
                }
            }

            for (int i = 0; i < WMITER * TM; i++) {
                for (int j = 0; j < WNITER * TN; j++) {
                    threadReslts[i][j] += atmp[i] * btmp[j];
                }
            }
        }

        __syncthreads();
    }

    for (int wm = 0; wm < WMITER; wm++) {
        for (int wn = 0; wn < WNITER; wn++) {
            for (int i = 0; i < TM; i++) {
                int row = block_start_row + warp_start_row + local_start_row + wm * WSUBM + i;
                for (int j = 0; j < TN; j++) {
                    int col = block_start_col + warp_start_col + local_start_col + wn * WSUBN + j;
                    if (row < m && col < n) {
                        c[row * n + col] = threadReslts[wm * TM + i][wn * TN + j];
                    }
                }
            }
        }
    }
}


WarptiledConfig default_warptiled_config() noexcept {
    using namespace gemm_warp_config;
    return {BM, BN, BK, WM, WN, TM, TN, WNITER};
}

const std::vector<WarptiledVariant>& warptiled_variants() {
    static const std::vector<WarptiledVariant> variants{
#define GEMM_WARPTILED_VARIANT(BM, BN, BK, WM, WN, TM, TN, WNITER) \
    {{BM, BN, BK, WM, WN, TM, TN, WNITER}, \
     &launch_warptiled_config<BM, BN, BK, WM, WN, TM, TN, WNITER>},
#include "warptiled_variants.inc"
#undef GEMM_WARPTILED_VARIANT
    };
    return variants;
}

const WarptiledVariant* find_warptiled_variant(const WarptiledConfig& config) {
    for (const auto& variant : warptiled_variants()) {
        const auto& candidate = variant.config;
        if (candidate.bm == config.bm && candidate.bn == config.bn && candidate.bk == config.bk &&
            candidate.wm == config.wm && candidate.wn == config.wn && candidate.tm == config.tm &&
            candidate.tn == config.tn && candidate.wniter == config.wniter) {
            return &variant;
        }
    }
    return nullptr;
}
