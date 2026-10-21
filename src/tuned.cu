#include "tuned.cuh"
#include "tuned_config.h"

// TODO: Set true after implementing your next measured optimization below.
bool tuned_available() noexcept {
    return true;
}

cudaError_t launch_tuned(const float* a, const float* b, float* c, int m, int n, int k,
                         cudaStream_t stream) {

    return launch_tuned_config<gemm_tuned_config::BM, gemm_tuned_config::BN,
                               gemm_tuned_config::BK, gemm_tuned_config::TM,
                               gemm_tuned_config::TN>(a, b, c, m, n, k, stream);
}

template <int BM, int BN, int BK, int TM, int TN>
cudaError_t launch_tuned_config(const float* a, const float* b, float* c, int m, int n, int k,
                                cudaStream_t stream) {
    static_assert(BM > 0 && BN > 0 && BK > 0 && TM > 0 && TN > 0);
    static_assert(BM % TM == 0 && BN % TN == 0);
    static_assert((BM / TM) * (BN / TN) <= 1024);
    static_assert(BK % 4 == 0 && BN % 4 == 0);
    static_assert(4LL * BK * (BM + BN) <= 48 * 1024);

    dim3 block(BN / TN, BM / TM);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);

    sgemm_tuned<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(a, b, c, m, n, k);

    return cudaGetLastError();
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_tuned(const float* a, const float* b, float* c, int m, int n, int k) {

    __shared__ __align__(16) float As[BM][BK];
    __shared__ __align__(16) float Bs[BK][BN];

    float threadResult[TM][TN]{0.0f};

    float atmp[TM];
    float btmp[TN];

    int block_start_row = BM * blockIdx.y;
    int block_start_col = BN * blockIdx.x;

    int ty = TM * threadIdx.y;
    int tx = TN * threadIdx.x;

    int row = block_start_row + ty;
    int col = block_start_col + tx;

    for (int p = 0; p < k; p += BK) {

        // load shared memory by vectorization
        int tid = blockDim.x * threadIdx.y + threadIdx.x;
        int nThreadsPerBlock = blockDim.x * blockDim.y;
        for (int t = 4 * tid; t < BM * BK; t += 4 * nThreadsPerBlock) {

            int local_row = t / BK;
            int local_col = t % BK;

            int arow = block_start_row + local_row;
            int acol = local_col + p;

            float4 av = make_float4({0.0f}, {0.0f}, {0.0f}, {0.0f});

            // load A
            if (arow < m) {
                size_t offset = static_cast<size_t>(arow * k + acol);
                if (acol + 3 < k && offset % 4 == 0 /* address alignement*/) {
                    av = *reinterpret_cast<const float4*>(a + offset);
                } else {
                    av.x = acol < k ? a[offset] : 0.0f;
                    av.y = acol + 1 < k ? a[offset + 1] : 0.0f;
                    av.z = acol + 2 < k ? a[offset + 2] : 0.0f;
                    av.w = acol + 3 < k ? a[offset + 3] : 0.0f;
                }
            }

            *reinterpret_cast<float4*>(&As[local_row][local_col]) = av;
        }

        //load B
        for (int t = 4 * tid; t < BK * BN; t += 4 * nThreadsPerBlock) {

            int local_row = t / BN;
            int local_col = t % BN;

            int brow = local_row + p;
            int bcol = block_start_col + local_col;

            float4 bv = make_float4({0.0f}, {0.0f}, {0.0f}, {0.0f});
            if (brow < k) {
                size_t offset = static_cast<size_t>(brow * n + bcol);
                if (bcol + 3 < n && offset % 4 == 0 /* address alignement*/) {
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

        for (int i = 0; i < BK; i++) {
            for (int r = 0; r < TM; r++) {
                atmp[r] = As[ty + r][i];
            }
            for (int c = 0; c < TN; c++) {
                btmp[c] = Bs[i][tx + c];
            }
            for (int r = 0; r < TM; r++) {
                for (int c = 0; c < TN; c++) {
                    threadResult[r][c] += atmp[r] * btmp[c];
                }
            }
        }

//         for (int i = 0; i < BK; i += 4) {
//             float4 a4[TM];

// // 每一行，沿 K 连续读取 4 个 A 元素。
// #pragma unroll
//             for (int r = 0; r < TM; ++r) {
//                 a4[r] = *reinterpret_cast<const float4*>(&As[ty + r][i]);
//             }

// // 依次完成这 4 个 K 位置上的计算。
// #pragma unroll
//             for (int q = 0; q < 4; ++q) {
// #pragma unroll
//                 for (int cc = 0; cc < TN; ++cc) {
//                     btmp[cc] = Bs[i + q][tx + cc];
//                 }

// #pragma unroll
//                 for (int r = 0; r < TM; ++r) {
//                     float av = (q == 0)   ? a4[r].x
//                                : (q == 1) ? a4[r].y
//                                : (q == 2) ? a4[r].z
//                                           : a4[r].w;

// #pragma unroll
//                     for (int cc = 0; cc < TN; ++cc) {
//                         threadResult[r][cc] += av * btmp[cc];
//                     }
//                 }
//             }
//         }

        __syncthreads();
    }

    for (int r = 0; r < TM; r++) {
        for (int cc = 0; cc < TN; cc++) {
            if (row + r < m && col + cc < n) {
                c[(row + r) * n + col + cc] = threadResult[r][cc];
            }
        }
    }
}
