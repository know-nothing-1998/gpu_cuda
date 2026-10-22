#include "doublebuffer.cuh"
#include "doublebuffer_config.h"

#include <sstream>

bool doublebuffer_available() noexcept {
    // Set true after writing sgemm_doublebuffer below, then run --mode check.
    return true;
}

cudaError_t launch_doublebuffer(const float* a, const float* b, float* c, int m, int n, int k,
                                cudaStream_t stream) {
    using namespace gemm_doublebuffer_config;
    return launch_doublebuffer_config<BM, BN, BK, WM, WN, TM, TN, WNITER>(a, b, c, m, n, k, stream);
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
cudaError_t launch_doublebuffer_config(const float* a, const float* b, float* c, int m, int n,
                                       int k, cudaStream_t stream) {
    static_assert(BM > 0 && BN > 0 && BK > 0 && WM > 0 && WN > 0);
    static_assert(TM > 0 && TN > 0 && WNITER > 0);
    static_assert(BM % WM == 0 && BN % WN == 0);
    static_assert(BK % 4 == 0 && BN % 4 == 0);
    constexpr int subTileArea = 32 * TM * TN * WNITER;
    static_assert((WM * WN) % subTileArea == 0);
    constexpr int WMITER = (WM * WN) / subTileArea;
    static_assert(WMITER > 0);
    static_assert(WM % WMITER == 0 && WN % WNITER == 0);
    static_assert((WM / WMITER) % TM == 0 && (WN / WNITER) % TN == 0);
    static_assert(32 * (BM / WM) * (BN / WN) <= 1024);
    static_assert(2LL * 4 * BK * (BM + BN) <= 49152,
                  "Two shared-memory stages must fit the 48 KiB static budget");

    if (!doublebuffer_available())
        return cudaErrorNotSupported;
    constexpr int THREADS_X = BN / (TN * WNITER);
    constexpr int THREADS_Y = BM / (TM * WMITER);
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    sgemm_doublebuffer<BM, BN, BK, WM, WN, TM, TN, WNITER>
        <<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    return cudaGetLastError();
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
__global__ void sgemm_doublebuffer(const float* a, const float* b, float* c, int m, int n, int k) {

    __shared__ __align__(16) float As[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN];

    constexpr int THREADS = 32 * (BM / WM) * (BN / WN);

    constexpr int A_VECTORS = BM * BK / 4;
    constexpr int B_VECTORS = BK * BN / 4;

    constexpr int A_SLOTS = (A_VECTORS + THREADS - 1) / THREADS;
    constexpr int B_SLOTS = (B_VECTORS + THREADS - 1) / THREADS;

    float4 nextA[A_SLOTS];
    float4 nextB[B_SLOTS];

    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);

    float atmp[TM * WMITER];
    float btmp[TN * WNITER];

    float threadResults[TM * WMITER][TN * WNITER]{0.0f};

    int WSUBM = WM / WMITER;
    int WSUBN = WN / WNITER;

    int block_start_row = blockIdx.y * BM;
    int block_start_col = blockIdx.x * BN;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int local_tid_in_warp = tid % 32;
    int thread_start_row = TM * (local_tid_in_warp / (WN / (WNITER * TN)));
    int thread_start_col = TN * (local_tid_in_warp % (WN / (WNITER * TN)));

    int warp_id = tid / 32;
    int warp_start_row = WM * (warp_id / (BN / WN));
    int warp_start_col = WN * (warp_id % (BN / WN));

    auto load4 = [](const float* matrix, int rows, int cols, int row, int col) -> float4 {
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (row >= rows || col >= cols) {
            return value;
        }

        size_t offset = static_cast<size_t>(row) * cols + col;

        if (col + 3 < cols && offset % 4 == 0) {
            return *reinterpret_cast<const float4*>(matrix + offset);
        }

        value.x = matrix[offset];
        if (col + 1 < cols)
            value.y = matrix[offset + 1];
        if (col + 2 < cols)
            value.z = matrix[offset + 2];
        if (col + 3 < cols)
            value.w = matrix[offset + 3];

        return value;
    };

    auto prefetch = [&](int tile_k) {
#pragma unroll
        for (int s = 0; s < A_SLOTS; ++s) {
            int vector_id = tid + s * THREADS;
            nextA[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (vector_id < A_VECTORS) {
                int element = 4 * vector_id;
                int local_row = element / BK;
                int local_col = element % BK;

                nextA[s] = load4(a, m, k, block_start_row + local_row, tile_k + local_col);
            }
        }

#pragma unroll
        for (int s = 0; s < B_SLOTS; ++s) {
            int vector_id = tid + s * THREADS;
            nextB[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (vector_id < B_VECTORS) {
                int element = 4 * vector_id;
                int local_row = element / BN;
                int local_col = element % BN;

                nextB[s] = load4(b, k, n, tile_k + local_row, block_start_col + local_col);
            }
        }
    };

    auto publish = [&](int stage) {
#pragma unroll
        for (int s = 0; s < A_SLOTS; ++s) {
            int vector_id = tid + s * THREADS;

            if (vector_id < A_VECTORS) {
                int element = 4 * vector_id;
                int local_row = element / BK;
                int local_col = element % BK;

                As[stage][local_col + 0][local_row] = nextA[s].x;
                As[stage][local_col + 1][local_row] = nextA[s].y;
                As[stage][local_col + 2][local_row] = nextA[s].z;
                As[stage][local_col + 3][local_row] = nextA[s].w;
            }
        }

#pragma unroll
        for (int s = 0; s < B_SLOTS; ++s) {
            int vector_id = tid + s * THREADS;

            if (vector_id < B_VECTORS) {
                int element = 4 * vector_id;
                int local_row = element / BN;
                int local_col = element % BN;

                *reinterpret_cast<float4*>(&Bs[stage][local_row][local_col]) = nextB[s];
            }
        }
    };

    int read_stage = 0;

    prefetch(0);
    publish(read_stage);
    __syncthreads();

    for (int p = 0; p < k; p += BK) {

        bool has_next = p + BK < k;
        int write_stage = read_stage ^ 1;

        if (has_next) {
            prefetch(p + BK);
        }

        for (int t = 0; t < BK; ++t) {
            for (int wm = 0; wm < WMITER; ++wm) {
                int local_row = warp_start_row + thread_start_row + wm * WSUBM;

                for (int i = 0; i < TM; ++i) {
                    atmp[wm * TM + i] = As[read_stage][t][local_row + i];
                }
            }

            for (int wn = 0; wn < WNITER; ++wn) {
                int local_col = warp_start_col + thread_start_col + wn * WSUBN;

                for (int j = 0; j < TN; ++j) {
                    btmp[wn * TN + j] = Bs[read_stage][t][local_col + j];
                }
            }

            for (int i = 0; i < TM * WMITER; ++i) {
                for (int j = 0; j < TN * WNITER; ++j) {
                    threadResults[i][j] += atmp[i] * btmp[j];
                }
            }
        }

        if (has_next) {
            publish(write_stage);
            __syncthreads();
            read_stage = write_stage;
        }
    }

    for (int wm = 0; wm < WMITER; wm++) {
        for (int wn = 0; wn < WNITER; wn++) {
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    int row = block_start_row + warp_start_row + thread_start_row + wm * WSUBM + i;
                    int col = block_start_col + warp_start_col + thread_start_col + wn * WSUBN + j;
                    if (row < m && col < n) {
                        c[row * n + col] = threadResults[wm * TM + i][wn * TN + j];
                    }
                }
            }
        }
    }
}

std::string doublebuffer_configuration() {
    using namespace gemm_doublebuffer_config;
    std::ostringstream out;
    out << "BM=" << BM << ";BN=" << BN << ";BK=" << BK << ";WM=" << WM << ";WN=" << WN
        << ";TM=" << TM << ";TN=" << TN << ";WNITER=" << WNITER << ";THREADS=" << THREADS
        << ";STAGES=" << STAGES << ";SHARED_BYTES=" << SHARED_BYTES;
    return out.str();
}
