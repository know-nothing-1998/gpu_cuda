#include "asyncpipeline.cuh"
#include "asyncpipeline_config.h"

#include <cuda/pipeline>
#include <sstream>

bool asyncpipeline_available() noexcept {
    return true;
}

cudaError_t launch_asyncpipeline(const float* a, const float* b, float* c, int m, int n, int k,
                                 cudaStream_t stream) {
    using namespace gemm_asyncpipeline_config;
    return launch_asyncpipeline_config<BM, BN, BK, WM, WN, TM, TN, WNITER>(a, b, c, m, n, k,
                                                                           stream);
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
cudaError_t launch_asyncpipeline_config(const float* a, const float* b, float* c, int m, int n,
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

    if (!asyncpipeline_available()) {
        return cudaErrorNotSupported;
    }

    constexpr int THREADS_X = BN / (TN * WNITER);
    constexpr int THREADS_Y = BM / (TM * WMITER);
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    sgemm_asyncpipeline<BM, BN, BK, WM, WN, TM, TN, WNITER>
        <<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    return cudaGetLastError();
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int WNITER>
__global__ void sgemm_asyncpipeline(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ __align__(16) float As[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN];

    constexpr int THREADS = 32 * (BM / WM) * (BN / WN);
    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int A_VECTORS = BM * BK / 4;
    constexpr int A_SLOTS = (A_VECTORS + THREADS - 1) / THREADS;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int warp_start_row = (warp_id / (BN / WN)) * WM;
    const int warp_start_col = (warp_id % (BN / WN)) * WN;
    const int thread_start_row = (lane / (WSUBN / TN)) * TM;
    const int thread_start_col = (lane % (WSUBN / TN)) * TN;
    const int block_start_row = blockIdx.y * BM;
    const int block_start_col = blockIdx.x * BN;

    float atmp[TM * WMITER];
    float btmp[TN * WNITER];
    float threadResults[TM * WMITER][TN * WNITER]{};
    float4 nextA[A_SLOTS];

    // Each thread tracks its own B async-copy batches. Block barriers below make
    // the tiles safe for consumption by other threads and for buffer reuse.
    auto pipe = cuda::make_pipeline();

    // A uses ordinary vectorized global loads into registers. Delay the
    // transposed shared stores until after computing the current tile.
    auto prefetch_a = [&](int tile_k) {
#pragma unroll
        for (int s = 0; s < A_SLOTS; ++s) {
            const int vector_id = tid + s * THREADS;
            float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (vector_id < A_VECTORS) {
                const int element = vector_id * 4;
                const int row = block_start_row + element / BK;
                const int col = tile_k + element % BK;

                if (row < m && col < k) {
                    const size_t offset = static_cast<size_t>(row) * k + col;
                    if (col + 3 < k && offset % 4 == 0) {
                        value = *reinterpret_cast<const float4*>(a + offset);
                    } else {
                        value.x = a[offset];
                        if (col + 1 < k) {
                            value.y = a[offset + 1];
                        }
                        if (col + 2 < k) {
                            value.z = a[offset + 2];
                        }
                        if (col + 3 < k) {
                            value.w = a[offset + 3];
                        }
                    }
                }
            }
            nextA[s] = value;
        }
    };

    auto publish_a = [&](int stage) {
#pragma unroll
        for (int s = 0; s < A_SLOTS; ++s) {
            const int vector_id = tid + s * THREADS;
            if (vector_id < A_VECTORS) {
                const int element = vector_id * 4;
                const int local_row = element / BK;
                const int local_col = element % BK;
                As[stage][local_col + 0][local_row] = nextA[s].x;
                As[stage][local_col + 1][local_row] = nextA[s].y;
                As[stage][local_col + 2][local_row] = nextA[s].z;
                As[stage][local_col + 3][local_row] = nextA[s].w;
            }
        }
    };

    auto enqueue_b = [&](int tile_k, int stage) {
        pipe.producer_acquire();
        // B stays row-major. Copy four floats together when both addresses
        // are aligned; otherwise copy valid scalars and zero the tile tail.
        for (int element = tid * 4; element < BK * BN; element += THREADS * 4) {
            const int local_row = element / BN;
            const int local_col = element % BN;
            const int row = tile_k + local_row;
            const int col = block_start_col + local_col;
            const size_t offset = static_cast<size_t>(row) * n + col;

            if (row < k && col + 3 < n && offset % 4 == 0) {
                cuda::memcpy_async(&Bs[stage][local_row][local_col], b + offset,
                                   cuda::aligned_size_t<16>(sizeof(float4)), pipe);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (row < k && col + j < n) {
                        cuda::memcpy_async(&Bs[stage][local_row][local_col + j], b + offset + j,
                                           cuda::aligned_size_t<4>(sizeof(float)), pipe);
                    } else {
                        Bs[stage][local_row][local_col + j] = 0.0f;
                    }
                }
            }
        }

        // Reconverge after boundary predicates before committing the batch.
        __syncwarp();
        pipe.producer_commit();
    };

    // Prologue: the first tile must arrive before any thread consumes it.
    int read_stage = 0;
    enqueue_b(0, read_stage);
    prefetch_a(0);
    publish_a(read_stage);
    pipe.consumer_wait();
    __syncthreads();

    for (int p = 0; p < k; p += BK) {
        const bool has_next = p + BK < k;
        const int write_stage = read_stage ^ 1;

        // B moves asynchronously into shared; A is prefetched into registers.
        // Both are issued before the independent current-tile computation.
        if (has_next) {
            enqueue_b(p + BK, write_stage);
            prefetch_a(p + BK);
        }

        for (int t = 0; t < BK; ++t) {
            for (int wm = 0; wm < WMITER; ++wm) {
                const int local_row = warp_start_row + thread_start_row + wm * WSUBM;
                for (int i = 0; i < TM; ++i) {
                    atmp[wm * TM + i] = As[read_stage][t][local_row + i];
                }
            }
            for (int wn = 0; wn < WNITER; ++wn) {
                const int local_col = warp_start_col + thread_start_col + wn * WSUBN;
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

        pipe.consumer_release();
        if (has_next) {
            publish_a(write_stage);
            pipe.consumer_wait();
            // All A shared stores and B async copies are complete AND all
            // old-tile readers have finished before reusing the old buffer.
            __syncthreads();
            read_stage = write_stage;
        }
    }

    for (int wm = 0; wm < WMITER; ++wm) {
        for (int wn = 0; wn < WNITER; ++wn) {
            for (int i = 0; i < TM; ++i) {
                const int row =
                    block_start_row + warp_start_row + thread_start_row + wm * WSUBM + i;
                for (int j = 0; j < TN; ++j) {
                    const int col =
                        block_start_col + warp_start_col + thread_start_col + wn * WSUBN + j;
                    if (row < m && col < n) {
                        c[static_cast<size_t>(row) * n + col] =
                            threadResults[wm * TM + i][wn * TN + j];
                    }
                }
            }
        }
    }
}

std::string asyncpipeline_configuration() {
    using namespace gemm_asyncpipeline_config;
    std::ostringstream out;
    out << "BM=" << BM << ";BN=" << BN << ";BK=" << BK << ";WM=" << WM << ";WN=" << WN
        << ";TM=" << TM << ";TN=" << TN << ";WNITER=" << WNITER << ";THREADS=" << THREADS
        << ";STAGES=" << STAGES << ";SHARED_BYTES=" << SHARED_BYTES
        << ";A_FLOAT4_PREFETCH=1;B_ASYNC=1";
    return out.str();
}
