#include "register.cuh"
#define TILED 64
#define TM 4
#define TN 4
bool register_available() noexcept {
    return true;
}

cudaError_t launch_register(const float* a, const float* b, float* c, int m, int n, int k,
                            cudaStream_t stream) {

    dim3 blockDim((TILED + TN - 1) / TN, (TILED + TM - 1) / TM);
    dim3 gridDim((n + TN * blockDim.x - 1) / (TN * blockDim.x),
                 (m + TM * blockDim.y - 1) / (TM * blockDim.y));

    sgemm_kernel_register_rowTM_colTN<<<gridDim, blockDim, 0, stream>>>(a, b, c, m, n, k);

    return cudaGetLastError();
}

__global__ void sgemm_kernel_register_rowTM_colTN(const float* a, const float* b, float* c, int m,
                                                  int n, int k) {

    __shared__ float As[TILED][TILED];
    __shared__ float Bs[TILED][TILED];

    float threadResults[TM][TN]{0.0f};

    int row = TM * (threadIdx.y + blockDim.y * blockIdx.y);
    int col = TN * (threadIdx.x + blockDim.x * blockIdx.x);

    int tx = TN * threadIdx.x;
    int ty = TM * threadIdx.y;

    int start_row = blockIdx.y * blockDim.y * TM;
    int start_col = blockIdx.x * blockDim.x * TN;

    for (int p = 0; p < k; p += TILED) {

        // if (tx + p < k && row < m) {
        //     for (int i = 0; i < TM; i++) {
        //         for (int j = 0; j < TN; j++) {
        //             if (row + i < m && tx+j+p < k) {
        //                 As[ty + i][tx + j] =
        //                     a[(blockIdx.y * blockDim.y * TM + ty + i) * k + tx + j + p];
        //             } else {
        //                 As[ty + i][tx + j] = 0.0;
        //             }
        //         }
        //     }
        // } else {
        //     for (int i = 0; i < TM; i++) {
        //         for (int j = 0; j < TN; j++) {
        //             As[ty + i][tx + j] = 0.0;
        //         }
        //     }
        // }

        // if (ty + p < k && col < n) {
        //     for (int i = 0; i < TM; i++) {
        //         for (int j = 0; j < TN; j++) {
        //             if (ty + i + p < k && col + j < n) {
        //                 Bs[ty + i][tx + j] =
        //                     b[(ty + i + p) * n + (tx + j + blockIdx.x * blockDim.x * TN)];
        //             } else {
        //                 Bs[ty + i][tx + j] = 0.0;
        //             }
        //         }
        //     }
        // } else {
        //     for (int i = 0; i < TM; i++) {
        //         for (int j = 0; j < TN; j++) {
        //             Bs[ty + i][tx + j] = 0.0;
        //         }
        //     }
        // }

        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int nThreadPerBlock = blockDim.x * blockDim.y;

        for (int t = 4*tid; t < TILED * TILED; t += 4*nThreadPerBlock) {

            int local_row = t / TILED;
            int local_col = t % TILED;

            // bool a_in_range = (start_row + local_row < m) && (p + local_col < k);
            // bool b_in_range = (p + local_row < k) && (start_col + local_col < n);
            // As[local_row][local_col] =
            //     a_in_range ? a[(start_row + local_row) * k + local_col + p] : 0.0f;
            // Bs[local_row][local_col] =
            //     b_in_range ? b[(p + local_row) * n + start_col + local_col] : 0.0f;

            int a_row = start_row + local_row;
            int a_col = p + local_col;

            int b_row = p + local_row;
            int b_col = start_col + local_col;

            float4 av = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 bv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if(a_row<m){
                size_t offset = static_cast<size_t>((a_row)*k + a_col);
                if(a_col+3<k&&offset%4==0){
                    av = *reinterpret_cast<const float4*>(a + offset);
                }else{
                    av.x = a_col < k ? a[offset] : 0.0f;
                    av.y = a_col + 1 < k ? a[offset + 1] : 0.0f;
                    av.z = a_col + 2 < k ? a[offset + 2] : 0.0f;
                    av.w = a_col + 3 < k ? a[offset + 3] : 0.0f;
                }
            }

            if (b_row < k) {
                size_t offset = static_cast<size_t>(b_row) * n + b_col;

                if (b_col + 3 < n && offset % 4 == 0) {
                    bv = *reinterpret_cast<const float4*>(b + offset);
                } else {
                    bv.x = b_col < n ? b[offset] : 0.0f;
                    bv.y = b_col + 1 < n ? b[offset + 1] : 0.0f;
                    bv.z = b_col + 2 < n ? b[offset + 2] : 0.0f;
                    bv.w = b_col + 3 < n ? b[offset + 3] : 0.0f;
                }
            }

            *reinterpret_cast<float4*>(&As[local_row][local_col]) = av;
            *reinterpret_cast<float4*>(&Bs[local_row][local_col]) = bv;
        }

        __syncthreads();

        for (int i = 0; i < TILED; i++) {
            float btmp[TN];
            float atmp[TM];
            for (int r = 0; r < TM; r++) {
                atmp[r] = As[ty + r][i];
            }
            for (int c = 0; c < TN; c++) {
                btmp[c] = Bs[i][tx + c];
            }

            for (int r = 0; r < TM; r++) {
                for (int c = 0; c < TN; c++) {
                    threadResults[r][c] += atmp[r] * btmp[c];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            if (row + i < m && col + j < n) {
                c[(row + i) * n + col + j] = threadResults[i][j];
            }
        }
    }
}
