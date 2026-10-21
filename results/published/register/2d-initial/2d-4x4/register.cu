#include "register.cuh"
#define TILED 32
#define TM 4
#define TN 4
bool register_available() noexcept {
    return true;
}

cudaError_t launch_register(const float* a, const float* b, float* c, int m, int n, int k,
    cudaStream_t stream) {

    dim3 blockDim((TILED+TN-1)/TN,(TILED + TM - 1) / TM);
    dim3 gridDim((n + TN*blockDim.x - 1) / (TN*blockDim.x), (m + TM * blockDim.y - 1) / (TM * blockDim.y));

    sgemm_kernel_register_rowTM_colTN<<<gridDim, blockDim, 0, stream>>>(a, b, c, m, n, k);

    return cudaGetLastError();
}

__global__ void sgemm_kernel_register_rowTM_colTN(const float* a, const float* b, float* c, int m,
                                                  int n, int k) {

    __shared__ float As[TILED][TILED];
    __shared__ float Bs[TILED][TILED];

    float threadResults[TM][TN] { 0.0f };

    int row = TM * (threadIdx.y + blockDim.y * blockIdx.y);
    int col = TN * (threadIdx.x + blockDim.x * blockIdx.x);

    int tx = TN * threadIdx.x;
    int ty = TM * threadIdx.y;

    for (int p = 0; p < k; p += TILED) {

        if (tx + p < k && row < m) {
            for (int i = 0;i < TM;i++) {
                for (int j = 0; j < TN;j++){
                    if (row + i < m && col+j<n) {
                        As[ty + i][tx + j] =
                            a[(blockIdx.y * blockDim.y * TM + ty + i) * k + tx + j + p];
                    } else {
                        As[ty + i][tx+j] = 0.0;
                    }
                }
            }
        } else {
            for (int i = 0;i < TM;i++) {
                for (int j = 0; j < TN;j++){
                    As[ty + i][tx+j] = 0.0;
                }
            }
        }

        if (ty + p < k && col < n) {
            for (int i = 0;i < TM;i++) {
                for (int j = 0; j < TN;j++){
                    if (ty + i + p < k&&col+j<n) {
                        Bs[ty + i][tx + j] =
                            b[(ty + i + p) * n + (tx + j + blockIdx.x * blockDim.x * TN)];
                    } else {
                        Bs[ty + i][tx+j] = 0.0;
                    }
                }
            }
        } else {
            for (int i = 0;i < TM;i++) {
                for (int j = 0; j < TN; j++) {
                    Bs[ty + i][tx+j] = 0.0;
                }
            }
        }

        __syncthreads();

        for (int i = 0; i < TILED; i++) {
            float btmp[TN];
            float atmp[TM];
            for (int j = 0; j < TN; j++) {
                btmp[j] = Bs[i][tx + j];
                for (int r = 0; r < TM; r++) {
                    threadResults[r][j] += As[ty + r][i] * btmp[j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0;i < TM;i++) {
        for (int j = 0; j < TN;j++){
            if (row + i < m && col+j < n) {
                c[(row + i) * n + col+j] = threadResults[i][j];
            }
        }
    }
}

