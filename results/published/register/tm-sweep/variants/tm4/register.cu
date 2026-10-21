#include "register.cuh"
#define TILED 32
#define TM 4
bool register_available() noexcept {
    return true;
}

cudaError_t launch_register(const float* a, const float* b, float* c, int m, int n, int k,
    cudaStream_t stream) {

    dim3 blockDim(TILED,(TILED + TM - 1) / TM);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + TM * blockDim.y - 1) / (TM * blockDim.y));

    sgemm_kernel_register<<<gridDim, blockDim, 0, stream>>>(a, b, c, m, n, k);

    return cudaGetLastError();
}

__global__ void sgemm_kernel_register(const float* a, const float* b, float* c, int m, int n,
    int k) {

    __shared__ float As[TILED][TILED];
    __shared__ float Bs[TILED][TILED];

    float threadResults[TM]{ 0.0f };

    int row = TM * (threadIdx.y + blockDim.y * blockIdx.y);
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    int tx = threadIdx.x;
    int ty = TM * threadIdx.y;

    for (int p = 0; p < k; p += TILED) {

        if (tx + p < k && row < m) {
            for (int i = 0;i < TM;i++) {
                if (row + i < m) {
                    As[ty + i][tx] = a[(blockIdx.y * blockDim.y*TM + ty + i) * k + tx + p];
                } else {
                    As[ty + i][tx] = 0.0;
                }
            }
        } else {
            for (int i = 0;i < TM;i++) {
                As[ty + i][tx] = 0.0;
            }
        }

        if (ty + p < k && col < n) {
            for (int i = 0;i < TM;i++) {
                if (ty+i+p< k) {
                    Bs[ty + i][tx] = b[(ty + i + p) * n + (tx + blockIdx.x * blockDim.x)];
                } else {
                    Bs[ty + i][tx] = 0.0;
                }
            }
        } else {
            for (int i = 0;i < TM;i++) {
                Bs[ty + i][tx] = 0.0;
            }
        }

        __syncthreads();

        for (int i = 0; i < TILED; i++) {
            float btmp = Bs[i][tx];
            for (int r = 0;r < TM;r++) {
                threadResults[r] += As[ty + r][i] * btmp;
            }
        }

        __syncthreads();
    }

    for (int i = 0;i < TM;i++) {
        if (row + i < m && col < n) {
            c[(row + i) * n + col] = threadResults[i];
        }
    }

}
