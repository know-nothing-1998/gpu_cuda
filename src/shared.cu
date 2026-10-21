#include "shared.cuh"
#define TILED 32

// TODO: Set true after implementing your shared-memory tiled kernel below.
bool shared_available() noexcept {
    return true;
}

cudaError_t launch_shared(const float* a, const float* b, float* c, int m, int n, int k,
                          cudaStream_t stream) {
    // TODO: Cooperatively load A/B tiles, synchronize, and accumulate across K.
    // Handle partial tiles; synchronize before reusing shared storage.
    // Replace this stub with the launch and return cudaGetLastError().

    dim3 blockDim(TILED, TILED);
    dim3 gridDim((n + blockDim.y - 1) / blockDim.y, (m + blockDim.x - 1) / blockDim.x);

    sgemm_kernel_shared<<<gridDim, blockDim, 0, stream>>>(a, b, c, m, n, k);

    return cudaGetLastError();
}

__global__ void sgemm_kernel_shared(const float* a, const float* b, float* c, int m, int n, int k) {

    __shared__ float As[TILED][TILED];
    __shared__ float Bs[TILED][TILED];

    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float tmp = 0.0f;

    for (int p = 0; p < k; p += TILED) {

        if (tx + p < k&& row<m) {
            As[ty][tx] = a[(blockIdx.y * blockDim.y + ty) * k + tx + p];
        } else {
            As[ty][tx] = 0.0;
        }
        
        if(ty+p<k&& col<n){
            Bs[ty][tx] = b[(ty + p) * n + (tx + blockIdx.x * blockDim.x)];
        }else{
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILED; i++) {
            tmp += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }
    
    if (row < m && col < n){
        c[row * n + col] = tmp;
    }

}
