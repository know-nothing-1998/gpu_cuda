#include "naive.cuh"

bool naive_available() noexcept { return true; }

cudaError_t launch_naive(const float* a, const float* b, float* c,
                         int m, int n, int k, cudaStream_t stream) {
    dim3 blockDim(32, 32);
    // This baseline maps x to rows and y to columns.
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    sgemm_kernel_naive<<<gridDim, blockDim,0,stream>>>(a, b, c, m, n, k);

    return cudaGetLastError();
}

__global__ void sgemm_kernel_naive(const float* a, const float* b, float* c,
                             int m, int n, int k) {
    
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    
    float sum=0;
    if(row<m && col<n){
        for(int i=0;i<k;i++){
            sum+=a[row*k+i]*b[col+i*n];
        }
        c[row*n+col]=sum;
    }
}
