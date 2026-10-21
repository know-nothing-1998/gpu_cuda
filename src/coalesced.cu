#include "coalesced.cuh"

// TODO: Set true after implementing your coalesced-access kernel below.
bool coalesced_available() noexcept { return true; }

cudaError_t launch_coalesced(const float* a, const float* b, float* c,
                             int m, int n, int k, cudaStream_t stream) {
    dim3 blockDim(32, 32);
    // This baseline maps x to cols and y to rows.
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    sgemm_kernel_coalesced<<<gridDim, blockDim,0,stream>>>(a, b, c, m, n, k);

    return cudaGetLastError();
}

__global__ void sgemm_kernel_coalesced(const float* a, const float* b, float* c,
                             int m, int n, int k) {
    
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    
    float sum=0;
    if(row<m && col<n){
        for(int i=0;i<k;i++){
            sum+=a[row*k+i]*b[col+i*n];
        }
        c[row*n+col]=sum;
    }
}
