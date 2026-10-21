#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>

// Custom implementations share the same row-major FP32 launch contract.
// cuBLAS remains the reference implementation managed by the harness.
struct KernelEntry {
    const char* name;
    const char* description;
    const char* source;
    bool (*available)() noexcept;
    cudaError_t (*launch)(const float*, const float*, float*, int, int, int,
                          cudaStream_t);
};

const std::vector<KernelEntry>& custom_kernels();
const KernelEntry* find_custom_kernel(const std::string& name);
