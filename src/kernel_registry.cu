#include "kernel_registry.cuh"

#include "naive.cuh"
#include "coalesced.cuh"
#include "shared.cuh"
#include "register.cuh"
#include "tuned.cuh"
#include "warptiled.cuh"
#include "doublebuffer.cuh"
#include "asyncpipeline.cuh"

const std::vector<KernelEntry>& custom_kernels() {
    static const std::vector<KernelEntry> kernels{
        {"naive", "Uncoalesced baseline: thread x -> row, y -> column",
         "src/naive.cu", naive_available, launch_naive},
        {"coalesced", "Coalesced global-memory access: thread x -> column",
         "src/coalesced.cu", coalesced_available, launch_coalesced},
        {"shared", "Block tiling with shared-memory reuse",
         "src/shared.cu", shared_available, launch_shared},
        {"register", "Register tiling: multiple output values per thread",
         "src/register.cu", register_available, launch_register},
        {"tuned", "Further tile, vectorized-load, and scheduling experiments",
         "src/tuned.cu", tuned_available, launch_tuned},
        {"warptiled", "Warp tiling: user implementation",
         "src/warptiled.cu", warptiled_available, launch_warptiled},
        {"doublebuffer", "Double-buffered warp tiling: user implementation",
         "src/doublebuffer.cu", doublebuffer_available, launch_doublebuffer},
        {"doublebuffer_opt", "Double-buffered warp tiling with As swizzle and vectorized C stores",
         "src/doublebuffer.cu", doublebuffer_opt_available, launch_doublebuffer_opt},
        {"asyncpipeline", "Warp tiling with float4 A prefetch and hardware async B copies",
         "src/asyncpipeline.cu", asyncpipeline_available, launch_asyncpipeline},
    };
    return kernels;
}

const KernelEntry* find_custom_kernel(const std::string& name) {
    for (const auto& kernel : custom_kernels()) {
        if (name == kernel.name) return &kernel;
    }
    return nullptr;
}
