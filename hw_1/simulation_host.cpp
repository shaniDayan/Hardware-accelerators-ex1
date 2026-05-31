#include "simulation.cuh"

#include <algorithm>

void gpu_device_sync(void)
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

float *host_alloc_pinned(size_t count)
{
    float *ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&ptr), count * sizeof(float)));
    return ptr;
}

float *host_alloc_pinned_zero(size_t count)
{
    float *ptr = host_alloc_pinned(count);
    std::fill(ptr, ptr + count, 0.0f);
    return ptr;
}

void host_free_pinned(float *ptr)
{
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}
