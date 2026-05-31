#include "simulation.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "physics.cuh"

__global__ void reference_heat_step_kernel(
    const float *current,
    const float *weights,
    float *next,
    int width,
    int height,
    int timestep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float source = heat_source(timestep, x, y, width, height);
    for (int dy = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++) {
        for (int dx = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++) {
            source += sample_clamped(current, width, height, x + dx, y + dy)
                    * weights[(dy + STENCIL_RADIUS) * STENCIL_SIZE + (dx + STENCIL_RADIUS)];
        }
    }

    next[y * width + x] = source;
}

static void swap_buffers(float **a, float **b)
{
    float *tmp = *a;
    *a = *b;
    *b = tmp;
}

void heat_simulate_reference_init(
    int width,
    int height,
    int num_simulations,
    const float *weights)
{
}

void heat_simulate_reference_finalize(void)
{
}

void heat_simulate_reference(
    float **initial_states,
    const float *weights,
    float **final_states,
    int num_simulations,
    int width,
    int height,
    int start_step,
    int num_steps)
{
    float *d_current = NULL;
    float *d_next = NULL;
    float *d_weights = NULL;
    size_t grid_bytes = (size_t)width * (size_t)height * sizeof(float);

    for (int s = 0; s < num_simulations; s++) {
        CUDA_CHECK(cudaMalloc(&d_current, grid_bytes));
        CUDA_CHECK(cudaMalloc(&d_next, grid_bytes));
        CUDA_CHECK(cudaMalloc(&d_weights, WEIGHTS_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_current, initial_states[s], grid_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights, weights, WEIGHTS_COUNT * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(32, 32);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        for (int step = start_step; step < start_step + num_steps; step++) {
            reference_heat_step_kernel<<<grid, block>>>(
                d_current, d_weights, d_next, width, height, step);
            CUDA_CHECK(cudaGetLastError());
            swap_buffers(&d_current, &d_next);
        }

        CUDA_CHECK(cudaMemcpy(final_states[s], d_current, grid_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));
        CUDA_CHECK(cudaFree(d_weights));
    }
}

