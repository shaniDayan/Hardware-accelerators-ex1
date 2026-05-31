#include "simulation.cuh"

#include <cuda_runtime.h>
#include "physics.cuh"

/* Optimization 1: weights are allocated and copied once */
static float *d_weights_optimized = NULL;

/* Optimization 2: main simulation buffers are allocated once */
static float *d_buffer_a = NULL;
static float *d_buffer_b = NULL;
static size_t optimized_grid_bytes = 0;

static void swap_buffers(float **a, float **b)
{
    float *tmp = *a;
    *a = *b;
    *b = tmp;
}

/* Optimization 3: shared memory tiled stencil kernel */
__global__ void optimized_heat_step_kernel(
    const float *current,
    const float *weights,
    float *next,
    int width,
    int height,
    int timestep)
{
    extern __shared__ float shared_tile[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int shared_width = blockDim.x + 2 * STENCIL_RADIUS;
    int shared_height = blockDim.y + 2 * STENCIL_RADIUS;

    int block_start_x = blockIdx.x * blockDim.x;
    int block_start_y = blockIdx.y * blockDim.y;

    /*
     * Optimization 3:
     * Load tile + halo into shared memory.
     */
    for (int sy = threadIdx.y; sy < shared_height; sy += blockDim.y) {
        for (int sx = threadIdx.x; sx < shared_width; sx += blockDim.x) {
            int gx = block_start_x + sx - STENCIL_RADIUS;
            int gy = block_start_y + sy - STENCIL_RADIUS;

            if (gx < 0 || gx >= width || gy < 0 || gy >= height) {
                shared_tile[sy * shared_width + sx] = 0.0f;
            } else {
                shared_tile[sy * shared_width + sx] =
                    current[gy * width + gx];
            }
        }
    }

    /*
     * Optimization 3:
     * Make sure the whole tile + halo is loaded before using it.
     */
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    int local_x = threadIdx.x + STENCIL_RADIUS;
    int local_y = threadIdx.y + STENCIL_RADIUS;

    float source = heat_source(timestep, x, y, width, height);

    /*
     * Optimization 3:
     * Read stencil values from shared memory instead of global memory.
     */
    for (int dy = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++) {
        for (int dx = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++) {
            float value =
                shared_tile[(local_y + dy) * shared_width + (local_x + dx)];

            float weight =
                weights[(dy + STENCIL_RADIUS) * STENCIL_SIZE +
                        (dx + STENCIL_RADIUS)];

            source += value * weight;
        }
    }

    next[y * width + x] = source;
}

void heat_simulate_optimized_init(
    int width,
    int height,
    int num_simulations,
    const float *weights)
{
    (void)num_simulations;

    /*
     * Optimization 1:
     * Allocate and copy weights once.
     */
    CUDA_CHECK(cudaMalloc(
        &d_weights_optimized,
        WEIGHTS_COUNT * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_weights_optimized,
        weights,
        WEIGHTS_COUNT * sizeof(float),
        cudaMemcpyHostToDevice));

    optimized_grid_bytes =
        (size_t)width * (size_t)height * sizeof(float);

    /*
     * Optimization 2:
     * Allocate current/next buffers once.
     */
    CUDA_CHECK(cudaMalloc(
        &d_buffer_a,
        optimized_grid_bytes));

    CUDA_CHECK(cudaMalloc(
        &d_buffer_b,
        optimized_grid_bytes));
}

void heat_simulate_optimized_finalize(void)
{
    /*
     * Optimization 1:
     * Free weights once.
     */
    if (d_weights_optimized != NULL) {
        CUDA_CHECK(cudaFree(d_weights_optimized));
        d_weights_optimized = NULL;
    }

    /*
     * Optimization 2:
     * Free reusable buffers once.
     */
    if (d_buffer_a != NULL) {
        CUDA_CHECK(cudaFree(d_buffer_a));
        d_buffer_a = NULL;
    }

    if (d_buffer_b != NULL) {
        CUDA_CHECK(cudaFree(d_buffer_b));
        d_buffer_b = NULL;
    }

    optimized_grid_bytes = 0;
}

void heat_simulate_optimized(
    float **initial_states,
    const float *weights,
    float **final_states,
    int num_simulations,
    int width,
    int height,
    int start_step,
    int num_steps)
{
    (void)weights;

    /*
    * Optimization 3:
    * Block size chosen for shared memory tiling.
    */
    dim3 block(16, 16);
    dim3 block(16, 16);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    /*
     * Optimization 3:
     * Shared memory size = tile + halo.
     */
    size_t shared_width = block.x + 2 * STENCIL_RADIUS;
    size_t shared_height = block.y + 2 * STENCIL_RADIUS;
    size_t shared_bytes =
        shared_width * shared_height * sizeof(float);

    for (int s = 0; s < num_simulations; s++) {
        /*
         * Optimization 2:
         * Reuse the already allocated buffers.
         */
        float *d_current = d_buffer_a;
        float *d_next = d_buffer_b;

        CUDA_CHECK(cudaMemcpy(
            d_current,
            initial_states[s],
            optimized_grid_bytes,
            cudaMemcpyHostToDevice));

        for (int step = start_step; step < start_step + num_steps; step++) {
            optimized_heat_step_kernel<<<grid, block, shared_bytes>>>(
                d_current,
                d_weights_optimized,
                d_next,
                width,
                height,
                step);

            CUDA_CHECK(cudaGetLastError());

            swap_buffers(&d_current, &d_next);
        }

        CUDA_CHECK(cudaMemcpy(
            final_states[s],
            d_current,
            optimized_grid_bytes,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaDeviceSynchronize());
    }
}