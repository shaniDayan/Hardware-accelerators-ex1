#include "simulation.cuh"

#include <cuda_runtime.h>
#include "physics.cuh"


/* Optimization 5: weights are stored in constant memory */
__constant__ float d_weights_const[WEIGHTS_COUNT];

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
/*
 * Optimization 7:
 * Use __restrict__ because current and next are different buffers.
 */
__global__ void optimized_heat_step_kernel(
    const float *__restrict__ current,
    float *__restrict__ next,
    int width,
    int height,
    int timestep)
{
    extern __shared__ float shared_tile[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /*
     * Optimization 9:
     * Use the z dimension of the CUDA grid to process all simulations
     * in the same kernel launch.
     */
    int sim = blockIdx.z;
    size_t grid_cells = (size_t)width * (size_t)height;
    size_t sim_offset = (size_t)sim * grid_cells;

    const float *sim_current = current + sim_offset;
    float *sim_next = next + sim_offset;

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
                    sim_current[gy * width + gx];
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
     * Optimization 7:
     * Unroll the fixed-size stencil loops.
     */
    #pragma unroll
    for (int dy = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++) {
        #pragma unroll
        for (int dx = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++) {
            float value =
                shared_tile[(local_y + dy) * shared_width + (local_x + dx)];

            float weight =
                d_weights_const[(dy + STENCIL_RADIUS) * STENCIL_SIZE +
                                (dx + STENCIL_RADIUS)];

            source += value * weight;
        }
    }

    sim_next[y * width + x] = source;
}

void heat_simulate_optimized_init(
    int width,
    int height,
    int num_simulations,
    const float *weights)
{

    /*
    * Optimization 5:
    * Copy weights once into constant memory.
    */
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_weights_const,
        weights,
        WEIGHTS_COUNT * sizeof(float)));
    optimized_grid_bytes =
        (size_t)width * (size_t)height * sizeof(float);

    /*
    * Optimization 9:
    * Allocate enough space for all simulations.
    */
    size_t total_bytes = optimized_grid_bytes * (size_t)num_simulations;

    CUDA_CHECK(cudaMalloc(
        &d_buffer_a,
        total_bytes));

    CUDA_CHECK(cudaMalloc(
        &d_buffer_b,
        total_bytes));
}

void heat_simulate_optimized_finalize(void)
{

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
    * Optimization 4:
    * Selected block size after benchmarking multiple options.
    * 32x8 was fastest on lambda for this workload.
    */
    dim3 block(32,8);

    /*
    * Optimization 9:
    * Add simulation dimension to the CUDA grid.
    */
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        num_simulations);

    /*
     * Optimization 3:
     * Shared memory size = tile + halo.
     */
    size_t shared_width = block.x + 2 * STENCIL_RADIUS;
    size_t shared_height = block.y + 2 * STENCIL_RADIUS;
    size_t shared_bytes =
        shared_width * shared_height * sizeof(float);

    size_t grid_cells = (size_t)width * (size_t)height;

    /*
    * Optimization 9:
    * Copy all simulation inputs into one large device buffer.
    */
    for (int s = 0; s < num_simulations; s++) {
        CUDA_CHECK(cudaMemcpy(
            d_buffer_a + (size_t)s * grid_cells,
            initial_states[s],
            optimized_grid_bytes,
            cudaMemcpyHostToDevice));
    }

    /*
    * Optimization 9:
    * Run all simulations together.
    */
    float *d_current = d_buffer_a;
    float *d_next = d_buffer_b;

    for (int step = start_step; step < start_step + num_steps; step++) {
        optimized_heat_step_kernel<<<grid, block, shared_bytes>>>(
            d_current,
            d_next,
            width,
            height,
            step);

        CUDA_CHECK(cudaGetLastError());

        swap_buffers(&d_current, &d_next);
    }

    /*
    * Optimization 9:
    * Copy all simulation outputs back.
    */
    for (int s = 0; s < num_simulations; s++) {
        CUDA_CHECK(cudaMemcpy(
            final_states[s],
            d_current + (size_t)s * grid_cells,
            optimized_grid_bytes,
            cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}