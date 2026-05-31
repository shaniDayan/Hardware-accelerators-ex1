#ifndef SIMULATION_H
#define SIMULATION_H

#include <stddef.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define STENCIL_RADIUS 6
#define STENCIL_SIZE (2 * STENCIL_RADIUS + 1)
#define WEIGHTS_COUNT (STENCIL_SIZE * STENCIL_SIZE)

void heat_simulate_reference_init(
    int width,
    int height,
    int num_simulations,
    const float *weights);

void heat_simulate_reference_finalize(void);

void heat_simulate_reference(
    float **initial_states,
    const float *weights,
    float **final_states,
    int num_simulations,
    int width,
    int height,
    int start_step,
    int num_steps);

void heat_simulate_optimized_init(
    int width,
    int height,
    int num_simulations,
    const float *weights);

void heat_simulate_optimized_finalize(void);

void heat_simulate_optimized(
    float **initial_states,
    const float *weights,
    float **final_states,
    int num_simulations,
    int width,
    int height,
    int start_step,
    int num_steps);



#define CUDA_CHECK(call)                                                 \
do {                                                                     \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(1);                                                         \
    }                                                                    \
} while (0)


void gpu_device_sync(void);
float *host_alloc_pinned(size_t count);
float *host_alloc_pinned_zero(size_t count);
void host_free_pinned(float *ptr);

#endif
