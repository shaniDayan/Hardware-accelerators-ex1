#include "simulation.cuh"

#include <cuda_runtime.h>
#include "physics.cuh"

void heat_simulate_optimized_init(
    int width,
    int height,
    int num_simulations,
    const float *weights)
{
}

void heat_simulate_optimized_finalize(void)
{
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
}
