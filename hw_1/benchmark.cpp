#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "simulation.cuh"
#include "utils/utils.h"

static const int WIDTH = 128;
static const int HEIGHT = 128;
static const int STEPS = 300;
static const int NUM_SIMULATIONS = 10;
static const int WARMUP_RUNS = 10;
static const int REPEATS = 20;
static const float COMPARE_TOLERANCE = 1e-5f;

static float stencil_weights[WEIGHTS_COUNT] = {
    0.10f, 0.12f, 0.14f, 0.16f, 0.18f, 0.20f, 0.22f, 0.20f, 0.18f, 0.16f, 0.14f, 0.12f, 0.10f,
    0.15f, 0.18f, 0.21f, 0.24f, 0.27f, 0.30f, 0.33f, 0.30f, 0.27f, 0.24f, 0.21f, 0.18f, 0.15f,
    0.22f, 0.26f, 0.30f, 0.34f, 0.38f, 0.42f, 0.46f, 0.42f, 0.38f, 0.34f, 0.30f, 0.26f, 0.22f,
    0.32f, 0.38f, 0.44f, 0.50f, 0.56f, 0.62f, 0.68f, 0.62f, 0.56f, 0.50f, 0.44f, 0.38f, 0.32f,
    0.45f, 0.54f, 0.63f, 0.72f, 0.81f, 0.90f, 0.99f, 0.90f, 0.81f, 0.72f, 0.63f, 0.54f, 0.45f,
    0.62f, 0.74f, 0.86f, 0.98f, 1.10f, 1.22f, 1.34f, 1.22f, 1.10f, 0.98f, 0.86f, 0.74f, 0.62f,
    0.85f, 1.00f, 1.15f, 1.30f, 1.45f, 1.60f, 2.50f, 1.60f, 1.45f, 1.30f, 1.15f, 1.00f, 0.85f,
    1.20f, 1.40f, 1.60f, 1.80f, 2.00f, 2.20f, 2.40f, 2.20f, 2.00f, 1.80f, 1.60f, 1.40f, 1.20f,
    1.55f, 1.80f, 2.05f, 2.30f, 2.55f, 2.80f, 3.05f, 2.80f, 2.55f, 2.30f, 2.05f, 1.80f, 1.55f,
    1.85f, 2.15f, 2.45f, 2.75f, 3.05f, 3.35f, 3.65f, 3.35f, 3.05f, 2.75f, 2.45f, 2.15f, 1.85f,
    2.05f, 2.40f, 2.75f, 3.10f, 3.45f, 3.80f, 4.15f, 3.80f, 3.45f, 3.10f, 2.75f, 2.40f, 2.05f,
    2.15f, 2.50f, 2.85f, 3.20f, 3.55f, 3.90f, 4.25f, 3.90f, 3.55f, 3.20f, 2.85f, 2.50f, 2.15f,
    2.10f, 2.45f, 2.80f, 3.15f, 3.50f, 3.85f, 4.20f, 3.85f, 3.50f, 3.15f, 2.80f, 2.45f, 2.10f,
};

static float *initial[NUM_SIMULATIONS];
static float *reference[NUM_SIMULATIONS];
static float *optimized[NUM_SIMULATIONS];

typedef void (*SimulateFn)(
    float **initial_states,
    const float *weights,
    float **final_states,
    int num_simulations,
    int width,
    int height,
    int start_step,
    int num_steps);

static void normalize_stencil_weights(float *weights)
{
    float sum = 0.0f;
    for (int i = 0; i < WEIGHTS_COUNT; i++) {
        sum += weights[i];
    }
    for (int i = 0; i < WEIGHTS_COUNT; i++) {
        weights[i] /= sum + 0.0001f;
    }
}

static void free_buffers(void)
{
    for (int s = 0; s < NUM_SIMULATIONS; s++) {
        host_free_pinned(initial[s]);
        host_free_pinned(reference[s]);
        host_free_pinned(optimized[s]);
    }
}

static bool alloc_buffers(size_t grid_size)
{
    for (int s = 0; s < NUM_SIMULATIONS; s++) {
        initial[s] = host_alloc_pinned(grid_size);
        reference[s] = host_alloc_pinned(grid_size);
        optimized[s] = host_alloc_pinned(grid_size);
    }
    return true;
}

static void init_random_states(size_t grid_size)
{
    srand(42);
    for (int s = 0; s < NUM_SIMULATIONS; s++) {
        for (size_t i = 0; i < grid_size; i++) {
            initial[s][i] = (float)rand() / (float)RAND_MAX * 0.1f;
        }
    }
}

static void run_simulation(SimulateFn simulate, float **out, int num_steps)
{
    simulate(initial, stencil_weights, out, NUM_SIMULATIONS, WIDTH, HEIGHT, 0, num_steps);
}

static bool results_match(size_t grid_size)
{
    for (int s = 0; s < NUM_SIMULATIONS; s++) {
        if (!compare_grid(reference[s], optimized[s], (int)grid_size, COMPARE_TOLERANCE)) {
            fprintf(stderr, "Mismatch at simulation %d\n", s);
            return false;
        }
    }
    return true;
}

static void print_stats(const char *label, double times[], int count)
{
    double sum = 0.0;
    double min_t = times[0];
    double max_t = times[0];

    for (int i = 0; i < count; i++) {
        sum += times[i];
        if (times[i] < min_t) min_t = times[i];
        if (times[i] > max_t) max_t = times[i];
    }

    double mean = sum / count;

    double var_sum = 0.0;
    for (int i = 0; i < count; i++) {
        double d = times[i] - mean;
        var_sum += d * d;
    }
    double stddev = sqrt(var_sum / count);

    printf("%s time (%d runs): mean=%.2f ms  min=%.2f  max=%.2f  stddev=%.2f\n",
           label, count, mean, min_t, max_t, stddev);
}

static void benchmark(SimulateFn simulate, float **out, const char *label)
{
    for (int i = 0; i < WARMUP_RUNS; i++) {
        run_simulation(simulate, out, STEPS);
        gpu_device_sync();
    }

    double times[REPEATS];
    for (int i = 0; i < REPEATS; i++) {
        gpu_device_sync();
        auto t0 = std::chrono::steady_clock::now();
        run_simulation(simulate, out, STEPS);
        gpu_device_sync();
        auto t1 = std::chrono::steady_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    print_stats(label, times, REPEATS);
}

int main()
{
    size_t grid_size = (size_t)WIDTH * HEIGHT;

    printf("Heat diffusion benchmark\n");
    printf("Grid: %d x %d, steps: %d, simulations: %d\n",
           WIDTH, HEIGHT, STEPS, NUM_SIMULATIONS);

    normalize_stencil_weights(stencil_weights);

    if (!alloc_buffers(grid_size)) {
        return 1;
    }

    init_random_states(grid_size);

    heat_simulate_reference_init(WIDTH, HEIGHT, NUM_SIMULATIONS, stencil_weights);
    heat_simulate_optimized_init(WIDTH, HEIGHT, NUM_SIMULATIONS, stencil_weights);

    run_simulation(heat_simulate_reference, reference, 1);
    run_simulation(heat_simulate_optimized, optimized, 1);
    gpu_device_sync();

    if (results_match(grid_size)) {
        printf("PASS: optimized GPU matches reference (%d simulations)\n", NUM_SIMULATIONS);
    } else {
        printf("FAIL: optimized GPU differs from reference\n");
    }

    benchmark(heat_simulate_reference, reference, "Reference");
    benchmark(heat_simulate_optimized, optimized, "Optimized");

    heat_simulate_reference_finalize();
    heat_simulate_optimized_finalize();

    free_buffers();
    printf("Done.\n");
    return 0;
}
