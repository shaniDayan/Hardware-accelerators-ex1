#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "simulation.cuh"
#include "utils/utils.h"

static const int WIDTH = 128;
static const int HEIGHT = 128;
static const int STEPS = 500;
static const int SAVE_EVERY = 10;

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

static void print_usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s <reference|optimized>\n"
            "\n"
            "Examples:\n"
            "  %s reference\n"
            "  %s optimized\n",
            program, program, program);
}

static void run_plot(
    SimulateFn simulate,
    void (*init)(int, int, int, const float *),
    void (*finalize)(void),
    const char *name,
    size_t grid_size)
{
    float *initial = host_alloc_pinned_zero(grid_size);
    float *buf_a = host_alloc_pinned(grid_size);
    float *buf_b = host_alloc_pinned(grid_size);
    std::vector<float> snapshots(grid_size * (STEPS + 1));

    printf("Heat diffusion plot (%s)\n", name);
    printf("Grid: %d x %d, steps: %d (chunk size: %d)\n",
           WIDTH, HEIGHT, STEPS, SAVE_EVERY);

    init(WIDTH, HEIGHT, 1, stencil_weights);

    memcpy(snapshots.data(), initial, grid_size * sizeof(float));

    float *current = buf_a;
    float *next = buf_b;
    memcpy(current, initial, grid_size * sizeof(float));

    for (int start = 0; start < STEPS; start += SAVE_EVERY) {
        int chunk = SAVE_EVERY;
        if (start + chunk > STEPS) {
            chunk = STEPS - start;
        }

        float *inputs[] = {current};
        float *outputs[] = {next};
        simulate(inputs, stencil_weights, outputs, 1, WIDTH, HEIGHT, start, chunk);

        float *tmp = current;
        current = next;
        next = tmp;

        int frame = start + chunk;
        if (frame % SAVE_EVERY == 0) {
            memcpy(snapshots.data() + frame * grid_size, current, grid_size * sizeof(float));
        }
    }

    gpu_device_sync();
    finalize();

    if (save_frames("frames", snapshots.data(), WIDTH, HEIGHT, STEPS + 1, SAVE_EVERY) != 0) {
        host_free_pinned(initial);
        host_free_pinned(buf_a);
        host_free_pinned(buf_b);
        exit(1);
    }

    host_free_pinned(initial);
    host_free_pinned(buf_a);
    host_free_pinned(buf_b);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    normalize_stencil_weights(stencil_weights);

    size_t grid_size = (size_t)WIDTH * HEIGHT;

    if (strcmp(argv[1], "reference") == 0) {
        run_plot(heat_simulate_reference,
                 heat_simulate_reference_init,
                 heat_simulate_reference_finalize,
                 "reference",
                 grid_size);
    } else if (strcmp(argv[1], "optimized") == 0) {
        run_plot(heat_simulate_optimized,
                 heat_simulate_optimized_init,
                 heat_simulate_optimized_finalize,
                 "optimized",
                 grid_size);
    } else {
        print_usage(argv[0]);
        return 1;
    }

    printf("Done.\n");
    return 0;
}
