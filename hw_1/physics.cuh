#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#include <cuda_runtime.h>
#include <math.h>

#define SOURCE_RADIUS 8.0f
#define SOURCE_HEAT   1.0f


__host__ __device__ inline float sample_clamped(
    const float *grid, int width, int height, int x, int y)
{
    return (x < 0 || x >= width || y < 0 || y >= height) ? 0.0f : grid[y * width + x];
}

__host__ __device__ inline void source_position(
    int t, int width, int height, float *sx, float *sy)
{
    *sx = width  * (0.5f + 0.4f * cosf(0.01f * (float)t));
    *sy = height * (0.88f + 0.04f * sinf(0.017f * (float)t));
}

__host__ __device__ inline float heat_source(
    int t, int x, int y, int width, int height)
{
    float sx, sy;
    source_position(t, width, height, &sx, &sy);

    float dx = (float)x - sx;
    float dy = (float)y - sy;
    float dist_sq = dx * dx + dy * dy;

    if (dist_sq <= SOURCE_RADIUS * SOURCE_RADIUS) {
        return SOURCE_HEAT;
    }
    return 0.0f;
}

__host__ __device__ inline float heat_equation(
    float north, float south, float west, float east, float source)
{
    return 0.1f * north + 0.3f * south + 0.299f * west + 0.299f * east + source;
}


#endif
