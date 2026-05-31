#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <vector>

#include "physics.cuh"

namespace {

struct Rgb {
    unsigned char r, g, b;
};

struct Image {
    int width = 0;
    int height = 0;
    std::vector<Rgb> pixels;
};

bool ensure_directory(const char *path)
{
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    char buffer[512];
    const size_t len = std::strlen(path);
    if (len >= sizeof(buffer)) {
        return false;
    }
    std::strcpy(buffer, path);

    for (size_t i = 1; buffer[i] != '\0'; ++i) {
        if (buffer[i] != '/') {
            continue;
        }
        buffer[i] = '\0';
        if (mkdir(buffer, 0755) != 0 && errno != EEXIST) {
            buffer[i] = '/';
            return false;
        }
        buffer[i] = '/';
    }

    return mkdir(buffer, 0755) == 0 || errno == EEXIST;
}

float find_max_temperature(
    const float *snapshots, int width, int height, int num_frames)
{
    const size_t grid_size = static_cast<size_t>(width) * height;
    float max_value = snapshots[0];

    for (int frame = 0; frame < num_frames; ++frame) {
        const float *grid = snapshots + frame * grid_size;
        for (size_t i = 0; i < grid_size; ++i) {
            max_value = std::max(max_value, grid[i]);
        }
    }

    return max_value > 0.0f ? max_value : 1.0f;
}

Rgb lerp_rgb(Rgb a, Rgb b, float t)
{
    auto mix = [&](unsigned char x, unsigned char y) {
        return static_cast<unsigned char>(x + (y - x) * t + 0.5f);
    };
    return {mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b)};
}

Rgb colormap_inferno(float t)
{
    static const Rgb stops[] = {
        {  4,   2,  18}, {  38,   4,  54}, {  68,  10,  84},
        {  88,  20, 110}, {110,  40, 120}, {130,  60, 110},
        {160,  70,  80}, {195,  80,  50}, {230, 120,  40},
        {252, 180,  40}, {252, 230, 120}, {255, 255, 220},
    };
    constexpr int stop_count = sizeof(stops) / sizeof(stops[0]);

    const float clamped_t = std::max(0.0f, std::min(t, 1.0f));
    const float scaled = clamped_t * (stop_count - 1);
    const int idx = static_cast<int>(scaled);
    const float frac = scaled - idx;

    if (idx >= stop_count - 1) {
        return stops[stop_count - 1];
    }
    return lerp_rgb(stops[idx], stops[idx + 1], frac);
}

void fill_rect(Image &image, int x0, int y0, int x1, int y1, Rgb color)
{
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);
    x1 = std::min(x1, image.width - 1);
    y1 = std::min(y1, image.height - 1);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            image.pixels[y * image.width + x] = color;
        }
    }
}

bool write_bmp(const char *path, const Image &image)
{
    const int width = image.width;
    const int height = image.height;
    const int row_stride = ((width * 3 + 3) / 4) * 4;
    const int file_size = 54 + row_stride * height;

    std::vector<unsigned char> row(row_stride, 0);

    FILE *file = std::fopen(path, "wb");
    if (file == nullptr) {
        return false;
    }

    unsigned char header[54] = {};
    header[0] = 'B';
    header[1] = 'M';
    header[2] = static_cast<unsigned char>(file_size);
    header[3] = static_cast<unsigned char>(file_size >> 8);
    header[4] = static_cast<unsigned char>(file_size >> 16);
    header[5] = static_cast<unsigned char>(file_size >> 24);
    header[10] = 54;
    header[14] = 40;
    header[18] = static_cast<unsigned char>(width);
    header[19] = static_cast<unsigned char>(width >> 8);
    header[20] = static_cast<unsigned char>(width >> 16);
    header[21] = static_cast<unsigned char>(width >> 24);
    header[22] = static_cast<unsigned char>(height);
    header[23] = static_cast<unsigned char>(height >> 8);
    header[24] = static_cast<unsigned char>(height >> 16);
    header[25] = static_cast<unsigned char>(height >> 24);
    header[26] = 1;
    header[28] = 24;

    std::fwrite(header, 1, 54, file);

    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            const Rgb &pixel = image.pixels[y * width + x];
            const int idx = x * 3;
            row[idx + 0] = pixel.b;
            row[idx + 1] = pixel.g;
            row[idx + 2] = pixel.r;
        }
        std::fwrite(row.data(), 1, row_stride, file);
    }

    std::fclose(file);
    return true;
}

bool save_frame_bmp(
    const char *path,
    const float *grid,
    int grid_w,
    int grid_h,
    int timestep,
    int cell_px,
    float vmax)
{
    constexpr int grid_gap = 1;
    constexpr int bar_w = 48;
    const Rgb grid_color = {28, 32, 48};
    const Rgb ring_color = {255, 255, 255};

    const int field_w = grid_w * cell_px + (grid_w + 1) * grid_gap;
    const int field_h = grid_h * cell_px + (grid_h + 1) * grid_gap;

    Image image;
    image.width = field_w + bar_w;
    image.height = field_h;
    image.pixels.assign(image.width * image.height, grid_color);

    for (int gy = 0; gy < grid_h; ++gy) {
        for (int gx = 0; gx < grid_w; ++gx) {
            const float t = (vmax > 0.0f) ? (grid[gy * grid_w + gx] / vmax) : 0.0f;
            const Rgb color = colormap_inferno(t);

            const int x0 = grid_gap + gx * (cell_px + grid_gap);
            const int y0 = grid_gap + gy * (cell_px + grid_gap);
            fill_rect(image, x0, y0, x0 + cell_px - 1, y0 + cell_px - 1, color);
        }
    }

    float sx, sy;
    source_position(timestep, grid_w, grid_h, &sx, &sy);
    const int cx = static_cast<int>(sx * (cell_px + grid_gap) + cell_px * 0.5f);
    const int cy = static_cast<int>(sy * (cell_px + grid_gap) + cell_px * 0.5f);
    const int radius = static_cast<int>(SOURCE_RADIUS * (cell_px + grid_gap) + 0.5f);
    const int r_outer = radius + 1;
    const int r_inner = std::max(radius - 1, 0);

    for (int dy = -r_outer; dy <= r_outer; ++dy) {
        for (int dx = -r_outer; dx <= r_outer; ++dx) {
            const int dist_sq = dx * dx + dy * dy;
            if (dist_sq > r_outer * r_outer || dist_sq < r_inner * r_inner) {
                continue;
            }
            const int x = cx + dx;
            const int y = cy + dy;
            if (x >= 0 && y >= 0 && x < image.width && y < image.height) {
                image.pixels[y * image.width + x] = ring_color;
            }
        }
    }

    return write_bmp(path, image);
}

}  // namespace

int compare_grid(const float *a, const float *b, int count, float tolerance)
{
    for (int i = 0; i < count; ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        if (diff > tolerance) {
            std::fprintf(stderr,
                         "Mismatch at index %d: a=%f b=%f diff=%f\n",
                         i, a[i], b[i], diff);
            return 0;
        }
    }
    return 1;
}

int save_frames(
    const char *output_dir,
    const float *snapshots,
    int width,
    int height,
    int num_frames,
    int save_every)
{
    if (!ensure_directory(output_dir)) {
        std::fprintf(stderr, "Failed to create output directory: %s\n", output_dir);
        return -1;
    }

    const size_t grid_size = static_cast<size_t>(width) * height;
    const float vmax = find_max_temperature(snapshots, width, height, num_frames);
    constexpr int cell_px = 4;

    std::printf("Saving frames to %s/ (every %d steps, vmax=%.4f)\n",
                output_dir, save_every, vmax);

    for (int frame = 0; frame < num_frames; ++frame) {
        if (frame % save_every != 0) {
            continue;
        }

        char path[512];
        std::snprintf(path, sizeof(path), "%s/frame_%04d.bmp", output_dir, frame);

        const float *grid = snapshots + frame * grid_size;
        if (!save_frame_bmp(path, grid, width, height, frame, cell_px, vmax)) {
            std::fprintf(stderr, "Failed to write %s\n", path);
            return -1;
        }
    }

    return 0;
}
