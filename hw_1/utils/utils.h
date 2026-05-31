#ifndef UTILS_H
#define UTILS_H

int compare_grid(const float *a, const float *b, int count, float tolerance);

int save_frames(
    const char *output_dir,
    const float *snapshots,
    int width,
    int height,
    int num_frames,
    int save_every);

#endif
