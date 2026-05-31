## Optimization 1: Avoid Reallocating and Recopying Weights per Simulation

### Current behavior

Currently, the reference implementation performs the following operations inside the loop over simulations:

```text
cudaMalloc
cudaMemcpy weights
run simulation
cudaFree
```

This means that for every simulation, the code allocates GPU memory for `d_weights`, copies the same `weights` array from host to device, and then frees it.

### Problem

The `weights` array is identical for all simulations.

Therefore, repeatedly allocating and copying it is unnecessary overhead.

### Optimization idea

Move the allocation and copy of `d_weights` outside the per-simulation loop.

Instead:

```text
Allocate d_weights once in init
Copy weights once to the GPU
Reuse d_weights for all simulations
Free d_weights once in finalize
```

### Expected benefit

This reduces overhead from repeated `cudaMalloc`, `cudaMemcpy`, and `cudaFree` calls.

It is especially useful when `num_simulations` is large, because the same weights are reused across all simulations.

---
## Optimization 2: Avoid Reallocating `d_current` and `d_next` per Simulation

### Current behavior

Currently, the reference implementation allocates and frees the main device buffers inside the loop over simulations:

```text
cudaMalloc d_current
cudaMalloc d_next
copy initial state
run simulation
copy final state
cudaFree d_current
cudaFree d_next
```
This means that for every simulation, the code allocates GPU memory for `d_current` and `d_next`, uses them, and then frees them.

### Problem

All simulations use the same width and height.

Therefore, the required buffer size is the same for every simulation:

`grid_bytes = width * height * sizeof(float);`

Repeatedly allocating and freeing `d_current` and `d_next` is unnecessary overhead.

### Optimization idea

Move the allocation of `d_current` and `d_next` outside the per-simulation loop.

Instead:
```text
Allocate d_current once in init
Allocate d_next once in init
Reuse both buffers for all simulations
Free both buffers once in finalize
```
For every simulation, we still copy the new input state into d_current.

So we reuse the GPU memory allocation, but not the old simulation data.

---

## Optimization 3: Shared Memory Tiling

Each thread computes one output cell, but neighboring threads read many of the same input cells from global memory.

To reduce repeated global memory reads, each CUDA block can load a rectangular tile of the input grid into shared memory.

Because stencil computation needs neighboring cells, the block must load not only its own tile, but also a halo around the tile.

For a block of size:
```c
blockDim.x by blockDim.y
```
and stencil radius:
```c
STENCIL_RADIUS
```
the shared memory tile size is:

`(blockDim.x + 2 * STENCIL_RADIUS)` by
`(blockDim.y + 2 * STENCIL_RADIUS)`

Then:

Threads cooperatively load the tile and halo from global memory into shared memory.
`__syncthreads()` ensures all values are loaded.
Each thread computes its output cell using shared memory.
Each thread writes one result to global memory.

This reduces repeated global memory reads because values loaded once into shared memory can be reused by many threads in the same block.

---
## Optimization 4: Tune Block Size for Shared Memory Tiling

### Current behavior

The reference implementation uses:

```c
dim3 block(32, 32);
```
This means each block has:

32 * 32 = 1024 threads

which is usually the maximum number of threads allowed per block.

### Problem

After adding shared memory tiling, each block needs to load not only its own output tile, but also a halo around it.

Since:

`STENCIL_RADIUS = 6`

a 32 x 32 block needs a shared-memory tile of:

(32 + 2*6) x (32 + 2*6)
= 44 x 44
= 1936 floats

This may be too heavy and may reduce performance.

### Optimization idea

Try different block sizes and benchmark them.

For example:

`dim3 block(8, 8);`
`dim3 block(16, 16);`
`dim3 block(32, 8);`
`dim3 block(8, 32);`
`dim3 block(32, 16);`

### Block-size benchmark

After adding shared memory tiling, we benchmarked several block sizes on lambda.

| Block size | PASS/FAIL | Optimized mean time |
|---|---|---:|
| 8x8 | PASS | 17.00 ms |
| 16x16 | PASS | 15.79 ms |
| 32x8 | PASS | 12.82 ms |
| 8x32 | PASS | 16.21 ms |
| 64x4 | PASS | 14.48 ms |
| 64x8 | PASS | 17.42 ms |
| 32x16 | PASS | 16.04 ms |

The best tested configuration was `32x8`.

This likely works well because the grid is stored in row-major order, so a block that is wider in the x direction gives better memory access behavior while keeping shared memory usage moderate.

### Expected benefit

Changing the block size can improve occupancy, shared memory usage, and memory access behavior.

The best block size is not guaranteed in advance, so we should benchmark several options on the target GPU and choose the fastest correct one.

---

## Optimization 5: Store Weights in Constant Memory

### Current behavior

The optimized implementation currently stores the stencil weights in regular device memory.

The weights are allocated once and copied once:

```c
cudaMalloc(&d_weights_optimized, WEIGHTS_COUNT * sizeof(float));
cudaMemcpy(d_weights_optimized, weights, WEIGHTS_COUNT * sizeof(float), cudaMemcpyHostToDevice);
```
Then each thread reads the weights from global memory inside the stencil loop.

### Problem

The weights are small, read-only, and identical for all threads, timesteps, and simulations.

Using regular global memory for this data may be unnecessary.

### Optimization idea

Store the weights in CUDA constant memory:
```c
__constant__ float d_weights_const[WEIGHTS_COUNT];
```
Copy the weights once in heat_simulate_optimized_init using:
```c
cudaMemcpyToSymbol(d_weights_const, weights, WEIGHTS_COUNT * sizeof(float));
```
Then the kernel reads directly from d_weights_const.

### Expected benefit

Constant memory is cached and is especially efficient when many threads in a warp read the same address.

In this stencil computation, all threads iterate over the same weight indices, so constant memory is a good fit.

---

## Optimization 6: Fast Path for Interior Blocks

### Current behavior

In the shared-memory tiled kernel, every block loads a tile plus halo into shared memory.

For every value loaded, the code checks whether the global coordinates are outside the grid:

```c
if (gx < 0 || gx >= width || gy < 0 || gy >= height) {
    shared_tile[...] = 0.0f;
} else {
    shared_tile[...] = current[gy * width + gx];
}
```
This is required for boundary blocks, because the reference behavior returns `0.0f` for out-of-bounds accesses.

### Problem

Most blocks are interior blocks.

For interior blocks, the entire tile plus halo is inside the grid, so the boundary check is unnecessary.

Performing this check for every shared-memory load adds extra overhead.

### Optimization idea

Add a fast path for interior blocks.

If the block’s tile plus halo is fully inside the grid, load directly from global memory without boundary checks.

Otherwise, use the safe boundary-handling version.

Conceptually:
```text
if block is interior:
    load tile + halo directly
else:
    load tile + halo with boundary checks
```
### Expected benefit

This reduces branch overhead during shared-memory loading for interior blocks.

It should help especially because the stencil radius is large (STENCIL_RADIUS = 6), so each block loads many halo values.

### Result

| Grid size | Without Opt 6 | With Opt 6 | Better? |
|---|---:|---:|---|
| 128x128 | 12.82 ms | 12.84 ms | Without Opt 6 |
| 256x256 | 26.94 ms | 26.80 ms | With Opt 6 |

### Conclusion

This optimization slightly improved performance for a larger `256x256` grid, but did not improve the provided `128x128` benchmark.

For now, we decided not to keep it in the final code. If the grading benchmark includes larger grids, we may restore this optimization later.

---

## Optimization 7: Loop Unrolling and Pointer Restriction

### Idea

The stencil radius is fixed at compile time:

```c
STENCIL_RADIUS = 6
```
Therefore, the stencil loop always performs:

13 x 13 = 169 iterations

We added `#pragma unroll` to the stencil loops to encourage the compiler to unroll them and reduce loop overhead.

We also marked kernel pointers with `__restrict__`:
```c
const float *__restrict__ current
float *__restrict__ next
```
This tells the compiler that current and next do not alias, which is true because they point to different device buffers.

### Result
| Version | PASS/FAIL | Optimized mean time |
|---|---|---:|
| Before Optimization 7 | PASS | 12.82 ms |
| After Optimization 7 | PASS | 12.69 ms |

### Conclusion

This gave a small improvement and preserved correctness, so we kept it.

---
## Optimization 9: Process All Simulations Together Using a 3D CUDA Grid
### Current behavior

Before this optimization, the optimized implementation still processed the simulations one after another:
```c
for (int s = 0; s < num_simulations; s++) {
    copy initial_states[s];

    for (int step = start_step; step < start_step + num_steps; step++) {
        launch kernel for simulation s;
        swap buffers;
    }

    copy final_states[s];
}
```

This means that for:

`num_simulations = 10`
`num_steps = 300`

the code launched:

10 * 300 = 3000 kernel launches
## Problem

Each simulation is independent from the others.

Therefore, running them sequentially creates unnecessary kernel-launch overhead and does not fully use the GPU parallelism across simulations.

## Optimization idea

Use the z dimension of the CUDA grid to process all simulations in the same kernel launch.

Instead of launching one kernel per simulation per timestep, we launch one kernel per timestep, where:

`blockIdx.z`

represents the simulation index.

Conceptually:
```text
Before:
for each simulation:
    for each timestep:
        launch kernel

After:
for each timestep:
    launch one kernel for all simulations
```
The CUDA grid becomes 3D:
```text
dim3 grid(
    (width + block.x - 1) / block.x,
    (height + block.y - 1) / block.y,
    num_simulations);
```
Inside the kernel:

int sim = blockIdx.z;

size_t grid_cells = (size_t)width * (size_t)height;
size_t sim_offset = (size_t)sim * grid_cells;

const float *sim_current = current + sim_offset;
float *sim_next = next + sim_offset;

Each simulation gets its own slice inside the large device buffers.

Buffer layout

Instead of allocating space for one simulation:

width * height

we allocate space for all simulations:

num_simulations * width * height

So the device buffers are arranged like this:

d_buffer_a:
[ simulation 0 grid ][ simulation 1 grid ][ simulation 2 grid ] ... [ simulation 9 grid ]

d_buffer_b:
[ simulation 0 grid ][ simulation 1 grid ][ simulation 2 grid ] ... [ simulation 9 grid ]

Each simulation is independent, but all of them are processed in the same kernel launch.

Expected benefit

This reduces the number of kernel launches from:

num_simulations * num_steps

to:

num_steps

For the benchmark:

10 * 300 = 3000 launches

becomes:

300 launches

This reduces kernel-launch overhead and increases the amount of parallel work per kernel launch.

Result
Version	PASS/FAIL	Optimized mean time
Before Optimization 9	PASS	12.69 ms
After Optimization 9	PASS	6.47 ms