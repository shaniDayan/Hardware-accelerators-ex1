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

## Optimization Idea: Shared Memory Tiling

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
__syncthreads() ensures all values are loaded.
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

For the first shared-memory version, we will start with:

`dim3 block(16, 16);`

because it gives:

16 * 16 = 256 threads per block

and the shared-memory tile is:

(16 + 2*6) x (16 + 2*6)
= 28 x 28
= 784 floats

This is a safer starting point than 32 x 32.

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

cudaMemcpyToSymbol(d_weights_const, weights, WEIGHTS_COUNT * sizeof(float));

Then the kernel reads directly from d_weights_const.

Expected benefit

Constant memory is cached and is especially efficient when many threads in a warp read the same address.

In this stencil computation, all threads iterate over the same weight indices, so constant memory is a good fit.