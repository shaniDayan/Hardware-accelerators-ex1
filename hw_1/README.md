# 02360509 - Advanced topics in hardware accelerators for deep learning

## HW 1 - Heat Diffusion on GPU - 20% of final grade

This assignment simulates **2D heat diffusion** on a GPU: each timestep applies a large **weighted stencil** over the temperature grid that includes a moving heat source. The weights and a **reference** CUDA implementation are provided. Your job is to make it run as **fast** as possible while staying **correct**.

The computation kernel itself is simple; the interesting part is **optimization**. You may use anything covered in the course and anything you find online, as long as you stick to **native CUDA** (kernels, memory types, streams, etc.). Do **not** use external libraries or tools that optimize or tune for you (e.g. autotuners, vendor BLAS, third-party GPU frameworks). You are expected to write the optimizations yourself.

**Grading (within this 20%):**


| Component                    | Weight      | Criteria                                                                                                                                                              |
| ---------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Correctness & speed**      | 80% of HW 1 | Output must match the reference (`PASS` in `benchmark`). For any points, your optimized kernel must also be **faster than the reference** on the course test machine. |
| **Optimization competition** | 20% of HW 1 | Ranked against other pairs on the course server - fastest correct submissions score highest.                                                                          |


### Building & Running

A `Makefile` is provided:

```bash
make          # build benchmark and plot
make clean    # remove binaries and frames/
```

`**benchmark**` - correctness check and timing (reference vs optimized):

```bash
./benchmark
```

Prints `**PASS**` or `**FAIL**` by comparing your output to the reference across multiple simulations, then mean / min / max / stddev timing for both implementations. If you see `**PASS**`, your result is numerically correct. For grading credit on the 80% portion, your optimized time must also beat the reference on the lambda server.

`**plot**` - writes BMP frames to `frames/`. Handy for debugging, and as a bonus you get to watch the heat spread and the source move around.

```bash
./plot reference
./plot optimized
```

### References

You may use ideas from lectures, CUDA samples, blogs, or papers. Do **not** copy code verbatim. If you adapt material from outside the course, cite it in your dry answer and include **links** to the sources.

### Environment

You are welcome to develop on any machine with a CUDA-compatible GPU or using Google Colab. However, your submission will be tested on the **lambda** server using an **NVIDIA GeForce RTX 2080 Ti**. Verify `**PASS`** and your speedup there before submitting.

For this assignment that matters especially: an optimization that helps on one GPU (e.g. your laptop or Colab) may not give the same speedup on another architecture, and can even hurt. Treat the RTX 2080 Ti on lambda as the machine that counts for grading and the competition.

### Dry Questions

Write your answer under the **A1.** label. A short but concrete write-up is enough (what you tried, what worked, what did not, and why you think so). We recommend updating this section continuously while you optimize.

---

**Q1.** Briefly describe your optimization work: what approaches did you try, what improved performance, what didn't, and your reasoning. If you used external references (outside course material), list them with links.

**A1.**
## Optimization 1: Avoid Reallocating and Recopying Weights per Simulation
The stencil weights are identical for all simulations, so there is no need to allocate and copy them for every simulation.
### Change
Move the weight allocation/copy outside the simulation loop.
Instead of:
```text
allocate weights
copy weights
run simulation
free weights
```
for every simulation, we allocate/copy once in init and free once in finalize.

## Optimization 2: Avoid Reallocating d_current and d_next per Simulation
All simulations use the same width and height, so the device buffers always have the same size.
### Change
Allocate d_current and d_next once in init, reuse them for all simulations, and free them once in finalize.
The simulation data is still copied fresh for every simulation; only the allocated memory is reused.

## Optimization 3: Shared Memory Tiling
Each thread computes one output cell, but nearby threads read many of the same input values from global memory.
### Change
Each block loads a tile of the grid into shared memory, including a halo of STENCIL_RADIUS cells around the tile.
The shared-memory tile size is:
```c
(blockDim.x + 2 * STENCIL_RADIUS) by
(blockDim.y + 2 * STENCIL_RADIUS)
```
After __syncthreads(), each thread reads stencil values from shared memory instead of repeatedly reading from global memory.

## Optimization 4: Tune Block Size for Shared Memory Tiling
The original block size was 32x32, which gives 1024 threads per block. With shared memory and halo loading, this may be too heavy.
We benchmarked several block sizes on lambda.

 Block size | PASS/FAIL | Optimized mean time |
|---|---|---:|
| 8x8 | PASS | 17.00 ms |
| 16x16 | PASS | 15.79 ms |
| 32x8 | PASS | 12.82 ms |
| 8x32 | PASS | 16.21 ms |
| 64x4 | PASS | 14.48 ms |
| 64x8 | PASS | 17.42 ms |
| 32x16 | PASS | 16.04 ms |

The best tested block size was 32x8.

## Optimization 5: Store Weights in Constant Memory
The stencil weights are small, read-only, and used by all threads.

### Change
Store the weights in CUDA constant memory:
```c
__constant__ float d_weights_const[WEIGHTS_COUNT];
```
Copy them once using:
```c
cudaMemcpyToSymbol(d_weights_const, weights, WEIGHTS_COUNT * sizeof(float));
```

## Optimization 6: Fast Path for Interior Blocks — Tested but Not Kept
Boundary checks are only needed for blocks whose halo goes outside the grid.
For interior blocks, the tile plus halo is fully inside the grid, so we tested loading without boundary checks.

| Grid size | Without Opt 6 | With Opt 6 | Better? |
|---|---:|---:|---|
| 128x128 | 12.82 ms | 12.84 ms | Without Opt 6 |
| 256x256 | 26.94 ms | 26.80 ms | With Opt 6 |

This slightly helped on a larger grid but did not improve the provided 128x128 benchmark, so we did not keep it.

## Optimization 7: Loop Unrolling and Pointer Restriction

STENCIL_RADIUS = 6, so the stencil loop always has:
13 x 13 = 169 iterations
The loop size is known at compile time.
### Chang
We added #pragma unroll to the stencil loops and marked kernel pointers with __restrict__ because current and next are different buffers.

| Version | PASS/FAIL | Optimized mean time |
|---|---|---:|
| Before Optimization 7 | PASS | 12.82 ms |
| After Optimization 7 | PASS | 12.69 ms |

This gave a small improvement, so we kept it.

## Optimization 8: Fused Multiply-Add (fmaf) — Tested but Not Kept
We tested replacing:
```c
source += value * weight;
``
with:
```c
source = fmaf(value, weight, source);
```
The code still passed correctness, but performance was slightly worse.
We did not keep this optimization.

## Optimization 9: Process All Simulations Together Using a 3D CUDA Grid

The simulations are independent, but the previous version still processed them one by one.
This caused:
num_simulations * num_steps
kernel launches.
For the benchmark:
10 * 300 = 3000 kernel launches

Use the z dimension of the CUDA grid for the simulation index:
int sim = blockIdx.z;
Each simulation gets its own slice in the large device buffers:
sim_offset = sim * width * height
The grid becomes:
dim3 grid(grid_x, grid_y, num_simulations);
Now each timestep launches one kernel for all simulations.

| Version | PASS/FAIL | Optimized mean time |
|---|---|---:|
| Before Optimization 9 | PASS | 12.69 ms |
| After Optimization 9 | PASS | 6.47 ms |

This was the largest improvement. Kernel launches were reduced from 3000 to 300.

## Optimization 10: Asynchronous Input and Output Copies Using CUDA Streams

After Optimization 9, all simulations are stored in one large device buffer. We tested asynchronous host-device copies using CUDA streams.
Create one stream per simulation and use cudaMemcpyAsync for input and output copies.
Input copies are synchronized before computation starts, and output copies are synchronized before returning.

| Version | PASS/FAIL | Optimized mean time |
|---|---|---:|
| Before Optimization 10 | PASS |  6.47 ms |
| After Optimization 10 | PASS | 6.41 ms |

This gave a small improvement, so we kept it.

## Optimization 11: CUDA Graphs for Timestep Kernel Launches
After Optimization 9, each timestep still required one kernel launch.
For the benchmark:
    num_steps = 300
this means 300 kernel launches per optimized run.
### Change
We tested CUDA Graphs to capture the repeated timestep kernel launches once and replay them with lower CPU-side launch overhead.
The graph contains the sequence of timestep kernels, including the alternating use of `d_buffer_a` and `d_buffer_b`.
We also made the graph rebuild only when the simulation parameters change, such as:

    width
    height
    num_simulations
    start_step
    num_steps
    shared memory size

| Version | PASS/FAIL | Optimized mean time |
|---|---|---:|
| Before Optimization 11 | PASS | 6.41 ms |
| After Optimization 11 | PASS | 6.25 ms |

CUDA Graphs gave a small additional improvement by reducing the CPU-side overhead of repeatedly launching timestep kernels.

We kept this optimization because it preserved correctness and improved runtime.

### References used

- Course CUDA introduction slides: blocks/threads/indexing, shared memory, `__syncthreads()`, constant memory, streams, and asynchronous copies.
- NVIDIA Developer Blog, "Using Shared Memory in CUDA C/C++": https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
- NVIDIA Developer Blog, "How to Optimize Data Transfers in CUDA C/C++": https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
- NVIDIA Developer Blog, "How to Overlap Data Transfers in CUDA C/C++": https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
- NVIDIA CUDA Graphs documentation: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- NVIDIA Developer Blog, "Getting Started with CUDA Graphs": https://developer.nvidia.com/blog/cuda-graphs/
---

### Submission

- Submission is in **pairs**. Submit a **zip file** named `<studentID1>_<studentID2>.zip`.
- The zip must contain a single folder with:
  - This `README.md`, with **A1** filled in
  - `**heat_cuda_optimized.cu`** only
- Do **not** include the `Makefile`, other source files, compiled binaries, `frames/`, or any other files.

