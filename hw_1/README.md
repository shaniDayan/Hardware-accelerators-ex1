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

---

### Submission

- Submission is in **pairs**. Submit a **zip file** named `<studentID1>_<studentID2>.zip`.
- The zip must contain a single folder with:
  - This `README.md`, with **A1** filled in
  - `**heat_cuda_optimized.cu`** only
- Do **not** include the `Makefile`, other source files, compiled binaries, `frames/`, or any other files.

