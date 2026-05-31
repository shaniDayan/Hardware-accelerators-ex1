NVCC = nvcc
NVCCFLAGS = -O2 -allow-unsupported-compiler -I.

UTILS = utils/utils.cpp
HOST = simulation_host.cpp
REF = heat_cuda_reference.cu
OPTIMIZED = heat_cuda_optimized.cu
CUDA_SRCS = $(REF) $(OPTIMIZED)

all: benchmark plot

benchmark: benchmark.cpp $(UTILS) $(HOST) $(CUDA_SRCS)
	$(NVCC) $(NVCCFLAGS) -o $@ benchmark.cpp $(UTILS) $(HOST) $(CUDA_SRCS)

plot: plot.cpp $(UTILS) $(HOST) $(CUDA_SRCS)
	$(NVCC) $(NVCCFLAGS) -o $@ plot.cpp $(UTILS) $(HOST) $(CUDA_SRCS)

clean:
	rm -f benchmark plot
	rm -rf frames

.PHONY: all clean
