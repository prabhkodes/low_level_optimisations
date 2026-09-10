# Low-Level Optimisations

![C](https://img.shields.io/badge/C-A8B9CC?style=flat-square&logo=c&logoColor=black)
![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-006DB8?style=flat-square&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LLVM](https://img.shields.io/badge/LLVM-262D3A?style=flat-square&logo=llvm&logoColor=white)

Performance-focused projects targeting cache behaviour, memory bandwidth, and GPU throughput. Each project includes benchmarks and profiling results.

## Projects

### blocked_matrix_multiplication
Cache-optimised GEMM in C using loop tiling. The naive triple-loop layout causes poor cache reuse on large matrices — blocking restructures the loop order so data stays in L1/L2 across iterations. OpenMP parallelises the outer loops.

Benchmarks cover block size tuning, thread scaling, strong/weak scaling, and a roofline comparison. All plots are in `benchmarks/`.

```bash
# Single-threaded
gcc -O3 -march=native -o matmul.x main.c -lm

# OpenMP
gcc -O3 -march=native -fopenmp -o matmul.x main.c -lm
OMP_NUM_THREADS=4 ./matmul.x
```

### fft
Cooley-Tukey FFT implemented from scratch in C (CPU reference) and progressively ported to CUDA across three versions. Benchmarks compare CPU, GPU naive, and GPU optimised on a P100/V100. Weak scaling results and plots are in `benchmarks/`.

| File | Description |
|---|---|
| `main.c` | CPU reference |
| `main_v1.cu` | Naive CUDA port |
| `main_v2.cu` | Shared memory optimisation |
| `main_v3.cu` | Further tuned |
| `validate.py` | Checks GPU output against NumPy FFT |

```bash
# CPU
gcc -O3 -fopenmp -o fft.x main.c -lm && ./fft.x <N>

# GPU
nvcc -O3 -arch=sm_70 -o fft_gpu.x main_v2.cu && ./fft_gpu.x <N>

# Cluster
sbatch slurm.sh
```

### llvm_pass_profiling
Out-of-tree LLVM analysis pass (`LoopStridePass`) that classifies every
load/store in a loop by its memory stride using ScalarEvolution — unit-stride,
strided, or gather/scatter — and predicts vectorisability before the
vectoriser runs. Validated against clang's own `-Rpass=loop-vectorize`
remarks, `llvm-mca` throughput models, and measured bandwidth on a 3-point
Jacobi stencil: the strided variant runs ~11x slower than unit-stride purely
from cache-line waste, and the gather variant loses ~30% purely from unproven
vectorisation. Pass plugin runs under `clang -fpass-plugin` or `opt`.

```bash
cd llvm_pass_profiling
scripts/build_pass.sh   # needs LLVM dev libs (brew install llvm / module load llvm)
scripts/stride.sh       # per-loop stride report
scripts/remarks.sh && scripts/mca.sh
```

### leonardo_booster
Architecture notes for the Leonardo Booster partition at Cineca. Documents the node topology (2 sockets, 8 NUMA nodes, 112 cores, 4x A100 per node) and explains the SLURM flags and process/thread pinning strategy used across the jobs in this repo.

Key considerations covered:
- NUMA locality and MPI rank placement to avoid cross-NUMA memory traffic
- OpenMP thread binding (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`) for stencil-heavy workloads
- GPU affinity — matching ranks to the socket their A100s attach to

The `arch/` folder has the node topology from `lstopo`.
