# low_level_optimisations

![LLVM](https://img.shields.io/badge/LLVM-262D3A?style=flat-square&logo=llvm&logoColor=white)
![C](https://img.shields.io/badge/C-A8B9CC?style=flat-square&logo=c&logoColor=black)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-006DB8?style=flat-square&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-064F8C?style=flat-square&logo=cmake&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)

Working out why code is slow, at four levels — from what the compiler can prove about a loop before it
runs, down to what the memory system actually does.

| | Project | Question |
|---|---|---|
| **1** | **[`llvm_pass_profiling/`](llvm_pass_profiling/)** | **Can you predict a loop's performance from the compiler's IR, before running it?** |
| 2 | [`blocked_matrix_multiplication/`](blocked_matrix_multiplication/) | Where's the ceiling, and is my kernel at it? |
| 3 | [`fft/`](fft/) | Does any of this survive the move to a GPU? |
| 4 | [`leonardo_booster/`](leonardo_booster/) | What hardware is underneath it all? |

---

## 1. Predicting performance from the IR

**An out-of-tree LLVM analysis pass that reads a loop's intermediate representation, classifies every
load and store by memory stride, and predicts whether it will vectorise — before anything runs.**

`LoopStridePass` uses `ScalarEvolution` add-recurrences: a pointer expressible as `{base, +, step}` is
provably a memory walk advancing `step` bytes per iteration. Three outcomes — unit-stride, strided, or
gather/scatter — and the verdict is checked against three independent instruments.

3-point Jacobi stencil, 2²⁴ doubles = 128 MiB per array, so the numbers are DRAM bandwidth:

| Variant | Pass verdict | Vectoriser | llvm-mca | Measured |
|---|---|---|---:|---:|
| **unit** | 3/3 unit-stride → contiguous | vectorised, VF=2 IC=4 | 11.3 cyc | **136.7 GB/s** |
| **strided** | 4/4 affine, 0/4 unit-stride | "not beneficial" | 89.0 cyc | **12.5 GB/s** |
| **gather** | contains gather/scatter | not vectorised | 14.0 cyc | **94.9 GB/s** |

All three produce an identical checksum — same arithmetic, same data, so every difference is access
pattern alone.

**The gather uses an identity permutation.** `idx[i] = i`, so at runtime it touches memory in exactly
the same order as the unit-stride version — same prefetcher behaviour, same cache behaviour. The only
variable is what the compiler can *prove*.

→ **The 136.7 → 94.9 GB/s gap is pure lost vectorisation, with cache effects held constant.** That
separation is not obtainable from profiling alone, and 30% is the *floor* cost of compiler opacity —
a random permutation would pay this plus the cache penalty.

→ **`llvm-mca` and the stopwatch disagree on the strided case, and the disagreement is the finding.**
The model says ~8× worse per block; reality says 11× overall. mca assumes an ideal L1-resident core and
has no concept of DRAM, so the gap between prediction and measurement is precisely where the damage
lives.

[Full write-up, build notes and limitations →](llvm_pass_profiling/)

---

## 2. Finding the ceiling — cache-blocked GEMM

Hand-tiled matrix multiply in C with OpenMP, measured against a roofline.

| | |
|---|---:|
| Compute peak (socket) | 307.2 GFLOP/s |
| Bandwidth peak | 59 GB/s |
| Ridge point | ≈ 5.2 FLOP/byte |
| Measured arithmetic intensity | **0.52 – 0.54 FLOP/byte** |
| Best achieved | **33.6 GFLOP/s** at 16 threads, N = 4096 |

Every configuration lands two orders of magnitude left of the ridge — bandwidth-limited, where the
ceiling is `AI × bandwidth ≈ 31 GFLOP/s`.

→ **At 33.6 GFLOP/s the tiled kernel is already at its roof.** Nothing more is available without
raising arithmetic intensity, which is what a tuned BLAS does with register blocking and packing.

Also measured: a partial-sum accumulator cost **6× in the parallel version** (2.6 vs 15.8 GFLOP/s).
Block-size sweeps, thread scaling and strong/weak scaling plots are in
[`blocked_matrix_multiplication/benchmarks/`](blocked_matrix_multiplication/benchmarks/).

---

## 3. Onto the GPU — Cooley–Tukey FFT

CPU reference in C, progressively ported to CUDA across three versions, plus OpenACC and cuFFT.

| Implementation | Time at N = 2²³ | vs best CUDA |
|---|---:|---:|
| cuFFT | ~2 ms | 4.5× faster |
| CUDA, shared memory | ~9 ms | baseline |
| CUDA, global memory | ~25 ms | 2.8× slower |
| OpenACC | ~45 ms | 5× slower |

→ **Shared memory is worth 4.6×, and directives won't do it for you.** Same lesson as §1 in a different
register: the compiler needs to be *told* about the memory hierarchy.

The full six-way comparison, including the P100-vs-V100 study and a caveat about mixed timing methods,
lives in [`fft-gpu-programming-models`](https://github.com/prabhkodes/fft-gpu-programming-models).

---

## 4. The hardware underneath

[`leonardo_booster/`](leonardo_booster/) — node topology for the Leonardo Booster partition at CINECA:
2 sockets, 8 NUMA nodes, 112 cores, 4× A100 per node, with `lstopo` output in `arch/`.

Covers the placement decisions every job in this repo depends on:

- NUMA locality and MPI rank placement, to avoid cross-socket memory traffic
- OpenMP binding — `OMP_PROC_BIND=close`, `OMP_PLACES=cores` — for stencil-heavy work
- GPU affinity, matching ranks to the socket their A100s attach to

---

## Quick start

```bash
# 1 — LLVM pass (needs LLVM dev headers; no LLVM rebuild required)
cd llvm_pass_profiling
scripts/build_pass.sh
scripts/stride.sh                 # per-loop stride report
scripts/remarks.sh && scripts/mca.sh

# 2 — blocked GEMM
gcc -O3 -march=native -fopenmp -o matmul.x blocked_matrix_multiplication/main.c -lm
OMP_NUM_THREADS=16 ./matmul.x

# 3 — FFT
gcc -O3 -fopenmp -o fft.x fft/main.c -lm && ./fft.x 1048576
nvcc -O3 -arch=sm_70 -o fft_gpu.x fft/main_v2.cu && ./fft_gpu.x 1048576
```

## Where this came from

| | |
|---|---|
| Course | Master in High Performance Computing, ICTP / SISSA Trieste, 2025–26 |
| Cluster | Leonardo, CINECA — except the LLVM study, measured on Apple M-series with LLVM 22 |
| Related | [`fft-gpu-programming-models`](https://github.com/prabhkodes/fft-gpu-programming-models) · [`matrix-multiplication-parallel`](https://github.com/prabhkodes/matrix-multiplication-parallel) · [`gpu-kernel-profiling`](https://github.com/prabhkodes/gpu-kernel-profiling) |
