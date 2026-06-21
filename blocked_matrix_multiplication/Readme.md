# Blocked Matrix Multiplication

Cache-optimised GEMM in C using loop tiling. The naive triple-loop layout causes poor cache reuse on large matrices — blocking restructures the loop order so data stays in L1/L2 across iterations. OpenMP parallelises the outer loops.

Benchmarks cover block size tuning, thread scaling, strong/weak scaling, and a roofline comparison. Plots are in `benchmarks/`.

## Build & Run

```bash
# Single-threaded
gcc -O3 -march=native -o matmul.x "main.c" -lm
./matmul.x

# With OpenMP
gcc -O3 -march=native -fopenmp -o matmul.x "main.c" -lm
OMP_NUM_THREADS=4 ./matmul.x
```

On a cluster:

```bash
module load gcc
gcc -O3 -march=native -fopenmp -o matmul.x "main.c" -lm
./matmul.x
```
