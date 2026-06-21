# FFT — CPU and GPU Implementations

Cooley-Tukey FFT implemented from scratch in C (CPU) and progressively ported to CUDA across three versions. Benchmarks compare CPU baseline, GPU naive, and GPU optimised (shared memory) on a V100. Plots and weak scaling results are in `benchmarks/`.

- `main.c` — CPU reference implementation
- `main_v1.cu` — naive CUDA port
- `main_v2.cu` — shared memory optimisation
- `main_v3.cu` — further tuned version
- `validate.py` — checks GPU output against NumPy FFT

## Build & Run

**CPU:**

```bash
gcc -O3 -fopenmp -o fft.x main.c -lm
./fft.x <N>   # N must be a power of 2, e.g. 1024
```

**GPU (local):**

```bash
nvcc -O3 -arch=sm_70 -o fft_gpu.x main_v2.cu -lm
./fft_gpu.x <N>
```

**GPU (cluster):**

```bash
module load cuda/11.8
sbatch slurm.sh
```

**Validate:**

```bash
python3 validate.py
```
