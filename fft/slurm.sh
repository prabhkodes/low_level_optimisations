#!/bin/bash
#SBATCH --job-name=fft_run
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=skyvolta
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --exclusive

# Load environment
module load cuda/11.8

# Define the problem size: 2^10
N=$((2**10))

# Compile 
nvcc fft.c -o fft.x -Xcompiler -std=c99 -arch=sm_70 -O3 -lm


# Only run if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running FFT with N=$N"
    srun ./fft.x $N
else
    echo "Compilation failed. Check the .err file."
    exit 1
fi