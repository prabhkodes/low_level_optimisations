#!/usr/bin/env bash
# Hardware-counter profile of the stencil binary (Linux / cluster nodes).
# Confirms what the static analysis predicts: the strided and gather variants
# burn the same FLOPs but miss in cache far more often.
#
# On the Leonardo login/compute nodes: module load llvm, then run under
# `srun` on an exclusive node so the counters aren't polluted.
set -euo pipefail
cd "$(dirname "$0")/.."

clang -O3 -march=native -g -fno-omit-frame-pointer -o stencil.x kernel/stencil.c

perf stat -o benchmarks/perf_stat.txt \
    -e cycles,instructions,cache-references,cache-misses,LLC-load-misses \
    ./stencil.x
cat benchmarks/perf_stat.txt

# Per-function breakdown: which variant eats the cache misses.
perf record -o benchmarks/perf.data -e cache-misses ./stencil.x
perf report -i benchmarks/perf.data --stdio | head -30 \
    > benchmarks/perf_report.txt
cat benchmarks/perf_report.txt
