#!/usr/bin/env bash
# Run the LoopStridePass over the stencil kernels via clang -fpass-plugin.
# The pass hooks the optimizer-early extension point, so it sees the IR after
# simplification (mem2reg, loop rotate) but before the vectoriser runs.
set -euo pipefail
cd "$(dirname "$0")/.."

PLUGIN=$(ls pass/build/libLoopStridePass.* 2>/dev/null | head -1)
[ -n "${PLUGIN}" ] || { echo "run scripts/build_pass.sh first"; exit 1; }

clang -O2 -g -fpass-plugin="${PLUGIN}" -c -o /dev/null kernel/stencil.c \
    2> benchmarks/stride_report.txt
cat benchmarks/stride_report.txt
