#!/usr/bin/env bash
# Ask clang's vectoriser to explain itself:
#   -Rpass=loop-vectorize          loops it vectorised (with VF/IC)
#   -Rpass-missed=loop-vectorize   loops it gave up on
#   -Rpass-analysis=loop-vectorize why it gave up
# Also dumps the full optimisation record as YAML for opt-viewer.
set -euo pipefail
cd "$(dirname "$0")/.."

clang -O3 -g -c -o /dev/null kernel/stencil.c \
    -Rpass=loop-vectorize \
    -Rpass-missed=loop-vectorize \
    -Rpass-analysis=loop-vectorize \
    2> benchmarks/vectorize_remarks.txt || true
cat benchmarks/vectorize_remarks.txt

# Machine-readable record of every transformation decision.
clang -O3 -g -c -o /dev/null kernel/stencil.c \
    -fsave-optimization-record \
    -foptimization-record-file=benchmarks/stencil.opt.yaml
echo "full record: benchmarks/stencil.opt.yaml"
