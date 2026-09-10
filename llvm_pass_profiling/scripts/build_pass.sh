#!/usr/bin/env bash
# Build the LoopStridePass plugin against the LLVM found via llvm-config.
# On macOS with Homebrew LLVM: export PATH="$(brew --prefix llvm)/bin:$PATH"
set -euo pipefail
cd "$(dirname "$0")/.."

LLVM_DIR=$(llvm-config --cmakedir)
echo "Using LLVM $(llvm-config --version) from ${LLVM_DIR}"

cmake -S pass -B pass/build -DLLVM_DIR="${LLVM_DIR}" \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_BUILD_TYPE=Release
cmake --build pass/build

ls pass/build/libLoopStridePass.*
