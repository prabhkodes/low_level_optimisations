#!/usr/bin/env bash
# Static throughput analysis of the three stencil loops with llvm-mca.
# Compiles each kernel to assembly and lets llvm-mca model how the loop body
# schedules on the host core: IPC, resource pressure, dependency stalls.
set -euo pipefail
cd "$(dirname "$0")/.."

ASM=$(mktemp -t stencil_asm)
trap 'rm -f "${ASM}" "${ASM}.fn"' EXIT
clang -O3 -mcpu=native -S -o "${ASM}" kernel/stencil.c

: > benchmarks/mca_report.txt
for fn in stencil_unit stencil_strided stencil_gather; do
    echo "=== ${fn} (-O3, native) ===" | tee -a benchmarks/mca_report.txt
    # Cut this function's asm (up to its `ret`) so mca models the hot loop
    # only, and drop .cfi directives so the fragment stands alone — mca
    # otherwise rejects the unfinished frame. Temp files, not pipes: awk's
    # early exit would SIGPIPE the producer and trip pipefail.
    awk -v f="_${fn}:" 'index($0,f)==1{p=1} p&&!/\.cfi_/{print} p&&/^[[:space:]]*ret/{exit}' \
        "${ASM}" > "${ASM}.fn"
    llvm-mca -mcpu=native --iterations=300 "${ASM}.fn" |
        sed -n '1,12p' | tee -a benchmarks/mca_report.txt
    echo | tee -a benchmarks/mca_report.txt
done
echo "full summaries: benchmarks/mca_report.txt"
