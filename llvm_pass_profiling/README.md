# llvm_pass_profiling

![LLVM](https://img.shields.io/badge/LLVM-262D3A?style=flat-square&logo=llvm&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![C](https://img.shields.io/badge/C-A8B9CC?style=flat-square&logo=c&logoColor=black)
![CMake](https://img.shields.io/badge/CMake-064F8C?style=flat-square&logo=cmake&logoColor=white)

Can you tell whether a loop will be slow **before running it**? An out-of-tree LLVM analysis pass reads
the compiler's own intermediate representation, classifies every load and store by its memory stride,
and predicts vectorisability — then three independent instruments check whether the prediction holds.

| Level | Instrument | What it sees |
|---|---|---|
| Static, from IR | [`LoopStridePass`](pass/LoopStridePass.cpp) — custom | Stride of every access, via ScalarEvolution add-recurrences |
| Static, from the compiler | `clang -Rpass=loop-vectorize` | What the vectoriser decided, and why |
| Static, core model | `llvm-mca` | Cycles per iteration assuming everything is in L1 |
| Dynamic | wall clock + achieved GB/s | What actually happened |

## Result

3-point Jacobi stencil, `out[i] = 0.25·in[i-1] + 0.5·in[i] + 0.25·in[i+1]`, 2²⁴ doubles = 128 MiB per
array — far larger than any cache, so the numbers are DRAM bandwidth, not cache bandwidth.

| Variant | Pass verdict | Vectoriser | llvm-mca | Measured |
|---|---|---|---:|---:|
| **unit** | 3/3 unit-stride → contiguous | vectorised, VF=2 IC=4 | 11.3 cyc | **136.7 GB/s** |
| **strided** | 4/4 affine, 0/4 unit-stride | "not beneficial", interleaved ×4 | 89.0 cyc | **12.5 GB/s** |
| **gather** | contains gather/scatter | not vectorised | 14.0 cyc | **94.9 GB/s** |

All three compute the same checksum (`8389146.60468194`) — they do identical arithmetic on identical
data, so every difference is access pattern alone.

<sub>The GB/s column is `SWEEPS · n · 4 · 8 / t` for all three variants, i.e. it assumes 32 B/point.
The gather actually moves 40 B/point, so its real touched bandwidth is ≈119 GB/s, not 94.9. See
*Known issues* below.</sub>

![Bandwidth by variant](benchmarks/bandwidth_by_variant.png)

**The experiment's one trick: the gather uses an identity permutation.**

`idx[i] = i`, so at runtime the gather touches memory in *exactly* the same order as the unit-stride
version. The hardware prefetcher sees a sequential stream; the caches behave identically.

This README used to conclude from that: *the only thing that changed is what the compiler can prove,
so the 136.7 → 94.9 GB/s gap is pure lost vectorisation.* **That conclusion was wrong, and the
control experiment below is what showed it.**

`stencil_gather` also reads `idx[i]` — 8 bytes per point the unit-stride loop never touches, a 25%
traffic increase in a loop this page itself calls DRAM-bound. The two variants are not
traffic-matched, so the comparison cannot isolate codegen from bandwidth.
[`kernel/stencil_control.c`](kernel/stencil_control.c) holds one effect fixed at a time
([`benchmarks/control_m4.txt`](benchmarks/control_m4.txt)):

| Comparison | What it isolates | Cost |
|---|---|---:|
| `unit_novec / unit` | vectorisation, at matched 32 B/point | 1.38× |
| `unit_idx / unit` | the extra 8 B/point, at matched codegen | 1.48× |
| `unit_idx_novec / unit` | both together | 1.46× |
| `gather / unit` | the original comparison | 1.61× |
| `gather / unit_idx_novec` | what is left once both are accounted for | **1.10×** |

- The extra traffic on its own costs **as much as losing vectorisation** on its own
- Once the loop moves 40 B/point, disabling the vectoriser changes nothing measurable — 1.48× vs
  1.46× is run-to-run noise. The loop is already bandwidth-saturated
- Only **1.10×** of the gather's 1.61× is unexplained by traffic plus scalar codegen

→ **In a bandwidth-bound loop, bytes moved dominate.** The identity permutation controls for *cache
behaviour*, which is real and worth having — but it does not control for *traffic volume*, and that
is what the original conclusion assumed. To isolate compiler opacity the index load has to be on
both sides of the comparison.

**Strided is 11× slower, and that's memory, not the core.**

A 64-byte stride touches a fresh cache line per element and uses 8 of its 64 bytes. Note the instrument
disagreement: `llvm-mca` says only ~8× worse per block, because it models an ideal L1-resident core and
has no concept of DRAM. **The gap between the model and the measurement is where the damage is** — that
is what makes running both worthwhile.

Also note the vectoriser's exact wording: *legal* but "cost-model indicates that vectorization is not
beneficial". Interleaving still happened (4 scalar iterations in flight). Legality and profitability
are different decisions.

![Core model vs measured](benchmarks/core_model_vs_measured.png)

## How the pass works

`ScalarEvolution` expresses a loop-varying value as an add-recurrence `{base, +, step}<loop>`. If a
pointer has that form, it provably advances by `step` bytes per iteration.

| SCEV form | Classification | Consequence |
|---|---|---|
| `SCEVAddRecExpr` in this loop, constant step = element size | **unit stride** | Contiguous, cheap vector loads |
| `SCEVAddRecExpr`, constant step ≠ element size | **strided** | Needs interleaving; poor cache-line utilisation |
| `SCEVAddRecExpr`, symbolic step | affine, stride unknown | Vectoriser needs a runtime check |
| Loop-invariant | same address each iteration | — |
| Neither | **gather/scatter** | Address depends on loaded data; no closed form exists |

Two implementation points that matter:

- **Only the innermost owning loop counts.** `L->getBlocks()` on an outer loop includes its children's
  blocks, so without an `LI.getLoopFor(BB) != L` filter every access is reported once per nesting
  level. Stride with respect to the innermost loop is what sets cache behaviour.
- **The pass hooks `OptimizerEarly`.** It needs the IR after `mem2reg` and loop rotation — otherwise
  every local is stack traffic and there are no clean induction variables — but *before* the vectoriser
  and unroller, or the report describes transformed code rather than yours.

An incidental demonstration that this placement is right: the pass reports the strided loop **8 times**.
`STRIDE` is a compile-time constant, so clang fully unrolled the outer pass-loop during simplification.
The pass sees the real post-simplification IR, not the source's fiction.

## Build

Needs an LLVM installation with development headers. No LLVM rebuild required — this is a plugin.

```bash
./scripts/build_pass.sh        # pins LLVM via `llvm-config --cmakedir`
```

Three build details that are easy to get wrong:

| Detail | Why |
|---|---|
| `-fno-rtti` | LLVM is built without RTTI. A plugin compiled *with* it references `typeinfo` symbols LLVM never emitted → unresolved symbols at load |
| `project(... C CXX)` | LLVM's config probes libedit with a **C** `check_include_file`; a CXX-only project fails at configure time |
| macOS `-undefined dynamic_lookup` | `MODULE` libraries error on undefined symbols by default; deferring resolution lets the plugin bind against whichever host loads it |

The plugin header moved from `llvm/Passes/` to `llvm/Plugins/` in LLVM 22, so the source guards it with
`__has_include`. LLVM's C++ API has no stability guarantee across major versions — out-of-tree passes
chase that churn permanently.

## Run

```bash
./scripts/stride.sh      # the custom pass      -> benchmarks/stride_report.txt
./scripts/remarks.sh     # vectoriser remarks   -> benchmarks/vectorize_remarks.txt + opt-record YAML
./scripts/mca.sh         # static core model    -> benchmarks/mca_report.txt
./scripts/perf.sh        # hardware counters    (Linux only)
python3 scripts/plots.py # figures              -> benchmarks/*.png
```

`remarks.sh` also emits `-fsave-optimization-record` YAML, which `opt-viewer.py` renders as annotated
source. That is the form that scales to a large codebase — grepping remark spam does not.

## Known issues and corrections

Re-reading this project in **September 2026** against its own source turned up three problems with
the original write-up. They are recorded here rather than quietly edited out, because how a wrong
conclusion survived three instruments is the more useful part of the story.

| # | Found | Issue | Status |
|---|---|---|---|
| 1 | Sep 2026 | **"The gather gap is pure lost vectorisation" does not follow.** `stencil_gather` reads `idx[i]`, 8 B/point more than `stencil_unit`, so the two are not traffic-matched. The control shows traffic alone costs as much as vectorisation alone, and only 1.10× of the 1.61× gap is left once both are accounted for | **Corrected.** Control added as [`kernel/stencil_control.c`](kernel/stencil_control.c); conclusion rewritten above |
| 2 | Sep 2026 | **The GB/s column under-counts the gather.** [`kernel/stencil.c`](kernel/stencil.c) computes bandwidth as `n · 4 · sizeof(double)` for every variant, but the gather moves five 8-byte quantities per point, not four. Its 94.9 GB/s should be ≈119 GB/s | **Documented,** not patched — the committed numbers in [`benchmarks/`](benchmarks/) were produced by this formula, and changing it would desynchronise them |
| 3 | Sep 2026 | **"30% is the floor cost of compiler opacity" was a guess.** It was never measured against a random permutation, so the floor claim had no evidence behind it either way | **Withdrawn.** The random-permutation run is still not done — see below |

**Why it held up for so long:** all three instruments agreed, and they agreed *because they were all
looking at the same thing*. The pass says "gather", the vectoriser says "not vectorised", `llvm-mca`
says "more cycles" — three confirmations that the code generation changed, and none of them can see
memory traffic at all. `llvm-mca` in particular assumes L1 residency, so it structurally cannot
report a bandwidth effect. Agreement between instruments that share a blind spot is not
corroboration. The measurement that would have caught it — a variant that changes traffic without
changing codegen — is the one that was missing.

**Still open**

- **The random-permutation run.** It is the natural next variant: it would separate the cache penalty
  from the traffic and codegen effects now quantified, and would give the "real gathers cost X"
  number this project originally claimed without evidence
- **A hardware-counter cross-check.** `scripts/perf.sh` exists but is Linux-only and these runs were
  on an M-series Mac, so bytes-from-DRAM was never measured directly — the whole argument above rests
  on wall-clock ratios between variants
- **`benchmarks/*.png` predate the correction.** The bandwidth plot still shows the 94.9 GB/s figure
  from issue 2

## Limitations

| Limitation | Detail |
|---|---|
| **Innermost stride only** | A column walk over a row-major 2-D array reports as "strided" without identifying which outer loop would fix it. Loop interchange detection is out of scope |
| **Structural verdict, no cost model** | "Hard to vectorise" is a heuristic, not a theorem — AVX-512 and SVE have hardware gathers |
| **Symbolic strides detected but not bounded** | `A[i*n]` is flagged affine-with-unknown-step; the vectoriser does better with runtime checks |
| **`llvm-mca` assumes L1 residency** | Its numbers are only meaningful comparatively, and alongside a bandwidth measurement |
| **Single-threaded by design** | Adding OpenMP would mix bandwidth saturation into what is a code-generation story |
| **Measured on Apple M-series** | LLVM 22, `-O3 -mcpu=native`, NEON. VF=2 is 128-bit vectors; AVX-512 would choose differently, and has gathers |

## Layout

```
pass/LoopStridePass.cpp   the analysis pass — SCEV add-rec classification, two PassBuilder hooks
pass/CMakeLists.txt       config-mode LLVM, -fno-rtti, macOS dynamic_lookup
kernel/stencil.c          three variants: unit, strided, gather (identity permutation)
kernel/stencil_control.c  five variants that separate traffic from codegen (Known issues #1)
scripts/                  build_pass, stride, remarks, mca, perf, plots
benchmarks/               captured reports and figures
benchmarks/control_m4.txt output of stencil_control.c, Apple M4 / clang 22
```

The optimisation-record YAML and `perf.data` are regenerable and not committed.
