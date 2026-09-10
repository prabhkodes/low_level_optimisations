#!/usr/bin/env python3
"""Generate result plots from the captured benchmark reports.

Reads benchmarks/runtime_m4.txt (measured) and benchmarks/mca_report.txt
(llvm-mca static model) so the figures always reflect the last real run.
Writes three PNGs into benchmarks/.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"

VARIANTS = ["unit", "strided", "gather"]
COLORS = {"unit": "#2a9d34", "strided": "#d62728", "gather": "#e69f00"}

# Compiler-side classification for each variant (from stride_report.txt and
# vectorize_remarks.txt — textual, so summarised here rather than parsed).
PASS_VERDICT = {
    "unit": "LoopStridePass: unit stride (8 B/iter)",
    "strided": "LoopStridePass: strided (64 B/iter)",
    "gather": "LoopStridePass: gather/scatter (non-affine)",
}
VECT_STATUS = {
    "unit": "vectorised (VF=2, IC=4)",
    "strided": "not vectorised (cost model)",
    "gather": "not vectorised (can't prove affine)",
}


def read_runtime():
    """Parse the table printed by stencil.x: variant, time, GB/s, checksum."""
    data = {}
    for line in (BENCH / "runtime_m4.txt").read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 4 and parts[0] in VARIANTS:
            data[parts[0]] = {"time": float(parts[1]), "gbs": float(parts[2])}
    assert set(data) == set(VARIANTS), f"missing variants in runtime_m4.txt: {data}"
    return data


def read_mca():
    """Parse IPC and Block RThroughput per variant from mca_report.txt."""
    text = (BENCH / "mca_report.txt").read_text()
    data = {}
    for m in re.finditer(
        r"=== stencil_(\w+).*?IPC:\s+([\d.]+).*?Block RThroughput:\s+([\d.]+)",
        text, re.S):
        data[m.group(1)] = {"ipc": float(m.group(2)), "rthr": float(m.group(3))}
    assert set(data) == set(VARIANTS), f"missing variants in mca_report.txt: {data}"
    return data


def bar_annotate(ax, bars, fmt, dy=0.01):
    top = ax.get_ylim()[1]
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * dy,
                fmt(b.get_height()), ha="center", va="bottom", fontsize=11,
                fontweight="bold")


def fig_bandwidth(rt):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    gbs = [rt[v]["gbs"] for v in VARIANTS]
    bars = ax.bar(VARIANTS, gbs, color=[COLORS[v] for v in VARIANTS],
                  width=0.55, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, max(gbs) * 1.22)
    bar_annotate(ax, bars, lambda h: f"{h:.1f}")
    # Per-variant compiler classification, boxed beside the bars where
    # there's guaranteed whitespace (in-bar text overflows the bar width).
    lines = [f"{v}:  {PASS_VERDICT[v]};  {VECT_STATUS[v]}" for v in VARIANTS]
    ax.text(0.98, 0.965, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="gray", alpha=0.9))
    ax.set_ylabel("effective memory bandwidth (GB/s)\n= 4 accesses x 8 B x N x sweeps / time")
    ax.set_xlabel("stencil variant (same arithmetic, different access pattern)")
    ax.set_title("3-point Jacobi stencil: measured bandwidth by memory access pattern\n"
                 r"N = $2^{24}$ doubles (128 MiB/array, DRAM-resident), 50 sweeps,"
                 "\nApple M4, clang -O3 -mcpu=native (LLVM 22)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(BENCH / "bandwidth_by_variant.png", dpi=150)
    print("wrote", BENCH / "bandwidth_by_variant.png")


def fig_slowdown(rt):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    base = rt["unit"]["time"]
    slow = [rt[v]["time"] / base for v in VARIANTS]
    bars = ax.bar(VARIANTS, slow, color=[COLORS[v] for v in VARIANTS],
                  width=0.55, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, max(slow) * 1.25)
    bar_annotate(ax, bars, lambda h: f"{h:.2f}x")
    causes = {
        "unit": "baseline:\ncontiguous + vectorised",
        "strided": "cause: cache-line waste\n(8 B used of every 64 B line)\n+ no vectorisation",
        "gather": "cause: lost vectorisation ONLY\n(identity permutation:\nruntime access order\nidentical to unit)",
    }
    for b, v in zip(bars, VARIANTS):
        ax.annotate(causes[v],
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 28), textcoords="offset points",
                    ha="center", fontsize=9,
                    arrowprops=dict(arrowstyle="-", lw=0.8))
    ax.set_ylabel("slowdown vs unit-stride variant\n(wall time ratio, same FLOPs and result checksum)")
    ax.set_xlabel("stencil variant")
    ax.set_title("Where the time goes: slowdown relative to the contiguous loop\n"
                 "strided pays the memory hierarchy; gather pays only the compiler's blindness")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(BENCH / "slowdown_vs_unit.png", dpi=150)
    print("wrote", BENCH / "slowdown_vs_unit.png")


def fig_core_vs_memory(rt, mca):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 5.5))
    ipc = [mca[v]["ipc"] for v in VARIANTS]
    gbs = [rt[v]["gbs"] for v in VARIANTS]

    bars = axl.bar(VARIANTS, ipc, color=[COLORS[v] for v in VARIANTS],
                   width=0.55, edgecolor="black", linewidth=0.6)
    axl.set_ylim(0, 6.4)
    bar_annotate(axl, bars, lambda h: f"{h:.2f}")
    axl.axhline(6.0, ls="--", lw=1, color="gray")
    axl.text(0.02, 6.05, "dispatch width = 6", fontsize=8.5, color="gray")
    axl.set_ylabel("IPC (instructions per cycle)")
    axl.set_title("llvm-mca static model\n(assumes all loads hit L1, no DRAM)")
    axl.set_xlabel("variant")

    bars = axr.bar(VARIANTS, gbs, color=[COLORS[v] for v in VARIANTS],
                   width=0.55, edgecolor="black", linewidth=0.6)
    axr.set_ylim(0, max(gbs) * 1.2)
    bar_annotate(axr, bars, lambda h: f"{h:.0f}")
    axr.set_ylabel("measured bandwidth (GB/s)")
    axr.set_title("measured on hardware\n(128 MiB arrays, DRAM-resident)")
    axr.set_xlabel("variant")

    for ax in (axl, axr):
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Core model vs reality: mca sees healthy cores for every variant "
                 "-> the 11x strided collapse happens in the memory hierarchy, not the core",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(BENCH / "core_model_vs_measured.png", dpi=150)
    print("wrote", BENCH / "core_model_vs_measured.png")


if __name__ == "__main__":
    rt = read_runtime()
    mca = read_mca()
    fig_bandwidth(rt)
    fig_slowdown(rt)
    fig_core_vs_memory(rt, mca)
