#!/usr/bin/env python3
# Copyright 2026 Oscar Amoros Huguet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Portability figure: the RTX 4090 and the RX 9060 on the same two experiments of the FKL-only harness, with the achieved memory-bandwidth fraction of the fused kernel.

  python scripts/plot_portability.py --what hf      --rtx4090 results/rtx4090 --rx9060 <results/rx9060/fkl-self> [--layout twin|panels] [--out figures/portability_hf_r2.pdf]
  python scripts/plot_portability.py --what numinst --rtx4090 results/rtx4090 --rx9060 <results/rx9060/fkl-self> [--out figures/portability_numinst_r2.pdf]   (the five-row table, CSV + markdown)

Horizontal fusion (what = hf): speedup of one horizontally fused kernel against one kernel per batch element, and the achieved bandwidth fraction of the fused kernel, against the batch size. Both series
are the 'NO_CPU_OVERHEAD' files (GPU time only, as figure F3) and the column 'Mean Speedup' (the mean over the iterations of the ratio of the two times); the fused time is 'Fused Batch MeanTime'.
Instructions per operation (what = numinst): the same quantities for the 100 points of M = 1, 6, ..., 496 instructions per operation, 500 instructions in total.

Achieved bandwidth fraction (definition fixed by the coordinator): bytes moved by the fused kernel (read plus written, from the sizes and element types that the harness allocates) divided by the measured mean time
of the fused kernel, divided by the vendor peak memory bandwidth of Table II: 1,008 GB/s for the RTX 4090, 288 GB/s for the RX 9060. Byte counts, from the harness sources:
  horizontal fusion (fkl-self/fusion/benchmark_horizontal_fusion.h, benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD): per crop 60 x 120 = 7,200 elements, read as uchar (1 byte) and written as float (4 bytes) by the
      fused kernel: 36,000 bytes per crop, 36,000 * B bytes for the batch B (the crops are the windows (i, i) of a 3840 x 2160 uchar image and overlap, the bytes are counted per crop all the same).
  instructions per operation (fkl-self/numinstructions/benchmark_numinstructions_x_operation.h): NUM_ELEMENTS = 3840 * 2160 * 8 = 66,355,200 floats read and 66,355,200 floats written by the 'All Ops' kernel:
      8 bytes per element, 530,841,600 bytes per launch, whatever M.
Caches: the small batches fit in the cache (72 MiB L2 of the RTX 4090, 32 MiB Infinity Cache of the RX 9060), so the fraction can exceed what DRAM alone allows, and it can exceed 100 %. The working set of
the horizontal-fusion kernel is the output tensor (28,800 * B bytes) plus the union of the input windows (computed exactly in bytes from the geometry, not in cache lines); the points where it exceeds the cache size are
drawn with hollow markers on the bandwidth curve. The working set of the 'All Ops' kernel of the instructions-per-operation experiment (input and output, 530,841,600 bytes) exceeds both caches at every point.
"""
import argparse
import csv
import os

import plot_common as pc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GPUS = [  # label, peak GB/s (Table II), cache bytes, cache name, csv subpaths
    dict(label="RTX 4090", peak=1008.0, cache=72 * 2**20, cache_name="72 MiB L2", key="rtx4090"),
    dict(label="RX 9060", peak=288.0, cache=32 * 2**20, cache_name="32 MiB Infinity Cache", key="rx9060"),
]
HF_NV = "e0-hf/benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_GPU_NVIDIA.csv"
HF_AMD = "horizontal_fusion/benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_GPU_AMD.csv"
NI_NV = "e0-numinstructions/testNumInstPerOp_GPU_NVIDIA.csv"
NI_AMD = "numinstructions_x_operation/testNumInstPerOp_GPU_AMD.csv"

CROP_W, CROP_H = 60, 120            # fkl-self/fusion/benchmark_horizontal_fusion.h: fk::Size cropSize(60, 120)
BYTES_PER_CROP = CROP_W * CROP_H * (1 + 4)          # uchar in, float out
OUT_BYTES_PER_CROP = CROP_W * CROP_H * 4
NI_ELEMENTS = 3840 * 2160 * 8       # benchmark_numinstructions_x_operation.h: NUM_ELEMENTS
NI_BYTES = NI_ELEMENTS * (4 + 4)    # float read + float written by the fused kernel
NI_ROWS = [1, 6, 101, 251, 496]


def union_input_bytes(batch):
    """Bytes of the union of the batch windows [i, i + 60) x [i, i + 120) of the uchar image (crop i at (i, i)): row by row, exact."""
    total = 0
    for y in range(batch - 1 + CROP_H):
        lo = max(0, y - CROP_H + 1)          # crops i with i <= y < i + CROP_H
        hi = min(batch - 1, y)
        if lo > hi:
            continue
        total += (hi + CROP_W) - lo           # union of the intervals [i, i + 60) for i in lo..hi (they overlap while hi - lo < 60, else the gaps are none: consecutive i differ by 1)
    return total


def working_set(batch):
    return batch * OUT_BYTES_PER_CROP + union_input_bytes(batch)


def column_of(path, name):
    _, blocks = pc.read_blocks(path)
    return pc.column(blocks[0], name)


def hf_data(path, gpu):
    batch = column_of(path, "Batch size")
    speedup = column_of(path, "Mean Speedup")
    fused_ms = column_of(path, "Fused Batch MeanTime")
    frac = [100.0 * (BYTES_PER_CROP * b) / (t * 1e-3) / (gpu["peak"] * 1e9) for b, t in zip(batch, fused_ms)]
    over = [working_set(int(b)) > gpu["cache"] for b in batch]
    return batch, speedup, fused_ms, frac, over


def numinst_data(path, gpu):
    m = column_of(path, "Number of instructions per Operation")
    speedup = column_of(path, "Mean Speedup")
    sep_ms = column_of(path, "Separated Ops MeanTime")
    all_ms = column_of(path, "All Ops MeanTime")
    frac = [100.0 * NI_BYTES / (t * 1e-3) / (gpu["peak"] * 1e9) for t in all_ms]
    return m, speedup, sep_ms, all_ms, frac


def draw_hf(args, paths):
    pc.apply_style()
    data = [hf_data(paths[i], g) for i, g in enumerate(GPUS)]
    panels = args.layout == "panels"
    if panels:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.1, 1.95), constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(3.4, 2.1), constrained_layout=True)
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.grid(False)
    notes = []
    for i, (g, (batch, speedup, fused_ms, frac, over)) in enumerate(zip(GPUS, data)):
        c, mk = pc.SERIES_COLORS[i], pc.SERIES_MARKERS[i]
        ax.plot(batch, speedup, color=c, marker=mk, markersize=2.6, markevery=8, linewidth=1.0, clip_on=False)
        ax2.plot(batch, frac, color=c, linestyle=(0, (3, 1.5)), linewidth=0.9)
        hollow = [(b, f) for b, f, o in zip(batch, frac, over) if o]
        if hollow:
            ax2.plot([b for b, _ in hollow], [f for _, f in hollow], linestyle="none", marker=mk, markersize=2.6, markerfacecolor="white", markeredgecolor=c, markeredgewidth=0.7)
        j = speedup.index(max(speedup))
        first_over = next((int(b) for b, o in zip(batch, over) if o), None)
        notes.append(f"{g['label']}: {len(batch)} points, batch {batch[0]:.0f} to {batch[-1]:.0f}; speedup maximum {max(speedup):.2f}x at batch {batch[j]:.0f}, at the last batch {speedup[-1]:.2f}x; bandwidth fraction of the fused kernel "
                     f"{min(frac):.1f} % to {max(frac):.1f} % of {g['peak']:.0f} GB/s (at the last batch {frac[-1]:.1f} %); working set exceeds the {g['cache_name']} from batch "
                     f"{first_over if first_over else 'none (not within the sweep)'} on (hollow markers); largest working set {working_set(int(batch[-1])) / 2**20:.1f} MiB")
    ax.set_xlim(0, 1200)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 1201, 200))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup vs one kernel\nper batch element")
    ax2.set_ylim(bottom=0)
    if panels:
        ax2.set_xlim(0, 1200)
        ax2.set_xticks(range(0, 1201, 200))
        ax2.set_xlabel("Batch size")
        ax2.set_ylabel("Achieved bandwidth,\n% of vendor peak")
    else:
        ax2.set_ylabel("Achieved bandwidth, % of peak (dashed)")
    handles = [Line2D([0], [0], color=pc.SERIES_COLORS[i], marker=pc.SERIES_MARKERS[i], markersize=2.6, linewidth=1.0, label=g["label"]) for i, g in enumerate(GPUS)]
    if not panels:
        handles += [Line2D([0], [0], color=pc.INK, linewidth=1.0, label="speedup (left)"), Line2D([0], [0], color=pc.INK, linewidth=0.9, linestyle=(0, (3, 1.5)), label="bandwidth (right)")]
    ax.legend(handles=handles, loc="upper left" if not panels else "lower right", ncol=1)
    return fig, notes


def table_numinst(args, paths):
    rows = []
    data = [numinst_data(paths[i], g) for i, g in enumerate(GPUS)]
    for mm in NI_ROWS:
        row = {"M": mm}
        for g, (m, speedup, sep_ms, all_ms, frac) in zip(GPUS, data):
            k = m.index(float(mm))
            tag = g["label"].replace(" ", "")
            row[f"{tag}_separated_ms"] = f"{sep_ms[k]:.4g}"
            row[f"{tag}_fused_ms"] = f"{all_ms[k]:.4g}"
            row[f"{tag}_speedup"] = f"{speedup[k]:.4g}"
            row[f"{tag}_bandwidth_pct"] = f"{frac[k]:.1f}"
        rows.append(row)
    return rows, data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["hf", "numinst"], required=True)
    ap.add_argument("--rtx4090", default=os.path.join(pc.REPO, "results", "rtx4090"))
    ap.add_argument("--rx9060", required=True, help="results/rx9060/fkl-self (or a checkout of the branch of the AMD session)")
    ap.add_argument("--layout", choices=["twin", "panels"], default="twin")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.what == "hf":
        paths = [os.path.join(args.rtx4090, HF_NV), os.path.join(args.rx9060, HF_AMD)]
        fig, notes = draw_hf(args, paths)
        out = args.out or os.path.join(pc.FIGURES, f"portability_hf_{args.layout}_r2.pdf")
        pc.save(fig, out, paths, "plot_portability.py", notes)
    else:
        paths = [os.path.join(args.rtx4090, NI_NV), os.path.join(args.rx9060, NI_AMD)]
        rows, data = table_numinst(args, paths)
        out = args.out or os.path.join(pc.FIGURES, "portability_numinst_r2")
        out = os.path.splitext(out)[0]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        head = "| M | RTX 4090 separated (ms) | RTX 4090 fused (ms) | RTX 4090 speedup | RTX 4090 bandwidth (%) | RX 9060 separated (ms) | RX 9060 fused (ms) | RX 9060 speedup | RX 9060 bandwidth (%) |"
        lines = [head, "|" + "---|" * 9]
        for r in rows:
            lines.append(f"| {r['M']} | " + " | ".join(r[k] for k in r if k != "M") + " |")
        open(out + ".md", "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
