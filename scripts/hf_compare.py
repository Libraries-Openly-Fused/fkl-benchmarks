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
"""The horizontal-fusion experiment (e0-hf) measured again, three times, on a quiet machine, against the original run.

  python scripts/hf_compare.py --original results/rtx4090/e0-hf --quiet results/rtx4090/e0-load-audit/spot/Q_hf --out results/rtx4090/e0-load-audit/spot/hf_comparison.csv

For each of the three files of the experiment (plain launches with the CPU overhead, NO_CPU_OVERHEAD, NO_CPU_OVERHEAD with CUDA Graphs; the second is figure F3's first series, the third its second) and each batch
size: the iterated (one kernel per batch element) time, the fused time and the speedup of the original run, the median and the range of the three quiet runs, and the ratio original / median of the quiet runs.
The summary says how many points differ from the quiet median by more than the tolerance (default 2 %) and whether those are outside the range of the quiet runs themselves.
"""
import argparse
import csv
import glob
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc  # noqa: E402

FILES = ["benchmark_Horizontal_Fusion_GPU_NVIDIA.csv", "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_GPU_NVIDIA.csv", "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_GPU_NVIDIA.csv"]
COLS = [("iterated_ms", "Iterated Batch MeanTime"), ("fused_ms", "Fused Batch MeanTime"), ("speedup", "Mean Speedup")]


def load(path):
    _, blocks = pc.read_blocks(path)
    b = blocks[0]
    batch = pc.column(b, "Batch size")
    return {int(x): {k: pc.column(b, c)[i] for k, c in COLS} for i, x in enumerate(batch)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True)
    ap.add_argument("--quiet", required=True, help="directory with run1, run2, run3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=2.0)
    args = ap.parse_args()
    runs = sorted(glob.glob(os.path.join(args.quiet, "run*")))
    rows, lines = [], []
    for fname in FILES:
        orig = load(os.path.join(args.original, fname))
        quiet = [load(os.path.join(r, fname)) for r in runs]
        flagged = []
        devs = []
        for batch in sorted(orig):
            row = {"file": fname.replace("benchmark_Horizontal_Fusion_", "").replace("_GPU_NVIDIA.csv", ""), "batch": batch}
            for key, _ in COLS:
                q = [r[batch][key] for r in quiet]
                med = st.median(q)
                row[f"{key}_original"] = orig[batch][key]
                row[f"{key}_quiet_median"] = med
                row[f"{key}_quiet_min"], row[f"{key}_quiet_max"] = min(q), max(q)
                row[f"{key}_ratio_original_over_quiet"] = orig[batch][key] / med
            dev = 100 * (row["speedup_ratio_original_over_quiet"] - 1)
            devs.append(abs(dev))
            inside = row["speedup_quiet_min"] <= orig[batch]["speedup"] <= row["speedup_quiet_max"]
            row["speedup_original_inside_quiet_range"] = int(inside)
            rows.append(row)
            if abs(dev) > args.tolerance:
                flagged.append((batch, dev, inside))
        lines.append(f"{fname}: {len(orig)} points, {len(runs)} quiet runs; speedup original / quiet median: median |difference| {st.median(devs):.2f} %, 90th percentile {sorted(devs)[int(0.9 * (len(devs) - 1))]:.2f} %, largest {max(devs):.2f} %; "
                     f"{len(flagged)} points above {args.tolerance} % (of which outside the range of the quiet runs: {sum(1 for f in flagged if not f[2])})")
        for batch, dev, inside in sorted(flagged, key=lambda f: -abs(f[1]))[:12]:
            lines.append(f"    batch {batch}: {dev:+.1f} % ({'inside' if inside else 'outside'} the range of the quiet runs)")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))
    open(os.path.splitext(args.out)[0] + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
