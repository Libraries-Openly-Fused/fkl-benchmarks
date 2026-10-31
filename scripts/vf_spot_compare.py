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
"""Some points of a long VF run measured again on a quiet machine, compared with the original run, point by point.

  python scripts/vf_spot_compare.py --original results/rtx4090/e0-vf-mul-add1 --quiet results/rtx4090/e0-load-audit/spot/Q_A/mul_add1 --name MulAdd1 --out results/rtx4090/e0-load-audit/spot/mul_add1_comparison.csv

--name is the part of the CSV names that identifies the chain (MulAdd1, Mul1, MulAdd): benchmark_vertical_fusion_loop<name>.csv (plain launches) and ..._CUDAGraphs.csv. For every point of the quiet run and each of the two
passes: the OpenCV time, the cvGS (FKL) time and the speedup of both runs and the ratio quiet / original of each. The summary gives the median and the largest absolute difference of the speedup and the points above
the tolerance.
"""
import argparse
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc  # noqa: E402


def load(path):
    _, blocks = pc.read_blocks(path)
    b = blocks[0]
    ops = pc.column(b, b[0][0])
    return {int(o): dict(ocv=pc.column(b, "OpenCV MeanTime")[i], cvgs=pc.column(b, "cvGS MeanTime")[i], speedup=pc.column(b, "Mean Speedup")[i]) for i, o in enumerate(ops)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True)
    ap.add_argument("--quiet", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=2.0)
    args = ap.parse_args()
    rows, lines = [], []
    for label, fname in (("plain", f"benchmark_vertical_fusion_loop{args.name}.csv"), ("graphs", f"benchmark_vertical_fusion_loop{args.name}_CUDAGraphs.csv")):
        orig, quiet = load(os.path.join(args.original, fname)), load(os.path.join(args.quiet, fname))
        devs = []
        for ops in sorted(quiet):
            if ops not in orig:
                continue
            o, q = orig[ops], quiet[ops]
            r = dict(pass_=label, ops=ops, opencv_original=o["ocv"], opencv_quiet=q["ocv"], cvgs_original=o["cvgs"], cvgs_quiet=q["cvgs"], speedup_original=o["speedup"], speedup_quiet=q["speedup"],
                     opencv_ratio=q["ocv"] / o["ocv"], cvgs_ratio=q["cvgs"] / o["cvgs"], speedup_ratio=q["speedup"] / o["speedup"])
            rows.append(r)
            devs.append((100 * (r["speedup_ratio"] - 1), ops))
        ab = sorted(abs(d) for d, _ in devs)
        over = [(d, ops) for d, ops in devs if abs(d) > args.tolerance]
        lines.append(f"{args.name} {label}: {len(devs)} points; speedup quiet / original: median |difference| {st.median(ab):.2f} %, largest {ab[-1]:.2f} %; above {args.tolerance} %: "
                     + (", ".join(f"{ops} ({d:+.1f} %)" for d, ops in over) if over else "none"))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))
    open(os.path.splitext(args.out)[0] + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
