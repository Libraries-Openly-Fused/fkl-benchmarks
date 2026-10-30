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
"""Compact E1 table for the paper (one column, 3.4 in at 8 pt): one row per experiment point, FKL against the hand-written kernels.

  python scripts/e1_compact.py [--vf results/rtx4090/e1-vf-chain-v2] [--hf ...e1-hf-batch-v2] [--pp ...e1-preproc-v3] [--out results/rtx4090/e1_compact.csv]

Two references per experiment, named in the sources of handwritten-cuda/:
  ref   the like-for-like hand-written kernel (it carries what FKL carries): E1a hand_params (constants as kernel arguments), E1b hand_generic (a 24 B
        descriptor per crop), E1c hand_padded (parameter block padded to the size of FKL's read IOp)
  best  the best hand-written kernel: E1a hand_imm (constants known at compile time), E1b and E1c hand (the lean per-crop parameters)
Columns: exp, point (E1a operations, E1b crops, E1c batch), ms_fkl (mean GPU time, CUDA events, 100 launches; median of the three runs run1..run3 of the
experiment directory), vs_ref and vs_best (time of FKL divided by the time of the reference, per run, then the median of the three), vs_ref_lo and vs_ref_hi (the
smallest and the largest of the three ratios to the reference: the run-to-run spread; drop them if the table has no room), regs_fkl, regs_ref, sass_fkl, sass_ref (instructions of the kernel), occ_fkl, occ_ref (theoretical occupancy in percent).
Left out of this compact table (they are in the full tables, e1-*/table.csv): gpu_min_ms, the instruction histogram (FFMA, FMUL, FADD, LDG, STG, LDL, STL, constant-bank
operands, IMAD, ISETP, branches, barriers), stack and local bytes (all 0 here), cmem0 (parameter block) bytes, code and cubin bytes, block and grid, active
blocks per SM, the ncu occupancy and duration, and every version other than the two references.
"""
import argparse
import csv
import glob
import statistics

REFS = {"E1a": ("hand_params", "hand_imm"), "E1b": ("hand_generic", "hand"), "E1c": ("hand_padded", "hand")}


def run_timings(d):
    """{point: {version: [mean ms of run1, run2, run3]}} from the timing CSVs of the three runs of an experiment directory."""
    out = {}
    for path in sorted(glob.glob(f"{d}/run*/*.csv")):
        if "gpu-at-end" in path:
            continue
        rows = list(csv.DictReader(l for l in open(path) if not l.startswith("#")))
        key = "ops" if "ops" in rows[0] else "batch"
        for r in rows:
            for col, v in r.items():
                if col.endswith("_mean_ms"):
                    out.setdefault(int(r[key]), {}).setdefault(col[:-len("_mean_ms")], []).append(float(v))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vf", default="results/rtx4090/e1-vf-chain-v2")
    ap.add_argument("--hf", default="results/rtx4090/e1-hf-batch-v2")
    ap.add_argument("--pp", default="results/rtx4090/e1-preproc-v3")
    ap.add_argument("--out", default="results/rtx4090/e1_compact.csv")
    args = ap.parse_args()
    out = []
    for d in (args.vf, args.hf, args.pp):
        rows = list(csv.DictReader(open(f"{d}/table.csv")))
        exp = rows[0]["experiment"]
        ref_v, best_v = REFS[exp]
        by = {(int(r["point"]), r["version"]): r for r in rows}
        times = run_timings(d)
        for point in sorted({p for p, _ in by}):
            f, ref, best = by[(point, "fkl")], by[(point, ref_v)], by[(point, best_v)]
            t = times[point]
            n = len(t["fkl"])
            assert n == len(t[ref_v]) == len(t[best_v]) and n >= 3, (d, point, n)
            r_ref = [a / b for a, b in zip(t["fkl"], t[ref_v])]
            r_best = [a / b for a, b in zip(t["fkl"], t[best_v])]
            out.append({"exp": exp, "point": point, "ms_fkl": f"{statistics.median(t['fkl']):.4f}", "vs_ref": f"{statistics.median(r_ref):.3f}", "vs_best": f"{statistics.median(r_best):.3f}",
                        "vs_ref_lo": f"{min(r_ref):.3f}", "vs_ref_hi": f"{max(r_ref):.3f}",
                        "regs_fkl": f["registers"], "regs_ref": ref["registers"], "sass_fkl": f["sass_instructions"], "sass_ref": ref["sass_instructions"],
                        "occ_fkl": f"{float(f['theoretical_occupancy_pct']):.0f}", "occ_ref": f"{float(ref['theoretical_occupancy_pct']):.0f}"})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
