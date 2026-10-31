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
"""The corrected series of a long VF run: the irregular points of the plain pass are replaced by their quiet re-measurement; every other point stays as measured.

  python scripts/vf_replace_points.py --original results/rtx4090/e0-vf-mul-add --quiet results/rtx4090/e0-load-audit/spot/Q_A/mul_add \
      --irregular results/rtx4090/e0-load-audit/irregular_points.csv --run "e0-vf-mul-add" --out results/rtx4090/e0-vf-mul-add-corrected [--tolerance 2]

Rule (agreed with the coordinator): the quiet value replaces the original only for a point that the smoothness test (scripts/audit_smoothness.py) flagged irregular and that was re-measured; for the other
re-measured points the quiet value is a check, and the original stays when it agrees within the tolerance (2 %). The output directory has the same file names as the original (so that scripts/plot_vf.py
reads it with --mul-add), the plain-pass CSV with four more columns (replaced, and the original OpenCV time, cvGS time and speedup), the CUDA Graphs CSV unchanged, corrections.csv (the replaced points),
checks.csv (every re-measured point, both passes: the ratio quiet / original of the OpenCV time, the cvGS time and the speedup) and CORRECTED.txt.
"""
import argparse
import csv
import os
import shutil
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc  # noqa: E402

PLAIN = "benchmark_vertical_fusion_loopMulAdd.csv"
GRAPHS = "benchmark_vertical_fusion_loopMulAdd_CUDAGraphs.csv"


def fmt(x):
    return str(int(x)) if isinstance(x, float) and x.is_integer() else repr(x)


def rows_by_ops(path):
    meta, blocks = pc.read_blocks(path)
    header, rows = blocks[0]
    return meta, header, {int(r[0]): r for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True)
    ap.add_argument("--quiet", required=True)
    ap.add_argument("--irregular", required=True)
    ap.add_argument("--run", default="e0-vf-mul-add")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=2.0)
    args = ap.parse_args()
    irregular = {int(r["ops"]) for r in csv.DictReader(open(args.irregular)) if r["run"] == args.run and r["pass_"] == "plain" and int(r["ops"]) > 2}
    os.makedirs(args.out, exist_ok=True)
    checks, corrections = [], []
    new_rows = {}
    for fname, label in ((PLAIN, "plain"), (GRAPHS, "graphs")):
        meta_o, head_o, orig = rows_by_ops(os.path.join(args.original, fname))
        meta_q, head_q, quiet = rows_by_ops(os.path.join(args.quiet, fname))
        io = {name: head_o.index(name) for name in head_o}
        for ops, q in sorted(quiet.items()):
            if ops not in orig:
                continue
            o = orig[ops]
            ratio = lambda col: q[head_q.index(col)] / o[io[col]]
            row = dict(pass_=label, ops=ops, opencv_ratio=ratio("OpenCV MeanTime"), cvgs_ratio=ratio("cvGS MeanTime"), speedup_ratio=ratio("Mean Speedup"),
                       opencv_orig=o[io["OpenCV MeanTime"]], opencv_quiet=q[head_q.index("OpenCV MeanTime")], speedup_orig=o[io["Mean Speedup"]], speedup_quiet=q[head_q.index("Mean Speedup")],
                       irregular=int(label == "plain" and ops in irregular))
            row["agree_within_tolerance"] = int(all(abs(100 * (row[k] - 1)) <= args.tolerance for k in ("opencv_ratio", "cvgs_ratio", "speedup_ratio")))
            checks.append(row)
        if label == "plain":
            repl = {ops for ops in irregular if ops in quiet}
            new_rows = {ops: (quiet[ops] if ops in repl else orig[ops], int(ops in repl), orig[ops]) for ops in orig}
            for ops in sorted(repl):
                corrections.append(next(c for c in checks if c["pass_"] == "plain" and c["ops"] == ops))
            # the corrected plain CSV: the original header and meta lines, the rows in order, four more columns
            with open(os.path.join(args.original, fname), newline="") as f:
                text = f.read().splitlines()
            out_lines = []
            hdr_done = False
            for line in text:
                if line.startswith("#"):
                    out_lines.append(line)
                    continue
                if not hdr_done:
                    out_lines.append(line + ", Replaced by the quiet re-measurement, Original OpenCV MeanTime, Original cvGS MeanTime, Original Mean Speedup")
                    hdr_done = True
                    continue
                cells = [c.strip() for c in line.split(",")]
                ops = int(float(cells[0]))
                new, replaced, old = new_rows[ops]
                cur = ", ".join(fmt(x) for x in new)
                out_lines.append(f"{cur}, {replaced}, {fmt(old[io['OpenCV MeanTime']])}, {fmt(old[io['cvGS MeanTime']])}, {fmt(old[io['Mean Speedup']])}")
            out_lines.insert(0, f"# CORRECTED SERIES: {len(repl)} points of the plain pass (irregular in the original run, re-measured on a quiet machine) are the quiet values; column 'Replaced' marks them; see CORRECTED.txt")
            open(os.path.join(args.out, fname), "w").write("\n".join(out_lines) + "\n")
        else:
            shutil.copy(os.path.join(args.original, fname), os.path.join(args.out, fname))
    for fname in os.listdir(args.original):
        if fname not in (PLAIN, GRAPHS) and os.path.isfile(os.path.join(args.original, fname)):
            shutil.copy(os.path.join(args.original, fname), os.path.join(args.out, fname))
    with open(os.path.join(args.out, "checks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)
    with open(os.path.join(args.out, "corrections.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corrections[0]) if corrections else ["ops"])
        w.writeheader()
        w.writerows(corrections)
    plain = [c for c in checks if c["pass_"] == "plain"]
    reg = [c for c in plain if not c["irregular"]]
    bad_reg = [c for c in reg if not c["agree_within_tolerance"]]
    text = [f"Corrected series of {args.run} (scripts/vf_replace_points.py).",
            f"Replaced: {len(corrections)} points of the plain pass (flagged irregular by the smoothness test, re-measured on a quiet machine): " + ", ".join(str(c['ops']) for c in corrections),
            f"Checked, not replaced: {len(reg)} plain-pass points re-measured and not flagged; {len(reg) - len(bad_reg)} agree within {args.tolerance} % in the OpenCV time, the cvGS time and the speedup" + (f"; DISAGREE: " + ", ".join(f"{c['ops']} ({100 * (c['speedup_ratio'] - 1):+.1f} % in the speedup)" for c in bad_reg) if bad_reg else ""),
            "The CUDA Graphs pass is unchanged (no irregular point); checks.csv has its re-measured points too."]
    open(os.path.join(args.out, "CORRECTED.txt"), "w").write("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
