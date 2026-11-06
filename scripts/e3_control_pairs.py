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
"""The same-day control of a compile-time difference between two library trees: compiles of the same translation unit alternating between the trees in pairs (the window script writes one CSV per
compile, pair<p>_<k>_<tree>.csv, k = the position in the pair), in the standing form.

  python scripts/e3_control_pairs.py DIR --old c8fcbe8 --new 9e20bd4e [--repeats OLD_REPEAT_DIR NEW_REPEAT_DIR]

Per quantity (nvcc wall time, peak memory, the phases cudafe++, cicc, ptxas, host g++): the compiles of each tree in the order they ran, the medians, spread = the larger (max - min) / median of the two trees,
change = (new median - old median) / old median, whether it exceeds the spread, the 5 % flag (host-bound), and the PAIRED differences (new - old) / old of each pair with their median. With --repeats, the
medians of the same length in the two controlled repeats are printed beside them, for the question whether the repeats' difference reappears in the alternating pairs. The code quantities (registers,
SASS instructions, .text and cubin bytes) are listed as equal or different. Writes DIR/control_pairs.md and DIR/control_pairs.csv.
"""
import argparse
import csv
import glob
import io
import os
import re
import statistics as st

TIMED = [("cuda_wall_s", "nvcc wall (s)"), ("cuda_peak_rss_mb", "peak memory (MiB)"), ("phase_cudafe_s", "cudafe++ (s)"), ("phase_cicc_s", "cicc (s)"), ("phase_ptxas_s", "ptxas (s)"), ("phase_gcc_s", "host g++ (s)")]
CODE = ["registers", "spill_stores", "spill_loads", "stack_bytes", "cmem0_bytes", "sass_instructions", "text_bytes", "cubin_bytes"]
NAME = re.compile(r"pair(\d+)_(\d+)_(.+)\.csv$")


def rows_of(path):
    lines = [l for l in open(path) if not l.startswith("#")]
    return [r for r in csv.DictReader(io.StringIO("".join(lines))) if r.get("status") == "ok"]


def spread(v):
    m = st.median(v)
    return (max(v) - min(v)) / m * 100 if m else 0.0


def repeat_median(folder, n, key):
    v = []
    for name in ("e3_repeat_A.csv", "e3_repeat_extra_A.csv"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            v += [float(r[key]) for r in rows_of(p) if int(r["n"]) == n and r.get(key) not in (None, "")]
    return st.median(v) if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--old", default="c8fcbe8")
    ap.add_argument("--new", default="9e20bd4e")
    ap.add_argument("--repeats", nargs=2, metavar=("OLD_REPEAT", "NEW_REPEAT"))
    a = ap.parse_args()
    comp = []                                                   # (pair, position, tree, row)
    for path in sorted(glob.glob(os.path.join(a.dir, "pair*_*_*.csv"))):
        m = NAME.search(os.path.basename(path))
        for r in rows_of(path):
            comp.append((int(m.group(1)), int(m.group(2)), m.group(3), r))
    comp.sort(key=lambda c: (c[0], c[1]))
    n_values = sorted({int(c[3]["n"]) for c in comp})
    assert len(n_values) == 1, f"one length expected, found {n_values}"
    n = n_values[0]
    order = ", ".join(f"({', '.join(c[2] for c in comp if c[0] == p)})" for p in sorted({c[0] for c in comp}))
    md = [f"# Same-day control at N = {n:,}: {a.old} and {a.new} alternating (scripts/e3_control_pairs.py)", "",
          f"Compiles in the order they ran, by pair: {order}. spread = the larger (max - min) / median of the two trees; 'exceeds' = |change| > spread; flag = |change| > 5 %; paired = (new - old) / old within each pair.", ""]
    head = "| quantity | " + a.old + " (in order) | " + a.new + " (in order) | median " + a.old + " | median " + a.new + " | spread % | change % | exceeds? | > 5 %? | paired changes % | median paired % |"
    if a.repeats:
        head += f" repeats: {a.old} -> {a.new} median | repeats' change % |"
    md += [head, "|" + "---|" * (head.count("|") - 1)]
    out_rows = []
    for key, label in TIMED:
        o = [float(c[3][key]) for c in comp if c[2] == a.old]
        w = [float(c[3][key]) for c in comp if c[2] == a.new]
        mo, mw = st.median(o), st.median(w)
        sp = max(spread(o), spread(w))
        ch = (mw - mo) / mo * 100
        paired = []
        for p in sorted({c[0] for c in comp}):
            po = [float(c[3][key]) for c in comp if c[0] == p and c[2] == a.old]
            pw = [float(c[3][key]) for c in comp if c[0] == p and c[2] == a.new]
            if po and pw:
                paired.append((pw[0] - po[0]) / po[0] * 100)
        row = dict(quantity=label, old_values=" / ".join(f"{x:g}" for x in o), new_values=" / ".join(f"{x:g}" for x in w), old_median=mo, new_median=mw, spread_pct=sp, change_pct=ch,
                   exceeds="yes" if abs(ch) > sp else "no", over_5pct="yes" if abs(ch) > 5 else "no", paired_pct=" / ".join(f"{x:+.2f}" for x in paired), paired_median_pct=st.median(paired))
        line = (f"| {label} | {row['old_values']} | {row['new_values']} | {mo:g} | {mw:g} | {sp:.2f} | {ch:+.2f} | {row['exceeds']} | {row['over_5pct']} | {row['paired_pct']} | "
                f"{row['paired_median_pct']:+.2f} |")
        if a.repeats:
            ro, rw = repeat_median(a.repeats[0], n, key), repeat_median(a.repeats[1], n, key)
            row.update(repeat_old=ro, repeat_new=rw, repeat_change_pct=(rw - ro) / ro * 100 if ro and rw else None)
            line += f" {ro:g} -> {rw:g} | {row['repeat_change_pct']:+.2f} |" if ro and rw else " | |"
        md.append(line)
        out_rows.append(row)
    same = {k: len({c[3][k] for c in comp}) == 1 for k in CODE}
    md += ["", "Code quantities over all " + str(len(comp)) + " compiles: " + ", ".join(f"{k} {'equal' if v else 'DIFFERENT'} ({', '.join(sorted({c[3][k] for c in comp}))})" for k, v in same.items())]
    open(os.path.join(a.dir, "control_pairs.md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(a.dir, "control_pairs.csv"), "w", newline="") as f:
        cols = list(out_rows[0])
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("\n".join(md))


if __name__ == "__main__":
    main()
