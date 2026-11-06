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
"""Two controlled compile repeats (distinct-operations chain, variant A), length by length, in the standing form: old value, new value, the values in the order they ran, the run-to-run spread of
each, the change, whether it exceeds the spread, and the 5 % flag for a host-bound measurement (a compile is host-bound: nvcc, cicc, ptxas and g++ on one core).

  python scripts/e3_repeat_pair.py OLD_DIR NEW_DIR [--old-label c8fcbe8] [--new-label 9e20bd4e] [--out PREFIX]

Reads e3_repeat_A.csv and e3_repeat_extra_A.csv (the ones that exist) of each folder; only compiles with status ok enter a value. Writes PREFIX.csv and PREFIX.md (default PREFIX: NEW_DIR/e3_repeat_vs_<old label>).
Quantities: nvcc wall time (cuda_wall_s), peak resident memory (cuda_peak_rss_mb, MiB: GNU time reports KiB and the sweep divides by 1024), the phase times of nvcc --time (cudafe++, cicc, ptxas, host g++), and the
quantities that do not vary between compiles of the same file (registers, spill stores and loads, stack, constant bank 0 bytes, SASS instructions, .text bytes, cubin bytes), which are compared as equal or different.
spread = (max - min) / median of the compiles of that quantity of one folder; the spread of a comparison is the larger of the two. change = (new median - old median) / old median. The change exceeds the
spread when |change| > spread; the flag is 5 %. A length that exists in one folder only is listed as such.
"""
import argparse
import csv
import io
import os
import statistics as st

TIMED = [("cuda_wall_s", "nvcc wall (s)"), ("cuda_peak_rss_mb", "peak memory (MiB)"), ("phase_cudafe_s", "cudafe++ (s)"), ("phase_cicc_s", "cicc (s)"), ("phase_ptxas_s", "ptxas (s)"), ("phase_gcc_s", "host g++ (s)")]
STATIC = [("registers", "registers"), ("spill_stores", "spill stores"), ("spill_loads", "spill loads"), ("stack_bytes", "stack bytes"), ("cmem0_bytes", "cmem[0] bytes"),
          ("sass_instructions", "SASS instructions"), ("text_bytes", ".text bytes"), ("cubin_bytes", "cubin bytes")]


def read_runs(folder):
    runs = {}
    for name in ("e3_repeat_A.csv", "e3_repeat_extra_A.csv"):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        lines = [l for l in open(path) if not l.startswith("#")]
        for r in csv.DictReader(io.StringIO("".join(lines))):
            if r.get("status") == "ok":
                runs.setdefault(int(r["n"]), []).append(r)
    return runs


def values(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (TypeError, ValueError, KeyError):
            pass
    return out


def spread(v):
    m = st.median(v)
    return (max(v) - min(v)) / m * 100 if m else 0.0


def fmt(v):
    return " / ".join(f"{x:g}" if abs(x) >= 1000 or float(x).is_integer() else f"{x:.3f}" for x in v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--old-label", default="c8fcbe8")
    ap.add_argument("--new-label", default="9e20bd4e")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(a.new, f"e3_repeat_vs_{a.old_label}")
    old, new = read_runs(a.old), read_runs(a.new)
    rows = []
    for n in sorted(set(old) | set(new)):
        if n not in old or n not in new:
            rows.append(dict(n=n, quantity="(length in one folder only)", old="" if n not in old else len(old[n]), new="" if n not in new else len(new[n])))
            continue
        for key, label in TIMED:
            o, w = values(old[n], key), values(new[n], key)
            if not o or not w:
                continue
            mo, mw = st.median(o), st.median(w)
            sp = max(spread(o), spread(w))
            ch = (mw - mo) / mo * 100 if mo else 0.0
            rows.append(dict(n=n, quantity=label, old=mo, new=mw, old_values=fmt(o), new_values=fmt(w), old_spread_pct=spread(o), new_spread_pct=spread(w), spread_pct=sp, change_pct=ch,
                             exceeds_spread="yes" if abs(ch) > sp else "no", over_5pct="yes" if abs(ch) > 5 else "no"))
        for key, label in STATIC:
            o, w = values(old[n], key), values(new[n], key)
            if not o or not w:
                continue
            same = set(o) == set(w) and len(set(o)) == 1
            rows.append(dict(n=n, quantity=label, old=st.median(o), new=st.median(w), old_values=fmt(o), new_values=fmt(w), old_spread_pct="", new_spread_pct="", spread_pct="",
                             change_pct=(st.median(w) - st.median(o)) / st.median(o) * 100 if st.median(o) else 0.0, exceeds_spread="equal" if same else "DIFFERENT", over_5pct=""))
    cols = ["n", "quantity", "old", "new", "old_values", "new_values", "old_spread_pct", "new_spread_pct", "spread_pct", "change_pct", "exceeds_spread", "over_5pct"]
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
    md = [f"# Compile repeat at {a.new_label} against the repeat at {a.old_label} (scripts/e3_repeat_pair.py)", "",
          f"old = {a.old}, new = {a.new}; values in the order they ran; spread = the larger (max - min) / median of the two folders; 'exceeds' = |change| > spread; flag = |change| > 5 %.", "",
          "| N | quantity | old | new | old values | new values | spread % | change % | exceeds spread? | > 5 %? |", "|---:|---|---:|---:|---|---|---:|---:|---|---|"]
    for r in rows:
        if "old_values" not in r:
            md.append(f"| {r['n']:,} | {r['quantity']} | {r['old']} | {r['new']} | | | | | | |")
            continue
        sp = f"{r['spread_pct']:.2f}" if r["spread_pct"] != "" else ""
        md.append(f"| {r['n']:,} | {r['quantity']} | {r['old']:g} | {r['new']:g} | {r['old_values']} | {r['new_values']} | {sp} | {r['change_pct']:+.2f} | {r['exceeds_spread']} | {r['over_5pct']} |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}.csv and {out}.md")


if __name__ == "__main__":
    main()
