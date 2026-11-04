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
"""E16 query-tile experiment: the decision rule of fkl-self/attention/e16_query_tile.cu applied to the three runs.

  python scripts/e16_query_tile_analyse.py results/rtx4090/e16-query-tile

Reads run0/run1/run2 e16_query_tile.csv. Per S and per run: excess(BQ) = time(int8_bqBQ) - time(raw_bqBQ_kv32) (same BQ, BKV = 32, same warps), ratio = excess(128) / excess(64),
share(BQ) = (time(int8) - time(int8nodq)) / (time(int8) - time(raw)); the rows of the int8nodq variants have timing_only = 1 and are used only for the share. Writes
e16_query_tile_derived.csv (per S: the median over the runs of every time, the spread (max - min) / median over the runs, the excess, ratio and share of every run and their median) and .md.
Reading of the ratio (fixed before the first run): <= 0.65 at S >= 2048 proportional to the number of query tiles; >= 0.85 not proportional; between: partly. A ratio near 0.5 means a cost
repeated per query tile, not the conversion; the share of the conversion is the arithmetic only.
"""
import csv
import os
import statistics as st
import sys

BQ = {"32": ("raw_bq32_kv32_w2", "int8_bq32_w2", "int8nodq_bq32_w2"), "64": ("raw_bq64_kv32_w4", "int8_bq64_w4", "int8nodq_bq64_w4"), "128": ("raw_bq128_kv32_w4", "int8_bq128_w4", "int8nodq_bq128_w4")}


def read(path):
    with open(path) as f:
        return list(csv.DictReader([l for l in f if not l.startswith("#")]))


def main():
    d = sys.argv[1]
    runs = []
    for r in sorted(x for x in os.listdir(d) if x.startswith("run")):
        rows = read(os.path.join(d, r, "e16_query_tile.csv"))
        runs.append({(int(x["seq"]), x["variant"]): x for x in rows})
    seqs = sorted({s for r in runs for (s, _) in r})
    variants = sorted({v for r in runs for (_, v) in r})
    out = []
    t = lambda run, s, v: float(run[(s, v)]["time_ms_median"]) if (s, v) in run and float(run[(s, v)]["time_ms_median"]) > 0 else None
    for s in seqs:
        rec = {"seq": s}
        for v in variants:
            vals = [t(r, s, v) for r in runs if t(r, s, v) is not None]
            if vals:
                rec[f"{v}_ms"] = st.median(vals)
                rec[f"{v}_spread_pct"] = (max(vals) - min(vals)) / st.median(vals) * 100
        ex = {}
        for bq, (raw, i8, nd) in BQ.items():
            per = []
            for r in runs:
                a, b = t(r, s, i8), t(r, s, raw)
                per.append(a - b if a is not None and b is not None else None)
            ex[bq] = per
            vals = [x for x in per if x is not None]
            rec[f"excess_bq{bq}_ms_runs"] = " / ".join("n/a" if x is None else f"{x:.4f}" for x in per)
            rec[f"excess_bq{bq}_ms_median"] = st.median(vals) if vals else ""
            shares = []
            for r in runs:
                a, b, c = t(r, s, i8), t(r, s, raw), t(r, s, nd)
                shares.append((a - c) / (a - b) if None not in (a, b, c) and a != b else None)
            sv = [x for x in shares if x is not None]
            rec[f"share_conversion_bq{bq}_runs"] = " / ".join("n/a" if x is None else f"{x:.3f}" for x in shares)
            rec[f"share_conversion_bq{bq}_median"] = st.median(sv) if sv else ""
        for lo, hi in (("64", "128"), ("32", "64")):
            ratios = [(h / l) if (h is not None and l is not None and l != 0) else None for h, l in zip(ex[hi], ex[lo])]
            rv = [x for x in ratios if x is not None]
            rec[f"ratio_{hi}_over_{lo}_runs"] = " / ".join("n/a" if x is None else f"{x:.3f}" for x in ratios)
            rec[f"ratio_{hi}_over_{lo}_median"] = st.median(rv) if rv else ""
            if hi == "128" and rv:
                m = st.median(rv)
                rec["reading_128_over_64"] = ("proportional to the number of query tiles" if m <= 0.65 else "not proportional" if m >= 0.85 else "partly") + ("" if s >= 2048 else " (S < 2048: rule applies at S >= 2048)")
        out.append(rec)
    cols = list(dict.fromkeys(k for r in out for k in r))
    with open(os.path.join(d, "e16_query_tile_derived.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    md = ["| S | excess BQ32 / BQ64 / BQ128 (ms, median of runs) | ratio 128/64 (runs) | median | reading | share of the conversion BQ64 / BQ128 | spread of int8_bq64 / raw_bq64 (%) |", "|---:|---|---|---:|---|---|---|"]
    for r in out:
        f3 = lambda k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else "n/a")
        md.append(f"| {r['seq']} | {f3('excess_bq32_ms_median')} / {f3('excess_bq64_ms_median')} / {f3('excess_bq128_ms_median')} | {r.get('ratio_128_over_64_runs', '')} | {f3('ratio_128_over_64_median')} | {r.get('reading_128_over_64', '')} | "
                  f"{f3('share_conversion_bq64_median')} / {f3('share_conversion_bq128_median')} | {r.get('int8_bq64_w4_spread_pct', float('nan')):.1f} / {r.get('raw_bq64_kv32_w4_spread_pct', float('nan')):.1f} |")
    open(os.path.join(d, "e16_query_tile_derived.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
