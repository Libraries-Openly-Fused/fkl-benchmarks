#!/usr/bin/env python3
"""Summary of testB_results.csv: per flag set and batch the median over the three rounds; the per-crop slope and the constant of the construction time (microseconds per call)."""
import csv, statistics as st, collections, sys
R = list(csv.DictReader(l for l in open(sys.argv[1] if len(sys.argv) > 1 else "testB_results.csv") if not l.startswith("#")))
cols = [("cvgs_all_us", "cvGS: resize + cvtColor + multiply + subtract + divide + split"), ("fastnpp_all_us", "FastNPP: resize + swap + MulC + SubC + DivC + CopyBatch"), ("cvgs_read_us", "cvGS: the read descriptor only"), ("fastnpp_read_us", "FastNPP: the read descriptor only")]
D = collections.defaultdict(list)
for r in R:
    for c, _ in cols: D[(r["flagset"], int(r["batch"]), c)].append(float(r[c]))
order = ["o3ndebug", "o2ndebug", "o1ndebug", "o0ndebug", "o3debugassert", "emptybuildtype"]
Bs = [2, 12, 42, 142]
med = lambda f, b, c: st.median(D[(f, b, c)])
spr = lambda f, b, c: (max(D[(f, b, c)]) - min(D[(f, b, c)])) / st.median(D[(f, b, c)]) * 100
out = []
for c, label in cols:
    out += [f"### {label} (microseconds per call; median of 3 rounds; the largest spread of the 3 rounds of a cell in the last column)", "",
            "| host flags | B = 2 | B = 12 | B = 42 | B = 142 | slope B 2..42 (ns per crop) | slope B 2..142 | constant (us) | largest spread % |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in order:
        v = [med(f, b, c) for b in Bs]
        s42, s142 = (v[2] - v[0]) / 40 * 1000, (v[3] - v[0]) / 140 * 1000
        out.append(f"| {f} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {s42:.1f} | {s142:.1f} | {v[0] - 2 * s142 / 1000:.3f} | {max(spr(f, b, c) for b in Bs):.0f} |")
    out.append("")
print("\n".join(out))
