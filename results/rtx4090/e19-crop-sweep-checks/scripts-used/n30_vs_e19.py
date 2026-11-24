#!/usr/bin/env python3
"""The N = 30 checks of the crop sweep against E19's checks window (results/rtx4090/e19-batched-box-fkl-batch-checks), row by row (nothing is run).

  n30_vs_e19.py <sweep checks.csv at N = 30> <E19 checks.csv> [--out report.txt]

Every row of E19's file is looked up in the sweep's by (version, step, compared_with, rule); the column crops is dropped. The rows of the sweep
that E19 does not have (the between rows of fkl_batch) are listed.
"""
import argparse, csv
ap = argparse.ArgumentParser(); ap.add_argument("new"); ap.add_argument("old"); ap.add_argument("--out", default=None); args = ap.parse_args()
rd = lambda p: list(csv.DictReader(l for l in open(p) if not l.startswith("#")))
new, old = rd(args.new), rd(args.old)
key = lambda r: (r["version"], r["step"], r["compared_with"], r["rule"].replace("of the 30 grey and 30 box", "of the 30 grey and 30 box"))
nd = {key(r): r for r in new}
cols = ["values", "differing_values", "max_abs_difference", "non_finite", "criterion", "ok"]
same, diff, lines = 0, 0, []
for r in old:
    n = nd.get(key(r))
    if n is None:
        lines.append(f"MISSING in the sweep: {key(r)}"); diff += 1; continue
    if all(n[c] == r[c] for c in cols): same += 1
    else:
        diff += 1; lines.append(f"DIFFERENT: {key(r)}: " + "; ".join(f"{c} {r[c]} -> {n[c]}" for c in cols if n[c] != r[c]))
extra = [k for k in nd if k not in {key(r) for r in old}]
lines.insert(0, f"E19's {len(old)} rows: {same} equal in every value column ({', '.join(cols)}), {diff} not; the sweep's rows E19 does not have: {len(extra)}")
lines += [f"  new: {k}" for k in extra]
t = "\n".join(lines); print(t)
if args.out: open(args.out, "w").write(t + "\n")
