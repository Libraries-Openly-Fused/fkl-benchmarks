#!/usr/bin/env python3
"""Task C-NV: the kernels that differ in a pr435_sass_identity.py CSV, grouped by DPP (B's name), with their changes
(instructions, registers, stack, LDL, STL, constant bank 0, shared memory) counted per combination (CPU only).

  diff_summary.py <kernels.csv> [--out summary.txt]
"""
import argparse, collections, csv, sys
csv.field_size_limit(sys.maxsize)
ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--out")
a = ap.parse_args()
rows = list(csv.DictReader(open(a.csv)))
def fam(r):
    n = r["B_demangled"] or r["A_demangled"]
    return n.split("<")[1].split(",")[0].replace("fk::", "") if n.startswith("void fk::launchDPP_Kernel<") else n.split("<")[0]
by = collections.defaultdict(list)
for r in rows:
    if r["status"] == "different":
        by[fam(r)].append(r)
out = [f"{a.csv}: {sum(len(v) for v in by.values())} kernels differ"]
for f in sorted(by):
    rs = by[f]
    out.append(f"{f}: {len(rs)} kernels")
    c = collections.Counter()
    for r in rs:
        d = lambda k: r["A_" + k] if r["A_" + k] == r["B_" + k] else f"{r['A_' + k]} -> {r['B_' + k]}"
        c[(f"instructions {int(r['A_instructions'])} -> {int(r['B_instructions'])} ({int(r['B_instructions']) - int(r['A_instructions']):+d})",
           f"registers {d('registers')}", f"stack {d('stack')} B", f"LDL {d('LDL')}", f"STL {d('STL')}",
           f"cbank0 {d('cbank0')} B", f"shared {d('shared')} B")] += 1
    for k, n in sorted(c.items(), key=lambda t: (-t[1], t[0])):
        out.append(f"  {n} x  " + "; ".join(k))
text = "\n".join(out)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
