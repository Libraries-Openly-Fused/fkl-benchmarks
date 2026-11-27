#!/usr/bin/env python3
"""RT-1: the resources of the kernels of e19_crop_sweep_cu, A (976e42b, FKL b10fc66) against B (main, FKL 304492a), per row and N, from
pr435_sass_identity.py's CSV (cuobjdump 13.4.3 -sass and -res-usage of both binaries; CPU only, reads the CSV).

  fkl_batch_resources.py <sass-e19-A-vs-B.csv> [--out table.txt] [--csv table.csv]

The kernels are told apart by their DPP: fkl_batch's MorphologyDPP (A: morphologyDPPKernel) with the batch size N of its BatchRead, fkl_loop's
LinearFilterDPP (one kernel, N = 1 crop at a time), and any other kernel. For each: registers, stack, local memory, LDL / STL, constant bank 0,
shared memory and SASS instructions in A and in B, and whether the SASS text is identical (addresses and module ids masked).
"""
import argparse
import csv
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("csv_in"); ap.add_argument("--out"); ap.add_argument("--csv")
a = ap.parse_args()
csv.field_size_limit(sys.maxsize)
rows = [r for r in csv.DictReader(open(a.csv_in)) if r["file"] == "e19_crop_sweep_cu"]


def family(name):
    if "MorphologyDPP" in name or "morphologyDPPKernel" in name:
        return "fkl_batch (MorphologyDPP)"
    if "LinearFilterDPP" in name or "linearFilterDPPKernel" in name:
        return "fkl_loop (LinearFilterDPP)"
    return "other: " + re.sub(r"<.*", "", name.replace("void ", ""))[:60]


def crops(name):
    m = re.search(r"BatchRead<\(fk::PlanePolicy\)\d+, (\d+)ul", name) or re.search(r"Batch\w*<(\d+)ul", name)
    return int(m.group(1)) if m else None


out = []
for r in rows:
    name = r["B_demangled"] or r["A_demangled"]
    out.append(dict(family=family(name), n=crops(name), status=r["status"],
                    **{f"{s}_{k}": r[f"{s}_{k}"] for s in "AB" for k in ("registers", "stack", "local", "LDL", "STL", "cbank0", "shared", "instructions")}))
out.sort(key=lambda x: (x["family"], x["n"] or 0))
cols = ("registers", "stack", "local", "LDL", "STL", "cbank0", "shared", "instructions")
lines = [f"e19_crop_sweep_cu: {len(out)} kernels; A = 976e42b with FKL b10fc66, B = main with FKL 304492a (cuobjdump 13.4.3, sm_89)",
         "row (kernel) | N | " + " | ".join(f"{c} A / B" for c in cols) + " | SASS"]
for x in out:
    lines.append(f"{x['family']} | {x['n'] if x['n'] is not None else '-'} | " +
                 " | ".join(f"{x['A_' + c]} / {x['B_' + c]}" for c in cols) + f" | {x['status']}")
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
if a.csv:
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
