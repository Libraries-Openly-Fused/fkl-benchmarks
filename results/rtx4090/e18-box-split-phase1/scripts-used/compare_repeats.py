#!/usr/bin/env python3
"""The repeated rows of e18_box_split_cu against e18_box_cu's record: every agreement row of the split run whose comparison e18_box.cu also
makes, and the exact-reference checks, field by field. Nothing is run.

  compare_repeats.py <split agreement.csv> <split checks.csv> <e18_box agreement.csv> <e18_box checks.csv> [--out report.txt]

The split program names npp's resized image (the intermediate of the npp row) "npp_intermediate"; e18_box.cu names it "npp" as a version in the
resize step and "npp_resize" as the comparison it is compared against. Those labels are mapped before the rows are matched. Every other label
is the same in both programs.
"""
import argparse
import csv

ap = argparse.ArgumentParser()
for a in ("split_agreement", "split_checks", "box_agreement", "box_checks"):
    ap.add_argument(a)
ap.add_argument("--out", default=None)
args = ap.parse_args()


def read(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


KEY = ("step", "version", "sampling", "against")


def key(r, mapped):
    k = [r[c] for c in KEY]
    if mapped and k[0] == "resize":
        k[1] = "npp" if k[1] == "npp_intermediate" else k[1]
        k[3] = "npp_resize" if k[3] == "npp_intermediate" else k[3]
    return tuple(k)


split = {key(r, True): r for r in read(args.split_agreement)}
shown = {key(r, True): tuple(r[c] for c in KEY) for r in read(args.split_agreement)}   # the labels as the split program writes them
box = {key(r, False): r for r in read(args.box_agreement)}
lines = []
same = differ = 0
for k, r in split.items():
    if k not in box:
        lines.append(f"new (not in e18_box): {' / '.join(shown[k])}: {r['values_differing']} values differ")
        continue
    b = box[k]
    cols = [c for c in r if c not in KEY]
    diffs = [f"{c}: {b[c]} -> {r[c]}" for c in cols if r[c] != b.get(c)]
    if diffs:
        differ += 1
        lines.append(f"DIFFERENT: {' / '.join(shown[k])} (e18_box: {' / '.join(k)}): " + "; ".join(diffs))
    else:
        same += 1
        lines.append(f"same: {' / '.join(shown[k])} (e18_box: {' / '.join(k)}): {r['values_differing']} values differ in both")
sc = {(r["check"], r["sampling"]): r for r in read(args.split_checks)}
bc = {(r["check"], r["sampling"]): r for r in read(args.box_checks)}
for k, r in sc.items():
    b = bc.get(k)
    if b is None:
        lines.append(f"check not in e18_box: {k}")
    elif any(r[c] != b[c] for c in r):
        differ += 1
        lines.append(f"DIFFERENT check: {k}: {b['value']} -> {r['value']}")
    else:
        same += 1
        lines.append(f"same check: {k[0]} ({k[1]}): {r['value']}")
lines.append(f"{same} rows the same, {differ} different")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
