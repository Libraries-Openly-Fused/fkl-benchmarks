#!/usr/bin/env python3
"""Task B: the outputs of the checks runs before (FKL 24478f0) against after (FKL 4b11bc6), bit for bit (CPU only; nothing is run).

  compare_outputs.py --runs <window folder>/runs --dumps <dump folder> [--out comparison.txt] [--csv comparison.csv]

<runs>/before/<run> and <runs>/after/<run> are the run folders of the same program and settings (<run> = e18_box, e18_box_split, e18_stencil,
e19_batched_box, e19_crop_sweep/N<NNN>); <dumps>/before/<run> and <dumps>/after/<run> their E19_DUMP folders (E19 only).
- Every CSV of a run folder: its rows that do not start with '#' (the header lines hold the commits, the clocks and the temperature, which are
  expected to differ) compared as text, row by row: equal or the first differing row (its number among the data rows, and both rows). The
  sha256 of the data rows on each side is the digest.
- Every dump file: its sha256 on each side, and byte for byte: equal or the first differing byte (for a .f32 file also the index of the value
  and both values; for a .u8 file both bytes).
- stdout.txt: the lines that differ are listed (reported only: it names the commits of the build).
Files on one side only are listed as differences. Exit 0 when every CSV's data rows and every dump file are equal on both sides.
"""
import argparse
import array
import csv
import hashlib
import os
import struct
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True); ap.add_argument("--dumps", required=True)
ap.add_argument("--out", default=None); ap.add_argument("--csv", default=None)
args = ap.parse_args()


def run_dirs(side):
    root = os.path.join(args.runs, side)
    out = []
    for d, dirs, files in os.walk(root):
        if "exit_code.txt" in files:
            out.append(os.path.relpath(d, root))
    return sorted(out)


def data_rows(path):
    with open(path, "rb") as f:
        return [l for l in f.read().split(b"\n") if l and not l.startswith(b"#")]


def sha(b):
    return hashlib.sha256(b).hexdigest()


lines, rows, bad = [], [], 0
before, after = run_dirs("before"), run_dirs("after")
lines.append(f"run folders: before {len(before)}, after {len(after)}; only before: {sorted(set(before) - set(after))}; only after: {sorted(set(after) - set(before))}")
bad += len(set(before) ^ set(after))
n_csv = n_csv_eq = n_dump = n_dump_eq = 0
for run in sorted(set(before) & set(after)):
    a_dir, b_dir = (os.path.join(args.runs, s, run) for s in ("before", "after"))
    csvs = sorted({f for f in os.listdir(a_dir) if f.endswith(".csv") and f != "gpu-at-end.csv"} |
                  {f for f in os.listdir(b_dir) if f.endswith(".csv") and f != "gpu-at-end.csv"})
    for f in csvs:
        pa, pb = os.path.join(a_dir, f), os.path.join(b_dir, f)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            lines.append(f"{run}/{f}: ON ONE SIDE ONLY ({'before' if os.path.exists(pa) else 'after'})"); bad += 1
            rows.append([run, f, "csv", "one side only", "", "", ""]); continue
        ra, rb = data_rows(pa), data_rows(pb)
        da, db = sha(b"\n".join(ra)), sha(b"\n".join(rb))
        n_csv += 1
        if ra == rb:
            n_csv_eq += 1
            lines.append(f"{run}/{f}: {len(ra)} data rows, EQUAL (sha256 {da})")
            rows.append([run, f, "csv", "equal", len(ra), da, db])
        else:
            bad += 1
            first = next((i for i in range(min(len(ra), len(rb))) if ra[i] != rb[i]), min(len(ra), len(rb)))
            lines.append(f"{run}/{f}: DIFFERENT: data rows before {len(ra)}, after {len(rb)}; sha256 {da} / {db}; first difference at data row {first + 1}:")
            lines.append(f"    before: {ra[first].decode(errors='replace') if first < len(ra) else '(none)'}")
            lines.append(f"    after:  {rb[first].decode(errors='replace') if first < len(rb) else '(none)'}")
            rows.append([run, f, "csv", f"different from data row {first + 1}", f"{len(ra)}/{len(rb)}", da, db])
    sa, sb = (os.path.join(s, "stdout.txt") for s in (a_dir, b_dir))
    if os.path.exists(sa) and os.path.exists(sb):
        la, lb = open(sa, errors="replace").read().splitlines(), open(sb, errors="replace").read().splitlines()
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y]
        lines.append(f"{run}/stdout.txt: {len(la)} / {len(lb)} lines, {len(diff)} differ (reported only)" + ("" if len(la) == len(lb) else "; THE LINE COUNTS DIFFER"))
        for i, x, y in diff[:6]:
            lines.append(f"    line {i + 1}: before: {x[:200]}")
            lines.append(f"    line {i + 1}: after:  {y[:200]}")
    da_dir, db_dir = (os.path.join(args.dumps, s, run) for s in ("before", "after"))
    if os.path.isdir(da_dir) or os.path.isdir(db_dir):
        files = sorted(set(os.listdir(da_dir) if os.path.isdir(da_dir) else []) | set(os.listdir(db_dir) if os.path.isdir(db_dir) else []))
        for f in files:
            pa, pb = os.path.join(da_dir, f), os.path.join(db_dir, f)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                lines.append(f"dump {run}/{f}: ON ONE SIDE ONLY"); bad += 1
                rows.append([run, f, "dump", "one side only", "", "", ""]); continue
            ba, bb = open(pa, "rb").read(), open(pb, "rb").read()
            ha, hb = sha(ba), sha(bb)
            n_dump += 1
            if ba == bb:
                n_dump_eq += 1
                lines.append(f"dump {run}/{f}: {len(ba)} bytes, EQUAL (sha256 {ha})")
                rows.append([run, f, "dump", "equal", len(ba), ha, hb])
            else:
                bad += 1
                first = next((i for i in range(min(len(ba), len(bb))) if ba[i] != bb[i]), min(len(ba), len(bb)))
                what = f"first differing byte {first}"
                if f.endswith(".f32") and first // 4 * 4 + 4 <= min(len(ba), len(bb)):
                    k = first // 4
                    va, vb = struct.unpack_from("<f", ba, 4 * k)[0], struct.unpack_from("<f", bb, 4 * k)[0]
                    m = min(len(ba), len(bb)) // 4 * 4
                    ua, ub = array.array("I", ba[:m]), array.array("I", bb[:m])   # the bit patterns, so that NaNs compare as bits
                    ndiff = sum(1 for x, y in zip(ua, ub) if x != y)
                    what += f" = value {k}: before {va!r}, after {vb!r}; values that differ: {ndiff} of {min(len(ba), len(bb)) // 4}"
                elif first < min(len(ba), len(bb)):
                    what += f": before {ba[first]}, after {bb[first]}"
                lines.append(f"dump {run}/{f}: DIFFERENT: {len(ba)} / {len(bb)} bytes; sha256 {ha} / {hb}; {what}")
                rows.append([run, f, "dump", what, f"{len(ba)}/{len(bb)}", ha, hb])
lines.insert(1, f"CSV files: {n_csv} on both sides, {n_csv_eq} with equal data rows; dump files: {n_dump} on both sides, {n_dump_eq} equal; "
                f"{'EVERYTHING EQUAL' if bad == 0 else f'{bad} DIFFERENCES'}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "file", "kind", "result", "rows_or_bytes", "sha256_before", "sha256_after"])
        w.writerows(rows)
sys.exit(0 if bad == 0 else 1)
