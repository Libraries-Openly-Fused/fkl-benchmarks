#!/usr/bin/env python3
"""The re-measured points of a long VF run that disagree with the original run, with the times that make the difference.

    vf_disagreements.py results/rtx4090/e0-load-audit/spot2/mul_add_comparison.csv [--tolerance 2] [--pass plain|graphs] [--ops 10002,12802,16602] [--out FILE.md]

Input: the CSV of scripts/vf_spot_compare.py (per point and pass: the OpenCV time, the cvGS time and the speedup of the original and of the quiet run). Output: a Markdown table of the points whose
speedup differs by more than the tolerance (percent), with, for each, which of the two times (OpenCV or cvGS, the fused kernel) moved and by how much, so that the text can say where a difference comes from.
"""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--pass", dest="which", help="only this pass (plain or graphs)")
    ap.add_argument("--ops", help="only these numbers of operations (comma separated)")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    if args.which:
        rows = [r for r in rows if r["pass_"] == args.which]
    if args.ops:
        wanted = {int(x) for x in args.ops.split(",")}
        rows = [r for r in rows if int(r["ops"]) in wanted]
    header = "| pass | operations | OpenCV ms (original, quiet, change) | cvGS ms (original, quiet, change) | speedup (original, quiet, change) | the time that moved |\n|---|---|---|---|---|---|\n"
    lines = []
    for r in rows:
        sr = float(r["speedup_ratio"]) - 1.0
        if abs(sr) * 100.0 <= args.tolerance:
            continue
        orr = float(r["opencv_ratio"]) - 1.0
        cr = float(r["cvgs_ratio"]) - 1.0
        moved = "cvGS (fused kernel)" if abs(cr) > abs(orr) else "OpenCV"
        lines.append(f"| {r['pass_']} | {int(r['ops']):,} | {float(r['opencv_original']):.3f}, {float(r['opencv_quiet']):.3f}, {orr * 100:+.1f} % | "
                     f"{float(r['cvgs_original']):.5f}, {float(r['cvgs_quiet']):.5f}, {cr * 100:+.1f} % | {float(r['speedup_original']):,.0f}, {float(r['speedup_quiet']):,.0f}, {sr * 100:+.1f} % | {moved} |")
    text = header + "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
