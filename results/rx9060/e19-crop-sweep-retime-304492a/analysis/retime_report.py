#!/usr/bin/env python3
"""RT-A1: the E19 crop sweep re-time against the record (no GPU).

  retime_report.py <this record's folder>

Reads summary/A and summary/B (the record's e19_crop_sweep_report.py on views/A and views/B, post_window.sh), the record's
../e19-crop-sweep/summary/, analysis/kernels-and-spin-{A,B}.tsv and the checks CSVs of every run. fkl_loop is left out (Oscar, 2026-11-20).
Writes:
  summary/retime.csv          per N, row (fkl_batch, rpp) and interval: old (the record), A and B, each the median of the three quiet runs' means
                              and its spread ((max - min) / median); B against A and A against old: the change, whether it exceeds the larger of
                              the two spreads, and a flag above 3 %; the GPU- or host-bound label of A and B (kernel_only within 3 % of whole_call)
  summary/ratios-retime.csv   per N and interval: rpp / fkl_batch, the median (min-max) of the three per-run ratios, old, A and B
  summary/checks-identity.txt every run's checks CSV (its data rows) against the record's quiet-run1 checks at the same N
  summary/kernels-retime.txt  the kernels per call and the spin coverage of A and B against the record's (kernels-and-spin.tsv)
  summary/retime-report.md    the tables in Markdown
  paper-data/{A,B}/e19_crop_sweep_times.csv, e19_crop_sweep_kernels.csv: the coordinator's package layout (2026-11-26):
      times:   gpu,crops,row,interval,median_us,min_us,max_us,spread_pct,ratio_row_over_fkl_batch_median,ratio_min,ratio_max
               (the three quiet runs' means; the ratio to fkl_batch, median and min-max over the runs, empty on the fkl_batch rows)
      kernels: gpu,crops,row,kernels_per_frame (the kernels of one call, from the rocprofv3 sweep: whole_call kernels / 101 calls)
"""
import csv
import os
import sys

HERE = sys.argv[1]
OLD = os.path.join(HERE, "..", "e19-crop-sweep")
SWEEP = [1] + list(range(10, 151, 10))
ROWS = ["fkl_batch", "rpp"]
INTERVALS = ["whole_call", "kernel_only"]
FLAG = 3.0
GPU = "rx9060"
RUNS = {"A": ["unused-A", "quiet-run1-A", "quiet-run2-A", "quiet-run3-A", "rocprofv3-A"],
        "B": ["unused-B", "quiet-run1-B", "quiet-run2-B", "quiet-run3-B", "rocprofv3-B"]}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def summ(folder):
    return {(int(r["crops"]), r["row"], r["interval"]): r for r in read(os.path.join(folder, "summary.csv"))}


def rats(folder):
    return {(int(r["crops"]), r["interval"]): r for r in read(os.path.join(folder, "ratios.csv"))}


def pct(new, old):
    return 100.0 * (new - old) / old


S = {"old": summ(os.path.join(OLD, "summary")), "A": summ(os.path.join(HERE, "summary", "A")), "B": summ(os.path.join(HERE, "summary", "B"))}
Q = {"old": rats(os.path.join(OLD, "summary")), "A": rats(os.path.join(HERE, "summary", "A")), "B": rats(os.path.join(HERE, "summary", "B"))}
BOUND = {x: {(int(r["crops"]), r["row"]): r["bound"] for r in read(os.path.join(HERE, "summary", x, "bound.csv"))} for x in ("A", "B")}
os.makedirs(os.path.join(HERE, "summary"), exist_ok=True)


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def compare(new, newsp, old, oldsp):
    ch = pct(new, old)
    return ch, abs(ch) > max(newsp, oldsp), abs(ch) > FLAG


retime = []
for n in SWEEP:
    for row in ROWS:
        for it in INTERVALS:
            v = {x: float(S[x][(n, row, it)]["median_of_means_ms"]) for x in S}
            sp = {x: float(S[x][(n, row, it)]["spread_pct"]) for x in S}
            ba = compare(v["B"], sp["B"], v["A"], sp["A"])
            ao = compare(v["A"], sp["A"], v["old"], sp["old"])
            retime.append({"crops": n, "row": row, "interval": it,
                           "old_ms": f"{v['old']:.6f}", "old_spread_pct": f"{sp['old']:.2f}", "A_ms": f"{v['A']:.6f}", "A_spread_pct": f"{sp['A']:.2f}",
                           "B_ms": f"{v['B']:.6f}", "B_spread_pct": f"{sp['B']:.2f}",
                           "B_vs_A_pct": f"{ba[0]:+.2f}", "B_vs_A_exceeds_spread": "yes" if ba[1] else "no", "B_vs_A_flag": "FLAG" if ba[2] else "no",
                           "A_vs_old_pct": f"{ao[0]:+.2f}", "A_vs_old_exceeds_spread": "yes" if ao[1] else "no", "A_vs_old_flag": "FLAG" if ao[2] else "no",
                           "bound_A": BOUND["A"][(n, row)], "bound_B": BOUND["B"][(n, row)]})
write(os.path.join(HERE, "summary", "retime.csv"), retime)

ratios = []
for n in SWEEP:
    for it in INTERVALS:
        r = {"crops": n, "interval": it}
        for x in ("old", "A", "B"):
            q = Q[x][(n, it)]
            r[f"{x}_median"], r[f"{x}_min"], r[f"{x}_max"] = q["median"], q["min"], q["max"]
        ratios.append(r)
write(os.path.join(HERE, "summary", "ratios-retime.csv"), ratios)


def data_rows(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f if not line.startswith("#")]


ident = []
counts = {}
for x, runs in RUNS.items():
    for run in runs:
        for n in SWEEP:
            mine = data_rows(os.path.join(HERE, "window", run, f"n{n:03d}", "e19_batched_box_checks_GPU_AMD.csv"))
            rec = data_rows(os.path.join(OLD, "quiet-run1", f"n{n:03d}", "e19_batched_box_checks_GPU_AMD.csv"))
            diff = [(i, a, b) for i, (a, b) in enumerate(zip(mine, rec)) if a != b]
            same = len(mine) == len(rec) and not diff
            counts[(x, same)] = counts.get((x, same), 0) + 1
            if not same:
                ident.append(f"{run} n{n:03d}: rows {len(mine)} against {len(rec)}; differing rows: " +
                             "; ".join(f"#{i}: {a!r} | record {b!r}" for i, a, b in diff[:4]))
with open(os.path.join(HERE, "summary", "checks-identity.txt"), "w") as f:
    f.write("Every run's checks CSV (the data rows: the header and the column names included, the # build lines left out) against the record's\n"
            "quiet-run1 checks CSV at the same N (../e19-crop-sweep/quiet-run1/nNNN/e19_batched_box_checks_GPU_AMD.csv).\n")
    for x in ("A", "B"):
        f.write(f"{x}: {counts.get((x, True), 0)} of {len(RUNS[x]) * len(SWEEP)} runs identical to the record's\n")
    f.write("\n".join(ident) + ("\n" if ident else ""))


def kern(path):
    with open(path) as f:
        return {(int(r["crops"]), r["row"]): r for r in csv.DictReader(f, delimiter="\t")}


K = {"old": kern(os.path.join(OLD, "analysis", "kernels-and-spin.tsv")), "A": kern(os.path.join(HERE, "analysis", "kernels-and-spin-A.tsv")),
     "B": kern(os.path.join(HERE, "analysis", "kernels-and-spin-B.tsv"))}
klines = ["The kernels per timed range and the spin coverage (kernels-and-spin.tsv), A and B against the record:"]
for x in ("A", "B"):
    ok = [k for k in K[x] if all(K[x][k][c] == K[x][k][c.replace("kernels", "expected")] for c in ("whole_call_kernels", "kernel_only_kernels"))]
    cov = [k for k in K[x] if K[x][k]["covered"] == "100 of 100"]
    same = [k for k in K[x] if all(K[x][k][c] == K["old"][k][c] for c in ("whole_call_kernels", "kernel_only_kernels", "covered"))]
    mn = min(K[x].values(), key=lambda r: float(r["margin_min_us"]))
    klines.append(f"{x}: {len(ok)} of {len(K[x])} (N, row) with the expected kernels; spin covers 100 of 100 in {len(cov)} of {len(K[x])}; "
                  f"kernels and coverage as the record's in {len(same)} of {len(K[x])}; smallest margin {mn['margin_min_us']} us "
                  f"({mn['row']}, N = {mn['crops']})")
with open(os.path.join(HERE, "summary", "kernels-retime.txt"), "w") as f:
    f.write("\n".join(klines) + "\n")

for x in ("A", "B"):
    times = []
    for n in SWEEP:
        for row in ROWS:
            for it in INTERVALS:
                s = S[x][(n, row, it)]
                r = {"gpu": GPU, "crops": n, "row": row, "interval": it, "median_us": f"{1000 * float(s['median_of_means_ms']):.3f}",
                     "min_us": f"{1000 * float(s['min_of_means_ms']):.3f}", "max_us": f"{1000 * float(s['max_of_means_ms']):.3f}",
                     "spread_pct": s["spread_pct"], "ratio_row_over_fkl_batch_median": "", "ratio_min": "", "ratio_max": ""}
                if row != "fkl_batch":
                    q = Q[x][(n, it)]
                    r["ratio_row_over_fkl_batch_median"], r["ratio_min"], r["ratio_max"] = q["median"], q["min"], q["max"]
                times.append(r)
    write(os.path.join(HERE, "paper-data", x, "e19_crop_sweep_times.csv"), times)
    kernels = [{"gpu": GPU, "crops": n, "row": row, "kernels_per_frame": f"{int(K[x][(n, row)]['whole_call_kernels']) / 101:g}"}
               for n in SWEEP for row in ROWS]
    write(os.path.join(HERE, "paper-data", x, "e19_crop_sweep_kernels.csv"), kernels)

md = ["# RT-A1: the E19 crop sweep re-time, old (../e19-crop-sweep/) / A (976e42b, FKL b10fc66) / B (73a9650, FKL 304492a)", "",
      "The median of the three quiet runs' means (ms) and its spread. B against A and A against old: the change; *exceeds* = larger than the larger",
      "of the two spreads; FLAG = above 3 %. Bound: kernel_only within 3 % of whole_call (the record's criterion).", ""]
for row in ROWS:
    for it in INTERVALS:
        md += [f"## {row} {it}", "", "| N | old | A | B | B vs A | exceeds? | flag | A vs old | exceeds? | flag | bound A / B |",
               "|--:|--:|--:|--:|--:|---|---|--:|---|---|---|"]
        for r in [r for r in retime if r["row"] == row and r["interval"] == it]:
            md.append(f"| {r['crops']} | {float(r['old_ms']):.4f} ({r['old_spread_pct']} %) | {float(r['A_ms']):.4f} ({r['A_spread_pct']} %) | "
                      f"{float(r['B_ms']):.4f} ({r['B_spread_pct']} %) | {r['B_vs_A_pct']} % | {r['B_vs_A_exceeds_spread']} | {r['B_vs_A_flag']} | "
                      f"{r['A_vs_old_pct']} % | {r['A_vs_old_exceeds_spread']} | {r['A_vs_old_flag']} | {r['bound_A']} / {r['bound_B']} |")
        md.append("")
md += ["## rpp / fkl_batch: the median (min-max) of the three per-run ratios", "", "| N | interval | old | A | B |", "|--:|---|--:|--:|--:|"]
for r in ratios:
    md.append(f"| {r['crops']} | {r['interval']} | " + " | ".join(f"{float(r[x + '_median']):.2f} ({float(r[x + '_min']):.2f}-{float(r[x + '_max']):.2f})"
                                                        for x in ("old", "A", "B")) + " |")
for name in ("checks-identity.txt", "kernels-retime.txt"):
    md += ["", f"## {name}", "", "```", open(os.path.join(HERE, "summary", name)).read().rstrip("\n")[:6000], "```"]
flags = [r for r in retime if r["B_vs_A_flag"] == "FLAG" or r["A_vs_old_flag"] == "FLAG"]
md += ["", f"## Flags: {sum(r['B_vs_A_flag'] == 'FLAG' for r in retime)} B vs A, {sum(r['A_vs_old_flag'] == 'FLAG' for r in retime)} A vs old, "
       f"of {len(retime)} (N, row, interval)", ""]
for r in flags:
    md.append(f"- N = {r['crops']} {r['row']} {r['interval']}: B vs A {r['B_vs_A_pct']} % ({r['B_vs_A_flag']}), A vs old {r['A_vs_old_pct']} % ({r['A_vs_old_flag']})")
with open(os.path.join(HERE, "summary", "retime-report.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
