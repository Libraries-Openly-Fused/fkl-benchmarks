#!/usr/bin/env python3
"""The report of the window e18-retime-304492a, RT-4 (CPU only: reads the CSVs; nothing is run). #63's retime_report.py
(results/rtx4090/e18-retime-b10fc66/scripts-used/), with "old" = #63's B (FKL b10fc66, whose timed code A repeats; see code-check/).

  retime_report.py <window folder> <results/rtx4090 folder> [--out report.txt] [--csv summary.csv] [--clocks clocks.txt] [--checks checks.txt]
                   [--md table.md]

Per program (e18_box, e18_box_split, e18_stencil), per row (version) and interval (whole_call, kernel_only):
  old  #63's B: the three quiet runs of its window (../e18-retime-b10fc66/window/<program>/quiet-run1..3-B/<program>.csv), recomputed here
       the way #63 computed them, and compared with #63's analysis/retime-summary.csv, its B columns (any difference is reported);
  A, B the three quiet runs of this window (<program>/quiet-run<k>-A/ and -B/);
  each: the median of the three runs' means (mean_ms, 100 iterations each), the min and max of the three means, and the spread
  (max - min) / median, in %, as in the records.
Changes: B against A, B against old, A against old, (x / y - 1) in % of the medians. A change exceeds the spread when its size is larger
than the larger spread of the two sides; the ranges (min-max of the means) are also said to overlap or not. Bound: a row's interval is
GPU-bound when its kernel_only and whole_call medians (of the same side) are within 10 % of each other and the host preparation of
kernel_only is below 1 % of its GPU time (as in the records, where every E18 row is GPU-bound); otherwise host-bound. The flag threshold is
3 % for GPU-bound and 5 % for host-bound numbers.
Clocks (--clocks): per run, the SM clock readings of gpu_clocks.csv (every 100 ms) from the run's start to its end (load.txt, to the
second), their distinct values with counts, and the largest reading.
Checks (--checks): every CSV of a run other than the timing CSV (agreement, checks, kernel), its data rows (lines not starting with '#')
against those of #63's B quiet-run1-B: equal, or the rows that differ.
"""
import argparse, csv, datetime, glob, os, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("window"); ap.add_argument("records")
ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--clocks"); ap.add_argument("--checks"); ap.add_argument("--md")
args = ap.parse_args()
PROGS = [("e18_box", "e18-box", "e18_box_summary.csv"), ("e18_box_split", "e18-box-split", "e18_box_split_summary.csv"),
         ("e18_stencil", "e18-stencil", "e18_summary.csv")]


def timing(path):
    rows = [r for r in csv.reader(l for l in open(path) if not l.startswith("#"))]
    h = rows[0]
    out = {}
    for r in rows[1:]:
        d = dict(zip(h, r))
        out[(d["version"], d["interval"])] = (float(d["mean_ms"]), float(d["host_prepare_mean_ms"]) if d.get("host_prepare_mean_ms") else None)
    return out, [ (dict(zip(h, r))["version"], dict(zip(h, r))["interval"]) for r in rows[1:] ]


def stats(vals):
    m = statistics.median(vals)
    return dict(med=m, lo=min(vals), hi=max(vals), spread=(max(vals) - min(vals)) / m * 100.0)


def data_rows(path):
    return [l.rstrip("\n") for l in open(path) if l.strip() and not l.startswith("#")]


lines, table, clock_lines, check_lines = [], [], [], []
for prog, rec, summ in PROGS:
    sides = {"old": [os.path.join(args.records, "e18-retime-b10fc66", "window", prog, f"quiet-run{k}-B", f"{prog}.csv") for k in (1, 2, 3)],
             "A": [os.path.join(args.window, prog, f"quiet-run{k}-A", f"{prog}.csv") for k in (1, 2, 3)],
             "B": [os.path.join(args.window, prog, f"quiet-run{k}-B", f"{prog}.csv") for k in (1, 2, 3)]}
    runs = {s: [timing(p) for p in ps] for s, ps in sides.items()}
    order = runs["A"][0][1]
    for s in ("old", "A", "B"):
        for t, o in runs[s]:
            if o != order:
                lines.append(f"{prog}: the rows or their order of a {s} run differ from A's quiet-run1: {o} against {order}")
    # #63's summary CSV, its B columns, for the cross-check of 'old' (in us there; in ms here)
    summary = {}
    for r in csv.DictReader(open(os.path.join(args.records, "e18-retime-b10fc66", "analysis", "retime-summary.csv"))):
        if r["program"] == prog:
            summary[(r["version"], r["interval"])] = float(r["B_us"]) * 1e-3
    st = {}
    for s in ("old", "A", "B"):
        for key in order:
            st[(s, key)] = stats([t[key][0] for t, _ in runs[s]])
            hp = [t[key][1] for t, _ in runs[s] if t[key][1] is not None]
            st[(s, key)]["hp"] = statistics.median(hp) if hp else None
    for key in order:
        if key not in summary:
            lines.append(f"{prog} {key}: not in #63's summary")
        elif abs(summary[key] - st[("old", key)]["med"]) > 1e-9:
            lines.append(f"{prog} {key}: #63's summary says {summary[key]} ms for B, recomputed {st[('old', key)]['med']} ms")
    for key in order:
        version, interval = key
        row = dict(program=prog, version=version, interval=interval)
        for s in ("old", "A", "B"):
            x = st[(s, key)]
            row.update({f"{s}_us": x["med"] * 1e3, f"{s}_min_us": x["lo"] * 1e3, f"{s}_max_us": x["hi"] * 1e3, f"{s}_spread_pct": x["spread"]})
        # bound, per side, from that side's own kernel_only and whole_call
        def bound(s):
            ko, wc = st[(s, (version, "kernel_only"))], st[(s, (version, "whole_call"))]
            gpu = abs(ko["med"] / wc["med"] - 1) <= 0.10 and (ko["hp"] is None or ko["hp"] < 0.01 * ko["med"])
            return "GPU" if gpu else "host"
        bounds = {bound(s) for s in ("old", "A", "B")}
        row["bound"] = "GPU" if bounds == {"GPU"} else "host"
        thr = 3.0 if row["bound"] == "GPU" else 5.0
        for name, x, y in (("B_vs_A", "B", "A"), ("B_vs_old", "B", "old"), ("A_vs_old", "A", "old")):
            d = (st[(x, key)]["med"] / st[(y, key)]["med"] - 1) * 100.0
            sp = max(st[(x, key)]["spread"], st[(y, key)]["spread"])
            overlap = not (st[(x, key)]["lo"] > st[(y, key)]["hi"] or st[(y, key)]["lo"] > st[(x, key)]["hi"])
            row[f"{name}_pct"] = d; row[f"{name}_exceeds_spread"] = abs(d) > sp; row[f"{name}_ranges_overlap"] = overlap
            row[f"{name}_flag"] = abs(d) > thr
        table.append(row)

hdr = ["program", "version", "interval", "bound", "old_us", "old_min_us", "old_max_us", "old_spread_pct", "A_us", "A_min_us", "A_max_us",
       "A_spread_pct", "B_us", "B_min_us", "B_max_us", "B_spread_pct"]
for n in ("B_vs_A", "B_vs_old", "A_vs_old"):
    hdr += [f"{n}_pct", f"{n}_exceeds_spread", f"{n}_ranges_overlap", f"{n}_flag"]
out = ["E18 re-time RT-4 at FKL b10fc66 (A) and 304492a (B); old = #63's B (FKL b10fc66). Times in us: median of the three quiet runs' means (min-max, spread %).",
       "Changes in % of the medians; 'exceeds' = larger than the larger spread of the two sides; FLAG = above 3 % (GPU-bound) or 5 % (host-bound)."]
out += lines if lines else ["consistency: every run has the same rows in the same order, and 'old' equals #63's summary (its B columns)"]
for r in table:
    def side(s):
        return f"{r[s + '_us']:.2f} ({r[s + '_min_us']:.2f}-{r[s + '_max_us']:.2f}, {r[s + '_spread_pct']:.2f} %)"
    def ch(n):
        return (f"{r[n + '_pct']:+.2f} %{' exceeds' if r[n + '_exceeds_spread'] else ' within'} the spread"
                f"{'' if r[n + '_ranges_overlap'] else ', ranges apart'}{' FLAG' if r[n + '_flag'] else ''}")
    out.append(f"{r['program']} {r['version']} {r['interval']} [{r['bound']}]: old {side('old')} | A {side('A')} | B {side('B')} || "
               f"B/A {ch('B_vs_A')}; B/old {ch('B_vs_old')}; A/old {ch('A_vs_old')}")
flags = [r for r in table if r["B_vs_A_flag"] or r["B_vs_old_flag"] or r["A_vs_old_flag"]]
out.append(f"flags: {len(flags)} of {len(table)} numbers have a change above the threshold"
           + (": " + "; ".join(f"{r['program']} {r['version']} {r['interval']}" for r in flags) if flags else ""))

# clocks per run
readings = []
for r in csv.reader(open(os.path.join(args.window, "gpu_clocks.csv"))):
    if len(r) < 2: continue
    try:
        readings.append((datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f"), int(r[1].strip().split()[0])))
    except ValueError:
        continue
clock_lines.append("SM clock readings (nvidia-smi every 100 ms) per run, from its start to its end second: count, distinct MHz:count, the largest")
for prog, _, _ in PROGS:
    for d in sorted(glob.glob(os.path.join(args.window, prog, "*")), key=lambda p: (("warmup" not in p), p)):
        txt = open(os.path.join(d, "load.txt")).read()
        stt = datetime.datetime.fromisoformat(re.search(r"start (\S+?)(;|$)", txt, re.M).group(1)).replace(tzinfo=None)
        en = datetime.datetime.fromisoformat(re.search(r"end (\S+)", txt).group(1)).replace(tzinfo=None) + datetime.timedelta(seconds=1)
        rs = [c for t, c in readings if stt <= t < en]
        counts = ", ".join(f"{c}:{rs.count(c)}" for c in sorted(set(rs)))
        clock_lines.append(f"  {prog}/{os.path.basename(d)}: {len(rs)} readings; {counts}; largest {max(rs) if rs else '-'}")

# the check CSVs against #63's B quiet-run1-B
for prog, rec, _ in PROGS:
    ref_dir = os.path.join(args.records, "e18-retime-b10fc66", "window", prog, "quiet-run1-B")
    for d in sorted(glob.glob(os.path.join(args.window, prog, "*"))):
        for f in sorted(os.listdir(ref_dir)):
            if not f.endswith(".csv") or f in (f"{prog}.csv", "gpu-at-end.csv"): continue
            p = os.path.join(d, f)
            if not os.path.exists(p):
                check_lines.append(f"{prog}/{os.path.basename(d)}/{f}: MISSING"); continue
            a, b = data_rows(os.path.join(ref_dir, f)), data_rows(p)
            if a == b:
                check_lines.append(f"{prog}/{os.path.basename(d)}/{f}: {len(b)} data rows, equal to #63's B")
            else:
                diff = [i for i in range(max(len(a), len(b))) if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)]
                check_lines.append(f"{prog}/{os.path.basename(d)}/{f}: {len(a)} / {len(b)} data rows, {len(diff)} differ from #63's B:")
                for i in diff[:12]:
                    check_lines.append(f"    row {i + 1}: #63 B:  {a[i] if i < len(a) else '(none)'}")
                    check_lines.append(f"    row {i + 1}: here:   {b[i] if i < len(b) else '(none)'}")

text = "\n".join(out)
print(text); print(); print("\n".join(clock_lines)); print(); print("\n".join(check_lines))
if args.out: open(args.out, "w").write(text + "\n")
if args.clocks: open(args.clocks, "w").write("\n".join(clock_lines) + "\n")
if args.checks: open(args.checks, "w").write("\n".join(check_lines) + "\n")
if args.md:
    md = ["| program | version | interval | old (us) | A (us) | B (us) | B / A | B / old | A / old |", "|---|---|---|--:|--:|--:|--:|--:|--:|"]
    def cell(r, s):
        return f"{r[s + '_us']:.2f} ({r[s + '_min_us']:.2f}-{r[s + '_max_us']:.2f})"
    def chg(r, n):
        return (f"{r[n + '_pct']:+.2f} %" + (" **exceeds**" if r[n + "_exceeds_spread"] else "") + ("" if r[n + "_ranges_overlap"] else " (apart)")
                + (" **FLAG**" if r[n + "_flag"] else ""))
    for r in table:
        md.append(f"| `{r['program']}` | `{r['version']}` | {r['interval']} | {cell(r, 'old')} | {cell(r, 'A')} | {cell(r, 'B')} | "
                  f"{chg(r, 'B_vs_A')} | {chg(r, 'B_vs_old')} | {chg(r, 'A_vs_old')} |")
    open(args.md, "w").write("\n".join(md) + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(table)
