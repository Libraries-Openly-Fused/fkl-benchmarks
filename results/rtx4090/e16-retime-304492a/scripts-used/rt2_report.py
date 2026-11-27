#!/usr/bin/env python3
"""RT-2: the report of the window e16-retime-304492a (CPU only: reads CSVs; nothing is run).

  rt2_report.py <old e16-summary folder> <A e16-summary folder> <B e16-summary folder> --out report.md --csv numbers.csv

old  = window D's cleaned side, the paper's E16 data (results/rtx4090/cleanup-window-d/cleaned/e16-summary/: run1..3 with the joined
       *_clock.csv, and scripts/e16_summary.py's outputs);
A, B = this window's two sides, laid out the same way by rt2_summaries.sh.
1. Every timed row of the FKL program (e16_attention, 152 rows a run) and of the Python baselines (e16_baselines, 180 rows), keyed by
   (variant, d, seq, bh, causal) as window D's report keyed them: time_ms_median of each of the three rounds; per side the median, the min,
   the max and the spread ((max - min) / median, %). The changes B against A, A against old and B against old, in % of the medians; whether a
   change exceeds the larger spread of its two sides, and whether the two min-max ranges overlap. Bound: every E16 row is GPU-bound, as in
   window D's report (a kernel time of a call, with the inputs already on the GPU), so the flag is a change above 3 %. The SM clock of each
   row (sm_mhz_median of the _clock file, median over the rounds) is given per side. Rows without a time (variants the scripts skip, with
   their note) are listed apart.
2. The paper's summary: e16_summary_core.csv's fraction FKL mma_raw / best baseline (fraction_median [min-max]) and the best baseline, per
   configuration, old / A / B, with the change of the fraction; e16_q2_compact.csv's ratio separate / fused (ratio_separate_over_fused
   [min-max]), per work and configuration, old / A / B.
3. The error columns of the FKL program (max_abs_err, mean_abs_err): A against old, and B against A, round by round, row by row.
4. The order of the variants: the position column of each round, A and B against old.
No cause is named.
"""
import argparse, csv, os, statistics

ap = argparse.ArgumentParser()
ap.add_argument("old"); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--out", required=True); ap.add_argument("--csv", required=True)
args = ap.parse_args()
SIDES = {"old": args.old, "A": args.a, "B": args.b}
KEY = ("variant", "d", "seq", "bh", "causal")
NROWS = {"e16_attention": 152, "e16_baselines": 180}


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def runs(side, name):
    out = []
    for i in (1, 2, 3):
        p = os.path.join(SIDES[side], f"run{i}", f"{name}_clock.csv")
        rs = rows_of(p)
        out.append({tuple(r[k] for k in KEY): r for r in rs})
    return out


def stats(v):
    m = statistics.median(v)
    return dict(med=m, lo=min(v), hi=max(v), spread=(max(v) - min(v)) / m * 100.0 if m else 0.0)


def change(x, y):
    d = (x["med"] / y["med"] - 1) * 100.0
    sp = max(x["spread"], y["spread"])
    overlap = not (x["lo"] > y["hi"] or y["lo"] > x["hi"])
    return dict(pct=d, exceeds=abs(d) > sp, overlap=overlap, flag=abs(d) > 3.0)


def label(k):
    v, d, s, bh, c = k
    return f"D={d} S={s} BH={bh} {'causal' if c == '1' else 'full'} {v}"


rep = ["# RT-2, E16 at FKL b10fc66 (A) and 304492a (B) against window D's cleaned side (old, the paper's E16 data)", ""]
allrows, consistency = [], []
for name, title in (("e16_attention", "the FKL program"), ("e16_baselines", "the Python baselines")):
    sets = {s: runs(s, name) for s in SIDES}
    for s, rs in sets.items():
        for i, r in enumerate(rs, 1):
            if len(r) != NROWS[name]:
                consistency.append(f"{name} {s} run{i}: {len(r)} rows, not {NROWS[name]}")
    keys = sorted(sets["old"][0], key=lambda k: (int(k[1]), k[4], int(k[2]), k[0]))
    for s in ("A", "B"):
        for i, r in enumerate(sets[s], 1):
            if set(r) != set(sets["old"][0]):
                consistency.append(f"{name} {s} run{i}: the rows differ from old's: only here {sorted(set(r) - set(sets['old'][0]))[:4]}, missing {sorted(set(sets['old'][0]) - set(r))[:4]}")
    timed, untimed = [], []
    for k in keys:
        if any(k not in r for s in SIDES for r in sets[s]):
            continue
        vals = {s: [r[k]["time_ms_median"] for r in sets[s]] for s in SIDES}
        if any(v == "" for s in SIDES for v in vals[s]):
            untimed.append((label(k), sorted({r[k]["note"] for s in SIDES for r in sets[s] if not r[k]["time_ms_median"]})))
            continue
        st = {s: stats([float(v) for v in vals[s]]) for s in SIDES}
        clk = {s: statistics.median([float(r[k]["sm_mhz_median"]) for r in sets[s] if r[k].get("sm_mhz_median")] or [0]) for s in SIDES}
        row = dict(program=name, label=label(k), bound="GPU")
        for s in SIDES:
            row.update({f"{s}_ms": st[s]["med"], f"{s}_min_ms": st[s]["lo"], f"{s}_max_ms": st[s]["hi"], f"{s}_spread_pct": st[s]["spread"], f"{s}_sm_mhz": clk[s]})
        for n, x, y in (("B_vs_A", "B", "A"), ("A_vs_old", "A", "old"), ("B_vs_old", "B", "old")):
            c = change(st[x], st[y])
            row.update({f"{n}_pct": c["pct"], f"{n}_exceeds": c["exceeds"], f"{n}_overlap": c["overlap"], f"{n}_flag": c["flag"]})
        timed.append(row)
    allrows += timed
    rep += [f"## {title} ({name}): time per call, ms (median of the three rounds; min-max; spread)", "",
            "| configuration, variant | old | A | B | B / A | A / old | B / old | SM MHz old / A / B |", "|---|---|---|---|---|---|---|---|"]

    def cell(r, s):
        return f"{r[s + '_ms']:.4f} ({r[s + '_min_ms']:.4f}-{r[s + '_max_ms']:.4f}; {r[s + '_spread_pct']:.2f} %)"

    def chg(r, n):
        return (f"{r[n + '_pct']:+.2f} %" + (" **exceeds**" if r[n + "_exceeds"] else "") + ("" if r[n + "_overlap"] else " (apart)")
                + (" **FLAG**" if r[n + "_flag"] else ""))
    for r in timed:
        rep.append(f"| {r['label']} | {cell(r, 'old')} | {cell(r, 'A')} | {cell(r, 'B')} | {chg(r, 'B_vs_A')} | {chg(r, 'A_vs_old')} | {chg(r, 'B_vs_old')} | "
                   f"{r['old_sm_mhz']:.0f} / {r['A_sm_mhz']:.0f} / {r['B_sm_mhz']:.0f} |")
    rep.append("")
    for n in ("B_vs_A", "A_vs_old", "B_vs_old"):
        v = [r[n + "_pct"] for r in timed]
        fl = [r for r in timed if r[n + "_flag"]]
        rep.append(f"- {n.replace('_vs_', ' against ')}: {min(v):+.2f} % to {max(v):+.2f} %; {sum(r[n + '_exceeds'] for r in timed)} of {len(timed)} exceed the larger "
                   f"spread, {sum(not r[n + '_overlap'] for r in timed)} apart; flags (above 3 %): {len(fl)}"
                   + ("".join(f"\n  - {r['label']}: {r[n + '_pct']:+.2f} % (spreads {r[n.split('_vs_')[0] + '_spread_pct']:.2f} %, {r[n.split('_vs_')[1] + '_spread_pct']:.2f} %)" for r in fl)))
    if untimed:
        rep.append(f"- Not timed on any side (the scripts skip them): " + "; ".join(f"{l} ({', '.join(n)})" for l, n in untimed))
    rep.append("")

# 2. the paper's summary
def core(side):
    return {(r["d"], r["causal"], r["seq"], r["bh"]): r for r in csv.DictReader(open(os.path.join(SIDES[side], "e16_summary_core.csv")))}


C = {s: core(s) for s in SIDES}
rep += ["## The paper's summary: fraction FKL mma_raw / best baseline (e16_summary_core.csv; median [min-max] of the per-run fractions)", "",
        "| D | causal | S | BH | old | A | B | B - A | A - old | B - old |", "|---|---|---|---|---|---|---|---|---|---|"]
moves = []
for k in sorted(C["old"], key=lambda k: (int(k[0]), k[1], int(k[2]))):
    f = {s: float(C[s][k]["fraction_median"]) for s in SIDES}
    cell = {s: f"{f[s]:.4f} [{float(C[s][k]['fraction_min']):.4f}-{float(C[s][k]['fraction_max']):.4f}] {C[s][k]['best_baseline']}" for s in SIDES}
    rep.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {cell['old']} | {cell['A']} | {cell['B']} | {f['B'] - f['A']:+.4f} | {f['A'] - f['old']:+.4f} | {f['B'] - f['old']:+.4f} |")
    moves.append((abs(f["B"] - f["A"]), abs(f["A"] - f["old"]), abs(f["B"] - f["old"])))
rep += ["", f"- The fraction moves by at most {max(m[0] for m in moves):.4f} (B - A), {max(m[1] for m in moves):.4f} (A - old) and {max(m[2] for m in moves):.4f} (B - old).", ""]


def q2(side):
    return {(r["work"], r["d"], r["seq"], r["causal"]): r for r in csv.DictReader(open(os.path.join(SIDES[side], "e16_q2_compact.csv")))}


Q = {s: q2(s) for s in SIDES}
rep += ["## The paper's summary: ratio separate / fused (e16_q2_compact.csv; median [min-max])", "",
        "| work | D | S | causal | old | A | B |", "|---|---|---|---|---|---|---|"]
for k in Q["old"]:
    rep.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(
        f"{float(Q[s][k]['ratio_separate_over_fused']):.4f} [{float(Q[s][k]['ratio_min']):.4f}-{float(Q[s][k]['ratio_max']):.4f}]" if k in Q[s] else "(missing)" for s in SIDES) + " |")
rep.append("")

# 3. the error columns of the FKL program
E = {s: runs(s, "e16_attention") for s in SIDES}
for a, b in (("A", "old"), ("B", "A")):
    eq = tot = 0
    diff = []
    for i in range(3):
        for k, r in E[a][i].items():
            if k not in E[b][i]:
                continue
            tot += 1
            same = all(r[c] == E[b][i][k][c] for c in ("max_abs_err", "mean_abs_err"))
            eq += same
            if not same and len(diff) < 12:
                diff.append(f"run{i + 1} {label(k)}: {b} {E[b][i][k]['max_abs_err']} / {E[b][i][k]['mean_abs_err']}, {a} {r['max_abs_err']} / {r['mean_abs_err']}")
    rep.append(f"- FKL error columns (max_abs_err, mean_abs_err), {a} against {b}: equal in {eq} of {tot} rows (three rounds)" + ("".join(f"\n  - {d}" for d in diff)))
# 4. the order
for name in ("e16_attention", "e16_baselines"):
    S = {s: runs(s, name) for s in SIDES}
    for s in ("A", "B"):
        same = all(S[s][i][k]["position"] == S["old"][i][k]["position"] for i in range(3) for k in S["old"][i] if k in S[s][i])
        rep.append(f"- {name}: the position of every variant in every round, {s} against old: {'the same' if same else 'DIFFERENT'}")
rep += ["", "## Consistency", ""] + (consistency or ["- every run of every side has the expected rows (152 FKL, 180 Python) and the rows of old"])
open(args.out, "w").write("\n".join(rep) + "\n")
hdr = list(allrows[0].keys())
with open(args.csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(allrows)
print("\n".join(l for l in rep if l.startswith("- ")))
