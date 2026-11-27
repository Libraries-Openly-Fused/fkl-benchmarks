#!/usr/bin/env python3
"""RT-2b: the report of the window e16-query-tile-retime-304492a (CPU only: reads CSVs; nothing is run).

  rt2b_report.py <old folder> <A folder> <B folder> --out report.md --csv numbers.csv

old  = the record results/rtx4090/e16-query-tile/ (2026-11-04; run0..2/e16_query_tile.csv; the binary 763e7e3c..., FKL c8fcbe8, the program
       before the cleanup);
A, B = this window's two sides, each with run0..2/e16_query_tile.csv (copies of the window's runs, made by rt2b_after.sh).
1. The quantities of the printed sentence, per run and S, with the definitions of scripts/e16_query_tile_analyse.py:
   excess(BQ) = time(int8_bqBQ_w4) - time(raw_bqBQ_kv32_w4), the extra time of the fused int8 kernel over the bf16 kernel at the same tiles;
   ratio = excess(128) / excess(64); share(BQ) = (time(int8) - time(int8nodq)) / (time(int8) - time(raw)), the share of the extra time that
   the timing-only variant (the nodq copy: no conversion arithmetic and no scale wait) leaves out. Per side: the median of the three runs,
   the min, the max and the spread ((max - min) / |median|, %); the changes B against A, A against old and B against old, in % of the
   medians and as differences; whether a change exceeds the larger spread and whether the min-max ranges overlap; the flag above 3 % (every
   number is a GPU time or a ratio of GPU times: GPU-bound).
2. Every timed row (variant, S): time_ms_median, the same statistics and changes.
3. Checks: the rows of every run (variants and S); split_kv and the number of kernels of every row (the kernels that ran: no split-KV);
   the error columns (max_abs_err, mean_abs_err) of the rows that are not timing_only, A against old and B against A; the positions.
No cause is named.
"""
import argparse, csv, os, statistics

ap = argparse.ArgumentParser()
ap.add_argument("old"); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--out", required=True); ap.add_argument("--csv", required=True)
args = ap.parse_args()
SIDES = {"old": args.old, "A": args.a, "B": args.b}
BQ = {"64": ("raw_bq64_kv32_w4", "int8_bq64_w4", "int8nodq_bq64_w4"), "128": ("raw_bq128_kv32_w4", "int8_bq128_w4", "int8nodq_bq128_w4")}
SENTENCE_S = (2048, 4096, 8192)


def read(p):
    with open(p) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


RUNS = {s: [{(int(r["seq"]), r["variant"]): r for r in read(os.path.join(d, f"run{k}", "e16_query_tile.csv"))} for k in (0, 1, 2)] for s, d in SIDES.items()}


def t(run, s, v):
    r = run.get((s, v))
    return float(r["time_ms_median"]) if r and r["time_ms_median"] and float(r["time_ms_median"]) > 0 else None


def stats(v):
    m = statistics.median(v)
    return dict(med=m, lo=min(v), hi=max(v), spread=(max(v) - min(v)) / abs(m) * 100.0 if m else 0.0, vals=v)


def change(x, y):
    d = (x["med"] / y["med"] - 1) * 100.0
    return dict(pct=d, diff=x["med"] - y["med"], exceeds=abs(d) > max(x["spread"], y["spread"]),
                overlap=not (x["lo"] > y["hi"] or y["lo"] > x["hi"]), flag=abs(d) > 3.0)


def quantity(name, fn):
    st = {}
    for s in SIDES:
        vals = [fn(run) for run in RUNS[s]]
        if any(v is None for v in vals):
            return None
        st[s] = stats(vals)
    row = dict(quantity=name)
    for s in SIDES:
        row.update({f"{s}": st[s]["med"], f"{s}_min": st[s]["lo"], f"{s}_max": st[s]["hi"], f"{s}_spread_pct": st[s]["spread"],
                    f"{s}_runs": " / ".join(f"{v:.4f}" for v in st[s]["vals"])})
    for n, x, y in (("B_vs_A", "B", "A"), ("A_vs_old", "A", "old"), ("B_vs_old", "B", "old")):
        c = change(st[x], st[y])
        row.update({f"{n}_pct": c["pct"], f"{n}_diff": c["diff"], f"{n}_exceeds": c["exceeds"], f"{n}_overlap": c["overlap"], f"{n}_flag": c["flag"]})
    return row


seqs = sorted({k[0] for k in RUNS["old"][0]})
sentence, rows = [], []
for s in seqs:
    for bq, (raw, i8, nd) in BQ.items():
        q = quantity(f"S={s} excess BQ{bq} (ms)", lambda run, s=s, raw=raw, i8=i8: (t(run, s, i8) - t(run, s, raw)) if None not in (t(run, s, i8), t(run, s, raw)) else None)
        if q: (sentence if s in SENTENCE_S else rows).append(q)
    q = quantity(f"S={s} ratio excess BQ128 / BQ64", lambda run, s=s: ((t(run, s, "int8_bq128_w4") - t(run, s, "raw_bq128_kv32_w4")) / (t(run, s, "int8_bq64_w4") - t(run, s, "raw_bq64_kv32_w4"))))
    if q: (sentence if s in SENTENCE_S else rows).append(q)
    for bq, (raw, i8, nd) in BQ.items():
        q = quantity(f"S={s} share left out by the timing-only variant, BQ{bq}", lambda run, s=s, raw=raw, i8=i8, nd=nd: ((t(run, s, i8) - t(run, s, nd)) / (t(run, s, i8) - t(run, s, raw))))
        if q: (sentence if s in SENTENCE_S else rows).append(q)
times = []
for s in seqs:
    for v in sorted({k[1] for k in RUNS["old"][0]}):
        q = quantity(f"S={s} {v} (ms)", lambda run, s=s, v=v: t(run, s, v))
        if q: times.append(q)


def table(title, qs, digits):
    out = [f"## {title}", "", "| quantity | old (runs; spread) | A (runs; spread) | B (runs; spread) | B / A | A / old | B / old |", "|---|---|---|---|---|---|---|"]
    def cell(r, s):
        return f"{r[s]:.{digits}f} ({r[s + '_runs']}; {r[s + '_spread_pct']:.2f} %)"
    def chg(r, n):
        return (f"{r[n + '_pct']:+.2f} % ({r[n + '_diff']:+.{digits}f})" + (" **exceeds**" if r[n + "_exceeds"] else "") + ("" if r[n + "_overlap"] else " (apart)")
                + (" **FLAG**" if r[n + "_flag"] else ""))
    for r in qs:
        out.append(f"| {r['quantity']} | {cell(r, 'old')} | {cell(r, 'A')} | {cell(r, 'B')} | {chg(r, 'B_vs_A')} | {chg(r, 'A_vs_old')} | {chg(r, 'B_vs_old')} |")
    out.append("")
    for n in ("B_vs_A", "A_vs_old", "B_vs_old"):
        v = [r[n + "_pct"] for r in qs]; fl = [r for r in qs if r[n + "_flag"]]
        out.append(f"- {n.replace('_vs_', ' against ')}: {min(v):+.2f} % to {max(v):+.2f} %; {sum(r[n + '_exceeds'] for r in qs)} of {len(qs)} exceed the larger spread, "
                   f"{sum(not r[n + '_overlap'] for r in qs)} apart; flags (above 3 %): {len(fl)}" + "".join(f"\n  - {r['quantity']}: {r[n + '_pct']:+.2f} %" for r in fl))
    return out + [""]


rep = ["# RT-2b, the E16 query tile at FKL b10fc66 (A) and 304492a (B) against the record of 2026-11-04 (old)", ""]
rep += table("The printed sentence's quantities, S = 2,048, 4,096, 8,192 (D = 128, non-causal)", sentence, 4)
rep += table("The same quantities at S = 512 and 1,024 (not in the sentence)", rows, 4)
rep += table("Every timed row: time_ms_median (ms)", times, 4)
# checks
chk = []
keys0 = set(RUNS["old"][0])
for s in SIDES:
    for k, run in enumerate(RUNS[s]):
        if set(run) != keys0:
            chk.append(f"- {s} run{k}: the rows differ from old's run0: only here {sorted(set(run) - keys0)[:4]}, missing {sorted(keys0 - set(run))[:4]}")
        split = {r["split_kv"] for r in run.values()}; kern = {r["kernels"] for r in run.values()}
        chk.append(f"- {s} run{k}: {len(run)} rows; split_kv {sorted(split)}; kernels per call {sorted(kern)}")
for a, b in (("A", "old"), ("B", "A")):
    eq = tot = 0; diff = []
    for k in range(3):
        for key, r in RUNS[a][k].items():
            o = RUNS[b][k].get(key)
            if not o or r.get("timing_only") == "1":
                continue
            tot += 1
            same = r["max_abs_err"] == o["max_abs_err"] and r["mean_abs_err"] == o["mean_abs_err"]
            eq += same
            if not same and len(diff) < 8:
                diff.append(f"run{k} S={key[0]} {key[1]}: {b} {o['max_abs_err']} / {o['mean_abs_err']}, {a} {r['max_abs_err']} / {r['mean_abs_err']}")
    chk.append(f"- error columns of the rows that are not timing_only, {a} against {b}: equal in {eq} of {tot}" + "".join(f"\n  - {d}" for d in diff))
for s in ("A", "B"):
    same = all(RUNS[s][k][key]["position"] == RUNS["old"][k][key]["position"] for k in range(3) for key in RUNS["old"][k] if key in RUNS[s][k])
    chk.append(f"- the position of every variant in every run, {s} against old: {'the same' if same else 'DIFFERENT'}")
rep += ["## Checks", ""] + chk
open(args.out, "w").write("\n".join(rep) + "\n")
allq = sentence + rows + times
with open(args.csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(allq[0].keys())); w.writeheader(); w.writerows(allq)
print("\n".join(l for l in rep if l.startswith("- ") or l.startswith("  - ")))
