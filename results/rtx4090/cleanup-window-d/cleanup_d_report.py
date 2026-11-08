#!/usr/bin/env python3
"""Report of the cleanup window D (2026-11-08): E16 (the FKL program and the Python baselines) published / reference-now / cleaned, with run-to-run
spreads, drift (reference-now against published), cleanup effect (cleaned against reference-now), the 3 % (GPU-bound) flag, the SM clock of the rows,
the paper's E16 summary (FKL / best baseline), the order of the variants in every round, the preparation column and the error columns; E2 at k = 5002
(the FKL program on both builds, the cleaned Python driver against the published large runs); the SM clock of every run; the E1 tables of the cleaned
build from the mangled ncu pass. Reads only (the clock joins and E16 summaries in {reference,cleaned}/e16-summary/ are made before by scripts/e16_join_clock.py and
scripts/e16_summary.py); writes the markdown report given as the first argument and a CSV of every compared number next to it."""
import bisect
import csv
import datetime
import os
import re
import statistics
import sys

R = "/home/oscar/tfkl-bench/fkl-benchmarks/results/rtx4090"
D = R + "/cleanup-window-d"
C = R + "/cleanup-window-c"
PUB16 = R + "/e16-attention"
OUT = sys.argv[1]
ALL_ROWS = []


def rows_of(path, expect=None):
    lines = [l for l in open(path) if l.strip() and not l.startswith("#")]
    table = list(csv.reader(lines))
    head = [c.strip() for c in table[0]]
    body = table[1:]
    bad = [r for r in body if len(r) != len(head)]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} rows with a field count other than the header's")
    out = [dict(zip(head, [c.strip() for c in r])) for r in body]
    if expect is not None and len(out) != expect:
        raise SystemExit(f"{path}: {len(out)} data rows, expected {expect}")
    if any("INCOMPLETE" in l for l in open(path)):
        raise SystemExit(f"{path}: INCOMPLETE")
    return out


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    med = statistics.median(values)
    spread = (max(values) - min(values)) / med * 100 if len(values) > 1 and med else None
    return med, spread, len(values)


def pct(a, b):
    return (a - b) / b * 100 if (a is not None and b not in (None, 0)) else None


def fmt(v, digits=4):
    if v is None:
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}g}"


def fp(v):
    return "-" if v is None else f"{v:+.1f} %"


def fs(v):
    return "one run" if v is None else f"{v:.1f} %"


def judge(change, spreads, bound):
    if change is None:
        return "-", False
    known = [s for s in spreads if s is not None]
    exceeds = bool(known) and abs(change) > max(known)
    limit = 5.0 if bound == "host" else 3.0
    flag = abs(change) > limit
    return ("exceeds" if exceeds else "within") + (" FLAG" if flag else ""), flag and exceeds


def section(title, key_name, numbers, causes, extra_head="", extra=None):
    """numbers: (label, bound, pub, ref, new); extra(label) -> text of an extra column (the SM clocks)"""
    lines = [f"### {title}", "",
             f"| {key_name} | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | {extra_head + ' | ' if extra_head else ''}cause |",
             "|---|---|---|---|---|---|---|---|---|" + ("---|" if extra_head else "") + "---|"]
    flagged = []
    for label, bound, pv, rv, nv in numbers:
        pm, ps, pn = stats(pv)
        rm, rs, rn = stats(rv)
        nm, ns, nn = stats(nv)
        drift, effect = pct(rm, pm), pct(nm, rm)
        dj, dflag = judge(drift, [ps, rs], bound)
        ej, eflag = judge(effect, [rs, ns], bound)
        cause = "; ".join(c for c in ((causes(label, "drift") if dflag else ""), (causes(label, "cleanup") if eflag else "")) if c)
        ex = f"{extra(label)} | " if extra else ""
        lines.append(f"| {label} | {bound} | {fmt(pm)} ({fs(ps)}) | {fmt(rm)} ({fs(rs)}) | {fmt(nm)} ({fs(ns)}) | {fp(drift)} | {dj} | {fp(effect)} | {ej} | {ex}{cause} |")
        ALL_ROWS.append([title, label, bound, pm, ps, pn, rm, rs, rn, nm, ns, nn, drift, dj, effect, ej, cause])
        if dflag:
            flagged.append((title, "drift", label, bound, f"{fp(drift)} (published {fmt(pm)}, reference now {fmt(rm)}; spreads {fs(ps)}, {fs(rs)})", causes(label, "drift")))
        if eflag:
            flagged.append((title, "cleanup", label, bound, f"{fp(effect)} (reference now {fmt(rm)}, cleaned {fmt(nm)}; spreads {fs(rs)}, {fs(ns)})", causes(label, "cleanup")))
    return lines + [""], flagged


def section2(title, key_name, numbers, causes):
    lines = [f"### {title}", "",
             f"| {key_name} | bound | published (spread) | cleaned (spread) | change cleaned vs pub | | cause |", "|---|---|---|---|---|---|---|"]
    flagged = []
    for label, bound, pv, nv in numbers:
        pm, ps, pn = stats(pv)
        nm, ns, nn = stats(nv)
        change = pct(nm, pm)
        j, flag = judge(change, [ps, ns], bound)
        cause = causes(label, "change") if flag else ""
        lines.append(f"| {label} | {bound} | {fmt(pm)} ({fs(ps)}) | {fmt(nm)} ({fs(ns)}) | {fp(change)} | {j} | {cause} |")
        ALL_ROWS.append([title, label, bound, pm, ps, pn, None, None, 0, nm, ns, nn, None, "", change, j, cause])
        if flag:
            flagged.append((title, "change", label, bound, f"{fp(change)} (published {fmt(pm)}, cleaned {fmt(nm)}; spreads {fs(ps)}, {fs(ns)})", cause))
    return lines + [""], flagged


def bound_of(normal, kernel):
    return "host" if (kernel is not None and normal and kernel < 0.8 * normal) else "GPU"


# ---------------------------------------------------------------- the clock sampler
SAMPLES = []   # (epoch ms, sm MHz, power W, temperature C, pstate, utilisation %)
for line in open(f"{D}/gpu_clocks.csv"):
    f = [x.strip() for x in line.split(",")]
    if len(f) < 7:
        continue
    t = datetime.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").astimezone()
    SAMPLES.append((t.timestamp() * 1000, float(f[1].split()[0]), float(f[5].split()[0]), float(f[6]), f[3], float(f[4].split()[0])))
SAMPLES.sort()
STIMES = [s[0] for s in SAMPLES]


def clock_span(t0, t1):
    lo, hi = bisect.bisect_left(STIMES, t0), bisect.bisect_right(STIMES, t1)
    return SAMPLES[lo:hi]


def run_span(folder):
    """(start, end) epoch ms from a run folder's load.txt"""
    txt = open(f"{folder}/load.txt").read()
    a = re.search(r"start (\S+)", txt).group(1)
    b = re.search(r"end (\S+)", txt).group(1)
    return datetime.datetime.fromisoformat(a).timestamp() * 1000, datetime.datetime.fromisoformat(b).timestamp() * 1000


report = ["# Cleanup window D, 2026-11-08: E16, E2 at k = 5,002 and the E1 tables, published, reference now and cleaned", ""]
flags_all = []

# ---------------------------------------------------------------- E16, the FKL program
KEY = ("variant", "d", "seq", "bh", "causal")


def e16_sets(kind):
    name = "e16_attention" if kind == "fkl" else "e16_baselines"
    sub = "e16-attention" if kind == "fkl" else "e16-baselines"
    n = 152 if kind == "fkl" else 180
    pub = [{tuple(r[k] for k in KEY): r for r in rows_of(f"{PUB16}/run{i}/{name}_clock.csv", n)} for i in (1, 2, 3)]
    ref = [{tuple(r[k] for k in KEY): r for r in rows_of(f"{D}/reference/e16-summary/run{i}/{name}_clock.csv", n)} for i in (1, 2, 3)]
    new = [{tuple(r[k] for k in KEY): r for r in rows_of(f"{D}/cleaned/e16-summary/run{i}/{name}_clock.csv", n)} for i in (1, 2, 3)]
    # the joined files must be the window's rows plus the clock columns: same keys and same times as the originals
    for side, sets in (("reference", ref), ("cleaned", new)):
        for i, s in enumerate(sets, 1):
            orig = {tuple(r[k] for k in KEY): r for r in rows_of(f"{D}/{side}/{sub}/run{i}/{name}.csv", n)}
            assert set(orig) == set(s) and all(orig[k]["time_ms_median"] == s[k]["time_ms_median"] for k in orig), (side, i, kind)
    return pub, ref, new


def label_of(k):
    v, d, s, bh, c = k
    return f"D={d} S={s} {'causal' if c == '1' else 'full'} {v}"


def clock_text(sets_by_side):
    def f(label):
        k = LABEL_KEY[label]
        out = []
        for side in ("published", "reference", "cleaned"):
            vals = [float(x[k]["sm_mhz_median"]) for x in sets_by_side[side] if x[k].get("sm_mhz_median")]
            out.append(f"{statistics.median(vals):.0f}" if vals else "-")
        return " / ".join(out)
    return f


LABEL_KEY = {}
SKIPPED = {}
E16_FLAG_CAUSES = {}
for kind, title in (("fkl", "E16, the FKL program (e16_attention), time per call (ms; median of the blocks; median of the three rounds)"),
                    ("py", "E16, the Python baselines (e16_baselines.py), time per call (ms; median of the blocks; median of the three rounds; the reference script makes its own torch.randn inputs, the cleaned one reads the FKL input files)")):
    pub, ref, new = e16_sets(kind)
    numbers, skipped = [], []
    for k in sorted(new[0], key=lambda k: (int(k[1]), k[4], int(k[2]), k[0])):
        LABEL_KEY[label_of(k)] = k
        if any(not x[k]["time_ms_median"] for x in pub + ref + new):
            skipped.append((label_of(k), sorted({x[k]["note"] for x in pub + ref + new if not x[k]["time_ms_median"]})))
            continue
        numbers.append((label_of(k), "GPU", [float(x[k]["time_ms_median"]) for x in pub], [float(x[k]["time_ms_median"]) for x in ref], [float(x[k]["time_ms_median"]) for x in new]))
    SKIPPED[kind] = skipped
    lines, fl = section(title, "configuration, variant", numbers, lambda label, kind_: E16_FLAG_CAUSES.get((label, kind_), "not examined"),
                        "SM MHz pub / ref / new", clock_text({"published": pub, "reference": ref, "cleaned": new}))
    report += lines
    if SKIPPED[kind]:
        report += [f"Not timed on any side (the script skips them): " + "; ".join(f"{l} ({', '.join(n)})" for l, n in SKIPPED[kind]), ""]
    flags_all += fl
    if kind == "fkl":
        FKL_SETS = (pub, ref, new)
    else:
        PY_SETS = (pub, ref, new)

# ---------------------------------------------------------------- E16, the paper's summary (scripts/e16_summary.py on each side)
def core(p):
    return {(r["d"], r["causal"], r["seq"]): r for r in csv.DictReader(open(p))}


CP, CR, CN = core(f"{PUB16}/e16_summary_core.csv"), core(f"{D}/reference/e16-summary/e16_summary_core.csv"), core(f"{D}/cleaned/e16-summary/e16_summary_core.csv")
report += ["### E16, the paper's summary: FKL mma_raw against the best baseline (scripts/e16_summary.py on each side: TFLOP/s; fraction FKL / best per run, median [smallest-largest])", "",
           "| D | mask | S | mma_raw TFLOP/s pub / ref / new | best baseline pub / ref / new | fraction pub | fraction ref | fraction new | clocks pub / ref / new |",
           "|---|---|---|---|---|---|---|---|---|"]
FRACTION_MOVES = []
for k in sorted(CN, key=lambda k: (int(k[0]), k[1], int(k[2]))):
    p, r, n = CP[k], CR[k], CN[k]
    def fr(x):
        return f"{float(x['fraction_median']):.3f} [{float(x['fraction_min']):.3f}-{float(x['fraction_max']):.3f}]"
    report.append(f"| {k[0]} | {'causal' if k[1] == '1' else 'full'} | {k[2]} | {float(p['mma_raw_tflops']):.1f} / {float(r['mma_raw_tflops']):.1f} / {float(n['mma_raw_tflops']):.1f} | "
                  f"{p['best_baseline']} / {r['best_baseline']} / {n['best_baseline']} | {fr(p)} | {fr(r)} | {fr(n)} | {p['different_clocks'] or '-'} / {r['different_clocks'] or '-'} / {n['different_clocks'] or '-'} |")
    if abs(float(n["fraction_median"]) - float(p["fraction_median"])) > 0.02:
        FRACTION_MOVES.append(f"D={k[0]} {'causal' if k[1] == '1' else 'full'} S={k[2]}: published {fr(p)} ({p['best_baseline']}), reference now {fr(r)} ({r['best_baseline']}), cleaned {fr(n)} ({n['best_baseline']})")
report.append("")

# ---------------------------------------------------------------- the order of the variants in every round
ABBR = {"mma_raw": "raw", "mma_int8kv_fused": "i8F", "mma_int8kv_unfused": "i8U", "mma_fp8kv_fused": "f8F", "mma_fp8kv_unfused": "f8U",
        "mma_epilogue_fused": "epF", "mma_epilogue_separate": "epS", "simt_raw": "simt"}
report += ["### E16: the order of the variants in every round (the position column; round r ran with E16_ROTATE = r - 1 and --rotate r - 1)", "",
           "Program order in each round: round 1 FKL reference, FKL cleaned, Python reference, Python cleaned; round 2 Python cleaned, Python reference, "
           "FKL cleaned, FKL reference; round 3 as round 1.", ""]
same_order = {}
for kind, sets, abbr in (("FKL", FKL_SETS, ABBR), ("Python", PY_SETS, None)):
    pub, ref, new = sets
    configs = sorted({k[1:] for k in new[0]}, key=lambda c: (int(c[0]), c[3], int(c[1])))
    for side, ss in (("published", pub), ("reference", ref)):
        same_order[(kind, side)] = all({k: x[k]["position"] for k in x} == {k: y[k]["position"] for k in y} for x, y in zip(ss, new))
    report += [f"{kind} variant order per configuration (the same in the reference and the cleaned runs of a round: {same_order[(kind, 'reference')]}; the same as the "
               f"published round of the same index: {same_order[(kind, 'published')]}):", "",
               "| configuration | round 1 | round 2 | round 3 |", "|---|---|---|---|"]
    for c in configs:
        cells = []
        for x in new:
            order = sorted((int(x[k]["position"]), k[0]) for k in x if k[1:] == c)
            cells.append(" ".join((abbr or {}).get(v, v) for _, v in order))
        d, s, bh, causal = c
        report.append(f"| D={d} S={s} {'causal' if causal == '1' else 'full'} | " + " | ".join(cells) + " |")
    report.append("")

# ---------------------------------------------------------------- preparation and errors
pub, ref, new = FKL_SETS
report += ["### E16, the FKL program: preparation of one call (cleaned build; prep_us_per_call, host time of building one argument set and releasing the old one, us; median of the three rounds)", "",
           "| variant | smallest | median | largest (over all configurations) |", "|---|---|---|---|"]
for v in sorted({k[0] for k in new[0]}):
    vals = [statistics.median(float(x[k]["prep_us_per_call"]) for x in new) for k in new[0] if k[0] == v]
    report.append(f"| {v} | {min(vals):.4g} | {statistics.median(vals):.4g} | {max(vals):.4g} |")
report.append("")
eq = sum(1 for k in new[0] if all((x[k]["max_abs_err"], x[k]["mean_abs_err"]) == (y[k]["max_abs_err"], y[k]["mean_abs_err"]) for x, y in zip(ref, new)))
eqp = sum(1 for k in new[0] if all((x[k]["max_abs_err"], x[k]["mean_abs_err"]) == (y[k]["max_abs_err"], y[k]["mean_abs_err"]) for x, y in zip(pub, new)))
ERRORS = (f"E16 FKL error columns (max_abs_err, mean_abs_err): equal between the reference and the cleaned run of every round in {eq} of {len(new[0])} rows; "
          f"equal to the published rows in {eqp} of {len(new[0])}")
ppub, pref, pnew = PY_SETS
eqpy = sum(1 for k in pref[0] if all((x[k]["max_abs_err"], x[k]["mean_abs_err"]) == (y[k]["max_abs_err"], y[k]["mean_abs_err"]) for x, y in zip(ppub, pref)))
ERRORS_PY = f"E16 Python error columns: the reference script's equal to the published rows in {eqpy} of {len(pref[0])} (the same script and seeds); the cleaned script reads the FKL inputs and checks the heads FKL checks"

# ---------------------------------------------------------------- E2 at k = 5002, the FKL program
def by_ops(dirs, fname, expect):
    return [{int(r["ops"]): r for r in rows_of(os.path.join(d, fname), expect)} for d in dirs]


PUB_E2 = [f"{R}/e2-chain", f"{R}/e2-chain-run3", f"{R}/e2-chain-run4"]
FP = by_ops(PUB_E2, "e2_fkl_chain.csv", 6)
FR = by_ops([f"{D}/reference/e2-fkl/run{i}" for i in (1, 2, 3)], "e2_fkl_chain.csv", 6)
FN = by_ops([f"{D}/cleaned/e2-fkl/run{i}" for i in (1, 2, 3)], "e2_fkl_chain.csv", 6)
FRC = by_ops([f"{C}/reference/e2-fkl/run{i}" for i in (1, 2, 3)], "e2_fkl_chain.csv", 6)
FNC = by_ops([f"{C}/cleaned/e2-fkl/run{i}" for i in (1, 2, 3)], "e2_fkl_chain.csv", 6)
numbers = []
for k in sorted(FN[0]):
    nv = [float(x[k]["fkl_mean_ms"]) for x in FN]
    kv = [float(x[k]["fkl_kernel_mean_ms"]) for x in FN]
    numbers.append((f"{k} warm", bound_of(statistics.median(nv), statistics.median(kv)), [float(x[k]["fkl_mean_ms"]) for x in FP], [float(x[k]["fkl_mean_ms"]) for x in FR], nv))
    numbers.append((f"{k} L2-flushed", "GPU", [float(x[k]["fkl_flushed_mean_ms"]) for x in FP], [float(x[k]["fkl_flushed_mean_ms"]) for x in FR], [float(x[k]["fkl_flushed_mean_ms"]) for x in FN]))
E2_CAUSES = {("5002 L2-flushed", "cleanup"): (
    "order of the passes, as in window C (-3.0 % there): the cleaned flushed pass follows the kernel-only pass of the same k (100 spins of 5 ms), the "
    "reference's follows the warm pass; the clock log does not resolve it: every E2 FKL run of both builds samples 2,670-2,685 MHz, and the k = 5,002 flushed "
    "pass lasts about 80 ms, about one 100 ms sample; the power medians differ (reference 126-196 W, cleaned 112-113 W: most of a cleaned run is the one-thread "
    "spin); not isolated")}
lines, fl = section("E2, the FKL program (e2_fkl_chain), GPU time per launch (ms; mean of 100; published: runs 2, 3, 4; window D)", "k, pass", numbers,
                    lambda label, kind_: E2_CAUSES.get((label, kind_), "not examined"))
report += lines
flags_all += fl
report += ["### E2, the FKL program in windows C and D (the same binaries; medians of three runs each, ms)", "",
           "| k | warm ref C / D | warm cleaned C / D | flushed ref C / D | flushed cleaned C / D | kernel-only cleaned C / D |", "|---|---|---|---|---|---|"]
for k in sorted(FN[0]):
    def m(sets, col):
        return statistics.median(float(x[k][col]) for x in sets)
    report.append(f"| {k} | {fmt(m(FRC, 'fkl_mean_ms'))} / {fmt(m(FR, 'fkl_mean_ms'))} | {fmt(m(FNC, 'fkl_mean_ms'))} / {fmt(m(FN, 'fkl_mean_ms'))} | "
                  f"{fmt(m(FRC, 'fkl_flushed_mean_ms'))} / {fmt(m(FR, 'fkl_flushed_mean_ms'))} | {fmt(m(FNC, 'fkl_flushed_mean_ms'))} / {fmt(m(FN, 'fkl_flushed_mean_ms'))} | "
                  f"{fmt(m(FNC, 'fkl_kernel_mean_ms'))} / {fmt(m(FN, 'fkl_kernel_mean_ms'))} |")
report.append("")
hashes = {}
for side in ("reference", "cleaned"):
    for i in (1, 2, 3):
        hashes[(side, i)] = sorted(l.split()[0] + " " + l.split()[1] for l in open(f"{D}/{side}/e2-fkl/run{i}/outputs-sha256.txt") if l.strip())
c_hash = sorted(l.split()[0] + " " + l.split()[1] for l in open(f"{C}/cleaned/e2-fkl/run1/outputs-sha256.txt") if l.strip())
E2_HASHES = f"E2 FKL outputs of window D: the six runs give the same six hashes: {len({tuple(h) for h in hashes.values()}) == 1}; equal to window C's (and so to the published outputs): {hashes[('reference', 1)] == c_hash}"

# ---------------------------------------------------------------- E2 at k = 5002, the Python driver
def py_rows(dirs, fname, expect):
    return [{(int(r["ops"]), r["version"]): r for r in rows_of(os.path.join(d, fname), expect)} for d in dirs]


PP = py_rows(PUB_E2, "e2_chain_large.csv", 7)
PN = py_rows([f"{D}/cleaned/e2-python-k5002/run{i}" for i in (1, 2, 3)], "e2_chain.csv", 7)
VERSIONS = ("triton_hand", "inductor_default", "inductor_wide", "inductor_graphs", "eager", "eager_graphs")
numbers, ko = [], []
for v in VERSIONS:
    k = (5002, v)
    nv = [float(x[k]["gpu_mean_ms"]) for x in PN]
    kv = [float(x[k]["gpu_kernel_mean_ms"]) for x in PN]
    numbers.append((f"5002 {v} warm", bound_of(statistics.median(nv), statistics.median(kv)), [float(x[k]["gpu_mean_ms"]) for x in PP], nv))
    numbers.append((f"5002 {v} L2-flushed", "GPU", [float(x[k]["gpu_flushed_mean_ms"]) for x in PP if x[k].get("gpu_flushed_mean_ms")], [float(x[k]["gpu_flushed_mean_ms"]) for x in PN]))
    numbers.append((f"5002 {v} host_launch_us", "host", [float(x[k]["host_launch_us"]) for x in PP], [float(x[k]["host_launch_us"]) for x in PN]))
    ko.append((v, stats(nv), stats(kv)))
PY_CAUSES = {
    "5002 inductor_wide host_launch_us": ("drift, as in window C (+5 to +10 % on six host numbers there): the code of these numbers is unchanged since the published "
                                          "runs (the cleaned script adds only the kernel-only pass, after them); this window pinned the driver to cores 0-3 and 16-19, "
                                          "the published runs' pinning is not recorded; not isolated"),
    "5002 eager warm": ("drift, as in window C (-3.0 to -3.7 % at k = 100 and 500): unchanged code; eager enqueues 5,004 PyTorch kernels per call; the SM clock of "
                        "the three runs is 2,685 MHz (median of each run; the published runs have no clock log); not isolated"),
    "5002 eager L2-flushed": "as the warm number of eager"}
lines, fl = section2("E2 at k = 5,002, the Python driver (cleaned script only; published: the large runs 2, 3, 4, e2_chain_large.csv)", "k, version, column", numbers,
                     lambda label, kind_: PY_CAUSES.get(label, "not examined"))
report += lines
flags_all += fl
fk = (5002, "fkl")
fkl_n = stats([float(x[fk]["gpu_mean_ms"]) for x in PN])
fkl_k = stats([float(x[fk]["gpu_kernel_mean_ms"]) for x in PN])
fkl_f = stats([float(x[fk]["gpu_flushed_mean_ms"]) for x in PN])
report += ["### E2 at k = 5,002: warm, kernel-only and flushed of every version (cleaned; medians of the three runs, ms; FKL from its program, run n with Python run n)", "",
           "| version | warm (spread) | kernel-only (spread) | kernel-only / warm | L2-flushed (spread) |", "|---|---|---|---|---|"]
for v, n_, k_ in ko + [("fkl", fkl_n, fkl_k)]:
    f_ = stats([float(x[(5002, v)]["gpu_flushed_mean_ms"]) for x in PN])
    report.append(f"| {v} | {fmt(n_[0])} ({fs(n_[1])}) | {fmt(k_[0])} ({fs(k_[1])}) | {fmt(k_[0] / n_[0], 3)} | {fmt(f_[0])} ({fs(f_[1])}) |")
report.append("")
ACC = ("kernels", "max_ulp_vs_exact", "bitexact_fraction_vs_triton_hand", "bitexact_fraction_vs_fkl", "max_ulp_vs_fkl", "sass_ffma", "sass_fmul", "sass_fadd", "status")
diff = [(key, c, r[c], sorted({p[key][c] for p in PP})) for x in PN for key, r in x.items() for c in ACC if r[c] not in {p[key][c] for p in PP}]
PY_ACC = f"E2 k = 5,002 Python accuracy, kernel and SASS columns ({', '.join(ACC)}) against the published large runs: " + ("all equal" if not diff else f"{len(diff)} differences: {diff[:6]}")

# ---------------------------------------------------------------- the SM clock of every run
report += ["## The SM clock of every run (the 100 ms sampler, between the run's start and end)", "",
           "| run | start | s | SM MHz median | SM MHz min | SM MHz max | power W median | power W max | temperature max | pstates |", "|---|---|---|---|---|---|---|---|---|---|"]
RUNS = []
for side in ("reference", "cleaned"):
    for sub in sorted(os.listdir(f"{D}/{side}")):
        p = f"{D}/{side}/{sub}"
        if not os.path.isdir(p):
            continue
        for run in sorted(os.listdir(p)):
            if os.path.exists(f"{p}/{run}/load.txt"):
                RUNS.append((run_span(f"{p}/{run}"), f"{side}/{sub}/{run}"))
for (t0, t1), name in sorted(RUNS):
    s = clock_span(t0, t1)
    if not s:
        report.append(f"| {name} | - | - | no sample | | | | | | |")
        continue
    sm = [x[1] for x in s]; pw = [x[2] for x in s]
    report.append(f"| {name} | {datetime.datetime.fromtimestamp(t0 / 1000).strftime('%H:%M:%S')} | {(t1 - t0) / 1000:.0f} | {statistics.median(sm):.0f} | {min(sm):.0f} | {max(sm):.0f} | "
                  f"{statistics.median(pw):.0f} | {max(pw):.0f} | {max(x[3] for x in s):.0f} | {' '.join(sorted({x[4] for x in s}))} |")
report.append("")

# ---------------------------------------------------------------- E1: the tables of the cleaned build
report += ["## E1: the tables of the cleaned build (the mangled ncu pass of this window, window C's cleaned binaries and run 1's timings)", ""]
STATIC = ["sass_instructions", "FFMA", "FMUL", "FADD", "LDG", "STG", "LDL", "STL", "const_bank_operands", "IMAD", "ISETP", "BRA_etc", "barriers", "registers", "stack_bytes",
          "local_bytes", "cmem0_bytes", "code_bytes", "block_threads", "grid", "theoretical_occupancy_pct", "active_blocks_per_sm", "theoretical_occupancy_ncu_pct"]
for exp, pubdir in (("e1-vf-chain", "e1-vf-chain-v2"), ("e1-hf-batch", "e1-hf-batch-v2"), ("e1-preproc", "e1-preproc-v3")):
    a = {(r["point"], r["version"]): r for r in csv.DictReader(open(f"{R}/{pubdir}/table.csv"))}
    b = {(r["point"], r["version"]): r for r in csv.DictReader(open(f"{D}/cleaned/e1-ncu/{exp}/table.csv"))}
    nd = sum(1 for k in a if k in b for c in STATIC if a[k][c] != b[k][c])
    occ = max(abs(float(a[k]["achieved_occupancy_pct"]) - float(b[k]["achieved_occupancy_pct"])) for k in a if k in b)
    report.append(f"- {exp}: {len(b)} kernels (published {len(a)}, the same points and versions: {set(a) == set(b)}); static columns (SASS counts, registers, stack, local, "
                  f"cmem0, code bytes, block, grid, theoretical occupancy) different from the published table: {nd}; achieved occupancy within {occ:.2f} points")
pc = {(r["exp"], r["point"]): r for r in csv.DictReader(open(f"{R}/e1_compact.csv"))}
nc = {(r["exp"], r["point"]): r for r in csv.DictReader(open(f"{D}/cleaned/e1_compact.csv"))}
same_static = all(pc[k][c] == nc[k][c] for k in pc for c in ("regs_fkl", "regs_ref", "sass_fkl", "sass_ref", "occ_fkl", "occ_ref"))
report += [f"- e1_compact.csv of the cleaned build (`cleaned/e1_compact.csv`): registers, SASS and occupancy columns equal to the published `../e1_compact.csv`: {same_static}; the timing columns:", "",
           "| exp, point | ms_fkl pub / new | vs_ref pub / new | vs_best pub / new | vs_ref lo-hi pub / new |", "|---|---|---|---|---|"]
for k in pc:
    p, n = pc[k], nc[k]
    report.append(f"| {k[0]} {k[1]} | {p['ms_fkl']} / {n['ms_fkl']} | {p['vs_ref']} / {n['vs_ref']} | {p['vs_best']} / {n['vs_best']} | {p['vs_ref_lo']}-{p['vs_ref_hi']} / {n['vs_ref_lo']}-{n['vs_ref_hi']} |")
report.append("")

# ---------------------------------------------------------------- agreement
report += ["## Agreement", "", f"- {ERRORS}", f"- {ERRORS_PY}", f"- {E2_HASHES}", f"- {PY_ACC}", ""]


def short(title):
    return title.split(" (")[0].replace("E2, the FKL program", "E2 FKL program")


groups = {}
for title, kind, label, bound, numbers_text, cause in flags_all:
    groups.setdefault((short(title), kind, cause), []).append(f"{label} ({bound}) {numbers_text}")
flag_lines = [f"- **{t}, {kind}**, {len(items)} number{'s' if len(items) > 1 else ''}: " + "; ".join(items) + f". Cause: {cause}" for (t, kind, cause), items in groups.items()]
FINDINGS = [l.rstrip("\n") for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleanup_d_findings.md"))] if os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleanup_d_findings.md")) else []
report[2:2] = (FINDINGS + [""] if FINDINGS else []) + ["## Changes above the spread and above the 3 % (GPU) or 5 % (host) threshold", "", f"{len(flags_all)} numbers of {len(ALL_ROWS)} compared:", ""] + (flag_lines or ["- none"]) + [""] + \
    ["## E16: the paper's fraction FKL / best baseline, configurations where it moves by more than 0.02 (published against cleaned)", ""] + ([f"- {m}" for m in FRACTION_MOVES] or ["- none"]) + [""]
open(OUT, "w").write("\n".join(report) + "\n")
with open(OUT.replace(".md", ".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["table", "number", "bound", "published_median", "published_spread_pct", "published_runs", "reference_median", "reference_spread_pct", "reference_runs",
                "cleaned_median", "cleaned_spread_pct", "cleaned_runs", "drift_pct", "drift_judgement", "cleanup_pct", "cleanup_judgement", "cause"])
    w.writerows(ALL_ROWS)
print(f"{len(ALL_ROWS)} numbers; {len(flags_all)} flagged; fraction moves {len(FRACTION_MOVES)}")
