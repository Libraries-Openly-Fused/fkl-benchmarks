#!/usr/bin/env python3
"""Report of the cleanup window C (2026-11-08, E1 and E2): published / reference-now / cleaned for every GPU time, with run-to-run spreads, drift
(reference-now against published), cleanup effect (cleaned against reference-now), the 3 % (GPU-bound) / 5 % (host-bound) flags; the new columns of
the cleaned build (kernel-only with the 5 ms spin and with the 10 ms spin, preparation); the Python driver (cleaned script only, against the published
runs); agreement (bit-exact flags, output hashes, accuracy columns); the ncu pass. Reads only; writes the markdown report given as the first
argument and a CSV of every compared number next to it."""
import csv
import os
import re
import statistics
import subprocess
import sys

R = "/home/oscar/tfkl-bench/fkl-benchmarks/results/rtx4090"
C = R + "/cleanup-window-c"
OUT = sys.argv[1]
ALL_ROWS = []   # every compared number, for the CSV


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
    """'exceeds' when |change| is larger than every spread involved (a missing spread is ignored); the flag at 3 % (GPU) or 5 % (host)."""
    if change is None:
        return "-", False
    known = [s for s in spreads if s is not None]
    exceeds = bool(known) and abs(change) > max(known)
    limit = 5.0 if bound == "host" else 3.0
    flag = abs(change) > limit
    text = ("exceeds" if exceeds else "within") + (" FLAG" if flag else "")
    return text, flag and exceeds


def section(title, key_name, numbers, causes):
    """numbers: list of (label, bound, pub_values, ref_values, new_values); causes(label, 'drift' or 'cleanup'): the cause of a flagged row."""
    lines = [f"### {title}", "",
             f"| {key_name} | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    flagged = []
    for label, bound, pv, rv, nv in numbers:
        pm, ps, pn = stats(pv)
        rm, rs, rn = stats(rv)
        nm, ns, nn = stats(nv)
        drift, effect = pct(rm, pm), pct(nm, rm)
        dj, dflag = judge(drift, [ps, rs], bound)
        ej, eflag = judge(effect, [rs, ns], bound)
        cause = "; ".join(c for c in ((causes(label, "drift") if dflag else ""), (causes(label, "cleanup") if eflag else "")) if c)
        lines.append(f"| {label} | {bound} | {fmt(pm)} ({fs(ps)}) | {fmt(rm)} ({fs(rs)}) | {fmt(nm)} ({fs(ns)}) | {fp(drift)} | {dj} | {fp(effect)} | {ej} | {cause} |")
        ALL_ROWS.append([title, label, bound, pm, ps, pn, rm, rs, rn, nm, ns, nn, drift, dj, effect, ej, cause])
        if dflag:
            flagged.append((title, "drift", label, bound, f"{fp(drift)} (published {fmt(pm)}, reference now {fmt(rm)}; spreads {fs(ps)}, {fs(rs)})", causes(label, "drift")))
        if eflag:
            flagged.append((title, "cleanup", label, bound, f"{fp(effect)} (reference now {fmt(rm)}, cleaned {fmt(nm)}; spreads {fs(rs)}, {fs(ns)})", causes(label, "cleanup")))
    return lines + [""], flagged


def section2(title, key_name, numbers, causes):
    """Two sides only (published and cleaned): change = cleaned against published, drift and cleanup together."""
    lines = [f"### {title}", "",
             f"| {key_name} | bound | published (spread) | cleaned (spread) | change cleaned vs pub | | cause |",
             "|---|---|---|---|---|---|---|"]
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


def kernel_only_table(title, key_name, items):
    """items: (label, normal values (3 runs), kernel values at 5 ms (3 runs), kernel value at 10 ms (one run), normal value of the 10 ms run)."""
    lines = [f"### {title}", "",
             f"| {key_name} | normal (spread) | kernel-only, 5 ms spin (spread) | kernel-only, 10 ms spin (one run) | 10 ms vs 5 ms | | kernel-only / normal | normal in the 10 ms run |",
             "|---|---|---|---|---|---|---|---|"]
    notes = []
    for label, nv, kv, k2, n2 in items:
        nm, ns, _ = stats(nv)
        km, ks, _ = stats(kv)
        d = pct(k2, km)
        j = "-" if d is None else ("exceeds the 5 ms spread" if ks is not None and abs(d) > ks else "within the 5 ms spread")
        ratio = km / nm if (km is not None and nm) else None
        lines.append(f"| {label} | {fmt(nm)} ({fs(ns)}) | {fmt(km)} ({fs(ks)}) | {fmt(k2)} | {fp(d)} | {j} | {fmt(ratio, 3)} | {fmt(n2)} |")
        ALL_ROWS.append([title, label, "kernel-only", nm, ns, 3, km, ks, 3, k2, None, 1, d, j, ratio, "kernel/normal", ""])
        if d is not None and ks is not None and abs(d) > ks and abs(d) > 3:
            notes.append(f"{title} / {label}: 10 ms spin {fp(d)} against the 5 ms spin (spread {fs(ks)})")
    return lines + [""], notes


def bound_of(normal, kernel):
    """host when the kernel-only time is below 80 % of the normal time (more than a fifth of the interval is the dispatch of the launch), GPU otherwise"""
    return "host" if (kernel is not None and normal and kernel < 0.8 * normal) else "GPU"


report = ["# Cleanup window C, 2026-11-08: E1 and E2, published, reference now and cleaned", ""]
flags_all, spin_notes = [], []

# ---------------------------------------------------------------- E1
E1 = (("E1a", "e1-vf-chain", "e1-vf-chain-v2", "e1_vf_chain.csv", "ops", 6, ("hand_params", "hand_imm", "fkl")),
      ("E1b", "e1-hf-batch", "e1-hf-batch-v2", "e1_hf_batch.csv", "batch", 5, ("hand", "hand_generic", "fkl", "iterated")),
      ("E1c", "e1-preproc", "e1-preproc-v3", "e1_preproc.csv", "batch", 5, ("hand", "hand_generic", "hand_padded", "fkl")))
ORDER_E1 = ("order of the passes: in the cleaned build every variant's normal pass follows the previous variant's kernel-only pass (100 one-thread spins of 5 ms, "
            "about 0.5 s), in the reference the previous variant's normal pass; at 5,002 and 10,002 operations the reference's hand_imm and fkl (timed second and "
            "third) are 3-5 % slower than hand_params (timed first), as in the published runs, and the cleaned build's three variants agree within 0.8 %; probable "
            "GPU clock or power state, not isolated")
E1C_DRIFT = ("drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 "
             "(10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state "
             "file); the ratios between the versions are unchanged (ratio table); not isolated")


def e1_causes(exp):
    def cause(label, kind):
        if exp == "E1a" and kind == "cleanup" and label.split()[0] in ("5002", "10002", "19902"):
            return ORDER_E1
        if exp == "E1b" and label == "1 iterated" and kind == "cleanup":
            return ("what precedes the pass: in the cleaned build fkl's kernel-only pass runs between fkl's and iterated's timing; the launch code inside the "
                    "interval is the same in both builds (one executeOperations per crop); the extra run with the 10 ms spin gave 0.002819 ms, as the reference "
                    "(0.002867); a 3 us host-bound number; not isolated")
        if exp == "E1c" and kind == "drift":
            return E1C_DRIFT
        if exp == "E1c" and label == "10 fkl" and kind == "cleanup":
            return ("the dispatch part of a 3.7 us interval: fkl's kernel-only time is 3.08 us, as hand's (3.08 us), so the change (0.17 us) is outside the kernel; "
                    "what precedes the pass differs (hand_padded's kernel-only pass); the extra 10 ms-spin run gave 3.49 us; not isolated")
        return "not examined"
    return cause


BITEXACT = {}


def by_point(dirs, fname, key, expect):
    out = []
    for d in dirs:
        out.append({int(r[key]): r for r in rows_of(os.path.join(d, fname), expect)})
    return out


for exp, name, pub, fname, key, n, versions in E1:
    P = by_point([f"{R}/{pub}/run{i}" for i in (1, 2, 3)], fname, key, n)
    Rf = by_point([f"{C}/reference/{name}/run{i}" for i in (1, 2, 3)], fname, key, n)
    N = by_point([f"{C}/cleaned/{name}/run{i}" for i in (1, 2, 3)], fname, key, n)
    S = by_point([f"{C}/cleaned/{name}/spin2x"], fname, key, n)[0]
    numbers, ko, prep = [], [], []
    for point in sorted(N[0]):
        for v in versions:
            col = f"{v}_mean_ms"
            nv = [float(x[point][col]) for x in N]
            kv = [float(x[point][f"{v}_kernel_mean_ms"]) for x in N]
            bound = bound_of(statistics.median(nv), statistics.median(kv))
            numbers.append((f"{point} {v}", bound, [float(x[point][col]) for x in P], [float(x[point][col]) for x in Rf], nv))
            ko.append((f"{point} {v}", nv, kv, float(S[point][f"{v}_kernel_mean_ms"]), float(S[point][col])))
            prep.append((point, v, [float(x[point][f"{v}_prep_mean_ms"]) for x in N], [float(x[point][f"{v}_prep_min_ms"]) for x in N],
                         [float(x[point][f"{v}_total_mean_ms"]) for x in N], nv))
        for side, sets in (("published", P), ("reference", Rf), ("cleaned", N + [S])):
            for x in sets:
                for c, val in x[point].items():
                    if c.endswith("_bitexact"):
                        BITEXACT.setdefault((exp, side), []).append(val)
    lines, fl = section(f"{exp} ({fname[:-4]}), GPU time per launch (ms; CUDA events around one launch, mean of 100; median of the three runs)",
                        f"{key}, version", numbers, e1_causes(exp))
    report += lines
    flags_all += fl
    lines, notes = kernel_only_table(f"{exp}: kernel-only (cleaned build; a GPU spin before the start event), 5 ms spin (three runs) and 10 ms spin (one run, E1_SPIN_US=10000)",
                                     f"{key}, version", ko)
    report += lines
    spin_notes += notes
    report += [f"### {exp}: preparation of one launch (cleaned build; host time, outside the GPU interval; median of the three runs)", "",
               f"| {key}, version | prep mean ms (spread) | prep min ms (smallest of the three) | total mean ms (GPU + prep) | prep / GPU |", "|---|---|---|---|---|"]
    for point, v, pm_, pmin, tot, gpu in prep:
        m, s, _ = stats(pm_)
        report.append(f"| {point} {v} | {fmt(m)} ({fs(s)}) | {fmt(min(pmin))} | {fmt(statistics.median(tot))} | {fmt(m / statistics.median(gpu), 3)} |")
        ALL_ROWS.append([f"{exp} preparation", f"{point} {v}", "host", None, None, 0, None, None, 0, m, s, 3, None, "", None, "", ""])
    report.append("")

# ---------------------------------------------------------------- E1, the ratios of the paper's compact table
REFS = {"E1a": ("hand_params", "hand_imm"), "E1b": ("hand_generic", "hand"), "E1c": ("hand_padded", "hand")}   # as scripts/e1_compact.py
report += ["### E1, the ratios of the paper's compact table (scripts/e1_compact.py: FKL's time over the reference's, per run, median of the three runs; smallest and largest of the three in brackets)", "",
           "vs_ref: E1a hand_params, E1b hand_generic, E1c hand_padded; vs_best: E1a hand_imm, E1b and E1c hand. Kernel-only: the cleaned build's kernel-only times.", "",
           "| exp, point | vs_ref published | reference now | cleaned | cleaned kernel-only | vs_best published | reference now | cleaned | cleaned kernel-only |",
           "|---|---|---|---|---|---|---|---|---|"]
RATIO_ROWS = []
for exp, name, pub, fname, key, n, versions in E1:
    sets = {"published": by_point([f"{R}/{pub}/run{i}" for i in (1, 2, 3)], fname, key, n),
            "reference": by_point([f"{C}/reference/{name}/run{i}" for i in (1, 2, 3)], fname, key, n),
            "cleaned": by_point([f"{C}/cleaned/{name}/run{i}" for i in (1, 2, 3)], fname, key, n)}
    ref_v, best_v = REFS[exp]
    for point in sorted(sets["cleaned"][0]):
        cells = []
        for other in (ref_v, best_v):
            for side, suffix in (("published", ""), ("reference", ""), ("cleaned", ""), ("cleaned", "_kernel")):
                r = [float(x[point][f"fkl{suffix}_mean_ms"]) / float(x[point][f"{other}{suffix}_mean_ms"]) for x in sets[side]]
                cells.append(f"{statistics.median(r):.3f} ({min(r):.3f}-{max(r):.3f})")
                RATIO_ROWS.append((exp, point, other, side + suffix, statistics.median(r)))
        report.append(f"| {exp} {point} | " + " | ".join(cells) + " |")
report.append("")
moved = [(e, p, o) for (e, p, o, s_, m) in RATIO_ROWS if s_ == "published"
         for (e2, p2, o2, s2, m2) in RATIO_ROWS if (e2, p2, o2, s2) == (e, p, o, "cleaned") and abs(m2 - m) > 0.02]
RATIO_MOVES = []
for e, p_, o in moved:
    pub_ = next(m for (e2, p2, o2, s2, m) in RATIO_ROWS if (e2, p2, o2, s2) == (e, p_, o, "published"))
    new_ = next(m for (e2, p2, o2, s2, m) in RATIO_ROWS if (e2, p2, o2, s2) == (e, p_, o, "cleaned"))
    ref_ = next(m for (e2, p2, o2, s2, m) in RATIO_ROWS if (e2, p2, o2, s2) == (e, p_, o, "reference"))
    RATIO_MOVES.append(f"{e} {p_}, FKL over {o}: published {pub_:.3f}, reference now {ref_:.3f}, cleaned {new_:.3f}")

# ---------------------------------------------------------------- E2, the FKL program
PUB_E2 = [f"{R}/e2-chain", f"{R}/e2-chain-run3", f"{R}/e2-chain-run4"]   # runs 2, 3 and 4: the published set (e2_compact.csv)
P = by_point(PUB_E2, "e2_fkl_chain.csv", "ops", 6)
Rf = by_point([f"{C}/reference/e2-fkl/run{i}" for i in (1, 2, 3)], "e2_fkl_chain.csv", "ops", 6)
N = by_point([f"{C}/cleaned/e2-fkl/run{i}" for i in (1, 2, 3)], "e2_fkl_chain.csv", "ops", 6)
S = by_point([f"{C}/cleaned/e2-fkl/spin2x"], "e2_fkl_chain.csv", "ops", 6)[0]
numbers, ko = [], []
for k in sorted(N[0]):
    nv = [float(x[k]["fkl_mean_ms"]) for x in N]
    kv = [float(x[k]["fkl_kernel_mean_ms"]) for x in N]
    numbers.append((f"{k} warm", bound_of(statistics.median(nv), statistics.median(kv)), [float(x[k]["fkl_mean_ms"]) for x in P], [float(x[k]["fkl_mean_ms"]) for x in Rf], nv))
    numbers.append((f"{k} L2-flushed", "GPU", [float(x[k]["fkl_flushed_mean_ms"]) for x in P], [float(x[k]["fkl_flushed_mean_ms"]) for x in Rf],
                    [float(x[k]["fkl_flushed_mean_ms"]) for x in N]))
    ko.append((f"{k} fkl", nv, kv, float(S[k]["fkl_kernel_mean_ms"]), float(S[k]["fkl_mean_ms"])))
def e2_cause(label, kind):
    if label == "10 warm" and kind == "drift":
        return "at the edge: the change (-3.8 %) equals the spreads of the published and the reference runs (3.8 % each); reference and cleaned agree (0.02042, 0.02037 ms)"
    if label == "5002 L2-flushed" and kind == "cleanup":
        return ("order of the passes: in the cleaned build the flushed pass follows the kernel-only pass of the same k (100 spins of 5 ms), in the reference the warm "
                "pass; the cleaned flushed time equals its kernel-only time (0.5234 against 0.5217 ms); probable GPU clock or power state, as E1a; not isolated")
    return "not examined"


lines, fl = section("E2, the FKL program (e2_fkl_chain), GPU time per launch (ms; mean of 100; published: runs 2, 3, 4)", "k, pass", numbers, e2_cause)
report += lines
flags_all += fl
lines, notes = kernel_only_table("E2, the FKL program: kernel-only (cleaned build), 5 ms spin (three runs) and 10 ms spin (one run)", "k", ko)
report += lines
spin_notes += notes
report += ["### E2, the FKL program: preparation (cleaned build; host time of building the operations of one launch, outside the GPU interval)", "",
           "| k | warm pass prep mean ms (spread) | flushed pass prep mean ms (spread) |", "|---|---|---|"]
for k in sorted(N[0]):
    a = stats([float(x[k]["fkl_prep_mean_ms"]) for x in N])
    b = stats([float(x[k]["fkl_flushed_prep_mean_ms"]) for x in N])
    report.append(f"| {k} | {fmt(a[0])} ({fs(a[1])}) | {fmt(b[0])} ({fs(b[1])}) |")
report.append("")

# ---------------------------------------------------------------- E2, the Python driver
def py_rows(dirs):
    out = []
    for d in dirs:
        out.append({(int(r["ops"]), r["version"]): r for r in rows_of(os.path.join(d, "e2_chain.csv"), 35)})
    return out


PP = py_rows(PUB_E2)
PN = py_rows([f"{C}/cleaned/e2-python/run{i}" for i in (1, 2, 3)])
PS = py_rows([f"{C}/cleaned/e2-python/spin2x"])[0]
VERSIONS = ("triton_hand", "inductor_default", "inductor_wide", "inductor_graphs", "eager", "eager_graphs")
numbers, ko = [], []
for k in sorted({k for k, _ in PN[0]}):
    for v in VERSIONS:
        nv = [float(x[(k, v)]["gpu_mean_ms"]) for x in PN]
        kv = [float(x[(k, v)]["gpu_kernel_mean_ms"]) for x in PN]
        numbers.append((f"{k} {v} warm", bound_of(statistics.median(nv), statistics.median(kv)), [float(x[(k, v)]["gpu_mean_ms"]) for x in PP], nv))
        numbers.append((f"{k} {v} L2-flushed", "GPU", [float(x[(k, v)]["gpu_flushed_mean_ms"]) for x in PP], [float(x[(k, v)]["gpu_flushed_mean_ms"]) for x in PN]))
        numbers.append((f"{k} {v} host_launch_us", "host", [float(x[(k, v)]["host_launch_us"]) for x in PP], [float(x[(k, v)]["host_launch_us"]) for x in PN]))
        ko.append((f"{k} {v}", nv, kv, float(PS[(k, v)]["gpu_kernel_mean_ms"]), float(PS[(k, v)]["gpu_mean_ms"])))
PY_HOST = ("drift: the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them in the same "
           "process); window C pinned the driver to cores 0-3 and 16-19 (four cores and their SMT siblings), the published runs' pinning is not recorded; not isolated")


def py_cause(label, kind):
    if label.endswith("host_launch_us") or label == "2 inductor_graphs warm":
        return PY_HOST
    if " eager " in f" {label} ":
        return "drift: the code of these numbers is unchanged since the published runs (as above); eager enqueues one PyTorch kernel per operation; not isolated"
    return "not examined"


lines, fl = section2("E2, the Python driver (triton/e2_chain.py; cleaned script only, no reference run: the change is drift and cleanup together; published: runs 2, 3, 4)",
                     "k, version, column", numbers, py_cause)
report += lines
flags_all += fl
lines, notes = kernel_only_table("E2, the Python driver: kernel-only (torch.cuda._sleep before the start event), 5 ms spin (three runs) and 10 ms spin (one run)", "k, version", ko)
report += lines
spin_notes += notes

# ---------------------------------------------------------------- agreement
report += ["## Agreement", ""]
for (exp, side), vals in sorted(BITEXACT.items()):
    report.append(f"- {exp} bit-exact flags, {side}: {len(vals)} values, all 1: {all(v == '1' for v in vals)}")
hashes = {}
for side in ("reference", "cleaned"):
    for i in (1, 2, 3):
        hashes[f"{side} run{i}"] = sorted(l.split()[0] + " " + os.path.basename(l.split()[1]) for l in open(f"{C}/{side}/e2-fkl/run{i}/outputs-sha256.txt") if l.strip())
first = hashes["reference run1"]
report.append(f"- E2 FKL outputs (6 files per run): the SHA-256 of every run equal to reference run 1's: {all(h == first for h in hashes.values())}")
ACC = ("kernels", "max_ulp_vs_exact", "bitexact_fraction_vs_triton_hand", "bitexact_fraction_vs_fkl", "max_ulp_vs_fkl", "sass_ffma", "sass_fmul", "sass_fadd", "status")
diff = []
for x in PN + [PS]:
    for key, r in x.items():
        for c in ACC:
            ref = {p[key][c] for p in PP}
            if r[c] not in ref:
                diff.append((key, c, r[c], sorted(ref)))
report.append(f"- E2 Python, accuracy, kernel and SASS columns ({', '.join(ACC)}), every cleaned run and the 10 ms run against the published runs: "
              + ("all equal" if not diff else f"{len(diff)} differences: {diff[:8]}"))
report.append("")

# ---------------------------------------------------------------- ncu
report += ["## E1 ncu pass of the cleaned build", ""]


def ncu_rows(path):
    lines = [l for l in open(path) if l.startswith('"')]
    return [r for r in csv.DictReader(lines) if r.get("ID", "").isdigit()]


def base(name):
    """the kernel's function name and its first template argument: handParams<1>, launchTransformDPP_Kernel<...> -> launchTransformDPP_Kernel"""
    n = name
    if n.startswith("_Z"):
        n = subprocess.run(["c++filt", n], capture_output=True, text=True).stdout.strip()
    n = re.sub(r"^void ", "", n)
    depth, cut = 0, len(n)
    for i, ch in enumerate(n):
        if ch == "<":
            if depth == 0 and cut == len(n):
                cut = i
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            cut = min(cut, i)
            break
    head = n[:cut].split("::")[-1]
    first = re.match(r"<\s*(?:\([\w:]+\))?(-?\d+)", n[cut:])
    return head + (f"<{first.group(1)}>" if first and head.startswith("hand") else "")


for exp, name, pub, fname, key, n, versions in E1:
    a = ncu_rows(f"{R}/{pub}/profile/ncu.csv")
    b = ncu_rows(f"{C}/cleaned/{name}/profile/ncu.csv")
    same_order = len(a) == len(b) and all(base(x["Kernel Name"]) == base(y["Kernel Name"]) for x, y in zip(a, b))
    eq = {}
    for m in ("launch__registers_per_thread", "launch__block_size", "launch__grid_size", "launch__shared_mem_per_block_dynamic"):
        if m in a[0] and m in b[0]:
            eq[m] = all(x[m].replace(",", "") == y[m].replace(",", "") for x, y in zip(a, b))
    occ = []
    for m in ("sm__warps_active.avg.pct_of_peak_sustained_active", "sm__maximum_warps_per_active_cycle_pct"):
        if m in a[0] and m in b[0]:
            occ.append((m, max(abs(float(x[m].replace(",", "")) - float(y[m].replace(",", ""))) for x, y in zip(a, b))))
    report.append(f"- {exp}: {len(b)} kernels (published {len(a)}); the same kernels in the same launch order (function name and, for the hand-written ones, the first "
                  f"template argument): {same_order}; equal launch values: {eq}; largest difference in points of percent: " + ", ".join(f"{m} {d:.2f}" for m, d in occ))
report += ["- The pass ran with ncu's default kernel names (demangled); the published pass has the mangled names, which scripts/e1_table.py needs to join the ncu",
           "  rows with the SASS of the binary. The rows above are matched by launch order (profile mode launches every kernel once, in a fixed order). The tables",
           "  (e1_table.py, e1_compact.py) of the cleaned build need the pass again with --print-kernel-base mangled.", ""]

# ---------------------------------------------------------------- flags first
def short(title):
    return title.split(" (")[0].replace("E2, the FKL program", "E2 FKL program").replace("E2, the Python driver", "E2 Python driver")


groups = {}
for title, kind, label, bound, numbers_text, cause in flags_all:
    groups.setdefault((short(title), kind, cause), []).append(f"{label} ({bound}) {numbers_text}")
flag_lines = []
for (t, kind, cause), items in groups.items():
    flag_lines.append(f"- **{t}, {kind}**, {len(items)} number{'s' if len(items) > 1 else ''}: " + "; ".join(items) + f". Cause: {cause}")
FINDINGS = [
    "## Findings", "",
    "1. **Order of the passes.** The cleaned programs time every variant's normal pass and then its kernel-only pass (100 launches, each behind a one-thread GPU spin "
    "of 5 ms: about 0.5 s of a nearly idle GPU), so the next variant (E1) or pass (E2: warm, kernel-only, flushed) starts after that pass instead of after the "
    "previous variant's normal pass. In the reference and the published runs, E1a's variants timed second and third at 5,002 and 10,002 operations (hand_imm, fkl) "
    "are 3-5 % slower than the first (hand_params); in the cleaned build the three agree within 0.8 %. The paper's E1a vs_ref at 5,002 and 10,002 operations moves "
    "from 1.026 and 1.018 (published) to 0.994 and 0.989 (cleaned; kernel-only 0.998 and 0.998). E2's flushed pass at k = 5,002 is 3.0 % faster (it now follows "
    "the kernel-only pass). Probable cause: the GPU's clock or power state after sustained load; not isolated (no clock log in window C).",
    "2. **E1c drift.** The published set (e1-preproc-v3) is 3-6 % slower than reference now on every version; reference now matches e1-preproc-v2 of the same day "
    "within 2.3 %. The ratios move only at batch 10 to 100, where the interval is 3.5-8 us and 4-17 % of it is dispatch (kernel-only / normal 0.83-0.96); there "
    "the kernel-only ratios against hand_padded are 0.975-1.004.",
    "3. **Kernel-only against normal.** At 10,002 and 19,902 operations (E1a) kernel-only is 4 % below normal: 40-90 us, far more than a dispatch, and a 10 ms spin "
    "lowers it by only 0.3-0.5 % more. There the kernel-only time is the kernel after a nearly idle GPU, not only the kernel without its dispatch. E2's FKL warm "
    "interval holds no visible dispatch (kernel-only / normal 1.01-1.02 at k <= 500). The Python driver's warm intervals of Triton and Inductor are 37-71 % dispatch "
    "at k <= 100 (kernel-only / normal 0.29-0.63).",
    "4. **E2 without the dispatch.** Kernel-only, FKL takes 0.0205 ms at k = 2 and 0.0653 ms at k = 500, Triton 0.0180 and 0.0592, Inductor (default) 0.0180 and "
    "0.0594: FKL's kernel is 10-38 % slower than theirs at k <= 500. That is the order of the published L2-flushed times; the warm times compare calls with their dispatch.",
    "5. **The 10 ms spin against the 5 ms spin.** Within 1.9 % for the 72 kernel-only numbers above 10 us. Of the 22 numbers at or below 10 us, three differ by "
    "more: E1b batch 1 hand_generic +16.5 % and iterated +10.5 %, and E1c batch 100 hand_generic +4.5 % (one run each).",
    "6. **Agreement.** All bit-exact flags are 1 on all three sides. The E2 FKL output hashes are identical in all six runs and equal to the outputs in "
    "e2-work/ (2026-10-31, the published runs). The Python accuracy, kernel and SASS columns equal the published ones.",
    "7. **ncu.** The pass ran with ncu's default (demangled) kernel names; the published pass and scripts/e1_table.py use mangled names. Matched by launch order: "
    "the same kernels, registers, launch configurations and theoretical occupancy; achieved occupancy within 0.6 (E1a), 5.5 (E1b) and 5.0 (E1c) points. The "
    "cleaned E1 tables need the pass again with --print-kernel-base mangled.", ""]
report[2:2] = (FINDINGS + ["## Changes above the spread and above the 3 % (GPU) or 5 % (host) threshold", "", f"{len(flags_all)} numbers of {len(ALL_ROWS)} compared:", ""]
               + (flag_lines or ["- none"]) + [""]
               + ["## The paper's E1 ratios that move by more than 0.02 (published against cleaned)", ""] + ([f"- {m}" for m in RATIO_MOVES] or ["- none"]) + [""]
               + ["## Kernel-only: the 10 ms spin against the 5 ms spin, beyond the 5 ms spread and above 3 %", ""] + ([f"- {s}" for s in spin_notes] or ["- none"]) + [""]
               + ["Bound: host when the kernel-only time of the cleaned build is below 80 % of its normal time (more than a fifth of the interval is the dispatch of the",
                  "launch), GPU otherwise; the L2-flushed passes are GPU (the flush keeps the GPU busy while the launch is enqueued); host_launch_us is host.", ""])
open(OUT, "w").write("\n".join(report) + "\n")
with open(OUT.replace(".md", ".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["table", "number", "bound", "published_median", "published_spread_pct", "published_runs", "reference_median", "reference_spread_pct", "reference_runs",
                "cleaned_median", "cleaned_spread_pct", "cleaned_runs", "drift_pct", "drift_judgement", "cleanup_pct", "cleanup_judgement", "cause"])
    w.writerows(ALL_ROWS)
print(f"{len(ALL_ROWS)} numbers; {len(flags_all)} flagged; {len(spin_notes)} spin notes")
