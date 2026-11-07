#!/usr/bin/env python3
# Copyright 2026 Oscar Amoros Huguet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E16 summary: the complete runs of fkl-self/attention/e16_attention.cu (e16_attention.csv) and pytorch-attention/e16_baselines.py (e16_baselines.csv) of one GPU.

  python scripts/e16_summary.py results/rtx4090/e16-attention          (the folder holds run1, run2, ... each with the two CSV files)

Every number is the median over the runs of the median time of the run, with the smallest and the largest of the run medians in brackets (the run-to-run spread). Two tables:
  core    the attention core with nothing fused around it: TFLOP/s of mma_raw (FKL) and of the baselines, the best baseline and the fraction FKL / best (per run, then median [smallest-largest])
  fusion  the work around the pattern, fused against separate kernels: K and V from int8 (and fp8 e4m3 for FKL), and the epilogue o * a + b. Time ratios separate / fused on FKL's own kernels, and the PyTorch form of the same
          work (dequant_flash, dequant_compiled_flash, epilogue_flash) against the FKL fused variant. For the epilogue pair the two forms run the same attention kernel, so the difference is the effect of the fusion alone; for the
          prologue pairs the two attention kernels also differ in schedule details and tile sizes (the in-kernel shared-memory dequantisation of cp.async-copied quantised bytes against a separate dequantisation kernel and the bf16
          cp.async kernel), so it is not. Every pair gets a verdict (which form is faster, whether the run medians of the two forms overlap) and the SM clock check (more than 2 % apart: DIFFERENT CLOCKS).
and the kernels of one call with their registers, local memory, shared memory, blocks per SM, tile and schedule (from the first run), and the largest error of each variant over the runs; the text numbers; the compact Q2 table.
Run scripts/e16_join_clock.py first so that the rows have their SM clock. Writes e16_summary_core.csv, e16_summary_fusion.csv, e16_summary_kernels.csv, e16_q2_compact.csv and e16_text_numbers.txt in the folder.
"""
import argparse
import csv
import glob
import os
import re
import statistics as stats

BASELINES = ["flash", "cudnn", "efficient", "flash_attn", "triton"]   # the attention kernels the core is compared with (math is not: it materialises S x S)


def read(path):
    with open(path, newline="") as f:
        lines = [l for l in f if not l.startswith("#")]
    return list(csv.DictReader(lines))


def span(values):
    return stats.median(values), min(values), max(values)


def cell(values, fmt):
    m, lo, hi = span(values)
    return f"{fmt.format(m)} [{fmt.format(lo)}-{fmt.format(hi)}]"


def load(folder):
    """{run: {(variant, d, seq, causal): row}} from every run*/ folder."""
    runs = {}
    for r in sorted(glob.glob(os.path.join(folder, "run*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
        rows = {}
        for name in ("e16_attention.csv", "e16_baselines.csv"):
            path = os.path.join(r, name[:-4] + "_clock.csv")   # the rows with the SM clock of their blocks (scripts/e16_join_clock.py), when joined
            if not os.path.exists(path):
                path = os.path.join(r, name)
            if os.path.exists(path):
                for row in read(path):
                    rows[(row["variant"], int(row["d"]), int(row["seq"]), int(row["causal"]))] = row
        runs[os.path.basename(r)] = rows
    return runs


def t(row, col="time_ms_median"):
    try:
        v = float(row[col])
        return v if v > 0 else None
    except (KeyError, ValueError):
        return None


def tflops(row):
    return t(row, "tflops_median")


def clock(row):
    """SM clock (MHz) of the blocks of a row, from the join with the sampler; None when not joined."""
    try:
        v = float(row["sm_mhz_median"])
        return v if v > 0 else None
    except (KeyError, ValueError):
        return None


def tile(row):
    """BLOCK_Q x BLOCK_KV x warps of the attention kernel of a row of the C++ program (from its template arguments), '' when absent."""
    try:
        return f"{int(row['block_q'])}x{int(row['block_kv'])}x{int(row['num_warps'])}"
    except (KeyError, ValueError, TypeError):
        return ""


def global_bytes(work, form, tokens, d):
    """COMPUTED, not measured: the compulsory global traffic of one call in bytes, every element counted once (K and V for the prologue pairs, the output O for the epilogue pair); what the kernels re-read from L2 or DRAM
    across query tiles is not in it. t = tokens = BH x S, a token has d elements; scales are fp32 per token (4 B), in the FKL kernels and in the PyTorch composition (the same input file).
      prologue, fused:      int8 or e4m3 K and V read once                                  = 2 t d + 2 t 4
      prologue, unfused:    dequantisation kernel reads them and writes bf16, attention reads bf16 = (2 t d + 2 t 4) + 4 t d + 4 t d
      prologue, PyTorch:    two elementwise kernels read int8 and fp32 scales, write bf16, flash reads bf16 = 2 (t d + 4 t) + 4 t d + 4 t d = 10 t d + 8 t
      raw bf16 (reference): K and V bf16 read once                                          = 4 t d
      epilogue, fused:      the output written once                                          = 2 t d
      epilogue, separate:   the output written by the attention kernel, read and written by the elementwise kernel = 2 t d + 2 t d + 2 t d
    """
    t = tokens
    if work.startswith("epilogue"):
        return 2 * t * d if form == "fused" else 6 * t * d
    if form == "fused":
        return 2 * t * d + 8 * t
    if form == "separate":
        return (2 * t * d + 8 * t) + 8 * t * d
    if form == "pytorch":
        return 2 * (t * d + 4 * t) + 8 * t * d
    return 4 * t * d


CLOCK_TOLERANCE = 2.0   # percent: two members of a comparison whose SM clocks differ by more than this are marked


def attn_facts(row):
    """(registers, local bytes) of the attention kernel of a call, from the kernel facts of the C++ program; None for the baselines (names only)."""
    for part in row.get("kernel_facts", "").split(" ; "):
        if "FlashAttention" in part:
            m = re.search(r"regs (\d+) local_B (\d+)", part)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--rep", default="128:2048:0,128:8192:0", help="representative configurations of the compact Q2 table: D:S:causal, comma separated")
    args = ap.parse_args()
    folder = args.folder
    runs = load(folder)
    names = list(runs)
    configs = sorted({(d, c, s) for rows in runs.values() for (_, d, s, c) in rows})
    print(f"{len(runs)} runs of {folder}: {', '.join(names)}")

    # ---- core: the attention with nothing fused around it
    core = []
    print("\nCORE (nothing fused around the pattern): TFLOP/s, median over the runs [smallest-largest of the run medians]; fraction = mma_raw / best baseline, per run then median [smallest-largest]")
    print(f"{'D':>3s} {'causal':>6s} {'S':>5s} {'BH':>4s} {'mma_raw':>22s} " + " ".join(f"{b:>22s}" for b in BASELINES) + f" {'best baseline':>14s} {'fraction':>22s} {'raw tile':>10s} {'fraction, equal clock':>22s}")
    for d, c, s in configs:
        per_run = []
        for rows in runs.values():
            vals = {v: tflops(rows[(v, d, s, c)]) for v in ["mma_raw", "simt_raw"] + BASELINES if (v, d, s, c) in rows}
            per_run.append(vals)
        bh = next((int(rows[("mma_raw", d, s, c)]["bh"]) for rows in runs.values() if ("mma_raw", d, s, c) in rows), 0)
        row = {"d": d, "causal": c, "seq": s, "bh": bh}
        raw_row = next((rows[("mma_raw", d, s, c)] for rows in runs.values() if ("mma_raw", d, s, c) in rows), {})
        row["mma_raw_tile_BQxBKVxwarps"] = tile(raw_row)
        cells = []
        for v in ["mma_raw"] + BASELINES:
            xs = [p[v] for p in per_run if p.get(v)]
            row[f"{v}_tflops"] = f"{stats.median(xs):.3f}" if xs else ""
            row[f"{v}_tflops_min"], row[f"{v}_tflops_max"] = (f"{min(xs):.3f}", f"{max(xs):.3f}") if xs else ("", "")
            cells.append(cell(xs, "{:.1f}") if xs else "-")
        fracs, best_names, core_clock_diffs, fracs_equal_clock = [], [], [], []
        for p, rows in zip(per_run, runs.values()):
            cand = {b: p[b] for b in BASELINES if p.get(b)}
            if p.get("mma_raw") and cand:
                best = max(cand, key=cand.get)
                best_names.append(best)
                fracs.append(p["mma_raw"] / cand[best])
                if clock(rows.get(("mma_raw", d, s, c), {})) and clock(rows.get((best, d, s, c), {})):
                    core_clock_diffs.append(100.0 * (clock(rows[("mma_raw", d, s, c)]) / clock(rows[(best, d, s, c)]) - 1.0))
                    fracs_equal_clock.append(p["mma_raw"] / cand[best] * clock(rows[(best, d, s, c)]) / clock(rows[("mma_raw", d, s, c)]))   # TFLOP/s per MHz of FKL over TFLOP/s per MHz of the best baseline
        if fracs_equal_clock:
            row["fraction_equal_clock_median"], row["fraction_equal_clock_min"], row["fraction_equal_clock_max"] = (f"{x:.4f}" for x in span(fracs_equal_clock))
        if core_clock_diffs:
            row["clock_diff_percent_median"], row["clock_diff_percent_largest"] = f"{stats.median(core_clock_diffs):.2f}", f"{max(core_clock_diffs, key=abs):.2f}"
            if max(abs(x) for x in core_clock_diffs) > CLOCK_TOLERANCE:
                row["different_clocks"] = "yes"
        best_name = max(set(best_names), key=best_names.count) if best_names else "-"
        row["best_baseline"] = best_name
        if fracs:
            row["fraction_median"], row["fraction_min"], row["fraction_max"] = (f"{x:.4f}" for x in span(fracs))
        simt = [p["simt_raw"] for p in per_run if p.get("simt_raw")]
        row["simt_raw_tflops"] = f"{stats.median(simt):.3f}" if simt else ""
        core.append(row)
        print(f"{d:3d} {c:6d} {s:5d} {bh:4d} {cells[0]:>22s} " + " ".join(f"{x:>22s}" for x in cells[1:]) + f" {best_name:>14s} " + (f"{cell(fracs, '{:.3f}'):>22s}" if fracs else f"{'-':>22s}") + f" {row['mma_raw_tile_BQxBKVxwarps']:>10s}" + (f" {cell(fracs_equal_clock, '{:.3f}'):>22s}" if fracs_equal_clock else f" {'-':>22s}")
              + ("  DIFFERENT CLOCKS (FKL against the best baseline: up to " + row["clock_diff_percent_largest"] + " %)" if row.get("different_clocks") else ""))
    with open(os.path.join(folder, "e16_summary_core.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in core for k in r}, key=lambda k: list(core[0]).index(k) if k in core[0] else 99))
        w.writeheader()
        w.writerows(core)

    # ---- fusion: the work before and after the pattern
    pairs = [("int8 K,V prologue", "mma_int8kv_fused", "mma_int8kv_unfused", "dequant_flash", "flash", "dequant_compiled_flash"),
             ("fp8 e4m3 K,V prologue", "mma_fp8kv_fused", "mma_fp8kv_unfused", None, None, None),
             ("epilogue o*a+b", "mma_epilogue_fused", "mma_epilogue_separate", "epilogue_flash", "flash", None)]
    fusion = []
    print("\nFUSION (work around the pattern): time ratios, per run then median [smallest-largest]; >1 means the fused form is faster")
    print("  READ WITH: the fused int8 / fp8 variants run the library's quantised schedule (the raw quantised bytes are copied with cp.async into shared memory and dequantised there, shared memory to shared memory, inside the kernel); the unfused variants dequantise in a separate kernel that writes bf16 to global memory and then run the bf16 cp.async schedule. The two attention kernels differ in schedule details and tile sizes (block_q, block_kv columns of the CSV), so a prologue pair is not a pure fusion effect; the epilogue pair runs the same kernel and schedule: only the epilogue differs.")
    print("  dequant_flash / dequant_compiled_flash / epilogue_flash are compositions of off-the-shelf pieces that a user has today, not a hand-fused quantised-cache attention kernel and not the best that could be written.")
    print(f"{'work':>22s} {'D':>3s} {'causal':>6s} {'S':>5s} {'fused ms':>20s} {'FKL separate/fused':>22s} {'fused vs separate (runs)':>58s} {'PyTorch form/FKL fused':>24s} {'compiled form/FKL fused':>24s} {'PyTorch form/plain flash':>26s} {'kernels f/s/t':>14s} {'regs / local B fused | separate':>32s} {'tiles fused | separate':>26s}")
    for label, fused, sep, torch_form, plain, torch_form2 in pairs:
        for d, c, s in configs:
            fz, sp, tf, pl = ([r[(v, d, s, c)] for r in runs.values() if v and (v, d, s, c) in r] for v in (fused, sep, torch_form, plain))
            if not fz:
                continue
            rat_tf2 = [t(runs[n][(torch_form2, d, s, c)]) / t(runs[n][(fused, d, s, c)]) for n in names if torch_form2 and (torch_form2, d, s, c) in runs[n] and t(runs[n].get((fused, d, s, c), {})) and t(runs[n][(torch_form2, d, s, c)])]
            rat_sep = [t(runs[n][(sep, d, s, c)]) / t(runs[n][(fused, d, s, c)]) for n in names if (sep, d, s, c) in runs[n] and t(runs[n].get((fused, d, s, c), {})) and t(runs[n][(sep, d, s, c)])]
            rat_tf = [t(runs[n][(torch_form, d, s, c)]) / t(runs[n][(fused, d, s, c)]) for n in names if torch_form and (torch_form, d, s, c) in runs[n] and t(runs[n].get((fused, d, s, c), {})) and t(runs[n][(torch_form, d, s, c)])]
            rat_pl = [t(runs[n][(torch_form, d, s, c)]) / t(runs[n][(plain, d, s, c)]) for n in names if torch_form and (torch_form, d, s, c) in runs[n] and (plain, d, s, c) in runs[n] and t(runs[n][(torch_form, d, s, c)]) and t(runs[n][(plain, d, s, c)])]
            fms = [t(r) for r in fz if t(r)]
            first_run = runs[names[0]]
            kf = (first_run.get((fused, d, s, c), {}).get("kernels", "?"), first_run.get((sep, d, s, c), {}).get("kernels", "?"), first_run.get((torch_form, d, s, c), {}).get("kernels", "-") if torch_form else "-")
            fa, sa = attn_facts(first_run.get((fused, d, s, c), {})), attn_facts(first_run.get((sep, d, s, c), {}))
            med_ms = lambda v: (lambda xs: f"{stats.median(xs):.5f}" if xs else "")([t(runs[n][(v, d, s, c)]) for n in names if v and (v, d, s, c) in runs[n] and t(runs[n][(v, d, s, c)])])
            tokens = int(first_run.get((fused, d, s, c), {}).get("bh", 0) or 0) * s
            row_bytes = {"fused_global_bytes_computed": global_bytes(label, "fused", tokens, d), "separate_global_bytes_computed": global_bytes(label, "separate", tokens, d)}
            if torch_form and not label.startswith("epilogue"):
                row_bytes["pytorch_global_bytes_computed"] = global_bytes(label, "pytorch", tokens, d)
            row = {"work": label, "d": d, "causal": c, "seq": s, "fused_tile": tile(first_run.get((fused, d, s, c), {})), "separate_tile": tile(first_run.get((sep, d, s, c), {})), **row_bytes, "fused_ms_median": f"{stats.median(fms):.5f}" if fms else "", "separate_ms_median": med_ms(sep), "pytorch_form_ms_median": med_ms(torch_form),
                   "pytorch_form2_ms_median": med_ms(torch_form2), "kernels_fused": kf[0], "kernels_separate": kf[1], "kernels_pytorch_form": kf[2],
                   "kernels_pytorch_form2": first_run.get((torch_form2, d, s, c), {}).get("kernels", "") if torch_form2 else "",
                   "fused_regs": fa[0] if fa else "", "fused_local_B": fa[1] if fa else "", "separate_regs": sa[0] if sa else "", "separate_local_B": sa[1] if sa else ""}
            for name, vals in (("fkl_separate_over_fused", rat_sep), ("pytorch_form_over_fkl_fused", rat_tf), ("pytorch_form_over_plain_flash", rat_pl), ("pytorch_form2_over_fkl_fused", rat_tf2)):
                if vals:
                    row[name + "_median"], row[name + "_min"], row[name + "_max"] = (f"{x:.4f}" for x in span(vals))
            # sign and size of the difference between the fused and the separate form, and whether it exceeds the run-to-run spread: the range of the run medians of the two forms do not overlap
            sep_runs = [t(runs[n][(sep, d, s, c)]) for n in names if (sep, d, s, c) in runs[n] and t(runs[n][(sep, d, s, c)])]
            if fms and sep_runs:
                faster = stats.median(fms) < stats.median(sep_runs)
                exceeds = max(fms) < min(sep_runs) or max(sep_runs) < min(fms)
                verdict = ("fused faster" if faster else "fused slower") + (", exceeds the spread" if exceeds else ", within the spread")
                row["difference_percent"] = f"{100.0 * (stats.median(sep_runs) / stats.median(fms) - 1.0):.2f}"   # separate time over fused time, minus 1: positive = the fused form is faster
                row["verdict"] = verdict
            else:
                verdict = "-"
            diffs = [100.0 * (clock(runs[n][(fused, d, s, c)]) / clock(runs[n][(sep, d, s, c)]) - 1.0) for n in names
                     if (fused, d, s, c) in runs[n] and (sep, d, s, c) in runs[n] and clock(runs[n][(fused, d, s, c)]) and clock(runs[n][(sep, d, s, c)])]
            eq = [t(runs[n][(sep, d, s, c)]) * clock(runs[n][(sep, d, s, c)]) / (t(runs[n][(fused, d, s, c)]) * clock(runs[n][(fused, d, s, c)])) for n in names
                  if (fused, d, s, c) in runs[n] and (sep, d, s, c) in runs[n] and clock(runs[n][(fused, d, s, c)]) and clock(runs[n][(sep, d, s, c)]) and t(runs[n][(fused, d, s, c)]) and t(runs[n][(sep, d, s, c)])]
            if eq:
                row["ratio_equal_clock_median"], row["ratio_equal_clock_min"], row["ratio_equal_clock_max"] = (f"{x:.4f}" for x in span(eq))   # separate / fused in SM cycles (time x clock)
            if diffs:
                row["clock_diff_percent_median"], row["clock_diff_percent_largest"] = f"{stats.median(diffs):.2f}", f"{max(diffs, key=abs):.2f}"
                row["fused_sm_mhz"] = f"{stats.median(clock(runs[n][(fused, d, s, c)]) for n in names if (fused, d, s, c) in runs[n] and clock(runs[n][(fused, d, s, c)])):.0f}"
                row["separate_sm_mhz"] = f"{stats.median(clock(runs[n][(sep, d, s, c)]) for n in names if (sep, d, s, c) in runs[n] and clock(runs[n][(sep, d, s, c)])):.0f}"
                if max(abs(x) for x in diffs) > CLOCK_TOLERANCE:
                    verdict += f", DIFFERENT CLOCKS (up to {max(diffs, key=abs):+.1f} %)"
                    row["different_clocks"] = "yes"
            if "verdict" in row:
                row["verdict"] = verdict
            fusion.append(row)
            print(f"{label:>22s} {d:3d} {c:6d} {s:5d} {cell(fms, '{:.4f}') if fms else '-':>20s} {cell(rat_sep, '{:.3f}') if rat_sep else '-':>22s} {verdict:>58s} {cell(rat_tf, '{:.3f}') if rat_tf else '-':>24s} {cell(rat_tf2, '{:.3f}') if rat_tf2 else '-':>24s} {cell(rat_pl, '{:.3f}') if rat_pl else '-':>26s} {'/'.join(str(x) for x in kf):>14s} {(f"{fa[0]} / {fa[1]}" if fa else "-") + " | " + (f"{sa[0]} / {sa[1]}" if sa else "-"):>32s} {row['fused_tile'] + ' | ' + row['separate_tile']:>26s}")
    if fusion:
        with open(os.path.join(folder, "e16_summary_fusion.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in fusion for k in r}, key=lambda k: list(fusion[0]).index(k) if k in fusion[0] else 99))
            w.writeheader()
            w.writerows(fusion)

    # ---- the numbers for the text, and the compact table of Q2
    gpu = next((row["gpu"] for rows in runs.values() for row in rows.values() if row.get("gpu")), "?")
    text = [f"E16 numbers for the text, {gpu}, {len(runs)} runs ({', '.join(names)}); a value is the median over the runs of the per-run value, with [smallest-largest] over the runs",
            "Q1, the attention core with nothing fused around it: fraction = FKL mma_raw TFLOP/s / TFLOP/s of the best of {flash, cudnn, efficient, flash_attn, triton} at the same configuration (GPU-bound numbers)"]
    for d in sorted({r["d"] for r in core}):
        for c in sorted({r["causal"] for r in core}):
            pts = [r for r in core if r["d"] == d and r["causal"] == c and r.get("fraction_median")]
            if not pts:
                continue
            fr = [(float(r["fraction_median"]), r["seq"]) for r in pts]
            lo, hi = min(fr), max(fr)
            text.append(f"  D={d}, {'causal' if c else 'non-causal'}: fraction of the best baseline from {lo[0]:.2f} (S={lo[1]}) to {hi[0]:.2f} (S={hi[1]}); by S: "
                        + ", ".join(f"{r['seq']}: {float(r['fraction_median']):.2f} [{float(r['fraction_min']):.2f}-{float(r['fraction_max']):.2f}], best {r['best_baseline']}" for r in pts))
    withfrac = [r for r in core if r.get("fraction_median")]
    label = lambda r: f"D={r['d']} {'causal' if str(r['causal']) == '1' else 'non-causal'} S={r['seq']} ({float(r['fraction_median']):.3f})"
    above = [r for r in withfrac if float(r["fraction_median"]) > 1.0]
    within1 = [r for r in withfrac if 0.99 <= float(r["fraction_median"]) <= 1.0]
    one_run = [r for r in withfrac if float(r["fraction_median"]) <= 1.0 and float(r["fraction_max"]) > 1.0]
    if withfrac:
        fr_all = [float(r["fraction_median"]) for r in withfrac]
        text.append(f"FKL mma_raw against the best baseline over all {len(withfrac)} points: fraction from {min(fr_all):.3f} to {max(fr_all):.3f}; ABOVE the best baseline by the median of the runs in {len(above)} of {len(withfrac)} points: " + "; ".join(label(r) for r in above)
                    + f". Within 1 % below it (0.990 to 1.000) in {len(within1)} points: " + "; ".join(label(r) for r in within1)
                    + f". Points where one run is above 1 but the median is not: {len(one_run)}: " + "; ".join(label(r) + f" run maximum {float(r['fraction_max']):.3f}" for r in one_run) + ".")
    text.append("Q2, the work around the pattern: ratio separate / fused (> 1: the fused form is faster), range of the median ratio over the configurations of the group (GPU-bound numbers)")
    for label in dict.fromkeys(r["work"] for r in fusion):
        for key, what in (("fkl_separate_over_fused", "FKL separate kernels / FKL fused"), ("pytorch_form_over_fkl_fused", "PyTorch composition / FKL fused"), ("pytorch_form2_over_fkl_fused", "torch.compile dequant + flash / FKL fused")):
            sel = [r for r in fusion if r["work"] == label and r.get(key + "_median")]
            if not sel:
                continue
            for c in sorted({r["causal"] for r in sel}):
                for d in sorted({r["d"] for r in sel}):
                    g = [r for r in sel if r["causal"] == c and r["d"] == d]
                    if g:
                        vals = [(float(r[key + "_median"]), r["seq"]) for r in g]
                        text.append(f"  {label}, {what}, D={d}, {'causal' if c else 'non-causal'}: {min(vals)[0]:.3f} (S={min(vals)[1]}) to {max(vals)[0]:.3f} (S={max(vals)[1]})")
    text.append("Q2, sign and size of the fused-against-separate difference at every S (separate time / fused time, median [smallest-largest] over the runs; verdict: which form is faster, and whether the run medians of the two forms do not overlap = exceeds the spread)")
    for label in dict.fromkeys(r["work"] for r in fusion):
        for c in sorted({r["causal"] for r in fusion if r["work"] == label}):
            for d in sorted({r["d"] for r in fusion if r["work"] == label}):
                g = sorted((r for r in fusion if r["work"] == label and r["causal"] == c and r["d"] == d and r.get("verdict")), key=lambda r: r["seq"])
                if g:
                    text.append(f"  {label}, D={d}, {'causal' if c else 'non-causal'}: " + "; ".join(
                        f"S={r['seq']}: {float(r['fkl_separate_over_fused_median']):.3f} [{float(r['fkl_separate_over_fused_min']):.3f}-{float(r['fkl_separate_over_fused_max']):.3f}] {r['verdict']}" for r in g))
    text.append("Global bytes of one call, COMPUTED from the sizes and not measured (compulsory traffic, every element once; re-reads across query tiles are not counted); t = BH x S tokens, d = head dimension: "
                "prologue fused (int8 or e4m3 K and V read once, fp32 scales) 2 t d + 8 t; prologue unfused (dequantisation kernel reads them, writes bf16; attention reads bf16) 10 t d + 8 t; PyTorch composition (int8 and fp32 scales read, bf16 written, flash reads bf16) 10 t d + 8 t; "
                "raw bf16 K and V 4 t d; epilogue (output O): fused 2 t d (written once), separate 6 t d (written, read, written). Per configuration in the columns *_global_bytes_computed of e16_summary_fusion.csv.")
    flagged = [r for r in fusion if r.get("different_clocks")]
    text.append(f"SM clock check: {sum(1 for r in fusion if 'clock_diff_percent_largest' in r)} fused/separate comparisons have the SM clock of both members from the sampler; {len(flagged)} of them differ by more than {CLOCK_TOLERANCE:.0f} % and are marked 'DIFFERENT CLOCKS' "
                "in the table and the CSV" + ("" if not flagged else ": " + "; ".join(f"{r['work']} D={r['d']} S={r['seq']} causal={r['causal']} ({r['clock_diff_percent_largest']} %)" for r in flagged[:12]) + (" ..." if len(flagged) > 12 else "")))
    flagged_core = [r for r in core if r.get("different_clocks")]
    text.append(f"SM clock check of the core: {len(flagged_core)} of {len(core)} configurations have an SM clock of mma_raw that differs by more than {CLOCK_TOLERANCE:.0f} % from the one of the best baseline.")
    text.append("Fused-prologue pairs: in-kernel shared-memory dequantisation (cp.async of the quantised bytes) against a separate dequantisation kernel that writes bf16 to global memory; the two attention kernels differ in schedule details and tile sizes (block_q, block_kv), so these pairs are not a pure fusion effect. Only the epilogue pair (same kernel and schedule) is.")
    text.append("dequant_flash, dequant_compiled_flash and epilogue_flash are compositions of off-the-shelf pieces that a user has today, not a hand-fused quantised-cache attention kernel and not the best that could be written for these cases.")
    with open(os.path.join(folder, "e16_text_numbers.txt"), "w") as f:
        f.write("\n".join(text) + "\n")
    print("\n" + "\n".join(text))
    reps = [tuple(int(x) for x in item.split(":")) for item in args.rep.split(",") if item]
    compact = []
    for label in dict.fromkeys(r["work"] for r in fusion):
        for d, s, c in reps:
            r = next((x for x in fusion if x["work"] == label and x["d"] == d and x["seq"] == s and x["causal"] == c), None)
            if r:
                compact.append({"gpu": gpu, "work": label, "d": d, "seq": s, "causal": c, "fkl_fused_ms": r["fused_ms_median"], "fkl_separate_ms": r["separate_ms_median"], "ratio_separate_over_fused": r.get("fkl_separate_over_fused_median", ""),
                                "ratio_min": r.get("fkl_separate_over_fused_min", ""), "ratio_max": r.get("fkl_separate_over_fused_max", ""), "verdict": r.get("verdict", ""), "kernels_fused": r["kernels_fused"], "kernels_separate": r["kernels_separate"],
                                "pytorch_composition_ms": r["pytorch_form_ms_median"], "pytorch_composition_kernels": r["kernels_pytorch_form"], "pytorch_over_fkl_fused": r.get("pytorch_form_over_fkl_fused_median", ""),
                                "pytorch_compiled_ms": r["pytorch_form2_ms_median"], "pytorch_compiled_kernels": r["kernels_pytorch_form2"], "pytorch_compiled_over_fkl_fused": r.get("pytorch_form2_over_fkl_fused_median", ""),
                                "fused_regs": r["fused_regs"], "fused_local_B": r["fused_local_B"], "separate_regs": r["separate_regs"], "separate_local_B": r["separate_local_B"],
                                "fused_tile": r["fused_tile"], "separate_tile": r["separate_tile"], "fused_global_bytes_computed": r["fused_global_bytes_computed"], "separate_global_bytes_computed": r["separate_global_bytes_computed"],
                                "pytorch_global_bytes_computed": r.get("pytorch_global_bytes_computed", ""),
                                "fused_sm_mhz": r.get("fused_sm_mhz", ""), "separate_sm_mhz": r.get("separate_sm_mhz", ""), "clock_diff_percent_median": r.get("clock_diff_percent_median", ""), "clock_diff_percent_largest": r.get("clock_diff_percent_largest", ""),
                                "different_clocks": r.get("different_clocks", ""), "ratio_equal_clock_median": r.get("ratio_equal_clock_median", ""), "ratio_equal_clock_min": r.get("ratio_equal_clock_min", ""), "ratio_equal_clock_max": r.get("ratio_equal_clock_max", "")})
    if compact:
        with open(os.path.join(folder, "e16_q2_compact.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(compact[0]))
            w.writeheader()
            w.writerows(compact)

    # ---- kernels of one call, and the errors
    first = runs[names[0]]
    kernels = []
    print("\nKERNELS of one call (first run): variant, D, S, causal: count; per kernel registers / local bytes / static+dynamic shared bytes / blocks per SM")
    for (v, d, s, c), row in sorted(first.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][3], kv[0][2])):
        facts = row.get("kernel_facts", "")
        parsed = []
        for part in [p for p in facts.split(" ; ") if p.strip()]:
            m = re.search(r"regs (\d+) local_B (\d+) static_smem_B (\d+) dynamic_smem_B (\d+) blocks_per_SM (\d+)", part)
            parsed.append(f"{part[:44].strip()}: {m.group(1)} / {m.group(2)} / {int(m.group(3)) + int(m.group(4))} / {m.group(5)}" if m else part[:70])
        worst = max((float(r[(v, d, s, c)]["max_abs_err"]) for r in runs.values() if (v, d, s, c) in r and r[(v, d, s, c)].get("max_abs_err") not in (None, "")), default=None)
        grid_x = re.search(r"FlashAttention\S* grid (\d+)x", facts)
        kernels.append({"variant": v, "d": d, "seq": s, "causal": c, "bh": row.get("bh", ""), "kernels": row.get("kernels", ""), "query_tile_BQ": (s // int(grid_x.group(1))) if grid_x and "Mma" in facts else "",
                        "max_abs_err_over_runs": "" if worst is None else f"{worst:.6g}", "kernel_facts": facts})
        if s == 2048 and c == 0:   # one representative size in the printout, all of them in the CSV
            print(f"  {v:22s} D={d:3d}: {row.get('kernels', '?'):>2s} kernel(s); " + " | ".join(parsed) + ("" if worst is None else f"; largest error {worst:.3g}"))
    with open(os.path.join(folder, "e16_summary_kernels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "d", "seq", "causal", "bh", "kernels", "query_tile_BQ", "max_abs_err_over_runs", "kernel_facts"])
        w.writeheader()
        w.writerows(kernels)


if __name__ == "__main__":
    main()
