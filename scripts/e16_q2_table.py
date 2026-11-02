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
"""E16, the Q2 table of the paper (one column): the work around the attention pattern, fused against separate kernels, three sequence lengths as columns and the head sizes as row blocks.

  python scripts/e16_q2_table.py results/rtx4090/e16-attention --out-dir figures [--seqs 512,2048,8192] [--causal]

Reads e16_summary_fusion.csv of scripts/e16_summary.py and writes e16_q2_table_r2.csv (long form, every number with its run-to-run range and the clock mark) and e16_q2_table_r2.txt (the sidecar with the one-column layout and the notes).
Rows per head size: the epilogue o*a+b (separate time / fused time), the int8 K and V (separate / fused), the fp8 K and V (CSV and sidecar only), the kernels per call (fused / separate) and the computed global megabytes of K and V
(fused / unfused; they do not depend on S: BH x S is the same for every S). A ratio above 1 means that the fused form is faster. Medians over the runs of the run median with [smallest-largest of the run medians];
'*' = the SM clocks of the two forms differ by more than 2 % in some run (the ratio at equal clock is in e16_summary_fusion.csv).
"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seqs", default="512,2048,8192")
    ap.add_argument("--causal", action="store_true", help="add a block per head size for the causal case")
    a = ap.parse_args()
    seqs = [int(x) for x in a.seqs.split(",")]
    fus = list(csv.DictReader(open(os.path.join(a.folder, "e16_summary_fusion.csv"))))
    gpu = next(iter(csv.DictReader(open(os.path.join(a.folder, "e16_q2_compact.csv")))))["gpu"]

    def cell(work, d, s, c, key="fkl_separate_over_fused"):
        r = next((x for x in fus if x["work"] == work and int(x["d"]) == d and int(x["seq"]) == s and int(x["causal"]) == c), None)
        if not r:
            return None
        return dict(median=float(r[key + "_median"]), lo=float(r[key + "_min"]), hi=float(r[key + "_max"]), star=bool(r.get("different_clocks")), exceeds="exceeds" in r["verdict"], r=r)

    long_rows, lines = [], []
    works = [("epilogue o*a+b", "epilogue o*a+b"), ("int8 K,V prologue", "int8 K,V"), ("fp8 e4m3 K,V prologue", "fp8 K,V")]
    header = f"{'':34s}" + "".join(f"{'S = ' + str(s):>24s}" for s in seqs)
    for c in ([0, 1] if a.causal else [0]):
        for d in (64, 128):
            lines.append(f"D = {d}, {'causal' if c else 'non-causal'}")
            lines.append(header)
            kv_bytes = None
            for work, label in works:
                cells = [cell(work, d, s, c) for s in seqs]
                txt = ""
                for cl in cells:
                    txt += f"{(f'{cl['median']:.2f} [{cl['lo']:.2f}-{cl['hi']:.2f}]' + ('*' if cl['star'] else '')) if cl else '-':>24s}"
                lines.append(f"  {label + ' (separate / fused)':32s}{txt}")
                for s, cl in zip(seqs, cells):
                    if cl:
                        r = cl["r"]
                        long_rows.append({"gpu": gpu, "head_dim": d, "mask": "causal" if c else "non-causal", "seq": s, "work": work, "ratio_separate_over_fused_median": f"{cl['median']:.4f}", "ratio_min": f"{cl['lo']:.4f}", "ratio_max": f"{cl['hi']:.4f}",
                                          "different_clocks": "yes" if cl["star"] else "", "exceeds_spread": "yes" if cl["exceeds"] else "no", "kernels_fused": r["kernels_fused"], "kernels_separate": r["kernels_separate"],
                                          "global_MB_fused_computed": f"{float(r['fused_global_bytes_computed']) / 1e6:.1f}", "global_MB_separate_computed": f"{float(r['separate_global_bytes_computed']) / 1e6:.1f}",
                                          "fused_ms": r["fused_ms_median"], "separate_ms": r["separate_ms_median"], "fused_tile": r["fused_tile"], "separate_tile": r["separate_tile"],
                                          "fused_regs": r["fused_regs"], "fused_local_B": r["fused_local_B"], "separate_regs": r["separate_regs"], "separate_local_B": r["separate_local_B"]})
                        if work.startswith("int8") and kv_bytes is None:
                            kv_bytes = (float(r["fused_global_bytes_computed"]) / 1e6, float(r["separate_global_bytes_computed"]) / 1e6)
            first = next((x for x in fus if int(x["d"]) == d and int(x["causal"]) == c and x["work"].startswith("int8")), None)
            lines.append(f"  {'kernels per call (fused / sep.)':32s}" + "epilogue 1 / 2; K,V prologue 1 / 2 (at every S)")
            if kv_bytes:
                lines.append(f"  {'K,V global MB (fused / unfused)':32s}{kv_bytes[0]:.0f} / {kv_bytes[1]:.0f}  (computed, not measured; the same at every S)")
            lines.append("")
    with open(os.path.join(a.out_dir, "e16_q2_table_r2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)
    notes = f"""e16_q2_table_r2.csv, e16_q2_table_r2.txt: the Q2 table of the paper for the {gpu}, drawn by scripts/e16_q2_table.py from {a.folder} (3 complete runs; e16_summary_fusion.csv holds every configuration).
LAYOUT FOR ONE COLUMN (what fits): the three sequence lengths as columns and the two head sizes as row blocks, five rows per block (epilogue ratio, int8 ratio, kernels per call, K,V megabytes; the fp8 ratio stays in the CSV and here). Six rows (head size x sequence length) with one column per quantity
do not fit one column once the run-to-run ranges are printed (a ratio with its range is about 19 characters); the layout below does, at about 6.5 pt, with the ranges in the cells.
Ratios are separate time / fused time (> 1: the fused form is faster), median over the 3 runs of the run median with [smallest-largest of the run medians]; '*' = the SM clocks of the two forms differ by more than 2 % in some run. All numbers GPU-bound.
KERNELS per call: epilogue 1 fused / 2 separate; K,V prologue 1 fused / 2 separate. The PyTorch compositions have 3 kernels (int8: two elementwise + flash) and 2 kernels (epilogue: flash + addcmul); torch.compile'd dequantisation also 3 (two Triton kernels, one for K and one for V, then flash).
K,V GLOBAL MB (COMPUTED from the sizes, NOT measured; compulsory traffic, every element counted once, K and V together, BH x S = 262,144 tokens at every S): fused int8 or e4m3 read once with fp32 per-token scales; unfused = the dequantisation kernel reads them, writes bf16, and the attention reads bf16.
What differs between the two forms: for the epilogue pair only the epilogue (same attention kernel and schedule). For the prologue pairs the fused kernel copies the raw quantised bytes with cp.async and dequantises them from shared memory to shared memory (tile BLOCK_Q x 32), the unfused form runs a dequantisation kernel and then the bf16 cp.async kernel (tile BLOCK_Q x 64 or 128):
the two attention kernels differ in schedule details and tile size as well as in fusion. No cause of the prologue result is stated; registers (174 against 254 at D = 128, 196 against 244 or 158 at D = 64), local bytes (0) and blocks per SM do not distinguish the two kernels.
Power: the RTX 4090 reaches its 450 W limit in the long runs; the median SM clock of the FKL raw kernel is 2588 MHz against 2640 to 2655 MHz for cuDNN, flash-attn and Triton in the Q1 rows (see e16_core_r2.txt); marks '*' show where the two forms of a pair differ by more than 2 %.

"""
    open(os.path.join(a.out_dir, "e16_q2_table_r2.txt"), "w").write(notes + "\n".join(lines) + "\n")
    # the compact table: the representative rows of e16_q2_compact.csv with the clock of both members and the ratio at equal clock
    comp = list(csv.DictReader(open(os.path.join(a.folder, "e16_q2_compact.csv"))))
    with open(os.path.join(a.out_dir, "e16_q2_compact_r2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0].keys()))
        w.writeheader()
        w.writerows(comp)
    clk = []
    for r in comp:
        sign_raw = "faster" if float(r["ratio_separate_over_fused"]) > 1 else "slower"
        sign_eq = "faster" if r["ratio_equal_clock_median"] and float(r["ratio_equal_clock_median"]) > 1 else "slower"
        clk.append(f"  {r['work']:22s} D={r['d']} S={r['seq']:>4s}: fused {r['fused_sm_mhz']} MHz, separate {r['separate_sm_mhz']} MHz, difference {float(r['clock_diff_percent_median']):+.1f} % (largest of the runs {float(r['clock_diff_percent_largest']):+.1f} %)"
                   + ("  MORE THAN 2 % in some run" if r.get("different_clocks") else "  within 2 %") + f"; ratio {float(r['ratio_separate_over_fused']):.3f} [{float(r['ratio_min']):.3f}-{float(r['ratio_max']):.3f}] (fused {sign_raw}), at equal clock {float(r['ratio_equal_clock_median']):.3f} "
                   f"[{float(r['ratio_equal_clock_min']):.3f}-{float(r['ratio_equal_clock_max']):.3f}] (fused {sign_eq}): " + ("same sign" if sign_raw == sign_eq else "DIFFERENT SIGN"))
    ctxt = f"""e16_q2_compact_r2.csv: the compact Q2 table of the paper for the {gpu}: D = 128, non-causal, S = 2048 and 8192, drawn by scripts/e16_summary.py (--rep 128:2048:0,128:8192:0) and scripts/e16_q2_table.py from {a.folder} (3 complete runs); one row per work (int8 and fp8 prologue, epilogue) and S.
Every column is explained in e16_q2_table_r2.txt; this file adds, per row, the SM clock of the two members of the pair (fused_sm_mhz, separate_sm_mhz: median of the row medians over the 3 runs, from the 100 ms sampler), their difference (clock_diff_percent_median, clock_diff_percent_largest: the largest difference of any run), the flag different_clocks (more than 2 % in some run) and the
ratio at equal SM clock (ratio_equal_clock_*: separate time x separate clock over fused time x fused clock, i.e. in SM cycles).
SM CLOCK OF THE TWO MEMBERS, row by row ({sum(1 for r in comp if r.get('different_clocks'))} of {len(comp)} rows differ by more than 2 % in some run) and the sign at equal clock:
""" + "\n".join(clk) + """
The RTX 4090 is power-limited at 450 W in the long runs, so the clock differs between kernels with different power draw; the raw ratio is what the GPU does, the equal-clock ratio shows what remains if the clocks were equal. No cause of the prologue result is stated.
"""
    open(os.path.join(a.out_dir, "e16_q2_compact_r2.txt"), "w").write(ctxt)
    print(ctxt)
    print(notes + "\n".join(lines))


if __name__ == "__main__":
    main()
