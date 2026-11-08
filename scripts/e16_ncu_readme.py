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
"""E16, the Nsight Compute pass: write the README of results/<gpu>/e16-attention/ncu from the exports of scripts/e16_ncu_summary.py and scripts/e16_ncu_sass.py (every number in it is computed here from the CSV files).

  python scripts/e16_ncu_readme.py results/rtx4090/e16-attention/ncu
"""
import csv
import os
import re
import statistics as stats
import sys

IDS = [("0", "mma_raw attention (bf16 path)"), ("1", "mma_int8kv_fused attention"), ("2", "dequantKV (int8 unfused)"), ("3", "mma_int8kv_unfused attention (bf16 path)"), ("4", "mma_fp8kv_fused attention"),
       ("5", "mma_epilogue_fused attention"), ("6", "mma_epilogue_separate attention (bf16 path)"), ("7", "affineBf16 (epilogue separate)")]
CONTROL = ["0", "3", "6"]   # the same instantiation, three launches


def num(text):
    try:
        return float(str(text).replace(",", "").strip())
    except ValueError:
        return None


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    folder = sys.argv[1]
    summ = {r["id"]: r for r in csv.DictReader(open(os.path.join(folder, "ncu_summary.csv")))}
    raw_rows = list(csv.reader([l for l in open(os.path.join(folder, "ncu_raw.csv")) if not l.startswith("==")]))
    head = raw_rows[0]
    raw = {str(i): r for i, r in enumerate(raw_rows[2:])}

    def m(i, name):
        return num(raw[i][head.index(name)]) if name in head else None

    ops = {}
    for r in csv.DictReader(open(os.path.join(folder, "ncu_opcodes.csv"))):
        ops.setdefault(r["kernel"], {})[r["opcode"]] = (float(r["warp_instructions_executed"]), float(r["stall_samples"]))
    label = dict(IDS)

    def count(i, op):
        return ops[label[i]].get(op, (0.0, 0.0))[0]

    def samples(i):
        return sum(v[1] for v in ops[label[i]].values())

    def spread(values):
        return (max(values) - min(values)) / stats.mean(values) * 100.0 if stats.mean(values) else 0.0

    dur = {i: m(i, "gpu__time_duration.sum") / 1000.0 for i, _ in IDS}   # us
    inst = {i: m(i, "smsp__inst_executed.sum") for i, _ in IDS}
    # the kernels
    krows = []
    for i, name in IDS:
        kn = summ[i]["kernel_name"]
        tile = re.search(r", (\d+), (\d+), (\d+), fk::NoScoreMod", kn)
        reads = re.findall(r"ReadInstantiableOperation<fk::(\w+)>", kn)
        epi = "Mul+Add" if "Mul<" in kn else ("identity" if "AttentionIdentityEpilogue" in kn else "-")
        krows.append([i, name, f"{tile.group(1)} x {tile.group(2)} x {tile.group(3)}" if tile else "-", ", ".join(r.replace("AttentionRead", "").replace("TokenDequantRead", "") for r in reads) if reads else "-", epi,
                      summ[i]["registers_per_thread"], summ[i]["block_size"], summ[i]["grid_size"], summ[i]["dynamic_smem_per_block"]])
    ktable = md(["id", "kernel", "tile BQ x BKV x warps", "Q, K, V reads", "epilogue", "registers", "block", "grid", "dynamic smem B"], krows)
    # time, instructions, issue
    trows = []
    for i, name in IDS:
        s = summ[i]
        tens = m(i, "sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_active")
        trows.append([i, name, f"{dur[i]:.1f}", f"{inst[i] / 1e6:.1f}", f"{count(i, 'HMMA') / 1e6:.1f}", s["issue_slots_busy_pct"], s["executed_ipc_active"], s["no_eligible_pct"], f"{tens:.1f}" if tens is not None else "-", f"{tens * dur[i] / 100:.1f}" if tens else "-",
                      s["warp_cycles_per_issued_instruction"], s["active_warps_per_scheduler"], s["theoretical_occupancy_pct"], s["achieved_occupancy_pct"]])
    ttable = md(["id", "kernel", "duration us", "warp instr. M", "HMMA M", "issue slots busy %", "IPC", "no eligible %", "tensor pipe active %", "tensor busy us", "warp cycles / issued instr.", "active warps / scheduler", "theor. occ. %", "achieved occ. %"], trows)
    # stalls
    strows = []
    for i, name in IDS:
        st = sorted(((m(i, h), re.sub(r"^smsp__average_warps_issue_stalled_|_per_issue_active\.ratio$", "", h)) for h in head if re.match(r"smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio$", h) and m(i, h)), reverse=True)[:5]
        tot = samples(i)
        top = sorted(ops[label[i]].items(), key=lambda kv: -kv[1][1])[:6]
        strows.append([i, name, ", ".join(f"{n} {v:.2f}" for v, n in st), ", ".join(f"{o} {100 * v[1] / tot:.1f}%" for o, v in top), f"{tot:,.0f}"])
    sttable = md(["id", "kernel", "stall reasons, warp cycles per issued instruction (top 5)", "stall samples by opcode (top 6)", "samples"], strows)
    # memory
    mrows = []
    for i, name in IDS:
        rd, wr = m(i, "dram__bytes_read.sum") / 1e6, m(i, "dram__bytes_write.sum") / 1e6
        wf, bc = m(i, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"), m(i, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")
        mrows.append([i, name, f"{rd:.1f}", f"{wr:.1f}", f"{rd + wr:.1f}", summ[i]["dram_throughput_pct"], summ[i]["l1_hit_rate_pct"], summ[i]["l2_hit_rate_pct"], f"{wf / 1e6:.1f}", f"{bc / 1e6:.2f}", f"{100 * bc / wf:.2f}"])
    mtable = md(["id", "kernel", "DRAM read MB", "DRAM write MB", "read + write MB", "DRAM throughput %", "L1 hit %", "L2 hit %", "shared wavefronts M", "bank conflicts M", "conflicts % of wavefronts"], mrows)
    # opcodes
    otable = open(os.path.join(folder, "ncu_opcodes.md")).read().strip()
    # controls and pairs
    c_dur = [dur[i] for i in CONTROL]
    ctl = (f"Three launches of the same instantiation (ids 0, 3, 6): duration {c_dur[0]:.1f} / {c_dur[1]:.1f} / {c_dur[2]:.1f} us, spread {spread(c_dur):.2f} %; executed instructions {spread([inst[i] for i in CONTROL]):.4f} %; "
           f"DRAM bytes {spread([m(i, 'dram__bytes_read.sum') + m(i, 'dram__bytes_write.sum') for i in CONTROL]):.2f} %; shared wavefronts {spread([m(i, 'l1tex__data_pipe_lsu_wavefronts_mem_shared.sum') for i in CONTROL]):.3f} %; "
           f"shared bank conflicts {spread([m(i, 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum') for i in CONTROL]):.1f} %; issue slots busy {min(num(summ[i]['issue_slots_busy_pct']) for i in CONTROL)} to {max(num(summ[i]['issue_slots_busy_pct']) for i in CONTROL)} %.")
    base = stats.mean(c_dur)
    pair_i8, pair_epi = dur["2"] + dur["3"], dur["6"] + dur["7"]
    fus = {(r["work"], r["d"], r["seq"], r["causal"]): r for r in csv.DictReader(open(os.path.join(folder, "..", "e16_summary_fusion.csv")))}
    b_i8, b_epi = fus[("int8 K,V prologue", "128", "2048", "0")], fus[("epilogue o*a+b", "128", "2048", "0")]
    pairs = (f"- bf16 attention kernel (mean of ids 0, 3, 6): {base:.1f} us. Fused int8 {dur['1']:.1f} us = x{dur['1'] / base:.3f}; fused fp8 {dur['4']:.1f} us = x{dur['4'] / base:.3f}; epilogue-fused {dur['5']:.1f} us = {100 * (dur['5'] / base - 1):+.2f} % "
             f"({'within' if abs(100 * (dur['5'] / base - 1)) <= spread(c_dur) else 'outside'} the {spread(c_dur):.2f} % spread of the identical launches).\n"
             f"- int8 pair: dequantKV {dur['2']:.1f} + attention {dur['3']:.1f} = {pair_i8:.1f} us against the fused kernel {dur['1']:.1f} us: separate / fused = {pair_i8 / dur['1']:.4f} (the fused kernel is {100 * (dur['1'] / pair_i8 - 1):.1f} % slower). "
             f"The benchmark window at the same configuration (D = 128, S = 2048, non-causal, boost clock, warm caches): {num(b_i8['fkl_separate_over_fused_median']):.4f} [{num(b_i8['fkl_separate_over_fused_min']):.4f}-{num(b_i8['fkl_separate_over_fused_max']):.4f}].\n"
             f"- epilogue pair: attention {dur['6']:.1f} + affineBf16 {dur['7']:.1f} = {pair_epi:.1f} us against the fused kernel {dur['5']:.1f} us: separate / fused = {pair_epi / dur['5']:.4f}; benchmark window {num(b_epi['fkl_separate_over_fused_median']):.4f} [{num(b_epi['fkl_separate_over_fused_min']):.4f}-{num(b_epi['fkl_separate_over_fused_max']):.4f}].\n"
             f"- tensor work: HMMA executed {count('0', 'HMMA'):,.0f} (mma_raw) and {count('1', 'HMMA'):,.0f} (fused int8) and {count('4', 'HMMA'):,.0f} (fused fp8); LDSM {count('0', 'LDSM'):,.0f} in all three.\n"
             f"- warp instructions executed: mma_raw {inst['0'] / 1e6:.1f} M, fused int8 {inst['1'] / 1e6:.1f} M (x{inst['1'] / inst['0']:.3f}, {(inst['1'] - inst['0']) / 1e6:+.1f} M), fused fp8 {inst['4'] / 1e6:.1f} M (x{inst['4'] / inst['0']:.3f}, {(inst['4'] - inst['0']) / 1e6:+.1f} M), "
             f"dequantKV {inst['2'] / 1e6:.1f} M, epilogue-fused {inst['5'] / 1e6:.1f} M ({(inst['5'] - inst['0']) / 1e6:+.1f} M).\n"
             f"- I2F executed: fused int8 {count('1', 'I2F'):,.0f}, dequantKV {count('2', 'I2F'):,.0f}: ratio {count('1', 'I2F') / count('2', 'I2F'):.1f}; the number of query tiles per head is S / BLOCK_Q = 2048 / 64 = {2048 // 64}.\n"
             f"- DRAM (read + write): fused int8 {(m('1', 'dram__bytes_read.sum') + m('1', 'dram__bytes_write.sum')) / 1e6:.1f} MB; unfused pair {(m('2', 'dram__bytes_read.sum') + m('2', 'dram__bytes_write.sum') + m('3', 'dram__bytes_read.sum') + m('3', 'dram__bytes_write.sum')) / 1e6:.1f} MB. "
             f"Reads: mma_raw {m('0', 'dram__bytes_read.sum') / 1e6:.1f} MB (Q + K + V in bf16 are 3 x 67.1 MB); fused int8 {m('1', 'dram__bytes_read.sum') / 1e6:.1f} MB, that is Q (67.1 MB) plus {m('1', 'dram__bytes_read.sum') / 1e6 - 67.1:.1f} MB of K and V as int8 with their scales "
             f"(computed in `../e16_summary_fusion.csv`: 69 MB). The writes are lower than the O (67.1 MB) and the bf16 K,V of dequantKV (134.2 MB) because lines still dirty in the 72 MiB L2 at the end of a kernel are not counted by the DRAM counters.")
    # the rules of the details page, verbatim, for ids 0, 1 and 2
    rules, seen = [], set()
    for r in csv.DictReader([l for l in open(os.path.join(folder, "ncu_details.csv")) if not l.startswith("==")]):
        if r["Rule Name"] and r["ID"] in ("0", "1") and r["Rule Type"] == "OPT" and (r["ID"], r["Rule Name"]) not in seen:
            seen.add((r["ID"], r["Rule Name"]))
            rules.append(f"- id {r['ID']} `{r['Rule Name']}` (the tool's estimated speedup {r['Estimated Speedup'] or '-'} %): {r['Rule Description'].strip()}")
    head_path = os.path.join(folder, "README_head.md")   # the fixed text of the README: question, what ran, how to read it, how to reproduce
    head_text = open(head_path).read() if os.path.exists(head_path) else ""
    scaling_path = os.path.join(folder, "..", "e16_prologue_scaling.md")   # scripts/e16_prologue_scaling.py, from the benchmark runs
    scaling = open(scaling_path).read().split("### ")[1] if os.path.exists(scaling_path) else ""
    scaling = scaling.split("\n", 1)[1].strip() if scaling else ""
    tail_path = os.path.join(folder, "README_tail.md")   # the fixed text after the tables: what the scaling shows, the source lines, the status of the claim
    tail_text = open(tail_path).read() if os.path.exists(tail_path) else ""
    out = [head_text.rstrip(), "",
           "## The eight kernels (launch order of the pass)", "", ktable, "",
           "## Time, instructions, issue and occupancy", "", ttable, "",
           "## Stalls (Nsight Compute warp-state statistics and PC sampling)", "", sttable, "",
           "## Executed warp instructions by SASS opcode (top 16 over all kernels; every opcode is in `ncu_opcodes.csv`, with its modifiers in `ncu_opcodes_full.csv`)", "", otable, "",
           "## Memory (DRAM bytes as the DRAM counters saw them, L1 and L2 hit rates, shared-memory wavefronts and bank conflicts)", "", mtable, "",
           "## Controls and ratios", "", ctl, "", pairs, "",
           "## Scaling of the extra time of the fused int8 kernel with S (COMPUTED FROM THE BENCHMARK RUNS, not from Nsight Compute)", "",
           "From the three complete runs of the E16 harness (`../run*/e16_attention.csv`: GPU time per call at the boost clock, back-to-back calls, BH x S = 262,144 tokens at every S, non-causal): per run the time of `mma_int8kv_fused` minus that of `mma_raw`, and of `mma_int8kv_unfused` minus that of `mma_raw` "
           "(the separate dequantisation kernel and the effect of its bf16 output), then the median over the runs of the per-run difference with [smallest-largest]; `scripts/e16_prologue_scaling.py` writes the table for both head sizes, both masks and fp8 as well (`../e16_prologue_scaling.csv`, `../e16_prologue_scaling.md`).", "",
           scaling, "", tail_text.rstrip(), "",
           "## What Nsight Compute's own rules said (verbatim, optimisation hints of the tool for ids 0 and 1; they are heuristics of the tool, not results of this study)", ""] + rules + [""]
    open(os.path.join(folder, "README.md"), "w").write("\n".join(out))
    print(os.path.join(folder, "README.md"), len("\n".join(out).splitlines()), "lines")


if __name__ == "__main__":
    main()
