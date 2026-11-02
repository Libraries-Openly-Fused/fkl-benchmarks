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
"""E16, the Nsight Compute pass: executed warp instructions and warp-stall samples of every kernel by SASS opcode.

  python scripts/e16_ncu_sass.py <report.ncu-repz> results/rtx4090/e16-attention/ncu [--ncu <path of ncu>]

For each of the 8 kernels of the pass the source page of the report is exported (`ncu --import <report> --page source --print-source sass --csv --kernel-id ::regex:<pattern>:<n>`: one row per SASS
instruction with 'Instructions Executed' and the warp-stall samples of the PC sampling) and summed by opcode. Writes
  ncu_opcodes.csv       kernel, opcode (without modifiers), warp instructions executed, share of the kernel's instructions, stall samples, share of the kernel's samples
  ncu_opcodes_full.csv  the same by opcode with its modifiers (LDGSTS.E.BYPASS.LTC128B.64, LDS.U8, ...)
  ncu_opcodes.md        the table of the README: the opcodes with the largest counts, one column per kernel
The sum of 'Instructions Executed' of a kernel must equal smsp__inst_executed.sum of the raw page; a mismatch is printed. No metric is interpreted here.
"""
import argparse
import csv
import io
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict

# (label, kernel-name regex, nth launch matching it): the launch order of the pass (see scripts/e16_ncu_summary.py)
KERNELS = [("mma_raw attention (bf16 path)", "FlashAttentionMma", 1), ("mma_int8kv_fused attention", "FlashAttentionMma", 2), ("dequantKV (int8 unfused)", "dequantKV", 1),
           ("mma_int8kv_unfused attention (bf16 path)", "FlashAttentionMma", 3), ("mma_fp8kv_fused attention", "FlashAttentionMma", 4), ("mma_epilogue_fused attention", "FlashAttentionMma", 5),
           ("mma_epilogue_separate attention (bf16 path)", "FlashAttentionMma", 6), ("affineBf16 (epilogue separate)", "affineBf16", 1)]


def number(text):
    try:
        return float(str(text).replace(",", "").strip())
    except ValueError:
        return None


def export(ncu, report, pattern, nth):
    cmd = ["taskset", "-c", "8-15,24-31", ncu, "--import", report, "--page", "source", "--print-source", "sass", "--csv", "--kernel-id", f"::regex:{pattern}:{nth}"]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def parse(text):
    """([(opcode with modifiers, executed, samples)], kernel name) from the csv of the source page."""
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("==")]
    name = ""
    start = None
    for i, l in enumerate(lines):
        if l.startswith('"Kernel Name"'):
            name = next(csv.reader([l]))[1]
        if l.startswith('"Address","Source"'):
            start = i
            break
    if start is None:
        raise SystemExit("no source table in the export")
    rows = []
    for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        src = r["Source"].strip()
        executed = number(r.get("Instructions Executed", ""))
        if not src or executed is None:
            continue
        src = re.sub(r"^@!?U?P[T0-9]+\s+", "", src)   # the predicate
        samples = number(r.get("Warp Stall Sampling (All Samples)", "")) or 0.0
        rows.append((src.split()[0].rstrip(";"), executed, samples))
    return rows, name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report")
    ap.add_argument("folder")
    ap.add_argument("--ncu", default=os.path.join(os.environ.get("FKLB_CUDA_HOME", "/home/oscar/tfkl-bench/toolchains/cuda-13.4.3"), "bin", "ncu"))
    a = ap.parse_args()
    raw_totals = {}
    raw_path = os.path.join(a.folder, "ncu_raw.csv")
    if os.path.exists(raw_path):
        rows = list(csv.reader([l for l in open(raw_path) if not l.startswith("==")]))
        col = rows[0].index("smsp__inst_executed.sum")
        raw_totals = {i: number(r[col]) for i, r in enumerate(rows[2:])}
    per_kernel, full = OrderedDict(), OrderedDict()
    for i, (label, pattern, nth) in enumerate(KERNELS):
        rows, name = parse(export(a.ncu, a.report, pattern, nth))
        base, withmod, base_s = Counter(), Counter(), Counter()
        for op, executed, samples in rows:
            base[op.split(".")[0]] += executed
            withmod[op] += executed
            base_s[op.split(".")[0]] += samples
        total = sum(base.values())
        ok = "matches" if raw_totals.get(i) == total else f"DOES NOT MATCH smsp__inst_executed.sum {raw_totals.get(i)}"
        print(f"{label:46s} {len(rows):5d} SASS instructions, {total:,.0f} warp instructions executed ({ok}); {name[:60]}")
        per_kernel[label] = (base, base_s)
        full[label] = withmod
    with open(os.path.join(a.folder, "ncu_opcodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "opcode", "warp_instructions_executed", "share_of_instructions", "stall_samples", "share_of_samples"])
        for label, (base, base_s) in per_kernel.items():
            tot, tot_s = sum(base.values()), sum(base_s.values())
            for op, n in base.most_common():
                w.writerow([label, op, f"{n:.0f}", f"{n / tot:.5f}" if tot else "", f"{base_s[op]:.0f}", f"{base_s[op] / tot_s:.5f}" if tot_s else ""])
    with open(os.path.join(a.folder, "ncu_opcodes_full.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "opcode_with_modifiers", "warp_instructions_executed"])
        for label, c in full.items():
            for op, n in c.most_common():
                w.writerow([label, op, f"{n:.0f}"])
    ops = Counter()
    for base, _ in per_kernel.values():
        for op, n in base.items():
            ops[op] += n
    top = [op for op, _ in ops.most_common(16)]
    head = "| opcode | " + " | ".join(l.split(" (")[0] if " (" in l else l for l in per_kernel) + " |"
    lines = [head, "|---|" + "---:|" * len(per_kernel)]
    for op in top:
        lines.append(f"| {op} | " + " | ".join(f"{per_kernel[l][0][op]:,.0f}" for l in per_kernel) + " |")
    lines.append("| all | " + " | ".join(f"{sum(per_kernel[l][0].values()):,.0f}" for l in per_kernel) + " |")
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.folder, "ncu_opcodes.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
