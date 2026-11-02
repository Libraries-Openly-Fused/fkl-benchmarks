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
"""E16, the Nsight Compute pass (results/rtx4090/e16-attention/ncu): tables of the measured metrics of the attention kernels.

  python scripts/e16_ncu_summary.py results/rtx4090/e16-attention/ncu

Reads ncu_details.csv and ncu_raw.csv (`ncu --import <report> --page details|raw --csv --print-units base`) and writes
  ncu_details_tidy.csv   every row of the details page: launch id, kernel label, section, metric, unit, value
  ncu_summary.csv        one row per kernel with the metrics that answer the questions of the pass (the names in SELECTED)
  ncu_summary.md         the same as a table for the README, plus the stall reasons and the pipe utilisations (top entries of the raw page)
The kernel label is the place of the launch in the program (E16_PROFILE=1 launches the kernels in a fixed order) and is checked against the kernel name; a mismatch is printed.
No metric is interpreted here.
"""
import csv
import os
import re
import statistics as stats
import sys

# launch order of the 8 kernels of the pass (E16_VARIANTS=mma_raw,mma_int8kv,mma_epilogue,mma_fp8kv_fused, rotation 0): label and a pattern of the kernel name
EXPECTED = [("mma_raw attention (bf16 path)", "FlashAttentionMma"), ("mma_int8kv_fused attention", "FlashAttentionMma"), ("dequantKV (int8 unfused)", "dequantKV"),
            ("mma_int8kv_unfused attention (bf16 path)", "FlashAttentionMma"), ("mma_fp8kv_fused attention", "FlashAttentionMma"), ("mma_epilogue_fused attention", "FlashAttentionMma"),
            ("mma_epilogue_separate attention (bf16 path)", "FlashAttentionMma"), ("affineBf16 (epilogue separate)", "affineBf16")]

# (column of the summary, section of the details page, metric name of the details page)
SELECTED = [("duration_us", "GPU Speed Of Light Throughput", "Duration"), ("sm_frequency", "GPU Speed Of Light Throughput", "SM Frequency"), ("elapsed_cycles", "GPU Speed Of Light Throughput", "Elapsed Cycles"),
            ("compute_sm_throughput_pct", "GPU Speed Of Light Throughput", "Compute (SM) Throughput"), ("memory_throughput_pct", "GPU Speed Of Light Throughput", "Memory Throughput"),
            ("dram_throughput_pct", "GPU Speed Of Light Throughput", "DRAM Throughput"), ("l1_throughput_pct", "GPU Speed Of Light Throughput", "L1/TEX Cache Throughput"),
            ("l2_throughput_pct", "GPU Speed Of Light Throughput", "L2 Cache Throughput"),
            ("issue_slots_busy_pct", "Compute Workload Analysis", "Issue Slots Busy"), ("executed_ipc_active", "Compute Workload Analysis", "Executed Ipc Active"), ("issued_ipc_active", "Compute Workload Analysis", "Issued Ipc Active"),
            ("sm_busy_pct", "Compute Workload Analysis", "SM Busy"),
            ("one_or_more_eligible_pct", "Scheduler Statistics", "One or More Eligible"), ("no_eligible_pct", "Scheduler Statistics", "No Eligible"), ("active_warps_per_scheduler", "Scheduler Statistics", "Active Warps Per Scheduler"),
            ("eligible_warps_per_scheduler", "Scheduler Statistics", "Eligible Warps Per Scheduler"),
            ("warp_cycles_per_issued_instruction", "Warp State Statistics", "Warp Cycles Per Issued Instruction"), ("executed_instructions", "Instruction Statistics", "Executed Instructions"),
            ("issued_instructions", "Instruction Statistics", "Issued Instructions"),
            ("l1_hit_rate_pct", "Memory Workload Analysis", "L1/TEX Hit Rate"), ("l2_hit_rate_pct", "Memory Workload Analysis", "L2 Hit Rate"), ("mem_busy_pct", "Memory Workload Analysis", "Mem Busy"),
            ("max_bandwidth_pct", "Memory Workload Analysis", "Max Bandwidth"), ("mem_pipes_busy_pct", "Memory Workload Analysis", "Mem Pipes Busy"),
            ("registers_per_thread", "Launch Statistics", "Registers Per Thread"), ("block_size", "Launch Statistics", "Block Size"), ("grid_size", "Launch Statistics", "Grid Size"),
            ("dynamic_smem_per_block", "Launch Statistics", "Dynamic Shared Memory Per Block"), ("static_smem_per_block", "Launch Statistics", "Static Shared Memory Per Block"), ("waves_per_sm", "Launch Statistics", "Waves Per SM"),
            ("theoretical_occupancy_pct", "Occupancy", "Theoretical Occupancy"), ("achieved_occupancy_pct", "Occupancy", "Achieved Occupancy"), ("achieved_active_warps_per_sm", "Occupancy", "Achieved Active Warps Per SM"),
            ("block_limit_registers", "Occupancy", "Block Limit Registers"), ("block_limit_shared_mem", "Occupancy", "Block Limit Shared Mem")]


def number(text):
    try:
        return float(str(text).replace(",", "").strip())
    except ValueError:
        return None


def read_details(path):
    """{launch id: {'name': kernel name, 'rows': [(section, metric, unit, value)]}} from the details page (one row per metric)."""
    kernels = {}
    with open(path, newline="") as f:
        lines = [l for l in f if not l.startswith("==")]   # ncu prints '==PROF==' lines on stdout
    for r in csv.DictReader(lines):
        k = kernels.setdefault(r["ID"], {"name": r["Kernel Name"], "rows": []})
        if r.get("Metric Name"):
            k["rows"].append((r["Section Name"], r["Metric Name"], r["Metric Unit"], r["Metric Value"]))
    return kernels


def read_raw(path):
    """{launch id: {metric name: (unit, value text)}} from the raw page (one column per metric, a row of units under the header)."""
    with open(path, newline="") as f:
        lines = [l for l in f if not l.startswith("==")]
    rows = list(csv.reader(lines))
    head, units = rows[0], rows[1]
    out = {}
    for r in rows[2:]:
        out[r[head.index("ID")]] = {h: (units[i], r[i]) for i, h in enumerate(head) if i < len(r)}
    return out


def main():
    folder = sys.argv[1]
    details = read_details(os.path.join(folder, "ncu_details.csv"))
    raw = read_raw(os.path.join(folder, "ncu_raw.csv")) if os.path.exists(os.path.join(folder, "ncu_raw.csv")) else {}
    ids = sorted(details, key=int)
    print(f"{len(ids)} profiled launches (expected {len(EXPECTED)})")
    labels = {}
    for i, kid in enumerate(ids):
        label, pattern = EXPECTED[i] if i < len(EXPECTED) else (f"launch {kid}", "")
        ok = bool(re.search(pattern, details[kid]["name"])) if pattern else False
        labels[kid] = label
        print(f"  id {kid}: {label:46s} name {'matches' if ok else 'DOES NOT MATCH ' + pattern}: {details[kid]['name'][:150]}")
    with open(os.path.join(folder, "ncu_details_tidy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "kernel", "section", "metric", "unit", "value"])
        for kid in ids:
            for section, metric, unit, value in details[kid]["rows"]:
                w.writerow([kid, labels[kid], section, metric, unit, value])
    table = []
    for kid in ids:
        lookup = {(s, m): (u, v) for s, m, u, v in details[kid]["rows"]}
        row = {"id": kid, "kernel": labels[kid], "kernel_name": details[kid]["name"]}
        for col, section, metric in SELECTED:
            u, v = lookup.get((section, metric), ("", ""))
            row[col] = v
            if col == "duration_us" and number(v) is not None:   # --print-units base: the duration is in ns
                row[col] = f"{number(v) / 1000.0:.2f}" if u.lower().startswith("n") else v
        table.append(row)
    with open(os.path.join(folder, "ncu_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    # the stall reasons and the pipe utilisations: top entries of the raw page, per kernel
    extra = []
    for kid in ids:
        r = raw.get(kid, {})
        stalls = sorted(((number(v[1]), re.sub(r"^smsp__average_warps_issue_stalled_|_per_issue_active\.ratio$", "", m)) for m, v in r.items()
                         if re.match(r"smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio$", m) and number(v[1]) is not None), reverse=True)[:5]
        pipes = sorted(((number(v[1]), re.sub(r"\.avg\.pct_of_peak_sustained_active$", "", m)) for m, v in r.items()
                        if re.match(r"sm__(inst_executed_)?pipe_.*\.avg\.pct_of_peak_sustained_active$", m) and number(v[1]) is not None), reverse=True)[:5]
        extra.append((labels[kid], stalls, pipes))
    lines = ["| id | kernel | " + " | ".join(c for c, _, _ in SELECTED[:12]) + " |", "|---|---|" + "---:|" * 12]
    for r in table:
        lines.append(f"| {r['id']} | {r['kernel']} | " + " | ".join(str(r[c]) for c, _, _ in SELECTED[:12]) + " |")
    lines += ["", "Top stall reasons (warp cycles stalled per issued instruction, raw page):", ""]
    for label, stalls, _ in extra:
        lines.append(f"- {label}: " + ", ".join(f"{n} {v:.2f}" for v, n in stalls))
    lines += ["", "Top pipe utilisations (% of peak sustained active, raw page):", ""]
    for label, _, pipes in extra:
        lines.append(f"- {label}: " + ", ".join(f"{n} {v:.1f}" for v, n in pipes))
    text = "\n".join(lines) + "\n"
    open(os.path.join(folder, "ncu_summary.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
