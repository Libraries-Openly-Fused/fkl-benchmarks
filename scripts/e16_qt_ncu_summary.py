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
"""E16 query-tile experiment, the Nsight Compute pass: the selected metrics of the four attention kernels from the raw page.

  python scripts/e16_qt_ncu_summary.py <report.ncu-rep> results/rtx4090/e16-query-tile/ncu

Writes in the folder: ncu_raw.csv (`ncu --import <report> --page raw --csv --print-units base`), ncu_selected.csv/.md (one row per kernel, the metrics of SELECTED) and, through
scripts/e16_ncu_sass.py, ncu_opcodes.csv/_full.csv/.md (executed warp instructions by SASS opcode). The kernels are in the launch order of the pass (E16_ROTATE = 0).
No metric is interpreted here.
"""
import csv
import os
import subprocess
import sys

LABELS = ["raw_bq64_kv32_w4", "int8_bq64_w4", "int8_bq128_w4", "int8nodq_bq64_w4 (TIMING ONLY)"]
SELECTED = ["gpu__time_duration.sum", "smsp__inst_executed.sum", "smsp__thread_inst_executed_per_inst_executed.ratio", "smsp__inst_issued.sum", "smsp__issue_active.avg.pct_of_peak_sustained_active",
            "sm__warps_active.avg.pct_of_peak_sustained_active", "launch__registers_per_thread", "launch__shared_mem_per_block_dynamic", "launch__occupancy_limit_registers",
            "launch__occupancy_limit_shared_mem", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "dram__bytes_read.sum", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
            "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
            "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active", "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active"]


def main():
    report, folder = sys.argv[1], sys.argv[2]
    ncu = os.path.join(os.environ.get("FKLB_CUDA_HOME", "/home/oscar/tfkl-bench/toolchains/cuda-13.4.3"), "bin", "ncu")
    raw = subprocess.run(["taskset", "-c", "8-15,24-31", ncu, "--import", report, "--page", "raw", "--csv", "--print-units", "base"], capture_output=True, text=True, check=True).stdout
    open(os.path.join(folder, "ncu_raw.csv"), "w").write(raw)
    rows = list(csv.reader([l for l in raw.splitlines() if not l.startswith("==")]))
    head, units, data = rows[0], rows[1], rows[2:]
    kn = head.index("Kernel Name")
    if len(data) != len(LABELS):
        print(f"WARNING: {len(data)} kernels in the report, {len(LABELS)} expected: {[d[kn][:50] for d in data]}")
    with open(os.path.join(folder, "ncu_selected.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel"] + [f"{m} [{units[head.index(m)] if m in head else ''}]" for m in SELECTED])
        for lab, d in zip(LABELS, data):
            w.writerow([lab] + [d[head.index(m)] if m in head else "not collected" for m in SELECTED])
    md = ["| metric | " + " | ".join(LABELS[:len(data)]) + " |", "|---|" + "---:|" * len(data)]
    for m in SELECTED:
        md.append(f"| {m} [{units[head.index(m)] if m in head else ''}] | " + " | ".join(d[head.index(m)] if m in head else "n/a" for d in data) + " |")
    open(os.path.join(folder, "ncu_selected.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e16_ncu_sass.py")
    kernels = ";".join(f"{lab}|FlashAttentionMma|{i + 1}" for i, lab in enumerate(LABELS[:len(data)]))
    subprocess.run([sys.executable, script, report, folder, "--kernels", kernels], check=True)


if __name__ == "__main__":
    main()
