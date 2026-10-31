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
"""Summary of the Nsight Compute runs of the E2 kernels (one launch each, caches flushed before it, clocks not locked): one row per file ncu_<impl>_k<k>.csv of a directory.

  python scripts/e2_ncu_summary.py results/rtx4090/e2-chain/ncu --out results/rtx4090/e2-chain/ncu/e2_ncu_summary.csv

Columns: duration, DRAM throughput as a percentage of the peak, DRAM bytes read and written during the launch, the share of the 1,008 GB/s that the bytes moved over the duration make up,
grid and block size, registers per thread, theoretical and achieved occupancy, instructions executed, global load/store requests and sectors at the L1, sectors at the L2 and the L2 hit rate.
"""
import argparse
import csv
import glob
import io
import os
import re

PEAK_GBS = 1008.0
COLS = [("gpu__time_duration.sum", "duration_us", 1e-3), ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram_throughput_pct", 1.0),
        ("dram__bytes_read.sum", "dram_read_MB", 1e-6), ("dram__bytes_write.sum", "dram_write_MB", 1e-6),
        ("launch__grid_size", "grid", 1.0), ("launch__block_size", "block", 1.0), ("launch__registers_per_thread", "regs_allocated", 1.0),
        ("sm__maximum_warps_per_active_cycle_pct", "theoretical_occupancy_pct", 1.0), ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved_occupancy_pct", 1.0),
        ("smsp__inst_executed.sum", "inst_executed", 1.0),
        ("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "l1_ld_requests", 1.0), ("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "l1_st_requests", 1.0),
        ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "l1_ld_sectors", 1.0), ("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "l1_st_sectors", 1.0),
        ("lts__t_sectors_op_read.sum", "l2_read_sectors", 1.0), ("lts__t_sectors_op_write.sum", "l2_write_sectors", 1.0), ("lts__t_sector_hit_rate.pct", "l2_hit_rate_pct", 1.0)]


def num(s):
    return float(s.replace(",", ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, "ncu_*_k*.csv"))):
        m = re.match(r"ncu_(.+)_k(\d+)\.csv", os.path.basename(path))
        text = open(path).read()
        i = text.find('"ID"')
        if i < 0:
            print(f"{path}: no profiled kernel")
            continue
        table = list(csv.reader(io.StringIO(text[i:])))
        head = table[0]
        for data in table[2:3]:
            d = dict(zip(head, data))
            row = {"impl": m.group(1), "k": int(m.group(2)), "kernel": d["Kernel Name"][:48]}
            for metric, name, scale in COLS:
                row[name] = round(num(d[metric]) * scale, 3) if d.get(metric) not in (None, "") else ""
            row["bytes_over_duration_GBs"] = round((row["dram_read_MB"] + row["dram_write_MB"]) * 1e6 / (row["duration_us"] * 1e-6) / 1e9, 1)
            row["bytes_over_duration_pct_of_peak"] = round(100 * row["bytes_over_duration_GBs"] / PEAK_GBS, 1)
            rows.append(row)
    for r in rows:
        print(f"{r['impl']:17s} k={r['k']:<4d} {r['duration_us']:7.2f} us  DRAM {r['dram_throughput_pct']:5.1f} % of peak  read {r['dram_read_MB']:6.2f} MB write {r['dram_write_MB']:6.2f} MB  grid {int(r['grid']):6d} x {int(r['block']):3d}  "
              f"regs {int(r['regs_allocated']):2d}  occupancy {r['achieved_occupancy_pct']:5.1f} % of {r['theoretical_occupancy_pct']:.0f} %  inst {int(r['inst_executed'])}  L1 ld req {int(r['l1_ld_requests'])} st req {int(r['l1_st_requests'])}")
    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
