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
"""E16: how the extra time of the fused quantised-KV attention kernel over mma_raw grows with S, at constant tokens, next to the cost of the separate dequantisation of the unfused pair.

  python scripts/e16_prologue_scaling.py results/rtx4090/e16-attention

Per work (int8, fp8), head size D, mask, S (BH = tokens / S, so BH x S is the same for every S) and run:
  extra_fused  = time(mma_<work>kv_fused)   - time(mma_raw)    what the fused kernel costs over the bf16 kernel
  extra_unfused = time(mma_<work>kv_unfused) - time(mma_raw)   what the unfused pair costs over the bf16 kernel (the dequantisation kernel and the effect of its bf16 output on the attention kernel)
  per_S = extra_fused / S
The table gives the median over the runs of each value with [smallest-largest of the runs]. Writes e16_prologue_scaling.csv and e16_prologue_scaling.md in the folder.
COMPUTED FROM THE BENCHMARK RUNS (GPU time per call, boost clock, back-to-back calls), NOT from Nsight Compute; no cause is stated.
"""
import csv
import os
import statistics as stats
import sys

import e16_summary as es

WORKS = [("int8 K,V", "mma_int8kv_fused", "mma_int8kv_unfused"), ("fp8 e4m3 K,V", "mma_fp8kv_fused", "mma_fp8kv_unfused")]


def span(values):
    return f"{stats.median(values):.3f} [{min(values):.3f}-{max(values):.3f}]"


def main():
    folder = sys.argv[1]
    runs = es.load(folder)
    keys = sorted({k for rows in runs.values() for k in rows if k[0] == "mma_raw"}, key=lambda k: (k[3], k[1], k[2]))
    gpu = next((row["gpu"] for rows in runs.values() for row in rows.values() if row.get("gpu")), "?")
    out, lines = [], []
    for work, fused, unfused in WORKS:
        for causal in (0, 1):
            lines += [f"### {work}, {'causal' if causal else 'non-causal'}: the time of the fused kernel and of the unfused pair over the bf16 kernel (ms)", "",
                      "| D | S | BH | BLOCK_Q of the fused kernel | query tiles per head | mma_raw ms | fused - raw ms | (fused - raw) / S us | unfused - raw ms | (fused - raw) / (unfused - raw) |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for (_, d, s, c) in keys:
                if c != causal:
                    continue
                ex_f, ex_u, raw_ms, bq, bh = [], [], [], None, None
                for rows in runs.values():
                    raw, fu, un = rows.get(("mma_raw", d, s, c)), rows.get((fused, d, s, c)), rows.get((unfused, d, s, c))
                    if raw and fu and un and es.t(raw) and es.t(fu) and es.t(un):
                        ex_f.append(es.t(fu) - es.t(raw))
                        ex_u.append(es.t(un) - es.t(raw))
                        raw_ms.append(es.t(raw))
                        bq, bh = int(fu["block_q"]), int(raw["bh"])
                if not ex_f:
                    continue
                per_s = [1000.0 * x / s for x in ex_f]
                ratio = [a / b if b > 0 else float("nan") for a, b in zip(ex_f, ex_u)]
                lines.append(f"| {d} | {s} | {bh} | {bq} | {s // bq} | {stats.median(raw_ms):.3f} | {span(ex_f)} | {span(per_s)} | {span(ex_u)} | {span(ratio)} |")
                out.append({"gpu": gpu, "work": work, "d": d, "causal": c, "seq": s, "bh": bh, "block_q_fused": bq, "query_tiles_per_head": s // bq, "raw_ms_median": f"{stats.median(raw_ms):.4f}",
                            "extra_fused_ms_median": f"{stats.median(ex_f):.4f}", "extra_fused_ms_min": f"{min(ex_f):.4f}", "extra_fused_ms_max": f"{max(ex_f):.4f}",
                            "extra_fused_per_S_us_median": f"{stats.median(per_s):.4f}", "extra_fused_per_S_us_min": f"{min(per_s):.4f}", "extra_fused_per_S_us_max": f"{max(per_s):.4f}",
                            "extra_unfused_ms_median": f"{stats.median(ex_u):.4f}", "extra_unfused_ms_min": f"{min(ex_u):.4f}", "extra_unfused_ms_max": f"{max(ex_u):.4f}",
                            "ratio_fused_over_unfused_extra_median": f"{stats.median(ratio):.4f}", "runs": len(ex_f)})
            lines.append("")
    with open(os.path.join(folder, "e16_prologue_scaling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    text = "\n".join(lines)
    open(os.path.join(folder, "e16_prologue_scaling.md"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
