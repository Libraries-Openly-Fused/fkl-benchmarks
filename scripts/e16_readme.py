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
"""E16: write the README of a results folder (results/<gpu>/e16-attention) from the summaries of scripts/e16_summary.py and the fixed text of scripts/e16_readme_static.md.

  python scripts/e16_readme.py results/rtx4090/e16-attention "RTX 4090 (sm_89)"
"""
import csv
import os
import sys


def table(rows, columns, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if i > 1 else "---" for i in range(len(header))) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(f(r)) if callable(f) else str(r.get(f, "")) for f in columns) + " |")
    return "\n".join(lines)


def num(r, key, fmt="{:.1f}"):
    try:
        return fmt.format(float(r[key]))
    except (KeyError, ValueError):
        return "-"


def main():
    folder, gpu = sys.argv[1], sys.argv[2]
    here = os.path.dirname(os.path.abspath(__file__))
    core = list(csv.DictReader(open(os.path.join(folder, "e16_summary_core.csv"))))
    fusion = list(csv.DictReader(open(os.path.join(folder, "e16_summary_fusion.csv"))))
    compact = list(csv.DictReader(open(os.path.join(folder, "e16_q2_compact.csv")))) if os.path.exists(os.path.join(folder, "e16_q2_compact.csv")) else []
    text = open(os.path.join(folder, "e16_text_numbers.txt")).read()
    static = open(os.path.join(here, "e16_readme_static.md")).read()
    out = [f"# E16, forward attention with FKL: the attention core and the work around it ({gpu}, CUDA 13.4.3)", "",
           "Question Q1 (nothing fused around the pattern): how close is FKL's tensor-core attention DPP, as shipped, to the best attention kernels? Question Q2 (the work before and after the pattern): what does fusing it buy, on FKL's own kernel and against "
           "the same work as separate PyTorch kernels? Every number below is the median over three complete runs of the run median, with [smallest-largest] of the run medians; no result is tuned, and a ratio below 1 is reported as it is.", ""]
    out += ["## Q1: the attention core (TFLOP/s, GPU-bound)", "",
            table([r for r in core if r.get("fraction_median")], [lambda r: r["d"], lambda r: "causal" if r["causal"] == "1" else "non-causal", "seq", "bh", lambda r: num(r, "mma_raw_tflops"), lambda r: r.get("mma_raw_tile_BQxBKVxwarps", ""),
                                                                       "best_baseline", lambda r: f"{num(r, 'fraction_median', '{:.3f}')} [{num(r, 'fraction_min', '{:.3f}')}-{num(r, 'fraction_max', '{:.3f}')}]", lambda r: "DIFFERENT CLOCKS" if r.get("different_clocks") else ""],
                  ["D", "mask", "S", "BH", "FKL mma_raw TFLOP/s", "tile BQxBKVxwarps", "best baseline", "FKL / best baseline", "clock check"]), "",
            "All baselines per point (TFLOP/s, flash = FlashAttention-2 of PyTorch, cuDNN, efficient, flash-attn 2.8.3, Triton) are in `e16_summary_core.csv`; the time per call in the run CSVs.", ""]
    out += ["## Q2: the work around the pattern (fused against separate)", ""]
    if compact:
        out += [table(compact, ["work", "d", "seq", lambda r: "causal" if r["causal"] == "1" else "non-causal", lambda r: num(r, "fkl_fused_ms", "{:.3f}"), lambda r: num(r, "fkl_separate_ms", "{:.3f}"),
                                lambda r: f"{num(r, 'ratio_separate_over_fused', '{:.3f}')} [{num(r, 'ratio_min', '{:.3f}')}-{num(r, 'ratio_max', '{:.3f}')}]", "verdict", lambda r: f"{r['kernels_fused']} / {r['kernels_separate']}",
                                lambda r: f"{num(r, 'pytorch_composition_ms', '{:.3f}')} ({r['pytorch_composition_kernels']} kernels)" if r.get("pytorch_composition_ms") else "-", lambda r: f"{r['fused_regs']} / {r['fused_local_B']}", "fused_tile", "separate_tile"],
                      ["work", "D", "S", "mask", "fused ms", "separate ms", "separate / fused", "verdict", "kernels fused / separate", "PyTorch composition ms", "regs / local B (fused)", "tile fused", "tile separate"]), ""]
    out += ["All configurations are in `e16_summary_fusion.csv`. The numbers for the text:", "", "```", text.rstrip(), "```", ""]
    out += [static]
    sha = os.path.join(folder, "original-sha256.txt")
    if os.path.exists(sha):
        out += ["", "## Repaired files (the first version of the FKL program wrote a comma into a field)", "",
                "The `schedule` string of the quantised variants of the first version of `e16_attention.cu` was `cp.async quantised bytes, smem dequant`, with a comma, and the CSV writer does not quote: 40 rows of each `run*/e16_attention.csv` (the fused int8 and fp8 rows, 120 rows in all) had 27 fields instead of 26. "
                "No value was lost: the columns before `schedule` were intact and the shifted columns held the same text. The three files were repaired by one text replacement, the field `cp.async quantised bytes, smem dequant` wrapped in double quotes (the text is unchanged, a standard CSV reader now gives 26 fields in every row), "
                "and the `_clock.csv` files and the summaries were regenerated from the repaired files (the summary tables are identical cell by cell to the ones made from the originals with a tolerant reader). The source is fixed (the string has no comma any more); the binary that made this run is the one in `binary-sha256.txt`. "
                "The originals are kept in `original-csv/` (small); their sha256 (and those of the repaired files) are below, so the repair can be checked: `sed 's/cp.async quantised bytes, smem dequant/\"cp.async quantised bytes, smem dequant\"/'` applied to an original must give the repaired file byte for byte.", "", "```", open(sha).read().rstrip(), "```", ""]
    open(os.path.join(folder, "README.md"), "w").write("\n".join(out))
    print(os.path.join(folder, "README.md"), len("\n".join(out).splitlines()), "lines")


if __name__ == "__main__":
    main()
