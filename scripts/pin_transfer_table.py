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
"""The table of results/rtx4090/e0-pin-transfer: every binary that was compared, the results it produced and the kernel counts, from the sass_<group>[_demangled].txt files that
scripts/pin_transfer.sh wrote.

  python scripts/pin_transfer_table.py results/rtx4090/e0-pin-transfer

Writes transfer_table.md and transfer_table.csv. A group is read from its _demangled file when the mangled-name comparison left kernels on one side only (the names then differ for a reason that is
not the code: the defaulted ResizeSampling template argument of resize kernels, which the library gained between fb9ac3a and 38c4f32, and the per-compile hash of kernels in an anonymous namespace).
"""
import csv
import os
import re
import sys

# group -> (benchmarks commit, library of the compared baseline, what the baseline is, [(binary, results it produced)])
GROUPS = [
    ("a334424", "a334424", "fb9ac3a", "frozen binary of the measurement", [("benchmark_latency_hiding_cu", "e0-latency-hiding"), ("benchmark_numinstructions_x_operation_cu", "e0-numinstructions")]),
    ("269d229", "269d229", "fb9ac3a", "frozen binary of the measurement", [("benchmark_CPUandGPU_cvGS_vs_fk_cu", "e0-wrapper-overhead"), ("benchmark_CPU_OpenCV_vs_cvGS_cu", "e0-cpu-time-opencv"), ("benchmark_data_type_cu", "e0-data-type")]),
    ("5aac244", "5aac244", "fb9ac3a", "frozen binary of the measurement", [("benchmark_CPU_NPP_vs_FastNPP_cu", "e0-cpu-time-npp"), ("benchmark_npp_batchresize_x_split3D_cu", "e0-npp")]),
    ("1a70cab", "1a70cab", "fb9ac3a", "frozen binary of the measurement", [("benchmark_image_resolution_MAD_loop_cu", "e0-data-size"), ("benchmark_horizontal_fusion_cu", "e0-hf"),
                                                                              ("benchmark_CPUandGPU_cvGS_vs_fk_cu", "e0-wrapper-overhead-order"), ("resize_convention_cu", "e6-resize-convention")]),
    ("a266bc9", "a266bc9", "fb9ac3a", "frozen binary of the measurement", [("graph_instantiate_time_cu", "e0-graph-instantiate"), ("benchmark_vertical_fusionCompareMulAdd1_cu", "e0-vf-mul-add1, e0-vf-mul-add1-run1"),
                                                                              ("benchmark_vertical_fusionCompareMul1_cu", "e0-vf-mul1"), ("benchmark_vertical_fusionCompareMulAdd_cu", "e0-vf-mul-add")]),
    ("e1", "74f47c6", "38c4f32", "frozen binary of the measurement", [("e1_vf_chain_cu", "e1-vf-chain-v2"), ("e1_hf_batch_cu", "e1-hf-batch-v2"), ("e1_preproc_cu", "e1-preproc-v3")]),
    ("e2-old", "e9e599d", "38c4f32", "rebuild at 38c4f32 against the frozen binary (source check)", [("e2_fkl_chain_cu", "e2-chain (same binary rebuilt at the old library)")]),
    ("e2-new", "e9e599d", "38c4f32", "frozen binary of the measurement", [("e2_fkl_chain_cu", "e2-chain")]),
    ("mul4-old", "e9e599d", "38c4f32", "rebuild at 38c4f32 against the frozen binary (source check)", [("benchmark_image_resolution_MAD_loop_mul4_cu", "e0-data-size-mul4 (same binary rebuilt at the old library)")]),
    ("mul4-new", "e9e599d", "38c4f32", "frozen binary of the measurement", [("benchmark_image_resolution_MAD_loop_mul4_cu", "e0-data-size-mul4, e0-data-size-parity")]),
    ("nan_saturate", "f3afadc", "fb9ac3a", "frozen binary of the measurement", [("nan_saturate_cu", "e6-nan-saturate")]),
    ("nv12", "f81306d", "fb9ac3a", "rebuild at fb9ac3a (no frozen binary exists)", [("benchmark_nv12_pipeline_cu", "e4b-nv12")]),
]
LINE = re.compile(r"^\s+binary (\S+): kernels in A (\d+), in B (\d+), identical (\d+), different (\d+), only in A (\d+), only in B (\d+)")


def read(path):
    out = {}
    for line in open(path):
        m = LINE.match(line)
        if m:
            out[m.group(1)] = tuple(int(x) for x in m.groups()[1:])
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/rtx4090/e0-pin-transfer"
    rows = []
    for tag, sha, lib, what, bins in GROUPS:
        dem = os.path.join(d, f"sass_{tag}_demangled.txt")
        plain = os.path.join(d, f"sass_{tag}.txt")
        if not os.path.exists(plain):
            print(f"{tag}: no result file")
            continue
        used, how = (dem, "demangled name, ResizeSampling argument and anonymous-namespace hash ignored") if os.path.exists(dem) else (plain, "mangled name")
        res = read(used)
        for exe, results in bins:
            if exe not in res:
                print(f"{tag}: {exe} not in {used}")
                continue
            a, b, same, diff, oa, ob = res[exe]
            rows.append(dict(group=tag, benchmarks_commit=sha, baseline_library=lib, baseline=what, binary=exe, results=results, kernels_baseline=a, kernels_new=b, identical=same,
                             different=diff, only_baseline=oa, only_new=ob, matched_by=how))
    with open(os.path.join(d, "transfer_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| group | benchmarks commit | baseline (library) | binary | results it produced | kernels in baseline | in c8fcbe8 build | identical | different | only on one side | matched by |", "|---|---|---|---|---|--:|--:|--:|--:|--:|---|"]
    for r in rows:
        lines.append(f"| {r['group']} | {r['benchmarks_commit']} | {r['baseline']} ({r['baseline_library']}) | `{r['binary']}` | {r['results']} | {r['kernels_baseline']} | {r['kernels_new']} | {r['identical']} | {r['different']} | {r['only_baseline'] + r['only_new']} | {r['matched_by']} |")
    tot = lambda key: sum(r[key] for r in rows)
    new_rows = [r for r in rows if not r["group"].endswith("-old")]
    lines.append(f"| **total (new tree against the baseline)** | | | {len(new_rows)} binaries | | **{sum(r['kernels_baseline'] for r in new_rows)}** | **{sum(r['kernels_new'] for r in new_rows)}** | **{sum(r['identical'] for r in new_rows)}** | **{sum(r['different'] for r in new_rows)}** | **{sum(r['only_baseline'] + r['only_new'] for r in new_rows)}** | |")
    open(os.path.join(d, "transfer_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nbinaries {len(rows)}, kernels identical {tot('identical')}, different {tot('different')}")


if __name__ == "__main__":
    main()
