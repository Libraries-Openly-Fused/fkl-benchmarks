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
"""E3, the StaticLoop axis: compile time of the vertical-fusion translation units (cvGPUSpeedup's mul_add1 chain, 2 to 19,902 operations) with the nvcc phase split.

  python scripts/e3_staticloop.py --build build/local-rtx4090 --instance mul_add1 --core 10 --out results/rtx4090/e3-compile

Compiles generated/<instance>/200_experiments/kernel<k>.cu (k = 1 .. 200, operations = 100 * (k - 1) + 2) with the compile line CMake used for the benchmark
(flags.make of the target, --threads 1 instead of --threads 0), pinned to ONE core, repeated three times when the first compile takes under 30 s, and records
what scripts/e3_sweep.py records. Compile only: nothing is executed (the GPU is not touched).
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import e3_sweep as e3  # noqa: E402
import e1_table  # noqa: E402
import sass_tools as st  # noqa: E402
from fklb_csv import CsvWriter  # noqa: E402

TARGETS = {"mul_add1": "benchmark_vertical_fusionCompareMulAdd1_cu", "mul1": "benchmark_vertical_fusionCompareMul1_cu", "mul_add": "benchmark_vertical_fusionCompareMulAdd_cu"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--instance", default="mul_add1", choices=list(TARGETS))
    ap.add_argument("--ks", default="1,2,3,6,11,21,41,61,81,101,126,151,176,200")
    ap.add_argument("--core", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="sm_89")
    args = ap.parse_args()
    vf = os.path.join(os.path.abspath(args.build), "opencv-cuda", "verticalfusion")
    flags = open(os.path.join(vf, "CMakeFiles", TARGETS[args.instance] + ".dir", "flags.make")).read()
    get = lambda key: re.search(rf"^{key} = (.*)$", flags, re.M).group(1)
    nvcc = os.path.join(os.environ["FKLB_CUDA_HOME"], "bin", "nvcc")
    defines = get("CUDA_DEFINES").split()
    rsp = get("CUDA_INCLUDES").replace("--options-file ", "").strip()
    rsp = rsp if os.path.isabs(rsp) else os.path.join(vf, rsp)
    cflags = [f.strip('"') for f in re.findall(r'"[^"]*"|\S+', get("CUDA_FLAGS"))]
    cflags = [("1" if cflags[i - 1] == "--threads" else f) for i, f in enumerate(cflags)]     # one thread: the compile is pinned to one core
    gen = os.path.join(vf, "generated", args.instance, "200_experiments")
    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(ROOT, "build", "e3-staticloop")
    os.makedirs(work, exist_ok=True)
    cols = ["instance", "k", "ops", "rep", "status", "why", "cuda_wall_s", "cuda_user_s", "cuda_sys_s", "cuda_peak_rss_mb", "phase_cudafe_s", "phase_cicc_s", "phase_ptxas_s",
            "phase_gcc_s", "phase_other_s", "registers", "spill_stores", "spill_loads", "stack_bytes", "cmem0_bytes", "sass_instructions", "text_bytes", "cubin_bytes"]
    notes = f"one core ({args.core}), the compile line of {TARGETS[args.instance]} with --threads 1 and --resource-usage; the programs are not executed; translation units from {gen}"
    with CsvWriter(os.path.join(args.out, f"e3_staticloop_{args.instance}.csv"), "E3 StaticLoop axis: compile time of the vertical-fusion TUs", cols, notes=notes) as csv:
        for k in (int(x) for x in args.ks.split(",")):
            ops = 100 * (k - 1) + 2
            src = os.path.join(gen, f"kernel{k}.cu")
            obj = os.path.join(work, f"{args.instance}_kernel{k}.o")
            for rep in range(3):
                timefile = os.path.join(work, f"{args.instance}_kernel{k}.time.csv")
                cmd = [nvcc] + defines + ["--options-file", rsp] + cflags + ["--resource-usage", "--time", timefile, "-c", src, "-o", obj]
                if os.path.exists(timefile):
                    os.remove(timefile)      # nvcc --time appends to an existing file
                status, wall, user, sys_, rss, err = e3.run_measured(cmd, args.core, args.timeout, cwd=vf)
                row = dict.fromkeys(cols, "")
                row.update(instance=args.instance, k=k, ops=ops, rep=rep, status=status, cuda_wall_s=f"{wall:.3f}", cuda_user_s=f"{user:.3f}", cuda_sys_s=f"{sys_:.3f}", cuda_peak_rss_mb=f"{rss:.0f}")
                if status == "ok":
                    ph = e3.phase_times(timefile)
                    row.update(phase_cudafe_s=f"{ph.get('cudafe++', 0):.3f}", phase_cicc_s=f"{ph.get('cicc', 0):.3f}", phase_ptxas_s=f"{ph.get('ptxas', 0):.3f}",
                               phase_gcc_s=f"{ph.get('gcc', 0):.3f}", phase_other_s=f"{sum(v for n, v in ph.items() if n not in ('cudafe++', 'cicc', 'ptxas', 'gcc')):.3f}")
                    row.update(e3.ptxas_resources(err))
                    funcs = st.split_functions(subprocess.run(["cuobjdump", "-arch", args.arch, "-sass", obj], capture_output=True, text=True).stdout)
                    if funcs:
                        name, body = max(funcs.items(), key=lambda kv: len(kv[1]))
                        row["sass_instructions"] = st.classify(st.normalise_sass(body, rename=False))[0]
                    import shutil, tempfile
                    tmp = tempfile.mkdtemp(prefix="e3cubin")
                    subprocess.run(["cuobjdump", "-arch", args.arch, "-xelf", "all", obj], cwd=tmp, capture_output=True)
                    cubins = [f for f in os.listdir(tmp) if f.endswith(".cubin")]
                    if cubins:
                        path = os.path.join(tmp, cubins[0])
                        row["cubin_bytes"] = os.path.getsize(path)
                        sizes = e1_table.text_section_sizes(path)
                        if sizes:
                            row["text_bytes"] = max(sizes.values())
                    shutil.rmtree(tmp, ignore_errors=True)
                else:
                    row["why"] = e3.first_error(err)
                csv.row(*[row[c] for c in cols])
                print(f"{args.instance} k={k:3d} ops={ops:6d} rep {rep}: {status:20s} {wall:8.2f} s {rss:7.0f} MB  cicc {row['phase_cicc_s']} ptxas {row['phase_ptxas_s']} cudafe {row['phase_cudafe_s']} regs {row.get('registers')} SASS {row.get('sass_instructions')}", flush=True)
                if status != "ok" or wall >= 30:
                    break


if __name__ == "__main__":
    main()
