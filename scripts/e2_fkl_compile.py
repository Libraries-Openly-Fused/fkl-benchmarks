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
"""Cold compile time of the FKL chain of E2 (handwritten-cuda/e2_fkl_chain.cu), one k per compile: the column 'compile time' of the FKL row of the E2 table.

  python scripts/e2_fkl_compile.py --ks 2,10,50,100,500 [--core 14] [--reps 3] [--out e2_fkl_compile.csv]

Measured as the E3 sweep measures a translation unit (scripts/e3_sweep.py): nvcc 13.4.92 on ONE core, --threads 1, -O3 -DNDEBUG -std=c++17, sm_89 + PTX,
-ftemplate-depth=1000, -lineinfo, the time of the nvcc phases from nvcc --time, GNU time for the peak memory; the program is not executed (-c: an object file, no link).
The translation unit is the program itself with E2_FKL_K_ONE = k, so it includes the (small) CSV and main headers; the chain is the only thing that changes with k.
One row per k: the median over --reps compiles of nvcc_s and of each phase (the E2 driver reads ops and nvcc_s). Compile only: nothing runs on the GPU.
"""
import argparse
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import e3_sweep as sw  # noqa: E402
from fklb_csv import CsvWriter  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", default="2,10,50,100,500")
    ap.add_argument("--core", type=int, default=14)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--out", default="e2_fkl_compile.csv")
    ap.add_argument("--include", default=os.path.join(ROOT, "external", "fkl", "include"))
    ap.add_argument("--build-info-dir", default=os.path.join(ROOT, "build", "e4", "generated"), help="directory of the CMake-generated fklb_build_info.h (it does not change the generated code)")
    args = ap.parse_args()
    cuda_home = os.environ.get("FKLB_CUDA_HOME")
    if not cuda_home:
        sys.exit("source toolchains/env-cuda13.4.sh first (FKLB_CUDA_HOME)")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    src = os.path.join(ROOT, "handwritten-cuda", "e2_fkl_chain.cu")
    cols = ["ops", "reps", "status", "nvcc_s", "cudafe_s", "cicc_s", "ptxas_s", "gcc_s", "peak_rss_mb", "all_nvcc_s"]
    notes = (f"one core ({args.core}), nvcc --threads 1, -O3 -DNDEBUG -std=c++17 {args.arch} +PTX -ftemplate-depth=1000 -lineinfo, object file (-c), the program is not executed; "
             f"median of {args.reps} compiles; fkl include {args.include}")
    with CsvWriter(args.out, "E2 FKL chain: cold compile time", cols, notes=notes) as csv, tempfile.TemporaryDirectory(prefix="e2fklcompile") as tmp:
        for k in (int(x) for x in args.ks.split(",")):
            walls, ph, rss, status = [], {"cudafe++": [], "cicc": [], "ptxas": [], "gcc": []}, [], "ok"
            for rep in range(args.reps):
                timefile = os.path.join(tmp, f"k{k}.time.csv")
                if os.path.exists(timefile):
                    os.remove(timefile)
                obj = os.path.join(tmp, f"k{k}.o")
                cmd = [nvcc, "-std=c++17", "-O3", "-DNDEBUG", f"--generate-code=arch=compute_{args.arch[3:]},code=[compute_{args.arch[3:]},{args.arch}]", "--threads", "1",
                       "-ftemplate-depth=1000", "-lineinfo", "--time", timefile, f"-DE2_FKL_K_ONE={k}", "-I", args.include, "-I", ROOT, "-I", os.path.join(ROOT, "handwritten-cuda"), "-I", args.build_info_dir,
                       "-c", src, "-o", obj]
                st, wall, _, _, peak, err = sw.run_measured(cmd, args.core, args.timeout)
                if st != "ok":
                    status = st + ": " + sw.first_error(err)
                    break
                walls.append(wall)
                rss.append(peak)
                t = sw.phase_times(timefile)
                for name in ph:
                    ph[name].append(t.get(name, 0.0))
            if status == "ok":
                csv.row(k, len(walls), status, f"{statistics.median(walls):.3f}", *[f"{statistics.median(ph[n]):.3f}" for n in ("cudafe++", "cicc", "ptxas", "gcc")],
                        f"{max(rss):.0f}", ";".join(f"{w:.2f}" for w in walls))
            else:
                csv.row(k, len(walls), status, "", "", "", "", "", "", "")
            print(f"k={k}: {status}" + (f" nvcc {statistics.median(walls):.2f} s (median of {len(walls)})" if status == "ok" else ""), flush=True)


if __name__ == "__main__":
    main()
