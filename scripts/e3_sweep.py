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
"""E3 compile-time sweep: how long and how much memory it takes to compile a fused kernel of N distinct parameterised operations, and where it stops.

  python scripts/e3_sweep.py --variants A,B --ns 4,14,54,100,200,300,500,750,1000 --core 8 --out results/rtx4090/e3-compile

For each variant (fkl-self/compiletime/gen_chain.py) and N one translation unit is generated and compiled with nvcc 13.4 (the library's flags: -O3 -DNDEBUG
-std=c++17 -arch sm_89 with PTX, -ftemplate-depth=1000, -lineinfo, --threads 1), pinned to ONE core, and, with --host, the same source as plain C++ with g++ (the
CPU back-end, no device code). Per compile: wall, user and system time and the peak resident memory (GNU time), the nvcc phase times (--time), the ptxas resource
usage of the fused kernel (registers, spills, stack, parameter block), the SASS instructions and the .text bytes of the kernel and the size of the cubin. The sweep
stops a variant at the first failure and records why (the first error line, or timeout, or the process being killed). The programs are NEVER executed here: this
script does not touch the GPU, it can run while the experiments of E0 are running (the core it uses must not be one of E0's).
Rows: scripts/e3_sweep.py writes e3_compile.csv with one row per compile (repetitions: 3 when the first compile takes under 30 s, else 1).
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import e1_table  # noqa: E402  (text_section_sizes)
import sass_tools as st  # noqa: E402
from fklb_csv import CsvWriter  # noqa: E402


def run_measured(cmd, core, timeout, cwd=None):
    """Run cmd pinned to a core under GNU time. Returns (status, wall, user, sys, peak_rss_mb, stderr_text)."""
    with tempfile.NamedTemporaryFile("r", suffix=".time") as tf:
        full = ["taskset", "-c", str(core), "/usr/bin/time", "-f", "%e %U %S %M", "-o", tf.name] + cmd
        t0 = time.perf_counter()
        p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", start_new_session=True, cwd=cwd)
        try:
            out, err = p.communicate(timeout=timeout)
            status = "ok" if p.returncode == 0 else f"failed (exit {p.returncode})"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, err = p.communicate()
            status = f"timeout ({timeout} s)"
        wall = time.perf_counter() - t0
        fields = (tf.read().strip().splitlines() or [""])[-1].split()
        try:
            _, user, sys_, rss = float(fields[0]), float(fields[1]), float(fields[2]), int(fields[3])
        except (ValueError, IndexError):
            user = sys_ = float("nan"); rss = 0
        if "Command terminated by signal 9" in err or "Killed" in err:
            status = "killed (out of memory?)" if status.startswith("failed") else status
        return status, wall, user, sys_, rss / 1024.0, err


def first_error(err):
    for line in err.splitlines():
        if re.search(r"error|fatal|Killed|bad_alloc|out of memory|too much parameter|exceeds", line, re.I) and "deprecated" not in line:
            return line.strip()[:300]
    return err.strip().splitlines()[-1][:300] if err.strip() else ""


def phase_times(path):
    import csv
    out = {}
    try:
        for r in csv.reader(open(path), skipinitialspace=True):
            if len(r) == 8 and r[6].replace(".", "", 1).isdigit():
                name = r[1].strip().split(" ")[0]
                out[name] = out.get(name, 0.0) + float(r[6]) / 1000.0     # ms to s
    except OSError:
        pass
    return out


def ptxas_resources(err):
    m = re.search(r"Function properties for \S+\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\s+ptxas info\s+: Used (\d+) registers.*?, (\d+) bytes cmem\[0\]", err, re.S)
    if not m:
        return {}
    stack, ss, sl, regs, cmem = (int(x) for x in m.groups())
    return {"registers": regs, "spill_stores": ss, "spill_loads": sl, "stack_bytes": stack, "cmem0_bytes": cmem}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="A,B")
    ap.add_argument("--ns", default="4,14,54,100,200,300,500,750,1000")
    ap.add_argument("--core", type=int, default=8, help="the one core every compile is pinned to (its SMT sibling should be idle)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per compile")
    ap.add_argument("--host", action="store_true", help="also build the same TU as plain C++ with g++ (CPU back-end)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=os.path.join(ROOT, "build", "e3"))
    ap.add_argument("--arch", default="sm_89")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    cuda_home = os.environ.get("FKLB_CUDA_HOME")
    if not cuda_home:
        sys.exit("source scripts/env-cuda13.4.sh first (FKLB_CUDA_HOME)")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    include = os.path.join(ROOT, "external", "fkl", "include")
    gen = os.path.join(ROOT, "fkl-self", "compiletime", "gen_chain.py")
    notes = (f"one core ({args.core}), nvcc --threads 1, -O3 -DNDEBUG -std=c++17 {args.arch} +PTX -ftemplate-depth=1000 -lineinfo; GNU time for the peak RSS; "
             f"timeout {args.timeout} s per compile; the programs are not executed; fkl include {include}")
    cols = ["variant", "n", "rep", "status", "why", "cuda_wall_s", "cuda_user_s", "cuda_sys_s", "cuda_peak_rss_mb", "phase_cudafe_s", "phase_cicc_s", "phase_ptxas_s",
            "phase_gcc_s", "phase_other_s", "registers", "spill_stores", "spill_loads", "stack_bytes", "cmem0_bytes", "sass_instructions", "text_bytes", "cubin_bytes",
            "host_status", "host_wall_s", "host_peak_rss_mb"]
    with CsvWriter(os.path.join(args.out, "e3_compile.csv"), "E3 compile-time sweep", cols, notes=notes) as csv:
        for variant in args.variants.split(","):
            stop = False
            for n in (int(x) for x in args.ns.split(",")):
                if stop:
                    break
                src = os.path.join(args.work, f"{variant}_{n}.cu")
                binary = os.path.join(args.work, f"{variant}_{n}")
                subprocess.run([sys.executable, gen, variant, str(n), src], check=True)
                rep = 0
                while rep < 3:
                    timefile = os.path.join(args.work, f"{variant}_{n}.time.csv")
                    cmd = [nvcc, "-std=c++17", "-O3", "-DNDEBUG", f"--generate-code=arch=compute_{args.arch[3:]},code=[compute_{args.arch[3:]},{args.arch}]", "--threads", "1",
                           "-ftemplate-depth=1000", "-lineinfo", "--resource-usage", "--time", timefile, "-I", include, src, "-o", binary]
                    status, wall, user, sys_, rss, err = run_measured(cmd, args.core, args.timeout)
                    row = dict.fromkeys(cols, "")
                    row.update(variant=variant, n=n, rep=rep, status=status, cuda_wall_s=f"{wall:.3f}", cuda_user_s=f"{user:.3f}", cuda_sys_s=f"{sys_:.3f}",
                               cuda_peak_rss_mb=f"{rss:.0f}")
                    if status == "ok":
                        ph = phase_times(timefile)
                        row.update(phase_cudafe_s=f"{ph.get('cudafe++', 0):.3f}", phase_cicc_s=f"{ph.get('cicc', 0):.3f}", phase_ptxas_s=f"{ph.get('ptxas', 0):.3f}",
                                   phase_gcc_s=f"{sum(v for k, v in ph.items() if k == 'gcc'):.3f}",
                                   phase_other_s=f"{sum(v for k, v in ph.items() if k not in ('cudafe++', 'cicc', 'ptxas', 'gcc')):.3f}")
                        row.update(ptxas_resources(err))
                        sass = subprocess.run(["cuobjdump", "-arch", args.arch, "-sass", binary], capture_output=True, text=True).stdout
                        funcs = st.split_functions(sass)
                        if funcs:
                            name, body = max(funcs.items(), key=lambda kv: len(kv[1]))
                            row["sass_instructions"] = st.classify(st.normalise_sass(body, rename=False))[0]
                        tmp = tempfile.mkdtemp(prefix="e3cubin")
                        subprocess.run(["cuobjdump", "-arch", args.arch, "-xelf", "all", binary], cwd=tmp, capture_output=True)
                        cubins = [f for f in os.listdir(tmp) if f.endswith(".cubin")]
                        if cubins:
                            path = os.path.join(tmp, cubins[0])
                            row["cubin_bytes"] = os.path.getsize(path)
                            sizes = e1_table.text_section_sizes(path)
                            if sizes:
                                row["text_bytes"] = max(sizes.values())
                        subprocess.run(["rm", "-rf", tmp])
                    else:
                        row["why"] = first_error(err)
                    if args.host and rep == 0:
                        hcmd = ["g++", "-std=c++17", "-O3", "-DNDEBUG", "-DE3_CPU", "-march=native", "-mtune=native", "-ftemplate-depth=1000", "-x", "c++", "-I", include, src,
                                "-o", binary + "_cpu"]
                        hstatus, hwall, _, _, hrss, herr = run_measured(hcmd, args.core, args.timeout)
                        row.update(host_status=hstatus if hstatus == "ok" else f"{hstatus}: {first_error(herr)}", host_wall_s=f"{hwall:.3f}", host_peak_rss_mb=f"{hrss:.0f}")
                    csv.row(*[row[c] for c in cols])
                    print(f"{variant} N={n:5d} rep {rep}: {status:24s} nvcc {wall:8.2f} s, {rss:7.0f} MB" + (f"  regs {row.get('registers')} spills {row.get('spill_stores')} cmem0 {row.get('cmem0_bytes')} SASS {row.get('sass_instructions')}" if status == 'ok' else f"  {row['why'][:120]}")
                          + (f"  | g++ {row['host_status'][:30]} {row['host_wall_s']} s" if row['host_wall_s'] else ""), flush=True)
                    if status != "ok":
                        stop = True
                        break
                    rep += 1
                    if wall >= 30:
                        break


if __name__ == "__main__":
    main()
