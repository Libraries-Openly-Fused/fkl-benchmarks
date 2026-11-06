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
"""What the SASS of the E3 chains of variant A contains: instructions per operation, per operation type, and how the parameters are read. Compile only; no GPU.

  python scripts/e3_sass_check.py --work DIR [--mixed 6,12,600,1000,2000] [--single-n 100,200] [--jobs 8] [--analyse-only]

Generates (fkl-self/compiletime/gen_chain.py, variant A: Mul, Add, Sub, Div, Max, Min on float, distinct parameters) the mixed chains of the lengths --mixed and, for each of the six operation types, chains of that type alone of the
lengths --single-n, compiles them with the flags of the E3 repeat (nvcc, one core each, --threads 1, -O3 -DNDEBUG -std=c++17, sm_89 + PTX, -ftemplate-depth=1000, -lineinfo, --resource-usage; the library is external/fkl) and reads
`cuobjdump -sass` of the kernel (the largest function, instruction lines counted exactly as the E3 tables do, scripts/sass_tools.py). Writes DIR/e3_sass_check.md: totals and the marginal instructions per operation, the fixed part, the
slope per operation type, the opcodes of a 6-operation cycle, the instructions with a constant-bank operand c[0x0][offset], and LDC / LDG / STG / LDL / STL counts (the parameters of the chain are read as constant-bank operands: no LDC, no local memory).
"""
import argparse
import collections
import importlib.util
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import sass_tools as st  # noqa: E402

CONST = re.compile(r"c\[0x0\]\[(0x[0-9a-f]+)\]")
TYPES = ["Mul", "Add", "Sub", "Div", "Max", "Min"]


def generator():
    spec = importlib.util.spec_from_file_location("gen_chain", os.path.join(ROOT, "fkl-self", "compiletime", "gen_chain.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def generate(work, mixed, single_n):
    os.makedirs(os.path.join(work, "src"), exist_ok=True)
    g = generator()
    full_period = list(g.VARIANTS["A"]["period"])
    jobs = []
    for n in mixed:
        g.VARIANTS["A"]["period"] = full_period
        sys.argv = ["gen_chain.py", "A", str(n), os.path.join(work, "src", f"A_{n}.cu")]
        g.main()
        jobs.append(f"A_{n}")
    for k in range(6):
        for n in single_n:
            g.VARIANTS["A"]["period"] = [full_period[k]]
            sys.argv = ["gen_chain.py", "A", str(n), os.path.join(work, "src", f"S{k}_{n}.cu")]
            g.main()
            jobs.append(f"S{k}_{n}")
    g.VARIANTS["A"]["period"] = full_period
    return jobs


def compile_all(work, jobs, ncpu_list):
    os.makedirs(os.path.join(work, "bin"), exist_ok=True)
    os.makedirs(os.path.join(work, "log"), exist_ok=True)
    cuda_home = os.environ.get("FKLB_CUDA_HOME", "/usr/local/cuda-13.4")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    include = os.path.join(ROOT, "external", "fkl", "include")

    def one(i_name):
        i, name = i_name
        core = ncpu_list[i % len(ncpu_list)]
        cmd = ["taskset", "-c", str(core), nvcc, "-std=c++17", "-O3", "-DNDEBUG", "--generate-code=arch=compute_89,code=[compute_89,sm_89]", "--threads", "1", "-ftemplate-depth=1000", "-lineinfo",
               "--resource-usage", "-I", include, os.path.join(work, "src", name + ".cu"), "-o", os.path.join(work, "bin", name)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        open(os.path.join(work, "log", name + ".txt"), "w").write(r.stdout + r.stderr)
        return name, r.returncode

    with ThreadPoolExecutor(max_workers=len(ncpu_list)) as ex:
        for name, rc in ex.map(one, list(enumerate(jobs))):
            print(f"{name}: nvcc exit {rc}", flush=True)


def analyse(binary):
    out = subprocess.run(["cuobjdump", "-arch", "sm_89", "-sass", binary], capture_output=True, text=True).stdout
    funcs = st.split_functions(out)
    name, body = max(funcs.items(), key=lambda kv: len(kv[1]))          # the E3 convention: the largest function
    lines = st.normalise_sass(body, rename=False)
    ops = collections.Counter(st.opcode(l) for l in lines if st.opcode(l))
    const_lines = [l for l in lines if st.opcode(l) and CONST.search(l)]
    return dict(instr=sum(ops.values()), ops=ops, const_instr=len(const_lines), const_by_op=collections.Counter(st.opcode(l) for l in const_lines),
                offsets=len({m.group(1) for l in const_lines for m in CONST.finditer(l)}), functions=len(funcs))


def ptxas_line(path):
    t = open(path).read() if os.path.exists(path) else ""
    m = re.search(r"Used (\d+) registers.*?(\d+) bytes cmem\[0\]", t)
    return (m.group(1), m.group(2)) if m else ("?", "?")


def report(work, mixed, single_n):
    L = ["# E3 variant A: what the SASS contains (compile only; scripts/e3_sass_check.py)", "",
         "| N | instructions | per Op | constant-bank-operand instructions | distinct c[0x0] offsets | LDC | LDG | STG | LDL | STL | registers | cmem[0] bytes |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    res = {}
    for n in mixed:
        p = os.path.join(work, "bin", f"A_{n}")
        if not os.path.exists(p):
            continue
        r = res[n] = analyse(p)
        reg, cm = ptxas_line(os.path.join(work, "log", f"A_{n}.txt"))
        L.append(f"| {n} | {r['instr']} | {r['instr'] / n:.3f} | {r['const_instr']} | {r['offsets']} | {r['ops'].get('LDC', 0)} | {r['ops'].get('LDG', 0)} | {r['ops'].get('STG', 0)} | {r['ops'].get('LDL', 0)} | {r['ops'].get('STL', 0)} | {reg} | {cm} |")
    ns = sorted(res)
    L += ["", "Marginal instructions per Op between consecutive lengths: " + "; ".join(f"{a}->{b}: {(res[b]['instr'] - res[a]['instr']) / (b - a):.3f}" for a, b in zip(ns, ns[1:]))]
    if len(ns) >= 2:
        a, b = ns[-2], ns[-1]
        slope = (res[b]['instr'] - res[a]['instr']) / (b - a)
        L.append(f"Fixed part (extrapolated to N = 0 from {a} and {b}): {res[b]['instr'] - slope * b:.0f} instructions (thread index, bounds, addresses, the load and the store, the division slow-path routine, padding).")
    big = [n for n in ns if n % 6 == 0 and n >= 12]
    if len(big) >= 2:
        a, b = big[0], big[-1]
        cyc = (b - a) / 6
        d = collections.Counter(res[b]['ops']); d.subtract(res[a]['ops'])
        dc = collections.Counter(res[b]['const_by_op']); dc.subtract(res[a]['const_by_op'])
        L += ["", f"Per 6-Op cycle (N = {b} minus N = {a}, {cyc:.0f} cycles): {sum(d.values()) / cyc:.2f} instructions: " + ", ".join(f"{k} {v / cyc:.2f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if abs(v) / cyc > 0.005),
              f"Instructions with a constant-bank operand per cycle: {(res[b]['const_instr'] - res[a]['const_instr']) / cyc:.2f}: " + ", ".join(f"{k} {v / cyc:.2f}" for k, v in dc.items() if abs(v) / cyc > 0.005)]
    L += ["", "Single-type chains (the slope between the two lengths), instructions per Op:", "", "| type | instructions per Op | constant-bank-operand instructions per Op | opcodes per Op | LDC | LDL | STL |", "|---|---:|---:|---|---:|---:|---:|"]
    if len(single_n) >= 2:
        lo, hi = single_n[0], single_n[-1]
        tot = 0.0
        for k, name in enumerate(TYPES):
            pa, pb = os.path.join(work, "bin", f"S{k}_{lo}"), os.path.join(work, "bin", f"S{k}_{hi}")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            a, b = analyse(pa), analyse(pb)
            span = hi - lo
            d = collections.Counter(b['ops']); d.subtract(a['ops'])
            tot += (b['instr'] - a['instr']) / span
            L.append(f"| {name} | {(b['instr'] - a['instr']) / span:.2f} | {(b['const_instr'] - a['const_instr']) / span:.2f} | " + ", ".join(f"{o} {v / span:.2f}" for o, v in sorted(d.items(), key=lambda kv: -kv[1]) if v / span > 0.3) +
                     f" | {b['ops'].get('LDC', 0)} | {b['ops'].get('LDL', 0)} | {b['ops'].get('STL', 0)} |")
        L.append(f"\nSum of the six isolated slopes: {tot:.2f} per 6-Op cycle (the mixed chain is smaller where a Mul followed by an Add is fused into one FFMA).")
    open(os.path.join(work, "e3_sass_check.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--mixed", default="6,12,600,1000,2000")
    ap.add_argument("--single-n", default="100,200")
    ap.add_argument("--cores", default="8,9,10,11,12,13,14,15", help="the cores the compiles are pinned to (one compile per core at a time)")
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    mixed = [int(x) for x in a.mixed.split(",")]
    single_n = [int(x) for x in a.single_n.split(",")]
    if not a.analyse_only:
        jobs = generate(a.work, mixed, single_n)
        compile_all(a.work, jobs, [int(c) for c in a.cores.split(",")])
    report(a.work, mixed, single_n)


if __name__ == "__main__":
    main()
