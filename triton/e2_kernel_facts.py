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
"""Facts about the kernels of the E2 chain (FKL, the hand-written Triton kernel, TorchInductor): the memory instructions of their SASS and PTX, registers, instruction counts.

  python triton/e2_kernel_facts.py --out e2_kernel_facts.csv \
      fkl:10=build/e4/bin/e2_fkl_chain_cu  fkl:100=build/e4/bin/e2_fkl_chain_cu \
      triton_hand:10=<TRITON_CACHE_DIR of the worker> inductor_default:10=<cache dir> ...

A spec is impl:k=path. A directory is searched for *.cubin and *.ptx (what Triton and Inductor leave in their caches); an executable (FKL) is read with cuobjdump -sass / -ptx, and its kernel of
the chain of k operations is the one whose SASS has k/2 FFMA (one FFMA per Mul+Add pair). Per kernel: registers, SASS instructions, the global/local/shared loads and stores of the SASS with their
modifiers (LDG.E.U8.CONSTANT, STG.E.128, ...), the FFMA/FMUL/FADD, and the global loads and stores of the PTX with their types (ld.global.nc.u8, st.global.v4.f32, ...).
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import sass_tools as st  # noqa: E402

MEM = re.compile(r"^(LDG|STG|LD|ST|LDS|STS|LDL|STL|LDC|ATOM|ATOMG|RED)(\.|$)")
PRED = re.compile(r"^@!?U?P\w+\s+")


def sass_of(path, arch="sm_89"):
    out = subprocess.run(["cuobjdump", "-arch", arch, "-sass", path], capture_output=True, text=True).stdout
    return {name: st.normalise_sass(body, rename=False) for name, body in st.split_functions(out).items()}


def regs_of(path, arch="sm_89"):
    out = subprocess.run(["cuobjdump", "-arch", arch, "-res-usage", path], capture_output=True, text=True).stdout
    res, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"^\s*Function (\S+):", line)
        if m:
            cur = m.group(1)
        elif cur and "REG:" in line:
            res[cur] = int(re.search(r"REG:(\d+)", line).group(1))
            cur = None
    return res


def mem_histogram(lines):
    h = collections.Counter()
    for line in lines:
        if line.endswith(":"):
            continue
        body = PRED.sub("", line)
        op = body.split(" ")[0]
        if MEM.match(op):
            h[op] += 1
    return h


def op_count(lines, name):
    return sum(1 for l in lines if not l.endswith(":") and st.opcode(l) == name)


PTX_ENTRY = re.compile(r"^\.(?:visible\s+)?\.entry\s+(\S+?)\(", re.M)
PTX_MEM = re.compile(r"^\s*(?:@!?%p\d+\s+)?((?:ld|st)\.global[\w.]*)", re.M)


def ptx_entries(text):
    """{entry name: body} of a PTX file."""
    starts = [(m.start(), m.group(1)) for m in PTX_ENTRY.finditer(text)]
    out = {}
    for i, (pos, name) in enumerate(starts):
        out[name] = text[pos:(starts[i + 1][0] if i + 1 < len(starts) else len(text))]
    return out


def ptx_mem(body):
    h = collections.Counter()
    for m in PTX_MEM.finditer(body):
        h[m.group(1)] += 1
    return h


def fmt(h):
    return "; ".join(f"{k} x{v}" for k, v in sorted(h.items())) if h else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arch", default="sm_89")
    args = ap.parse_args()
    rows = []
    for spec in args.specs:
        left, path = spec.split("=", 1)
        impl, k = left.split(":")
        k = int(k)
        if os.path.isdir(path):
            # TorchInductor compiles several configurations of a pointwise kernel and keeps the fastest: its *.best_config names the cache entry of the one that runs
            best = {}
            for bc in glob.glob(os.path.join(path, "**", "*.best_config"), recursive=True):
                cfg = json.load(open(bc))
                best[cfg["triton_cache_hash"]] = cfg
            cubins = sorted(glob.glob(os.path.join(path, "**", "*.cubin"), recursive=True))
            if best:
                skipped = [c for c in cubins if os.path.basename(os.path.dirname(c)) not in best]
                cubins = [c for c in cubins if os.path.basename(os.path.dirname(c)) in best]
                print(f"{impl} k={k}: {len(cubins)} kernel(s) of the best configuration kept, {len(skipped)} other compiled configuration(s) left out")
            for cubin in cubins:
                sas, regs = sass_of(cubin, args.arch), regs_of(cubin, args.arch)
                meta = json.load(open(os.path.splitext(cubin)[0] + ".json")) if os.path.exists(os.path.splitext(cubin)[0] + ".json") else {}
                bcfg = best.get(os.path.basename(os.path.dirname(cubin)), {})
                nw = bcfg.get("num_warps", meta.get("num_warps"))
                config = (f"XBLOCK={bcfg['XBLOCK']} num_warps={nw} (best of the compiled configurations)" if bcfg else f"num_warps={nw}" if nw else "")
                threads = nw * 32 if nw else ""
                base = os.path.splitext(cubin)[0]
                ptx_text = open(base + ".ptx").read() if os.path.exists(base + ".ptx") else ""
                ptx = ptx_entries(ptx_text)
                for name, lines in sas.items():
                    body = next((b for n, b in ptx.items() if n == name), "")
                    rows.append(dict(impl=impl, k=k, kernel=name, config=config, threads_per_block=threads, regs=regs.get(name, -1), instr=sum(1 for l in lines if not l.endswith(":")),
                                     sass_mem=fmt(mem_histogram(lines)), ffma=op_count(lines, "FFMA"), fmul=op_count(lines, "FMUL"), fadd=op_count(lines, "FADD"), ptx_mem=fmt(ptx_mem(body)), source=cubin))
        else:
            sas, regs = sass_of(path, args.arch), regs_of(path, args.arch)
            ptx_text = subprocess.run(["cuobjdump", "-ptx", path], capture_output=True, text=True).stdout
            ptx = ptx_entries(ptx_text)
            for name, lines in sas.items():
                if op_count(lines, "FFMA") != k // 2 or "launchTransformDPP_Kernel" not in subprocess.run(["c++filt", name], capture_output=True, text=True).stdout:
                    continue
                body = ptx.get(name, "")
                rows.append(dict(impl=impl, k=k, kernel=name[:60] + "...", config="one element per thread, 32 x 8 threads per block", threads_per_block=256, regs=regs.get(name, -1), instr=sum(1 for l in lines if not l.endswith(":")),
                                 sass_mem=fmt(mem_histogram(lines)), ffma=op_count(lines, "FFMA"), fmul=op_count(lines, "FMUL"), fadd=op_count(lines, "FADD"), ptx_mem=fmt(ptx_mem(body)), source=path))
    for r in rows:
        print(f"{r['impl']:18s} k={r['k']:<4d} {r['config']}\n    regs {r['regs']:3d}  SASS instr {r['instr']:5d}  FFMA {r['ffma']:4d} FMUL {r['fmul']:3d} FADD {r['fadd']:3d}\n    SASS mem: {r['sass_mem']}\n    PTX mem:  {r['ptx_mem']}")
    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
