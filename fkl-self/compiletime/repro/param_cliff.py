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
"""Minimal reproducers (no FKL) for the jump in registers and spills of kernels with many parameters (E3, see README.md).

  python param_cliff.py --out <dir> [--core 14] [--arch sm_89]

Two families, compiled only (nothing is executed), each kernel reading ALL of its parameters with compile-time indices:
  bytes  one struct parameter of B bytes (B = 4000 .. 4200 and larger), in four forms:
           plain   `Blob b`                                (by value)
           gc      `const __grid_constant__ Blob b`
           gcref   `const __grid_constant__ Blob b`, read through a `const Blob&` passed to a __device__ function (how FKL passes its operations)
           split   the same bytes as two `const __grid_constant__` parameters of B / 2 bytes each
  count  K parameters of one float each, `const __grid_constant__ float`, K around 1024 (1020 .. 1030, 1500), summed with a fold
         and K parameters of 4-byte structs, to see whether the threshold is the NUMBER of kernel parameters or their bytes.
For each: the PTX (mov/ld.param, cvta.param, st.local, .local) and the cubin (registers, spill stores and loads in bytes, STL, LDL, instructions).
"""
import argparse
import csv
import os
import re
import subprocess
import sys

HEAD = "#include <cuda_runtime.h>\n"


def blob_kernels(b):
    n = b // 4
    idx = sorted(set(list(range(0, n, max(1, n // 40))) + [n - 1]))
    read = " + ".join(f"X.v[{i}]" for i in idx)
    s = f"{HEAD}struct Blob {{ float v[{n}]; }};\n"
    s += f"__device__ __forceinline__ float readAll(const Blob& X) {{ return {read}; }}\n"
    s += f"__global__ void k_plain(Blob b, float* out) {{ const Blob& X = b; out[threadIdx.x] = {read}; }}\n"
    s += f"__global__ void k_gc(const __grid_constant__ Blob b, float* out) {{ const Blob& X = b; out[threadIdx.x] = {read}; }}\n"
    s += "__global__ void k_gcref(const __grid_constant__ Blob b, float* out) { out[threadIdx.x] = readAll(b); }\n"
    h = (b // 2) // 4
    idx2 = sorted(set(list(range(0, h, max(1, h // 20))) + [h - 1]))
    read2 = " + ".join(f"a.v[{i}] + c.v[{i}]" for i in idx2)
    s += f"struct Half {{ float v[{h}]; }};\n"
    s += f"__global__ void k_split(const __grid_constant__ Half a, const __grid_constant__ Half c, float* out) {{ out[threadIdx.x] = {read2}; }}\n"
    return s


def count_kernel(k, struct):
    t = "P" if struct else "float"
    s = HEAD + ("struct P { float x; };\n" if struct else "")
    s += "template <typename... T> __global__ void k_count(float* out, const __grid_constant__ T... p) {\n"
    s += "    out[threadIdx.x] = (" + ("p.x" if struct else "p") + " + ...);\n}\n"
    args = ", ".join(f"{t}{{{i}.0f}}" if struct else f"{i}.0f" for i in range(k))
    s += f"void launch(float* out) {{ k_count<<<1, 32>>>(out, {args}); }}\n"
    return s


def getarg_kernel(k):
    """K `const __grid_constant__ float` parameters read ONLY through fk::get_arg (the first and the last), as FKL's executor does"""
    s = HEAD + "#include <fused_kernel/core/utils/parameter_pack_utils.h>\n"
    s += "template <typename... T> __global__ void k_getarg(float* out, const __grid_constant__ T... p) {\n"
    s += "    out[threadIdx.x] = fk::get_arg<0>(p...) + fk::get_arg<sizeof...(p) - 1>(p...);\n}\n"
    s += f"void launch(float* out) {{ k_getarg<<<1, 32>>>(out, {', '.join(f'{i}.0f' for i in range(k))}); }}\n"
    return s


def run(cmd, core):
    p = subprocess.run(["taskset", "-c", str(core)] + cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def analyse(path, name_filter, core, arch, out, include=None):
    """returns the PTX and cubin figures of the (first) kernel whose name matches"""
    r = {}
    inc = ["-I", include] if include else []
    rc, _, err = run(["nvcc", "-std=c++17", "-O3", f"-arch={arch}"] + inc + ["-ptx", path, "-o", out + ".ptx"], core)
    if rc != 0:
        return {"error": err.strip().splitlines()[-1][:200]}
    ptx = open(out + ".ptx").read()
    r["ptx_ld_param"] = len(re.findall(r"ld\.param", ptx))
    r["ptx_cvta_param"] = len(re.findall(r"cvta\.param", ptx))
    r["ptx_st_local"] = len(re.findall(r"st\.local", ptx))
    r["ptx_ld_local"] = len(re.findall(r"ld\.local", ptx))
    m = re.search(r"\.local \.align \d+ \.b8 __local_depot\d+\[(\d+)\]", ptx)
    r["ptx_local_depot_bytes"] = int(m.group(1)) if m else 0
    rc, _, err = run(["nvcc", "-std=c++17", "-O3", f"-arch={arch}", "--resource-usage"] + inc + ["-cubin", path, "-o", out + ".cubin"], core)
    if rc != 0:
        r["error"] = err.strip().splitlines()[-1][:200]
        return r
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads.*?Used (\d+) registers.*?, (\d+) bytes cmem\[0\]", err, re.S)
    if m:
        r.update(stack_bytes=int(m[1]), spill_stores_bytes=int(m[2]), spill_loads_bytes=int(m[3]), registers=int(m[4]), cmem0_bytes=int(m[5]))
    sass = subprocess.run(["cuobjdump", "-sass", out + ".cubin"], capture_output=True, text=True).stdout
    r["sass_instructions"] = len(re.findall(r"^\s+/\*[0-9a-f]{4}\*/", sass, re.M))
    r["sass_STL"] = len(re.findall(r"\bSTL", sass))
    r["sass_LDL"] = len(re.findall(r"\bLDL", sass))
    r["sass_LDC"] = len(re.findall(r"\bLDC", sass))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--core", type=int, default=14)
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--fkl-include", help="include directory of the library: adds the family getarg (fk::get_arg on K parameters)")
    ap.add_argument("--patched-include", help="include directory of a copy of the library with get_arg changed: the getarg family is also run against it")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []

    def add(family, form, size, path, flt, include=None):
        r = analyse(path, flt, args.core, args.arch, os.path.join(args.out, os.path.basename(path)[:-3]), include)
        rows.append({"family": family, "form": form, "size": size, **r})
        print(family, form, size, {k: v for k, v in r.items() if k != "error"} if "error" not in r else r["error"], flush=True)

    for b in (4000, 4096, 4200, 8192):
        src = os.path.join(args.out, f"bytes_{b}.cu")
        open(src, "w").write(blob_kernels(b))
        # one TU holds the four forms; analyse each kernel by compiling a TU with only that kernel would be cleaner: do that by splitting
        text = open(src).read()
        for form in ("plain", "gc", "gcref", "split"):
            keep = []
            for line in text.splitlines():
                if line.startswith("__global__ void k_") and f"k_{form}(" not in line:
                    continue
                keep.append(line)
            p = os.path.join(args.out, f"bytes_{b}_{form}.cu")
            open(p, "w").write("\n".join(keep) + "\n")
            add("bytes", form, b, p, form)
    for k in (1000, 1020, 1023, 1024, 1025, 1030, 1500):
        for struct in (False, True):
            p = os.path.join(args.out, f"count_{k}_{'struct' if struct else 'float'}.cu")
            open(p, "w").write(count_kernel(k, struct))
            add("count", "struct4" if struct else "float", k, p, "k_count")
    if args.fkl_include:
        for label, inc in (("library", args.fkl_include), ("patched", args.patched_include)):
            if not inc:
                continue
            for k in (1000, 1022, 1023, 1024, 1025, 1026, 1030, 1500, 3000):
                p = os.path.join(args.out, f"getarg_{label}_{k}.cu")
                open(p, "w").write(getarg_kernel(k))
                add("getarg", label, k, p, "k_getarg", inc)
    cols = sorted({c for r in rows for c in r}, key=lambda c: (c not in ("family", "form", "size"), c))
    with open(os.path.join(args.out, "param_cliff.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
