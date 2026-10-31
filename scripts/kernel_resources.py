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
"""The same kernels of the same program built against several trees of the library: registers, stack, shared memory, the constant bank 0 (which holds the kernel parameters, so its size follows the
parameter size) and the SASS (instruction count and a hash of the normalised SASS, registers NOT renamed), side by side.

  python scripts/kernel_resources.py --out kernels.csv --match 'BatchRead' tag1=path/to/binary tag2=path/to/binary ...

Kernels are matched across the binaries by their mangled name (the per-compile hash of the anonymous namespace, _GLOBAL__N__<hash>_, is removed). --match is a regex on the demangled name; only the
kernels it matches are listed. One CSV row per (kernel, tag); the text summary says how many kernels the tags have in common and how many of them have the same SASS, the same registers and the same
constant bank 0.
"""
import argparse
import collections
import csv
import hashlib
import re
import subprocess
import sys

import sass_tools as st

FUNC_RES = re.compile(r"^\s*Function (\S+):")
RES = {name: re.compile(name + r":(\d+)") for name in ("REG", "STACK", "SHARED", "LOCAL", r"CONSTANT\[0\]")}
HASH = re.compile(r"_GLOBAL__N__[0-9a-f]+_")


def key_of(mangled):
    return HASH.sub("_GLOBAL__N__X_", mangled)


def resources(binary, arch):
    out = subprocess.run(["cuobjdump", "-arch", arch, "-res-usage", binary], capture_output=True, text=True).stdout
    res, cur = {}, None
    for line in out.splitlines():
        m = FUNC_RES.match(line)
        if m:
            cur = m.group(1)
            continue
        if cur and "REG:" in line:
            res[cur] = tuple(int(RES[n].search(line).group(1)) if RES[n].search(line) else 0 for n in ("REG", "STACK", "SHARED", "LOCAL", r"CONSTANT\[0\]"))
            cur = None
    return res


def sass(binary, arch):
    out = subprocess.run(["cuobjdump", "-arch", arch, "-sass", binary], capture_output=True, text=True).stdout
    return {name: st.normalise_sass(body, rename=False) for name, body in st.split_functions(out).items()}


def demangle(names):
    d = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, d))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binaries", nargs="+", help="tag=path")
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--match", default=".", help="regex on the demangled kernel name")
    ap.add_argument("--out", default=None, help="CSV")
    args = ap.parse_args()
    rx = re.compile(args.match)
    tags, per_tag = [], {}
    for spec in args.binaries:
        tag, path = spec.split("=", 1)
        tags.append(tag)
        res, sas = resources(path, args.arch), sass(path, args.arch)
        dem = demangle(list(sas))
        rows = {}
        for name, lines in sas.items():
            if not rx.search(dem[name]):
                continue
            instr = sum(1 for l in lines if not l.endswith(":"))
            r = res.get(name, (-1,) * 5)
            rows[key_of(name)] = dict(demangled=dem[name], regs=r[0], stack=r[1], shared=r[2], local=r[3], cmem0=r[4], instr=instr,
                                      sass_sha1=hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12])
        per_tag[tag] = rows
    common = set.intersection(*[set(r) for r in per_tag.values()])
    lines = [f"binaries: {', '.join(args.binaries)}", f"arch {args.arch}, kernels matching /{args.match}/: " + ", ".join(f"{t}: {len(per_tag[t])}" for t in tags) + f"; in all: {len(common)}"]
    same_sass = sum(1 for k in common if len({per_tag[t][k]["sass_sha1"] for t in tags}) == 1)
    same_regs = sum(1 for k in common if len({per_tag[t][k]["regs"] for t in tags}) == 1)
    same_cmem = sum(1 for k in common if len({per_tag[t][k]["cmem0"] for t in tags}) == 1)
    lines.append(f"of the {len(common)} kernels in all binaries: same SASS {same_sass}, same registers {same_regs}, same constant bank 0 {same_cmem}")
    for t in tags:
        only = set(per_tag[t]) - common
        if only:
            lines.append(f"  only in some binaries, listed under {t}: {len(only)}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "tag", "regs", "stack", "shared", "local", "cmem0", "sass_instructions", "sass_sha1", "demangled"])
            for k in sorted(set().union(*[set(r) for r in per_tag.values()])):
                for t in tags:
                    r = per_tag[t].get(k)
                    if r:
                        w.writerow([k, t, r["regs"], r["stack"], r["shared"], r["local"], r["cmem0"], r["instr"], r["sass_sha1"], r["demangled"]])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
