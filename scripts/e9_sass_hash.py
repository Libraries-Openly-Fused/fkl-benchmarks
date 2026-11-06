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
"""SASS identity of the kernels of the E9 binary: per kernel the number of instruction lines and two hashes.

  python scripts/e9_sass_hash.py [BINARY] [--out FILE]

text hash    sha256 of the instruction lines of `cuobjdump -arch sm_89 -sass` with the address (/*0010*/) and the encoding words (/* 0x... */) removed, whitespace collapsed (the instruction text);
code hash    sha256 of the same lines WITH their encoding words (the 128-bit instruction and its control word: the machine code, scheduling included), addresses removed.
The kernels are told apart by their demangled type (the plane policy of the resize read: 1 = cvGS, 0 = FastNPP; VectorReorderRT = FastNPP's swap; short_listing = the type of the short listing).
Used to back the statement that the kernel of the short listing (its own C++ instantiation, its own Gamma type) has the machine code of M1.
"""
import hashlib
import re
import subprocess
import sys


def kernels(binary):
    out = subprocess.run(["cuobjdump", "-arch", "sm_89", "-sass", binary], capture_output=True, text=True).stdout
    res, cur = [], None
    for line in out.splitlines():
        m = re.match(r"\s+Function : (\S+)", line)
        if m:
            cur = [m.group(1), []]
            res.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"\s+/\*([0-9a-f]{4})\*/\s+(.*?)\s*/\*\s*(0x[0-9a-f]+)\s*\*/\s*$", line)
        if m:                                   # instruction line: address, text, first encoding word
            cur[1].append([re.sub(r"\s+", " ", m.group(2)).strip(), m.group(3)])
            continue
        m = re.match(r"\s+/\*\s*(0x[0-9a-f]+)\s*\*/\s*$", line)
        if m and cur[1]:                        # second encoding word (control word) of the previous instruction
            cur[1][-1].append(m.group(1))
    return res


def name_of(mangled):
    d = subprocess.run(["c++filt", mangled], capture_output=True, text=True).stdout
    if "launchTransformDPP_Kernel" not in d:
        return None
    cv = "PlanePolicy)1" in d
    swap = "VectorReorderRT" in d
    if "short_listing" in d:
        return "short listing"
    if cv:
        return "M1 (= long listing)" if swap else "R1"
    return "R2" if swap else "M2"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    binary = args[0] if args else "build/local-rtx4090/bin/e9_cross_library_cu"
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    sha = hashlib.sha256(open(binary, "rb").read()).hexdigest()
    lines = [f"binary {binary} sha256 {sha}", "", f"{'kernel':22s} {'instruction lines':>17s}  {'text hash':16s}  {'code hash':16s}"]
    rows = {}
    for mangled, ins in kernels(binary):
        n = name_of(mangled)
        if n is None:
            continue
        text = "\n".join(i[0] for i in ins)
        code = "\n".join(" ".join(i) for i in ins)
        rows[n] = (len(ins), hashlib.sha256(text.encode()).hexdigest()[:16], hashlib.sha256(code.encode()).hexdigest()[:16])
    for n in ("M1 (= long listing)", "short listing", "R1", "M2", "R2"):
        if n in rows:
            lines.append(f"{n:22s} {rows[n][0]:17d}  {rows[n][1]}  {rows[n][2]}")
    if "M1 (= long listing)" in rows and "short listing" in rows:
        a, b = rows["M1 (= long listing)"], rows["short listing"]
        lines += ["", f"short listing against M1: instruction lines {'equal' if a[0] == b[0] else 'DIFFERENT'} ({b[0]} / {a[0]}), text {'IDENTICAL' if a[1] == b[1] else 'DIFFERENT'}, machine code (with encodings) {'IDENTICAL' if a[2] == b[2] else 'DIFFERENT'}"]
    print("\n".join(lines))
    if out:
        open(out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
