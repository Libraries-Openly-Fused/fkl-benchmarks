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
"""Whole-build SASS comparison: does a change of the library change the code generated for any kernel?

  sass_compare_builds.py <bin_dir_A> <bin_dir_B> [--arch sm_89] [--glob '*_cu'] [--out report.txt]

For every executable matching --glob that exists in both directories, cuobjdump -sass is split per kernel, normalised as in
sass_tools.py (addresses and encodings removed) and compared kernel by kernel, by mangled name. Reports the number of
kernels identical, different, and present in only one of the two builds, and lists the different ones with their
instruction counts. (cuobjdump pads columns differently per file, so raw text diffs of whole dumps show false differences;
this is why the comparison is per normalised kernel.)
"""
import argparse
import fnmatch
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sass_tools as st  # noqa: E402


def kernels_of(path, arch):
    res = subprocess.run(["cuobjdump", "-arch", arch, "-sass", path], capture_output=True, text=True)
    if res.returncode != 0:
        return None
    out = {}
    for name, body in st.split_functions(res.stdout).items():
        out[name] = st.normalise_sass(body, rename=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a"); ap.add_argument("dir_b")
    ap.add_argument("--arch", default="sm_89"); ap.add_argument("--glob", default="*_cu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names_a = {f for f in os.listdir(args.dir_a) if fnmatch.fnmatch(f, args.glob) and os.access(os.path.join(args.dir_a, f), os.X_OK)}
    names_b = {f for f in os.listdir(args.dir_b) if fnmatch.fnmatch(f, args.glob) and os.access(os.path.join(args.dir_b, f), os.X_OK)}
    common = sorted(names_a & names_b)
    lines = [f"A = {args.dir_a}", f"B = {args.dir_b}", f"arch {args.arch}, executables in both: {len(common)}, only in A: {sorted(names_a - names_b)}, only in B: {sorted(names_b - names_a)}"]
    lines.append(st.tool_version("cuobjdump"))
    same = diff = only_a = only_b = 0
    different = []
    for exe in common:
        ka, kb = kernels_of(os.path.join(args.dir_a, exe), args.arch), kernels_of(os.path.join(args.dir_b, exe), args.arch)
        if ka is None or kb is None:
            lines.append(f"{exe}: cuobjdump failed on {'A' if ka is None else 'B'}")
            continue
        for name in sorted(set(ka) | set(kb)):
            if name not in kb:
                only_a += 1
            elif name not in ka:
                only_b += 1
            elif ka[name] == kb[name]:
                same += 1
            else:
                diff += 1
                different.append((exe, name, len(ka[name]), len(kb[name])))
    lines.append(f"kernels identical: {same}, different: {diff}, only in A: {only_a}, only in B: {only_b}")
    for exe, name, na, nb in different:
        short = subprocess.run(["c++filt", name], capture_output=True, text=True).stdout.strip()
        lines.append(f"  DIFFERENT in {exe}: {na} -> {nb} lines  {short[:200]}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
