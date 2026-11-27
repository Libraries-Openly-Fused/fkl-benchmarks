#!/usr/bin/env python3
"""RT-4, the code check of A against "old" (CPU only: the binaries are only read with cuobjdump; nothing is run).

  sass_linecounts_vs_60.py <#60's sass_identity.py> <#60's sass-identity-24478f0-vs-b10fc66.txt> <dir of sass_tools.py> <bin dir> <program>...

"old" = #63's B = #60's head build (scratch/pin-433/head-b10fc66, fkl-benchmarks 2160c70, FKL b10fc66), removed after #63. What #60's
record keeps of its device code: for every kernel whose SASS differed from 24478f0's, the B side's normalised SASS line count
("DIFFERENT  <program>: <A lines> -> <B lines> lines  <key, first 240 characters>"). This script runs #60's own kernels() function
(cuobjdump -sass, sass_tools.normalise_sass, its pairing key) on the given binaries and compares their line counts with those recorded
B-side counts. The record cuts every key at 240 characters, and several E18 kernels share those first 240 characters, so the comparison
is per group of kernels with the same 240-character prefix: the sorted line counts recorded for the group against the sorted line counts of
this build's kernels with that prefix (equal, or different). Run it with #60's cuobjdump (CUDA 13.3) first in PATH.
"""
import os, re, sys

script, record, tools, bindir, progs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
src = open(script).read()
cut = src.index("ea, eb = listing(")
sys.argv = [script, bindir, bindir, "--tools", tools]
g = {"__name__": "sass_identity_60", "__file__": script}
exec(compile(src[:cut], script, "exec"), g)
rec = {}
for l in open(record):
    m = re.match(r"\s+DIFFERENT\s+(\S+): (\d+) -> (\d+) lines\s+(.*)$", l.rstrip("\n"))
    if m:
        rec.setdefault(m.group(1), {}).setdefault(m.group(4), []).append(int(m.group(3)))
print(g["st"].tool_version("cuobjdump"))
tot = dict(equal=0, different=0, kernels=0)
for p in progs:
    exe = f"{p}_cu"
    ka, _, _ = g["kernels"](os.path.join(bindir, exe))
    groups = rec.get(exe, {})
    print(f"{exe}: {len(ka)} kernels here; {sum(len(v) for v in groups.values())} recorded for #63's B, in {len(groups)} prefix groups")
    for key, counts in groups.items():
        here = sorted(len(v[1]) for k, v in ka.items() if k[:240] == key)
        ok = here == sorted(counts)
        tot["equal" if ok else "different"] += 1; tot["kernels"] += len(counts)
        print(f"  {'equal    ' if ok else 'DIFFERENT'} group of {len(counts)}: recorded {sorted(counts)}, here {here}  {key[:110]}")
print(f"TOTAL: {tot['kernels']} recorded kernels in {tot['equal'] + tot['different']} groups: {tot['equal']} groups equal, {tot['different']} different")
sys.exit(0 if tot["different"] == 0 else 1)
