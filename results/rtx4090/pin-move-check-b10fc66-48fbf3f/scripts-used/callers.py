#!/usr/bin/env python3
"""C-PM-CUDA: the functions of a binary that call a given function directly, and whether each caller is a .cold clone (CPU only: objdump).

  callers.py <binary> <substring of the callee's demangled name> [--within <substring>]
"""
import re, subprocess, sys
binary, callee = sys.argv[1], sys.argv[2]
within = sys.argv[sys.argv.index("--within") + 1] if "--within" in sys.argv else None
out = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", "-w", "-j", ".text", binary], capture_output=True, text=True).stdout
cur, hits = None, {}
for l in out.splitlines():
    m = re.match(r"^[0-9a-f]+ <(.*)>:$", l)
    if m:
        cur = m.group(1); continue
    if cur and "\tcall" in l and callee in l.split("<", 1)[-1]:
        hits[cur] = hits.get(cur, 0) + 1
for f, n in sorted(hits.items()):
    if within and within not in f:
        continue
    print(f"{'COLD' if '[clone .cold]' in f else 'HOT '} {n:3d} {f[:220]}")
