#!/usr/bin/env python3
"""C-NV-A: where the cudaFuncSetAttribute calls of a binary run, per function (CPU only: objdump -d -C; nothing is run).

  attr_calls.py <binary> [--match SUBSTRING] [--out report.txt]

For every .text function that calls cudaFuncSetAttribute, the sequence of its calls of interest in address order:
  G+ __cxa_guard_acquire, G- __cxa_guard_release (the one-time initialization of a function-local static), A cudaFuncSetAttribute,
  O cudaOccupancyMaxActiveBlocksPerMultiprocessor, D cudaGetDevice, V cudaDeviceGetAttribute, L cudaLaunchKernel (the launch of a host stub),
  S a call of a kernel's host stub (__device_stub__ or a function named after a kernel), F a call of another function that calls
  cudaFuncSetAttribute itself (its name follows).
An A inside an innermost guarded block, a G+ followed by its G- with no other G+ or G- between them in address order, is in the static's
initializer: it runs once, the first time. Any other A runs on every call of the function (where GCC interleaves the inlined code of several
statics, a site after a G+ that is followed by another G+ is counted as on every call). Totals: the functions, the A sites in an initializer
and the A sites on every call.
"""
import argparse
import re
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("binary"); ap.add_argument("--match", default=""); ap.add_argument("--out")
a = ap.parse_args()
out = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", "-w", "-j", ".text", a.binary], capture_output=True, text=True, check=True).stdout
HDR = re.compile(r"^[0-9a-f]+ <(.*)>:$")
CALL = re.compile(r"\tcall\s+[0-9a-f]+ <(.+)>$")
funcs, cur = {}, None
for line in out.splitlines():
    h = HDR.match(line)
    if h:
        cur = h.group(1); funcs.setdefault(cur, []); continue
    m = CALL.search(line)
    if m and cur is not None:
        funcs[cur].append(m.group(1))
setters = {f for f, calls in funcs.items() if any(c.startswith("cudaFuncSetAttribute@plt") for c in calls)}


def code(c):
    if c.startswith("__cxa_guard_acquire@plt"): return "G+"
    if c.startswith("__cxa_guard_release@plt"): return "G-"
    if c.startswith("cudaFuncSetAttribute@plt"): return "A"
    if c.startswith("cudaOccupancyMaxActiveBlocksPerMultiprocessor"): return "O"
    if c.startswith("cudaGetDevice@plt"): return "D"
    if c.startswith("cudaDeviceGetAttribute@plt"): return "V"
    if c.startswith("cudaLaunchKernel@plt"): return "L"
    if "__device_stub__" in c or re.search(r"launchDPP_Kernel<", c.split("(")[0]): return "S"
    base = c.split("+")[0]
    if base in setters: return "F:" + base[:90]
    return None


lines, tot_f, tot_once, tot_every = [], 0, 0, 0
for f in sorted(setters):
    if a.match and a.match not in f:
        continue
    seq = [x for x in (code(c) for c in funcs[f]) if x]
    once, every, pending = 0, 0, None   # pending: the A sites seen since the last G+, when no G+ or G- came after it
    for x in seq:
        if x == "G+":
            if pending: every += pending
            pending = 0
        elif x == "G-":
            if pending is not None: once += pending
            pending = None
        elif x == "A":
            if pending is None: every += 1
            else: pending += 1
    if pending: every += pending
    tot_f += 1; tot_once += once; tot_every += every
    lines.append(f"{f[:230]}\n    A in an initializer {once}, A on every call {every}; calls: {' '.join(seq)}")
lines.insert(0, f"{a.binary}: {tot_f} functions call cudaFuncSetAttribute; A sites in a static's initializer {tot_once}, on every call {tot_every}")
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
