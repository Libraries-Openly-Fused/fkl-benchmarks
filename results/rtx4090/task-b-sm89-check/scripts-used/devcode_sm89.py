#!/usr/bin/env python3
"""Task B: the sm_89 device code of the same CUDA executables built against two FKL commits, kernel by kernel (nothing is run).

  devcode_sm89.py --tools <dir of sass_tools.py> --a <tag>=<dir> --b <tag>=<dir> [--glob '*_cu'] [--out report.txt] [--csv kernels.csv]

For every executable that matches --glob, found anywhere under each directory (by file name; a name found twice under one directory is an
error), cuobjdump -arch sm_89 -sass is split per kernel and normalised by sass_tools.normalise_sass (the /*address*/ comments and the
encodings removed, registers NOT renamed), as scripts/sass_compare_builds.py and scripts/kernel_resources.py do, and cuobjdump -res-usage gives
each kernel's registers, stack, shared memory, local memory and constant bank 0 (the bank that holds the kernel parameters).
Kernels are paired by their mangled name with the per-compile hash of the anonymous namespace (_GLOBAL__N__<hash>_) removed, as
kernel_resources.py does; nothing else of the name is changed. A kernel whose mangled name changed is therefore "only in A" and "only in B".
Per executable: kernels in A and B, identical, different, only in A, only in B; for every different and every unpaired kernel its demangled
name and, before and after, the instruction count (the normalised lines that are not labels), registers, stack, shared and local memory and
constant bank 0. Executables found on one side only are listed. Exit 0 when every executable is on both sides and every kernel is identical.
"""
import argparse
import csv
import fnmatch
import hashlib
import os
import re
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--tools", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--glob", default="*_cu"); ap.add_argument("--arch", default="sm_89")
ap.add_argument("--out", default=None); ap.add_argument("--csv", default=None)
args = ap.parse_args()
sys.path.insert(0, args.tools)
import sass_tools as st  # noqa: E402

FUNC_RES = re.compile(r"^\s*Function (\S+):")
RES_NAMES = ("REG", "STACK", "SHARED", "LOCAL", r"CONSTANT\[0\]")
RES = {n: re.compile(n + r":(\d+)") for n in RES_NAMES}
HASH = re.compile(r"_GLOBAL__N__[0-9a-f]+_")


def key_of(mangled):
    return HASH.sub("_GLOBAL__N__X_", mangled)


def executables(root):
    found = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("CMakeFiles", ".cmake")]
        for f in files:
            p = os.path.join(d, f)
            if fnmatch.fnmatch(f, args.glob) and os.path.isfile(p) and os.access(p, os.X_OK):
                if f in found:
                    sys.exit(f"{f} is found twice under {root}: {found[f]} and {p}")
                found[f] = p
    return found


def cuobjdump(what, path):
    r = subprocess.run(["cuobjdump", "-arch", args.arch, what, path], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"cuobjdump {what} failed on {path}: {r.stderr.strip()[:300]}")
    return r.stdout


def kernels(path):
    res, cur = {}, None
    for line in cuobjdump("-res-usage", path).splitlines():
        m = FUNC_RES.match(line)
        if m:
            cur = m.group(1); continue
        if cur and "REG:" in line:
            res[cur] = tuple(int(RES[n].search(line).group(1)) if RES[n].search(line) else 0 for n in RES_NAMES)
            cur = None
    out = {}
    for name, body in st.split_functions(cuobjdump("-sass", path)).items():
        lines = st.normalise_sass(body, rename=False)
        k = key_of(name)
        if k in out:
            sys.exit(f"{path}: two kernels with the key {k}")
        out[k] = dict(mangled=name, lines=lines, instr=sum(1 for l in lines if not l.endswith(":")),
                      res=res.get(name, (-1,) * 5), sha=hashlib.sha256("\n".join(lines).encode()).hexdigest())
    return out


def demangle(names):
    d = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, d))


def fmt(k):
    r = k["res"]
    return f"{k['instr']} instructions, {r[0]} registers, stack {r[1]} B, shared {r[2]} B, local {r[3]} B, cbank0 {r[4]} B"


tag_a, dir_a = args.a.split("=", 1); tag_b, dir_b = args.b.split("=", 1)
ea, eb = executables(dir_a), executables(dir_b)
common = sorted(set(ea) & set(eb))
tool = subprocess.run(["cuobjdump", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
lines = [f"A = {tag_a}: {dir_a}", f"B = {tag_b}: {dir_b}",
         f"arch {args.arch}; {tool}; executables matching {args.glob}: A {len(ea)}, B {len(eb)}, in both {len(common)}",
         f"only in A: {sorted(set(ea) - set(eb))}", f"only in B: {sorted(set(eb) - set(ea))}"]
tot = dict(a=0, b=0, same=0, diff=0, only_a=0, only_b=0)
details, rows, per_exe = [], [], []
for exe in common:
    ka, kb = kernels(ea[exe]), kernels(eb[exe])
    dem = demangle(sorted({k["mangled"] for k in ka.values()} | {k["mangled"] for k in kb.values()}))
    n = dict(a=len(ka), b=len(kb), same=0, diff=0, only_a=0, only_b=0)
    for key in sorted(set(ka) | set(kb)):
        x, y = ka.get(key), kb.get(key)
        if x and y:
            status = "identical" if x["lines"] == y["lines"] else "different"
            n["same" if status == "identical" else "diff"] += 1
        else:
            status = "only in A" if x else "only in B"
            n["only_a" if x else "only_b"] += 1
        name = dem[(x or y)["mangled"]]
        rows.append([exe, status, name, *(["", "", "", "", "", "", ""] if not x else [x["instr"], *x["res"], x["sha"][:16]]),
                     *(["", "", "", "", "", "", ""] if not y else [y["instr"], *y["res"], y["sha"][:16]])])
        if status != "identical":
            details.append(f"  {exe}: {status.upper()}  {name[:400]}")
            if x: details.append(f"      A: {fmt(x)}")
            if y: details.append(f"      B: {fmt(y)}")
    for k in tot: tot[k] += n[k]
    per_exe.append(f"  {exe}: kernels A {n['a']}, B {n['b']}; identical {n['same']}, different {n['diff']}, only in A {n['only_a']}, only in B {n['only_b']}"
                   + ("" if n["diff"] or n["only_a"] or n["only_b"] else "  (all identical)"))
lines.append(f"kernels: A {tot['a']}, B {tot['b']}; identical {tot['same']}, different {tot['diff']}, only in A {tot['only_a']}, only in B {tot['only_b']}")
lines.append("per executable:"); lines += per_exe
lines.append("every kernel that is different or on one side only (A = before, B = after):"); lines += details or ["  none"]
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["executable", "status", "demangled",
                    "A_instructions", "A_registers", "A_stack", "A_shared", "A_local", "A_cbank0", "A_sass_sha256_16",
                    "B_instructions", "B_registers", "B_stack", "B_shared", "B_local", "B_cbank0", "B_sass_sha256_16"])
        w.writerows(rows)
ok = not (set(ea) ^ set(eb)) and tot["diff"] == 0 and tot["only_a"] == 0 and tot["only_b"] == 0
sys.exit(0 if ok else 1)
