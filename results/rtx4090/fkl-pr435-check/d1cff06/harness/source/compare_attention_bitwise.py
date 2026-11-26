#!/usr/bin/env python3
"""C-NV-A: the outputs of pr435_attention at FKL d1cff06 against those of the C-NV window at 48fbf3f, bit for bit (CPU only: reads files).

  compare_attention_bitwise.py <stdout of the 48fbf3f run> <its output folder> <stdout of the d1cff06 run> <its output folder>
                               [--sha256 <the C-NV window's harness-outputs-sha256.txt>] [--out report.txt]

The outputs of a run are the files <name>.bin of its "OUT <name> <bytes> bytes" lines. Both runs must list the same names. Every file must be
byte for byte the same in both folders. With --sha256, each 48fbf3f file must also still have the sha256 that the C-NV window recorded for
./head-48fbf3f/<name>.bin. Exit 0 when everything holds.
"""
import argparse
import hashlib
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("stdout_a"); ap.add_argument("dir_a"); ap.add_argument("stdout_b"); ap.add_argument("dir_b")
ap.add_argument("--sha256"); ap.add_argument("--out")
a = ap.parse_args()
OUT = re.compile(r"^OUT (\S+)\s+(\d+) bytes$")


def outputs(stdout):
    names = []
    for line in open(stdout):
        m = OUT.match(line.rstrip("\n"))
        if m:
            names.append((m.group(1), int(m.group(2))))
    return names


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


na, nb = outputs(a.stdout_a), outputs(a.stdout_b)
lines, bad = [f"A {a.dir_a} ({len(na)} outputs), B {a.dir_b} ({len(nb)} outputs)"], 0
if [n for n, _ in na] != [n for n, _ in nb]:
    bad += 1
    lines.append(f"  FAIL the two runs list different outputs: only A {sorted(set(n for n, _ in na) - set(n for n, _ in nb))}, "
                 f"only B {sorted(set(n for n, _ in nb) - set(n for n, _ in na))}")
recorded = {}
if a.sha256:
    for line in open(a.sha256):
        h, p = line.split(None, 1)
        recorded[p.strip()] = h
for (n, size_a), (_, size_b) in zip(na, nb):
    pa, pb = os.path.join(a.dir_a, n + ".bin"), os.path.join(a.dir_b, n + ".bin")
    ha, hb = sha(pa), sha(pb)
    same = ha == hb and open(pa, "rb").read() == open(pb, "rb").read()
    rec = recorded.get(f"./head-48fbf3f/{n}.bin") if a.sha256 else None
    rec_ok = (rec == ha) if a.sha256 else True
    if not same or not rec_ok or size_a != size_b or os.path.getsize(pa) != size_a:
        bad += 1
    lines.append(f"  {'ok  ' if same and rec_ok else 'FAIL'} {n}: {size_a} B, sha256 {ha[:16]} / {hb[:16]}"
                 + ("" if not a.sha256 else f", recorded by the C-NV window {'the same' if rec_ok else 'DIFFERENT (' + str(rec)[:16] + ')'}"))
lines.append(f"result: {len(na) - bad if bad <= len(na) else 0} of {len(na)} outputs identical bit for bit"
             + (", every 48fbf3f output as the C-NV window recorded it" if a.sha256 and not bad else "") + ("" if not bad else f"; {bad} FAIL"))
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
sys.exit(1 if bad else 0)
