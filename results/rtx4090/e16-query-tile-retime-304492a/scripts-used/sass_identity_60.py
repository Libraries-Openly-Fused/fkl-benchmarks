#!/usr/bin/env python3
"""Device-code identity check of an FKL change that adds a defaulted template parameter to Sub, Mul and Div.

  sass_identity.py <bin dir A> <bin dir B> --tools <dir of sass_tools.py> [--arch sm_89] [--glob '*_cu'] [--out report.txt]

For every executable matching --glob in both directories, cuobjdump -sass is split per kernel and normalised with
sass_tools.py (addresses and encodings removed, registers NOT renamed), as scripts/sass_compare_builds.py does. Kernels
are paired by their demangled name, whitespace removed, in which ',fk::BinaryType' is removed where it is the 4th
template argument of fk::Sub, fk::Mul or fk::Div (the default argument that B adds; in A those templates have 3
parameters). Nothing else of the name is changed, so two kernels pair only if their names are otherwise identical.

Reports per executable and in total: kernels in A and B, pairs, pairs whose mangled names differ (renamed), pairs with
identical SASS, pairs with different SASS, kernels without a pair, and lists every difference and every unpaired kernel.
Exit status 0 when every kernel has a pair and every pair has identical SASS, 1 otherwise.
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("dir_a"); ap.add_argument("dir_b")
ap.add_argument("--tools", required=True); ap.add_argument("--arch", default="sm_89")
ap.add_argument("--glob", default="*_cu"); ap.add_argument("--out", default=None)
args = ap.parse_args()
sys.path.insert(0, args.tools)
import sass_tools as st  # noqa: E402

TEMPLATES = ("fk::Sub<", "fk::Mul<", "fk::Div<")
OPEN, CLOSE = "<({[", ">)}]"


def top_level_args(s, start):
    """s[start] is just after '<': returns (list of (begin, end) of the top-level arguments, index of the matching '>')."""
    depth, begin, spans, i = 0, start, [], start
    while i < len(s):
        c = s[i]
        if c in OPEN:
            depth += 1
        elif c in CLOSE:
            if depth == 0:
                spans.append((begin, i))
                return spans, i
            depth -= 1
        elif c == "," and depth == 0:
            spans.append((begin, i))
            begin = i + 1
        i += 1
    raise ValueError(f"unbalanced template argument list in {s[:200]}")


def drop_default(name):
    """Removes ',fk::BinaryType' where it is the 4th argument of fk::Sub/Mul/Div; returns (new name, number removed)."""
    removed, i = 0, 0
    while True:
        hits = [(name.find(t, i), t) for t in TEMPLATES]
        hits = [(p, t) for p, t in hits if p >= 0 and (p == 0 or not (name[p - 1].isalnum() or name[p - 1] == "_"))]
        if not hits:
            return name, removed
        p, t = min(hits)
        start = p + len(t)
        spans, close = top_level_args(name, start)
        if len(spans) == 4 and name[spans[3][0]:spans[3][1]] == "fk::BinaryType":
            name = name[:spans[2][1]] + name[spans[3][1]:]
            removed += 1
        i = start


def kernels(path):
    res = subprocess.run(["cuobjdump", "-arch", args.arch, "-sass", path], capture_output=True, text=True)
    if res.returncode != 0:
        return None
    funcs = st.split_functions(res.stdout)
    names = list(funcs)
    dem = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    out, seen, collisions, removed = {}, {}, 0, 0
    for mangled, d in zip(names, dem):
        key, n = drop_default(re.sub(r"\s+", "", d))
        removed += n
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            collisions += 1
            key = f"{key}#{seen[key]}"
        out[key] = (mangled, st.normalise_sass(funcs[mangled], rename=False))
    return out, collisions, removed


def listing(d):
    return {f for f in os.listdir(d) if fnmatch.fnmatch(f, args.glob) and os.access(os.path.join(d, f), os.X_OK)}


ea, eb = listing(args.dir_a), listing(args.dir_b)
common = sorted(ea & eb)
lines = [f"A = {args.dir_a}", f"B = {args.dir_b}",
         f"arch {args.arch}, executables in both: {len(common)}, only in A: {sorted(ea - eb)}, only in B: {sorted(eb - ea)}",
         st.tool_version("cuobjdump"),
         "pairing: demangled name, whitespace removed, ',fk::BinaryType' removed where it is the 4th template argument of "
         "fk::Sub, fk::Mul or fk::Div"]
tot = dict(a=0, b=0, pairs=0, renamed=0, same=0, diff=0, only_a=0, only_b=0, coll=0, removed_a=0, removed_b=0)
per, details = [], []
for exe in common:
    ra, rb = kernels(os.path.join(args.dir_a, exe)), kernels(os.path.join(args.dir_b, exe))
    if ra is None or rb is None:
        lines.append(f"{exe}: cuobjdump failed on {'A' if ra is None else 'B'}")
        tot["diff"] += 1
        continue
    (ka, ca, rma), (kb, cb, rmb) = ra, rb
    e = dict(a=len(ka), b=len(kb), pairs=0, renamed=0, same=0, diff=0, only_a=0, only_b=0, coll=ca + cb,
             removed_a=rma, removed_b=rmb)
    for key in sorted(set(ka) | set(kb)):
        if key not in kb:
            e["only_a"] += 1; details.append(f"  ONLY IN A  {exe}: {key[:240]}")
        elif key not in ka:
            e["only_b"] += 1; details.append(f"  ONLY IN B  {exe}: {key[:240]}")
        else:
            e["pairs"] += 1
            (ma, sa), (mb, sb) = ka[key], kb[key]
            if ma != mb:
                e["renamed"] += 1
            if sa == sb:
                e["same"] += 1
            else:
                e["diff"] += 1
                details.append(f"  DIFFERENT  {exe}: {len(sa)} -> {len(sb)} lines  {key[:240]}")
    for k in tot:
        tot[k] += e[k]
    per.append(f"  {exe}: kernels A {e['a']}, B {e['b']}; pairs {e['pairs']} (renamed {e['renamed']}), identical {e['same']}, "
               f"different {e['diff']}, only in A {e['only_a']}, only in B {e['only_b']}; key collisions {e['coll']}; "
               f"BinaryType arguments removed A {e['removed_a']}, B {e['removed_b']}")
lines.append(f"TOTAL: kernels A {tot['a']}, B {tot['b']}; pairs {tot['pairs']}, of which renamed {tot['renamed']}; "
             f"identical SASS {tot['same']}, different {tot['diff']}; only in A {tot['only_a']}, only in B {tot['only_b']}; "
             f"key collisions {tot['coll']}; BinaryType arguments removed from the names: A {tot['removed_a']}, B {tot['removed_b']}")
lines += per + details
text = "\n".join(lines)
print(text)
if args.out:
    with open(args.out, "w") as f:
        f.write(text + "\n")
sys.exit(0 if tot["diff"] == 0 and tot["only_a"] == 0 and tot["only_b"] == 0 and tot["coll"] == 0 else 1)
