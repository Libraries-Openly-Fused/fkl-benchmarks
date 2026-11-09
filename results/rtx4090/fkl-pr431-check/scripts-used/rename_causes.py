#!/usr/bin/env python3
"""Why the kernels of two builds have different mangled names (names only, no SASS).

  rename_causes.py <bin dir A> <bin dir B> [--arch sm_89] [--glob '*_cu']

Kernels are paired as in sass_identity.py: demangled name, whitespace removed, ',fk::BinaryType' removed where it is the
4th template argument of fk::Sub/Mul/Div. For every pair whose mangled names differ, the cause is classified:
  hash      the names are equal once the module id of the anonymous namespace (_GLOBAL__N__<8 hex>_) is masked;
            nvcc derives that id from the build path, so it differs between two build folders
  default   the names still differ after masking (the defaulted template argument of Sub/Mul/Div)
and 'default' pairs are also counted by whether their names contain an anonymous namespace.
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("dir_a"); ap.add_argument("dir_b")
ap.add_argument("--arch", default="sm_89"); ap.add_argument("--glob", default="*_cu")
args = ap.parse_args()

TEMPLATES = ("fk::Sub<", "fk::Mul<", "fk::Div<")
OPEN, CLOSE = "<({[", ">)}]"
HASH = re.compile(r"_GLOBAL__N__[0-9a-f]{8}_")


def top_level_args(s, start):
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
    raise ValueError("unbalanced")


def drop_default(name):
    i = 0
    while True:
        hits = [(name.find(t, i), t) for t in TEMPLATES]
        hits = [(p, t) for p, t in hits if p >= 0 and (p == 0 or not (name[p - 1].isalnum() or name[p - 1] == "_"))]
        if not hits:
            return name
        p, t = min(hits)
        start = p + len(t)
        spans, _ = top_level_args(name, start)
        if len(spans) == 4 and name[spans[3][0]:spans[3][1]] == "fk::BinaryType":
            name = name[:spans[2][1]] + name[spans[3][1]:]
        i = start


def names(path):
    out = subprocess.run(["cuobjdump", "-arch", args.arch, "-sass", path], capture_output=True, text=True).stdout
    mangled = re.findall(r"Function : (\S+)", out)
    dem = subprocess.run(["c++filt"], input="\n".join(mangled), capture_output=True, text=True).stdout.splitlines()
    keyed, seen = {}, {}
    for m, d in zip(mangled, dem):
        k = drop_default(re.sub(r"\s+", "", d))
        seen[k] = seen.get(k, 0) + 1
        if seen[k] > 1:
            k = f"{k}#{seen[k]}"
        keyed[k] = m
    return keyed


def listing(d):
    return {f for f in os.listdir(d) if fnmatch.fnmatch(f, args.glob) and os.access(os.path.join(d, f), os.X_OK)}


tot = dict(pairs=0, same=0, hash_only=0, default=0, default_with_anon=0, unpaired=0)
for exe in sorted(listing(args.dir_a) & listing(args.dir_b)):
    a, b = names(os.path.join(args.dir_a, exe)), names(os.path.join(args.dir_b, exe))
    e = dict(pairs=0, same=0, hash_only=0, default=0, default_with_anon=0, unpaired=len(set(a) ^ set(b)))
    for k in set(a) & set(b):
        e["pairs"] += 1
        ma, mb = a[k], b[k]
        if ma == mb:
            e["same"] += 1
        elif HASH.sub("_GLOBAL__N__X_", ma) == HASH.sub("_GLOBAL__N__X_", mb):
            e["hash_only"] += 1
        else:
            e["default"] += 1
            if HASH.search(ma):
                e["default_with_anon"] += 1
    for k in tot:
        tot[k] += e[k]
    print(f"  {exe}: " + ", ".join(f"{k} {v}" for k, v in e.items()))
print("TOTAL: " + ", ".join(f"{k} {v}" for k, v in tot.items()))
