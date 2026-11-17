#!/usr/bin/env python3
"""From the nsys trace of an E18_PROFILE=1 run of e18_box_cu (exported to SQLite): the kernels and the CUDA API calls of one call of every
implementation.

  profile_census.py <e18-box-profile.sqlite> --calls N [--out report.txt]

The profile mode calls every selected implementation once outside any range (a warm-up), then N times inside an NVTX range named after it. For every
such range:
  - the kernels whose launch API call started inside the range (joined to the GPU kernels by correlation id), per call (all of them / N), by name
  - every API call that started inside the range, by name, with its count per call
  - the API calls that allocate, free, copy or set memory, or create or use a texture, an array or a surface object (names holding Malloc, Free,
    Alloc, Memcpy, Memset, Tex, Array, Surf or Mipmap), and the memory copies and sets the GPU ran for API calls of the range
In the whole trace: every API call whose name holds Tex, Array, Surf or Mipmap, the warm-up calls and the setup included.
"""
import argparse
import collections
import re
import sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("--calls", type=int, required=True); ap.add_argument("--out", default=None)
args = ap.parse_args()
c = sqlite3.connect(args.db).cursor()
tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
names = dict(c.execute("select id, value from StringIds"))
ranges = [(t, s, e) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text is not null and end is not null order by start")]
api = []   # (start, end, name, correlation id)
for corr, nid, s, e in c.execute("select correlationId, nameId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME"):
    api.append((s, e, names.get(nid, "?"), corr))
kernels = {}   # correlation id -> kernel name
for corr, nid in c.execute("select correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    kernels[corr] = names.get(nid, "?")
gpuMem = collections.defaultdict(list)   # correlation id -> ["memcpy" | "memset"]
for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
    if table in tables:
        for (corr,) in c.execute(f"select correlationId from {table}"):
            gpuMem[corr].append(kind)
MEMORY = re.compile(r"Malloc|Free|Alloc|Memcpy|Memset|Tex|Array|Surf|Mipmap", re.I)
OBJECTS = re.compile(r"Tex|Array|Surf|Mipmap", re.I)
lines = [f"calls per range: {args.calls}"]
for text, rs, re_ in ranges:
    if ":" in text:
        continue
    inRange = [a for a in api if rs <= a[0] <= re_]
    ks = [kernels[a[3]] for a in inRange if a[3] in kernels]
    byName = collections.Counter(a[2] for a in inRange)
    flagged = {n: k for n, k in byName.items() if MEMORY.search(n)}
    mem = collections.Counter(kind for a in inRange for kind in gpuMem.get(a[3], []))
    lines.append(f"{text}: {len(ks)} kernels in the range, {len(ks) / args.calls:g} per call; by name (per call): "
                 + ", ".join(f"{n} {k / args.calls:g}" for n, k in sorted(collections.Counter(ks).items())))
    lines.append(f"  API calls in the range (per call): " + ", ".join(f"{n} {k / args.calls:g}" for n, k in sorted(byName.items())))
    lines.append(f"  memory, texture, array or surface API calls in the range: {dict(flagged) if flagged else 'none'}; "
                 f"GPU memory copies and sets of the range: {dict(mem) if mem else 'none'}")
objects = collections.Counter(a[2] for a in api if OBJECTS.search(a[2]))
lines.append(f"texture, array, surface or mipmapped-array API calls in the whole trace: {sum(objects.values())} {dict(objects)}")
report = "\n".join(lines)
print(report)
if args.out:
    open(args.out, "w").write(report + "\n")
