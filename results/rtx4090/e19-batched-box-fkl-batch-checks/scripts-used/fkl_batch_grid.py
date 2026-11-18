#!/usr/bin/env python3
"""fkl_batch's kernels in the SQLite export of the profile-mode nsys run (nothing is run): the kernels whose runtime API call starts inside an
NVTX range 'fkl_batch:call', with their grid, block, registers and shared memory, and their GPU duration (median, min, max).

  fkl_batch_grid.py <run.sqlite>
"""
import sqlite3, sys, statistics
c = sqlite3.connect(sys.argv[1]).cursor()
names = dict(c.execute("select id, value from StringIds"))
ranges = [(s, e) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text='fkl_batch:call' and end is not null")]
api = dict(c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))
cols = [r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
want = [x for x in ("gridX","gridY","gridZ","blockX","blockY","blockZ","registersPerThread","staticSharedMemory","dynamicSharedMemory") if x in cols]
rows = c.execute(f"select start, end, shortName, correlationId, {', '.join(want)} from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
inside = [r for r in rows if any(s <= api.get(r[3], -1) <= e for s, e in ranges)]
print(f"fkl_batch: {len(ranges)} calls in 'fkl_batch:call', {len(inside)} kernels launched from them")
seen = {}
for r in inside:
    key = (names.get(r[2], '?'),) + tuple(r[4:])
    seen[key] = seen.get(key, 0) + 1
for key, n in seen.items():
    print(f"  {n} x {key[0]}: " + ", ".join(f"{w} {v}" for w, v in zip(want, key[1:])))
d = [(r[1] - r[0]) / 1000 for r in inside]
print(f"  GPU duration per kernel, us: median {statistics.median(d):.2f}, min {min(d):.2f}, max {max(d):.2f}")
