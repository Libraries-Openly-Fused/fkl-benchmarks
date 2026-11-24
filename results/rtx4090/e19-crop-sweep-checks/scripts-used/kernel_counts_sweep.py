#!/usr/bin/env python3
"""Kernels per call of every E19 row at every N, from the SQLite exports of the profile-mode nsys runs of the crop sweep (nothing is run).

  kernel_counts_sweep.py <folder of profile-N<NNN>.sqlite> [--out report.txt] [--csv counts.csv]

Each profile run (E19_PROFILE=1, E19_PROFILE_CALLS=10) calls every row once outside any range, then 10 times, each call in the NVTX range
"<row>:call". A kernel belongs to a call when the runtime API call that launched it (the same correlationId) starts inside the call's range.
Per N and row: the kernels per call (every call), the expected count (fkl_loop N, fkl_batch 1, opencv_cuda 4N, npp 3N), the kernel names,
fkl_batch's grid and block, the host's enqueue of one call (the range's length) and the GPU span of one call (first kernel start to last kernel
end), medians over the 10 calls.
"""
import argparse, collections, csv, glob, os, re, sqlite3, statistics
ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--out", default=None); ap.add_argument("--csv", default=None)
args = ap.parse_args()
EXPECT = {"fkl_loop": lambda n: n, "fkl_batch": lambda n: 1, "opencv_cuda": lambda n: 4 * n, "npp": lambda n: 3 * n}
lines, out = [], []
for db in sorted(glob.glob(os.path.join(args.folder, "profile-N*.sqlite"))):
    n = int(re.search(r"profile-N(\d+)\.sqlite", db).group(1))
    c = sqlite3.connect(db).cursor()
    names = dict(c.execute("select id, value from StringIds"))
    ranges = sorted((s, e, t) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text is not null and end is not null"))
    api_start = dict(c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))
    kernels = [(s, e, names.get(nm, "?"), corr, gx, gy, gz, bx, by, bz) for s, e, nm, corr, gx, gy, gz, bx, by, bz in
               c.execute("select start, end, shortName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ from CUPTI_ACTIVITY_KIND_KERNEL")]
    lines.append(f"N = {n}:")
    for row in ("fkl_loop", "fkl_batch", "opencv_cuda", "npp"):
        calls = [(s, e) for s, e, t in ranges if t == row + ":call"]
        per, host, span, byname, grids = [], [], [], [], set()
        for s, e in calls:
            ks = sorted(k for k in kernels if k[3] in api_start and s <= api_start[k[3]] <= e)
            per.append(len(ks)); host.append((e - s) / 1000.0); byname.append(collections.Counter(k[2] for k in ks))
            if ks: span.append((max(k[1] for k in ks) - ks[0][0]) / 1000.0)
            if row == "fkl_batch": grids |= {(k[4], k[5], k[6], k[7], k[8], k[9]) for k in ks}
        exp = EXPECT[row](n)
        ok = len(calls) == 10 and all(p == exp for p in per)
        g = "; grid x,y,z / block: " + ", ".join(f"{a} x {b} x {z} / {bx} x {by} x {bz}" for a, b, z, bx, by, bz in sorted(grids)) if grids else ""
        lines.append(f"  {row}: {len(calls)} calls; kernels per call {sorted(set(per))}, expected {exp}: {'yes' if ok else 'NO'}; "
                     f"names {dict(byname[0]) if byname else {}}{g}; host enqueue median {statistics.median(host):.1f} us; GPU span median "
                     f"{statistics.median(span):.1f} us")
        out.append({"crops": n, "row": row, "calls": len(calls), "kernels_per_call": "/".join(map(str, sorted(set(per)))), "expected": exp,
                    "as_expected": "yes" if ok else "no", "host_enqueue_median_us": f"{statistics.median(host):.1f}",
                    "gpu_span_median_us": f"{statistics.median(span):.1f}"})
t = "\n".join(lines); print(t)
if args.out: open(args.out, "w").write(t + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
