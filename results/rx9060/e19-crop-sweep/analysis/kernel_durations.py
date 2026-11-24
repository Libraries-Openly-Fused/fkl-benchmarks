#!/usr/bin/env python3
"""The GPU durations of every kernel inside each timed ROCTx range of rocprofv3 timing traces, to compare two windows (no GPU).

  kernel_durations.py <label>=<trace folder> [<label>=<trace folder> ...]

For every trace (e19_kernel_trace.csv or .csv.gz, e19_hip_api_trace.csv[.gz], e19_marker_api_trace.csv): per range (whole_call:<row>,
kernel_only:<row>, for fkl_batch and rpp) and kernel name, the number of dispatches and the median and mean duration (End - Start, us); and
per range, the median of the per-call GPU span (kernel_only: from the first kernel after each spin to the last kernel before the next spin;
whole_call: the time from one call's first kernel to the next call's first kernel is not used, only the sum of the kernel durations per
call). A kernel belongs to a range when the host API call that launched it (by correlation id) started inside the range.
"""
import csv
import gzip
import os
import re
import statistics
import sys


def rows(folder, name):
    p = os.path.join(folder, name)
    f = gzip.open(p + ".gz", "rt", newline="") if os.path.exists(p + ".gz") else open(p, newline="")
    with f:
        return list(csv.DictReader(f))


print("trace\trange\tkernel\tdispatches\tmedian_us\tmean_us")
spans = []
for arg in sys.argv[1:]:
    label, folder = arg.split("=", 1)
    marks = [m for m in rows(folder, "e19_marker_api_trace.csv") if re.match(r"(whole_call|kernel_only):(fkl_batch|rpp)$", m["Function"])]
    api = {a["Correlation_Id"]: int(a["Start_Timestamp"]) for a in rows(folder, "e19_hip_api_trace.csv")}
    ks = rows(folder, "e19_kernel_trace.csv")
    for m in marks:
        s, e = int(m["Start_Timestamp"]), int(m["End_Timestamp"])
        inside = sorted((k for k in ks if k["Correlation_Id"] in api and s <= api[k["Correlation_Id"]] <= e), key=lambda k: int(k["Start_Timestamp"]))
        by = {}
        for k in inside:
            name = re.match(r"(?:void )?([\w:]+)", k["Kernel_Name"]).group(1)
            by.setdefault(name, []).append((int(k["End_Timestamp"]) - int(k["Start_Timestamp"])) / 1e3)
        for name, d in by.items():
            print(f"{label}\t{m['Function']}\t{name}\t{len(d)}\t{statistics.median(d):.2f}\t{statistics.mean(d):.2f}")
        if m["Function"].startswith("kernel_only"):
            calls, cur = [], None
            for k in inside:
                if "gpuSpin" in k["Kernel_Name"]:
                    cur = []
                    calls.append(cur)
                elif cur is not None:
                    cur.append(k)
            span = [(int(c[-1]["End_Timestamp"]) - int(c[0]["Start_Timestamp"])) / 1e3 for c in calls if c]
            busy = [sum(int(k["End_Timestamp"]) - int(k["Start_Timestamp"]) for k in c) / 1e3 for c in calls if c]
            spans.append(f"{label}\t{m['Function']}\tper call after the spin: GPU span median {statistics.median(span):.2f} us, "
                         f"sum of kernel durations median {statistics.median(busy):.2f} us ({len(span)} calls)")
print()
print("\n".join(spans))
