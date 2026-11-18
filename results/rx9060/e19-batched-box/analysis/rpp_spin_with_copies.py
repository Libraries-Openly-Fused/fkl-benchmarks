#!/usr/bin/env python3
"""The spin coverage of kernel_only:rpp with rpp's 30 copy blit kernels counted as part of its call.
rocprof_calls.py --spin leaves out the kernels of hipMemcpy* API calls (in E18 those were a preparation enqueued before the spin). In E19 the
30 hipMemcpy2DAsync are the batch building inside rpp's call, so here every kernel launched after a gpuSpin, up to the next one, is in the
iteration.
  rpp_spin_with_copies.py <trace folder> <prefix>
"""
import csv, statistics, sys
P = f"{sys.argv[1]}/{sys.argv[2]}_"
mk = [m for m in csv.DictReader(open(P + "marker_api_trace.csv")) if m["Function"] == "kernel_only:rpp"][0]
s, e = int(mk["Start_Timestamp"]), int(mk["End_Timestamp"])
api = {a["Correlation_Id"]: a for a in csv.DictReader(open(P + "hip_api_trace.csv"))}
ks = [k for k in csv.DictReader(open(P + "kernel_trace.csv")) if k["Correlation_Id"] in api and s <= int(api[k["Correlation_Id"]]["Start_Timestamp"]) <= e]
ks.sort(key=lambda k: int(k["Start_Timestamp"]))
its, cur = [], None
for k in ks:
    if "gpuSpin" in k["Kernel_Name"]:
        cur = [k, []]
        its.append(cur)
    elif cur is not None:
        cur[1].append(k)
covered, gaps, margins, counts = 0, [], [], set()
for spin, call in its:
    if not call:
        continue
    end = int(spin["End_Timestamp"])
    last = max(int(api[k["Correlation_Id"]]["End_Timestamp"]) for k in call)
    covered += last <= end
    margins.append((end - last) / 1e3)
    gaps.append((int(call[0]["Start_Timestamp"]) - end) / 1e3)
    counts.add(len(call))
print(f"kernel_only:rpp with its copies: {len(its)} iterations, kernels per iteration {sorted(counts)}, covered {covered} of {len(margins)}; "
      f"margin min {min(margins):.1f} us, median {statistics.median(margins):.1f} us; spin end -> first kernel median {statistics.median(gaps):.1f} us, "
      f"max {max(gaps):.1f} us (the first kernel is {its[1][1][0]['Kernel_Name'][:40]})")
