#!/usr/bin/env python3
"""The GPU work of every ROCTx range of an E18 box rocprofv3 trace (rocprofv3 --kernel-trace --memory-copy-trace --memory-allocation-trace
--hip-runtime-trace --marker-trace --output-format csv).

  rocprof_calls.py <trace folder> <file prefix> [--out calls.txt] [--tsv calls.tsv]

A range is one call (E18_PROFILE=1: every call of every selected variant in a range named after the variant). For every range:
  - the HIP runtime API calls the program's thread made inside it (by their host start time), counted by function;
  - the kernels dispatched by those calls (joined by correlation id), in order, with the registers, LDS, scratch and grid the runtime reports;
  - the memory copies of those calls (joined by correlation id) and the memory allocations or frees inside the range (by correlation id or time).
The summary gives, per range name, the kernels of each call and whether any call allocates, frees, copies or sets memory. A range with more than
20 kernels (the whole_call and kernel_only ranges of a timed run: 101 calls each) is listed by kernel name and count.
--spin (phase 2, for the trace of a complete timed run): for every range kernel_only:<version> (fklb::timePreparedLaunch: one warm-up call without
a spin, then 100 iterations of the preparation, gpuSpin and the call), the iterations: the kernels launched after each gpuSpin by the API calls
before the next one, not counting the kernels of hipMemcpy* calls (a preparation that copies enqueues them before the spin; the ROI writes of rpp
and rpp_ltrb are host stores, with no copy). Covered: the
host's launch call of the iteration's last kernel returned before the spin ended on the GPU (the GPU never waited for the host inside the timed
interval); the margin is the spin's end minus that return. The gaps on the GPU: from the end of the spin to the start of the first kernel, and
between consecutive kernels. rocprofv3 gives the host and the GPU timestamps in one time base (ns).
"""
import argparse
import collections
import csv
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("prefix")
ap.add_argument("--out", default=None)
ap.add_argument("--tsv", default=None)
ap.add_argument("--spin", action="store_true")
args = ap.parse_args()


def read(kind):
    path = os.path.join(args.folder, f"{args.prefix}_{kind}.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


markers = [m for m in read("marker_api_trace") if m["Domain"] == "MARKER_CORE_RANGE_API"]
api = read("hip_api_trace")
kernels = read("kernel_trace")
copies = read("memory_copy_trace")
allocations = read("memory_allocation_trace")
kernels_by_corr = collections.defaultdict(list)
for k in kernels:
    kernels_by_corr[k["Correlation_Id"]].append(k)
copies_by_corr = collections.defaultdict(list)
for c in copies:
    copies_by_corr[c["Correlation_Id"]].append(c)


def short(name):
    s = name[5:] if name.startswith("void ") else name
    m = re.match(r"[A-Za-z_][\w:]*", s)
    return m.group(0) if m else s


out = []
tsv = [["range", "call", "kernel", "vgpr", "sgpr", "lds_bytes", "scratch_bytes", "workgroup", "grid"]]
summary = collections.OrderedDict()
calls = collections.Counter()
for m in markers:
    start, end, thread = int(m["Start_Timestamp"]), int(m["End_Timestamp"]), m["Thread_Id"]
    inside = [a for a in api if a["Thread_Id"] == thread and start <= int(a["Start_Timestamp"]) <= end]
    corr = {a["Correlation_Id"] for a in inside}
    ks = sorted((k for c in corr for k in kernels_by_corr[c]), key=lambda k: int(k["Start_Timestamp"]))
    cs = [c for cid in corr for c in copies_by_corr[cid]]
    al = [a for a in allocations if a["Correlation_Id"] in corr or start <= int(a["Start_Timestamp"]) <= end]
    functions = collections.Counter(a["Function"] for a in inside)
    name = m["Function"]
    calls[name] += 1
    out.append(f"range {name} #{calls[name]}: {len(inside)} HIP API calls ({', '.join(f'{f} {n}' for f, n in sorted(functions.items()))})")
    if len(ks) > 20:
        for kn, n in collections.Counter(short(k["Kernel_Name"]) for k in ks).most_common():
            out.append(f"    kernel {kn}: {n}")
    for k in ks if len(ks) <= 20 else []:
        out.append(f"    kernel {short(k['Kernel_Name'])}: VGPR {k['VGPR_Count']}, SGPR {k['SGPR_Count']}, LDS {k['LDS_Block_Size']} B, scratch "
                   f"{k['Scratch_Size']} B, workgroup {k['Workgroup_Size_X']}x{k['Workgroup_Size_Y']}x{k['Workgroup_Size_Z']}, grid "
                   f"{k['Grid_Size_X']}x{k['Grid_Size_Y']}x{k['Grid_Size_Z']} (correlation {k['Correlation_Id']})")
        tsv.append([name, calls[name], short(k["Kernel_Name"]), k["VGPR_Count"], k["SGPR_Count"], k["LDS_Block_Size"], k["Scratch_Size"],
                    f"{k['Workgroup_Size_X']}x{k['Workgroup_Size_Y']}x{k['Workgroup_Size_Z']}", f"{k['Grid_Size_X']}x{k['Grid_Size_Y']}x{k['Grid_Size_Z']}"])
    for c in cs if len(cs) <= 20 else []:
        out.append(f"    memory copy {c['Direction']} (correlation {c['Correlation_Id']})")
    if len(cs) > 20:
        out.append(f"    memory copies: {len(cs)}")
    for a in al:
        out.append(f"    memory {a['Operation']} {a['Allocation_Size']} B")
    s = summary.setdefault(name, {"kernels": [], "copy_api": 0, "alloc_api": 0, "memset_api": 0, "copies": 0, "allocations": 0})
    s["kernels"].append(tuple(short(k["Kernel_Name"]) for k in ks))
    s["copy_api"] += sum(n for f, n in functions.items() if f.startswith("hipMemcpy"))
    s["alloc_api"] += sum(n for f, n in functions.items() if re.match(r"hip(Malloc|Free|HostMalloc|HostFree|ExtMalloc|HostAlloc|MallocAsync|FreeAsync)", f))
    s["memset_api"] += sum(n for f, n in functions.items() if f.startswith("hipMemset"))
    s["copies"] += len(cs)
    s["allocations"] += len(al)
out.append("")
out.append("summary per range name (one range = one call):")
for name, s in summary.items():
    distinct = sorted(set(s["kernels"]))
    out.append(f"  {name}: {len(s['kernels'])} calls; kernels per call {sorted(set(len(k) for k in s['kernels']))}; "
               f"the same kernels in every call: {'yes' if len(distinct) == 1 else 'no'}")
    for d in distinct:
        out.append(f"    {' -> '.join(d)}")
    out.append(f"    inside the calls: hipMemcpy* API calls {s['copy_api']}, memory copies {s['copies']}, allocation/free API calls {s['alloc_api']}, "
               f"memory allocations/frees {s['allocations']}, hipMemset* API calls {s['memset_api']}")
if args.spin:
    out.append("")
    out.append("spin coverage of the kernel_only ranges:")
    api_by_corr = {a["Correlation_Id"]: a for a in api}
    if not any(m["Function"].startswith("kernel_only:") for m in markers):
        out.append("  no kernel_only range in this trace")
    for m in markers:
        if not m["Function"].startswith("kernel_only:"):
            continue
        start, end, thread = int(m["Start_Timestamp"]), int(m["End_Timestamp"]), m["Thread_Id"]
        inside = sorted((a for a in api if a["Thread_Id"] == thread and start <= int(a["Start_Timestamp"]) <= end),
                        key=lambda a: int(a["Start_Timestamp"]))
        iterations, current = [], None
        for a in inside:
            for k in sorted(kernels_by_corr[a["Correlation_Id"]], key=lambda k: int(k["Start_Timestamp"])):
                if "gpuSpin" in k["Kernel_Name"]:
                    current = {"spin": k, "kernels": []}
                    iterations.append(current)
                elif current is not None and not a["Function"].startswith("hipMemcpy"):
                    current["kernels"].append(k)
        covered, margins, first_gaps, inner_gaps, counts = 0, [], [], [], collections.Counter()
        for it in iterations:
            ks = sorted(it["kernels"], key=lambda k: int(k["Start_Timestamp"]))
            counts[len(ks)] += 1
            if not ks:
                continue
            spin_end = int(it["spin"]["End_Timestamp"])
            last_return = max(int(api_by_corr[k["Correlation_Id"]]["End_Timestamp"]) for k in ks)
            covered += last_return <= spin_end
            margins.append(spin_end - last_return)
            first_gaps.append(int(ks[0]["Start_Timestamp"]) - spin_end)
            inner_gaps += [int(b["Start_Timestamp"]) - int(a2["End_Timestamp"]) for a2, b in zip(ks, ks[1:])]
        if not iterations:
            out.append(f"  {m['Function']}: no gpuSpin in the range")
            continue
        out.append(f"  {m['Function']}: {len(iterations)} iterations, kernels per iteration {dict(counts)}; covered {covered} of {len(margins)}; "
                   f"margin min {min(margins) / 1000:.1f} us, median {sorted(margins)[len(margins) // 2] / 1000:.1f} us; spin end -> first kernel "
                   f"max {max(first_gaps) / 1000:.2f} us; between kernels max {max(inner_gaps) / 1000 if inner_gaps else 0:.2f} us")
text = "\n".join(out) + "\n"
if args.out:
    with open(args.out, "w") as f:
        f.write(text)
if args.tsv:
    with open(args.tsv, "w") as f:
        f.writelines("\t".join(str(x) for x in row) + "\n" for row in tsv)
print(text, end="")
