#!/usr/bin/env python3
"""The kernel parameter sizes of e19_crop_sweep_cu, from `cuobjdump -elf` of the binary (nothing is run).

  kernel_param_sizes.py <elf.txt> [--out report.txt]

Per kernel of the cubin: EIATTR_CBANK_PARAM_SIZE (the bytes of the kernel's parameters in constant bank 0) and each parameter's size from
EIATTR_KPARAM_INFO. fkl_batch's kernel (morphologyDPPKernel) is named by its N (BatchRead<PROCESS_ALL, N, ...> in its mangled name).
"""
import argparse, re
ap = argparse.ArgumentParser(); ap.add_argument("elf"); ap.add_argument("--out", default=None); args = ap.parse_args()
text = open(args.elf).read()
out, rows = [], []
for sec in re.split(r"\n(?=\.nv\.info\._Z)", text)[1:]:
    name = sec.split("\n", 1)[0]
    m = re.search(r"PlanePolicyE0ELm(\d+)E", name)
    if m: key, order = f"fkl_batch morphologyDPPKernel, N = {int(m.group(1))}", int(m.group(1))
    elif "linearFilterDPPKernel" in name: key, order = "fkl_loop linearFilterDPPKernel (the same kernel at every N)", 1000
    elif "gpuSpin" in name: key, order = "gpuSpin", 1001
    else: key, order = name, 1002
    size = int(re.search(r"EIATTR_CBANK_PARAM_SIZE\s*\n\s*Format:[^\n]*\n\s*Value:\s*(0x[0-9a-f]+)", sec).group(1), 16)
    params = [(int(o, 16), int(s, 16)) for o, s in re.findall(r"Ordinal : (0x[0-9a-f]+)\s+Offset\s*: 0x[0-9a-f]+\s+Size\s*: (0x[0-9a-f]+)", sec)]
    rows.append((order, key, size, sorted(params)))
for order, key, size, params in sorted(rows):
    out.append(f"{key}: {size} bytes of parameters ({', '.join(f'parameter {o}: {s}' for o, s in params)})")
batch = sorted((o, s) for o, k, s, p in rows if o < 1000)
if len(batch) > 1:
    (n0, s0), (n1, s1) = batch[0], batch[-1]
    step = (s1 - s0) / (n1 - n0)
    fits = all(s == s0 + step * (n - n0) for n, s in batch)
    out.append(f"fkl_batch: {s0} bytes at N = {n0}, {s1} at N = {n1}: {step:g} bytes per crop{' at every N' if fits else ' (NOT linear at every N)'}; "
               f"{s0 - step * n0:g} + {step:g} N")
t = "\n".join(out); print(t)
if args.out: open(args.out, "w").write(t + "\n")
