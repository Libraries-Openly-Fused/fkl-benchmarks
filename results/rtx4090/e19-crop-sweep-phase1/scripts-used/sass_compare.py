#!/usr/bin/env python3
"""The SASS of e19_crop_sweep_cu's kernels against e19_batched_box_cu's (cuobjdump -sass of both; nothing is run).

  sass_compare.py <old.sass> <new.sass> [--out report.txt]

Per function: its name (gpuSpin's anonymous-namespace prefix, which holds a hash of the file, is dropped; fkl_batch's kernel is named by its N), its
instruction count, and whether its instructions (the text without the /*address*/ comments and the encodings) equal the old binary's
function of the same name. Every new kernel is listed; for each old one, the new one of the same name.
"""
import argparse, re
ap = argparse.ArgumentParser(); ap.add_argument("old"); ap.add_argument("new"); ap.add_argument("--out", default=None); args = ap.parse_args()

def functions(path):
    text = open(path).read()
    out = {}
    for block in re.split(r"\n\s*Function : ", text)[1:]:
        name, body = block.split("\n", 1)
        name = re.sub(r"_ZN\d+_GLOBAL__N__[0-9a-f]+_\d+_[A-Za-z0-9_]+_cu_[0-9a-f]+", "<anon>", name.strip())
        ins = []
        for line in body.split("\n"):
            m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
            if m: ins.append(m.group(1))
        out[name] = ins
    return out

def label(name):
    m = re.search(r"PlanePolicyE0ELm(\d+)E", name)
    if m: return f"morphologyDPPKernel (fkl_batch), N = {m.group(1)}", int(m.group(1))
    if "linearFilterDPPKernel" in name: return "linearFilterDPPKernel (fkl_loop)", 1000
    if "gpuSpin" in name: return "gpuSpin", 1001
    return name[:80], 1002

old, new = functions(args.old), functions(args.new)
lines = []
for name in sorted(new, key=lambda n: label(n)[1]):
    lab = label(name)[0]
    if name in old:
        same = old[name] == new[name]
        lines.append(f"{lab}: {len(new[name])} instructions; the old binary's function of the same name: {len(old[name])} instructions; "
                     f"{'IDENTICAL' if same else 'DIFFERENT'}")
    else:
        lines.append(f"{lab}: {len(new[name])} instructions; not in the old binary")
for name in old:
    if name not in new: lines.append(f"old {label(name)[0]}: MISSING in the new binary")
t = "\n".join(lines); print(t)
if args.out: open(args.out, "w").write(t + "\n")
