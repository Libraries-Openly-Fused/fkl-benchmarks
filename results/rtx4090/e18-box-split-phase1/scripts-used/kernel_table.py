#!/usr/bin/env python3
"""The kernels of e18_box_split_cu from cuobjdump (no program is run), and their SASS against a reference binary's SASS dump.

  kernel_table.py --tools <dir of sass_tools.py> --binary <e18_box_split_cu> --sass <its cuobjdump -sass dump>
                  --ref-sass <the reference binary's cuobjdump -sass dump> --ref-name <label> --out <tsv>

For every kernel of the binary: the row of e18_box_split.cu that launches it (from its template arguments), its SASS instruction count
(sass_tools.classify, the E1-SASS count), LDL / STL, and from cuobjdump -res-usage REG, STACK, SHARED, LOCAL and CONSTANT[0]. Then its
normalised SASS (sass_tools.normalise_sass without register renaming: every instruction, operand and label as cuobjdump prints it, with
the addresses, encodings and column padding removed) against the kernel of the same mangled name in the reference dump (gpuSpin: of the
same demangled name, since its anonymous namespace carries the name of its source file): identical, different, or absent there.
"""
import argparse
import re
import subprocess
import sys

ap = argparse.ArgumentParser()
for a in ("--tools", "--binary", "--sass", "--ref-sass", "--ref-name", "--out"):
    ap.add_argument(a, required=True)
args = ap.parse_args()
sys.path.insert(0, args.tools)
import sass_tools as st  # noqa: E402

RES = re.compile(r"Function (.+?):\s*\n\s*(REG:[^\n]*)", re.S)


def field(line, key):
    m = re.search(re.escape(key) + r":(\d+)", line)
    return m.group(1) if m else "?"


def demangle(names):
    return subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True).stdout.splitlines()


def label(dm):
    """The row of e18_box_split.cu that launches the kernel, from its demangled name."""
    if dm.startswith("(anonymous namespace)::gpuSpin") or dm.startswith("gpuSpin"):
        return "gpuSpin (the spin of kernel_only)"
    if "launchTransformDPP_Kernel" in dm:
        return "fkl_resize"
    if "linearFilterDPPKernel" in dm:
        resize = "ResizeComplete" in dm
        sampling = "HALF_PIXEL" if "(fk::ResizeSampling)1" in dm else "ASYMMETRIC"
        identity = "PerThreadRead<(fk::ND)2, float3>" in dm
        div = "fk::Div<float3" in dm
        if identity:
            return f"read chain check, {sampling} (identity weights)"
        if not resize:
            return "fkl_box" if div else "box alone, rounding"
        if div:
            return "fkl_trunc_halfpixel" if sampling == "HALF_PIXEL" else "fkl_trunc_asym"
        return "fkl_box_asym" if sampling == "ASYMMETRIC" else "fkl_box_halfpixel"
    return "?"


res = subprocess.run(["cuobjdump", "-res-usage", args.binary], capture_output=True, text=True, check=True).stdout
usage = {re.sub(r"\s+", "", m.group(1)): m.group(2) for m in RES.finditer(res)}
funcs = st.split_functions(open(args.sass).read())
ref = st.split_functions(open(args.ref_sass).read())
mangled = list(funcs)
dem = dict(zip(mangled, demangle(mangled)))
refDem = dict(zip(list(ref), demangle(list(ref))))
refByDem = {d.replace("(anonymous namespace)::", ""): m for m, d in refDem.items()}
rows = []
for mg in mangled:
    dm = dem[mg]
    lines = st.normalise_sass(funcs[mg], rename=False)
    instr, hist = st.classify(lines)
    u = usage.get(mg, "")
    refMg = mg if mg in ref else refByDem.get(dm.replace("(anonymous namespace)::", ""))
    if refMg is None:
        cmp = f"absent from {args.ref_name}"
    else:
        same = st.normalise_sass(ref[refMg], rename=False) == lines
        cmp = f"identical to {args.ref_name}" if same else f"DIFFERENT from {args.ref_name}"
    rows.append((label(dm), instr, hist.get("LDL", 0), hist.get("STL", 0), field(u, "REG"), field(u, "STACK"), field(u, "SHARED"),
                 field(u, "LOCAL"), field(u, "CONSTANT[0]"), cmp, mg))
with open(args.out, "w") as f:
    f.write("row\tSASS_instructions\tLDL\tSTL\tREG\tSTACK\tSHARED_bytes\tLOCAL_bytes\tCONSTANT0_bytes\tSASS_against_reference\tmangled\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
for r in rows:
    print("\t".join(str(x) for x in r[:10]))
