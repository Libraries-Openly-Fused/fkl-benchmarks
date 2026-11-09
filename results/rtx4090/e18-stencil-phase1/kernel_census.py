#!/usr/bin/env python3
"""Per-kernel census of how the kernel parameters are read, from the SASS of existing binaries (no program is run).

  param_census.py --tools <dir of sass_tools.py> --out <dir> <binary>...

For every kernel of every binary (cuobjdump -sass, sm_89, CUDA 13.4 tools in PATH):
  instructions        normalised instruction lines (sass_tools.classify, the E1-SASS count)
  c0_operand_instr    instructions with an immediate operand in constant bank 0, c[0x0][0x...] (the parameter bank)
  c0_ldc              LDC instructions that load from bank 0 with a register index, c[0x0][R..+0x...]
  other_bank_instr    instructions with an operand in another constant bank (c[0x3], ...)
  LDL, STL            local-memory loads and stores
and from cuobjdump -res-usage: REG, STACK, LOCAL, CONSTANT[0] (bytes of bank 0: the parameters plus the driver's area).
Writes <out>/<binary>.sass (the full dump) and <out>/census.tsv; prints one summary line per binary.
"""
import argparse
import os
import re
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--tools", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--arch", default="sm_89"); ap.add_argument("binaries", nargs="+")
args = ap.parse_args()
sys.path.insert(0, args.tools)
import sass_tools as st  # noqa: E402

C0_IMM = re.compile(r"\bc\[0x0\]\[0x[0-9a-f]+\]")
C0_IDX = re.compile(r"\bc\[0x0\]\[U?R\w+")
OTHER = re.compile(r"\bc\[0x([1-9a-f][0-9a-f]*)\]\[")
RES = re.compile(r"Function (.+?):\s*\n\s*(REG:[^\n]*)", re.S)


def field(line, key):
    m = re.search(re.escape(key) + r":(\d+)", line)
    return m.group(1) if m else "0"

os.makedirs(args.out, exist_ok=True)
rows = []
for b in args.binaries:
    name = os.path.basename(b)
    sass = subprocess.run(["cuobjdump", "-arch", args.arch, "-sass", b], capture_output=True, text=True, check=True).stdout
    with open(os.path.join(args.out, name + ".sass"), "w") as f:
        f.write(sass)
    res = subprocess.run(["cuobjdump", "-res-usage", b], capture_output=True, text=True, check=True).stdout
    usage = {re.sub(r"\s+", "", m.group(1)): tuple(field(m.group(2), k) for k in ("REG", "STACK", "SHARED", "LOCAL", "CONSTANT[0]"))
             for m in RES.finditer(res)}  # long names wrap over lines; fields in any order (CONSTANT[2] may come first)
    funcs = st.split_functions(sass)
    mangled = list(funcs)
    dem = subprocess.run(["c++filt"], input="\n".join(mangled), capture_output=True, text=True).stdout.splitlines()
    for mg, dm in zip(mangled, dem):
        lines = st.normalise_sass(funcs[mg], rename=False)
        instr, hist = st.classify(lines)
        body = [l for l in lines if not l.endswith(":")]
        c0imm = sum(1 for l in body if C0_IMM.search(l))
        c0ldc = sum(1 for l in body if st.opcode(l) == "LDC" and C0_IDX.search(l))
        other = sum(1 for l in body if OTHER.search(l))
        reg, stack, shared, local, cmem0 = usage.get(mg, ("?",) * 5)
        short = dm.replace("(anonymous namespace)::", "").split("(")[0][:90]
        rows.append((name, short, instr, c0imm, c0ldc, other, hist.get("LDL", 0), hist.get("STL", 0), reg, stack, local, cmem0, mg))
    sel = [r for r in rows if r[0] == name]
    print(f"{name}: {len(sel)} kernels; LDL total {sum(r[6] for r in sel)}, STL total {sum(r[7] for r in sel)}; "
          f"STACK>0 in {sum(1 for r in sel if r[9] not in ('?', '0'))}, LOCAL>0 in {sum(1 for r in sel if r[10] not in ('?', '0'))}; "
          f"instructions with a c[0x0] operand: min {min(r[3] for r in sel)} max {max(r[3] for r in sel)}; "
          f"indexed LDC from c[0x0]: kernels with any {sum(1 for r in sel if r[4])}; CONSTANT[0] bytes min {min(int(r[11]) for r in sel)} "
          f"max {max(int(r[11]) for r in sel)}")
with open(os.path.join(args.out, "census.tsv"), "w") as f:
    f.write("binary\tkernel (demangled, cut at the argument list, 90 chars)\tinstructions\tc0_operand_instr\tc0_ldc\tother_bank_instr\t"
            "LDL\tSTL\tREG\tSTACK\tLOCAL\tCONSTANT0\tmangled\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
