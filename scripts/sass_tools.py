#!/usr/bin/env python3
# Copyright 2026 Oscar Amoros Huguet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SASS / PTX tooling for experiment E1-SASS (hand-written kernel vs FKL-generated kernel).

  sass_tools.py dump <binary|cubin> <out_dir> [--arch sm_89] [--rename]
      cuobjdump -sass (and -ptx) of every kernel of the file, normalised, one file per kernel in <out_dir>,
      plus <out_dir>/_toolchain.txt with the tools that produced it (nvcc, cuobjdump, driver, GPU).
  sass_tools.py hist <kernel.sass>...
      instruction count and the per-class histogram of normalised SASS files.
  sass_tools.py diff <a.sass> <b.sass> [--context N]
      the two histograms side by side and the unified diff of the normalised instruction streams.
  sass_tools.py ptx <file.ptx> [--rename]
      normalised PTX on stdout (comments, .loc and .file lines removed; registers renamed with --rename).

Normalisation of SASS: the address of every instruction (/*0010*/) and its encoding (/* 0x... */) are removed, so two
kernels with the same instructions compare equal wherever they sit; with --rename the registers (R, UR, P, UP) are numbered
by first appearance, so that two kernels that only differ in the register allocation compare equal. Constant-bank operands
(c[0x0][0x...]) are kept: a parameter at another offset is a real difference and the report says so.

Run it with the tools of CUDA 13.4 in PATH (source scripts/env-cuda13.4.sh): the tool records which cuobjdump it used.
"""
import argparse
import collections
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys

ADDR = re.compile(r"^\s*/\*[0-9a-fA-F]{4,}\*/\s*")          # /*0010*/ at the start of an instruction
ENC = re.compile(r"\s*/\*\s*0x[0-9a-fA-F]+\s*\*/\s*$")       # /* 0x... */ at the end
ENC_ONLY = re.compile(r"^\s*/\*\s*0x[0-9a-fA-F]+\s*\*/\s*$")  # the second encoding line of an instruction
FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)")
REG = re.compile(r"\b(UR|UP|R|P)(\d+)\b")
PRED_PREFIX = re.compile(r"^\s*@!?U?P\w+\s+")

# Instruction classes of the E1-SASS table. The first match wins, so the order matters.
CLASSES = [
    ("FFMA", re.compile(r"^(FFMA|FFMA2)\b")),
    ("FMUL", re.compile(r"^(FMUL|FMUL2)\b")),
    ("FADD", re.compile(r"^(FADD|FADD2)\b")),
    ("other FP32/FP16 ALU", re.compile(r"^(FMNMX|FSEL|FSETP|FCHK|MUFU|F2F|F2I|I2F|I2FP|F2FP|FRND|HFMA2|HADD2|HMUL2|HSETP2|HMNMX2)\b")),
    ("tensor core (HMMA/IMMA/QMMA/OMMA)", re.compile(r"^(HMMA|IMMA|QMMA|OMMA|DMMA|BMMA)\b")),
    ("LDSM", re.compile(r"^LDSM\b")),
    ("LDGSTS (cp.async)", re.compile(r"^LDGSTS\b")),
    ("LDG", re.compile(r"^LDG\b")),
    ("STG", re.compile(r"^STG\b")),
    ("LDL", re.compile(r"^LDL\b")),
    ("STL", re.compile(r"^STL\b")),
    ("LDS", re.compile(r"^LDS\b")),
    ("STS", re.compile(r"^STS\b")),
    ("LDC (constant load)", re.compile(r"^LDC\b")),
    ("IMAD", re.compile(r"^IMAD\b")),
    ("ISETP", re.compile(r"^ISETP\b")),
    ("BRA/BSSY/BSYNC/CALL/RET/EXIT", re.compile(r"^(BRA|BRX|BSSY|BSYNC|CALL|RET|EXIT|JMP|BREAK|WARPSYNC|YIELD|NANOSLEEP)\b")),
    ("barriers (BAR, MEMBAR, DEPBAR, ERRBAR)", re.compile(r"^(BAR|MEMBAR|DEPBAR|ERRBAR|FENCE|SYNCS)\b")),
    ("other integer ALU", re.compile(r"^(IADD3|IADD|LEA|LOP3|SHF|SHL|SHR|IABS|ISCADD|IMNMX|SEL|PRMT|POPC|FLO|BFE|BFI|VIADD|VIMNMX|UIADD3|ULOP3|UMOV|USHF|ULEA|UIMAD|UISETP|USEL|UPRMT)\b")),
    ("moves and special registers", re.compile(r"^(MOV|S2R|S2UR|CS2R|R2UR|SHFL|VOTE|MATCH|P2R|R2P|PLOP3|UP2UR|UR2UP|CCTL|ELECT|REDUX|NOP)\b")),
]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)


def tool_version(tool):
    path = shutil.which(tool)
    if not path:
        return f"{tool}: not found"
    out = run([path, "--version"]).stdout.strip().splitlines()
    return f"{tool}: {path}  ({out[-1].strip() if out else '?'})"


def normalise_sass(text, rename):
    """text = cuobjdump -sass output of one function. Returns the list of normalised instruction lines."""
    out = []
    names = {}

    def rename_reg(m):
        key = (m.group(1), m.group(2))
        if key not in names:
            counter = sum(1 for k in names if k[0] == m.group(1))
            names[key] = counter
        return f"{m.group(1)}{names[key]}"

    for line in text.splitlines():
        if FUNC.match(line) or line.strip().startswith((".headerflags", ".section", ".sectionflags", ".sectioninfo", ".align", ".type", ".size", ".weak", ".global", ".text", "code for", "Fatbin", "arch =", "identifier")):
            continue
        if ENC_ONLY.match(line) or not line.strip():
            continue
        line = ADDR.sub("", line)
        line = ENC.sub("", line)
        line = line.strip().rstrip(";").strip()
        line = re.sub(r"\s+", " ", line)
        if not line or line.startswith((".", "//")):
            # labels (".L_x_0:") are kept: they carry the control flow; directives are not instructions
            if line.endswith(":"):
                out.append(line)
            continue
        if rename:
            line = REG.sub(rename_reg, line)
        out.append(line)
    return out


def opcode(line):
    """The opcode of a normalised line: the first word after the optional predicate, without modifiers (FFMA.FTZ -> FFMA)."""
    if line.endswith(":"):
        return None
    body = PRED_PREFIX.sub("", line)
    return body.split(" ")[0].split(".")[0]


def classify(lines):
    hist = collections.Counter()
    instr = 0
    for line in lines:
        op = opcode(line)
        if op is None:
            continue
        instr += 1
        for name, rx in CLASSES:
            if rx.match(op):
                hist[name] += 1
                break
        else:
            hist["other"] += 1
        if re.search(r"\bc\[0x[0-9a-f]+\]\[0x[0-9a-f]+\]", line):
            hist["instructions with a constant-bank operand c[bank][off]"] += 1
    return instr, hist


def split_functions(text):
    funcs = collections.OrderedDict()
    cur = None
    for line in text.splitlines():
        m = FUNC.match(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
        elif cur is not None:
            funcs[cur].append(line)
    return {k: "\n".join(v) for k, v in funcs.items()}


def kernel_file(index, name):
    """Mangled template names are long and share long prefixes: number the file and add a hash of the full name."""
    short = re.sub(r"[^A-Za-z0-9_]", "_", name)[:60]
    return f"k{index:03d}_{hashlib.sha1(name.encode()).hexdigest()[:8]}_{short}.sass"


def cmd_dump(args):
    os.makedirs(args.out_dir, exist_ok=True)
    base = [shutil.which("cuobjdump") or "cuobjdump"]
    arch = ["-arch", args.arch] if args.arch else []
    res = run(base + arch + ["-sass", args.binary])
    if res.returncode != 0 or "Function" not in res.stdout:
        sys.exit(f"cuobjdump -sass failed on {args.binary}: {res.stderr.strip() or res.stdout[:200]}")
    funcs = split_functions(res.stdout)
    index = []
    for idx, (name, body) in enumerate(funcs.items()):
        lines = normalise_sass(body, args.rename)
        fname = kernel_file(idx, name)
        with open(os.path.join(args.out_dir, fname), "w") as f:
            f.write(f"# kernel: {name}\n" + "\n".join(lines) + "\n")
        n, _ = classify(lines)
        index.append((name, fname, n))
    ptx = run(base + arch + ["-ptx", args.binary])
    if ptx.returncode == 0 and ptx.stdout.strip():
        with open(os.path.join(args.out_dir, "_all.ptx"), "w") as f:
            f.write(ptx.stdout)
    smi = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).stdout.strip().replace("\n", " | ")
    with open(os.path.join(args.out_dir, "_toolchain.txt"), "w") as f:
        f.write(f"input: {os.path.abspath(args.binary)}\narch: {args.arch or 'all'}  rename-registers: {args.rename}\n")
        f.write(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', '')}\n")
        f.write(tool_version("nvcc") + "\n" + tool_version("cuobjdump") + "\n" + f"GPU, driver: {smi}\n")
    with open(os.path.join(args.out_dir, "_index.tsv"), "w") as f:
        f.write("kernel\tfile\tinstructions\n")
        for name, fname, n in index:
            f.write(f"{name}\t{fname}\t{n}\n")
    print(f"{len(index)} kernels of {args.binary} -> {args.out_dir}")


def read_lines(path):
    with open(path) as f:
        return [l.rstrip("\n") for l in f if not l.startswith("#")]


def print_hist(title, instr, hist):
    print(f"{title}: {instr} instructions")
    for name, count in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {count:6d}  {name}")


def cmd_hist(args):
    for p in args.files:
        instr, hist = classify(read_lines(p))
        print_hist(os.path.basename(p), instr, hist)


def cmd_diff(args):
    a, b = read_lines(args.a), read_lines(args.b)
    ia, ha = classify(a)
    ib, hb = classify(b)
    print(f"A = {args.a}\nB = {args.b}\n")
    print(f"{'class':58s} {'A':>7s} {'B':>7s}")
    for name in sorted(set(ha) | set(hb), key=lambda n: (-(ha.get(n, 0) + hb.get(n, 0)), n)):
        mark = "" if ha.get(name, 0) == hb.get(name, 0) else "   <-- differs"
        print(f"{name:58s} {ha.get(name, 0):7d} {hb.get(name, 0):7d}{mark}")
    print(f"{'TOTAL instructions':58s} {ia:7d} {ib:7d}{'' if ia == ib else '   <-- differs'}\n")
    if a == b:
        print("normalised instruction streams: IDENTICAL")
        return 0
    print("normalised instruction streams: DIFFER")
    for l in difflib.unified_diff(a, b, "A", "B", n=args.context, lineterm=""):
        print(l)
    return 1


def cmd_ptx(args):
    names = {}

    def rn(m):
        key = m.group(0)
        if key not in names:
            names[key] = f"%{m.group(1)}{sum(1 for k in names if re.match(r'%' + m.group(1) + r'\d', k))}"
        return names[key]

    with open(args.file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("//", ".file", ".loc")):
                continue
            line = re.sub(r"//.*$", "", line).rstrip()
            if args.rename:
                line = re.sub(r"%([a-zA-Z]+)\d+", rn, line)
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("binary"); d.add_argument("out_dir")
    d.add_argument("--arch", default=None); d.add_argument("--rename", action="store_true"); d.set_defaults(fn=cmd_dump)
    h = sub.add_parser("hist"); h.add_argument("files", nargs="+"); h.set_defaults(fn=cmd_hist)
    f = sub.add_parser("diff"); f.add_argument("a"); f.add_argument("b"); f.add_argument("--context", type=int, default=3)
    f.set_defaults(fn=cmd_diff)
    p = sub.add_parser("ptx"); p.add_argument("file"); p.add_argument("--rename", action="store_true"); p.set_defaults(fn=cmd_ptx)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
