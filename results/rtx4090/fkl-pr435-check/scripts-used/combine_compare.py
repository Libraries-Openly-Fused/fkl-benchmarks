#!/usr/bin/env python3
"""Task C-NV: the split-KV combine kernel at b10fc66 (flashFwdSplitCombineKernel<float, D, true>) against #435's
launchDPP_Kernel<FlashAttentionSplitCombineDPP<GPU_NVIDIA, float, D, true, AttentionIdentityEpilogue>, ...> (CPU only).
For each D: the normalised SASS with the registers renamed in order of appearance (sass_tools.normalise_sass(rename=True)),
branch targets and the trailing NOP padding removed; then the lines that differ, and whether the sequences of the
floating-point instructions (opcode F*, MUFU) and of the global and constant memory accesses (LDG, STG, LDC, ULDC and the
c[0x0][...] parameter reads) are the same, operands included.

  combine_compare.py --tools <dir of sass_tools.py> <base cuda file> <head cuda file>
"""
import difflib, re, subprocess, sys
tools, base, head = sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, tools)
import sass_tools as st


def kernels(path):
    f = st.split_functions(subprocess.run(["cuobjdump", "-arch", "sm_89", "-sass", path], capture_output=True, text=True, check=True).stdout)
    names = list(f)
    dem = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return {d: f[m] for m, d in zip(names, dem)}


def norm(body):
    out = []
    for l in st.normalise_sass(body, rename=True):
        l = re.sub(r"\b(BRA|BSSY|CALL\.REL\.NOINC|MOV R\d+,) 0x[0-9a-f]+", r"\1 <target>", l)
        out.append(l)
    while out and out[-1] == "NOP":
        out.pop()
    return out


FP = re.compile(r"^(@!?U?P\w+ )?(F\w+|MUFU)\b")
MEM = re.compile(r"^(@!?U?P\w+ )?(LDG|STG|LDC|ULDC)\b|c\[0x0\]")
kb, kh = kernels(base), kernels(head)
for D in (64, 128):
    a = norm(kb[f"void fk::flashFwdSplitCombineKernel<float, {D}, true>(float const*, float*, int, int)"])
    hn = [n for n in kh if n.startswith(f"void fk::launchDPP_Kernel<fk::FlashAttentionSplitCombineDPP<(fk::ParArch)1, float, {D}, true, fk::AttentionIdentityEpilogue>")]
    assert len(hn) == 1, hn
    b = norm(kh[hn[0]])
    print(f"D = {D}: {len(a)} / {len(b)} lines (registers renamed, branch targets and the trailing NOPs removed)")
    d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if not l.startswith(("---", "+++"))]
    print("  the lines that differ (- base, + head):")
    for l in d:
        print("    " + l)
    fa, fb = [l for l in a if FP.search(l)], [l for l in b if FP.search(l)]
    ma, mb = [l for l in a if MEM.search(l)], [l for l in b if MEM.search(l)]
    print(f"  floating-point instructions: {len(fa)} / {len(fb)}, same sequence with the same operands: {fa == fb}")
    print(f"  memory accesses and parameter reads: {len(ma)} / {len(mb)}, same sequence with the same operands: {ma == mb}")
