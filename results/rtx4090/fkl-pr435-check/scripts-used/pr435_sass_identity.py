#!/usr/bin/env python3
"""Task C-NV: the sm_89 device code of FKL's CUDA tests built at b10fc66 (A, the base) and at e322cad (B, the head of #435),
kernel by kernel, with the kernel names paired by #435's rename rule (CPU only: cuobjdump, nothing is run).

  pr435_sass_identity.py --tools <dir of sass_tools.py> --a <tag>=<dir> --b <tag>=<dir> [--glob '*_cu'] [--out report.txt] [--csv kernels.csv]

Files: every executable named --glob (default *_cu) and every shared library lib<glob>.so under each directory (CMakeFiles skipped),
paired by file name.
SASS: cuobjdump -arch sm_89 -sass, split per kernel and normalised by sass_tools.normalise_sass (the /*address*/ comments and the
encodings removed, registers NOT renamed), as in ../task-b-sm89-check/ and ../fkl-pr433-pin-check/. Resources: cuobjdump -res-usage
(registers, stack, shared memory, local memory, constant bank 0, which holds the kernel parameters). Per kernel also the instruction count
(the normalised lines that are not labels), the LDL and STL counts (local memory loads and stores) and, for a pair that differs, the
opcodes whose counts differ.

Pairing. A kernel's key is its demangled name with the whitespace removed (the anonymous namespace demangles as "(anonymous namespace)",
without nvcc's module id). B's keys are used as they are. A's names are rewritten into the names #435 gives the same kernels, and nothing
else is changed:
  1. "(fk::ParArch)1," is inserted after every "fk::SoftmaxDPP<", "fk::FlashAttentionDPP<", "fk::FlashAttentionMmaDPP<" and
     "fk::FlashDecodeDPP<" (the attention DPPs take a leading ParArch; only GPU_NVIDIA = 1 exists), so their Params types match.
  2. The old kernel templates become launchDPP_Kernel<DPP, DPPDetails, IOps...>(DPPDetails, IOps...):
     launchTransformDPP_Kernel<PA, TFEN, TD, D, IOps...>                -> <fk::TransformDPP<PA, TFEN, D, TD, void>, D, IOps...>
     launchDivergentBatchTransformDPP_Kernel<PA, SS, D, Seqs...>       -> <fk::DivergentBatchTransformDPP<PA, SS>, D, Seqs...>
     linearFilterDPPKernel<PA, D, R, M, A, W>                          -> <fk::LinearFilterDPP<PA, D>, D, R, M, A, W>
     medianFilterDPPKernel<PA, D, R, W, S>                             -> <fk::MedianFilterDPP<PA, D>, D, R, W, S>
     morphologyDPPKernel<PA, D, I, O, Red>                             -> <fk::MorphologyDPP<PA, D>, D, I, O, Red>
     launch{BoxFilter,Conv,Median,Morph}QuadDPP_Kernel<DPP, IOps...>   -> <DPP, fk::{BoxFilter,Conv,Median,Morph}QuadDetails, IOps...>
     launchSoftmaxDPP_Kernel<In, OT, BS>                               -> <S, S::Params>, S = fk::SoftmaxDPP<(fk::ParArch)1, In, OT, BS>
     launchFlashAttentionDPP_Kernel<OT, D, Q, K, V, E, BN, W>          -> <F, F::Params>, F = fk::FlashAttentionDPP<(fk::ParArch)1, ...>
     launchFlashAttentionMmaDPP_Kernel<OT, ..., SM, false>             -> <M, M::Params>, M = fk::FlashAttentionMmaDPP<(fk::ParArch)1, OT, ..., SM>
     launchFlashAttentionMmaDPP_Kernel<OT, ..., SM, true>              -> <fk::FlashAttentionMmaSplitKVDPP<(fk::ParArch)1, OT, ..., SM>, M::Params>
     launchFlashDecodeDPP_Kernel<OT, D, Q, K, V, NW>                   -> <X, X::Params>, X = fk::FlashDecodeDPP<(fk::ParArch)1, ...>
     flashFwdSplitCombineKernel<OT, D, LOG2D>(float const*, OT*, int, int)
                                -> <C, C::Params>, C = fk::FlashAttentionSplitCombineDPP<(fk::ParArch)1, OT, D, LOG2D, fk::AttentionIdentityEpilogue>
                                   (the base's combine had no epilogue: its counterpart is the combine with the identity epilogue)
     The function parameters become the template arguments after the DPP.
A key found twice in one file is an error. Exit 0 when every file is on both sides and every kernel pairs and is identical.
"""
import argparse
import collections
import csv
import fnmatch
import hashlib
import os
import re
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--tools", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--arch", default="sm_89"); ap.add_argument("--glob", default="*_cu"); ap.add_argument("--out", default=None); ap.add_argument("--csv", default=None)
args = ap.parse_args()
sys.path.insert(0, args.tools)
import sass_tools as st  # noqa: E402

FUNC_RES = re.compile(r"^\s*Function (\S+):")
RES_NAMES = ("REG", "STACK", "SHARED", "LOCAL", r"CONSTANT\[0\]")
RES = {n: re.compile(n + r":(\d+)") for n in RES_NAMES}
OPEN, CLOSE = "<({[", ">)}]"
PA_DPPS = ("fk::SoftmaxDPP<", "fk::FlashAttentionDPP<", "fk::FlashAttentionMmaDPP<", "fk::FlashDecodeDPP<")
QUADS = {"launchBoxFilterQuadDPP_Kernel": "fk::BoxFilterQuadDetails", "launchConvQuadDPP_Kernel": "fk::ConvQuadDetails",
         "launchMedianQuadDPP_Kernel": "fk::MedianQuadDetails", "launchMorphQuadDPP_Kernel": "fk::MorphQuadDetails"}
KERNEL = re.compile(r"^voidfk::(\w+)<")
RENAMED = {"launchTransformDPP_Kernel", "launchDivergentBatchTransformDPP_Kernel", "linearFilterDPPKernel", "medianFilterDPPKernel",
           "morphologyDPPKernel", *QUADS, "launchSoftmaxDPP_Kernel", "launchFlashAttentionDPP_Kernel",
           "launchFlashAttentionMmaDPP_Kernel", "launchFlashDecodeDPP_Kernel", "flashFwdSplitCombineKernel"}


def split_args(s, start):
    """s[start] is just after '<': returns (list of the top-level arguments, index of the matching '>')."""
    depth, begin, out, i = 0, start, [], start
    while i < len(s):
        c = s[i]
        if c in OPEN:
            depth += 1
        elif c in CLOSE:
            if depth == 0:
                out.append(s[begin:i])
                return out, i
            depth -= 1
        elif c == "," and depth == 0:
            out.append(s[begin:i])
            begin = i + 1
        i += 1
    raise ValueError(f"unbalanced template argument list: {s[:300]}")


def insert_pa(s):
    for t in PA_DPPS:
        s = re.sub(r"(?<![\w:])" + re.escape(t), t + "(fk::ParArch)1,", s)
    return s


def rewrite(key):
    """A's key -> the key #435 gives the same kernel; (new key, rule) where rule is '' when the name is not one #435 renames."""
    s = insert_pa(key)
    m = KERNEL.match(s)
    if not m or m.group(1) not in RENAMED:
        return s, ("ParArch inserted" if s != key else "")
    name = m.group(1)
    a, close = split_args(s, m.end())
    rest = s[close + 1:]
    if not (rest.startswith("(") and rest.endswith(")")):
        raise ValueError(f"unexpected kernel name: {key[:300]}")
    if name == "launchTransformDPP_Kernel":
        new = [f"fk::TransformDPP<{a[0]},{a[1]},{a[3]},{a[2]},void>", *a[3:]]
    elif name == "launchDivergentBatchTransformDPP_Kernel":
        new = [f"fk::DivergentBatchTransformDPP<{a[0]},{a[1]}>", *a[2:]]
    elif name == "linearFilterDPPKernel":
        new = [f"fk::LinearFilterDPP<{a[0]},{a[1]}>", *a[1:]]
    elif name == "medianFilterDPPKernel":
        new = [f"fk::MedianFilterDPP<{a[0]},{a[1]}>", *a[1:]]
    elif name == "morphologyDPPKernel":
        new = [f"fk::MorphologyDPP<{a[0]},{a[1]}>", *a[1:]]
    elif name in QUADS:
        new = [a[0], QUADS[name], *a[1:]]
    elif name == "launchSoftmaxDPP_Kernel":
        d = f"fk::SoftmaxDPP<(fk::ParArch)1,{','.join(a)}>"
        new = [d, d + "::Params"]
    elif name == "launchFlashAttentionDPP_Kernel":
        d = f"fk::FlashAttentionDPP<(fk::ParArch)1,{','.join(a)}>"
        new = [d, d + "::Params"]
    elif name == "launchFlashAttentionMmaDPP_Kernel":
        if len(a) != 11 or a[10] not in ("true", "false"):
            raise ValueError(f"unexpected Mma kernel: {key[:300]}")
        mma = f"fk::FlashAttentionMmaDPP<(fk::ParArch)1,{','.join(a[:10])}>"
        d = f"fk::FlashAttentionMmaSplitKVDPP<(fk::ParArch)1,{','.join(a[:10])}>" if a[10] == "true" else mma
        new = [d, mma + "::Params"]
    elif name == "launchFlashDecodeDPP_Kernel":
        d = f"fk::FlashDecodeDPP<(fk::ParArch)1,{','.join(a)}>"
        new = [d, d + "::Params"]
    elif name == "flashFwdSplitCombineKernel":
        d = f"fk::FlashAttentionSplitCombineDPP<(fk::ParArch)1,{','.join(a)},fk::AttentionIdentityEpilogue>"
        new = [d, d + "::Params"]
    return f"voidfk::launchDPP_Kernel<{','.join(new)}>({','.join(new[1:])})", name


def files(root):
    found = {}
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("CMakeFiles", ".cmake")]
        for f in fs:
            p = os.path.join(d, f)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            if (fnmatch.fnmatch(f, args.glob) and os.access(p, os.X_OK)) or fnmatch.fnmatch(f, "lib" + args.glob + ".so"):
                if f in found:
                    sys.exit(f"{f} is found twice under {root}: {found[f]} and {p}")
                found[f] = p
    return found


def cuobjdump(what, path):
    r = subprocess.run(["cuobjdump", "-arch", args.arch, what, path], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"cuobjdump {what} failed on {path}: {r.stderr.strip()[:300]}")
    return r.stdout


LLVM_CXXFILT = "/usr/lib/llvm-21/bin/llvm-cxxfilt"
FALLBACKS = []


def demangle(names):
    """GNU c++filt; a name it returns unchanged (it gives up on very long names) is demangled by llvm-cxxfilt instead (listed)."""
    d = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    if len(d) != len(names):
        sys.exit("c++filt returned a different number of lines")
    out = dict(zip(names, d))
    for n in names:
        if out[n] == n and n.startswith("_Z"):
            r = subprocess.run([LLVM_CXXFILT], input=n, capture_output=True, text=True).stdout.strip()
            if r == n or not r:
                sys.exit(f"no demangler demangles {n[:200]}")
            out[n] = r
            FALLBACKS.append(n)
    return out


def kernels(path, side):
    res, cur = {}, None
    for line in cuobjdump("-res-usage", path).splitlines():
        m = FUNC_RES.match(line)
        if m:
            cur = m.group(1); continue
        if cur and "REG:" in line:
            res[cur] = tuple(int(RES[n].search(line).group(1)) if RES[n].search(line) else 0 for n in RES_NAMES)
            cur = None
    funcs = st.split_functions(cuobjdump("-sass", path))
    dem = demangle(list(funcs))
    out = {}
    for mangled, body in funcs.items():
        lines = st.normalise_sass(body, rename=False)
        instr = [l for l in lines if not l.endswith(":")]
        ops = collections.Counter(st.opcode(l) for l in instr)
        key0 = re.sub(r"\s+", "", dem[mangled])
        key, rule = rewrite(key0) if side == "A" else (key0, "")
        if key in out:
            sys.exit(f"{path}: two kernels with the key {key[:300]}")
        out[key] = dict(mangled=mangled, demangled=dem[mangled], rule=rule, lines=lines, instr=len(instr), ops=ops,
                        ldl=ops.get("LDL", 0), stl=ops.get("STL", 0), res=res.get(mangled, (-1,) * 5),
                        sha=hashlib.sha256("\n".join(lines).encode()).hexdigest())
    return out


def fmt(k):
    r = k["res"]
    return (f"{k['instr']} instructions, {r[0]} registers, stack {r[1]} B, LDL {k['ldl']}, STL {k['stl']}, shared {r[2]} B, "
            f"local {r[3]} B, cbank0 {r[4]} B")


def family(k):
    m = re.match(r"^void fk::(\w+)<\s*([\w:]+)", k["demangled"])
    if not m:
        return k["demangled"].split("(")[0][:60]
    return m.group(1) + ("<" + m.group(2).split("<")[0] + ">" if m.group(1) == "launchDPP_Kernel" else "")


tag_a, dir_a = args.a.split("=", 1); tag_b, dir_b = args.b.split("=", 1)
fa, fb = files(dir_a), files(dir_b)
common = sorted(set(fa) & set(fb))
tool = subprocess.run(["cuobjdump", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
lines = [f"A = {tag_a}: {dir_a}", f"B = {tag_b}: {dir_b}",
         f"arch {args.arch}; {tool}; files ({args.glob} executables, lib{args.glob}.so): A {len(fa)}, B {len(fb)}, in both {len(common)}",
         f"only in A: {sorted(set(fa) - set(fb))}", f"only in B: {sorted(set(fb) - set(fa))}"]
tot = collections.Counter()
rules_used = collections.Counter()
by_family = collections.defaultdict(collections.Counter)
details, rows, per_file = [], [], []
for f in common:
    ka, kb = kernels(fa[f], "A"), kernels(fb[f], "B")
    n = collections.Counter(a=len(ka), b=len(kb))
    for key in sorted(set(ka) | set(kb)):
        x, y = ka.get(key), kb.get(key)
        if x and y:
            status = "identical" if x["lines"] == y["lines"] else "different"
            renamed = x["mangled"] != y["mangled"]
            n["renamed"] += renamed
            if renamed:
                rules_used[x["rule"] or "(none: mangled names differ otherwise)"] += 1
        else:
            status = "only in A" if x else "only in B"
        n[status] += 1
        fam = family(y or x)
        by_family[fam][status] += 1
        rows.append([f, status, (x or y)["rule"] if x else "", (x or {}).get("demangled", ""), (y or {}).get("demangled", ""),
                     *([""] * 10 if not x else [x["instr"], *x["res"], x["ldl"], x["stl"], x["sha"][:16]]),
                     *([""] * 10 if not y else [y["instr"], *y["res"], y["ldl"], y["stl"], y["sha"][:16]])])
        if status != "identical":
            details.append(f"  {f}: {status.upper()}  {(y or x)['demangled'][:600]}")
            if x and x["demangled"] != (y or x)["demangled"]:
                details.append(f"      A name: {x['demangled'][:600]}")
            if x: details.append(f"      A: {fmt(x)}")
            if y: details.append(f"      B: {fmt(y)}")
            if x and y:
                delta = {o: y["ops"].get(o, 0) - x["ops"].get(o, 0) for o in set(x["ops"]) | set(y["ops"])}
                delta = {o: d for o, d in delta.items() if d}
                details.append("      opcode counts B - A: " + (", ".join(f"{o} {d:+d}" for o, d in sorted(delta.items(), key=lambda t: (-abs(t[1]), t[0])))
                                                             or "none (the same opcodes, in another order or with other operands)"))
    tot.update(n)
    per_file.append(f"  {f}: kernels A {n['a']}, B {n['b']}; renamed pairs {n['renamed']}; identical {n['identical']}, different {n['different']}, "
                    f"only in A {n['only in A']}, only in B {n['only in B']}"
                    + ("" if n["different"] or n["only in A"] or n["only in B"] else "  (all identical)"))
lines.append(f"kernels: A {tot['a']}, B {tot['b']}; pairs {tot['identical'] + tot['different']} (renamed {tot['renamed']}); identical {tot['identical']}, "
             f"different {tot['different']}, only in A {tot['only in A']}, only in B {tot['only in B']}")
lines.append(f"names demangled by llvm-cxxfilt because GNU c++filt returned them unchanged: {len(FALLBACKS)}"
             + "".join(f"\n  {n[:160]}..." for n in FALLBACKS))
lines.append("renamed pairs by rule: " + (", ".join(f"{r} {c}" for r, c in sorted(rules_used.items())) or "none"))
lines.append("by kernel family (B's name, or A's for a kernel only in A): identical / different / only in A / only in B")
for fam in sorted(by_family):
    c = by_family[fam]
    lines.append(f"  {fam}: {c['identical']} / {c['different']} / {c['only in A']} / {c['only in B']}")
lines.append("per file:"); lines += per_file
lines.append("every kernel that is different or on one side only (A = base, B = head):"); lines += details or ["  none"]
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "status", "rename_rule", "A_demangled", "B_demangled",
                    "A_instructions", "A_registers", "A_stack", "A_shared", "A_local", "A_cbank0", "A_LDL", "A_STL", "A_sass_sha256_16",
                    "B_instructions", "B_registers", "B_stack", "B_shared", "B_local", "B_cbank0", "B_LDL", "B_STL", "B_sass_sha256_16"])
        w.writerows(rows)
ok = not (set(fa) ^ set(fb)) and tot["different"] == 0 and tot["only in A"] == 0 and tot["only in B"] == 0
sys.exit(0 if ok else 1)
