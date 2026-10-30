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
"""E1-SASS: are the kernels FKL generates for the vertical-fusion chain the same machine code as the hand-written ones?

  e1_sass.py <e1_vf_chain_cu binary> <out_dir> [--arch sm_89]

Takes the SASS of every kernel of the E1a binary (handwritten-cuda/e1_vf_chain.cu), pairs them by number of operations and writes,
for each number of operations N, hand_params.sass, hand_imm.sass, hand_vec.sass, fkl.sass (thread fusion off) and fkl_tf.sass (thread
fusion on), normalised (addresses and encodings removed, registers renamed by first appearance), plus the unified diffs
hand_params-vs-fkl.diff and hand_vec-vs-fkl_tf.diff, and report.md with, per N and per kernel: total instructions, the histogram
classes of the paper's table (FFMA/FMUL/FADD, LDG/STG, LDL/STL, constant-bank operands, IMAD/ISETP/BRA control, barriers),
registers, stack frame, local memory and constant-bank size (cuobjdump -res-usage), and whether the instruction streams are
identical. The kernels are paired by their template argument (hand-written ones: the number of pairs) or by the number of FFMA
instructions (FKL: one FFMA per Mul+Add pair per element, four elements per thread with thread fusion). Tools: the cuobjdump in PATH,
which must be the CUDA 13.4 one (source scripts/env-cuda13.4.sh); the report records it and the GPU.
"""
import argparse
import collections
import difflib
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sass_tools as st  # noqa: E402


def res_usage(binary, arch):
    out = subprocess.run(["cuobjdump", "-arch", arch, "-res-usage", binary], capture_output=True, text=True).stdout
    usage, name = {}, None
    for line in out.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            name = m.group(1)
        elif name and "REG:" in line:
            usage[name] = {k: int(v) for k, v in re.findall(r"(REG|STACK|SHARED|LOCAL|CONSTANT\[0\]):(\d+)", line)}
            name = None
    return usage


CHAIN_OPS = re.compile(r"^(FFMA|FMUL|FADD|LDG|STG|I2FP|F2FP|I2F|F2F|FMNMX)\b")


def chain_signature(lines):
    """The arithmetic and memory instructions of a kernel, in order, with register numbers and constant-bank offsets blanked."""
    sig = []
    for line in lines:
        body = st.PRED_PREFIX.sub("", line)
        if CHAIN_OPS.match(body):
            body = re.sub(r"c\[0x[0-9a-f]+\]\[0x[0-9a-f]+\]", "c[]", body)
            body = re.sub(r"\b(UR|R)\d+\b", "R", body)
            body = re.sub(r"\.reuse", "", body)
            sig.append(body)
    return sig


def classify_kernels(funcs):
    """name -> (variant, pairs) using the template argument of hand kernels and the FFMA count of FKL kernels."""
    out = {}
    for name, body in funcs.items():
        lines = st.normalise_sass(body, rename=True)
        instr, hist = st.classify(lines)
        ffma = hist.get("FFMA", 0)
        m = re.search(r"(handParams|handImmediates|handVectorized)ILi(\d+)E", name)
        if m:
            variant = {"handParams": "hand_params", "handImmediates": "hand_imm", "handVectorized": "hand_vec"}[m.group(1)]
            out[name] = (variant, int(m.group(2)), lines)
        elif "launchTransformDPP_Kernel" in name:
            demangled = subprocess.run(["c++filt", name], capture_output=True, text=True).stdout
            thread_fusion = "(fk::TF)1" in demangled
            if thread_fusion and hist.get("LDG", 0) >= 2:
                # the variant for widths that are not a multiple of the thread-fusion vector: a vector path plus a scalar tail
                # (4 + 1 FFMA per pair); the divisible-width variant is the one that runs for the 4096-wide image
                out[name] = ("fkl_tf_tail", ffma // 5, lines)
            else:
                out[name] = ("fkl_tf" if thread_fusion else "fkl", ffma // 4 if thread_fusion else ffma, lines)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary"); ap.add_argument("out_dir"); ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--keep-sass-up-to", type=int, default=10**9, help="write the .sass and .diff files only for kernels of at most this many operations (the report always covers all)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sass = subprocess.run(["cuobjdump", "-arch", args.arch, "-sass", args.binary], capture_output=True, text=True).stdout
    funcs = st.split_functions(sass)
    usage = res_usage(args.binary, args.arch)
    kernels = classify_kernels(funcs)
    by_n = collections.defaultdict(dict)
    names = {}
    for name, (variant, pairs, lines) in kernels.items():
        by_n[pairs][variant] = lines
        names[(pairs, variant)] = name

    # kernels whose pair count does not match any hand-written kernel (e.g. the scalar-tail variant of a very small chain, whose tail loop
    # the compiler unrolled so that its FFMA count is not a multiple of the pairs) cannot be paired: leave them out of the report
    for pairs in [p for p in by_n if "hand_params" not in by_n[p]]:
        del by_n[pairs]

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().replace("\n", " | ")
    md = ["# E1-SASS: hand-written kernels against FKL, vertical-fusion chain", "",
          f"binary `{args.binary}`, arch {args.arch}, registers renamed by first appearance.", "",
          "```", st.tool_version("nvcc"), st.tool_version("cuobjdump"), f"GPU, driver: {smi}", "```", ""]
    order = ["hand_params", "hand_imm", "hand_vec", "fkl", "fkl_tf", "fkl_tf_tail"]
    for pairs in sorted(by_n):
        group = by_n[pairs]
        ops = pairs * 2
        keep = ops <= args.keep_sass_up_to
        d = os.path.join(args.out_dir, f"ops{ops}")
        if keep:
            os.makedirs(d, exist_ok=True)
        md += [f"## {ops} operations ({pairs} Mul+Add pairs per element)", "",
               "| kernel | instructions | FFMA | FMUL | FADD | LDG | STG | LDL | STL | const-bank operands | IMAD | ISETP | BRA etc. | barriers | regs | stack | local | cmem[0] B |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v in order:
            if v not in group:
                continue
            lines = group[v]
            if keep:
                with open(os.path.join(d, f"{v}.sass"), "w") as f:
                    f.write("\n".join(lines) + "\n")
            n, h = st.classify(lines)
            u = usage.get(names[(pairs, v)], {})
            md.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                v, n, h.get("FFMA", 0), h.get("FMUL", 0), h.get("FADD", 0), h.get("LDG", 0), h.get("STG", 0), h.get("LDL", 0), h.get("STL", 0),
                h.get("instructions with a constant-bank operand c[bank][off]", 0), h.get("IMAD", 0), h.get("ISETP", 0),
                h.get("BRA/BSSY/BSYNC/CALL/RET/EXIT", 0), h.get("barriers (BAR, MEMBAR, DEPBAR, ERRBAR)", 0),
                u.get("REG", "?"), u.get("STACK", "?"), u.get("LOCAL", "?"), u.get("CONSTANT[0]", "?")))
        md.append("")
        for a, b in (("hand_params", "fkl"), ("hand_vec", "fkl_tf")):
            if a in group and b in group:
                sa, sb = chain_signature(group[a]), chain_signature(group[b])
                relax = lambda sig: sorted(re.sub(r"\.CONSTANT", "", x) for x in sig)
                if sa == sb:
                    verdict = "**identical, in the same order**"
                elif relax(sa) == relax(sb):
                    only_constant = [re.sub(r"\.CONSTANT", "", x) for x in sa] == [re.sub(r"\.CONSTANT", "", x) for x in sb]
                    verdict = ("**identical except the `.CONSTANT` modifier of the loads** (the hand kernel reads its input through `const __restrict__`, "
                               "the read-only data path; FKL emits a plain load)" if only_constant else
                               "**the same instructions** (ignoring the `.CONSTANT` modifier of the loads) **in a different order** "
                               "(instruction scheduling of the conversions)")
                else:
                    extra_a = collections.Counter(relax(sa)) - collections.Counter(relax(sb))
                    extra_b = collections.Counter(relax(sb)) - collections.Counter(relax(sa))
                    verdict = f"**DIFFERENT**: only in `{a}`: {dict(extra_a)}; only in `{b}`: {dict(extra_b)}"
                    brA = st.classify(group[a])[1].get("BRA/BSSY/BSYNC/CALL/RET/EXIT", 0)
                    brB = st.classify(group[b])[1].get("BRA/BSSY/BSYNC/CALL/RET/EXIT", 0)
                    if brA > brB and not extra_a:
                        verdict += (f" (`{a}` has {brA} control instructions against {brB}: nvcc did not fully unroll the hand-written loop "
                                    f"despite `#pragma unroll`, it compiled a loop with a partially unrolled body, while FKL's kernel is straight-line code)")
                md.append(f"- `{a}` vs `{b}`, the {len(sa)} arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers "
                          f"and constant-bank offsets ignored): {verdict}. Other instructions (thread index, bounds, address arithmetic): "
                          f"{len(group[a]) - len(sa)} in `{a}`, {len(group[b]) - len(sb)} in `{b}`.")
                diff = list(difflib.unified_diff(group[a], group[b], a, b, n=1, lineterm=""))
                if keep:
                    with open(os.path.join(d, f"{a}-vs-{b}.diff"), "w") as f:
                        f.write("\n".join(diff) + "\n")
                changed = [l for l in diff if l[:1] in "+-" and l[:3] not in ("+++", "---")]
                same = not changed
                md.append(f"- `{a}` vs `{b}`: " + ("**identical instruction streams**" if same else
                          f"differ: {len(group[a])} vs {len(group[b])} lines, {len(changed)} changed lines in the diff" + (f" (`ops{ops}/{a}-vs-{b}.diff`)" if keep else "")))
        md.append("")
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
