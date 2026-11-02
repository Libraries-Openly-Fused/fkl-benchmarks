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
"""E16: what the attention kernels of the E16 program are, from the binary: template arguments (demangled) and SASS instruction counts per kernel instantiation.

  python scripts/e16_sass_counts.py build/local-rtx4090/bin/e16_attention_cu sm_89 results/rtx4090/e16-attention/e16_sass_counts.csv

For every launchFlashAttentionMmaDPP_Kernel / launchFlashAttentionDPP_Kernel in the cubin of the architecture: head dimension, the K and V read types, the epilogue type, BLOCK_Q, BLOCK_KV, NUM_WARPS, SPLIT_KV, registers and
stack bytes (cuobjdump --dump-resource-usage), and the number of SASS instructions of the families that tell the load schedule: LDGSTS (cp.async global to shared), LDG (global to registers), LDS / LDSM (shared to registers),
STS (registers to shared), HMMA (tensor-core mma), BAR, MUFU. The 'schedule' column is what the instruction counts show: 'cp.async' when the kernel has LDGSTS, 'registers' when it loads from global memory with LDG and has none.
"""
import csv
import re
import subprocess
import sys
from collections import Counter

FAMILIES = ["LDGSTS", "LDG", "LDS", "LDSM", "STS", "STG", "HMMA", "BAR", "MUFU", "LDL", "STL"]


def split_args(text):
    """Top-level comma separated template arguments of the first <...> of text."""
    start = text.index("<") + 1
    depth, cur, out = 1, [], []
    for ch in text[start:]:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                break
        if ch == "," and depth == 1:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


def demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, out))


def short(t):
    m = re.findall(r"fk::(\w+)", t)
    return m[-1] if m else t


def main():
    binary, arch, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sass = subprocess.run(["cuobjdump", "-sass", "-arch", arch, binary], capture_output=True, text=True).stdout
    usage = subprocess.run(["cuobjdump", "--dump-resource-usage", "-arch", arch, binary], capture_output=True, text=True).stdout
    regs = {}
    for name, line in re.findall(r"Function (\S+):\s*\n\s*(.*)", usage):
        regs[name] = (int(re.search(r"REG:(\d+)", line).group(1)), int(re.search(r"STACK:(\d+)", line).group(1)), int(re.search(r"SHARED:(\d+)", line).group(1)))
    funcs = {}
    cur = None
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = Counter()
            continue
        if cur:
            m = re.match(r"\s*/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)", line)
            if m:
                funcs[cur][m.group(1)] += 1
    attn = [n for n in funcs if "FlashAttention" in n and "Kernel" in n]
    dem = demangle(attn)
    rows = []
    for n in attn:
        d = dem[n]
        mma = "launchFlashAttentionMmaDPP_Kernel<" in d
        key = "launchFlashAttentionMmaDPP_Kernel<" if mma else "launchFlashAttentionDPP_Kernel<"
        a = split_args(d[d.index(key) - 1 + len(key) - 1:] if False else d[d.index(key) + len(key) - 1:])
        row = {"arch": arch, "kind": "mma" if mma else "simt", "head_dim": a[1]}
        if mma:   # OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_Q, BLOCK_KV, NUM_WARPS, ScoreModOp, SPLIT_KV
            row.update({"q_read": short(a[2]), "k_read": short(a[3]), "v_read": short(a[4]), "epilogue": short(a[5]), "block_q": a[6], "block_kv": a[7], "num_warps": a[8], "split_kv": a[10] if len(a) > 10 else "false"})
        else:     # OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK
            row.update({"q_read": short(a[2]), "k_read": short(a[3]), "v_read": short(a[4]), "epilogue": short(a[5]), "block_q": a[7], "block_kv": a[6], "num_warps": a[7], "split_kv": "false"})
        r = regs.get(n, (0, 0, 0))
        row.update({"registers": r[0], "stack_bytes": r[1], "static_smem_bytes": r[2]})
        for fam in FAMILIES:
            row[f"sass_{fam}"] = funcs[n][fam]
        row["sass_total"] = sum(funcs[n].values())
        row["schedule"] = "cp.async" if funcs[n]["LDGSTS"] > 0 else ("registers" if funcs[n]["LDG"] > 0 else "?")
        row["symbol"] = n
        rows.append(row)
    rows.sort(key=lambda r: (r["kind"], int(r["head_dim"]), r["k_read"], r["epilogue"], r["split_kv"], int(r["block_q"]), int(r["block_kv"])))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} attention kernels of {binary} for {arch} -> {out_path}")
    print(f"{'kind':5s} {'D':>3s} {'K read':>24s} {'epilogue':>26s} {'BQ':>4s} {'BKV':>4s} {'NW':>3s} {'split':>5s} {'regs':>4s} {'stack':>5s} {'LDGSTS':>7s} {'LDG':>5s} {'LDS':>5s} {'LDSM':>5s} {'STS':>5s} {'HMMA':>6s} {'schedule':>9s}")
    for r in rows:
        print(f"{r['kind']:5s} {r['head_dim']:>3s} {r['k_read']:>24s} {r['epilogue']:>26s} {r['block_q']:>4s} {r['block_kv']:>4s} {r['num_warps']:>3s} {r['split_kv']:>5s} {r['registers']:>4d} {r['stack_bytes']:>5d} {r['sass_LDGSTS']:>7d} {r['sass_LDG']:>5d} {r['sass_LDS']:>5d} {r['sass_LDSM']:>5d} {r['sass_STS']:>5d} {r['sass_HMMA']:>6d} {r['schedule']:>9s}")


if __name__ == "__main__":
    main()
