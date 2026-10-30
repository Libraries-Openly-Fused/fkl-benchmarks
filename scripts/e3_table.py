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
"""The E3 tables: one row per (tree, variant, N) from the CSVs of scripts/e3_sweep.py, e3_staticloop.py and the kernel-time runs.

  python scripts/e3_table.py results/rtx4090/e3-compile [--kernel-times results/rtx4090/e3-kernel-time/kernel_time_by_binary.csv]

Writes e3_table_chains.csv (variants A and B, trees current / patched / getarg-pr), e3_table_trees.csv (variant L on the four FKL trees),
e3_table_staticloop.csv and e3_table.md. The parameter block is the parameter space ptxas checks against 0x7ffc = 32,764 B, computed from
the layout of the kernel parameters in PTX (alignment and size of every parameter; the model reproduces ptxas' 0x8508 at N = 4,000 of
variant B and the PTX declarations at N = 1,023 and 1,024 of variant A); ptxas' own cmem[0] figure is NOT that number (it includes reserved bytes).
"""
import csv
import glob
import io
import os
import statistics
import sys


def read_csv(path):
    lines = [l for l in open(path) if not l.startswith("#")]
    return list(csv.DictReader(io.StringIO("".join(lines))))


def layout(params):
    off = 0
    for a, s in params:
        off = (off + a - 1) // a * a + s
    return off


def param_space(variant, n):
    """bytes of the kernel parameter space of the fused kernel of N operations: [details (1 B), read (24 B, align 8), N operations, write (24 B, align 8)]"""
    if variant == "A" or variant == "L":
        ops = [(4, 4)] * n
    else:   # B: Mul, Add, VectorReorder, Cast, Add<int3>, Cast, Sub, Div; a closing cast when the chain ends on an int3
        period = [(4, 12), (4, 12), (1, 1), (1, 1), (4, 12), (1, 1), (4, 12), (4, 12)]
        ops = [period[i % 8] for i in range(n)]
        if n % 8 in (4, 5):
            ops.append((1, 1))
    return layout([(1, 1), (8, 24)] + ops + [(8, 24)])


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def med(rows, key):
    v = [num(r.get(key)) for r in rows if num(r.get(key)) is not None]
    return statistics.median(v) if v else ""


def main():
    d = sys.argv[1]
    kt = {}
    if "--kernel-times" in sys.argv:
        for r in read_csv(sys.argv[sys.argv.index("--kernel-times") + 1]):
            kt[(r["tree"], r["variant"], int(r["n"]))] = r["kernel_ms_median"]
    sources = {"e3_compile.csv": "current", "e3_compile_B.csv": "current", "e3_compile_A_knee.csv": "current", "e3_compile_A_knee_fine.csv": "current",
               "e3_limit_B_3847.csv": "current", "e3_limit_B_3848.csv": "current", "e3_patched_A.csv": "patched", "e3_patched_B.csv": "patched",
               "e3_getarg_pr_A.csv": "getarg-pr", "compile-times-1-100-500-2000/without_fix.csv": "current", "compile-times-1-100-500-2000/with_fix.csv": "getarg-pr"}
    groups = {}
    for fn, tree in sources.items():
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        for r in read_csv(p):
            groups.setdefault((tree, r["variant"], int(r["n"])), []).append(r)
    out = []
    for (tree, variant, n), rows in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        first = sorted(rows, key=lambda r: int(r["rep"]))[0]
        ok = first["status"] == "ok"
        out.append({
            "tree": tree, "variant": variant, "n": n, "status": first["status"], "first_error": first["why"] if not ok else "",
            "param_space_bytes": param_space(variant, n), "cmem0_bytes_ptxas": first.get("cmem0_bytes", ""),
            "registers": first["registers"], "spill_stores_bytes": first["spill_stores"], "spill_loads_bytes": first["spill_loads"], "stack_bytes": first["stack_bytes"],
            "sass_instructions": first["sass_instructions"], "text_bytes": first["text_bytes"], "cubin_bytes": first["cubin_bytes"],
            "nvcc_s": med(rows, "cuda_wall_s"), "cudafe_s": med(rows, "phase_cudafe_s"), "cicc_s": med(rows, "phase_cicc_s"), "ptxas_s": med(rows, "phase_ptxas_s"),
            "peak_rss_mb": med(rows, "cuda_peak_rss_mb"), "gpp_host_s": med(rows, "host_wall_s"), "gpp_host_rss_mb": med(rows, "host_peak_rss_mb"),
            "host_status": first.get("host_status", ""), "kernel_ms": kt.get((tree, variant, n), ""),
        })
    cols = list(out[0].keys())
    with open(os.path.join(d, "e3_table_chains.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    trees = []
    for p in sorted(glob.glob(os.path.join(d, "e3_trees_*.csv")) + glob.glob(os.path.join(d, "e3_limit_*_*.csv"))):
        if "_B_" in os.path.basename(p):
            continue
        for r in read_csv(p):
            if r["rep"] == "0":
                trees.append({"tree": r["tree"], "n": int(r["n"]), "status": r["status"], "first_error": r["why"][:200] if r["status"] != "ok" else "", "nvcc_s": r["cuda_wall_s"],
                              "registers": r["registers"], "sass_instructions": r["sass_instructions"], "peak_rss_mb": r["cuda_peak_rss_mb"]})
    trees.sort(key=lambda r: (r["tree"], r["n"]))
    with open(os.path.join(d, "e3_table_trees.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trees[0].keys()))
        w.writeheader()
        w.writerows(trees)
    sl = os.path.join(d, "e3_staticloop_mul_add1.csv")
    sl_rows = []
    if os.path.exists(sl):
        by = {}
        for r in read_csv(sl):
            by.setdefault(int(r["ops"]), []).append(r)
        for ops, rows in sorted(by.items()):
            sl_rows.append({"ops": ops, "nvcc_s": med(rows, "cuda_wall_s"), "cudafe_s": med(rows, "phase_cudafe_s"), "cicc_s": med(rows, "phase_cicc_s"), "ptxas_s": med(rows, "phase_ptxas_s"),
                            "peak_rss_mb": med(rows, "cuda_peak_rss_mb"), "registers": rows[0]["registers"], "spills": rows[0]["spill_stores"], "sass_instructions": rows[0]["sass_instructions"],
                            "text_bytes": rows[0]["text_bytes"], "cubin_bytes": rows[0]["cubin_bytes"]})
        with open(os.path.join(d, "e3_table_staticloop.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sl_rows[0].keys()))
            w.writeheader()
            w.writerows(sl_rows)
    # without and with the get_arg change side by side: "after" is the include tree of the PR branch where it was measured, else the scratch copy with the same function
    by = {(r["tree"], r["variant"], r["n"]): r for r in out}
    side = []
    for variant in ("A", "B"):
        for n in sorted({r["n"] for r in out if r["variant"] == variant and r["tree"] == "current"} | {r["n"] for r in out if r["variant"] == variant and r["tree"] in ("getarg-pr", "patched")}):
            b = by.get(("current", variant, n))
            a = by.get(("getarg-pr", variant, n)) or by.get(("patched", variant, n))
            if b is None and a is None:
                continue
            g = lambda r, k: (r[k] if r else "")
            side.append({"variant": variant, "n": n, "param_space_bytes": param_space(variant, n),
                         "registers_without": g(b, "registers"), "registers_with": g(a, "registers"),
                         "spill_bytes_without": g(b, "spill_stores_bytes"), "spill_bytes_with": g(a, "spill_stores_bytes"),
                         "sass_without": g(b, "sass_instructions"), "sass_with": g(a, "sass_instructions"),
                         "nvcc_s_without": g(b, "nvcc_s"), "nvcc_s_with": g(a, "nvcc_s"), "cicc_s_without": g(b, "cicc_s"), "cicc_s_with": g(a, "cicc_s"),
                         "ptxas_s_without": g(b, "ptxas_s"), "ptxas_s_with": g(a, "ptxas_s"), "rss_mb_without": g(b, "peak_rss_mb"), "rss_mb_with": g(a, "peak_rss_mb"),
                         "kernel_ms_without": g(b, "kernel_ms"), "kernel_ms_with": g(a, "kernel_ms"),
                         "status_without": g(b, "status"), "status_with": g(a, "status"), "with_tree": g(a, "tree")})
    with open(os.path.join(d, "e3_table_before_after.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(side[0].keys()))
        w.writeheader()
        w.writerows(side)
    # a readable summary
    md = ["# E3 tables (generated by scripts/e3_table.py)", ""]
    for variant in ("A", "B"):
        md += [f"## Variant {variant}: chains of N distinct parameterised operations", "",
               "| tree | N | parameter space B | registers | spill stores B | SASS | nvcc s | cicc s | ptxas s | RSS MB | g++ host s | kernel ms | status |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in out:
            if r["variant"] == variant:
                fmt = lambda x: f"{x:.1f}" if isinstance(x, float) else x
                md.append(f"| {r['tree']} | {r['n']} | {r['param_space_bytes']} | {r['registers']} | {r['spill_stores_bytes']} | {r['sass_instructions']} | {fmt(r['nvcc_s'])} | {fmt(r['cicc_s'])} | {fmt(r['ptxas_s'])} | {fmt(r['peak_rss_mb'])} | {fmt(r['gpp_host_s'])} | {r['kernel_ms']} | {'ok' if r['status'] == 'ok' else r['status'] + ': ' + r['first_error'][:80]} |")
        md.append("")
    fmt2 = lambda x: f"{x:.1f}" if isinstance(x, float) else x
    md += ["## Without / with the get_arg change, side by side (the after column: " + "the include tree of the PR branch where measured, else a copy with the same function)", "",
           "| variant | N | parameter space B | registers | spill stores B | SASS | nvcc s | cicc s | ptxas s | RSS MB | kernel ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in side:
        pair = lambda k: f"{fmt2(r[k + '_without'])} / {fmt2(r[k + '_with'])}"
        st = r["status_without"] if r["status_without"] not in ("ok", "") else ""
        st += (" (with: " + r["status_with"] + ")") if r["status_with"] not in ("ok", "") else ""
        md.append(f"| {r['variant']} | {r['n']}{(' ' + st) if st else ''} | {r['param_space_bytes']} | {pair('registers')} | {pair('spill_bytes')} | {pair('sass')} | {pair('nvcc_s')} | {pair('cicc_s')} | {pair('ptxas_s')} | {pair('rss_mb')} | {pair('kernel_ms')} |")
    md.append("")
    md += ["## Fold versus recursion (Mul/Add<float>, variant L)", "", "| tree | N | nvcc s | registers | status |", "|---|---:|---:|---:|---|"]
    for r in trees:
        md.append(f"| {r['tree']} | {r['n']} | {r['nvcc_s']} | {r['registers']} | {'ok' if r['status'] == 'ok' else r['status'] + ': ' + r['first_error'][:90]} |")
    md += ["", "## StaticLoop axis (cvGS mul_add1)", "", "| ops | nvcc s | cudafe++ s | cicc s | ptxas s | RSS MB | registers | spills | SASS |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sl_rows:
        md.append(f"| {r['ops']} | {r['nvcc_s']} | {r['cudafe_s']} | {r['cicc_s']} | {r['ptxas_s']} | {r['peak_rss_mb']} | {r['registers']} | {r['spills']} | {r['sass_instructions']} |")
    open(os.path.join(d, "e3_table.md"), "w").write("\n".join(md) + "\n")
    print(f"{len(out)} chain rows, {len(trees)} tree rows, {len(sl_rows)} StaticLoop rows -> {d}")


if __name__ == "__main__":
    main()
