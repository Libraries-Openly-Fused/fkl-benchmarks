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
"""Tables of the controlled compile repeat (results/rtx4090/e3-compile/repeat-c8fcbe8/): per length the compiles, their median, minimum, maximum and the run-to-run spread.

  python scripts/e3_repeat_table.py results/rtx4090/e3-compile/repeat-c8fcbe8

Reads e3_repeat_A.csv (scripts/e3_sweep.py, the distinct-operations chain, variant A) and e3_repeat_staticloop.csv (scripts/e3_staticloop.py) and, when they exist, e3_repeat_extra_A.csv and
e3_repeat_extra_staticloop.csv (more lengths for the figure, compiled after the requested ones in the same window). The warm-up compile (e3_repeat_warmup_A.csv) is not read.
Writes
  e3_repeat_summary.csv / e3_repeat_summary.md   one row per length: the values of the compiles in the order they ran, median, minimum, maximum and spread = (max - min) / median of the wall
                                                 time and of the peak memory; the median of every nvcc phase and its share of the median wall time; the cubin size; whether the code is identical
                                                 in all compiles (registers, spills, SASS instructions, .text bytes)
  e3_table_chains.csv, e3_table_staticloop.csv   the layout scripts/e3_table.py writes and scripts/plot_compiletime.py reads; each cell is the median of the compiles of that length
Nothing is left out: a compile that failed is counted in the summary ("compiles" against "ok") and its status is listed; the cells are computed from the compiles with status ok.
"""
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e3_table as t3  # noqa: E402  (read_csv, med, num, param_space)

SWEEPS = [("e3_repeat_A.csv", "requested"), ("e3_repeat_extra_A.csv", "extra")]
LOOPS = [("e3_repeat_staticloop.csv", "requested"), ("e3_repeat_extra_staticloop.csv", "extra")]
PHASES = [("cudafe_s", "phase_cudafe_s"), ("cicc_s", "phase_cicc_s"), ("ptxas_s", "phase_ptxas_s"), ("gcc_s", "phase_gcc_s"), ("other_s", "phase_other_s")]
SAME = ["registers", "spill_stores", "sass_instructions", "text_bytes", "cubin_bytes"]


def load(d, files, key):
    groups = {}
    for fn, block in files:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            for r in t3.read_csv(p):
                g = groups.setdefault(key(r), {"block": block, "rows": []})
                g["rows"].append(r)
    return groups


def med4(rows, col):
    """median of a column, rounded to 4 decimals (the tables of scripts/e3_table.py print the float noise of the median)"""
    v = t3.med(rows, col)
    return round(v, 4) if v != "" else ""


def values(rows, col):
    return [t3.num(r[col]) for r in rows if t3.num(r.get(col)) is not None]


def summary_row(series, n, block, rows):
    rows = sorted(rows, key=lambda r: int(r["rep"]))
    ok = [r for r in rows if r["status"] == "ok"]
    out = {"series": series, "n": n, "block": block, "compiles": len(rows), "ok": len(ok),
           "status_not_ok": "; ".join(sorted({r["status"] for r in rows if r["status"] != "ok"}))}
    wall, rss = values(ok, "cuda_wall_s"), values(ok, "cuda_peak_rss_mb")
    out["nvcc_s_values"] = " / ".join(f"{v:.3f}" for v in wall)
    for name, v, fmt in (("nvcc_s", wall, "{:.3f}"), ("rss_mb", rss, "{:.0f}")):
        if v:
            m = statistics.median(v)
            out[name + "_median"] = fmt.format(m)
            out[name + "_min"] = fmt.format(min(v))
            out[name + "_max"] = fmt.format(max(v))
            out[name + "_spread_pct"] = f"{(max(v) - min(v)) / m * 100:.2f}" if m else ""
    out["rss_mb_values"] = " / ".join(f"{v:.0f}" for v in rss)
    for name, col in (("user_s", "cuda_user_s"), ("sys_s", "cuda_sys_s")):
        v = values(ok, col)
        out[name + "_median"] = f"{statistics.median(v):.3f}" if v else ""
    wall_med = statistics.median(wall) if wall else None
    shares = 0.0
    for name, col in PHASES:
        v = values(ok, col)
        m = statistics.median(v) if v else None
        out[name + "_median"] = f"{m:.3f}" if m is not None else ""
        out[name.replace("_s", "_share_pct")] = f"{m / wall_med * 100:.1f}" if (m is not None and wall_med) else ""
        shares += (m / wall_med * 100) if (m is not None and wall_med) else 0.0
    out["phases_share_pct"] = f"{shares:.1f}" if wall_med else ""
    for col in SAME:
        v = sorted({r[col] for r in ok})
        out[col] = v[0] if len(v) == 1 else ("differs: " + " / ".join(v) if v else "")
    return out


def main():
    d = sys.argv[1]
    chains = load(d, SWEEPS, lambda r: (r["tree"], r["variant"], int(r["n"])))
    loops = load(d, LOOPS, lambda r: int(r["ops"]))
    if not chains and not loops:
        sys.exit(f"no e3_repeat_*.csv in {d}")
    summary = []
    for (tree, variant, n), g in sorted(chains.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        summary.append(summary_row("distinct operations (variant %s, tree %s)" % (variant, tree), n, g["block"], g["rows"]))
    for ops, g in sorted(loops.items()):
        summary.append(summary_row("StaticLoop (cvGS mul_add1)", ops, g["block"], g["rows"]))
    cols = list(dict.fromkeys(k for r in summary for k in r))
    with open(os.path.join(d, "e3_repeat_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(summary)
    # the tables that scripts/plot_compiletime.py reads (the layout of scripts/e3_table.py)
    out = []
    for (tree, variant, n), g in sorted(chains.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        rows = sorted(g["rows"], key=lambda r: int(r["rep"]))
        ok = [r for r in rows if r["status"] == "ok"]
        first = ok[0] if ok else rows[0]
        out.append({"tree": tree, "variant": variant, "n": n, "status": "ok" if len(ok) == len(rows) else "; ".join(sorted({r["status"] for r in rows if r["status"] != "ok"})),
                    "first_error": "" if ok else first["why"], "param_space_bytes": t3.param_space(variant, n), "cmem0_bytes_ptxas": first.get("cmem0_bytes", ""),
                    "registers": first["registers"], "spill_stores_bytes": first["spill_stores"], "spill_loads_bytes": first["spill_loads"], "stack_bytes": first["stack_bytes"],
                    "sass_instructions": first["sass_instructions"], "text_bytes": first["text_bytes"], "cubin_bytes": first["cubin_bytes"],
                    "nvcc_s": med4(ok, "cuda_wall_s"), "cudafe_s": med4(ok, "phase_cudafe_s"), "cicc_s": med4(ok, "phase_cicc_s"), "ptxas_s": med4(ok, "phase_ptxas_s"),
                    "peak_rss_mb": med4(ok, "cuda_peak_rss_mb"), "gpp_host_s": "", "gpp_host_rss_mb": "", "host_status": "", "kernel_ms": ""})
    if out:
        with open(os.path.join(d, "e3_table_chains.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    sl_rows = []
    for ops, g in sorted(loops.items()):
        rows = sorted(g["rows"], key=lambda r: int(r["rep"]))
        ok = [r for r in rows if r["status"] == "ok"]
        first = ok[0] if ok else rows[0]
        sl_rows.append({"ops": ops, "nvcc_s": med4(ok, "cuda_wall_s"), "cudafe_s": med4(ok, "phase_cudafe_s"), "cicc_s": med4(ok, "phase_cicc_s"), "ptxas_s": med4(ok, "phase_ptxas_s"),
                        "peak_rss_mb": med4(ok, "cuda_peak_rss_mb"), "registers": first["registers"], "spills": first["spill_stores"], "sass_instructions": first["sass_instructions"],
                        "text_bytes": first["text_bytes"], "cubin_bytes": first["cubin_bytes"]})
    if sl_rows:
        with open(os.path.join(d, "e3_table_staticloop.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sl_rows[0].keys()))
            w.writeheader()
            w.writerows(sl_rows)
    # a readable table
    md = ["# E3 controlled repeat: compile time and memory per length (generated by scripts/e3_repeat_table.py)", "",
          "Values of the compiles in the order they ran; median; spread = (max - min) / median. Phase times and shares are medians (share = phase median / wall-time median).", ""]
    for title, pick in (("Distinct operations (variant A)", lambda r: r["series"].startswith("distinct")), ("StaticLoop (cvGS mul_add1, operations = 100 (k - 1) + 2)", lambda r: r["series"].startswith("StaticLoop"))):
        sel = [r for r in summary if pick(r)]
        if not sel:
            continue
        md += [f"## {title}", "", "| length | block | compiles (ok) | nvcc wall s (in order) | median | spread % | peak RSS MB (in order) | spread % | cudafe++ s | cicc s | ptxas s | host g++ s | other s | cicc % | ptxas % | host % | cubin bytes |",
               "|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sel:
            g = lambda k: r.get(k, "")
            md.append(f"| {r['n']:,} | {r['block']} | {r['compiles']} ({r['ok']}) | {g('nvcc_s_values')} | {g('nvcc_s_median')} | {g('nvcc_s_spread_pct')} | {g('rss_mb_values')} | {g('rss_mb_spread_pct')} | "
                      f"{g('cudafe_s_median')} | {g('cicc_s_median')} | {g('ptxas_s_median')} | {g('gcc_s_median')} | {g('other_s_median')} | {g('cicc_share_pct')} | {g('ptxas_share_pct')} | {g('gcc_share_pct')} | {g('cubin_bytes')} |")
        md.append("")
    open(os.path.join(d, "e3_repeat_summary.md"), "w").write("\n".join(md) + "\n")
    print(f"{len(summary)} lengths ({sum(r['compiles'] for r in summary)} compiles) -> {d}/e3_repeat_summary.csv, .md, e3_table_chains.csv, e3_table_staticloop.csv")


if __name__ == "__main__":
    main()
