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
"""The E1 table: one row per (experiment point, version) with everything the paper's table needs, so that nothing is measured twice.

  e1_table.py probe-input --ncu ncu.csv > probe_input.tsv        kernel / threads / shared memory lines for build/tools/occupancy_probe
  e1_table.py cubin --binary <e1 binary> --out file.cubin         the sm_89 cubin of the binary, for occupancy_probe
  e1_table.py table --experiment vf|hf|pp --binary <e1 binary> --timing <e1 csv> --ncu ncu.csv --occupancy probe_output.tsv \\
                    --out table.csv [--arch sm_89]

Columns (one row per launched kernel):
  experiment, point (operations or crops), version
  gpu_mean_ms, gpu_min_ms                       from the timing CSV of the benchmark (CUDA events, 100 launches)
  sass_instructions and the histogram classes   cuobjdump -sass of the kernel (FFMA, FMUL, FADD, LDG, STG, LDL, STL = the spills,
                                                instructions with a constant-bank operand, IMAD, ISETP, BRA/BSSY/..., barriers)
  registers, stack_bytes, local_bytes, cmem0_bytes   cuobjdump -res-usage (local_bytes: spills and local arrays; cmem0: parameter block)
  code_bytes, cubin_bytes_total                 size of the kernel's .text section, and of the whole cubin of the binary
  block_threads, grid                           the launch configuration the profiler saw
  theoretical_occupancy_pct, active_blocks_per_sm   cudaOccupancyMaxActiveBlocksPerMultiprocessor at that block size (occupancy_probe)
  theoretical_occupancy_ncu_pct                 sm__maximum_warps_per_active_cycle_pct of ncu (cross-check of the previous column)
  achieved_occupancy_pct                        sm__warps_active.avg.pct_of_peak_sustained_active of ncu
  ncu_duration_us                               gpu__time_duration.sum under ncu (informative: ncu locks the clocks)
The kernels come from the ncu run (E1_PROFILE=1, one launch per kernel), joined with the static data by their mangled names.
"""
import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sass_tools as st  # noqa: E402
import e1_sass  # noqa: E402

M_ACHIEVED = "sm__warps_active.avg.pct_of_peak_sustained_active"
M_THEORETICAL = "sm__maximum_warps_per_active_cycle_pct"
M_DURATION = "gpu__time_duration.sum"


def read_ncu(path):
    """list of dicts, one per profiled kernel launch: name (mangled), block, grid and the metric values (floats)."""
    text = open(path, errors="replace").read()
    m = re.search(r'^"?ID"?,', text, re.M)    # ncu prints a few lines of its own before the CSV
    if not m:
        raise SystemExit(f"{path}: no CSV header (a line starting with \"ID\") in the ncu output")
    rows = list(csv.reader(io.StringIO(text[m.start():])))
    header = rows[0]
    out = []
    for row in rows[1:]:
        if len(row) != len(header) or not row[0].strip().isdigit():
            continue                      # the units row, or noise
        rec = dict(zip(header, row))
        metrics = {}
        for k, v in rec.items():
            try:
                metrics[k] = float(v.replace(",", ""))
            except ValueError:
                pass
        grid = [int(x) for x in re.findall(r"\d+", rec.get("Grid Size", ""))]
        block = [int(x) for x in re.findall(r"\d+", rec.get("Block Size", ""))]
        out.append({"name": rec["Kernel Name"], "grid": grid, "block": block, "metrics": metrics})
    return out


def cmd_probe_input(args):
    seen = set()
    for k in read_ncu(args.ncu):
        threads = 1
        for b in k["block"]:
            threads *= b
        key = (k["name"], threads)
        if key not in seen:
            seen.add(key)
            print(f"{k['name']} {threads} 0")


def text_section_sizes(path):
    """name -> size in bytes of the .text.<name> sections of an ELF64 file (readelf truncates the long names of FKL's kernels)."""
    import struct
    data = open(path, "rb").read()
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]   # name, type, flags, addr, offset, size, ...
    strtab_off = sections[shstrndx][4]
    sizes = {}
    for sec in sections:
        end = data.index(b"\0", strtab_off + sec[0])
        name = data[strtab_off + sec[0]:end].decode()
        if name.startswith(".text."):
            sizes[name[len(".text."):]] = sec[5]
    return sizes


def code_sizes(binary, arch):
    """name -> .text size, and the cubin file size, from the cubin cuobjdump extracts."""
    tmp = tempfile.mkdtemp(prefix="e1cubin")
    try:
        subprocess.run(["cuobjdump", "-arch", arch, "-xelf", "all", os.path.abspath(binary)], cwd=tmp, capture_output=True, check=True)
        cubins = [f for f in os.listdir(tmp) if f.endswith(".cubin")]
        if len(cubins) != 1:
            raise SystemExit(f"expected one {arch} cubin in {binary}, found {cubins}")
        path = os.path.join(tmp, cubins[0])
        sizes = text_section_sizes(path)
        return sizes, os.path.getsize(path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_cubin(args):
    tmp = tempfile.mkdtemp(prefix="e1cubin")
    try:
        subprocess.run(["cuobjdump", "-arch", args.arch, "-xelf", "all", os.path.abspath(args.binary)], cwd=tmp, capture_output=True, check=True)
        cubins = [f for f in os.listdir(tmp) if f.endswith(".cubin")]
        if len(cubins) != 1:
            raise SystemExit(f"expected one {args.arch} cubin in {args.binary}, found {cubins}")
        shutil.copy(os.path.join(tmp, cubins[0]), args.out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def read_timing(path):
    lines = [l for l in open(path) if not l.startswith("#")]
    rows = list(csv.DictReader(io.StringIO("".join(lines))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cubin"); c.add_argument("--binary", required=True); c.add_argument("--out", required=True); c.add_argument("--arch", default="sm_89")
    c.set_defaults(fn=cmd_cubin)
    p = sub.add_parser("probe-input"); p.add_argument("--ncu", required=True); p.set_defaults(fn=cmd_probe_input)
    t = sub.add_parser("table")
    t.add_argument("--experiment", choices=["vf", "hf", "pp"], required=True)
    t.add_argument("--binary", required=True); t.add_argument("--timing", required=True); t.add_argument("--ncu", required=True)
    t.add_argument("--occupancy"); t.add_argument("--out", required=True); t.add_argument("--arch", default="sm_89")
    t.set_defaults(fn=cmd_table)
    args = ap.parse_args()
    args.fn(args)


def cmd_table(args):
    sass = subprocess.run(["cuobjdump", "-arch", args.arch, "-sass", args.binary], capture_output=True, text=True).stdout
    funcs = st.split_functions(sass)
    usage = e1_sass.res_usage(args.binary, args.arch)
    sizes, cubin_total = code_sizes(args.binary, args.arch)
    timing = {}
    for row in read_timing(args.timing):
        timing[int(row["ops" if args.experiment == "vf" else "batch"])] = row
    occupancy = {}
    if args.occupancy:
        for line in open(args.occupancy):
            f = line.rstrip("\n").split("\t")
            if len(f) == 5:
                occupancy[f[0]] = (int(f[1]), float(f[4]))
    ncu = {}
    for k in read_ncu(args.ncu):
        ncu.setdefault(k["name"], k)          # the same kernel launched twice profiles the same

    classified = e1_sass.classify_kernels(funcs) if args.experiment == "vf" else None
    rows = []
    for name, k in ncu.items():
        if name not in funcs:
            print(f"warning: {name[:80]} is not in the SASS of the binary", file=sys.stderr)
            continue
        lines = st.normalise_sass(funcs[name], rename=True)
        instr, hist = st.classify(lines)
        u = usage.get(name, {})
        if args.experiment == "vf":
            version, pairs, _ = classified[name]
            point = pairs * 2
        else:
            demangled = subprocess.run(["c++filt", name], capture_output=True, text=True).stdout
            point = k["grid"][2] if len(k["grid"]) > 2 else 1
            if "launchTransformDPP" in name:
                version = "fkl_tf" if "(fk::TF)1" in demangled else "fkl"
            else:
                version = "hand_generic" if "Generic" in name else "hand"
        tr = timing.get(point, {})
        threads = 1
        for b in k["block"]:
            threads *= b
        api = occupancy.get(name)
        rows.append({
            "experiment": {"vf": "E1a", "hf": "E1b", "pp": "E1c"}[args.experiment], "point": point, "version": version,
            "gpu_mean_ms": tr.get(f"{version}_mean_ms", ""), "gpu_min_ms": tr.get(f"{version}_min_ms", ""),
            "sass_instructions": instr,
            "FFMA": hist.get("FFMA", 0), "FMUL": hist.get("FMUL", 0), "FADD": hist.get("FADD", 0), "LDG": hist.get("LDG", 0), "STG": hist.get("STG", 0),
            "LDL": hist.get("LDL", 0), "STL": hist.get("STL", 0),
            "const_bank_operands": hist.get("instructions with a constant-bank operand c[bank][off]", 0),
            "IMAD": hist.get("IMAD", 0), "ISETP": hist.get("ISETP", 0), "BRA_etc": hist.get("BRA/BSSY/BSYNC/CALL/RET/EXIT", 0),
            "barriers": hist.get("barriers (BAR, MEMBAR, DEPBAR, ERRBAR)", 0),
            "registers": u.get("REG", ""), "stack_bytes": u.get("STACK", ""), "local_bytes": u.get("LOCAL", ""), "cmem0_bytes": u.get("CONSTANT[0]", ""),
            "code_bytes": sizes.get(name, ""), "cubin_bytes_total": cubin_total,
            "block_threads": threads, "grid": "x".join(str(g) for g in k["grid"]),
            "theoretical_occupancy_pct": f"{api[1]:.2f}" if api else "", "active_blocks_per_sm": api[0] if api else "",
            "theoretical_occupancy_ncu_pct": f"{k['metrics'].get(M_THEORETICAL, float('nan')):.2f}",
            "achieved_occupancy_pct": f"{k['metrics'].get(M_ACHIEVED, float('nan')):.2f}",
            "ncu_duration_us": f"{k['metrics'].get(M_DURATION, float('nan')):.2f}",
        })
    order = {"hand_params": 0, "hand": 0, "hand_imm": 1, "hand_generic": 1, "hand_vec": 2, "fkl": 3, "fkl_tf": 4}
    rows.sort(key=lambda r: (r["point"], order.get(r["version"], 9)))
    if not rows:
        raise SystemExit("no kernel of the ncu run was found in the binary")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
