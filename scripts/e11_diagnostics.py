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
"""E11: what the compilers say about three deliberately wrong chains (fkl-self/diagnostics/e11_*.cu).

  python scripts/e11_diagnostics.py --compilers g++-13,clang++-23,/opt/rocm/core-10.1/bin/amdclang++ --label unchanged-c8fcbe8 --out results/rx9060/e11-diagnostics
  python scripts/e11_diagnostics.py --compilers nvcc --label unchanged-c8fcbe8 --out results/rtx4090/e11-diagnostics      (the NVIDIA machine)

For every translation unit and compiler: the correct chain (no -DE11_WRONG) must compile (control); the wrong chain (-DE11_WRONG) is compiled 5 times
with the flags of the E3 sweeps (-O3 -DNDEBUG -std=c++17 -ftemplate-depth=1000; amdclang++ -x hip --offload-arch=gfx1200; nvcc -arch sm_89; g++ and clang++ -march=native,
the CPU back-end) and once with the limit of one error (-fmax-errors=1 for g++, -ferror-limit=1 for clang based compilers; nvcc has none: its time is the total),
which is the time to the first error. Recorded per cell: the exit status, the number of lines, errors and notes of the diagnostics, whether a static_assert fired
(and its text), the first error line, and whether the names of the two operations involved appear in that line and in the first diagnostic and in the
headline: the error line plus the innermost context line (g++: the first "In instantiation of" line without its "[with ...]" part, which prints the whole list
of operations of the chain; clang based compilers: the first note after the error). The output of every compile of the wrong chain goes to logs/.
--include selects the FKL tree (default external/fkl/include): the same command with the include directory of a changed tree gives the "after" rows.
"""
import argparse
import csv
import datetime
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIAG = os.path.join(ROOT, "fkl-self", "diagnostics")
# the two operations involved in each mistake: (previous, next)
CASES = {
    "e11_type_mismatch": ("Mul", "Add"),
    "e11_read_in_middle": ("Mul", "PerThreadRead"),
    "e11_batch_mismatch": ("PerThreadRead", "PerThreadWrite"),
}
COLUMNS = ["date", "label", "note", "tu", "compiler", "version", "control_ok", "control_wall_s", "outcome", "exit", "lines", "errors", "notes", "static_assert", "first_error",
           "names_previous_op_in_error_line", "names_next_op_in_error_line", "names_both_in_headline", "points_to_user_line", "wall_s", "wall_first_error_s", "first_error_limit"]
ERR = re.compile(r"(?:^|[:\s)])(?:fatal )?error(?: #\d+-D)?:")
NOTE = re.compile(r"(?:^|[:\s)])note:")
SA = re.compile(r"static assertion failed|static_assert failed|static_assert")


def kind(cc):
    b = os.path.basename(cc)
    return "hip" if "amdclang" in b or "hipcc" in b else "cuda" if "nvcc" in b else "cpu"


def command(cc, tu, include, wrong, limit, offload_arch, cuda_arch, ccbin=None):
    k = kind(cc)
    base = ["-O3", "-DNDEBUG", "-std=c++17", "-ftemplate-depth=1000"]
    if k == "hip":
        cmd = [cc, "-x", "hip", f"--offload-arch={offload_arch}"] + base + (["-ferror-limit=1"] if limit else [])
    elif k == "cuda":
        cmd = [cc, "-x", "cu", "-arch", cuda_arch] + (["-ccbin", ccbin] if ccbin else []) + base
    else:
        cmd = [cc, "-x", "c++", "-march=native"] + base + ([("-fmax-errors=1" if "g++" in os.path.basename(cc) else "-ferror-limit=1")] if limit else [])
    return cmd + (["-DE11_WRONG"] if wrong else []) + [f"-I{include}", "-c", tu, "-o", os.devnull]


def run(cmd):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    return p.returncode, time.time() - t0, p.stdout + p.stderr


def short(text, include, limit=300):
    text = text.replace(os.path.abspath(include) + "/", "").replace(DIAG + "/", "")
    text = re.sub(r"/[^ :'\"]*/(?=fused_kernel/)", "", text)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def analyse(out, prev_name, next_name, tu_name):
    lines = out.splitlines()
    errs = [i for i, l in enumerate(lines) if ERR.search(l)]
    first = errs[0] if errs else None
    block = []
    if first is not None:
        end = errs[1] if len(errs) > 1 else len(lines)
        block = [l for l in lines[:end] if not l.startswith("In file included from") and not re.match(r"\s+from ", l)]
    sa = [l for l in lines if SA.search(l) and ERR.search(l)]
    inner = ""
    if first is not None:
        inner = next((l for l in block if "In instantiation of" in l), None) or next((l for l in lines[first + 1:] if NOTE.search(l)), "")
    head = (lines[first] + " " + re.sub(r"\[with .*$", "", inner)) if first is not None else ""
    return dict(both_in_headline=bool(head and prev_name in head and next_name in head), user_line=(tu_name + ":") in out, lines=len(lines), errors=len(errs), notes=sum(1 for l in lines if NOTE.search(l)), first_idx=first,
                first_error=lines[first] if first is not None else "", static_assert=sa[0] if sa else "",
                prev_in_line=bool(first is not None and prev_name in lines[first]), next_in_line=bool(first is not None and next_name in lines[first]),
                )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compilers", required=True, help="comma separated compilers")
    ap.add_argument("--label", required=True, help="the FKL tree, e.g. unchanged-c8fcbe8")
    ap.add_argument("--include", default=os.path.join(ROOT, "external", "fkl", "include"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--offload-arch", default="gfx1200")
    ap.add_argument("--cuda-arch", default="sm_89")
    ap.add_argument("--ccbin", help="nvcc: the host compiler (-ccbin)")
    ap.add_argument("--note", default="", help="free text written to the CSV, e.g. where and how the compiler was run")
    a = ap.parse_args()
    include = os.path.abspath(a.include)
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    path = os.path.join(a.out, "e11_diagnostics.csv")
    new = not os.path.exists(path)
    rows = []
    for cc in a.compilers.split(","):
        version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
        version = next((l for l in version if re.search(r"\d+\.\d+", l)), version[0] if version else "?")
        for name, (pn, nn) in CASES.items():
            tu = os.path.join(DIAG, name + ".cu")
            crc, cw, cout = run(command(cc, tu, include, False, False, a.offload_arch, a.cuda_arch, a.ccbin))
            runs = [run(command(cc, tu, include, True, False, a.offload_arch, a.cuda_arch, a.ccbin)) for _ in range(a.reps)]
            rc, _, out = runs[0]
            lim = [run(command(cc, tu, include, True, True, a.offload_arch, a.cuda_arch, a.ccbin)) for _ in range(a.reps)] if kind(cc) != "cuda" else runs
            an = analyse(out, pn, nn, os.path.basename(tu))
            outcome = "compiles, no diagnostic" if rc == 0 and not out.strip() else "compiles, with output" if rc == 0 else "static_assert" if an["static_assert"] else "error"
            open(os.path.join(a.out, "logs", f"{a.label}_{name}_{os.path.basename(cc)}.txt"), "w").write(short(out, include, 10 ** 9))
            row = dict(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label=a.label, note=a.note, tu=name, compiler=os.path.basename(cc), version=version,
                       control_ok="yes" if crc == 0 else f"NO (exit {crc})", control_wall_s=round(cw, 2), outcome=outcome, exit=rc, lines=an["lines"], errors=an["errors"],
                       notes=an["notes"], static_assert=short(an["static_assert"], include), first_error=short(an["first_error"], include),
                       names_previous_op_in_error_line="yes" if an["prev_in_line"] else "no", names_next_op_in_error_line="yes" if an["next_in_line"] else "no",
                       names_both_in_headline="yes" if an["both_in_headline"] else "no", points_to_user_line="yes" if an["user_line"] else "no", wall_s=round(statistics.median(r[1] for r in runs), 2),
                       wall_first_error_s=round(statistics.median(r[1] for r in lim), 2), first_error_limit="none (total time)" if kind(cc) == "cuda" else "1 error")
            rows.append(row)
            print(f"{a.label} {name} {os.path.basename(cc)}: control {row['control_ok']}, {outcome}, {an['lines']} lines, {an['errors']} errors, {an['notes']} notes, "
                  f"{row['wall_s']} s ({row['wall_first_error_s']} s to the first error) | {row['first_error'][:170]}", flush=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
