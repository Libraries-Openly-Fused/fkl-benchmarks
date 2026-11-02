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
"""E3, the limits of the clang based compilers: where a fused chain of N distinct operations stops compiling.

One translation unit per compile (variant A of fkl-self/compiletime/gen_chain.py: float, Mul/Add/Sub/Div/Max/Min cycled, each operation with its own parameter,
all in ONE fused kernel), compiled with amdclang++ (HIP, one gfx target), clang++ (the CPU back-end) or g++ (the CPU back-end, for comparison), with the flags of
scripts/e3_sweep.py (-O3 -DNDEBUG -std=c++17 -ftemplate-depth=1000). One compile at a time, under a monitor that samples the resident memory of the whole process
tree at 4 Hz and kills it at a cap (--cap-gb, 40 GiB) or a time limit (--timeout, 3600 s) and says so. run and bisect never execute the programs; check does (below).
The memory figures are GiB (GNU time's KiB and the sampled VmRSS in kB, divided by 1,048,576; the cap is cap_gb * 1,048,576 kB): the CSV columns peak_rss_gb and peak_tree_rss_gb are in GiB, object_mb is in MiB.

  python scripts/e3_clang_limits.py run    --backend hip   --ns 1100,2046,2047 --out results/rx9060/e3-compile-limits
  python scripts/e3_clang_limits.py run    --backend clang --ns 2100 --bracket-depth auto --stack-kb unlimited --out ...
  python scripts/e3_clang_limits.py bisect --backend hip   --lo 1100 --hi 2046 --no-aslr --out ...
  python scripts/e3_clang_limits.py check  --backend hip   --ns 1100,2046 --stack-kb 32768 --bracket-depth auto --out ...

check builds the program of the same translation unit (-DE3_CHECK: 64 x 64 floats of a deterministic pattern through the chain, written to a file) for the GPU with
amdclang++ and for the CPU back-end with g++, runs both and compares the outputs (scripts/e3_check_compare.py): the question is whether the kernel with N operations
LAUNCHES (its kernel argument block is larger than 4 KiB from N = 1,000) and computes the same. The check chain is variant A with Mul and Add in place of Max and Min
(which clamp the values and hide the early part of the chain) and parameters that keep the values finite, without FMA contraction on either side. One row per N is appended to <out>/e3_check.csv
(a program that does not finish in 600 s gets its row with run exit "timeout"; compare_exit is 0 only when the two outputs are byte for byte identical; check needs --backend hip).

One CSV row per compile is appended to <out>/e3_limits.csv; the log of every compile that did not succeed (the first 400 lines) goes to <out>/logs/.
The HIP compiles ask for the resource usage remark of the kernel (-Rpass-analysis=kernel-resource-usage) unless --no-remarks is given; it costs time and memory (section 3 of the README).
The outcome is one of: ok, error (a diagnostic; the first one is in first_error), crash (the compiler died of a signal or printed its crash dump: first_error has
the declaration it was generating code for, if it said), cut (killed by the monitor). A cut says nothing about whether the compiler would have succeeded or failed on that N,
so bisect stops with an error when a compile is cut; the cut row is in e3_limits.csv when --out is given.
--stack-kb sets the stack limit of the compiler (RLIMIT_STACK; default: whatever the calling shell has, 8192 kB); --no-aslr runs it with the address space
randomisation off (setarch -R), which makes a stack overflow happen at the same N every time; --bracket-depth N|auto adds -fbracket-depth (auto: N + 2, the
smallest value that lifts the limit on the fold expression over the N operations and the read and write of the chain).
"""
import argparse
import csv
import datetime
import gzip
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEN = os.path.join(ROOT, "fkl-self", "compiletime", "gen_chain.py")
COLUMNS = ["date", "series", "backend", "compiler", "fkl", "n", "bracket_depth", "stack_kb", "aslr", "extra_flags", "remarks", "outcome", "exit", "wall_s", "user_s", "sys_s",
           "peak_rss_gb", "peak_tree_rss_gb", "object_mb", "sgpr", "vgpr", "scratch_bytes_per_lane", "occupancy", "sgpr_spills", "vgpr_spills", "first_error"]


def tree_rss_kb(pid):
    """Sum of VmRSS (kB) of pid and its descendants."""
    total, todo, seen = 0, [pid], set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            for line in open(f"/proc/{p}/status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
            with open(f"/proc/{p}/task/{p}/children") as f:
                todo += [int(x) for x in f.read().split()]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return total


def runner_busy():
    return subprocess.run(["pgrep", "-f", "[R]unner.Worker"], stdout=subprocess.DEVNULL).returncode == 0


def short(text, limit=260):
    text = re.sub(r"/[^ :'\"]*/(?=(?:fused_kernel|include)/)", "", text)
    text = re.sub(r"/[^ :'\"]*/(chain_\d+\.cu)", r"\1", text)
    return text if len(text) <= limit else text[:limit - 3] + "..."


class Setup:
    def __init__(self, a, backend=None, compiler=None):
        self.a = a
        self.backend = backend or a.backend
        self.inc = os.path.abspath(a.fkl_include or os.path.join(ROOT, "external", "fkl", "include"))
        self.pool = os.path.join(os.path.dirname(self.inc), "thread-pool", "include")
        if self.backend == "hip":
            self.cc = compiler or a.compiler or os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "amdclang++")
        elif self.backend == "clang":
            self.cc = compiler or a.compiler or "clang++-23"
        else:
            self.cc = compiler or (a.compiler if a.backend == "gcc" else None) or "g++-13"
        self.version = subprocess.run([self.cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        try:
            self.fkl = subprocess.run(["git", "-C", self.inc, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "?"
        except OSError:
            self.fkl = "?"
        self.work = a.work or tempfile.mkdtemp(prefix="e3limits_")

    def command(self, n, bracket, check=False):
        """The compile command and the file it writes: an object, or with check the program of the translation unit (-DE3_CHECK)."""
        a = self.a
        d = os.path.join(self.work, f"{self.backend}_{n}")
        os.makedirs(d, exist_ok=True)
        tu, obj = os.path.join(d, f"chain_{n}.cu"), os.path.join(d, f"chain_{n}.exe" if check else f"chain_{n}.o")
        subprocess.run([sys.executable, GEN, "A", str(n), tu], check=True)
        src = open(tu).read()
        if check:
            # the Max and Min of variant A clamp every value after the first few operations, and its parameters (1 + 0.000123 (i + 1)) make the product of
            # the Mul and Div overflow from N = 2,900: either way the output would hide most of the chain. The check program has Mul and Add in place of Max and
            # Min and parameters 1 + 0.0001 ((i mod 13) + 1) (Mul, Div) and 1 + 0.001 ((i mod 13) + 1) (Add, Sub), which keep the values finite (about N / 6 at the
            # end) and make the output depend on every parameter; no contraction of a Mul and an Add into an FMA on either side: bit-identical results expected.
            src = src.replace("fk::Max<float>::build(", "fk::Mul<float>::build(").replace("fk::Min<float>::build(", "fk::Add<float>::build(")
            count = [0]

            def param(m):
                i = count[0]
                count[0] += 1
                scale = 0.0001 if m.group(1) in ("Mul", "Div") else 0.001
                return f"fk::{m.group(1)}<float>::build({1.0 + scale * ((i % 13) + 1):.6f}f)"
            src = re.sub(r"fk::(Mul|Add|Sub|Div)<float>::build\([0-9.]+f\)", param, src)
        if self.backend == "hip":
            src = src.replace("GPU_NVIDIA", "GPU_AMD").replace("cudaMemset", "hipMemset")
        open(tu, "w").write(src)
        if self.backend == "hip":
            cmd = [self.cc, "-x", "hip", f"--offload-arch={a.offload_arch}", "-O3", "-DNDEBUG", "-std=c++17", "-ftemplate-depth=1000"]
            cmd += ["-DE3_CHECK", "-ffp-contract=off"] if check else ([] if a.no_remarks else ["-Rpass-analysis=kernel-resource-usage"])
        else:
            cmd = [self.cc, "-x", "c++", "-DE3_CPU", "-O3", "-DNDEBUG", "-std=c++17", "-march=native", "-ftemplate-depth=1000", f"-I{self.pool}"]
            cmd += ["-DE3_CHECK", "-ffp-contract=off"] if check else []
        if bracket and self.backend != "gcc":
            cmd.append(f"-fbracket-depth={bracket}")
        cmd += a.flags.split() + [f"-I{self.inc}"] + ([tu] if check else ["-c", tu]) + ["-o", obj]
        return cmd, obj


def hip_resources(err):
    """The resource usage remark of the fused kernel (launchTransformDPP_Kernel)."""
    for block in re.split(r"Function Name: ", err)[1:]:
        if "launchTransformDPP_Kernel" not in block.split("\n")[0]:
            continue
        def g(key):
            m = re.search(re.escape(key) + r":\s*(\d+)", block)
            return int(m.group(1)) if m else ""
        return dict(sgpr=g("TotalSGPRs"), vgpr=g("VGPRs"), scratch_bytes_per_lane=g("ScratchSize [bytes/lane]"), occupancy=g("Occupancy [waves/SIMD]"),
                    sgpr_spills=g("SGPRs Spill"), vgpr_spills=g("VGPRs Spill"))
    return {}


def compile_once(s, n, series, check=False):
    a = s.a
    bracket = (n + 2 if a.bracket_depth == "auto" else int(a.bracket_depth)) if a.bracket_depth else 0
    cmd, obj = s.command(n, bracket, check)
    stack = a.stack_kb
    soft = resource.RLIM_INFINITY if stack == "unlimited" else (int(stack) * 1024 if stack else resource.getrlimit(resource.RLIMIT_STACK)[0])
    stack_kb = "unlimited" if soft == resource.RLIM_INFINITY else soft // 1024
    tfile = tempfile.mktemp(prefix="e3time_")
    full = ["/usr/bin/time", "-f", "%e %U %S %M", "-o", tfile] + (["setarch", "-R"] if a.no_aslr else []) + cmd

    def prep():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_STACK, (soft, resource.getrlimit(resource.RLIMIT_STACK)[1]))
    while runner_busy():
        print("a CI job is running on this machine: waiting", file=sys.stderr)
        time.sleep(30)
    t0 = time.time()
    # TMPDIR: when amdclang++ crashes it writes a reproducer of about 12 MB to the temporary directory; that goes to the work directory, which is removed at the end
    p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", preexec_fn=prep, env=dict(os.environ, TMPDIR=s.work))
    buf = []
    th = threading.Thread(target=lambda: buf.extend(p.stdout))
    th.start()
    peak, cut = 0, None
    while p.poll() is None:
        rss = tree_rss_kb(p.pid)
        peak = max(peak, rss)
        if rss > a.cap_gb * 1048576:
            cut = "rss cap"
        elif time.time() - t0 > a.timeout:
            cut = "time limit"
        if cut:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            break
        time.sleep(0.25)
    p.wait()
    th.join()
    wall = time.time() - t0
    out = "".join(buf)
    status, sig, nums = 0, 0, [0.0, 0.0, 0.0, 0.0]
    try:
        lines = [l.strip() for l in open(tfile) if l.strip()]
        os.unlink(tfile)
        for l in lines:
            if l.startswith("Command terminated by signal"):
                sig = int(l.split()[-1])
            elif l.startswith("Command exited with non-zero status"):
                status = int(l.split()[-1])
        nums = [float(x) for x in lines[-1].split()]
    except (OSError, ValueError, IndexError):
        pass
    rc = -sig if sig else status
    if cut:
        outcome, first = "cut", f"killed at the {cut}"
    elif rc == 0:
        outcome, first = "ok", ""
    elif sig or "PLEASE submit a bug report" in out or "failed due to signal" in out:
        ctx = re.search(r"^\d+\.\s+(.*Generating code for declaration.*)$", out, re.M)
        outcome = "crash"
        first = ("signal %d" % sig if sig else "crash dump") + (" while: " + short(ctx.group(1), 300) if ctx else (", no message" if not out.strip() else ""))
    else:
        outcome = "error"
        m = re.search(r"^.*(?:fatal error|error):.*$", out, re.M)
        first = short(m.group(0)) if m else short(out.strip().splitlines()[-1] if out.strip() else "")
    row = dict(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), series=series, backend=s.backend, compiler=s.version, fkl=s.fkl, n=n,
               bracket_depth=bracket or "default", stack_kb=stack_kb, aslr="off" if a.no_aslr else "on", extra_flags=a.flags,
               remarks="kernel-resource-usage" if s.backend == "hip" and not check and not a.no_remarks else "", outcome=outcome, exit=rc,
               wall_s=round(wall, 1), user_s=nums[1], sys_s=nums[2], peak_rss_gb=round(nums[3] / 1048576, 2), peak_tree_rss_gb=round(peak / 1048576, 2),
               object_mb=round(os.path.getsize(obj) / 1048576, 1) if os.path.exists(obj) else "", first_error=first)
    if s.backend == "hip" and not check:
        row.update(hip_resources(out))
    row["_path"] = obj
    if os.path.exists(obj) and not check:
        os.unlink(obj)
    if outcome != "ok" and a.out:
        os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
        name = os.path.join(a.out, "logs", f"{s.backend}_N{n}{'_check' if check else ''}_{outcome}_stack{stack_kb}_{'noaslr' if a.no_aslr else 'aslr'}_{int(time.time())}.log.gz")
        ls = short(out, 10 ** 9).splitlines()
        with gzip.open(name, "wt") as f:
            f.write("\n".join(ls[:400]) + (f"\n... ({len(ls) - 400} more lines)\n" if len(ls) > 400 else "\n"))
    if a.out and not check:
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, "e3_limits.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, COLUMNS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in COLUMNS})
    res = f" sgpr {row.get('sgpr', '')} vgpr {row.get('vgpr', '')} scratch {row.get('scratch_bytes_per_lane', '')}" if s.backend == "hip" and outcome == "ok" and not check else ""
    print(f"{s.backend}{' program' if check else ''} N={n} stack={stack_kb} aslr={row['aslr']} bracket={row['bracket_depth']}: {outcome} (exit {rc}) wall {row['wall_s']} s, peak {row['peak_rss_gb']} GiB{res}"
          + (f" | {first}" if first else ""), flush=True)
    return row


def kernel_metadata(exe, rocm, arch):
    """kernarg_segment_size, vgpr_count, sgpr_count and private_segment_fixed_size of the fused kernel in the code object of a HIP program."""
    llvm = os.path.join(rocm, "llvm", "bin")
    with tempfile.TemporaryDirectory() as d:
        fat, code = os.path.join(d, "fat.bin"), os.path.join(d, "code.o")
        subprocess.run([os.path.join(llvm, "llvm-objcopy"), "--dump-section", ".hip_fatbin=" + fat, exe, os.devnull], check=True, capture_output=True)
        subprocess.run([os.path.join(llvm, "clang-offload-bundler"), "--type=o", "--unbundle", "--input=" + fat, f"--targets=hip-amdgcn-amd-amdhsa--{arch}",
                        "--output=" + code], check=True, capture_output=True)
        notes = subprocess.run([os.path.join(llvm, "llvm-readobj"), "--notes", code], capture_output=True, text=True, check=True).stdout
    for block in re.split(r"\n  - (?=\.)", notes.split("amdhsa.kernels:", 1)[-1] if "amdhsa.kernels:" in notes else notes):
        if "launchTransformDPP_Kernel" in block:
            def g(key):
                m = re.search(re.escape(key) + r":\s*(\d+)", block)
                return int(m.group(1)) if m else ""
            return dict(kernarg_bytes=g(".kernarg_segment_size"), vgpr=g(".vgpr_count"), sgpr=g(".sgpr_count"), scratch=g(".private_segment_fixed_size"))
    return {}


CHECK_COLUMNS = ["date", "series", "compiler", "reference_compiler", "fkl", "n", "bracket_depth", "stack_kb", "build_outcome", "build_wall_s", "kernarg_bytes", "vgpr",
                 "sgpr", "scratch", "gpu_run_exit", "cpu_run_exit", "compare_exit", "compare"]


def check_once(s, ref, n, series):
    """Build the check program for the GPU and for the CPU back-end, run both, compare the outputs."""
    a = s.a
    g = compile_once(s, n, series, check=True)
    row = dict(date=g["date"], series=series, compiler=g["compiler"], reference_compiler=ref.version, fkl=s.fkl, n=n, bracket_depth=g["bracket_depth"],
               stack_kb=g["stack_kb"], build_outcome=g["outcome"], build_wall_s=g["wall_s"])
    if g["outcome"] == "ok":
        row.update(kernel_metadata(g["_path"], os.environ.get("ROCM_PATH", "/opt/rocm"), a.offload_arch))
        c = compile_once(ref, n, series, check=True)
        if c["outcome"] == "ok":
            outs = {}
            for tag, exe in (("gpu", g["_path"]), ("cpu", c["_path"])):
                outs[tag] = exe + ".bin"
                try:
                    r = subprocess.run([exe, outs[tag]], capture_output=True, text=True, timeout=600)
                except subprocess.TimeoutExpired:
                    row[f"{tag}_run_exit"] = "timeout"
                    row["compare"] = "; ".join(x for x in (row.get("compare", ""), f"the {tag} program was killed after 600 s") if x)
                    print(f"  the {tag} program did not finish in 600 s")
                    continue
                row[f"{tag}_run_exit"] = r.returncode
                if r.returncode != 0:
                    print(f"  the {tag} program exited with {r.returncode}: {r.stderr.strip()[:300]}")
            if row["gpu_run_exit"] == 0 and row["cpu_run_exit"] == 0:
                r = subprocess.run([sys.executable, os.path.join(HERE, "e3_check_compare.py"), outs["gpu"], outs["cpu"]], capture_output=True, text=True)
                row["compare_exit"], row["compare"] = r.returncode, r.stdout.strip()
        else:
            row["compare"] = "the CPU reference did not build: " + c["first_error"]
        for p in (g["_path"], c["_path"]):
            for f in (p, p + ".bin"):
                if os.path.exists(f):
                    os.unlink(f)
    if os.path.exists(g["_path"]):
        os.unlink(g["_path"])
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, "e3_check.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, CHECK_COLUMNS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in CHECK_COLUMNS})
    print(f"check {s.backend} N={n}: build {row['build_outcome']}, kernarg {row.get('kernarg_bytes', '')} B, gpu run exit {row.get('gpu_run_exit', '')}, cpu run exit {row.get('cpu_run_exit', '')}, "
          f"compare exit {row.get('compare_exit', '')}: {row.get('compare', '')}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["run", "bisect", "check"])
    ap.add_argument("--backend", choices=["hip", "clang", "gcc"], required=True)
    ap.add_argument("--ns", help="run, check: comma separated numbers of operations")
    ap.add_argument("--reps", type=int, default=1, help="run: repetitions of each N")
    ap.add_argument("--lo", type=int, help="bisect: an N that compiles")
    ap.add_argument("--hi", type=int, help="bisect: an N that does not")
    ap.add_argument("--series", default="", help="a label written to the CSV")
    ap.add_argument("--stack-kb", help="stack limit of the compiler in kB or 'unlimited' (default: the limit of the calling shell)")
    ap.add_argument("--no-aslr", action="store_true", help="run the compiler with address space randomisation off (setarch -R)")
    ap.add_argument("--bracket-depth", help="-fbracket-depth: a number or 'auto' (N + 2); clang only")
    ap.add_argument("--flags", default="", help="more compiler flags")
    ap.add_argument("--no-remarks", action="store_true", help="hip: do not ask for the resource usage remark (-Rpass-analysis=kernel-resource-usage), which costs time and memory")
    ap.add_argument("--compiler", help="the compiler (default: $ROCM_PATH/bin/amdclang++, clang++-23, g++-13)")
    ap.add_argument("--offload-arch", default="gfx1200")
    ap.add_argument("--reference-compiler", default="g++-13", help="check: the compiler of the CPU back-end program that the GPU output is compared with")
    ap.add_argument("--fkl-include", help="FKL include dir (default: external/fkl/include)")
    ap.add_argument("--cap-gb", type=float, default=40.0, help="cap of the resident memory of the process tree, in GiB")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--work", help="directory for the translation units (default: a temporary one)")
    ap.add_argument("--out", help="directory of e3_limits.csv and logs/")
    a = ap.parse_args()
    if a.mode == "check" and a.backend != "hip":
        ap.error("check compares a GPU program with the CPU reference and needs --backend hip")
    s = Setup(a)
    try:
        if a.mode == "run":
            for n in [int(x) for x in a.ns.split(",")]:
                for _ in range(a.reps):
                    compile_once(s, n, a.series)
        elif a.mode == "check":
            ref = Setup(a, backend="gcc", compiler=a.reference_compiler)
            ref.work = s.work
            for n in [int(x) for x in a.ns.split(",")]:
                check_once(s, ref, n, a.series)
        else:
            def compiles(n):
                row = compile_once(s, n, a.series)
                if row["outcome"] == "cut":
                    sys.exit(f"bisect: the compile at N = {n} was cut by the monitor ({row['first_error']}): no boundary is established; raise --cap-gb or --timeout")
                return row["outcome"] == "ok"
            lo, hi = a.lo, a.hi
            if not compiles(lo) or compiles(hi):
                sys.exit("bisect needs an N that compiles (--lo) and an N that does not (--hi)")
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if compiles(mid) else (lo, mid)
            print(f"{a.backend}: largest N that compiles {lo}, first N that does not {hi}")
    finally:
        if not a.work:
            shutil.rmtree(s.work, ignore_errors=True)


if __name__ == "__main__":
    main()
