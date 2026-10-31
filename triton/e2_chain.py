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
"""E2 (a): the element-wise chain of the paper through Triton and TorchInductor.

The workload is the one of E1a / benchmark_vertical_fusionCompareMulAdd1: a 4096 x 2160 uint8 image converted to float32, scaled by
alpha = 1.0, then k operations alternating Mul by c and Add c (k/2 pairs, c = 0.54), written as float32. k is the number of
operations, the x axis of the paper. Versions:

  triton_hand        one hand-written Triton kernel, the chain unrolled at trace time (tl.static_range), the upper bound of this section
  eager              PyTorch eager: one kernel per operation (what a user gets without fusion)
  inductor_default   torch.compile(fn) with Inductor's default fusion window (max_fusion_size = 64)
  inductor_wide      torch.compile(fn) with the window raised so that the whole chain can fuse
  inductor_graphs    torch.compile(fn, mode="reduce-overhead"): Inductor plus CUDA Graphs, reported separately
  eager_graphs       eager PyTorch captured in one CUDA graph by hand (the fair comparison for inductor_graphs)

Per version and k: GPU time (CUDA events around one call; mean and min of 100 calls after warm-up and compilation), host launch time of
one call (no synchronisation; mean of 100), the number of CUDA kernels one call launches (torch.profiler), compile or autotune time
(first call), and the agreement of the output with an exact float64 reference (maximum error in float32 ULPs of the exact value, fraction
of bit-identical elements against triton_hand). Tools: the CUDA 13.4 ptxas must be the one Triton uses (source scripts/triton_env.sh;
common/fklb_csv.py writes the version it resolved in the CSV header).

  python triton/e2_chain.py [--ks 2,10,50,100,500] [--out e2_chain.csv] [--versions triton_hand,eager,...] [--timeout 1800]
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from fklb_csv import CsvWriter  # noqa: E402
import sass_tools as st  # noqa: E402

H, W = 2160, 4096
ALPHA = 1.0
C = 0.54
ITERATIONS = 100
BLOCK = 1024


@triton.jit
def _chain_kernel(x_ptr, out_ptr, n, alpha, c, PAIRS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask).to(tl.float32) * alpha
    for _ in tl.static_range(PAIRS):
        v = v * c
        v = v + c
    tl.store(out_ptr + offs, v, mask=mask)


def triton_hand(x, out, pairs):
    n = x.numel()
    _chain_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, ALPHA, C, PAIRS=pairs, BLOCK=BLOCK)
    return out


def make_eager(ops):
    def fn(x):
        v = x.to(torch.float32) * ALPHA
        for i in range(ops):
            v = v * C if i % 2 == 0 else v + C
        return v
    return fn


def make_versions(ops):
    fn = make_eager(ops)
    versions = {}
    versions["triton_hand"] = (lambda x, out: triton_hand(x, out, ops // 2), None)
    versions["eager"] = (lambda x, out: fn(x), None)
    versions["inductor_default"] = (lambda x, out, f=torch.compile(fn): f(x), {"max_fusion_size": 64})
    versions["inductor_wide"] = (lambda x, out, f=torch.compile(make_eager(ops)): f(x), {"max_fusion_size": 10**6})
    versions["inductor_graphs"] = (lambda x, out, f=torch.compile(make_eager(ops), mode="reduce-overhead"): f(x), {"max_fusion_size": 10**6})
    return versions


def cuda_time(call, x, out):
    """Mean and min GPU time (ms) of 100 calls, host time of one call (us), after the warm-up."""
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(ITERATIONS):
        start.record()
        call(x, out)
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        call(x, out)
    host_us = (time.perf_counter() - t0) / ITERATIONS * 1e6
    torch.cuda.synchronize()
    return sum(times) / len(times), min(times), host_us


FLUSH_BYTES = 256 << 20


def cuda_time_flushed(call, x, out, flush_buf):
    """Mean and min GPU time (ms) of 100 calls with the L2 flushed before each one: a kernel that READS a 256 MB buffer (more than three times the 72 MB L2 of the RTX 4090; reading leaves
    clean lines, so the output of the previous call is written back during the flush) runs on the same stream before the start event, outside the timed interval."""
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(ITERATIONS):
        flush_buf.sum()
        start.record()
        call(x, out)
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return sum(times) / len(times), min(times)


def count_kernels(call, x, out):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call(x, out)
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and "Memcpy" not in e.name and "Memset" not in e.name)


def ulp_error(result, exact64):
    """Max |result - exact| in float32 ULPs of the exact value, over the tensor."""
    r = result.double()
    ulp = torch.nextafter(exact64.float(), torch.tensor(float("inf"), device=exact64.device)).double() - exact64.float().double()
    return ((r - exact64).abs() / ulp.abs().clamp_min(1e-300)).max().item()


def sass_counts(cache_dir):
    """FFMA, FMUL and FADD instructions of every kernel Triton compiled in this process (TorchInductor's kernels are Triton kernels too): tells whether the Mul and the
    Add of the chain were contracted into one FMA (FFMA) or kept as two instructions (FMUL + FADD). cuobjdump -sass of every cubin of the cache."""
    import glob
    totals = {"sass_ffma": 0, "sass_fmul": 0, "sass_fadd": 0}
    if not cache_dir:
        return totals
    cubins = glob.glob(os.path.join(cache_dir, "**", "*.cubin"), recursive=True)
    # TorchInductor compiles several configurations of a pointwise kernel (here 8 and 2 elements per thread) and keeps the fastest: its *.best_config names the cache entry of the one that
    # runs. Only that kernel is counted; summing every cubin of the cache would add the configurations that were compiled and not used.
    best = set()
    for bc in glob.glob(os.path.join(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/nonexistent"), "**", "*.best_config"), recursive=True):
        with open(bc) as f:
            best.add(json.load(f)["triton_cache_hash"])
    if best:
        cubins = [c for c in cubins if os.path.basename(os.path.dirname(c)) in best]
    for cubin in cubins:
        sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True).stdout
        for body in st.split_functions(sass).values():
            _, hist = st.classify(st.normalise_sass(body, rename=False))
            totals["sass_ffma"] += hist.get("FFMA", 0)
            totals["sass_fmul"] += hist.get("FMUL", 0)
            totals["sass_fadd"] += hist.get("FADD", 0)
    return totals


def load_input(path):
    """The image: the raw uint8 file written by handwritten-cuda/e2_fkl_chain.cu (the bytes FKL ran on), or torch.randint with seed 20261030 when no file is given."""
    if path:
        a = np.fromfile(path, dtype=np.uint8)
        if a.size != H * W:
            raise SystemExit(f"{path}: {a.size} bytes, expected {H * W}")
        return torch.from_numpy(a.reshape(H, W).copy()).cuda()
    torch.manual_seed(20261030)
    return torch.randint(0, 256, (H, W), dtype=torch.uint8, device="cuda")


def ulp_diff(result, reference):
    """Max |result - reference| in float32 ULPs of the reference value (reference: a float32 tensor)."""
    ulp = torch.nextafter(reference, torch.tensor(float("inf"), device=reference.device)).double() - reference.double()
    return ((result.double() - reference.double()).abs() / ulp.abs().clamp_min(1e-300)).max().item()


def worker(version, ops, ref_path, input_path=None, fkl_dir=None):
    """One (version, ops) in this process: cold compile time (the caches are fresh), steady GPU time, kernels, accuracy. Returns a dict."""
    x = load_input(input_path)
    out = torch.empty((H, W), dtype=torch.float32, device="cuda")
    res = {}
    fkl_path = os.path.join(fkl_dir, f"fkl_out_k{ops}.f32") if fkl_dir else None
    if version == "fkl":    # not run here: its output (written by e2_fkl_chain) is evaluated like the others; time and compile time come from the FKL CSVs (the driver)
        if not fkl_path or not os.path.exists(fkl_path):
            raise SystemExit(f"no FKL output {fkl_path}")
        res_t = torch.from_numpy(np.fromfile(fkl_path, dtype=np.float32).reshape(H, W)).cuda()
        exact = x.double() * ALPHA
        for i in range(ops):
            exact = exact * C if i % 2 == 0 else exact + C
        res["max_ulp_vs_exact"] = ulp_error(res_t, exact)
        res["bitexact_fraction_vs_fkl"], res["max_ulp_vs_fkl"] = 1.0, 0.0
        if os.path.exists(ref_path):
            res["bitexact_fraction_vs_triton_hand"] = (res_t.cpu() == torch.load(ref_path)).float().mean().item()
        res["kernels"] = 1
        return res
    if version == "eager_graphs":
        fn = make_eager(ops)
        t0 = time.perf_counter()
        with torch.cuda.stream(torch.cuda.Stream()):
            fn(x)   # warm-up outside the capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = fn(x)
        res["compile_s"] = time.perf_counter() - t0
        call = lambda x_, out_: g.replay()
        result = captured
    else:
        call, cfg = make_versions(ops)[version]
        for key, val in (cfg or {}).items():
            setattr(inductor_config, key, val)
        if version == "inductor_graphs":
            torch._dynamo.mark_static_address(x)    # no copy of the input into the graph's static buffer
        t0 = time.perf_counter()
        r = call(x, out)
        torch.cuda.synchronize()
        res["compile_s"] = time.perf_counter() - t0
        for _ in range(3):
            r = call(x, out)   # warm-up (CUDA graphs need a few runs)
        torch.cuda.synchronize()
        result = out if version == "triton_hand" else r
    mean, mn, host = cuda_time(call, x, out)
    res.update(gpu_mean_ms=mean, gpu_min_ms=mn, host_launch_us=host)
    flush_buf = torch.zeros(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    res["gpu_flushed_mean_ms"], res["gpu_flushed_min_ms"] = cuda_time_flushed(call, x, out, flush_buf)
    del flush_buf
    res["kernels"] = count_kernels(call, x, out) if version != "eager_graphs" else count_kernels(lambda a, b: g.replay(), x, out)
    if version not in ("eager", "eager_graphs"):    # eager runs PyTorch's own precompiled kernels, one per operation: nothing is contracted across operations
        res.update(sass_counts(os.environ.get("TRITON_CACHE_DIR")))
    res_t = result.clone()
    if fkl_path and os.path.exists(fkl_path):    # agreement with FKL, bit by bit and in ULPs of the FKL value
        fkl = torch.from_numpy(np.fromfile(fkl_path, dtype=np.float32).reshape(H, W)).cuda()
        res["bitexact_fraction_vs_fkl"] = (res_t == fkl).float().mean().item()
        res["max_ulp_vs_fkl"] = ulp_diff(res_t, fkl)
    exact = x.double() * ALPHA
    for i in range(ops):
        exact = exact * C if i % 2 == 0 else exact + C
    res["max_ulp_vs_exact"] = ulp_error(res_t, exact)
    if version == "triton_hand":
        torch.save(res_t.cpu(), ref_path)
        res["bitexact_fraction_vs_triton_hand"] = 1.0
    elif os.path.exists(ref_path):
        res["bitexact_fraction_vs_triton_hand"] = (res_t.cpu() == torch.load(ref_path)).float().mean().item()
    else:
        res["bitexact_fraction_vs_triton_hand"] = float("nan")
    return res


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="2,10,50,100,500", help="chain lengths; add 1000 (optional) and one bounded attempt at a large k such as 5002 with --timeout 900 to document the limit")
    ap.add_argument("--out", default="e2_chain.csv")
    ap.add_argument("--versions", default="triton_hand,eager,inductor_default,inductor_wide,inductor_graphs,eager_graphs")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per (version, ops): a worker that is slower is killed and recorded")
    ap.add_argument("--worker", nargs=2, metavar=("VERSION", "OPS"), help="internal: run one point in this process and print a JSON line")
    ap.add_argument("--ref", default="triton_ref.pt")
    ap.add_argument("--input", default=None, help="raw uint8 image 4096x2160 (written by handwritten-cuda e2_fkl_chain); default: torch.randint with seed 20261030")
    ap.add_argument("--fkl-dir", default=None, help="directory with fkl_out_k<k>.f32 of e2_fkl_chain: adds the FKL row and the agreement columns against FKL")
    ap.add_argument("--fkl-csv", default=None, help="e2_fkl_chain.csv (GPU time of FKL)")
    ap.add_argument("--fkl-compile", default=None, help="CSV with the columns ops, nvcc_s (cold compile time of the FKL chain; scripts/e2_fkl_compile.py)")
    args = ap.parse_args()
    if args.worker:
        print("E2JSON " + json.dumps(worker(args.worker[0], int(args.worker[1]), args.ref, args.input, args.fkl_dir)))
        return
    ks = [int(k) for k in args.ks.split(",")]
    image = (f"the file {args.input} (written by e2_fkl_chain: std::mt19937_64 seed 20261030, as E1a)" if args.input else "torch.randint, torch.manual_seed 20261030")
    notes = (f"k values: {','.join(str(k) for k in ks)}; 4096x2160 uint8 noise ({image}) -> float32, alpha {ALPHA}, c {C}, k operations alternating mul and add; each (version, k) in its own process "
             f"with fresh Triton and Inductor caches (compile_s = the first call, cold); {ITERATIONS} timed calls after warm-up; kernels counted with torch.profiler; "
             f"gpu_flushed_*: the same timing with the L2 flushed before every call (a kernel that reads 256 MB, outside the timed interval); "
             f"ULP error against an exact float64 evaluation (max_ulp_vs_exact) and against the FKL output (max_ulp_vs_fkl, in ULPs of the FKL value; bitexact_fraction_vs_fkl); "
             f"FKL row: time from {args.fkl_csv}, cold compile time from {args.fkl_compile}; timeout {args.timeout} s per point")
    cols = ["ops", "version", "status", "why", "gpu_mean_ms", "gpu_min_ms", "host_launch_us", "kernels", "compile_s", "max_ulp_vs_exact", "bitexact_fraction_vs_triton_hand",
            "bitexact_fraction_vs_fkl", "max_ulp_vs_fkl", "sass_ffma", "sass_fmul", "sass_fadd", "gpu_flushed_mean_ms", "gpu_flushed_min_ms"]
    with CsvWriter(args.out, "E2a element-wise chain: Triton and TorchInductor", cols, notes=notes) as csv:
        versions = args.versions.split(",")
        if args.fkl_dir and "fkl" not in versions:
            versions.insert(min(1, len(versions)), "fkl")     # after triton_hand, whose output the FKL row is also compared with
        fkl_times = {int(r["ops"]): r for r in read_rows(args.fkl_csv)} if args.fkl_csv else {}
        fkl_compile = {int(r["ops"]): r for r in read_rows(args.fkl_compile)} if args.fkl_compile else {}
        for ops in ks:
            ref = os.path.join(tempfile.gettempdir(), f"e2_triton_ref_{os.getpid()}_{ops}.pt")
            for version in versions:
                caches = tempfile.mkdtemp(prefix="e2cache")
                env = dict(os.environ, TRITON_CACHE_DIR=os.path.join(caches, "triton"), TORCHINDUCTOR_CACHE_DIR=os.path.join(caches, "inductor"),
                           TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_AUTOGRAD_CACHE="0")
                cmd = [sys.executable, os.path.abspath(__file__), "--worker", version, str(ops), "--ref", ref]
                if args.input:
                    cmd += ["--input", args.input]
                if args.fkl_dir:
                    cmd += ["--fkl-dir", args.fkl_dir]
                t0 = time.perf_counter()
                try:
                    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout, start_new_session=True)
                    line = [l for l in p.stdout.splitlines() if l.startswith("E2JSON ")]
                    if p.returncode == 0 and line:
                        r = json.loads(line[-1][7:])
                        status, why = "ok", ""
                    else:
                        r, status = {}, f"failed (exit {p.returncode})"
                        why = (p.stderr.strip().splitlines() or [""])[-1][:300]
                except subprocess.TimeoutExpired:
                    r, status, why = {}, f"timeout ({args.timeout} s)", ""
                shutil.rmtree(caches, ignore_errors=True)
                if version == "fkl" and status == "ok":   # the time of FKL and the cold compile time of its chain come from their own programs
                    if ops in fkl_times:
                        r["gpu_mean_ms"], r["gpu_min_ms"] = float(fkl_times[ops]["fkl_mean_ms"]), float(fkl_times[ops]["fkl_min_ms"])
                        if fkl_times[ops].get("fkl_flushed_mean_ms"):
                            r["gpu_flushed_mean_ms"], r["gpu_flushed_min_ms"] = float(fkl_times[ops]["fkl_flushed_mean_ms"]), float(fkl_times[ops]["fkl_flushed_min_ms"])
                    if ops in fkl_compile:
                        r["compile_s"] = float(fkl_compile[ops]["nvcc_s"])
                csv.row(ops, version, status, why, *[r.get(c, "") for c in cols[4:]])
                nan = float("nan")
                summary = (f"gpu {r.get('gpu_mean_ms', nan):9.4f} ms host {r.get('host_launch_us', nan):9.1f} us kernels {int(r.get('kernels', 0)):5d} compile {r.get('compile_s', nan):8.2f} s "
                           f"ulp_exact {r.get('max_ulp_vs_exact', nan):.3g} ulp_fkl {r.get('max_ulp_vs_fkl', nan):.3g} bitexact_fkl {r.get('bitexact_fraction_vs_fkl', nan):.4f}" if r else why)
                print(f"k={ops:6d} {version:17s} {status:18s} {summary}  [{time.perf_counter() - t0:.0f} s]", flush=True)
            if os.path.exists(ref):
                os.remove(ref)


if __name__ == "__main__":
    main()
