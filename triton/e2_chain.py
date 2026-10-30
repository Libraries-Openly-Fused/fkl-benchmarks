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

  python triton/e2_chain.py [--ks 2,10,50,100,500] [--out e2_chain.csv] [--versions triton_hand,eager,...]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from fklb_csv import CsvWriter  # noqa: E402

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


def worker(version, ops, ref_path):
    """One (version, ops) in this process: cold compile time (the caches are fresh), steady GPU time, kernels, accuracy. Returns a dict."""
    torch.manual_seed(20261030)
    x = torch.randint(0, 256, (H, W), dtype=torch.uint8, device="cuda")
    out = torch.empty((H, W), dtype=torch.float32, device="cuda")
    res = {}
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
    res["kernels"] = count_kernels(call, x, out) if version != "eager_graphs" else count_kernels(lambda a, b: g.replay(), x, out)
    res_t = result.clone()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="2,102,1002,5002,10002,19902")
    ap.add_argument("--out", default="e2_chain.csv")
    ap.add_argument("--versions", default="triton_hand,eager,inductor_default,inductor_wide,inductor_graphs,eager_graphs")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per (version, ops): a worker that is slower is killed and recorded")
    ap.add_argument("--worker", nargs=2, metavar=("VERSION", "OPS"), help="internal: run one point in this process and print a JSON line")
    ap.add_argument("--ref", default="triton_ref.pt")
    args = ap.parse_args()
    if args.worker:
        print("E2JSON " + json.dumps(worker(args.worker[0], int(args.worker[1]), args.ref)))
        return
    ks = [int(k) for k in args.ks.split(",")]
    notes = (f"4096x2160 uint8 noise (torch seed 20261030) -> float32, alpha {ALPHA}, c {C}, k operations alternating mul and add; each (version, k) in its own process "
             f"with fresh Triton and Inductor caches (compile_s = the first call, cold); {ITERATIONS} timed calls after warm-up; kernels counted with torch.profiler; "
             f"ULP error against a float64 reference; timeout {args.timeout} s per point")
    cols = ["ops", "version", "status", "why", "gpu_mean_ms", "gpu_min_ms", "host_launch_us", "kernels", "compile_s", "max_ulp_vs_exact", "bitexact_fraction_vs_triton_hand"]
    with CsvWriter(args.out, "E2a element-wise chain: Triton and TorchInductor", cols, notes=notes) as csv:
        for ops in ks:
            ref = os.path.join(tempfile.gettempdir(), f"e2_triton_ref_{os.getpid()}_{ops}.pt")
            for version in args.versions.split(","):
                caches = tempfile.mkdtemp(prefix="e2cache")
                env = dict(os.environ, TRITON_CACHE_DIR=os.path.join(caches, "triton"), TORCHINDUCTOR_CACHE_DIR=os.path.join(caches, "inductor"),
                           TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_AUTOGRAD_CACHE="0")
                cmd = [sys.executable, os.path.abspath(__file__), "--worker", version, str(ops), "--ref", ref]
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
                csv.row(ops, version, status, why, *[r.get(c, "") for c in cols[4:]])
                print(f"k={ops:6d} {version:17s} {status:18s} " + (f"gpu {r['gpu_mean_ms']:9.4f} ms (min {r['gpu_min_ms']:9.4f}) host {r['host_launch_us']:9.1f} us kernels {r['kernels']:5d} compile {r['compile_s']:8.2f} s ulp {r['max_ulp_vs_exact']:.3g} bitexact {r['bitexact_fraction_vs_triton_hand']:.4f}" if r else why) + f"  [{time.perf_counter() - t0:.0f} s]", flush=True)
            if os.path.exists(ref):
                os.remove(ref)


if __name__ == "__main__":
    main()
