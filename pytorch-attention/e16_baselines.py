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
"""E16: forward attention baselines, the counterpart of fkl-self/attention/e16_attention.cu (same configurations, same columns).

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python pytorch-attention/e16_baselines.py --out e16_baselines.csv --input-dir DIR
      [--dims 64,128] [--seqs 512,1024,2048,4096,8192] [--tokens 262144] [--bh 0] [--causal 0,1] [--backends flash,cudnn,efficient,math,flash_attn,triton,dequant_flash,dequant_compiled_flash,epilogue_flash]

The same configurations and the same input bytes as the C++ program: the file e16_input_d<D>_s<S>_bh<BH>.bin of --input-dir, written by fkl-self/attention/e16_attention.cu (Q, K, V in bf16, then K and V
in int8, each followed by its fp32 per-token scales, then K and V in e4m3 with their scales; the e4m3 data is not used here). bf16 Q, K, V of shape (BH, S, D) (contiguous, the memory layout of the FKL program;
PyTorch sees them as (1, BH, S, D), flash-attn as the view (1, S, BH, D)),
BH = max(min_bh, tokens / S) heads so that BH x S is the same for every S (or --bh heads for every S), causal or not, fp32 accumulation inside every kernel. Per variant and configuration: the time per call from
`--iters` blocks of back-to-back calls between two CUDA events (each block about `--block-ms` long: the GPU time of the call, with no gap for the host between calls) and the median of `--iters` single synchronised
calls (the launch latency of the call is in it), TFLOP/s = 4 BH S^2 D (halved for causal) / time, the largest and mean absolute error of the output against an fp32 attention of the same inputs on the same GPU (the
math backend in fp32, for `--error-heads` heads: the first ones), and the kernels of one call (CUPTI, one extra untimed call under torch.profiler). Variants:
  flash           torch.nn.functional.scaled_dot_product_attention with SDPBackend.FLASH_ATTENTION (FlashAttention-2)
  cudnn           the same with SDPBackend.CUDNN_ATTENTION
  efficient       the same with SDPBackend.EFFICIENT_ATTENTION
  math            the same with SDPBackend.MATH (materialises S x S; skipped when it would take more than --max-math-gb)
  flash_attn      flash_attn.flash_attn_func of the flash-attn package
  triton          the forward kernel of triton/e16_fused_attention_fwd.py (the fused-attention tutorial kernel of Triton, forward only, its autotuner as shipped)
  dequant_flash   the unfused alternative of the compressed KV cache: K and V are int8 with one fp32 scale per token (those of the FKL program), dequantised by one PyTorch elementwise kernel each
                  (int8 * fp32, rounded to bf16 in the same kernel: the arithmetic of the FKL program's dequantisation kernel), then flash
  dequant_compiled_flash   the same with the dequantisation of K and V compiled by torch.compile (TorchInductor), then flash; its kernel count is the CUPTI count of the row
  epilogue_flash  the unfused alternative of the epilogue o * a + b: flash, then one PyTorch elementwise kernel (torch.addcmul with the scalars as 0-dim tensors)
dequant_flash, dequant_compiled_flash and epilogue_flash are compositions of off-the-shelf pieces that a user has today (elementwise kernels around the flash kernel). They are NOT a hand-fused quantised-cache attention
kernel such as the ones in serving systems, and not the best that could be written for these two cases.
A variant that cannot run a configuration (not supported on the GPU, out of memory) writes a row with its message in `note` and no time.
Preparation: none (column prep_us_per_call = "none"). The tensors exist before timing, and torch.compile and Triton's autotuner run once before timing (the probe, the first call of every variant).
"""
import argparse
import csv
import datetime
import math
import os
import platform
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "triton"))

COLUMNS = ["gpu", "variant", "d", "seq", "bh", "causal", "position", "blocks", "calls_per_block", "t_start_ms", "t_end_ms", "time_ms_median", "time_ms_min", "time_ms_max", "single_ms_median", "tflops_median",
           "max_abs_err", "mean_abs_err", "kernels", "schedule", "block_q", "block_kv", "num_warps", "split_kv", "kernel_facts", "note", "prep_us_per_call"]


def flops(bh, seq, d, causal):
    return 4.0 * bh * seq * seq * d * (0.5 if causal else 1.0)


def measure(fn, warmup, blocks, block_ms, warmup_ms=300.0):
    """(median, smallest, largest per call over the blocks, median of the single calls, calls per block, wall clock of the first block start and of the last block end in ms since the epoch), times in ms. The warm-up lasts at least `warmup` calls and at least warmup_ms of GPU time (the clock of the GPU must be
    back at its boost before the first block)."""
    def elapsed(calls):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(calls):
            fn()
        e.record()
        e.synchronize()
        return s.elapsed_time(e)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    spent = 0.0
    while spent < warmup_ms:
        spent += elapsed(4)
    first = max(elapsed(1), 1e-3)
    reps = max(1, min(200, math.ceil(block_ms / first)))
    t_start = int(time.time() * 1000)
    per_call = [elapsed(reps) / reps for _ in range(blocks)]
    t_end = int(time.time() * 1000)
    singles = [elapsed(1) for _ in range(blocks)]
    return statistics.median(per_call), min(per_call), max(per_call), statistics.median(singles), reps, t_start, t_end


def kernels_of(fn):
    """(number of kernels, names joined by ' ; ') of one call, from CUPTI through torch.profiler; (-1, message) when the profiler cannot run."""
    try:
        from torch.profiler import ProfilerActivity, profile
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith(("Memcpy", "Memset"))]
        return len(names), " ; ".join(n[:70] for n in names)
    except Exception as e:  # noqa: BLE001
        return -1, f"profiler failed: {type(e).__name__}"


def _sdpa_call(backend, q, k, v, causal):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def reference(q, k, v, causal, heads):
    """fp32 attention of the first `heads` heads from the bf16 inputs (the math backend, fp32)."""
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q[:, :heads].float(), k[:, :heads].float(), v[:, :heads].float(), is_causal=causal)


def errors(out, ref, heads):
    diff = (out[:, :heads].float() - ref).abs()
    return diff.max().item(), diff.mean().item()


def clean(text):
    return str(text).replace(",", ";").replace("\n", " ").replace("\r", " ")


def first_line(e):
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0][:160] if text else ''}"


def probe(dev, calls):
    """One tiny call of every baseline (D = 64 and 128, causal and not, 2 heads of 128 tokens): {name: 'ok' or the error text}. Every baseline fails on its own, whatever the others do."""
    gen = torch.Generator(device=dev).manual_seed(1)
    out = {}
    for name, call in calls.items():
        problems = []
        for d in (64, 128):
            for causal in (False, True):
                q, k, v = (torch.randn(1, 2, 128, d, device=dev, dtype=torch.bfloat16, generator=gen) for _ in range(3))
                try:
                    call(q, k, v, causal)
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    problems.append(f"D={d} causal={int(causal)}: {first_line(e)}")
                    try:
                        torch.cuda.synchronize()
                    except Exception:  # noqa: BLE001
                        pass
                    torch.cuda.empty_cache()
        out[name] = "ok" if not problems else clean(problems[0] + (f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""))
    return out


def os_kernel_lines():
    """The os and kernel comment lines of every file of a run, as common/system_info.h writes them, read at run time."""
    try:
        name = platform.freedesktop_os_release()["PRETTY_NAME"]
    except (OSError, KeyError, AttributeError):
        name = "unavailable"
    u = os.uname()
    return [f"# os: {name}", f"# kernel: {u.sysname} {u.release}"]


def header_lines(probes, unavailable):
    lines = ["# experiment: E16 forward attention baselines (PyTorch SDPA backends, flash-attn, Triton)", f"# date: {datetime.datetime.now().isoformat(timespec='seconds')}"]
    props = torch.cuda.get_device_properties(0)
    lines += os_kernel_lines()
    lines.append(f"# gpu: {props.name}, cc {props.major}.{props.minor}, {props.multi_processor_count} SMs, {props.total_memory // 2**20} MiB")
    lines.append(f"# torch: {torch.__version__}, cuda {torch.version.cuda}, cudnn {torch.backends.cudnn.version()}")
    try:
        import flash_attn
        lines.append(f"# flash-attn: {flash_attn.__version__}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"# flash-attn: not importable ({type(e).__name__}: {clean(e)})")
    try:
        import triton
        lines.append(f"# triton: {triton.__version__}; TRITON_PTXAS_PATH={os.environ.get('TRITON_PTXAS_PATH', '(unset)')}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"# triton: not importable ({type(e).__name__}: {clean(e)})")
    try:
        smi = subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], "--query-gpu=driver_version,clocks.sm,clocks.max.sm,power.limit", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        lines.append(f"# nvidia-smi driver: {smi.split(',')[0].strip()}")
        lines.append(f"# nvidia-smi (driver, sm clock now, max sm clock, power limit): {smi}")
    except Exception:  # noqa: BLE001
        pass
    for name, result in probes.items():
        lines.append(f"# probe {name}: {result}")
    for name, why in unavailable.items():
        lines.append(f"# unavailable {name}: {why}")
    lines.append("# note: dequant_flash, dequant_compiled_flash and epilogue_flash are compositions of off-the-shelf PyTorch pieces that a user has today (elementwise kernels around the flash kernel); they are not a hand-fused "
                 "quantised-cache attention kernel such as the ones in serving systems, and not the best that could be written for these two cases")
    lines.append("# input: the files e16_input_d<D>_s<S>_bh<BH>.bin written by fkl-self/attention/e16_attention.cu (the bytes of the FKL program: bf16 Q, K, V, int8 K and V with fp32 per-token scales); "
                 "layout (BH, S, D) contiguous; time = per call from blocks of back-to-back calls between two CUDA events, single = one synchronised call; warm-up before; fp32 accumulation inside the kernels")
    lines.append("# preparation: none (prep_us_per_call = none): the tensors exist before timing, and torch.compile and Triton's autotuner run once before timing")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", default="64,128")
    ap.add_argument("--seqs", default="512,1024,2048,4096,8192")
    ap.add_argument("--tokens", type=int, default=262144, help="BH x S of every configuration")
    ap.add_argument("--bh", type=int, default=0, help="fixed number of heads for every S (0: tokens / S)")
    ap.add_argument("--min-bh", type=int, default=4)
    ap.add_argument("--causal", default="0,1")
    ap.add_argument("--backends", default="flash,cudnn,efficient,math,flash_attn,triton,dequant_flash,dequant_compiled_flash,epilogue_flash")
    ap.add_argument("--iters", type=int, default=15, help="blocks, and single calls")
    ap.add_argument("--block-ms", type=float, default=20.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--warmup-ms", type=float, default=300.0, help="the warm-up lasts at least this much GPU time")
    ap.add_argument("--rotate", type=int, default=0, help="offset of the order of the groups of variants (the long runs use 0, 1, 2)")
    ap.add_argument("--error-heads", type=int, default=2)
    ap.add_argument("--max-math-gb", type=float, default=2.0)
    ap.add_argument("--epi-a", type=float, default=1.25)
    ap.add_argument("--epi-b", type=float, default=0.5)
    ap.add_argument("--input-dir", required=True, help="the folder of the input files e16_input_d<D>_s<S>_bh<BH>.bin of the FKL program")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    gpu = torch.cuda.get_device_name(0)
    backends = a.backends.split(",")
    unavailable = {}
    triton_fwd = None
    if "triton" in backends:
        try:
            from e16_fused_attention_fwd import attention_forward as triton_fwd
        except Exception as e:  # noqa: BLE001
            unavailable["triton"] = clean(f"import failed, {first_line(e)}")
            print(f"triton backend unavailable: {unavailable['triton']}")
    flash_attn_func = None
    if "flash_attn" in backends:
        try:
            from flash_attn import flash_attn_func
        except Exception as e:  # noqa: BLE001
            unavailable["flash_attn"] = clean(f"import failed, {first_line(e)}")
            print(f"flash_attn backend unavailable: {unavailable['flash_attn']}")
    deq_compiled = None
    if "dequant_compiled_flash" in backends:
        try:
            from torch import _dynamo   # not "import torch._dynamo": inside a function that would make `torch` a local name of main()
            _dynamo.config.cache_size_limit = 256   # one compiled graph per configuration (the default of 8 would silently fall back to eager)
            _dynamo.config.accumulated_cache_size_limit = 1024
            deq_compiled = torch.compile(lambda k8_, ks_, v8_, vs_: ((k8_ * ks_[..., None]).to(torch.bfloat16), (v8_ * vs_[..., None]).to(torch.bfloat16)), dynamic=False)
        except Exception as e:  # noqa: BLE001
            unavailable["dequant_compiled_flash"] = clean(f"torch.compile failed, {first_line(e)}")
            print(f"dequant_compiled_flash unavailable: {unavailable['dequant_compiled_flash']}")
    sdpa = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}
    calls = {name: (lambda q, k, v, c, name=name: _sdpa_call(sdpa[name], q, k, v, c)) for name in ("flash", "cudnn", "efficient", "math") if name in backends}
    if flash_attn_func is not None:
        calls["flash_attn"] = lambda q, k, v, c: flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=c)
    if triton_fwd is not None:
        calls["triton"] = lambda q, k, v, c: triton_fwd(q, k, v, c)
    if deq_compiled is not None:
        calls["dequant_compiled_flash"] = lambda q, k, v, c: deq_compiled(torch.ones_like(k, dtype=torch.int8), torch.ones(k.shape[:-1], device=k.device, dtype=torch.float32),
                                                                             torch.ones_like(v, dtype=torch.int8), torch.ones(v.shape[:-1], device=v.device, dtype=torch.float32))
    probes = probe(dev, calls)
    for name, result in probes.items():
        print(f"probe {name}: {result}", flush=True)
    with open(a.out, "w", newline="") as f:
        for line in header_lines(probes, unavailable):
            f.write(line + "\n")
        w = csv.writer(f)
        w.writerow(COLUMNS)
        config_index = 0
        for d in [int(x) for x in a.dims.split(",")]:
            for seq in [int(x) for x in a.seqs.split(",")]:
                bh = a.bh if a.bh > 0 else max(a.min_bh, a.tokens // seq)
                # the inputs of the FKL program: Q, K, V (bf16) and K and V in int8 with their fp32 per-token scales, read from its file (the layout is in the docstring)
                path = os.path.join(a.input_dir, f"e16_input_d{d}_s{seq}_bh{bh}.bin")
                n, tokens = bh * seq * d, bh * seq
                if not os.path.isfile(path) or os.path.getsize(path) != 10 * n + 16 * tokens:
                    raise SystemExit(f"{path}: missing or not {10 * n + 16 * tokens} bytes (run fkl-self/attention/e16_attention.cu with E16_INPUT_DIR={a.input_dir} first)")
                raw = torch.from_file(path, shared=False, size=10 * n + 16 * tokens, dtype=torch.uint8)
                offsets = {}
                at = 0
                for name, size in (("q", 2 * n), ("k", 2 * n), ("v", 2 * n), ("k8", n), ("ks", 4 * tokens), ("v8", n), ("vs", 4 * tokens)):
                    offsets[name] = (at, at + size)
                    at += size
                part = lambda name: raw[offsets[name][0]:offsets[name][1]]
                q = part("q").view(torch.bfloat16).reshape(1, bh, seq, d).to(dev)
                k = part("k").view(torch.bfloat16).reshape(1, bh, seq, d).to(dev)
                v = part("v").view(torch.bfloat16).reshape(1, bh, seq, d).to(dev)
                k8 = part("k8").view(torch.int8).reshape(1, bh, seq, d).to(dev)
                v8 = part("v8").view(torch.int8).reshape(1, bh, seq, d).to(dev)
                ks = part("ks").view(torch.float32).reshape(1, bh, seq).to(dev)
                vs = part("vs").view(torch.float32).reshape(1, bh, seq).to(dev)
                del raw
                # the 0-dim tensors of the epilogue
                epi_a = torch.tensor(a.epi_a, device=dev, dtype=torch.bfloat16)
                epi_b = torch.tensor(a.epi_b, device=dev, dtype=torch.bfloat16)
                for causal in [bool(int(x)) for x in a.causal.split(",")]:
                    heads = min(a.error_heads, bh)
                    ref = reference(q, k, v, causal, heads)
                    ref_deq = ref_epi = None
                    if "dequant_flash" in backends or deq_compiled is not None:
                        kd32 = k8[:, :heads].float() * ks[:, :heads, :, None]
                        vd32 = v8[:, :heads].float() * vs[:, :heads, :, None]
                        with sdpa_kernel(SDPBackend.MATH):
                            ref_deq = F.scaled_dot_product_attention(q[:, :heads].float(), kd32, vd32, is_causal=causal)
                    if "epilogue_flash" in backends:
                        ref_epi = ref * a.epi_a + a.epi_b

                    def emit(name, position=0, **values):
                        row = {"gpu": gpu, "variant": name, "d": d, "seq": seq, "bh": bh, "causal": int(causal), "position": position}
                        row.update(values)
                        w.writerow([row.get(c, "") for c in COLUMNS])
                        f.flush()
                        print(" ".join(str(row.get(c, "")) for c in COLUMNS[:19] + ["note"]), flush=True)   # without the kernel names (long), with the note (the reason of a failure)

                    def run(name, fn, ref_for, position):
                        try:
                            out = fn()
                            ms, lo, hi, single, reps, t0, t1 = measure(fn, a.warmup, a.iters, a.block_ms, a.warmup_ms)
                            nk, names = kernels_of(fn)
                            mx, mean = errors(out.reshape(1, bh, seq, d), ref if ref_for is None else ref_for, heads)
                            emit(name, position, blocks=a.iters, calls_per_block=reps, t_start_ms=t0, t_end_ms=t1, time_ms_median=f"{ms:.5f}", time_ms_min=f"{lo:.5f}", time_ms_max=f"{hi:.5f}", single_ms_median=f"{single:.5f}",
                                 tflops_median=f"{flops(bh, seq, d, causal) / (ms * 1e-3) / 1e12:.3f}", max_abs_err=f"{mx:.6g}", mean_abs_err=f"{mean:.6g}", kernels=nk, kernel_facts=clean(names),
                                 prep_us_per_call="none")
                        except Exception as e:  # noqa: BLE001
                            emit(name, position, blocks=0, note=clean(first_line(e)))
                            torch.cuda.empty_cache()

                    for name, why in unavailable.items():
                        emit(name, 0, blocks=0, note=f"unavailable: {why}")

                    def sdpa_variant(name):
                        def fn():
                            with sdpa_kernel(sdpa[name]):
                                return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                        return fn

                    def fn_deq():
                        kk = torch.mul(k8, ks[..., None], out=torch.empty_like(k))    # int8 * fp32, rounded to bf16 in the same elementwise kernel
                        vv = torch.mul(v8, vs[..., None], out=torch.empty_like(v))
                        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            return F.scaled_dot_product_attention(q, kk, vv, is_causal=causal)

                    def fn_deq_compiled():
                        kk, vv = deq_compiled(k8, ks, v8, vs)    # compiled by TorchInductor
                        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            return F.scaled_dot_product_attention(q, kk, vv, is_causal=causal)

                    def fn_epi():
                        with sdpa_kernel(sdpa["flash"]):
                            o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                        return torch.addcmul(epi_b, o, epi_a)    # b + o * a in one elementwise kernel

                    # the variants of this configuration in GROUPS; a group is measured together (its members are adjacent): {flash, epilogue_flash, dequant_flash, dequant_compiled_flash}, then one group per other baseline.
                    # ORDER: the groups are rotated by (configuration index + --rotate) mod (number of groups) and the members of the first group are reversed when that shift is odd.
                    quad = []
                    if "flash" in backends:
                        quad.append(("flash", sdpa_variant("flash"), None))
                    if "epilogue_flash" in backends:
                        quad.append(("epilogue_flash", fn_epi, ref_epi))
                    if "dequant_flash" in backends:
                        quad.append(("dequant_flash", fn_deq, ref_deq))
                    if deq_compiled is not None:
                        quad.append(("dequant_compiled_flash", fn_deq_compiled, ref_deq))
                    groups = [quad] if quad else []
                    for name in ("cudnn", "efficient", "math"):
                        if name not in backends:
                            continue
                        if name == "math" and bh * seq * seq * 2 / 2**30 > a.max_math_gb:
                            emit(name, 0, blocks=0, note=f"skipped: S x S scores of {bh * seq * seq * 2 / 2**30:.1f} GiB")
                            continue
                        groups.append([(name, sdpa_variant(name), None)])
                    if flash_attn_func is not None:
                        qf, kf, vf = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # (1, S, BH, D) views of the same memory
                        groups.append([("flash_attn", (lambda qf=qf, kf=kf, vf=vf: flash_attn_func(qf, kf, vf, causal=causal).transpose(1, 2)), None)])
                    if triton_fwd is not None:
                        groups.append([("triton", (lambda: triton_fwd(q, k, v, causal)), None)])
                    if groups:
                        shift = (config_index + a.rotate) % len(groups)
                        groups = groups[shift:] + groups[:shift]
                        position = 0
                        for gi, group in enumerate(groups):
                            members = list(group)
                            if len(members) > 1 and shift % 2 == 1:
                                members.reverse()
                            for name, fn, ref_for in members:
                                position += 1
                                run(name, fn, ref_for, position)
                    config_index += 1
                del q, k, v, k8, v8, ks, vs
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
