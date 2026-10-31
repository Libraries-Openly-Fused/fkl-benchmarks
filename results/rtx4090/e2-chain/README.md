# E2: the element-wise chain through FKL, Triton and TorchInductor (RTX 4090, CUDA 13.4.3, Triton 3.8.0, PyTorch 2.14.1)

Workload: a 4,096 x 2,160 uint8 image converted to float32 and scaled by alpha = 1.0, then k operations alternating Mul by c and Add c (c = 0.54, k / 2 pairs), written as float32.
Every implementation reads the same bytes: `e2_input.u8` (std::mt19937_64 seed 20261030, uniform 0..255, as E1a; sha256 `726b9ef38a8277d6021009d344446bf5915fe75f746d07950e608f63614a8995`), written by
`handwritten-cuda/e2_fkl_chain.cu`, which also runs the FKL chain (32 x 8 threads per block) and writes one float32 output per k. `triton/e2_chain.py` runs Triton and TorchInductor on that
image, each (version, k) in its own process with fresh Triton and Inductor caches, and compares every output with FKL's and with an exact float64 evaluation. The image and the outputs (6 x 35 MB) stay in
`/home/oscar/tfkl-bench/e2-work`, not in the repository. k = 2, 10, 50, 100, 500 (`e2_chain.csv`) and the bounded larger attempt k = 5,002 with a 900 s limit per point (`e2_chain_large.csv`).

**Runs.** `../e2-chain-run1/`: the first run, warm timings only. This folder: the second run (the first with the L2-flushed timings). `../e2-chain-run3/` and `../e2-chain-run4/`: the same binaries and scripts again, on a quiet machine
(nothing else running; load average 1.0 to 1.3, the benchmark itself). **`e2_compact.csv` is the table for the paper: k = 10, 100, 500, 5,002, warm and L2-flushed rows, `gpu_ms` the median of the three runs (2, 3, 4) of the mean of 100 calls,
`spread_pct` = (largest - smallest) / median of the three means, `min_ms` the smallest single call of the three runs. All its rows are final (three runs each).** The library was at 38c4f32 for all of them; the FKL kernels at c8fcbe8 are identical
(`../e0-pin-transfer/`). Run 2 reproduces the warm timings of run 1 (median difference 0.7 % over the 42 points, largest 11.3 %).

Versions: FKL; `triton_hand` (one hand-written Triton kernel, the chain unrolled at trace time); `eager` (PyTorch, one kernel per operation); `inductor_default` (torch.compile, fusion window 64);
`inductor_wide` (window raised so that the whole chain may fuse); `inductor_graphs` (torch.compile with mode "reduce-overhead", CUDA Graphs); `eager_graphs` (eager captured in one CUDA graph by hand).

Two timing conditions, both by CUDA events, mean of 100 calls after warm-up (`gpu_mean_ms`, `gpu_flushed_mean_ms`; `fkl_mean_ms`, `fkl_flushed_mean_ms` for FKL):
- **warm**: the calls follow each other; the 8.8 MB input and the 35 MB output fit the 72 MB L2 of the 4090 (FKL's 0.021 ms at k = 2 is below the 0.044 ms that moving 44 MB through DRAM would take), the same as E1a.
- **L2-flushed**: before each timed call a kernel that reads 256 MB runs outside the timed interval, so the call starts with the input in DRAM and a clean L2 (no dirty lines of the previous output).

## Results
- **Agreement.** FKL, the hand Triton kernel and all three Inductor variants give bit-identical outputs at every k (bitwise fraction 1.0000, 0 ULP). Eager differs: 79 % of the elements are identical to FKL's at k = 2, 37 % at k = 10, 62 % at k = 50 and 99 % at k = 100, 500 and 5,002, with a maximum difference of 1 to 4 ULP (float32 ULPs of FKL's value). Against an exact float64 evaluation of the chain all of them are within 1.12 to 4.76 ULP (4.6 ULP for FKL at k = 10, 1.65 ULP for k >= 100). The eager differences fade with k because the chain converges to its fixed point (c / (1 - c) = 1.174), where the rounding of the individual operations stops mattering.
- **Contraction of Mul and Add (SASS of the kernel that runs; `kernel-facts/e2_kernel_facts.csv`).** FKL: k / 2 FFMA and 1 FMUL per element, no FADD (k = 10: 5 FFMA, 1 FMUL; k = 100: 50 FFMA, 1 FMUL). `triton_hand`: 8 elements per thread, one FFMA per Mul+Add pair per element (4 k FFMA for k = 10 ... 5,002: 40, 400, 2,000, 20,008) and 8 FMUL for the scale by alpha, no FADD. TorchInductor (the three variants): 2 elements per thread, FFMA only: k FFMA (10, 100, 500, 5,002), no FMUL (the scale by 1.0 is gone), no FADD; counted in runs 3 and 4, where the script counts the kernel that Inductor selected. (In `e2_chain.csv` of this second run the `sass_*` columns of Inductor add up every configuration that Inductor compiled, two, 8 and 2 elements per thread, which gave 5 k; and in the run-1 folder likewise.) PyTorch eager launches one kernel per operation (k + 2 kernels), so nothing can be contracted across operations: it rounds after every Mul and every Add, which is the whole difference with the others.
- **Kernels launched per call:** 1 for FKL, Triton and all Inductor variants (also at k = 5,002: Inductor fuses a pointwise chain into one kernel whatever the fusion window, so the default and the wide window give the same kernel); k + 2 for eager (4, 12, 52, 102, 502, 5,004).
- **GPU time, warm (ms, median of 3 runs; spread of the three means in brackets):** k = 10: FKL 0.0212 (3.8 %), Triton 0.0289 (8.5 %), Inductor 0.0447 (2.2 %); k = 100: 0.0255 (4.9 %), 0.0294 (2.1 %), 0.0470 (10.1 %); k = 500: 0.0643 (0.3 %), 0.0730 (3.3 %), 0.0883 (1.5 %); k = 5,002: FKL 0.527 (0.7 %), Inductor 0.566 (2.5 %), Triton 0.795 (0.8 %). Eager: 0.362 (k = 10), 2.92 (100), 14.3 (500), 141.4 (5,002), launch-bound.
- **GPU time, L2-flushed (ms, median of 3 runs; spread in brackets): the order changes.** k = 10: FKL 0.0269 (4.3 %), Triton 0.0195 (1.5 %), Inductor 0.0192 (1.2 %); k = 100: 0.0303 (1.7 %), 0.0201 (0.6 %), 0.0230 (0.4 %); k = 500: 0.0690 (0.6 %), 0.0622 (0.5 %), 0.0653 (0.8 %);
  k = 5,002: FKL 0.5445 (2.5 %), Inductor 0.5193 (1.2 %), Triton 0.7505 (3.9 %). With the input read from DRAM FKL is slower than Triton and Inductor at k = 10, 100 and 500, by 38 to 40 % at k = 10, 51 % and 32 % at k = 100, 11 % and 6 % at k = 500 (Triton, Inductor);
  at k = 5,002 it is 4.9 % slower than Inductor, and its time is 27 % lower than Triton's. The differences at k = 500 are many times the run-to-run spread of the three runs (0.5 to 0.8 %); the 4.9 % at k = 5,002 is about twice the larger of the two spreads (2.5 %, 1.2 %).
  Triton and Inductor are faster flushed than warm in every row; FKL is slower flushed than warm in every row. These are measurements: no cause is stated here (see the next two items for what the kernels and the profiler show).
- **The kernels (`kernel-facts/e2_kernel_facts.csv`; SASS and PTX of the kernel that runs, k = 10 / k = 100).**
  FKL: 1 output element per thread; one `LDG.E.U8` and one `STG.E` (32-bit) per thread (PTX `ld.global.u8`, `st.global.f32`); blocks of 32 x 8 = 256 threads, grid 34,560; 12 / 10 registers in the cubin (16 allocated); 40 / 88 SASS instructions.
  Triton (hand-written): 8 elements per thread; one `LDG.E.64` (eight uint8 in one 8-byte load), two `STG.E.128`, two `LDS.128` and two `STS.128` (shared-memory staging); 128 threads per block (4 warps), grid 8,640; 29 registers; 112 / 472 instructions.
  TorchInductor: it compiles two configurations of the pointwise kernel and keeps the faster (`best_config`: XBLOCK = 512, num_warps = 8): 2 elements per thread; one `LDG.E.U16` (two uint8) and one `STG.E.64` (two floats); 256 threads per block, grid 17,280; 10 / 12 registers; 40 / 136 instructions.
- **Nsight Compute (`ncu/`; one launch of each kernel, caches flushed before it, clocks not locked, `scripts/e2_ncu_summary.py`), k = 10 (k = 100 in brackets).**

  | | FKL | TorchInductor | Triton (hand-written) |
  |---|---|---|---|
  | duration (us) | 25.8 (30.0) | 18.1 (21.8) | 19.0 (19.1) |
  | DRAM throughput, % of peak | 40.4 (35.4) | 53.4 (44.4) | 60.6 (55.3) |
  | bytes moved over the duration (GB/s; % of 1,008) | 397 (347); 39.4 % (34.5 %) | 523 (435); 51.9 % (43.2 %) | 594 (542); 58.9 % (53.7 %) |
  | DRAM bytes read / written (MB) | 8.85 / 1.38 | 8.85 / 0.62 | 8.85 / 2.44 |
  | L1 global-load sectors / store sectors | 276,480 / 1,105,920 | 276,480 / 1,105,920 | 276,480 / 1,105,920 |
  | L1 load requests / store requests | 276,480 / 276,480 | 138,240 / 138,240 | 34,560 / 69,120 |
  | warp instructions executed (millions) | 8.29 (20.7) | 4.15 (16.6) | 3.42 (15.9) |
  | grid x block | 34,560 x 256 | 17,280 x 256 | 8,640 x 128 |
  | achieved occupancy (theoretical 100 % for all) | 64.4 % (70.3 %) | 71.2 % (74.8 %) | 84.0 % (86.8 %) |

  The three kernels move the same bytes (the same sectors at the L1 and at the L2, 8.85 MB read from DRAM; of the 35 MB written, 0.6 to 2.4 MB reached DRAM during the launch), so the difference in time is a difference in the bandwidth that each kernel reaches.
  This README does not say why. No variant of any kernel was run.
- **Cold compile time (s)** is not the same operation in each column: FKL is an nvcc compile of the translation unit with the chain (object file, one core, `e2_fkl_compile.csv`: 3.65 s for k = 2 ... 100, 3.72 s at 500, 5.23 s at 5,002); Triton is the first call in a fresh process (Triton compile and ptxas: 0.7 to 0.9 s, 5.8 s at 5,002); Inductor is the first `torch.compile` call (1.2 s at k = 2 to 3.4 s at k = 500, 26.5 s at 5,002). Python start-up and `import torch` are outside all of them.
- **Larger attempt (k = 5,002):** neither Triton nor Inductor failed or timed out (900 s limit): the limit that FKL has with distinct operations (E3) is not reached by these tools with this chain, which they unroll into one kernel with the constants folded.

Files: `e2_chain.csv`, `e2_chain_large.csv`, `e2_fkl_chain.csv`, `e2_fkl_compile.csv`, `e2_compact.csv` (runs 2 to 4), `kernel-facts/`, `ncu/` (raw `ncu_<impl>_k<k>.csv`, `e2_ncu_summary.csv`), `stdout*.txt`, `stderr*.txt`, `gpu-at-end.csv`.
Window scripts of the session: `gpu-window-final.sh` (run 2), `gpu-window-final2.sh` (kernel profiles, runs 3 and 4): FKL program first, then `triton/e2_chain.py --input ... --fkl-dir ... --fkl-csv ... --fkl-compile ...`.
