# E19 batched box, phase 1 (e19-batched-box-phase1), 2026-11-18: the checks on the RX 9060, NOT timed

This is the checks-only window of `rpp/e19_batched_box.h`, the AMD side of `opencv-cuda/stencil/e19_batched_box.cu`. The program's header describes the pipeline, the rows and the checks. **Nothing here is timed.**
- **The program:** 2ad7ac5, FKL 24478f0, RPP 3.2.0 (ROCm 10.1.0), amdclang++ 24.0.0, gfx1200.
- **The build (`step0/`):** a clean worktree, 0 errors and 0 warnings, sha256 `a193bae62d1e2d76d83e5d3a66f867c4d874c1e40b9c2c2ee76f40a94bb8f4d1`.
- **The window** was granted on receipt and ran from 16:25:32 to 16:25:33Z (`window-console.txt`). Before each of its two runs the guards passed: the binary's sha256, no `Runner.Worker`, and a load average below 0.8 (0.19). The rocprofv3 run ran only after the check run had exited 0 with "E19 checks passed".
- **Decision:** the coordinator session, 2026-11-18 (E19 on the RX 9060, phase 1).

## The check run (`check/`, `E19_NO_TIMING=1`, `E19_DUMP`)
Exit 0, empty stderr, "E19 checks passed". The table is `e19_batched_box_checks_GPU_AMD.csv`. Every comparison covers all 1,966,080 values, and every buffer was filled with 0xFF bytes before the call.

| row | check | result |
|---|---|---|
| `rpp` | grey: RPP's source rule, `(0.299f R + 0.587f G) + 0.114f B` with no FMA, ties to even | 154 values differ (max 1) |
| `rpp` | grey: the rule found by comparison, `t = 0.587f G; t = fma(0.299f, R, t); t = fma(0.114f, B, t)`, ties to even | **0 differ** (required: ok) |
| `rpp` | box against round(sum / 9) of its own grey, REPLICATE at each crop's edges | **0 differ** (required: ok). floor(sum / 9) differs at 873,625 |
| `rpp` | float against (v / 255 - 0.45) / 0.225 in double, v its own box | max 1.3e-7, all finite (required: ok) |
| `fkl_loop` | against FKL's pipeline in double, grey order `fma(0.587 G, 0.299 R, 0.114 B)` | max 6.4e-7, 0 beyond 1e-5, all finite (required: ok) |
| `fkl_batch` | the same | max 3.6e-7, 0 beyond 1e-5, all finite (required: ok) |
| reported | `rpp` against `fkl_loop` | max 0.0077 (the 8-bit rounding of RPP's box, against FKL's float box) |
| reported | `fkl_batch` against `fkl_loop` | max 7.7e-7 |

- **The grey rule.** RPP's grey (`librpp.so.3.2.0`) is the same FMA chain that nvcc gives FKL on NVIDIA, and that amdclang++ gives FKL here. In `isa/grey-instructions.txt`, both FKL kernels have `v_mul_f32` 0.587f G, then `v_fmac_f32` 0.299f R, then `v_fmac_f32` 0.114f B, then `v_rndne_f32`. That is `__float2int_rn`, the signed branch of `RGB2Gray<uchar3, float>`. The other evaluation orders give these differences:

  | evaluation order | rpp grey | fkl_loop / fkl_batch beyond 1e-5 |
  |---|--:|--:|
  | no FMA, left to right | 154 | 1,380 |
  | R first, then G and B | 178 | 1,593 |
  | the rounded R + G sum, then B | 141 | 1,260 |
  | the exact sum | 926 | 8,281 |

- **The dump** (`E19_DUMP`): RPP's grey, box and tensor and both FKL tensors, as raw bytes, about 27 MB. It is kept outside the repository; `check/dump-sha256.txt` gives the sizes and sha256 sums.

## The rocprofv3 run (`rocprofv3/`, `E19_PROFILE=1`)
Exit 0. One warm-up call of each row, then three calls, each in a ROCTx range. `calls.txt` comes from `../e18-box-phase1/rocprofv3/rocprof_calls.py`; the grids are in work-items.

| row (per frame) | GPU dispatches | grid, workgroup |
|---|---|---|
| `fkl_loop` | 30 x `fk::linearFilterDPPKernel` | 256 x 256 x 1, 16 x 8 x 1 (16 x 32 blocks) each |
| `fkl_batch` | 1 x `fk::morphologyDPPKernel` | **256 x 256 x 30**, 16 x 8 x 1 (16 x 32 x 30 blocks: one grid plane per crop) |
| `rpp` | 30 x `__amd_rocclr_copyBufferRectAligned` (the 30 `hipMemcpy2DAsync`), then `color_to_greyscale_pkd3_pln1_hip_tensor`, `box_filter_3x3_pln_hip_tensor`, `crop_mirror_normalize_pln_hip_tensor` | copies 48 x 256 x 1 or 192 x 256 x 1 (below), 16 x 16; RPP's three 32 x 256 x 30, 48 x 304 x 30 and 32 x 256 x 30 |

- **The copies are blit kernels.** HIP runs the 30 device-to-device `hipMemcpy2DAsync` as blit kernels; the memory-copy trace has no record of them as SDMA copies.
- **The copy grids** (the 120 traced copies: 4 calls x 30 crops, by launch order) take two forms, 60 each, the same crops in every call:
  - the crops of even columns (i mod 6 = 0, 2, 4; x = 64, 1264, 2464) get 48 x 256 x 1;
  - the crops of odd columns (x = 664, 1864, 3064) get 192 x 256 x 1.
  - A crop's source address in RPP's frame is `(y * 3840 + x) * 3` bytes from the frame's start. That is a multiple of 16 for the even columns and 8 modulo 16 for the odd ones. Every destination (`i * 196,608` bytes into the batch) is a multiple of 256.
  - A row is 768 bytes, so 48 work-items per row would move 16 bytes each and 192 would move 4 bytes each. This is an observation; the cause, in HIP's choice of the blit's element size, was not checked.

- **No allocations inside a call.** In no row does a call allocate, free or memset anything.
- **The same in every call:** every row's three profiled calls launch the same kernels.
