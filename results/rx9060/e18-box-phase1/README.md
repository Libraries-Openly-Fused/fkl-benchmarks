# E18 box, phase 1 (e18-box-phase1), 2026-11-17: the 3 x 3 box filter after a crop and a resize on the RX 9060, RPP against FKL, checks only, NOT timed

This folder is phase 1 of the redefined E18 on AMD: the program, the build, the checks, the kernels of every call and the metrics of the fused kernel. **Nothing here is timed.** The timing code is written (`phase2/`) and runs only in a GPU window scheduled with the coordinator after review.
- **The program:** `rpp/e18_box.h`, target `e18_box_hip`, at 664b06a (branch `e18-box-rx9060`). It is f97e949 with the changes of the review: RPP's parameter arrays moved to pinned host memory, and the variant `rpp_ltrb` added.
- **FKL:** 24478f0, the pin of `external/fkl` on main since #47.
- **RPP:** 3.2.0 of ROCm 10.1.0 (`librpp.so.3.2.0`, sha256 `c6d9bdc7800e105122632f01f408a709b38bea4775677040beda4d7e5614844d`), HIP back-end.
- **Decision:** E18 uses a pipeline that OpenCV-CUDA, NPP, RPP and FKL can all run, in 8 bits from start to end. Oscar chose the box filter on 2026-11-17, in the coordinator session. This replaces the sharpen pipeline of `e18-stencil-phase1` (#46), which RPP cannot run (`inventory/`).

## The pipeline
1. One 3840 x 2160 8UC3 frame: E4's seeded source (LCG, seed 12345), the same frame as `opencv-cuda/stencil/e18_stencil.cu`.
2. A crop of 2560 x 1440 at (640, 360).
3. A bilinear resize to 1920 x 1080, half-pixel sampling, with 8-bit input and 8-bit output.
4. A 3 x 3 box filter, anchor at the centre, border REPLICATE, with 8-bit input and 8-bit output.

## The variants
The order below is the order of the checks and of the timing.

**`rpp`:** RPP's own buffers, with no FKL object. Two calls, as a user writes them. One RPP handle (`rppCreate`, batch size 1) and every buffer are made once, outside the calls.
1. `rppt_resize`: U8, NHWC, batch 1, `RpptInterpolationType::BILINEAR`, `RPP_HIP_BACKEND`. The source is the whole frame, the crop is its XYWH ROI (`roiTensorPtrSrc`), and `dstImgSizes` is 1920 x 1080. It writes into an intermediate 1920 x 1080 U8 tensor.
2. `rppt_box_filter`: `kernelSize` 3, `RpptImageBorderType::REPLICATE`, with the whole intermediate as its XYWH ROI.

The buffers follow what RPP 3.2.0 needs:
- **The ROI arrays and `dstImgSizes` are in pinned host memory** (`hipHostMalloc`), as in the Figure 15 program (`rpp/benchmark_rpp_preprocessing.h`). RPP's kernels read them in place, and its conversion kernel rewrites them in place. RPP 3.2.0's header documents `roiTensorPtrSrc` as HIP memory for the HIP back-end; we use pinned host memory, as Figure 15's program does.
- **256 entries per ROI array.** RPP's ROI conversion kernels run 256 threads for a batch of 1, each rewriting one entry (`rpp_hip_roi_conversion.hpp`).
- **A 64-byte offset in the intermediate.** It is the box filter's source, and `rppt_box_filter` returns `RPP_ERROR_LOW_OFFSET` when `offsetInBytes < 12 * (kernelSize / 2)` (`rppt_tensor_filter_augmentations.cpp:133`).
- **64 bytes after every image**, because RPP stores 8 pixels at a time, up to 7 pixels past the ROI width.

**The ROI write is part of every call.** `rppt_resize` converts its ROI from XYWH to LTRB in place, so each call first writes the crop's ROI: a 16-byte store on the host into the pinned array. `rppt_box_filter` converts only an LTRB ROI (`box_filter.cpp:2050`), so its XYWH ROI is written once. The check run confirms both: after the calls, the crop's ROI reads LTRB 640 360 3199 1799 and the box filter's is unchanged (`e18_box_checks_GPU_AMD.csv`).

**The host write is safe** because no RPP kernel still reads the ROI when it is rewritten. Every call of this program follows a synchronisation:
- the check run synchronises after every call;
- `fklb::timeIt` and `fklb::timePreparedLaunch` wait for each call's stop event before the next call (`common/hip_timing.h:51` and `:93`);
- the profile mode synchronises after every call.

**From HIP memory to pinned host memory.** f97e949 kept the ROIs in HIP memory and wrote them with a 16-byte `hipMemcpyAsync`, which HIP ran as a blit kernel in every call. The outputs are bitwise the same in both forms. `rpp-roi-memory/digests.txt` gives the digests (FNV-1a 64) of the outputs and the intermediate from a scratch build of 6f21abf and from this folder's run: equal for `rpp`, `fkl_box_halfpixel` and `fkl_half_once`.

**`rpp_ltrb`:** `rpp` with the crop passed to `rppt_resize` as an LTRB ROI (640, 360, 3199, 1799). That is `rppt_resize`'s own form: it converts the ROI only when it is XYWH (`resize.cpp:944` on the single-image path that a batch of 1 takes, and `:730` on the batched path), so with LTRB it launches no conversion kernel. The ROI is in pinned host memory, in its own 256-entry array, and is written before every call like `rpp`'s. `rpp_ltrb` has its own intermediate and output, and shares the box filter's ROI with `rpp`. Both forms are documented. `rpp` is the form of the Figure 15 program; `rpp_ltrb` is RPP's native form for the resize.

**In the check run,** every RPP call is followed by `hipGetLastError` and `hipStreamSynchronize`, because RPP returns `RPP_SUCCESS` when a launch fails. The profile mode and the timing do not add them.

**`fkl_box_halfpixel`:** one `fk::executeLinearFilter` on `ParArch::GPU_AMD` with the default `LinearFilterDPPDetails<fk::float3>` (16 x 8 tile, 7 x 7 largest kernel), so one kernel. Every call builds the crop view, every IOp and the details.
- **The image Read IOp** is fused in the one kernel:
  - `PerThreadRead` of the frame cropped through `Ptr2D::crop2D`;
  - `fk::Resize<INTER_LINEAR, IGNORE_AR, NullType, ResizeSampling::HALF_PIXEL>` to 1920 x 1080, returning `float3`;
  - `fk::SaturateCast<float3, uchar3>` (round to nearest, ties to even), which rounds the resized image to 8 bits inside the kernel, as RPP rounds it between its two calls;
  - `fk::Cast<uchar3, float3>`.
- **The coefficient:** `fk::ConstantFilterRead<float3>::build(float3{1, 1, 1} / float3{9, 9, 9})`. Multiply and accumulate are `fk::Mul` and `fk::Add` on `float3` (`UnaryType`).
- **The write:** `fk::SaturateCast<float3, uchar3>`, then `PerThreadWrite`.

These are exactly the arguments `fk::executeBoxFilter` builds (`linear_filter.h:452-468` at 24478f0). `executeBoxFilter` itself does not compile for `LinearFilterDPPDetails<float3>`. Its coefficient `scale / static_cast<T>(details.kernelWidth * details.kernelHeight)` (lines 459-460 for GPU_AMD, 324-325 for GPU_NVIDIA) needs a conversion from `int` to the vector type, and FKL's vector types have none. The compile test in `executeBoxFilter-float3/` gives `linear_filter.h:460:9: error: no matching conversion for static_cast from 'int' to 'T' (aka 'fk::Float3')`. No FKL change is part of this PR.

**`fkl_half_once`:** `fkl_box_halfpixel` with the crop view, the IOps and the details built once. Every call is `fk::executeLinearFilter`.

**Differences from the FKL code of the CUDA program,** all of them back-end differences, as in `e18-stencil-phase1`:
- the stream is `fk::Stream_<fk::ParArch::GPU_AMD>`;
- the FKL vector types are written `fk::float3` and `fk::uchar3`, and the coefficient is `float3{1, 1, 1} / float3{9, 9, 9}` (CUDA: `make_float3`);
- the copies use the HIP runtime;
- the timing is `common/hip_timing.h` (`common/cuda_timing.h` with HIP events, copied from #46);
- the spin waits on `wall_clock64()`;
- the profiler ranges are ROCTx (`rocprofiler-sdk-roctx`) instead of NVTX.

## The build (`step0/`)
- amdclang++ 24.0.0 (ROCm 10.1.0), gfx1200, Release, the HIP configuration of the benchmark builds (`-DFKLB_BUILD_RPP=ON`, CUDA off). Only `e18_box_hip` was built: 0 errors, 0 warnings.
- **Source state** (`source-state.txt`): the worktree at 664b06a, with FKL 24478f0 clean. The only changed files were three scripts of this results folder, which are not part of the build.
- **Why `git status` is recorded directly:** the `-dirty` mark of the binaries covers `common cmake fkl-self opencv-cuda npp external CMakeLists.txt CMakePresets.json` (`cmake/fklb_write_build_info.cmake:22`), not `rpp/`.
- The binary records 664b06a and FKL 24478f0. Its sha256 is in `binary-sha256.txt` (`8d4a2b98…`).
- Before each GPU run there was no CI job (`Runner.Worker`) on this machine.
- The folder is `step0/`, not `build/`, because the repository's `.gitignore` ignores `build*/`.

## The checks (`console.txt` and the four CSVs)
One run with `E18_NO_TIMING=1`: exit 0, empty stderr, "E18 checks passed".

Every comparison covers all 1920 x 1080 x 3 = 6,220,800 values. It counts two regions separately, because RPP's resize never reads the last column and row of its ROI:
- the interior: 6,202,812 values;
- the last two columns and rows: 17,988 values.

The rows also give how many of the differing values are in the very last row or column.

### The exact reference
The scale is 2560 / 1920 = 4/3, so the half-pixel sample positions are multiples of 1/6. The weights in each direction are 1/6, 1/2 and 5/6, and every exact resized value is a multiple of 1/36. The program checks this for every value.

So many exact values lie exactly halfway between two integers. A computation with inexact weights rounds those by the sign of its own error, and that includes the double-precision reference of `common/resize_reference.h`. The reference is therefore computed in integers:
- **The resize:** `exactResizeHalfPixel` in `rpp/e18_box.h`. The value times (2 · 1920) · (2 · 1080) is an integer, so rounding to nearest with ties to even is exact.
- **The box:** round(sum / 9) in integers, with the border replicated.

The double reference rounded with `std::nearbyint` is kept as a cross-check only.

**The values of the exact resize and how each image rounds its ties** (`e18_box_ties_GPU_AMD.csv`, all 6,220,800 values):

| | count |
|---|--:|
| exact ties (k + 18/36) | 636,525 |
| within 1/36 of a tie (k + 17/36, k + 19/36) | 103,272 |
| other values | 5,481,003 |

| image | interior ties rounded up | rounded down | last 2 columns and rows: up / down / neither |
|---|--:|--:|--:|
| exact reference (ties to even) | 317,470 | 316,410 | 844 / 1,801 / 0 |
| double reference (`std::nearbyint`) | 311,743 | 322,137 | 919 / 1,726 / 0 |
| RPP's resize (the intermediate) | 322,662 | 311,218 | 992 / 1,098 / 555 |
| FKL's read chain (identity output) | 322,511 | 311,369 | 1,281 / 1,364 / 0 |

- Of the 633,880 interior ties, RPP rounds 50.9 % up and FKL 50.9 %. Both round in float: RPP through `V_CVT_PK_U8_F32`, which rounds to nearest with ties to even (`inventory/cvt-test/`), and FKL through `SaturateCast`, which also rounds ties to even. The side of a tie is decided by each library's float error in the position and the weights.
- RPP's 555 values that are neither neighbour of their tie are in its clamped last row and column.

### a. The resize alone (`e18_box_agreement_GPU_AMD.csv`, check a)
FKL's resized image is its image Read IOp written by `fk::executeLinearFilter` with the identity coefficients 0 0 0 / 0 1 0 / 0 0 0 (a 3 x 3 `fk::Ptr2D<float3>`) and the same write.

| image | against | region | values differing | at exact ties | within 1/36 | other | max (levels) |
|---|---|---|--:|--:|--:|--:|--:|
| RPP's resize | exact reference | interior | 201,737 | 201,068 | 669 | 0 | 1 |
| RPP's resize | exact reference | last 2 columns and rows | 9,559 | 1,184 | 223 | 8,152 | 196 |
| FKL's read chain | exact reference | interior | 200,675 | 200,389 | 286 | 0 | 1 |
| FKL's read chain | exact reference | last 2 columns and rows | 912 | 907 | 5 | 0 | 1 |
| FKL's read chain | RPP's resize | interior | 15,098 | 14,701 | 397 | 0 | 1 |
| FKL's read chain | RPP's resize | last 2 columns and rows | 8,958 | 583 | 223 | 8,152 | 196 |
| double reference (cross-check) | exact reference | interior | 201,223 | 201,223 | 0 | 0 | 1 |
| double reference (cross-check) | exact reference | last 2 columns and rows | 533 | 533 | 0 | 0 | 1 |

- **FKL's resize equals the exact resize at every value that is neither a tie nor within 1/36 of one.** This holds in both regions, at every one of the 5,481,003 such values.
- **RPP's resize does the same in the interior.** In the last two columns and rows it differs by up to 196 levels, at 8,931 values in the very last row or column. That is the edge clamp: RPP never reads the last column and row of the crop, so its last output column and row take the next-to-last source pixels at full weight. 8,931 of the 8,997 values in the last row or column differ.
- **FKL against RPP:** in the interior they differ only at 14,701 of the 633,880 ties and at 397 values within 1/36 of a tie. In the last two columns and rows they also differ at RPP's clamped last row and column.
- **The double reference** rounds 201,756 ties to the other side from ties-to-even, and agrees with the exact reference everywhere else.
- **Within 1/36 of a tie:** RPP differs at 669 interior values and FKL at 286. A float error larger than 1/36 of a level is needed for that. The cause was not checked; one candidate is the float error of the source position at large x.

### b. The box alone (check b)
The input is one 1920 x 1080 8UC3 image: E4's generator at 1920 x 1080, with the whole image as the ROI. Each library is compared with round(sum / 9), computed in integers with the border replicated.

| | interior | last 2 columns and rows |
|---|--:|--:|
| `rppt_box_filter` (kernelSize 3, REPLICATE) | **0** differing | **0** |
| FKL: `fk::executeLinearFilter` with the box coefficient on a plain read (`PerThreadRead`, `Cast<uchar3, float3>`) | **0** | **0** |

Both box filters equal round(sum / 9) at every value. 9 is odd, so no sum / 9 is a tie, and the nearest rounding boundary is 1/18 of a level away.

### c. The whole pipeline (checks c, `e18_box_attribution_GPU_AMD.csv`)
The reference is the exact resize followed by round(sum / 9).

| output | against | interior: differing, max | last 2 columns and rows: differing (in the last row or column), max |
|---|---|--:|--:|
| `rpp` | exact reference | 183,019, 1 | 17,500 (8,842), 104 |
| `fkl_box_halfpixel` | exact reference | 182,312, 1 | 773 (401), 1 |
| `fkl_box_halfpixel` | `rpp` | 14,737, 1 | 17,513 (8,842), 104 |

**Every difference comes from step (a).** Two checks show it:
- **No difference is added by the box step.** Each output equals round(sum / 9) of its own 8-bit resized image at every value: RPP's output against its intermediate, and FKL's against its identity output. These are required: 0 differing values in both regions.
- **Every differing output value has a differing resized value in its 3 x 3 window.** This holds at all 183,019 + 17,500 values for RPP against the reference, all 182,312 + 773 for FKL against the reference, and all 14,737 + 17,513 for FKL against RPP. It is required.

By (a), those resized differences are:
- ties, or values within 1/36 of a tie, everywhere for FKL and in RPP's interior;
- RPP's edge clamp in its last row and column, which the box spreads over the last two.

### d. The build-once variant and `rpp_ltrb` (check d and the `rpp_ltrb` rows)
- `fkl_half_once` is bitwise equal to `fkl_box_halfpixel` at all 6,220,800 values (required).
- `rpp_ltrb` is bitwise equal to `rpp` at all 6,220,800 values, in both its intermediate and its output (required).
- After the calls, `rpp_ltrb`'s ROI is unchanged (LTRB 640 360 3199 1799), so `rppt_resize` did not convert it (`e18_box_checks_GPU_AMD.csv`).

### The digests
The check run prints a digest (FNV-1a 64) of every output and intermediate, also in `e18_box_checks_GPU_AMD.csv`:
- `rpp` and `rpp_ltrb`: output `9d2e2d8cdf865beb`, intermediate `0c87ef655df949fd`;
- `fkl_box_halfpixel` and `fkl_half_once`: output `e0be817567fd29f1`.

### The fused kernel's arguments
Details 24 B; the two Reads 64 B (the image 48, the coefficient 12); multiply 1; accumulate 1; the write 24. That is 114 B of explicit arguments.

## e. The kernels of every call (`rocprofv3/`)
- **The run:** one run with `E18_PROFILE=1` under `rocprofv3 --kernel-trace --memory-copy-trace --memory-allocation-trace --hip-runtime-trace --marker-trace` (exit 0). It does no checks and no timing.
- **What the profile mode does:** after the setup, one warm-up call of every variant, then three calls of each. Each call is in a ROCTx range named after the variant and followed by a synchronisation of its stream.
- **The analysis:** `rocprof_calls.py` joins every HIP API call made inside a range to its kernels and copies by correlation id, and gives `calls.txt` and `calls.tsv`.
- **No timing:** the trace's timestamps are not a measurement and are not used.

| variant | GPU dispatches per call (the same in all 3 calls) | HIP API calls inside a call |
|---|---|---|
| `rpp` | 3: `roi_conversion_xywh_to_ltrb` (RPP's in-place ROI conversion) → `resize_bilinear_pkd_hip_tensor` → `box_filter_3x3_pkd_hip_tensor` | `hipLaunchKernel` 3, `hipGetLastError` 3 (RPP's own launch checks) |
| `rpp_ltrb` | 2: `resize_bilinear_pkd_hip_tensor` → `box_filter_3x3_pkd_hip_tensor` | `hipLaunchKernel` 2, `hipGetLastError` 2 |
| `fkl_box_halfpixel` | 1: `fk::linearFilterDPPKernel` | `hipLaunchKernel` 1, `hipGetLastError` 1 |
| `fkl_half_once` | 1: the same kernel | `hipLaunchKernel` 1, `hipGetLastError` 1 |

- **`rpp` launches 3 kernels per call,** one more than its two calls suggest, because `rppt_resize` converts its XYWH ROI in a kernel of its own. `rpp_ltrb` launches 2: an LTRB ROI needs no conversion.
- **No call allocates, frees, copies or sets memory:** 0 allocation or free API calls, 0 memory allocations, 0 `hipMemcpy*`, 0 memory copies and 0 `hipMemset*`. The ROI writes are host stores into pinned memory.
- **Before the review,** at f97e949, with the ROI in HIP memory, `rpp` made 4 GPU dispatches per call. The extra one was HIP's blit kernel `__amd_rocclr_copyBuffer` for the 16-byte `hipMemcpyAsync` of the ROI.
- **What the runtime reports for the kernels:**
  - the fused kernel: 32 VGPRs and 4,608 B of LDS, the compiler's 26 and 4,284 rounded up to the allocation granules;
  - RPP's resize: 96 VGPRs, no LDS;
  - RPP's box: 48 VGPRs, 6,144 B of LDS.

## f. The fused kernel on gfx1200 (`isa/`)
- `disassembly.s`: `llvm-objdump -d --symbolize-operands` of the gfx1200 code object.
- `notes.txt`: its metadata (`llvm-readelf --notes`).
- `resource-usage-remarks.txt`: `-Rpass-analysis=kernel-resource-usage` from a recompile of the same source with the build's command (`compile-cmd.txt`).
- `kernel-census.tsv`: the three combined by `isa_census.py`.

| | the fused kernel (`fkl_box_halfpixel`, `fkl_half_once`) |
|---|--:|
| instructions | 649 |
| VGPRs | 26 |
| SGPRs | 38 |
| scratch (private segment), spills | 0 B, 0 / 0 |
| LDS | 4,284 B per block |
| kernarg segment | 376 B, of which 114 B are the five explicit arguments |
| wave size | 32 |
| occupancy (compiler, waves per SIMD) | 16 (the maximum) |
| block, grid | 16 x 8 (128 threads = 4 waves), 120 x 135 |

- **One kernel for both variants:** `fkl_box_halfpixel` and `fkl_half_once` launch the same instantiation.
- **LDS:** the same 4,284 B as the stencil's kernel in `e18-stencil-phase1`. The shared arrays are sized for the largest kernel of the default details, 7 x 7: a 22 x 14 `float3` halo plus 7 x 7 coefficients.
  - The work follows the runtime window, not that size. A 3 x 3 kernel stages 18 x 10 values per 16 x 8 tile, about 1.4 resizes per output pixel, because the halo is `TILE_WIDTH + windowWidth - 1` by `TILE_HEIGHT + windowHeight - 1` (`neighborhood.h:85-86`).
  - So the 7 x 7 maximum changes the LDS size (4,284 B against 2,268 B for a 3 x 3 maximum), not the work.
- **The other kernels of the binary**, in the same table:
  - `gpuSpin`;
  - the identity kernel of check a: 708 instructions, 26 VGPRs, 40 SGPRs;
  - the box kernel of check b: 386 instructions, 14 VGPRs, 30 SGPRs.
- **Occupancy:** the compiler's static value. The achieved occupancy needs a profiler counter run, which was not done.

## Phase 2 (`phase2/`)
The timing is in the program. It follows the RTX 4090 protocol (`../../rtx4090/e18-stencil/`):
- **`whole_call`:** `fklb::timeIt`, events around one call, 100 calls after one warm-up.
- **`kernel_only`:** `fklb::timePreparedLaunch` with the 5 ms `gpuSpin` as `beforeLaunch`. RPP's kernels go behind one spin.
- **The preparation**, timed on the host:
  - `rpp`, `rpp_ltrb`: the ROI write (a host store);
  - `fkl_box_halfpixel`: the crop view, the IOps and the details;
  - `fkl_half_once`: nothing.
- **The order is fixed:** `rpp`, `rpp_ltrb`, `fkl_box_halfpixel`, `fkl_half_once`, each `whole_call` then `kernel_only`.
- **ROCTx ranges:** each interval is in a range named `whole_call:<version>` or `kernel_only:<version>`.
- **The clock warm-up** (added after the first window, 6fca6aa):
  - **The problem:** in the first window the GPU idled at 35-60 MHz during the checks, so `rpp`, the first variant, was timed while sclk ramped up (`../e18-box-window1/`).
  - **The fix:** now the timing file is opened first, then one `gpuSpin` of `E18_CLOCK_WARMUP_MS` (default 2000 ms) runs, then a synchronisation, and the first interval starts at once.
  - **The clock record:** sclk (`fklb::amdGpuClockMHz`, `common/hip_wallclock.h`) is read before and after every interval, outside the timed calls, into the columns `sclk_before_mhz` and `sclk_after_mhz`.
  - **Console lines:** the warm-up's sclk, and the realtime of the warm-up and the timing, for the analysis of the window's `gpu_clocks.csv`.
- **`clock-warmup-test/`:** a diagnostic, not a benchmark. One 2 s single-thread `gpuSpin` with sclk read every 100 ms:
  - sclk rises from 9 MHz to 3,097 MHz in 0.5 s and to 3,258 MHz at 2 s;
  - after the spin it falls to 1,513 MHz within 0.1 s and to 48 MHz within 0.2 s.
  - The timing file is therefore opened before the warm-up: its header runs `amd-smi` twice, 0.07-0.08 s each (`amd-smi-times.txt`).
  - `build.log` holds 6 warnings for unchecked HIP return values; the final `hipGetLastError` reports no error.

The scripts for the window:
- **`window.sh`:**
  - Guards: the binary's sha256 and commit (no `-dirty`), a new output folder, and before every run no `Runner.Worker` and a load average below 0.8.
  - Runs: an optional warm-up run, then three quiet runs, then one complete run under rocprofv3. It samples the GPU clocks every 100 ms.
  - If the runner is not idle within `IDLE_MAX` seconds (default 600) before a run, the window stops there. `GUARD_ONLY=1` runs the guards only and creates nothing; it passed with the binary of `step0/` and refused a wrong commit.
  - It ran the first window (`../e18-box-window1/`, superseded), and runs the second one (`../e18-box/`) with the clock warm-up.
- **`e18_box_report.py`:** the median of the three runs' means with their min-max, and the speedups of FKL over `rpp` and over `rpp_ltrb`.
  - Since 6fca6aa it also gives the lowest sclk at the interval boundaries, and the sclk of `gpu_clocks.csv` during each run's timing (`e18_box_clock.csv`), against 3.1 GHz.
  - It produced the summary of the first window. Its clock part is tested on made-up data only.
- **`../rocprofv3/rocprof_calls.py --spin`:** checks that the spin covers the host's enqueue of every kernel-only call, and gives the gaps. It ran on the first window's rocprofv3 trace: 100 of 100 iterations covered for every variant.

## `inventory/`: the inventory behind the redefinition (2026-11-13)
- `inventory-fkl.md`, `inventory-rpp.md`: the inventory of FKL's DPPs at 2c42a80 and of RPP 3.2.0's filters that led to this pipeline. Read only, as sent on 2026-11-13.
- `cvt-test/`: the one-kernel test of how `V_CVT_PK_U8_F32` (RPP's `rpp_hip_pack`) rounds on gfx1200, with one real `rppt_box_filter` 3 x 3 U8 call. The result: round to nearest, ties to even, saturating, and `rppt_box_filter` 3 x 3 U8 = round(sum / 9). Its `README.txt` gives the build and the files; the binary is not included.
