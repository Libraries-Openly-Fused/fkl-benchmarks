# E18 box, phase 1: correctness, kernels per call and kernel metrics (RTX 4090, CUDA 13.4.3; nothing is timed)

This is the record of the first phase of E18 with a box filter (`opencv-cuda/stencil/e18_box.cu`, target `e18_box_cu`). It covers:
- the checks of every step against exact references;
- the number of kernels and CUDA API calls in one call of every variant;
- Table III's metrics of the fused kernels.

Nothing is timed in this phase. The timing code is in the program, but it was not run. Phase 2 runs it in a window.

**Why a new pipeline.** RPP cannot run the sharpen pipeline of `e18_stencil.cu` (#43). E18 now uses a pipeline that OpenCV-CUDA, NPP, RPP and FKL can all run, in uchar from start to end. The sharpen run stays on main as a record (`results/rtx4090/e18-stencil/`).

**The pipeline**, the same in every library:
- **Input:** one 3840 x 2160 uchar3 frame, E4's seeded source (an LCG with seed 12345, as in `e18_stencil.cu`).
- **Steps:**
  1. Crop 2560 x 1440 at (640, 360).
  2. Resize bilinearly to 1920 x 1080, uchar in and uchar out. The scale is uniform (0.75), and each implementation uses its own sampling convention.
  3. A 3 x 3 box filter, anchor at the centre, border REPLICATE, with uchar3 output.

## Finding: OpenCV-CUDA's and NPP's 8-bit box filters truncate
OpenCV-CUDA's and NPP's 8-bit box filters give **floor(sum / 9)**. RPP's and FKL's round to the nearest.
- **The evidence (check b below):** on the same image, both libraries equal the truncated exact box at all 6,220,800 values. Against the rounded exact box they differ at 2,770,188 values, every one by 1 level below.
- **Why the two libraries agree:** OpenCV's `NPPBoxFilter::apply()` pads the image with `cv::cuda::copyMakeBorder` and then calls `nppiFilterBox_8u_C4R_Ctx` (opencv_contrib 4.14, `modules/cudafilters/src/filtering.cpp:137-175`). OpenCV-CUDA's box is NPP's 8-bit box.
- **What FKL does:** it has a variant of each kind, so every comparison has a variant with the same output.
  - The rounding variants `fkl_box_*` are the FKL code that AMD runs against RPP, which also rounds to the nearest.
  - The truncating variants `fkl_trunc_*` compute what OpenCV-CUDA and NPP compute.
  - The two kinds differ only in the last step: a sum over 9 rounded to the nearest, or truncated.

## The variants
They are described in the header comment of the program. Their order is the fixed pass order of the checks and of the timing.

| version | convention | box | what one call is |
|---|---|---|---|
| `fkl_box_asym` | ASYMMETRIC | rounds | one `fk::executeLinearFilter`; the crop view, every IOp and the details are built in the call |
| `fkl_box_halfpixel` | HALF_PIXEL | rounds | the same, with `fk::ResizeSampling::HALF_PIXEL` |
| `fkl_trunc_asym` | ASYMMETRIC | truncates | one `fk::executeLinearFilter`, built in the call |
| `fkl_trunc_halfpixel` | HALF_PIXEL | truncates | the same, with HALF_PIXEL |
| `opencv_cuda` | ASYMMETRIC (`cv::cuda::resize`) | truncates | GpuMat ROI, `cv::cuda::resize` (8UC3, `INTER_LINEAR`), `cvtColor` BGR to BGRA, the box filter (`createBoxFilter`, CV_8UC4, 3 x 3, anchor (-1, -1), `BORDER_REPLICATE`, created once), `cvtColor` BGRA to BGR |
| `npp` | HALF_PIXEL (`nppiResize`) | truncates | `nppiResize_8u_C3R_Ctx` (`NPPI_INTER_LINEAR`, from the ROI pointer and the frame's step), `nppiFilterBoxBorder_8u_C3R_Ctx` (mask 3 x 3, anchor (1, 1), `NPP_BORDER_REPLICATE`, the resized image as the source, offset 0) |
| `fkl_asym_once`, `fkl_half_once` | as above | rounds | everything built once; the call is `fk::executeLinearFilter` alone |
| `fkl_trunc_asym_once`, `fkl_trunc_halfpixel_once` | as above | truncates | the same for the truncating variants |

**The cuda box filter is 4-channel.** It takes only CV_8UC1, CV_8UC4 and CV_32FC1, so `opencv_cuda` gives the three channels a fourth before the filter and drops it after.

**How each library crops.** The baselines own their buffers: every intermediate is allocated once, outside the calls, and no FKL object is used. OpenCV crops with a GpuMat ROI, NPP with a pointer and a step, and FKL with `Ptr2D::crop2D`.

**FKL's call.** Every FKL variant is one `fk::executeLinearFilter`, with `LinearFilterDPPDetails<float3>` at its defaults: a 16 x 8 tile and a largest kernel of 7 x 7.
- **The image Read IOp:** the read of the crop view, then `fk::Resize` (`INTER_LINEAR`, ASYMMETRIC or HALF_PIXEL) to float3, then `fk::SaturateCast<float3, uchar3>`, then `fk::Cast<uchar3, float3>`.
  - The resized image is therefore rounded to uchar inside the one kernel, as the libraries round it between their two calls.
  - The DPP stages a tile of these values, with their replicated border, in shared memory.
- **Multiply and accumulate:** `fk::Mul<float3, float3, float3, fk::UnaryType>` and `fk::Add<float3, float3, float3, fk::UnaryType>`.
- **The rounding variants:** the coefficient Read IOp is `fk::ConstantFilterRead<float3>` with 1/9 in every channel. The write is `fk::SaturateCast<float3, uchar3>`, then the write of the output.
  - **Why not `fk::executeBoxFilter`:** it does not compile with float3 details at FKL 24478f0. It computes the coefficient as `scale / static_cast<T>(details.kernelWidth * details.kernelHeight)`, and float3 has no conversion from int. The line is `linear_filter.h:325` for CUDA, `:460` for HIP and `:501` for the CPU_MULTITHREAD back-end.
  - The call here passes the arguments that `executeBoxFilter`'s body builds for scale 1: `ConstantFilterRead<float3>` with 1/9, and `Mul` and `Add<float3, float3, float3, UnaryType>`. The argument types are the same, so the kernel is the same one.
- **The truncating variants:** the coefficient Read IOp gives the weight 1 in every channel.
  - The accumulator then holds the exact sum of the 9 values, an integer of at most 2,295, which float represents exactly.
  - The write divides by 9 (`fk::Div<float3>`), truncates to uchar3 (`fk::Cast<float3, uchar3>`), then writes the output.
  - A correctly rounded sum / 9 is exactly k when the sum is 9k, and lies strictly between k and k + 1 otherwise. So the result is floor(sum / 9).
  - This requires the IEEE division. `e18_box_cu` is compiled with `-O3 -lineinfo`, with no fast-math flag, so nvcc's defaults `-prec-div=true` and `-ftz=false` apply.
  - Check (b) confirms it: the truncating box equals floor(sum / 9) at every value.

## Setup
- **Machine:** RTX 4090, nvidia-smi index 1, with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. Driver 610.43.02. CPU AMD Ryzen 9 5950X.
- **Code:** fkl-benchmarks 36233ec, the commit of the program on the branch `e18-box`, based on main fa76331.
  - Submodules: FKL 24478f0 (the merge of FKL #432), cvGPUSpeedup 6d26c65, FastNPP a1b5e245.
  - The CSV headers show 36233ec without -dirty.
- **Build:** preset `local-rtx4090`, nvcc 13.4.92 for sm_89, GNU 15.2.0, Release, OpenCV 4.14.0.
  - Only the target `e18_box_cu` was built: `cmake --build build/local-rtx4090 --target e18_box_cu`.
  - Its only warnings are the 25 deprecation lines for the 4-component 64-bit vector types (`long4`, `ulong4`, `longlong4`, `ulonglong4`, `double4`), which every CUDA program of this repository has.
  - Binary sha256 `1af288033c5f1ef399ec0b428bb49ec03564953ca19a40124d7408559266e04b`.
- **The runs, 2026-11-17 CET:**
  - **The checks:** `scripts-used/run-checks.sh`, `E18_NO_TIMING=1`, 11:57:09-11:57:11. Exit 0, empty stderr, last line "E18 box checks passed".
  - **The nsys pass:** `scripts-used/run-profile.sh`, `E18_PROFILE=1 E18_PROFILE_CALLS=10`, 11:57:11-11:57:15, exit 0.
  - Both scripts source the frozen-program guard first (`refuse_blocked_binaries`).

## The references are exact
- **The resize.** With these sizes every source position is a multiple of 1/3 (ASYMMETRIC) or 1/6 (HALF_PIXEL), so every resized value is an integer over 9 or over 36.
  - The program resizes in integers, so the reference has no rounding error.
  - It rounds to the nearest with ties to even.
- **Why not the double resize** (`common/resize_reference.h`, the reference of #43):
  - HALF_PIXEL has **636,525 exact ties** (weights of 1/2). The double resize misplaces them because its positions are rounded: it lies within 8.0e-11 of the exact values, but its rounding (`std::nearbyint`) differs from the exact ties-to-even rounding at 201,756 values.
  - ASYMMETRIC has no ties. There the double resize is within 7.7e-11 and rounds the same at every value.
  - The double resize stays in the program as a cross-check (`e18_box_checks.csv`).
- **The box.** The integer sum of the 9 values, over 9: rounded to the nearest or truncated. sum / 9 is never a tie, and it is at least 1/18 of a level away from every half level.
- **The exact pipeline:** the exact resize, rounded, then the exact box, rounded or truncated.

## The checks (`e18_box_agreement.csv`, `console.txt`)
Every comparison covers all 1920 x 1080 x 3 = 6,220,800 values. Every difference found is 1 level.
- **Column definitions:**
  - `values_above` and `values_below`: the differing values where the first image is above or below the second.
  - `differing_at_exact_ties`: of the differing values, those where the unrounded exact value of the step is a tie (k + 0.5).
  - `max_distance_to_half_level_off_ties`: the largest distance of the other differing values to a half level, or -1 when there are none.
  - `exact_ties` and `exact_ties_where_version_rounds_up`: all ties of the step, and those where the first image has the upper integer.
  - These last three columns are empty for the pipeline, which has two roundings.

### (a) The resize alone
- **The libraries:** each library's uchar resize is its own intermediate of the pipeline's call, read after the call.
- **FKL's read chain:** the image Read IOp above, through `fk::executeLinearFilter` with the identity weights 0 0 0 / 0 1 0 / 0 0 0 (a 3 x 3 `fk::Ptr2D<float3>`) and the rounding write. It is the resized image that FKL's box filters receive.

| resize | against | values differing | of them at exact ties | the others: largest distance to a half level | ties rounded up |
|---|---|--:|--:|--:|--:|
| `opencv_cuda` (`cv::cuda::resize` 8UC3) | the exact resize, ASYMMETRIC | **0** | | | (no ties) |
| FKL's read chain, ASYMMETRIC | the exact resize, ASYMMETRIC | **0** | | | (no ties) |
| FKL's read chain, ASYMMETRIC | `opencv_cuda`'s resize | **0** | | | |
| `npp` (`nppiResize_8u_C3R`) | the exact resize, HALF_PIXEL | 318,328 (220,312 above, 98,016 below) | 318,037 | 1/36 (291 values) | 440,577 of 636,525 |
| FKL's read chain, HALF_PIXEL | the exact resize, HALF_PIXEL | 201,593 (103,564 above, 98,029 below) | 201,302 | 1/36 (291 values) | 323,816 of 636,525 |
| FKL's read chain, HALF_PIXEL | `npp`'s resize | 116,849 (44 above, 116,805 below) | **116,849** | | |

- **ASYMMETRIC:** OpenCV's and FKL's uchar resizes are exact.
- **HALF_PIXEL:**
  - Both resizes compute in float, so each places the exact ties on either side. This also measures NPP's 8-bit rounding, which the inventory had not: NPP rounds up 69 % of the ties, and FKL rounds up 51 %.
  - Away from the ties, NPP and FKL differ from the exact resize at the same 291 values, all within 1/36 of a level of a tie. FKL and NPP agree everywhere off the ties.
  - FKL and NPP differ only at exact ties: 116,849 values.

### (b) The box alone, on one and the same uchar image
- **The image:** 1920 x 1080 x 3 values of E4's generator (seed 12345), so every level occurs next to every other. Each library has its own copy.
- **The implementations:**
  - OpenCV's path after the resize: `cvtColor`, the same filter object, `cvtColor`;
  - NPP's `nppiFilterBoxBorder_8u_C3R_Ctx`;
  - FKL's two calls (rounding, `fkl_box`; truncating, `fkl_trunc`), with a plain read of the image followed by `fk::Cast<uchar3, float3>` as their image Read IOp.

| box | against round(sum / 9) | against floor(sum / 9) |
|---|--:|--:|
| `fkl_box` | **0** | |
| `fkl_trunc` | | **0** |
| `opencv_cuda` | 2,770,188 differ, all 1 below (the largest distance to a half level 7/18) | **0** |
| `npp` | 2,770,188 differ, all 1 below | **0** |

2,770,188 is 44.5 % of the values: those whose sum leaves a remainder of 5 to 8 when divided by 9.

### (c) The whole pipeline

| version | against the exact pipeline, rounded box | against the exact pipeline, truncated box | against the exact box of its own resize (rounded or truncated, as its box) |
|---|--:|--:|--:|
| `fkl_box_asym`, `fkl_asym_once` | **0** | | **0** |
| `fkl_box_halfpixel`, `fkl_half_once` | 183,078 | | **0** |
| `fkl_trunc_asym`, `fkl_trunc_asym_once` | 2,762,319 (all below) | **0** | **0** |
| `fkl_trunc_halfpixel`, `fkl_trunc_halfpixel_once` | 2,763,971 | 183,571 | **0** |
| `opencv_cuda` | 2,762,319 (all below) | **0** | **0** |
| `npp` | 2,648,272 | 280,340 | **0** |

- **Every difference from the exact pipeline comes from the resize (a), or from the choice between rounding and truncating the box.** Every implementation equals the exact box, of its own kind, of its own resize: 0 differences in the last column. FKL's own resize is its read chain; a library's is its intermediate.
- **FKL against the library of its convention:**

| FKL | against | values differing |
|---|---|--:|
| `fkl_trunc_asym`, `fkl_trunc_asym_once` | `opencv_cuda` | **0: bitwise equal** |
| `fkl_trunc_halfpixel`, `fkl_trunc_halfpixel_once` | `npp` | 115,927 (33 above, 115,894 below) |
| `fkl_box_asym`, `fkl_asym_once` | `opencv_cuda` | 2,762,319 (FKL 1 above at every one: the rounding) |
| `fkl_box_halfpixel`, `fkl_half_once` | `npp` | 2,653,824 (FKL 1 above at every one) |

- **`fkl_trunc_halfpixel` against `npp`:** both equal the truncated exact box of their own resize. Their resizes differ only at the 116,849 exact ties of (a). Every one of the 115,927 differing outputs therefore comes from NPP's and FKL's rounding of those ties.
  - The two counts differ because one resized value enters 9 sums, and it changes a truncated sum / 9 only where it crosses a multiple of 9.

### (d) The build-once variants
`fkl_asym_once`, `fkl_half_once`, `fkl_trunc_asym_once` and `fkl_trunc_halfpixel_once` are each **bitwise equal** to their per-call version (the rows `once`).

**The pass criterion of the program.** These checks must hold, or the program reports a failure and does not time:
- every FKL box against the exact box of its own kind;
- every FKL pipeline against the exact box of its own resize;
- every `_once` variant against its per-call version.

The libraries' results are recorded; they are not criteria.

## (e) Kernels and API calls of one call (`profile/`)
- **How they were counted:**
  - With `E18_PROFILE=1`, every variant is called once outside any range, as a warm-up, so that what a first call allocates is not counted.
    - OpenCV's warm-up call has one `cudaMallocPitch`, the padded image of `copyMakeBorder`.
    - The first calls also load the libraries' kernels (`cuLibraryLoadData`).
  - Then it is called 10 times inside an NVTX range named after it, after a synchronised setup. Nothing is compared or timed.
  - The trace: `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none`, nsys 2026.3.2 of CUDA 13.4.3, with its temporary files in `TMPDIR` on disk.
  - The count: `profile_census.py` on the SQLite export. It joins every kernel to its launch API call by correlation id, and counts everything that started inside each range.

| version | kernels per call | the kernels |
|---|--:|---|
| all eight FKL variants | **1** | `fk::linearFilterDPPKernel` |
| `opencv_cuda` | **5** | `resize_linear`, `transformSimple` (BGR to BGRA), `copy` (`copyMakeBorder`), `ForEachPixelNaiveInLargeImage` (NPP's box), `transformSimple` (BGRA to BGR) |
| `npp` | **2** | `resizeNaive`, `ForEachPixelNaiveInLargeImage` (the box) |

- **No allocation, free, copy or memset inside any call.** In none of the 10 ranges is there an API call that allocates, frees, copies or sets memory, and the GPU ran no memory copy or set for the calls of the ranges.
- **No texture, array or surface objects:** the whole trace has no such API call, the setup and the warm-up calls included.
- **The other API calls in the ranges, per call:**
  - one `cudaLaunchKernel` per kernel;
  - one `cuKernelGetName` per kernel, a driver call that appears next to every launch;
  - for `opencv_cuda`, one `cudaGetDeviceProperties` per call. OpenCV's `NppStreamHandler` builds the NPP stream context from the device properties in every call before it calls NPP (`opencv2/core/private.cuda.hpp:142-163`, CUDA 12.9 and later).
  - the ranges' one `cudaStreamSynchronize` (0.1 per call).

## (f) The fused kernels, Table III's metrics
- **Static data:** cuobjdump of the binary (`kernel-census.tsv`, from `../e18-stencil-phase1/kernel_census.py`, which counts as `scripts/sass_tools.py` counts).
- **Registers, shared memory, occupancy and argument sizes:** the CUDA runtime, from `cudaFuncGetAttributes`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the launch's 128-thread block, and `sizeof` (`e18_box_kernel.csv`).
  - The program queries the kernel instantiation that `fk::executeLinearFilter` launches for these argument types.
  - The binary holds exactly the 8 `linearFilterDPPKernel` instantiations that the program launches, so the queries add none: the four of the table, the two identity-weight kernels of check (a), and the two box-alone kernels of check (b).
  - The runtime's registers and shared memory agree with cuobjdump's.
- **The `_once` variants** launch the same kernels as their per-call versions.

| | fkl_box_asym | fkl_box_halfpixel | fkl_trunc_asym | fkl_trunc_halfpixel |
|---|--:|--:|--:|--:|
| SASS instructions | 800 | 808 | 864 | 872 |
| registers | 40 | 40 | 40 | 40 |
| stack, local memory | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| LDL / STL | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| shared memory (static) | 4,284 B | 4,284 B | 4,284 B | 4,284 B |
| CONSTANT[0] | 472 B | 472 B | 488 B | 488 B |
| instructions with a c[0x0] operand | 51 | 51 | 66 | 66 |
| kernel arguments | 114 B | 114 B | 130 B | 130 B |
| block | 16 x 8 (128 threads) | the same | the same | the same |
| blocks per SM (runtime occupancy calculator) | 12 | 12 | 12 | 12 |
| theoretical occupancy | 100 % (48 of 48 warps) | 100 % | 100 % | 100 % |

- **Arguments:** the sum of `sizeof` of the five.
  - The details: 24 B.
  - The two Reads: 64 B (the image 48 B, the constant coefficient 12 B).
  - Multiply and accumulate: 1 B each.
  - The write: 24 B (rounding) or 40 B (truncating: the divisor, a float3, is a parameter of `fk::Div`).
- **Shared memory:** the halo of the default tile (22 x 14 float3) plus the 7 x 7 coefficients, the defaults of `LinearFilterDPPDetails<float3>`.
- **Parameters:** they are read only as constant-bank operands. There is no indexed load from bank 0 and no local memory, because everything is inlined. `linearFilterDPPKernel` does not declare its parameters `__grid_constant__` (FKL 24478f0, `linear_filter.h:285`), which makes no difference here.
- **Left for phase 2:** the ncu achieved occupancy and duration, in the window.
- **The other kernels of the census** are not part of any timed call: the identity-weight kernels of check (a) (944 and 952 instructions), the box-alone kernels of check (b) (856 rounding, 920 truncating), and `gpuSpin`.

## The timing code (written, not run)
It follows #43's protocol. Phase 2 needs only a window.
- **The order:** the variants in the order of the table above (the per-call versions, then the build-once variants), the same in every run.
- **whole_call:** `fklb::timeIt`, E4's interval (Figure 15). CUDA events surround one call; 100 calls follow one warm-up call.
- **kernel_only:** `fklb::timePreparedLaunch` with `gpuSpin` as `beforeLaunch`, E1's interval (Table III).
  - The preparation is timed on the host: the crop view, the IOps and the details for FKL; the ROI for OpenCV; the ROI pointer for NPP; nothing for the build-once variants.
  - The whole call, every kernel of a multi-kernel baseline included, is enqueued behind one 5 ms spin (`E18_SPIN_US`).
- **NVTX ranges:** every interval of every variant is inside a range named `whole_call:<version>` or `kernel_only:<version>`, for the spin-coverage check of the window's nsys pass.
- **The output:** `e18_box.csv`, with the columns of #43's `e18_stencil.csv`.

## Files
- `console.txt`: the stdout of the checks run.
- `e18_box_agreement.csv`, `e18_box_checks.csv`, `e18_box_kernel.csv`: the program's output in the checks run.
- `profile/`:
  - `e18-box-profile.nsys-rep`, the nsys report;
  - `profile-census.txt`, the output of `profile_census.py`;
  - `profile-console.txt`, the program's stdout under nsys, with nsys's progress lines.
- `kernel-census.tsv`: the census of every kernel of the binary.
- `profile_census.py`: the script that counts the kernels and API calls.
- `scripts-used/run-checks.sh`, `scripts-used/run-profile.sh`: the scripts of the two runs.
- **Left out:**
  - the empty stderr of both runs;
  - the SQLite export of the profile (688 KiB, derived from the `.nsys-rep`) and its export log;
  - the profile run's copy of `e18_box_kernel.csv`, which is identical to the checks run's;
  - the SASS dump;
  - the build log.

  They stay in `scratch/` on this machine. Every file here is the complete output of the program or script named; nothing was filtered.
