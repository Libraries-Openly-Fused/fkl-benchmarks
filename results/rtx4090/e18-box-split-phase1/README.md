# E18 box split, phase 1: the program, its checks and its kernels (RTX 4090, CUDA 13.4.3; nothing is timed)

This is the record of the first phase of the E18 box split (`opencv-cuda/stencil/e18_box_split.cu`, target `e18_box_split_cu`). It covers:
- the checks of the seven rows against exact references, and of the three repeated rows against the window e18-box;
- the kernels of the binary from cuobjdump, and their SASS against the binary of the window e18-box.

Nothing is timed in this phase. The timing code is in the program, but it was not run. Phase 2 runs it in a window.

**The question.** In the window e18-box (`../e18-box/`), `npp` (two calls) took less time than `fkl_trunc_halfpixel` (one fused kernel) for the same output: 0.85x in the whole call. The split times each step alone, in both libraries:
- the box alone: FKL's box filter kernel against NPP's;
- the resize alone: FKL's resize against NPP's.

The fused kernel computes the resized value of every position of each tile's halo: a 16 x 8 tile with the 3 x 3 box reads 18 x 10 resized values (`neighborhood.h:85-86` at FKL 24478f0). The split rows run each step once per output pixel.

**The binary:**
- `e18_box_split_cu` at fkl-benchmarks 62202d1, on branch e18-box-split from main d1ae342. Its submodules are those of main: FKL 24478f0, FastNPP a1b5e245, cvGPUSpeedup 6d26c65.
- Built with the preset `local-rtx4090` (nvcc 13.4.92, sm_89, `-O3 -lineinfo`, no fast-math flag). The build printed 25 warnings, the same as `e18_box_cu`'s build: CUDA's deprecated 4-component vector types.
- sha256 `5b4c823103e91ea8666c98621a09ed2527d2799a47a65d5f221cc1671f4e507d` (`binary-sha256.txt`). The binary carries the commit 62202d1, not `-dirty`.

## The pipeline and the rows
The frame, the crop, the sizes, the NPP stream context, the references and the timing are those of `e18_box.cu`. The program copies its code.
- **The frame:** 3840 x 2160 uchar3, E4's seeded source (an LCG with seed 12345).
- **The crop:** 2560 x 1440 at (640, 360).
- **The resize:** bilinear to 1920 x 1080, uchar in and uchar out.
- **The box:** 3 x 3, anchor at the centre, border REPLICATE, uchar3 out.

The rows are in a fixed order, the same in every run. It is the order of the checks and of the timing.

| # | row | what one call is |
|--:|---|---|
| 1 | `fkl_box_asym` | `e18_box.cu`'s `fkl_box_asym`, with its code: one `fk::executeLinearFilter` (resize ASYMMETRIC, box rounded to the nearest). It comes first, as in the window e18-box |
| 2 | `fkl_trunc_halfpixel` | `e18_box.cu`'s `fkl_trunc_halfpixel`, with its code: one `fk::executeLinearFilter` (resize HALF_PIXEL, box truncated: floor(sum / 9)). Its output equals NPP's box of FKL's resize |
| 3 | `npp` | `e18_box.cu`'s `npp`, with its code: `nppiResize_8u_C3R_Ctx` to NPP's intermediate, then `nppiFilterBoxBorder_8u_C3R_Ctx` |
| 4 | `npp_resize` | `npp`'s first call alone: `nppiResize_8u_C3R_Ctx` (`NPPI_INTER_LINEAR`) from the crop to a 1920 x 1080 uchar3 image of its own |
| 5 | `fkl_resize` | FKL's resize alone, one `fk::executeOperations` (`fk::TransformDPP`, thread fusion off): the read of the crop view (`Ptr2D::crop2D`), `fk::Resize` (`INTER_LINEAR`, HALF_PIXEL) to float3, `fk::SaturateCast<float3, uchar3>`, then the write of a 1920 x 1080 uchar3 `fk::Ptr2D`. This is the image Read IOp of row 2 without the cast back to float3 |
| 6 | `npp_box` | `npp`'s second call alone: `nppiFilterBoxBorder_8u_C3R_Ctx` (mask 3 x 3, anchor (1, 1), `NPP_BORDER_REPLICATE`, `oSrcSize` the whole image, offset (0, 0)) on a 1920 x 1080 uchar3 image of its own |
| 7 | `fkl_box` | row 2's call with the read of a 1920 x 1080 uchar3 `fk::Ptr2D`, then `fk::Cast<uchar3, float3>`, as its image Read IOp. The weights (1), the write (`fk::Div<float3>` by 9, then `fk::Cast<float3, uchar3>`) and the details (`LinearFilterDPPDetails<float3>` at its defaults, the 16 x 8 tile) are row 2's |

- **The input of rows 6 and 7** is one and the same image: the exact HALF_PIXEL resize of the crop, rounded to the nearest with ties to even. It is the program's reference. It is uploaded once, before any call: into an NPP buffer (`cudaMallocPitch`) and into an `fk::Ptr2D`.
- **Every row writes its own output.**
- **Rows 5 and 7 build their view, their IOps and their details in every call**, as the per-call variants of `e18_box.cu` do. There are no build-once rows.
- **The NPP rows own their buffers and use no FKL object.** NPP crops with a pointer into its frame and the frame's step; FKL crops with `Ptr2D::crop2D`.
- **No helper functions** stand between the rows and the library calls.

## The checks (one run, `E18_NO_TIMING=1`)
The run started at 16:25:52 and ended at 16:25:53 (+01:00), with exit code 0. It printed "E18 box split checks passed" (`console.txt`, `run-header.txt`).
- **When it ran:** once, after the compile, on the coordinator's grant (option 1: one checks-only run before the window). Nothing was timed, and no profiler ran.
- **The CSVs:** `e18_box_split_agreement.csv` and `e18_box_split_checks.csv`. Every comparison covers all 1920 x 1080 x 3 = 6,220,800 values.
- **Which checks gate the timing:** the program times only when all of these pass:
  - the five equalities of the new rows below;
  - the FKL repeats against the exact box of their own resize.

**The new rows:**

| comparison | values that differ | |
|---|--:|---|
| `npp_resize` against `npp`'s intermediate | 0 | bitwise equal: the same call |
| `npp_resize` against the exact resize (HALF_PIXEL) | 318,328 | as `npp`'s intermediate; 318,037 of them at exact ties |
| `fkl_resize` against FKL's read chain of HALF_PIXEL (the check of `e18_box.cu`) | 0 | bitwise equal |
| `fkl_resize` against the exact resize | 201,593 | unchanged from the window e18-box; 201,302 of them at exact ties |
| `fkl_resize` against `npp_resize` | 116,849 | all at exact ties |
| `npp_box` against the exact box, truncated, of the input | 0 | floor(sum / 9) at all 6,220,800 values |
| `fkl_box` against the exact box, truncated, of the input | 0 | the same |
| `fkl_box` against `npp_box` | 0 | bitwise equal |

**FKL's read chain** is the image Read IOp of the fused kernels, run through `fk::executeLinearFilter` with the identity weights 0 0 0 / 0 1 0 / 0 0 0, as in `e18_box.cu`.

**The repeats against the window e18-box** (`compare-repeats.txt`):
- The program's output was compared field by field with the run quiet-run1 of `../e18-box/` (binary of 7dca510): every row that `e18_box.cu` also writes, and every exact-reference check.
- All 13 agreement rows and all 8 checks are the same in every field. Some examples:
  - `fkl_trunc_halfpixel` against `npp`: 115,927 values differ;
  - `fkl_trunc_halfpixel` against the exact box, truncated, of its own resize: 0;
  - `npp` against the exact resize: 318,328;
  - HALF_PIXEL has 636,525 exact ties.
- **Labels:** the split program calls `npp`'s resized image `npp_intermediate`. `e18_box.cu` calls it `npp` (as a version of the resize step) and `npp_resize` (as what it is compared against). The comparison maps these labels; every other label is the same.
- **The command:** `scripts-used/compare_repeats.py e18_box_split_agreement.csv e18_box_split_checks.csv ../e18-box/quiet-run1/e18_box_agreement.csv ../e18-box/quiet-run1/e18_box_checks.csv`.

## The kernels (cuobjdump; nothing run)
`kernel-table.tsv` comes from `scripts-used/kernel_table.py`. It takes:
- the instructions as `scripts/sass_tools.py` counts them;
- REG, STACK, SHARED, LOCAL and CONSTANT[0] from `cuobjdump -res-usage`.

It also compares each kernel's normalised SASS with the SASS of `e18_box_cu` of 7dca510, the binary of the window e18-box (sha256 af354faf…). That dump is kept on this machine; its census equals `../e18-box/kernel-census.tsv`. The normalisation removes the addresses, the encodings and cuobjdump's column padding; it keeps every instruction, operand and label. Kernels are matched by mangled name. `gpuSpin` is matched by demangled name, because its anonymous namespace carries its source file's name.

| kernel | row | SASS instructions | registers | static shared memory | local memory | stack | LDL / STL | CONSTANT[0] | SASS against 7dca510 |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| `linearFilterDPPKernel` | `fkl_box_asym` | 800 | 40 | 4,284 B | 0 | 0 | 0 / 0 | 472 B | identical |
| `linearFilterDPPKernel` | `fkl_trunc_halfpixel` | 872 | 40 | 4,284 B | 0 | 0 | 0 / 0 | 488 B | identical |
| `launchTransformDPP_Kernel` | `fkl_resize` | 208 | 31 | 0 | 0 | 0 | 0 / 0 | 432 B | not in that binary (new) |
| `linearFilterDPPKernel` | `fkl_box` | 920 | 38 | 4,284 B | 0 | 0 | 0 / 0 | 464 B | identical to `e18_box_cu`'s box-alone truncating check kernel: the same instantiation |
| `linearFilterDPPKernel` | the read chain check, ASYMMETRIC / HALF_PIXEL (identity weights; not timed) | 944 / 952 | 40 | 4,284 B | 0 | 0 | 0 / 0 | 480 B | identical |
| `gpuSpin` | the spin of kernel_only | 24 | 8 | 0 | 0 | 0 | 0 / 0 | 360 B | identical |

- **The binary holds these 7 kernels and no other.**
- **`fkl_box`'s kernel** is the instantiation that `e18_box.cu` launches in its box-alone check (b) with the truncating write. That check ran on a different image (seeded noise). Here the kernel reads the exact HALF_PIXEL resize.
- **`fkl_resize`'s kernel** is `fk::launchTransformDPP_Kernel<GPU_NVIDIA, TF::DISABLED, true, …>`, launched with FKL's default block for a 1920 x 1080 grid of threads.
- **`kernel-census.tsv`** is the census of every kernel by `../e18-stencil-phase1/kernel_census.py`: constant-bank operands, LDL/STL, REG, STACK, LOCAL, CONSTANT[0]. It is the format of the other E18 records.
- **No ncu** in this phase.

## The timing code (written, not run)
It is `e18_box.cu`'s: the protocol of the window e18-box.
- **whole_call:** `fklb::timeIt`. CUDA events surround one call; 100 calls follow one warm-up call.
- **kernel_only:** `fklb::timePreparedLaunch` with `gpuSpin` (5 ms, `E18_SPIN_US`) as `beforeLaunch`.
  - The preparation is timed on the host: for the FKL rows, the view, the IOps and the details; for `npp` and `npp_resize`, the ROI pointer; for `npp_box`, nothing.
  - The whole call is enqueued behind the spin.
- **The order:** the seven rows in the order of the table, the same in every run.
- **NVTX ranges:** `whole_call:<row>` and `kernel_only:<row>`, for the nsys pass.
- **The output:** `e18_box_split.csv`, with the columns of `e18_box.csv`.
- **The same environment variables as `e18_box.cu`:** `E18_ONLY`, `E18_NO_TIMING`, `E18_SPIN_US`, `E18_PROFILE` and `E18_PROFILE_CALLS`.

## Files
- **The run:**
  - `console.txt`: the program's stdout;
  - `run-header.txt`: the run script's start and end lines;
  - `exit_code.txt`;
  - `gpu-compute-apps-before.csv`: the processes on the GPUs at the start, the desktop's.
- **The program's output:** `e18_box_split_agreement.csv`, `e18_box_split_checks.csv`.
- **The binary:** `binary-sha256.txt`.
- **The kernels:** `kernel-table.tsv`, `kernel-census.tsv`.
- **The repeats:** `compare-repeats.txt`.
- **`scripts-used/`:** `run-checks.sh` (the checks run, with the blocked-programs guard), `kernel_table.py`, `compare_repeats.py`.
- **Left out:**
  - the empty stderr of the run;
  - the SASS dumps of this binary and of the 7dca510 binary;
  - the configure and build logs.

  They stay in `scratch/` on this machine. Every file here is the complete output of the program or script named; nothing was filtered.
