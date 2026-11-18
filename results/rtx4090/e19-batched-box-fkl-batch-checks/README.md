# E19 batched box, phase 3: the checks window e19-fkl-batch-checks (RTX 4090, 2026-11-18 17:28:49-17:28:55 +01:00)

`e19_batched_box_cu` at 8ff14e5. That commit adds the row **fkl_batch** to the code of the window e19-fkl-recheck (`../e19-batched-box-fkl-recheck/`, 69df252):
- **The call:** one `fk::executeBoxBlur<1>` per frame (morphology.h, MorphologyDPP, GPU_NVIDIA). That is one kernel, with grid z = the planes of the input, here the batch of 30.
- **The read:** a BatchRead of the 30 crop views (`fk::PerThreadRead<_2D, uchar3>::build` of a `std::array` of the 30 `crop2D`), then `.then()` fkl_loop's grey op.
- **The details:** `fk::MorphologyDPPDetails<float>` with its defaults: 256 x 256, mask 3 x 3, anchor (1, 1).
- **The write:** fkl_loop's Div 255, Sub 0.45 and Div 0.225, then one `fk::PerThreadWrite<_3D, float>` of FKL's whole tensor. The plane comes from Point.z.
- **The order** of the checks is now fkl_loop, fkl_batch, opencv_cuda, npp.
- **What did not change:** no call and no reference of the other rows.
  - The SASS of `linearFilterDPPKernel` (fkl_loop) and of `gpuSpin` is byte-identical to the 69df252 build. 79779b7, the record commit of #51, has the same source (`sass-compare.txt`, from `scripts-used/sass_compare.py`).
  - Only gpuSpin's anonymous-namespace hash in its mangled name changes, because it follows the file's content.
  - The OpenCV and NPP kernels are in their libraries.

**The build:** 17:24, with 25 warnings, all of them CUDA 13 deprecations inside FKL's headers, the same 25 lines as the 69df252 build (`build.log`). The build info is 8ff14e5 without -dirty, and the sha256 is `e989a41e…` (`binary-sha256.txt`). `kernel-census.tsv` comes from `../e18-stencil-phase1/kernel_census.py`.

The window had two steps (`window-script.sh`, which stops at a failed step):
1. **checks/**: one checks run, `E19_NO_TIMING=1`, with `E19_DUMP`. Exit 0: "E19 batched box checks passed".
2. **nsys/**: one run under the CUDA API and NVTX trace, `E19_PROFILE=1` with 10 calls per row after a warm-up call. Nothing is compared or timed. Exit 0.

Nothing was timed, and no ncu ran.

**The folder name:** the script wrote into `results/rtx4090/e19-fkl-batch-checks/`. That folder was renamed to this one after the window, before the commit. The paths in `window-script.sh`, `window-console.txt` and `nsys/stdout.txt` show the old name. No file was changed.

**Left out of this record:** two things, both on the measuring machine in `scratch/e19-fkl-batch/`. One is the dump (37.5 MiB, `dump/`; its sha256 sums are in `dump-comparison.txt`). The other is the SQLite export of the nsys report (`nsys-export/`).

## The checks (checks/e19_batched_box_checks.csv)
- **Every check passed.**
- **The 25 rows that the window e19-fkl-recheck also had are identical, value for value.** These are the 22 rows of opencv_cuda and npp and fkl_loop's 3 rows.
- **The 3 new rows** are fkl_batch's:

| fkl_batch compared with | criterion | values beyond 1e-5 / left | largest difference | not finite | ok |
|---|---|--:|--:|--:|---|
| fklBatchReference (FKL's grey rule; float(sum of 9 x float(1/9)); the conversion in double) | at most 1e-5, every value finite | 0 | 2.62e-7 | 0 | yes |
| the fill (values still 0xFFFFFFFF) | 0 left | 0 of 1,966,080 | | | yes |
| fkl_loop's tensor (reported) | | | 7.75e-7 | 0 | |

- **fkl_batch against fkl_loop:** fkl_loop multiplies each of the 9 taps by 1 / 9 and then sums the 9 products in float. fkl_batch sums the 9 integer grey values exactly and multiplies once. 7.75e-7 is within fkl_loop's bound of about 1.7e-6 plus fkl_batch's 5.1e-7.
- **The dump:** the 7 files of the old rows are byte-identical to the dump of the window e19-fkl-recheck (`dump-p2b`, `dump-comparison.txt`). These are fkl_loop's tensor and the grey, box and tensor of opencv_cuda and npp. fkl_batch's tensor is a new file.

## The kernels per call (nsys; `kernel-counts.txt`, `kernel-order.txt`, `fkl-batch-grid.txt`)
- Every call launches the same kernels. No memset or memcpy happens inside a call.
- The durations are taken under the CUDA trace; they are not timings.

| row | kernels per frame | kernel (grid; block; median GPU duration) | host enqueue of one frame (median, min-max) |
|---|--:|---|--:|
| fkl_loop | 30 | `linearFilterDPPKernel` (16 x 32 x 1; 16 x 8; 2.08 µs per crop) | 93.0 µs (91.6-106.7) |
| fkl_batch | 1 | `morphologyDPPKernel` (16 x 32 x 30; 16 x 8; 40 registers, 1,232 B static shared; 20.99 µs, min 20.83, max 21.54) | 4.1 µs (4.0-5.3) |
| opencv_cuda | 120 | `transformSimple` 1.15, `copy` 1.18, `ForEachLargeTupleByteQuadInLargeImage` 2.53, `transformSmart` 1.09 µs per crop | 401.8 µs (397.9-412.0) |
| npp | 90 | `ForEachPixelByteInLargeImage` 1.60, `ForEachLargeTupleByteQuadInLargeImage` 3.14, `ForEachTupleDWordInLargeImage` 1.15 µs per crop | 276.5 µs (271.7-303.9) |

- **The host enqueue** is the length of the call's NVTX range under the trace. All four rows are far below the 5 ms spin of the kernel-only interval.
- **The runtime API calls per frame:** 30, 1, 120 and 90 `cudaLaunchKernel`, plus opencv_cuda's 30 `cudaGetDeviceProperties`.
- **The per-crop listing doesn't fit fkl_batch:** `kernel-order.txt` lists the kernels of the first crop and assumes 30 crops per call. fkl_batch has one kernel for the whole frame, so that listing is empty for it and its line "every crop of every call launches the same kernels in this order: NO" doesn't apply. `fkl-batch-grid.txt` (`scripts-used/fkl_batch_grid.py`) gives its kernel and grid.

## The SM clock (`gpu_clocks.csv`, every 100 ms; `clock-readings.txt`)
- **17:28:49.712:** 210 MHz, P8.
- **From 17:28:50.419 on:** 2,550 MHz through the rest of the checks run and the whole nsys run. That is 35 readings, in P0 and P2, and none at 2,685 MHz.
- **The readings at the end of both runs:** 2,550 MHz (`checks/gpu-at-end.csv`, `nsys/gpu-at-end.csv`).
