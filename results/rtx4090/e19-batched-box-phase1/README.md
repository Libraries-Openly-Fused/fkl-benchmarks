# E19 batched box, phase 1: the checks window e19-checks (RTX 4090, 2026-11-18 10:49:02-10:49:09 +01:00)

`e19_batched_box_cu` at 9eb7abc (`opencv-cuda/stencil/e19_batched_box.cu`), the build of 2026-11-18 10:44 (`build.log`: 0 warnings; build info 9eb7abc without -dirty), sha256 `cad4fc35…` (`binary-sha256.txt`). Nothing was timed and no ncu ran. The window had two steps (`window-script.sh`, `window-console.txt`):
1. **checks/**: one checks run, `E19_NO_TIMING=1`, with `E19_DUMP`. Exit 1: one check failed (below).
2. **nsys/**: one run under the CUDA API and NVTX trace, `E19_PROFILE=1` with 10 calls per row after a warm-up call. Nothing is compared or timed. Exit 0.
   - The script did not stop after step 1's failure. Window scripts now stop at a failed step.

**In short:**
- **The checks run exited 1** because NPP's grey reference failed. The best of the program's six candidate rules still differed at 135 of 1,966,080 values, by 1 level each. None of the six was NPP's rounding.
- **The rule** was then found from the dump, as below: 0 differing values, and the next best differs at 64.
- **Every other check passed.**
- **Step 2 ran after the failure,** because the script did not stop. The script of the re-run window stops at a failed step, as every window script now does.

Left out of this record:
- the raw dump of step 1 (22.5 MiB: each row's grey and box images and its tensor);
- the SQLite export of the nsys report.

Both are on the measuring machine, in `scratch/e19-batched-box/`.

## The checks (checks/e19_batched_box_checks.csv; 30 x 256 x 256 = 1,966,080 values)

| | opencv_cuda | npp |
|---|---|---|
| grey against its reference | 0 differ | **FAILED**: 135 differ (no candidate rule had 0) |
| box against the box of its own grey, floor(sum / 9) | 0 differ | 0 differ |
| box against the box of its own grey, nearest of sum / 9 | 873,678 differ | 873,666 differ |
| float against (v / 255 - 0.45) / 0.225 in double, v its own box value | 1.283e-7, all finite | 3.485e-7, all finite |
| buffers set once: data pointers unchanged | 61 of 61 | (not applicable) |

- **OpenCV's grey** is its fixed point, (9798 R + 19235 G + 3735 B + 2^14) >> 15.
- **The C1 box of both libraries** truncates: floor(sum / 9).
- **NPP's Scale** maps [0, 255] onto [nMin, nMax]. A source range of 256 levels would differ by up to 0.017.
- **NPP's grey against the program's six candidate rules** (differing values; the largest difference is 1 in every row):

| rule | differing |
|---|--:|
| 0.299f R + 0.587f G + 0.114f B, exact sum, truncated | 983,428 |
| exact sum, rounded half up | 450 |
| float, left to right (R, G, B), truncated | 983,144 |
| float, left to right (R, G, B), + 0.5f, truncated | 135 (the fewest: chosen, and failed) |
| (77 R + 150 G + 29 B + 128) >> 8 | 262,927 |
| (9798 R + 19235 G + 3735 B + 2^14) >> 15 | 2,541 |

- **The reported rows of NPP against the reference pipeline** follow from that reference: box 143 values, float 0.0174295 (1 level / 57.375).
- **Between the libraries** (reported only): grey 2,541 values differ, box 2,497, float at most 0.0174296.

### NPP's grey rule, from the dump (`grey-rule-search.txt`, `scripts-used/grey_rule_search.py`; CPU only)
- **The search:**
  - 80 rules in all.
  - The float weights, in every order of the three terms: with each product and sum rounded, or with FMAs, and each with three roundings.
  - Fixed-point rules for shifts 8 to 20.
- **Exactly one rule gives 0 differing values:** t = 0.587f G; t = fmaf(0.299f, R, t); t = fmaf(0.114f, B, t); grey = (int)(t + 0.5f), with the + 0.5f a separate float addition.
  - The next best rule differs at 64 values: the FMA chain in the order G, B, R.
- **The rebuild of the frame and crops** gives OpenCV's grey with 0 differing values, as the run did.

## The kernels per call (nsys; `kernel-counts.txt`, `kernel-order.txt`)
- Every call and every crop launches the same kernels, in the same order.
- No memset or memcpy happens inside a call.
- The durations are taken under the CUDA trace; they are not timings.

| row | kernels per frame (per crop) | per crop, in launch order (median GPU duration) |
|---|--:|---|
| opencv_cuda | 120 (4) | `transformSimple` RGB_to_GRAY (1.15 µs), `copy` BrdReplicate (copyMakeBorder, 1.18 µs), `ForEachLargeTupleByteQuadInLargeImage` LargeFilterBoxReplicateBorder3x3 (2.53 µs), `transformSmart` Convertor (1.09 µs) |
| npp | 90 (3) | `ForEachPixelByteInLargeImage` ColorToGrayFunctor (1.60 µs), `ForEachLargeTupleByteQuadInLargeImage` LargeFilterBoxReplicateBorder3x3 (3.14 µs), `ForEachTupleDWordInLargeImage` ScaleFunctor (1.15 µs) |

- **The runtime API calls per frame:**
  - `opencv_cuda`: 120 `cudaLaunchKernel` and 30 `cudaGetDeviceProperties`, from the NppStreamHandler of each `apply()`.
  - `npp`: 90 `cudaLaunchKernel`.
  - The trace also lists one `cuKernelGetName` per launch.
- **The host's enqueue of one frame** (the call's NVTX range, under the trace):
  - `opencv_cuda`: median 410 µs (405-422);
  - `npp`: median 277 µs (271-300).
  - Both are far below the 5 ms spin of the kernel-only interval.

## The SM clock (`gpu_clocks.csv`, every 100 ms; `clock-readings.txt`)
- **The first readings:** the window starts at P8, 210 MHz.
- **From 10:49:03.570:** the clock reads 2,550 MHz.
  - In P2 at the end of the checks run, then in P0 through the nsys run.
  - 34 readings at 2,550 MHz and none at 2,685 MHz.
  - Utilisation reads 0-8 %.
