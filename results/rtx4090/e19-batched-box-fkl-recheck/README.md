# E19 batched box, phase 2: the re-check window e19-fkl-recheck (RTX 4090, 2026-11-18 11:13:58-11:14:04 +01:00)

`e19_batched_box_cu` at 69df252, which adds two commits to the code of the window e19-fkl-checks (`../e19-batched-box-fkl-checks/`):
- **11bac03 (commit A):** fkl_loop's reference takes FKL's actual grey rule. That rule is the kernel's FMA chain t = 0.587f G, t = fma(0.299f, R, t), t = fma(0.114f, B, t), rounded to the nearest integer with ties to even. `fk::RGB2Gray` rounds even with a float output (`color_conversion.h:55-69`: `std::is_signed_v<float>` is true). A host-only check (`scripts-used/fkl_rule_table.cpp`, `fkl_compare.py`) finds 0 of the 2^24 RGB values where the program's rule differs from the emulation that fitted e19-fkl-checks' dump.
- **69df252 (commit B):** the checks and the timing now run in the fixed order fkl_loop, opencv_cuda, npp. The checks' host references are now computed before the first GPU call.
- **What did not change:** no library call. The SASS of the binary is byte-identical to f487b3f's.

**The build:** 11:12, with 25 warnings, all of them CUDA 13 deprecations inside FKL's headers (`build.log`). The build info is 69df252 without -dirty, and the sha256 is `2aa1486c…` (`binary-sha256.txt`).

The window had two steps (`window-script.sh`, which stops at a failed step):
1. **checks/**: one checks run, `E19_NO_TIMING=1`, with `E19_DUMP`. Exit 0: "E19 batched box checks passed".
2. **nsys/**: one run under the CUDA API and NVTX trace, `E19_PROFILE=1` with 10 calls per row after a warm-up call. Nothing is compared or timed. Exit 0.

Nothing was timed, and no ncu ran. Two things are left out of this record, both on the measuring machine in `scratch/e19-batched-box/`: the dump (30.0 MiB, `dump-p2b/`) and the SQLite export of the nsys report (`nsys-export-p2b/`).

## The checks (checks/e19_batched_box_checks.csv)
- **Every check passed.**
- **The 24 rows that the window e19-fkl-checks also had are identical, value for value:** the 22 rows of opencv_cuda and npp and fkl_loop's two rows against the libraries' tensors.
- **The one new row** is fkl_loop against its corrected reference:

| fkl_loop compared with | values beyond 1e-5 | largest difference | not finite | ok |
|---|--:|--:|--:|---|
| fklReference (FKL's grey rule, the box and the conversion in double) | 0 | 6.37e-7 | 0 | yes |
| opencv_cuda's tensor (reported) | | 0.0193662 | 0 | |
| npp's tensor (reported) | | 0.0154930 | 0 | |

- **The dump:** all 7 files are byte-identical to the dump of the window e19-fkl-checks (`dump-comparison.txt`). That is fkl_loop's tensor and the grey, box and tensor of opencv_cuda and npp. The baselines' files therefore equal phase 1's as well.

## The kernels per call (nsys; `kernel-counts.txt`, `kernel-order.txt`)
- Every call and every crop launches the same kernels, in the same order.
- No memset or memcpy happens inside a call.
- The durations are taken under the CUDA trace; they are not timings.

| row | kernels per frame (per crop) | per crop (median GPU duration) | host enqueue of one frame (median, min-max) |
|---|--:|---|--:|
| fkl_loop | 30 (1) | `linearFilterDPPKernel` (grid 16 x 32, block 16 x 8; 2.08 µs) | 93.4 µs (91.3-107.8) |
| opencv_cuda | 120 (4) | `transformSimple` 1.15, `copy` 1.18, `ForEachLargeTupleByteQuadInLargeImage` 2.53, `transformSmart` 1.09 µs | 405.6 µs (398.3-434.5) |
| npp | 90 (3) | `ForEachPixelByteInLargeImage` 1.60, `ForEachLargeTupleByteQuadInLargeImage` 3.14, `ForEachTupleDWordInLargeImage` 1.15 µs | 272.3 µs (265.4-300.8) |

- **The host enqueue** is the length of the call's NVTX range, measured under the trace. All three rows are far below the 5 ms spin of the kernel-only interval.
- **The runtime API calls per frame:** 30, 120 and 90 `cudaLaunchKernel`, plus `opencv_cuda`'s 30 `cudaGetDeviceProperties`.

## The SM clock (`gpu_clocks.csv`, every 100 ms; `clock-readings.txt`)
- **11:13:58.626:** 210 MHz, P8.
- **From 11:13:59.634 on:** 2,550 MHz, through the end of the checks run and the whole nsys run. That is 31 readings, in P0 and P2, and none at 2,685 MHz.
- **The readings at the end of the runs:**
  - checks: 210 MHz (`checks/gpu-at-end.csv`);
  - nsys: 2,550 MHz (`nsys/gpu-at-end.csv`).
