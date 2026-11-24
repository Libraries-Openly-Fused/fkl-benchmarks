# E19 RX 9060 repeat check (e19-repeat-check), 2026-11-24, 19:01:11-19:02:14Z, RX 9060

The question: does the change of rpp's kernel_only at N = 30 follow the program or the window? The change is between E19's timing window (`../e19-batched-box/`, 2026-11-18, 0.2010 ms) and the crop sweep's (`../e19-crop-sweep/`, 2026-11-24 07:31-07:45Z, 0.1858 ms, -7.6 %, flagged).

**The answer, by the reading rule fixed before the window: the change does not follow the program.**
- In this window, B (the crop sweep's program at N = 30) agrees with A (E19's own binary) within their spreads in all four rows: -0.5 % to +0.1 %.
- A itself, the binary E19 timed, now gives rpp kernel_only 0.1860 ms. That is -7.5 % against E19's window, the same change the crop sweep showed.
- No further cause is claimed. Why E19's window gave 0.2010 ms is not determined.

- **`fkl_loop` is not compared** (Oscar, 2026-11-20). It is timed first in every run only to keep E19's row order. Its numbers are in the raw CSVs and in no table here.
- **Decision:** the coordinator session's task of 2026-11-24, given after the crop sweep's N = 30 repeat flag. The coordinator session verified the window script at 8f870f2 (sha256 `93ddfcf2…`) and granted the window the same day.

## What ran
- **The script:** `window.sh`, as committed at 8f870f2. The copy the script made of itself in the window folder is identical, and its sha256 is `93ddfcf2617ea3647505eff8876869747bc8136f3cf31d79f4db5b53ad8ea3c1`.
  - It was run as one foreground call. This session made no other tool call until the call returned, and no subagent was running.
  - The foreground check at 19:01:11Z found no `Runner.Worker` and a load of 0.25.
- **The programs** (`binaries.txt`, both FKL 24478f0, RPP 3.2.0):
  - **A:** `e19_batched_box_hip`, the binary E19's window timed: sha256 `a193bae62d1e2d76d83e5d3a66f867c4d874c1e40b9c2c2ee76f40a94bb8f4d1`, commit 2ad7ac5. It ran with no environment variable.
  - **B:** `e19_crop_sweep_hip`, the binary the crop sweep timed: sha256 `f9fcb2a43e96be2249734491e6fecf8e3baadc3484d0b8ccba895559a2192ee8`, commit c8eb575. It ran with `E19_CROPS=30`.
- **The runs, in this order.** Each run is complete: the checks, then the 2 s `gpuSpin`, then the timing in E19's order and protocol.
  1. `unused-A`, `unused-B`: not used;
  2. `quiet-run1-A`, `quiet-run1-B`, `quiet-run2-A`, `quiet-run2-B`, `quiet-run3-A`, `quiet-run3-B`;
  3. `rocprofv3-A`, `rocprofv3-B`: kernel, HIP runtime, ROCTx, memory copy and allocation traces. Their times are not used.
- **The outcome:** all 10 runs exited 0 and printed "E19 checks passed". Each took 4-6 s. No CSV has `INCOMPLETE` or `-nan`.
- **The guards:**
  - Before the start: both binaries' names, sha256 and commits, neither carrying `-dirty`, and a new output folder.
  - Before every run: no `Runner.Worker` and a load average below 0.8.
  - Only `rocprofv3-B` waited, 15 s at a load of 0.78.
- **The sampler:** `gpu_clocks.csv` reads sclk, mclk, `power1_average` and the edge, junction and memory temperatures from hwmon every 100 ms.
- **The summary** (`summary/`) was made after the window, without the GPU, by `analysis/repeat_check_report.py`. `summary/report.md` has its tables.
  - It is reproduced byte for byte from this folder with `repeat_check_report.py e19-repeat-check e19-batched-box e19-crop-sweep <out>`.
- **The traces:** `e19_kernel_trace.csv` and `e19_hip_api_trace.csv` are gzipped (`gzip -9`). The sha256 of the uncompressed files is in `rocprofv3-*/trace-csv-sha256.txt`.

## The times (the median of the three quiet runs' means, min-max, ms)
| program | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |
|---|--:|--:|--:|--:|
| A (E19's binary) | 0.1098 (0.1096-0.1101) | 0.1018 (0.1017-0.1018) | 0.1854 (0.1850-0.1861) | 0.1860 (0.1842-0.1868) |
| B (the sweep's, N = 30) | 0.1093 (0.1092-0.1116) | 0.1018 (0.1017-0.1019) | 0.1849 (0.1824-0.1866) | 0.1859 (0.1798-0.1860) |

The spreads ((max - min) / median of the three means), whole_call / kernel_only:
- A: fkl_batch 0.5 % / 0.1 %, rpp 0.6 % / 1.4 %;
- B: fkl_batch 2.2 % / 0.1 %, rpp 2.3 % / 3.4 %.

## The repeats (`summary/compare.csv`, `summary/bound.csv`)
The criterion for GPU- or host-bound is the crop sweep's: a row is GPU-bound when its kernel_only and whole_call medians differ by less than 3 %.
- rpp is GPU-bound in both programs: kernel_only is +0.35 % (A) and +0.49 % (B) of whole_call.
- fkl_batch's kernel_only is 7.3 % (A) and 6.8 % (B) below its whole_call. The criterion labels that host-bound, as in E19 and the crop sweep.

**B against A, in this window (the reading rule's comparison):**
| row, interval | A (ms) | B (ms) | change | spread A / B | exceeds the larger spread? | > 3 %? | cause | GPU- or host-bound |
|---|--:|--:|--:|--:|---|---|---|---|
| fkl_batch whole_call | 0.1098 | 0.1093 | -0.5 % | 0.5 / 2.2 % | no | no | none needed | host |
| fkl_batch kernel_only | 0.1018 | 0.1018 | +0.1 % | 0.1 / 0.1 % | no | no | none needed | host |
| rpp whole_call | 0.1854 | 0.1849 | -0.2 % | 0.6 / 2.3 % | no | no | none needed | GPU |
| rpp kernel_only | 0.1860 | 0.1859 | -0.1 % | 1.4 / 3.4 % | no | no | none needed | GPU |

**A against E19's window: the same binary, six days apart.**
| row, interval | old: E19 (ms) | new: A (ms) | change | spread old / new | exceeds the larger spread? | > 3 %? | cause | GPU- or host-bound (new) |
|---|--:|--:|--:|--:|---|---|---|---|
| fkl_batch whole_call | 0.1136 | 0.1098 | -3.4 % | 0.8 / 0.5 % | yes | FLAG | not determined; not the program | host |
| fkl_batch kernel_only | 0.1029 | 0.1018 | -1.1 % | 0.7 / 0.1 % | yes | no | not determined; not the program | host |
| rpp whole_call | 0.1887 | 0.1854 | -1.8 % | 0.4 / 0.6 % | yes | no | not determined; not the program | GPU |
| rpp kernel_only | 0.2010 | 0.1860 | -7.5 % | 0.2 / 1.4 % | yes | FLAG | not determined; not the program | GPU |

In E19's window, rpp's kernel_only was 6.5 % above its whole_call (0.2010 against 0.1887 ms). In this window the two are within 0.5 %.

**B against the crop sweep's N = 30: the same binary, about 11 hours apart.**
| row, interval | old: sweep (ms) | new: B (ms) | change | spread old / new | exceeds the larger spread? | > 3 %? | cause | GPU- or host-bound (new) |
|---|--:|--:|--:|--:|---|---|---|---|
| fkl_batch whole_call | 0.1116 | 0.1093 | -2.1 % | 1.7 / 2.2 % | no | no | none needed | host |
| fkl_batch kernel_only | 0.1018 | 0.1018 | -0.0 % | 0.2 / 0.1 % | no | no | none needed | host |
| rpp whole_call | 0.1850 | 0.1849 | -0.0 % | 2.7 / 2.3 % | no | no | none needed | GPU |
| rpp kernel_only | 0.1858 | 0.1859 | +0.1 % | 2.9 / 3.4 % | no | no | none needed | GPU |

## The kernels in the traces (`analysis/kernel-durations.txt`)
The durations were taken by `../e19-crop-sweep/analysis/kernel_durations.py` from four traces:
- E19's (`../e19-batched-box/rocprofv3/trace`);
- this window's A and B;
- the crop sweep's N = 30 (`../e19-crop-sweep/rocprofv3/n030/trace`).

The medians in kernel_only:rpp (µs):
| trace | greyscale | box filter 3x3 | crop-mirror-normalize | per-call GPU span | sum of kernel durations per call |
|---|--:|--:|--:|--:|--:|
| E19 (2026-11-18) | 25.72 | 35.44 | 18.80 | 303.08 | 155.82 |
| sweep N = 30 (2026-11-24 07:42Z) | 15.92 | 35.08 | 13.60 | 261.18 | 130.16 |
| A (this window) | 15.92 | 34.96 | 13.52 | 260.70 | 130.02 |
| B (this window) | 15.92 | 34.88 | 13.52 | 247.10 | 127.24 |

The shorter greyscale and crop-mirror-normalize kernels the crop sweep's trace showed are in A's trace too, with E19's own binary. fkl_batch's kernel in kernel_only is 94.72 µs in E19's trace and 93.24 (A) / 93.44 (B) / 93.40 µs (sweep) here. These are traced runs. Under rocprofv3 the means are 6-8 % (fkl_batch) and 43-49 % (rpp) above the quiet medians (`summary/unused.csv`).

## The clock, the power and the temperature (`summary/clock.csv`, `summary/summary.csv`)
- **In each run's timing** (13 hwmon samples):
  - No sample was below 3.1 GHz. The minimum sclk was 3,108 MHz (`quiet-run3-A`). The median was 3,351 MHz in `unused-A` and 3,444-3,480 MHz in the other runs.
  - The median power was 44-49 W. The maximum temperatures were 45 °C edge, 49 °C junction and 42 °C memory.
- **After the 2 s warm-up:** 3,481-3,482 MHz in all six quiet runs, 3,284 MHz in `unused-A` (the first run), and 3,383 / 3,408 MHz under rocprofv3.
- **sclk at the interval boundaries**, min-max over the three quiet runs (MHz):
  - A: fkl_batch whole_call 3,462-3,463 before and 3,181-3,249 after; rpp whole_call 3,480-3,484 before and 3,401-3,412 after.
  - B: fkl_batch whole_call 3,462-3,463 before and 3,177-3,254 after; rpp whole_call 3,477-3,481 before and 3,368-3,428 after.
  - E19's window had minima of 2,994 MHz (fkl_batch) and 3,381 MHz (rpp) at its boundaries (`../e19-batched-box/e19_batched_box_summary.csv`).
  - These are facts of the two windows, not a cause.

## The unused runs (`summary/unused.csv`)
Against their program's quiet median:
- `unused-A` is +0.6 % to +3.0 %; its fkl_batch kernel_only is +2.99 %.
- `unused-B` is -2.6 % to +1.9 %.
