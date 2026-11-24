# E18 box, window 2 (e18-box), 2026-11-17, 15:13:20-15:13:53Z: the 3 x 3 box filter after a crop and a resize, timed on the RX 9060

This window times `rpp/e18_box.h` (phase 1: `../e18-box-phase1/`). It supersedes the first window (`../e18-box-window1/`), whose first variant was timed during the clock ramp from idle.

**Note, 2026-11-24: re-timed at FKL b10fc66** (the merge of FKL #433, which changes the linear-filter kernel of the FKL rows), in `../e18-box-retime-b10fc66/`. That window timed this program's code at FKL 24478f0 (A) and at b10fc66 (B) alternately, with this window's protocol. No change is flagged, and the outputs are the same.
- **B against A:** `fkl_box_halfpixel` and `fkl_half_once` kernel_only are 0.9 % and 1.0 % lower, beyond their spreads; the other rows are within theirs.
- **A against this window:** `rpp_ltrb` is 3.5 % lower in both intervals (flagged; the same code, so not the program; cause not determined).
- The times below are FKL 24478f0's.

## What ran
- **The script:** `window.sh` at e2fca06 (`../e18-box-phase1/phase2/window.sh`; the copy is `window.sh`), with `WARMUP=1 IDLE_MAX=90`.
- **The program:** `e18_box_hip` at e2fca06, FKL 24478f0. The build record is `step0/`: a clean worktree, 0 errors and 0 warnings, sha256 `194a1dabc195e1dc381a3a78761acd026fdf9d5fe368fc666e62e4fc1b4cef2b`.
- **The step-0 check run** (`step0/check/`, `E18_NO_TIMING=1`) is identical to 664b06a's: every agreement, attribution, ties and checks row and all six output digests.
- **The window:** granted to start on receipt, it ran from 15:13:20 to 15:13:53Z. A grant for 12:00Z did not run: the background wait for that time ended with the session that held it.
- **The guards:**
  - Before the start: the binary's sha256 and commit (not `-dirty`), and a new folder.
  - Before every run: no `Runner.Worker` and a load average below 0.8. Every run started after a 0 s wait, at a load of 0.01-0.08.
- **The steps:**
  1. `warmup/`: one complete run, not used;
  2. `quiet-run1..3/`: three complete runs, back to back;
  3. `rocprofv3/`: one complete run in timing mode under rocprofv3, its times not used;
  4. the summary (`e18_box_summary.csv`, `e18_box_speedups.csv`, `e18_box_report.md`, `e18_box_clock.csv`);
  5. `rocprofv3/calls.txt`.
- **The outcome:** every run exited 0 and printed "E18 checks passed". Every run's agreement, attribution, ties and checks rows are identical to phase 1's.
- **Recorded throughout:** `gpu_clocks.csv` (sclk and mclk every 100 ms) and the environment before and after.
- **The order** in every run: `rpp`, `rpp_ltrb`, `fkl_box_halfpixel`, `fkl_half_once`, each `whole_call` then `kernel_only`.
- **New since window 1 (6fca6aa):**
  - a 2 s `gpuSpin` right before the timing (`E18_CLOCK_WARMUP_MS`), then a synchronisation;
  - sclk before and after every interval, outside the timed calls (`sclk_before_mhz`, `sclk_after_mhz`).
- **The timing file is opened before the warm-up.** Its header runs `amd-smi` twice, 0.07-0.08 s each. The diagnostic of `../e18-box-phase1/clock-warmup-test/` shows sclk falling from 3.2 GHz to about 1.5 GHz within 0.1 s of the GPU going idle, and to idle within 0.2 s. So opening the file after the warm-up would have undone it. Window 1 opened the file right before `rpp`'s timing, which probably explains part of its ramp.

## The times (`e18_box_report.md`)
The median of the three quiet runs' means, with their min-max, in ms:

| version | whole_call | kernel_only |
|---|--:|--:|
| `rpp` | 0.1677 (0.1656-0.1684) | 0.1584 (0.1560-0.1598) |
| `rpp_ltrb` | 0.1592 (0.1566-0.1594) | 0.1555 (0.1517-0.1563) |
| `fkl_box_halfpixel` | 0.2300 (0.2289-0.2344) | 0.2083 (0.2083-0.2099) |
| `fkl_half_once` | 0.2318 (0.2318-0.2326) | 0.2079 (0.2075-0.2090) |

| RPP / FKL (the median of the per-run ratios) | whole_call | kernel_only |
|---|--:|--:|
| `rpp` / `fkl_box_halfpixel` | 0.72 (0.72-0.73) | 0.76 (0.74-0.77) |
| `rpp` / `fkl_half_once` | 0.72 (0.71-0.73) | 0.76 (0.75-0.77) |
| `rpp_ltrb` / `fkl_box_halfpixel` | 0.69 (0.67-0.70) | 0.74 (0.73-0.75) |
| `rpp_ltrb` / `fkl_half_once` | 0.69 (0.68-0.69) | 0.75 (0.73-0.75) |

- **FKL against RPP:** FKL's one kernel is slower than RPP's two calls on the RX 9060.
- **The spread** of the three runs' means (whole_call / kernel_only): `rpp` 1.7 % / 2.4 %, `rpp_ltrb` 1.8 % / 3.0 %, `fkl_box_halfpixel` 2.4 % / 0.7 %, `fkl_half_once` 0.4 % / 0.7 %.
- **FKL's `kernel_only` is below its `whole_call`:** 0.2083 against 0.2300 ms. Its `kernel_only` intervals also end at a higher sclk than its `whole_call` intervals. No cause is claimed.

## The clock
The criterion was at least 3.1 GHz, meant to rule out the ramp from idle.
- **(a) The ramp is gone.** sclk is 66 / 1,316 / 80 MHz before the 2 s warm-up and 3,328 / 3,360 / 3,371 MHz after it, so `rpp`'s timing starts at 3.33-3.37 GHz.
- **(b) The criterion as written is not met.** At the boundaries after FKL's `whole_call` intervals, sclk is 2.90-3.00 GHz in all three runs, the lowest 2,904 MHz. In quiet-run2's `gpu_clocks.csv`, 2 of the 19 samples of its timing are below 3.1 GHz (min 2,969 MHz). The other runs' samples are all at least 3,191 MHz (`e18_box_clock.csv`).
- **The boundary clocks by interval type** (`analysis/boundary_clocks.csv`, every row of every run):
  - after `kernel_only` intervals: 3.37-3.47 GHz;
  - after `whole_call` intervals: `rpp` 3.16-3.19, `rpp_ltrb` 3.13-3.17, FKL 2.90-3.00 GHz.
- **A hypothesis, not verified:** the GPU settles at a load-dependent clock. It would run lower under the continuous kernels of a `whole_call` than under the 1-thread spin that fills `kernel_only`. Power and temperature were not recorded.

## The repeats (Oscar's rule): window 1 against window 2
The spread is the larger of the two windows' spreads of the three runs' means.

| version, interval | window 1 | window 2 | change | spread | exceeds the spread? | > 3 %? |
|---|--:|--:|--:|--:|---|---|
| `rpp_ltrb` whole_call | 0.1587 | 0.1592 | +0.3 % | 1.9 % | no | no |
| `rpp_ltrb` kernel_only | 0.1558 | 0.1555 | -0.2 % | 3.0 % | no | no |
| `fkl_box_halfpixel` whole_call | 0.2325 | 0.2300 | -1.1 % | 2.4 % | no | no |
| `fkl_box_halfpixel` kernel_only | 0.2242 | 0.2083 | -7.1 % | 0.7 % | **yes** | **FLAG** |
| `fkl_half_once` whole_call | 0.2321 | 0.2318 | -0.1 % | 0.4 % | no | no |
| `fkl_half_once` kernel_only | 0.2218 | 0.2079 | -6.3 % | 0.7 % | **yes** | **FLAG** |

- **`rpp` is left out of the flagging.** Its window-1 values were taken during the clock ramp and are biased (`../e18-box-window1/`): 0.2450 to 0.1677 ms for whole_call and 0.1931 to 0.1584 ms for kernel_only.
- **The cause of the two flags is not determined.** The candidate is sclk, tested below as far as the files allow, with no GPU.

### The candidate: sclk (`analysis/`)
- **(i) The boundary clocks of window 2:** `analysis/boundary_clocks.csv` gives `sclk_before_mhz` and `sclk_after_mhz` for every row, interval and run, the rocprofv3 run included. Window 1 has no such record.
- **(ii) The samples inside each interval:** `analysis/interval_clocks.py` writes `interval_clocks.csv`, the `gpu_clocks.csv` samples inside each interval of both windows.
  - The program records no per-interval host times, so the positions are estimated. They come from the order and the measured durations (whole_call about 101 x mean; kernel_only about 101 x (5 ms + mean)).
  - The estimates are fitted to the run's timing range. Window 2 has the range the program prints. Window 1 has none, so its range ends where the clock drops at the end of the run's timing.
  - The position error is up to about one sampling period (100 ms). A `whole_call` interval (16-25 ms) holds no sample, so only the `kernel_only` intervals (about 0.52 s, 4-5 samples) can be compared.
- **What the samples show** (mean sclk of the samples inside each `kernel_only` interval, over the three runs):

| `kernel_only` | window 1 | window 2 | time, window 1 to window 2 |
|---|--:|--:|--:|
| `rpp_ltrb` | 3,138-3,145 MHz | 3,382-3,423 MHz | 0.1558 to 0.1555 ms (-0.2 %) |
| `fkl_box_halfpixel` | 3,168-3,171 MHz | 3,404-3,438 MHz | 0.2242 to 0.2083 ms (-7.1 %) |
| `fkl_half_once` | 3,205-3,208 MHz | 3,358-3,453 MHz | 0.2218 to 0.2079 ms (-6.3 %) |

- **(iii)** The clock differed for both, by about 7-8 %: `rpp_ltrb`'s and FKL's `kernel_only` intervals all ran faster-clocked in window 2. `rpp_ltrb`'s time did not change, while FKL's fell by 6-7 %. So `rpp_ltrb`'s unchanged time would mean it depends on the clock less. Nothing further is claimed.
- **quiet-run2's low samples:** its `fkl_half_once` `kernel_only` interval holds one sample of 2,969 MHz, one of the two samples below 3.1 GHz in that run.

## The rocprofv3 run (`rocprofv3/`)
**Kernels per call**, from the ROCTx ranges:

| | `whole_call` kernels | per call | `kernel_only` kernels |
|---|--:|--:|--:|
| `rpp` | 303 | 3 (`roi_conversion_xywh_to_ltrb`, resize, box) | 403 = 3 + 100 x (spin + 3) |
| `rpp_ltrb` | 202 | 2 (resize, box) | 302 |
| `fkl_box_halfpixel` | 101 | 1 | 201 |
| `fkl_half_once` | 101 | 1 | 201 |

No range has a copy, an allocation, a free or a memset.

**The spin coverage** (`calls.txt`):
- For every variant, 100 of 100 `kernel_only` iterations are covered.
- The host had enqueued the whole call at least 4.997 ms before the spin ended (median margin 5.03-5.05 ms).
- Consecutive kernels of a call are at most 5.2 µs apart.

**The gap after the spin.** Under rocprofv3 there is a gap of up to 26.0-27.3 µs from the end of the spin to the start of the first kernel, as in window 1 (25-27 µs). It is the same for all four variants, and it lies inside every `kernel_only` interval. Its cause was not determined.

The rocprofv3 run's times are higher (`e18_box_GPU_AMD.csv` in `rocprofv3/`) and are not used.
