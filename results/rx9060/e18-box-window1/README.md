# E18 box, window 1 (e18-box-window1), 2026-11-17, 11:35:10-11:35:32Z: SUPERSEDED, kept as the record

**This folder is superseded by the second window (`../e18-box/`) and is kept only as the record.**
- `rpp` is the first variant of the fixed order. It was timed while the GPU's graphics clock was still ramping up from idle (see below).
- **Its times, and every `rpp` / FKL row, must not be used.**
- The second window adds a 2 s clock warm-up before the timing and records the clock at every interval boundary.

## What ran
- **The script:** `window.sh` at b7dd12b (`results/rx9060/e18-box-phase1/phase2/window.sh`; the copy is `window.sh`). It ran with `WARMUP=1 IDLE_MAX=90`.
- **The program:** the binary of `../e18-box-phase1/step0/`: `e18_box_hip` at 664b06a, FKL 24478f0, sha256 `8d4a2b9847ab7077225dfe8201c7025f62582315201c7d1185304064d0a6aeed`.
- **The window:** granted 11:35-11:50Z, ran 11:35:10-11:35:32Z.
- **The folder:** the window wrote to `results/rx9060/e18-box/`, and the folder was renamed to `e18-box-window1/` afterwards. The paths in `window-console.txt` name `e18-box`.
- **The guards:**
  - Before the start: the binary's sha256 and commit (not `-dirty`), and a new output folder.
  - Before every run: no `Runner.Worker` and a load average below 0.8. Every run started after a 0 s wait, with a load of 0.02 to 0.40.
- **The steps:**
  1. `warmup/`: one complete run, not used;
  2. `quiet-run1..3/`: three complete runs, back to back;
  3. `rocprofv3/`: one complete run in timing mode under rocprofv3, its times not used;
  4. the summary (`e18_box_summary.csv`, `e18_box_speedups.csv`, `e18_box_report.md`);
  5. `rocprofv3/calls.txt`, `calls.tsv`: the kernels of every range and the spin coverage.
- **The outcome:** every run exited 0 with an empty stderr (the rocprofv3 run's stderr holds the tool's log). Each run printed "E18 checks passed".
- **The checks:** every run's agreement, attribution, ties and checks rows are identical to those of `../e18-box-phase1/`.
- **Recorded throughout:** `gpu_clocks.csv` (sclk and mclk every 100 ms) and `environment-before.txt` / `environment-after.txt`.

## The clock ramp (`gpu_clocks.csv`)
- **During the checks:** in every quiet run the GPU idles at 35-60 MHz while the program computes its host reference, about 2 s.
- **When the timing starts:** sclk ramps up, for example 1,849 → 2,297 → 2,598 → 2,903 → 3,126 MHz in quiet-run1, about 0.5 s. It then stays at 3.1-3.24 GHz until the run ends.
- **What runs during the ramp:** `rpp` is timed first. Its `whole_call` (101 calls, about 25 ms) and most of its `kernel_only` (101 iterations behind a 5 ms spin, about 0.53 s) fall in the ramp.
- **How it shows in the times:** `rpp`'s `kernel_only` per-run minimum is 0.158 / 0.158 / 0.174 ms against means of 0.186-0.193 ms. `rpp_ltrb`, timed right after at full clock, has minima of 0.151-0.152 ms and means of 0.155-0.159 ms.
- **The other variants:** by `gpu_clocks.csv`, `rpp_ltrb`, `fkl_box_halfpixel` and `fkl_half_once` ran at 3.1-3.24 GHz. This window has no clock record per interval.

## The times, for the record (`e18_box_report.md`)
These are the median of the three quiet runs' means, with their min-max, in ms. **Do not use the `rpp` rows.**

| version | whole_call | kernel_only |
|---|--:|--:|
| `rpp` (**biased by the clock ramp**) | 0.2450 (0.2443-0.2475) | 0.1931 (0.1864-0.1934) |
| `rpp_ltrb` | 0.1587 (0.1570-0.1601) | 0.1558 (0.1551-0.1595) |
| `fkl_box_halfpixel` | 0.2325 (0.2322-0.2334) | 0.2242 (0.2239-0.2242) |
| `fkl_half_once` | 0.2321 (0.2321-0.2322) | 0.2218 (0.2214-0.2221) |

| RPP / FKL (the median of the per-run ratios) | whole_call | kernel_only |
|---|--:|--:|
| `rpp` / `fkl_box_halfpixel` (**biased, do not use**) | 1.05 (1.05-1.07) | 0.86 (0.83-0.86) |
| `rpp` / `fkl_half_once` (**biased, do not use**) | 1.06 (1.05-1.07) | 0.87 (0.84-0.87) |
| `rpp_ltrb` / `fkl_box_halfpixel` | 0.68 (0.67-0.69) | 0.70 (0.69-0.71) |
| `rpp_ltrb` / `fkl_half_once` | 0.68 (0.68-0.69) | 0.70 (0.70-0.72) |

- **The spread of the three runs' means:** `rpp` 1.3 % / 3.6 %, `rpp_ltrb` 1.9 % / 2.8 %, `fkl_box_halfpixel` 0.5 % / 0.1 %, `fkl_half_once` 0.07 % / 0.3 % (whole_call / kernel_only).
- **The host preparation of `kernel_only`:** 0.05-0.11 µs (mean) for every version.
- **FKL's `kernel_only` is below its `whole_call`:** 0.2242 against 0.2325, and 0.2218 against 0.2321 for the build-once form. `rpp_ltrb` goes the same way.

## The rocprofv3 run (`rocprofv3/`)
**Kernels per call**, from the ROCTx ranges:

| | `whole_call` kernels | per call | `kernel_only` kernels |
|---|--:|--:|--:|
| `rpp` | 303 | 3 (`roi_conversion_xywh_to_ltrb`, resize, box) | 403 = 3 + 100 x (spin + 3) |
| `rpp_ltrb` | 202 | 2 | 302 |
| `fkl_box_halfpixel` | 101 | 1 | 201 |
| `fkl_half_once` | 101 | 1 | 201 |

No range has a copy, an allocation, a free or a memset.

**The spin coverage** (`rocprof_calls.py --spin`):
- For every variant, 100 of 100 `kernel_only` iterations are covered.
- The host had enqueued the whole call at least 5.00 ms before the spin ended (median margin 5.03-5.05 ms).
- Consecutive kernels of a call are at most 5.3 µs apart.

**The gap after the spin.** Under rocprofv3 there is a 25-27 µs gap from the end of the spin to the start of the first kernel (median 25.4-27.0 µs, max 27.4-28.3 µs). It is the same for all four variants, and it lies inside every `kernel_only` interval. Its cause was not determined.

The clock during this run is not that of the quiet runs: the samples stay near idle for most of it.
