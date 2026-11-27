# E19 crop sweep, the timing window (e19-crop-sweep), 2026-11-24, 07:31:16-07:45:14Z, RX 9060

> **Timed again (dated note, 2026-11-27).** These times were measured at fkl-benchmarks `b7002e9` (CSV repo `c8eb575`) with FKL `24478f0`. The sweep was timed again at FKL 304492a in `../e19-crop-sweep-retime-304492a/` (#71, 7ae214f), with the builds of the pins b10fc66 and 304492a alternating, as Oscar ordered on 2026-11-26.

This is the timing of `rpp/e19_crop_sweep.h`: E19 at N = 1, 10, 20, ..., 150 crops. The phase-1 checks are in `../e19-crop-sweep-phase1/`.
- **`fkl_loop` is not compared** (Oscar, 2026-11-20). It is timed first in every run only to keep E19's row order. Its numbers are in the raw CSVs and in no table or ratio here.
- **Decision:** Oscar's order of 2026-11-24, relayed by the coordinator session the same day. The coordinator session verified phase 1 and granted this window.

## What ran
- **The script:** `window.sh`, as committed at cfda463 in `../e19-crop-sweep-phase1/phase2/window.sh` (its copy here is identical).
  - It was run as one detached call. This session made no other tool call until it exited.
  - The window ran from 07:31:16 to 07:45:14Z. The foreground check at 07:31:08Z found no `Runner.Worker` and a load of 0.06.
- **The program:** the binary of c8eb575 (FKL 24478f0, RPP 3.2.0), sha256 `f9fcb2a43e96be2249734491e6fecf8e3baadc3484d0b8ccba895559a2192ee8`. `step0/` is a copy of the phase-1 build record.
- **The sweeps.** Each sweep is 16 complete runs in ascending N, and each run is the checks, then a 2 s `gpuSpin`, then the timing in E19's order and protocol.
  1. `warmup/nNNN`, 07:31:16-07:32:49Z, not used;
  2. `quiet-run1..3/nNNN`, 07:32:49-07:41:59Z;
  3. `rocprofv3/nNNN`, 07:41:59-07:45:14Z, in timing mode, its times not used.
- **The outcome:** all 80 runs exited 0 and printed "E19 checks passed". No CSV has `INCOMPLETE` or `-nan`.
- **The guards:**
  - Before the start: the binary's name, sha256 and commit, not `-dirty`, and a new folder.
  - Before every run: no `Runner.Worker` and a load average below 0.8.
  - 21 of the 80 runs started after an idle wait, because the back-to-back runs had raised the 1-minute load past 0.8 (`summary/waits.csv`). The waits were 15 s before 15 runs, 30 s before 5 and 45 s before 1. Every run had its own 2 s warm-up spin after any wait.
- **The sampler:** `gpu_clocks.csv` reads sclk, mclk, `power1_average` and the edge, junction and memory temperatures from hwmon every 100 ms.
- **The summary** (`summary/`) was made after the window, without the GPU, by `analysis/e19_crop_sweep_report.py`. `summary/report.md` has every table below.

## The times (the median of the three quiet runs' means, min-max, ms)
| N | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |
|--:|--:|--:|--:|--:|
| 1 | 0.0137 (0.0136-0.0137) | 0.0163 (0.0162-0.0163) | 0.0231 (0.0231-0.0285) | 0.0278 (0.0278-0.0278) |
| 10 | 0.0521 (0.0514-0.0532) | 0.0533 (0.0523-0.0546) | 0.0838 (0.0800-0.0841) | 0.0783 (0.0777-0.0844) |
| 20 | 0.0808 (0.0795-0.0813) | 0.0715 (0.0715-0.0716) | 0.1323 (0.1322-0.1330) | 0.1351 (0.1350-0.1351) |
| 30 | 0.1116 (0.1104-0.1123) | 0.1018 (0.1017-0.1019) | 0.1849 (0.1799-0.1850) | 0.1858 (0.1810-0.1863) |
| 40 | 0.1448 (0.1429-0.1469) | 0.1314 (0.1307-0.1317) | 0.2639 (0.2581-0.2643) | 0.2580 (0.2522-0.2586) |
| 50 | 0.1786 (0.1783-0.1789) | 0.1615 (0.1615-0.1617) | 0.3565 (0.3552-0.3579) | 0.3531 (0.3498-0.3531) |
| 60 | 0.2123 (0.2118-0.2150) | 0.1952 (0.1951-0.1956) | 0.4264 (0.4246-0.4272) | 0.4209 (0.4208-0.4210) |
| 70 | 0.2541 (0.2523-0.2543) | 0.2330 (0.2324-0.2334) | 0.4994 (0.4977-0.5000) | 0.4952 (0.4944-0.4963) |
| 80 | 0.2924 (0.2920-0.2940) | 0.2697 (0.2684-0.2702) | 0.5770 (0.5663-0.5770) | 0.5731 (0.5683-0.5735) |
| 90 | 0.3256 (0.3242-0.3261) | 0.3019 (0.3010-0.3019) | 0.6731 (0.6721-0.6732) | 0.6674 (0.6666-0.6675) |
| 100 | 0.3615 (0.3593-0.3615) | 0.3319 (0.3303-0.3322) | 0.7587 (0.7466-0.7593) | 0.7502 (0.7464-0.7509) |
| 110 | 0.3916 (0.3904-0.3929) | 0.3628 (0.3625-0.3629) | 0.8462 (0.8460-0.8465) | 0.8384 (0.8381-0.8394) |
| 120 | 0.4288 (0.4275-0.4291) | 0.3936 (0.3931-0.3937) | 0.9357 (0.9350-0.9364) | 0.9283 (0.9282-0.9284) |
| 130 | 0.4597 (0.4578-0.4598) | 0.4221 (0.4217-0.4226) | 1.0123 (1.0114-1.0125) | 1.0055 (1.0055-1.0059) |
| 140 | 0.4909 (0.4888-0.4931) | 0.4506 (0.4504-0.4508) | 1.0879 (1.0871-1.0886) | 1.0816 (1.0807-1.0820) |
| 150 | 0.5211 (0.5209-0.5242) | 0.4798 (0.4789-0.4803) | 1.1583 (1.1581-1.1588) | 1.1530 (1.1525-1.1532) |

- **The spread of the three runs' means is at most 2.9 % from N = 20 on.** The largest is rpp kernel_only at N = 30.
- **At N = 1 and N = 10 it is larger:**
  - rpp whole_call at N = 1: 23.5 %. Quiet-run1 is 0.0285 ms; the other two runs are 0.0231.
  - At N = 10: fkl_batch 3.5 % (whole_call) and 4.4 % (kernel_only); rpp 4.9 % and 8.6 %.
  - These are calls of 14 to 84 µs. No cause is claimed.

## rpp / fkl_batch (the median of the three per-run ratios, min-max)
| N | whole_call | kernel_only |
|--:|--:|--:|
| 1 | 1.70 (1.68-2.09) | 1.71 (1.71-1.72) |
| 10 | 1.61 (1.50-1.64) | 1.49 (1.43-1.58) |
| 20 | 1.64 (1.64-1.66) | 1.89 (1.89-1.89) |
| 30 | 1.65 (1.63-1.66) | 1.82 (1.78-1.83) |
| 40 | 1.80 (1.78-1.85) | 1.96 (1.93-1.96) |
| 50 | 2.00 (1.99-2.01) | 2.18 (2.17-2.19) |
| 60 | 2.00 (1.99-2.01) | 2.16 (2.15-2.16) |
| 70 | 1.97 (1.96-1.98) | 2.13 (2.12-2.13) |
| 80 | 1.96 (1.94-1.98) | 2.12 (2.12-2.13) |
| 90 | 2.07 (2.06-2.07) | 2.21 (2.21-2.22) |
| 100 | 2.10 (2.08-2.10) | 2.26 (2.26-2.26) |
| 110 | 2.16 (2.15-2.17) | 2.31 (2.31-2.32) |
| 120 | 2.18 (2.18-2.19) | 2.36 (2.36-2.36) |
| 130 | 2.20 (2.20-2.21) | 2.38 (2.38-2.38) |
| 140 | 2.21 (2.21-2.23) | 2.40 (2.40-2.40) |
| 150 | 2.22 (2.21-2.22) | 2.40 (2.40-2.41) |

## GPU- or host-bound (`summary/bound.csv`)
The criterion is the RTX 4090 report's: a row is GPU-bound when its kernel_only and whole_call medians differ by less than 3 %.
- **rpp is GPU-bound from N = 20 on.** Its kernel_only is -2.3 % to +2.1 % of whole_call. At N = 1 the difference is +20.2 % and at N = 10 it is -6.5 %.
- **fkl_batch's kernel_only is 7.3 % to 11.5 % below its whole_call from N = 20 on.** The criterion labels that host-bound; E19 had the same gap (0.1029 against 0.1136 ms). At N = 1 kernel_only is 19.1 % above whole_call, and at N = 10 it is 2.3 % above.
- **No cause is claimed** for either difference.

## Kernels per call and the spin (`rocprofv3/`, `analysis/kernels_and_spin.py`, `kernels-and-spin.tsv`)
- **Kernel counts:** at every N, every range holds the expected kernels:
  - fkl_batch: 101 in whole_call and 201 in kernel_only (101 calls and 100 spins), so 1 per call;
  - rpp: 101 x (N + 3) in whole_call, plus 100 spins in kernel_only.
- **No call allocates or sets memory.** `calls.txt` comes from `rocprof_calls.py`.
- **The spin covers every kernel_only call at every N:** 100 of 100 iterations, for fkl_batch and for rpp. rpp is counted with its N copies (`rpp-spin-with-copies.txt`, from `../e19-batched-box/analysis/rpp_spin_with_copies.py`).

| N | fkl_batch smallest / median margin (us) | rpp smallest / median margin (us) |
|--:|--:|--:|
| 1 | 5,015.7 / 5,096.5 | 4,994.2 / 5,075.8 |
| 30 | 5,008.9 / 5,034.7 | 4,677.0 / 4,869.3 |
| 60 | 5,007.7 / 5,031.2 | 4,214.6 / 4,722.5 |
| 110 | 5,005.6 / 5,028.8 | 2,859.3 / 4,462.4 |
| 120 | 5,003.7 / 5,026.5 | **2,626.0** / 4,404.7 |
| 130 | 5,005.1 / 5,027.6 | 2,953.3 / 4,336.6 |
| 150 | 5,004.8 / 5,026.6 | 4,128.1 / 4,235.7 |

The full table, every N, is `analysis/kernels-and-spin.tsv`. The smallest margin of the window is 2.63 ms, rpp at N = 120.

The kernel and HIP API traces are gzip-compressed (`.csv.gz`). `trace-csv-sha256.txt` gives the sha256 of the uncompressed CSVs; together they are 35 MB.

## The clock, the power and the temperature (`summary/clock.csv`, `boundaries.csv`)
- **The warm-up:**
  - sclk is 31 to 1,573 MHz before each run's 2 s spin. It is higher when the run follows the previous one without a wait.
  - It is 3,468 to 3,482 MHz after the spin, in every quiet run.
- **During the timing,** in 3 x 16 quiet runs:
  - The hwmon samples' median sclk is 3,418-3,477 MHz at every N.
  - 10 of the 692 samples are below 3.1 GHz, at most 3 per N. The lowest is 2,997 MHz, at N = 150.
  - The median power rises with N, from 38 W at N = 1 to 66-70 W at N = 150.
  - The largest temperatures are 52 °C edge, 64 °C junction and 54 °C memory.
- **At the interval boundaries:**
  - After fkl_batch's whole_call, sclk is 2.86-3.45 GHz over all N, and 2.86-3.07 GHz from N = 50 on, below the 3.1 GHz criterion. This is as in E18 and E19.
  - After rpp's whole_call it is 3.37-3.48 GHz.
  - After a kernel_only interval it is 3.42-3.48 GHz.

| N | samples | sclk min / medians / max (MHz) | below 3.1 GHz | median power (W) | max edge / junction / memory (°C) |
|--:|--:|---|--:|--:|---|
| 1 | 37 | 3436 / 3477, 3476, 3477 / 3482 | 0 | 38, 38, 38 | 50 / 53 / 46 |
| 30 | 39 | 3012 / 3471, 3464, 3473 / 3476 | 1 | 49, 50, 50 | 51 / 57 / 50 |
| 60 | 41 | 3045 / 3450, 3464, 3460 / 3471 | 1 | 55, 55, 56 | 52 / 59 / 50 |
| 90 | 45 | 3011 / 3452, 3456, 3429 / 3475 | 1 | 55, 62, 61 | 52 / 63 / 52 |
| 120 | 46 | 3010 / 3439, 3439, 3441 / 3469 | 1 | 54, 59, 63 | 52 / 63 / 52 |
| 150 | 51 | 2997 / 3429, 3425, 3430 / 3468 | 3 | 70, 67, 66 | 51 / 64 / 52 |

Every N is in `summary/report.md`. The power is the driver's `power1_average`, sampled every 100 ms.

## The N = 30 repeat against `../e19-batched-box/` (Oscar's format)
The spread is the larger of the two windows' spreads of the three runs' means.

| row, interval | old (ms) | new (ms) | change | spread old / new | exceeds the spread? | > 3 %? | GPU- or host-bound |
|---|--:|--:|--:|--:|---|---|---|
| fkl_batch whole_call | 0.1136 | 0.1116 | -1.8 % | 0.8 / 1.7 % | yes | no | host (kernel_only -8.7 %) |
| fkl_batch kernel_only | 0.1029 | 0.1018 | -1.0 % | 0.7 / 0.2 % | yes | no | host |
| rpp whole_call | 0.1887 | 0.1849 | -2.0 % | 0.4 / 2.7 % | no | no | GPU (kernel_only +0.4 %) |
| rpp kernel_only | 0.2010 | 0.1858 | **-7.6 %** | 0.2 / 2.9 % | **yes** | **FLAG** | GPU |

**The flag: rpp kernel_only at N = 30 is 7.6 % faster than in E19's window.** The cause is not determined. What the files show, with no GPU:
- **The minimum barely moved.** The per-run minimum call times are 0.1755-0.1759 ms in E19 and 0.1713-0.1758 ms here. The change is in the mean of the 100 calls.
- **In E19, rpp's kernel_only mean was above its whole_call mean** (0.2010 against 0.1887). Here the two are within 0.4 %.
- **The traced runs' kernels were shorter** (`analysis/kernel_durations.py`, `kernel-durations-n030.txt`). This compares E19's rocprofv3 run with this window's N = 30 rocprofv3 run, so these are traced runs, not the quiet ones.
  - Inside kernel_only:rpp, RPP's grey kernel takes a median 25.7 µs in E19 and 15.9 µs here. Its normalise kernel takes 18.8 and 13.6 µs.
  - In whole_call the grey kernel is 16.0 and 13.3 µs, the normalise kernel 13.7 and 12.0 µs.
  - The box kernel and the copies are within 0.4 µs.
  - Per kernel_only call, the GPU span from the first copy to the last kernel is a median 303 µs in E19 and 261 µs here.
  - So in the traced runs the difference is in GPU time, in the duration of two of RPP's kernels, mostly inside kernel_only.
- **The clocks were alike in both windows:**
  - after rpp's kernel_only, 3,480-3,481 MHz in E19 and 3,471-3,476 MHz here;
  - after rpp's whole_call, 3,381-3,412 MHz in E19 and 3,368-3,405 MHz here.
- **The two programs differ only in their host code** (N is a template parameter here). The RPP library and its kernels are the same files.

## The unused runs (`summary/unused.csv`)
- **warmup:** every value is within ±3.0 % of the quiet median, except one, which is flagged:
  - **FLAG:** fkl_batch whole_call at N = 20 is 3.00 % below the quiet median (0.078388 against 0.080814 ms).
  - 2 of its timing samples are below 3.1 GHz, one at N = 120 and one at N = 140.
- **rocprofv3:** this sweep runs under tracing, so its times are higher, and they are not used.
  - 49 of its 64 values are more than 3 % above the quiet median.
  - rpp is 29-81 % above; fkl_batch is +1.1 % to +49 %, most at N = 1-40.
  - 9 of its timing samples are below 3.1 GHz.
