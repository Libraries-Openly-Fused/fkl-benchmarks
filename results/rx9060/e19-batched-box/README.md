# E19 batched box, the timing window (e19-batched-box), 2026-11-18, 16:30:11-16:30:35Z, RX 9060

The timing of `rpp/e19_batched_box.h`. The phase-1 checks are in `../e19-batched-box-phase1/`.

## What ran
- **The script:** `window.sh` at 5507edd (`../e19-batched-box-phase1/phase2/window.sh`; the copy is `window.sh`), with `WARMUP=1 IDLE_MAX=90`.
- **The program:** the binary of 2ad7ac5 (FKL 24478f0, RPP 3.2.0), sha256 `a193bae62d1e2d76d83e5d3a66f867c4d874c1e40b9c2c2ee76f40a94bb8f4d1`. Its build record is in `step0/`, a copy of the phase-1 record.
- **The window:** granted on receipt, it ran from 16:30:11 to 16:30:35Z.
- **The guards:**
  - Before the start: the binary's name, sha256 and commit, not `-dirty`, and a new folder.
  - Before every run: no `Runner.Worker` and a load average below 0.8. Every run started after a 0 s wait.
- **The steps:**
  1. `warmup/`, not used;
  2. `quiet-run1..3/`;
  3. `rocprofv3/`, in timing mode, its times not used;
  4. then the report (`e19_batched_box_summary.csv`, `_speedups.csv`, `_report.md`, `_clock.csv`) and `rocprofv3/calls.txt` (`rocprof_calls.py --spin`).
  The script stops at a failed step; none failed.
- **The outcome:** all five runs exited 0 and printed "E19 checks passed". Their check rows are identical to phase 1's.
- **The protocol** (the E18 RX 9060 one):
  - the timing file is opened first, then a 2 s `gpuSpin`;
  - each row is timed in the order `fkl_loop`, `fkl_batch`, `rpp`, with `whole_call` (100 calls after one warm-up), then `kernel_only` (100 iterations behind a 5 ms spin, nothing to prepare);
  - sclk is read before and after every interval.
- **The sampler:** `gpu_clocks.csv` reads sclk, mclk, `power1_average` and the edge, junction and memory temperatures from hwmon every 100 ms.

## The times (the median of the three quiet runs' means, min-max, ms)

| row | whole_call | kernel_only |
|---|--:|--:|
| `fkl_loop` (30 kernels per frame) | 0.2202 (0.2184-0.2210) | 0.2170 (0.2168-0.2171) |
| `fkl_batch` (1 kernel) | 0.1136 (0.1132-0.1141) | 0.1029 (0.1027-0.1034) |
| `rpp` (30 copies + 3 kernels) | 0.1887 (0.1887-0.1896) | 0.2010 (0.2006-0.2011) |

| ratio (the median of the per-run ratios) | whole_call | kernel_only |
|---|--:|--:|
| `rpp` / `fkl_loop` | 0.86 (0.85-0.86) | 0.93 (0.93-0.93) |
| `rpp` / `fkl_batch` | 1.66 (1.65-1.68) | 1.95 (1.95-1.95) |
| `fkl_loop` / `fkl_batch` | 1.94 (1.91-1.95) | 2.11 (2.10-2.11) |

- **The spread** of the three runs' means (whole_call / kernel_only): `fkl_loop` 1.2 % / 0.1 %, `fkl_batch` 0.8 % / 0.7 %, `rpp` 0.4 % / 0.2 %. The largest is 1.2 %.
- **The two intervals:** for both FKL rows, `kernel_only` is below `whole_call`. For `rpp` it is above (0.2010 against 0.1887). No cause is claimed.

## The clock, the power and the temperature
- **The warm-up:** sclk is 78 / 76 / 84 MHz before the 2 s spin and 3,388 / 3,453 / 3,444 MHz after it.
- **sclk at the interval boundaries** (`quiet-run*/e19_batched_box_GPU_AMD.csv`, MHz):
  - after a `kernel_only` interval: 3,423-3,482;
  - after a `whole_call` interval: `fkl_loop` 3,045-3,057, `fkl_batch` 2,994 / 3,229 / 3,048, `rpp` 3,381-3,412.
  - **The lowest is 2,994 MHz,** after `fkl_batch`'s `whole_call` in quiet-run1, so a boundary is below the 3.1 GHz criterion. The boundaries after FKL's `whole_call` intervals are 2.99-3.06 GHz in all three runs.
- **The samples during each run's timing** (`e19_batched_box_clock.csv`): none is below 3.1 GHz.

| run | samples in the timing | sclk min / median / max (MHz) | median power (W) | max edge / junction / memory temperature (°C) |
|---|--:|---|--:|---|
| quiet-run1 | 13 in 1.64 s | 3,351 / 3,447 / 3,481 | 46 | 46 / 54 / 46 |
| quiet-run2 | 13 in 1.64 s | 3,229 / 3,480 / 3,481 | 47 | 47 / 51 / 46 |
| quiet-run3 | 13 in 1.64 s | 3,255 / 3,478 / 3,481 | 47 | 47 / 53 / 46 |
| rocprofv3 | 13 in 1.69 s | 3,322 / 3,394 / 3,470 | 44 | 48 / 51 / 46 |

The power is the driver's `power1_average`, sampled every 100 ms, so each value is an average over the driver's own window. This window is too short to say anything about thermal throttling.

## The rocprofv3 run (`rocprofv3/`)
- **Kernels per call:** `whole_call` holds 101 calls and `kernel_only` 101 calls plus 100 spins.

  | row | `whole_call` | `kernel_only` |
  |---|---|---|
  | `fkl_loop` | 3,030 = 101 x 30 | 3,130 |
  | `fkl_batch` | 101 x 1 | 201 |
  | `rpp` | 3,333 = 101 x 33 (30 copy blits + 3 kernels) | 3,433 |

  No call allocates, frees or memsets anything.
- **The spin covered every `kernel_only` call:** in 100 of 100 iterations for every row, the host had enqueued the whole call before the spin ended.
  - `fkl_loop`: 30 kernels per iteration. Minimum margin 4.75 ms, at most 14.7 µs between kernels.
  - `fkl_batch`: 1 kernel. Minimum margin 5.01 ms.
  - `rpp`: 33 dispatches with its copies (`analysis/rpp_spin_with_copies.py`, `.txt`). Minimum margin 4.62 ms, at most 5.1 µs between RPP's kernels.
- **The `rpp` gap figure.** `rocprof_calls.py --spin` leaves out the kernels of `hipMemcpy*` calls, because in E18 those were a preparation enqueued before the spin. So in `calls.txt` it shows `rpp` with 3 kernels per iteration and a gap of up to 251 µs from the spin's end to the first of them. That gap is the 30 copy blits, which run first. Counted with the copies, the gap from the spin's end to the first kernel is 25.9 µs (median), 27.6 µs at most, as for the other rows (26-27 µs, cause not determined, as in E18).
- **The trace files:** `e19_kernel_trace.csv` and `e19_hip_api_trace.csv` are stored gzip-compressed (`.csv.gz`; `trace-csv-sha256.txt` gives the sha256 of the uncompressed files). The scripts read the uncompressed CSVs.
