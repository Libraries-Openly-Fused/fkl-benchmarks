# E18 box re-time at FKL b10fc66 (the merge of FKL #433), 2026-11-24, 22:41:52-22:42:56Z

E18 (`rpp/e18_box.h`) is the 3 x 3 box filter after a crop and a resize, RPP against FKL, on the RX 9060. Its record is `../e18-box/`
(2026-11-17, FKL 24478f0). FKL #433 changed the linear-filter kernel that E18's FKL rows use, and #60 moves this repository's pin to its
merge, b10fc66. This window times E18 again so that the E18 record matches what the repository builds now. The paper reports no E18 number.
Oscar ordered it, through the coordinator session on 2026-11-24.

## What ran

- **The binaries.** They are `e18_box_hip` from the two HIP builds of `../pin-move-devcode-24478f0-b10fc66/` (#61). No new build was made.
  - **A:** fkl-benchmarks a8f9daa with FKL 24478f0, sha256 `987b8a2ec372fec0bc483cb56c78ced73a7c0a4e62f7c9e83cddbd1c6b165bc7`.
  - **B:** fkl-benchmarks 2160c70 (#60) with FKL b10fc66, sha256 `3007b72e92eae1783ee6c4d5478066a31fa1413a5ccc3bee72cd88af4729e1f2`.
- **Against the record's binary** (`build-options.txt`). The record's binary is `194a1dab...`, built from e2fca06.
  - The three builds have the same options. Their CMake caches are equal. The four build commands of `e18_box_hip` are the same apart from
    the folders: Release, gfx1200, amdclang++ 24, ROCm 10.1.0.
  - The sources differ only where E18 is not touched: from e2fca06 to a8f9daa, `rpp/e18_box.h` and `common/` are unchanged.
  - **A has the record binary's code.** Its device code matches, 6 of 6 functions. Its host code matches in 127 of 128 functions; the one
    exception is `fklb::writeSystemInfoHeader`, because of its build strings.
  - **The build strings differ.** A and B were built from a `git archive` (#61's `pin_build.sh`), so they record fkl-benchmarks as
    `unknown`, and cvGPUSpeedup and FastNPP as `not checked out`, with no pinned commit. They do record FKL 24478f0 (A) and b10fc66 (B). The
    record's binary records e2fca06. The CSV headers of `window/` show these strings.
- **The script.** `window.sh` was run as `launch.sh`, the commit before this one on #62. The coordinator read both scripts and granted the
  window. `manifest.sha256` (both binaries, `window.sh`, `retime_report.py`) was checked at the start.
- **The protocol** is the record's own, unchanged:
  - Every run is complete, in its own process, with no environment variable. It runs the checks, then a 2 s `gpuSpin` clock warm-up, then
    `rpp`, `rpp_ltrb`, `fkl_box_halfpixel` and `fkl_half_once`, each `whole_call` then `kernel_only`, with sclk before and after every interval.
  - The order: `unused-A`, `unused-B`, then `quiet-run1-A`, `-B`, `quiet-run2-A`, `-B`, `quiet-run3-A`, `-B`, then `rocprofv3-A` and
    `rocprofv3-B` under the record's rocprofv3 command.
- **The guards.**
  - Before every run: no `Runner.Worker` and a load below 0.8. Every run started after a 0 s wait. The load rose from 0.04 to 0.79 during
    the window, from the window's own runs.
  - Throughout, `gpu_clocks.csv` holds sclk, mclk, power and the three temperatures every 100 ms.
- **The outcome.** All 10 runs exited 0 and printed "E18 checks passed". None of their CSVs is incomplete or holds a NaN.

## The check rows (`summary/checks-identity.txt`)

The data rows of every run's checks, agreement, attribution and ties files equal the record's (`../e18-box/quiet-run1/`), with one
exception. B's checks file has one different informational row: `fused kernel arguments (bytes)` gives `details 16` instead of `details 24`,
because #433's details lose `width` and `height`.

All six output digests, every agreement, attribution and ties row, and every checked criterion are the same in A, B and the record.

## The times (`summary/report.md`, `summary/summary.csv`)

These are the median of the three quiet runs' means, with their min-max, in ms. The spread is (max - min) / median of the three means.

| version | interval | old (`../e18-box/`) | A | B |
|---|---|--:|--:|--:|
| `rpp` | whole_call | 0.1677 (0.1656-0.1684), 1.7 % | 0.1645 (0.1627-0.1680), 3.2 % | 0.1672 (0.1651-0.1700), 2.9 % |
| `rpp` | kernel_only | 0.1584 (0.1560-0.1598), 2.4 % | 0.1541 (0.1530-0.1543), 0.9 % | 0.1542 (0.1540-0.1542), 0.1 % |
| `rpp_ltrb` | whole_call | 0.1592 (0.1566-0.1594), 1.8 % | 0.1537 (0.1537-0.1549), 0.8 % | 0.1537 (0.1534-0.1539), 0.3 % |
| `rpp_ltrb` | kernel_only | 0.1555 (0.1517-0.1563), 3.0 % | 0.1501 (0.1501-0.1503), 0.1 % | 0.1501 (0.1498-0.1501), 0.2 % |
| `fkl_box_halfpixel` | whole_call | 0.2300 (0.2289-0.2344), 2.4 % | 0.2292 (0.2284-0.2306), 1.0 % | 0.2281 (0.2276-0.2312), 1.6 % |
| `fkl_box_halfpixel` | kernel_only | 0.2083 (0.2083-0.2099), 0.7 % | 0.2064 (0.2062-0.2069), 0.4 % | 0.2045 (0.2043-0.2048), 0.3 % |
| `fkl_half_once` | whole_call | 0.2318 (0.2318-0.2326), 0.4 % | 0.2321 (0.2317-0.2362), 1.9 % | 0.2297 (0.2273-0.2308), 1.5 % |
| `fkl_half_once` | kernel_only | 0.2079 (0.2075-0.2090), 0.7 % | 0.2065 (0.2062-0.2069), 0.3 % | 0.2044 (0.2043-0.2048), 0.2 % |

## The repeats (Oscar's rule)

Rules for the tables below:
- **A change exceeds the spread** when it is larger than the larger of the two spreads.
- **GPU- or host-bound** follows the crop sweep's criterion (`../e19-repeat-check/`): a version is GPU-bound when its kernel_only and whole_call
  medians differ by less than 3 %.
  - It labels `rpp_ltrb` GPU-bound (kernel_only 2.4 % below whole_call, in A and B).
  - It labels `rpp` (6.3 % / 7.8 %), `fkl_box_halfpixel` (9.9 % / 10.4 %) and `fkl_half_once` (11.0 % / 11.0 %) host-bound.
  - This is the criterion's label, not a shown cause. The record found the boundary clocks lower after whole_call intervals than after
    kernel_only ones, and this window shows the same (below).
- **A flag** is a change above 3 % (GPU-bound) or 5 % (host-bound).

### B against A, in this window (the pin move)

| version, interval | A (ms) | B (ms) | change | spread A / B | exceeds? | flag | cause | bound |
|---|--:|--:|--:|--:|---|---|---|---|
| `rpp` whole_call | 0.1645 | 0.1672 | +1.7 % | 3.2 / 2.9 % | no | no | none needed | host |
| `rpp` kernel_only | 0.1541 | 0.1542 | +0.0 % | 0.9 / 0.1 % | no | no | none needed | host |
| `rpp_ltrb` whole_call | 0.1537 | 0.1537 | +0.0 % | 0.8 / 0.3 % | no | no | none needed | GPU |
| `rpp_ltrb` kernel_only | 0.1501 | 0.1501 | -0.0 % | 0.1 / 0.2 % | no | no | none needed | GPU |
| `fkl_box_halfpixel` whole_call | 0.2292 | 0.2281 | -0.5 % | 1.0 / 1.6 % | no | no | none needed | host |
| `fkl_box_halfpixel` kernel_only | 0.2064 | 0.2045 | -0.9 % | 0.4 / 0.3 % | **yes** | no | not determined; the candidate is the changed kernel (below) | host |
| `fkl_half_once` whole_call | 0.2321 | 0.2297 | -1.0 % | 1.9 / 1.5 % | no | no | none needed | host |
| `fkl_half_once` kernel_only | 0.2065 | 0.2044 | -1.0 % | 0.3 / 0.2 % | **yes** | no | not determined; the candidate is the changed kernel (below) | host |

- **No change is flagged.** The two that exceed their spreads are FKL's kernel_only rows, at -0.9 % and -1.0 %.
- **The candidate is the changed kernel.** An FKL kernel_only interval holds one kernel, `fk::linearFilterDPPKernel`, after the spin. Its
  device code is one of the 5 functions that #433 changes in this program (`../pin-move-devcode-24478f0-b10fc66/`). In the rocprofv3 runs
  (`summary/kernel-durations.txt`), its median GPU duration is lower in B in all four ranges:
  - kernel_only `fkl_box_halfpixel`: 202.72 to 200.80 µs (-1.9 µs, -0.9 %);
  - kernel_only `fkl_half_once`: 201.04 to 199.04 µs (-2.0 µs, -1.0 %);
  - whole_call: -3.1 µs and -2.4 µs.

  The quiet kernel_only medians differ by 1.9 µs and 2.1 µs.
- **Why the candidate is not a shown cause.**
  - Each traced run is a single run.
  - RPP's kernels have the same code in A and B, and between the same two traces they move by -2.2 % to +1.7 % (at most 0.9 µs).
  - The linear-filter kernel itself, with the same code, is 1.0 % and 0.9 % shorter in A's kernel_only trace than in the record's.
  - So the traces do not separate a 1 % change from run-to-run variation, and why the new code would be faster is not determined.
- **The rpp rows run the same device code in A and B.** Their kernels are in librpp, and the call lambdas of the RPP rows are identical in
  A's and B's host code (`host-code-A-B/`).
  - What differs in the host code is FKL's: `executeLinearFilter` and the kernel stubs; `fkl_box_halfpixel`'s call (`launch()::{lambda()#7}`)
    and its kernel-only helper; and `launch()` itself, which builds `fkl_half_once`'s details and holds the checks.
  - This record does not follow the timed calls through `launch()`, as `../pin-move-devcode-24478f0-b10fc66/e19-host-code.md` does for E19.
    So it claims nothing more about rpp's host path, and no rpp change exceeds its spread.

### A against the record (the same code, seven days apart)

| version, interval | old (ms) | A (ms) | change | spread old / A | exceeds? | flag | cause | bound |
|---|--:|--:|--:|--:|---|---|---|---|
| `rpp` whole_call | 0.1677 | 0.1645 | -2.0 % | 1.7 / 3.2 % | no | no | none needed | host |
| `rpp` kernel_only | 0.1584 | 0.1541 | -2.7 % | 2.4 / 0.9 % | **yes** | no | not determined; not the program | host |
| `rpp_ltrb` whole_call | 0.1592 | 0.1537 | -3.5 % | 1.8 / 0.8 % | **yes** | **FLAG** | not determined; not the program | GPU |
| `rpp_ltrb` kernel_only | 0.1555 | 0.1501 | -3.5 % | 3.0 / 0.1 % | **yes** | **FLAG** | not determined; not the program | GPU |
| `fkl_box_halfpixel` whole_call | 0.2300 | 0.2292 | -0.3 % | 2.4 / 1.0 % | no | no | none needed | host |
| `fkl_box_halfpixel` kernel_only | 0.2083 | 0.2064 | -0.9 % | 0.7 / 0.4 % | **yes** | no | not determined; not the program | host |
| `fkl_half_once` whole_call | 0.2318 | 0.2321 | +0.1 % | 0.4 / 1.9 % | no | no | none needed | host |
| `fkl_half_once` kernel_only | 0.2079 | 0.2065 | -0.7 % | 0.7 / 0.3 % | no | no | none needed | host |

- **The two flags are `rpp_ltrb`'s, at -3.5 % in both intervals.** A runs the record binary's code against the same librpp, so the change
  follows the window, not the program. Its cause is not determined.
- **The traced runs do not show it in the kernels.** `rpp_ltrb`'s kernel_only kernels are: box filter 98.60 µs (the record's trace) and 97.92
  (A); resize 43.76 and 45.16. Its per-call GPU span after the spin is 146.48 and 147.04 µs (`summary/kernel-durations.txt`).
- **Facts of the two windows, not a cause:**
  - After the 2 s warm-up, sclk was 3,437-3,450 MHz in this window's six quiet runs, against 3,328-3,371 MHz in the record's three.
  - The boundaries after kernel_only intervals were 3,462-3,480 MHz, against 3,366-3,465 MHz.
  - The boundaries after `rpp_ltrb`'s whole_call intervals were 3,166-3,178 MHz, against 3,134-3,172 MHz.

### B against the record

| version, interval | old (ms) | B (ms) | change | spread old / B | exceeds? | flag | cause | bound |
|---|--:|--:|--:|--:|---|---|---|---|
| `rpp` whole_call | 0.1677 | 0.1672 | -0.3 % | 1.7 / 2.9 % | no | no | none needed | host |
| `rpp` kernel_only | 0.1584 | 0.1542 | -2.6 % | 2.4 / 0.1 % | **yes** | no | as A against the record | host |
| `rpp_ltrb` whole_call | 0.1592 | 0.1537 | -3.5 % | 1.8 / 0.3 % | **yes** | **FLAG** | as A against the record (B equals A here: 0.0 %) | GPU |
| `rpp_ltrb` kernel_only | 0.1555 | 0.1501 | -3.5 % | 3.0 / 0.2 % | **yes** | **FLAG** | as A against the record (B equals A here: -0.0 %) | GPU |
| `fkl_box_halfpixel` whole_call | 0.2300 | 0.2281 | -0.8 % | 2.4 / 1.6 % | no | no | none needed | host |
| `fkl_box_halfpixel` kernel_only | 0.2083 | 0.2045 | -1.9 % | 0.7 / 0.3 % | **yes** | no | not determined; it is A against the record (-0.9 %) plus B against A (-0.9 %) | host |
| `fkl_half_once` whole_call | 0.2318 | 0.2297 | -0.9 % | 0.4 / 1.5 % | no | no | none needed | host |
| `fkl_half_once` kernel_only | 0.2079 | 0.2044 | -1.7 % | 0.7 / 0.2 % | **yes** | no | not determined; it is A against the record (-0.7 %) plus B against A (-1.0 %) | host |

## The unused runs, the clocks and the rocprofv3 runs

- **The unused runs** (`summary/unused.csv`), against their binary's quiet median: `unused-A` is -3.4 % to +5.5 % (+5.5 % for rpp
  kernel_only, in the window's first run), and `unused-B` is -1.2 % to +1.9 %.
- **The clocks** (`summary/clock.csv`, `summary/summary.csv`):
  - In the timing of the six quiet runs, sclk had a minimum of 3,020 MHz (`quiet-run2-B`) and a median of 3,460-3,467 MHz.
  - Power had a median of 48-52 W, and the junction temperature a maximum of 56 °C.
  - The boundaries after FKL's whole_call intervals were 2.79-3.05 GHz, as in the record (2.90-3.00 GHz).
- **The rocprofv3 runs** are apart: they are not in the times above (`summary/rocprofv3-times.csv`, `summary/kernel-durations.txt`, and
  `summary/rocprofv3-{A,B}-calls.txt` from the record's `rocprof_calls.py --spin`).
  - Their means are 2-10 % above the quiet medians.
  - Every call has the record's kernels: `rpp` 3, `rpp_ltrb` 2, FKL 1.
  - In A and in B, 100 of 100 `kernel_only` iterations are covered by the spin, with a margin of at least 4.99 ms.
  - The gap from the end of the spin to the first kernel is up to 26.2 µs, as in the record.

## Files

- `window.sh`, `launch.sh`, `retime_report.py`, `manifest.sha256`: the scripts, as run (the first commit of #62).
- `window/`: the window's folder.
  - The console; the environment before and after; `gpu_clocks.csv`.
  - The 10 runs, each with its CSVs, stdout, stderr and exit code; the rocprofv3 runs also hold their traces.
  - The copies of `window.sh` and `manifest.sha256` that the window made.
- `summary/`: `retime_report.py`'s output (`report.md` and its CSVs and text files).
- `build-options.txt`: the comparison with the record's binary.
- `host-code-A-B/`: the host code of `e18_box_hip`, A against B. It is `host_compare.py` of `../pin-move-devcode-24478f0-b10fc66/step0/`, in its
  two forms.
