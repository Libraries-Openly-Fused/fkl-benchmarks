# E19 crop sweep timed again at the new pin, FKL 304492a, against FKL b10fc66 (RTX 4090, RT-1)

> **Dated note, 2026-11-27.** Side B of this record (FKL 304492a: `window/quiet-run{1,2,3}-B/` and `differences-B/`) backs the paper's RTX 4090 crop-sweep factors and chart, in place of `../e19-crop-sweep/` (Oscar's answers of 2026-11-27).

**What.** The RTX 4090 half of the E19 crop sweep, `results/rtx4090/e19-crop-sweep/` (old), timed again after the pin move of
fkl-benchmarks to FKL #435:
- #435 was merged as `304492af4d777165bd556fdd6113439635724d02`, the tree of its last head d1cff06;
- the pin move is #66, merged as af96aef.

The window runs two builds of the same program, `opencv-cuda/stencil/e19_crop_sweep.cu`, alternately (`binaries.sha256`):
- **A:** fkl-benchmarks 976e42b with FKL b10fc66, the pin before the move;
- **B:** fkl-benchmarks main with FKL 304492a, the pin after it.

Decision: the coordinator's task RT-1 of 2026-11-26, Oscar's order with Q3. Oscar's rule of the same day applies: "Coordinate with fkl-reduce,
fkl-reduce has priority". So the window starts only in a quiet slot agreed with that session, and any run it disturbs is redone.

**State.** The window ran on the coordinator's GRANTED (fkl-reduce QUIET, HOLD COMPILES and WINDOW STARTS sent) from 2026-11-27 03:21:36
to 03:43:54 (+01:00; 02:21:36Z to 02:43:54Z, `window/window-console.txt`). The analysis below was done afterwards, on the CPU only.

## Result

**The window** (`window/`):
- **Runs:** all 8 sweeps ran, 128 runs. Every run exited 0, passed its checks and wrote complete CSVs: each run's line reads
  "csv 3; INCOMPLETE in: 0; -nan in: 0; checks passed: 1".
- **Guards:** the after-run check never fired, so no run is affected and none is redone. The idle wait never timed out: it waited 0 to 26 s
  before a run, and the 1-minute load at the start of a run was 0.36 to 0.49.
- **Clocks** (`gpu_clocks.csv`, `gpu_clocks.err` empty):
  - 2,942 readings have a utilisation of 50 % or more. Their SM clock was 2,670 MHz in 2,832 of them, 2,550 MHz in 105 and 2,580 MHz in 5, at P2
    (5 at P0).
  - The temperature was 50 to 55 °C.
  - The old record's runs reached 2,685 MHz (its README, "The quiet runs are not all at 2,685 MHz").

**How to read the numbers:**
- `analysis/retime-table.md` gives all 96 numbers (16 N × 3 rows × 2 intervals): old, A and B as median (spread), the two changes, whether
  each exceeds the larger spread or the ranges do not overlap, the bound, and the flag. The same is in `analysis/retime-summary.csv`, and
  the text report with the ratios is `analysis/retime-report.txt`.
- **Bound:** a number is GPU-bound when whole_call / kernel_only of the newer side's medians is within 3 % of 1, and host-bound otherwise.
- **Flag:** a change above 3 % (GPU-bound) or above 5 % (host-bound).
- **Cause:** not shown for any change.

**B against A, the pin move** (2 flags, both fkl_batch at N = 1; 34 of 96 changes exceed the larger spread, 36 have ranges that do not overlap):
- **fkl_batch at N = 1 is about 18 % faster with B** in both intervals: whole_call 4.80 → 3.93 µs (−18.04 %), kernel_only 4.27 → 3.46 µs
  (−18.87 %). The spreads are at most 1.57 %, and the ranges do not overlap.
  - The kernel table above shows one device-code difference at this N and only at it. A's fkl_batch kernel for N = 1 has 35 registers, a
    40 B stack, 2 LDL, 5 STL and 568 instructions. B's has 40 registers, no stack, no LDL or STL, and 544 instructions.
  - At N = 10 to 150 the two builds' kernels have the same resources.
  - This difference is not shown to be the cause.
- **fkl_batch at N = 10 to 150 is slower with B by +0.02 % to +2.75 %.** No change reaches the flag, and 28 of these 30 exceed the larger spread.
  From N = 30 the numbers are GPU-bound.
- **opencv_cuda and npp, whose libraries are the same in A and B:** at N = 10 to 150, B against A is −0.28 % to +0.13 %. At N = 1 it is
  −0.33 % to +1.41 %, with opencv_cuda whole_call +1.41 % exceeding the larger spread.

fkl_batch, per N (the "exceeds" mark means the change exceeds the larger spread):

| N | whole_call: old / A / B, µs | B vs A | kernel_only: old / A / B, µs | B vs A | bound (B) |
|---|---|---|---|---|---|
| 1 | 4.71 / 4.80 / 3.93 | **−18.04 %** (flag) | 4.13 / 4.27 / 3.46 | **−18.87 %** (flag) | host-bound |
| 10 | 9.59 / 9.60 / 9.75 | +1.57 % (exceeds) | 9.22 / 9.22 / 9.22 | +0.02 % | host-bound |
| 20 | 15.81 / 15.87 / 16.07 | +1.28 % (exceeds) | 15.38 / 15.43 / 15.56 | +0.84 % (exceeds) | host-bound |
| 30 | 22.04 / 22.18 / 22.33 | +0.67 % (exceeds) | 21.70 / 21.69 / 22.28 | +2.75 % (exceeds) | GPU-bound |
| 40 | 28.22 / 28.30 / 28.66 | +1.27 % (exceeds) | 28.01 / 28.07 / 28.67 | +2.13 % (exceeds) | GPU-bound |
| 50 | 34.33 / 34.53 / 34.75 | +0.62 % (exceeds) | 34.22 / 34.69 / 34.82 | +0.35 % | GPU-bound |
| 60 | 40.61 / 40.63 / 41.04 | +1.02 % (exceeds) | 40.46 / 40.88 / 40.96 | +0.21 % (exceeds) | GPU-bound |
| 70 | 46.77 / 46.86 / 47.28 | +0.91 % (exceeds) | 46.79 / 46.88 / 47.11 | +0.51 % (exceeds) | GPU-bound |
| 80 | 52.58 / 52.95 / 53.65 | +1.32 % (exceeds) | 52.71 / 52.91 / 53.26 | +0.66 % (exceeds) | GPU-bound |
| 90 | 59.02 / 59.27 / 60.22 | +1.60 % (exceeds) | 58.75 / 59.22 / 59.43 | +0.34 % (exceeds) | GPU-bound |
| 100 | 65.35 / 65.55 / 66.27 | +1.10 % (exceeds) | 64.98 / 65.21 / 65.62 | +0.63 % (exceeds) | GPU-bound |
| 110 | 71.52 / 71.74 / 72.47 | +1.01 % (exceeds) | 71.00 / 71.12 / 71.80 | +0.96 % (exceeds) | GPU-bound |
| 120 | 77.67 / 77.92 / 78.67 | +0.96 % (exceeds) | 77.06 / 77.35 / 77.95 | +0.77 % (exceeds) | GPU-bound |
| 130 | 83.84 / 84.01 / 85.04 | +1.22 % (exceeds) | 83.27 / 83.48 / 84.14 | +0.80 % (exceeds) | GPU-bound |
| 140 | 89.92 / 90.29 / 91.16 | +0.96 % (exceeds) | 89.33 / 89.65 / 90.43 | +0.87 % (exceeds) | GPU-bound |
| 150 | 96.08 / 96.38 / 97.34 | +0.99 % (exceeds) | 95.51 / 95.79 / 96.81 | +1.07 % (exceeds) | GPU-bound |

**A against old, the same timed code** (#60's records, above), 1 flag:
- **The flag:** opencv_cuda kernel_only at N = 1, 8.65 → 9.18 µs (+6.07 %), host-bound. The old spread is 3.47 % and A's is 0.22 %.
- **At N = 10 to 150, A is slower than old** in nearly every number: fkl_batch −0.09 % to +1.40 %, opencv_cuda +0.76 % to +2.37 %,
  npp +0.14 % to +1.27 %. 79 of 96 changes exceed the larger spread.
- The clock reached 2,670 MHz in this window and 2,685 MHz in the old record's (above). That difference is not shown to be the cause.

**The ratios** (the median of the per-sweep ratios, `analysis/retime-report.txt`):
- **At N = 1, B's ratios are about 19 % to 27 % higher than old's**, as fkl_batch's time falls:
  - opencv_cuda / fkl_batch: whole_call 2.060 → 2.493 (+21.0 %), kernel_only 2.092 → 2.647 (+26.5 %);
  - npp / fkl_batch: whole_call 1.799 → 2.141 (+19.0 %), kernel_only 1.991 → 2.366 (+18.8 %).
- **At N = 10 to 150, every ratio of B is within −2.41 % to +1.89 % of old's.** At N = 150, for example:
  - opencv_cuda / fkl_batch: whole_call 13.059 → 13.050, kernel_only 13.075 → 13.129;
  - npp / fkl_batch: whole_call 11.460 → 11.433, kernel_only 11.525 → 11.503.

**The differences** (`differences-A/`, `differences-B/`: `e19_crop_sweep_differences.csv` and `consistency.txt`):
- **Each side's runs agree:** at every N, the 4 between rows are the same in that side's 4 sweeps.
- **A's file and B's are byte-identical.** Their rows are also those that `rt1_differences.py` gives from the old record's sweeps.
- **What this shows:** fkl_batch's tensor has the same count of values not equal to each baseline, and the same largest difference, at
  both pins and in the old record.
- **What it does not show:** that the tensors are bitwise equal across pins; the program writes no dump.
- **Where the file comes from:** the program's own in-memory comparison, as in the record.

## The two binaries (compile only, no GPU visible)

Both are `e19_crop_sweep_cu` of a `scripts/check_fkl_tree.sh` build: a link farm of the checkout, with the FKLB_* options, OpenCV and nvcc of
the reference build `build/local-rtx4090`.
- **The same flags as the record.** That reference build is the preset `local-rtx4090` with which the record's binary was built. The CUDA
  flags of the link-farm builds are those of the reference build (`-O3 -DNDEBUG -std=c++17`, sm_89 with PTX, `-ftemplate-depth=1000`,
  `-lineinfo`), with CMAKE_BUILD_TYPE Release, nvcc 13.4.92 and OpenCV 4.14.0.
- **A** = `scratch/pin-435/base-b10fc66/bench-build/bin/e19_crop_sweep_cu`, sha256 `9cb5a18f...`.
  - This is the base build of #68's identity check: 976e42b with FKL b10fc66, cvGPUSpeedup 6d26c65 and FastNPP a1b5e24. Its sha256 is
    that record's (`../pin-move-check-b10fc66-48fbf3f/sass-identity/binary-sha256-base-b10fc66.txt`).
  - Its fkl_batch kernel and the host code of its timed calls equal those of the record's binary (246b1c8, FKL 24478f0), per #60's records.
    #60 compared exactly that source and FKL with main's, and found the timed calls of `fkl_batch`, `opencv_cuda` and `npp` unchanged
    (`../pin-move-check-b10fc66-48fbf3f/`, section 3.6).
- **B** = `scratch/pin-435/main-304492a/bench-build/bin/e19_crop_sweep_cu`, sha256 `acf58fc7...` (`compile-B/`, 2026-11-26 19:02:09 to
  19:03:33, nice 19, `-j8`, `scripts-used/check_fkl_tree_rt.sh`).
  - It was built from main 73a9650. That is 658a896 plus #67, which changes only `results/rx9060/`, so the program's source and headers are
    658a896's. FKL 304492a, cvGPUSpeedup 6d26c65 and FastNPP a1b5e24.
  - The build script is `scripts/check_fkl_tree.sh` with two changes: the checkout comes from `CHECK_ROOT`, and only six targets are built.
    These are this program, and `e16_attention_cu`, `e16_query_tile_cu`, `e18_box_cu`, `e18_box_split_cu` and `e18_stencil_cu` for the
    later re-times RT-2 and RT-4.
  - Its folder name has the length of A's, so the two builds' source paths have the same length.
- **What the binaries carry.** The link farm has no `.git`, so both binaries name the repository commit "unknown". They carry their FKL
  commit, and no `-dirty` commit.

## The kernels of the program, A against B (`resources/`)

From `cuobjdump -res-usage` and `-sass` (13.4.3, sm_89), paired with `scripts-used/pr435_sass_identity.py` and FKL #435's rename rule
(`resources/sass-e19-A-vs-B.txt`, `fkl-batch-resources.txt`, `res-usage-{A,B}.txt`). There are 18 kernels in each build:

| kernel | N | registers A / B | stack A / B | LDL, STL A / B | constant bank 0 A / B | instructions A / B | SASS |
|---|---|---|---|---|---|---|---|
| fkl_batch (MorphologyDPP) | 1 | 35 / 40 | 40 / 0 B | 2, 5 / 0, 0 | 472 / 472 B | 568 / 544 | different |
| fkl_batch (MorphologyDPP) | 10 to 150 (15 kernels) | 40 / 40 | 0 / 0 B | 0, 0 / 0, 0 | 688 to 4,048 B (both) | 544 / 544 | different |
| fkl_loop (LinearFilterDPP) | (one kernel) | 39 / 39 | 0 / 0 B | 0, 0 / 0, 0 | 456 / 456 B | 696 / 704 | different |
| gpuSpin | | 8 / 8 | 0 / 0 B | 0, 0 / 0, 0 | 360 / 360 B | 24 / 24 | identical |

- At every N the shared memory of fkl_batch's kernel is 1,232 B in both builds, and its constant bank 0 grows by 240 B per 10 crops.
- FKL #435 passes the details of the filter DPPs as a `__grid_constant__` kernel parameter (`../fkl-pr435-check/`); the SASS of these kernels
  differs with it. No cause of a time difference is claimed from this.

## The window (`window-script.sh`)

(Run on the coordinator's GRANTED, 2026-11-27 03:21:36 to 03:43:54 +01:00; see the result above.)

**The protocol is the record's.** Each sweep is 16 processes, one per N (`E19_CROPS` = 1, 10, ..., 150), in ascending order. Each process is
a complete run of the program:
- first the checks;
- then whole_call and kernel_only of every row, in the order fkl_loop, fkl_batch, opencv_cuda, npp.

fkl_loop runs first only to keep E19's order. It is in no table.

**A and B alternate, as in #63:**
- `warmup-A`, then `warmup-B` (not used);
- then `quiet-run1-A`, `quiet-run1-B`, `quiet-run2-A`, `quiet-run2-B`, `quiet-run3-A`, `quiet-run3-B`.

That is 128 runs. There is no nsys sweep.

**Before every run**, the script waits for an idle machine: no build, compile job or measurement, a 1-minute load below 0.5, and no other
program on the RTX 4090 (up to 300 s).

**After every run**, it checks again for a build, a compile job or another program on the RTX 4090. If one appeared during the run, the window
stops, and that run is the affected one, to be redone.

**The runs:** each is pinned to cores 0-3,16-19, under a 300 s timeout, and must exit 0, pass its checks and write complete CSVs. nvidia-smi
reads the GPU every 100 ms.

**The report** (`scripts-used/rt1_report.py`) gives, per N, row and interval:
- the medians and spreads of old, A and B;
- B against A, and A against old: the change, whether it exceeds the larger spread or the ranges do not overlap, the bound, and the flag
  (above 3 % for a GPU-bound number, 5 % for a host-bound one).

No cause is named unless shown.

**The differences** (`e19_crop_sweep_differences.csv` for A and for B, written after the window on the CPU by
`scripts-used/rt1_differences.py`):
- **What they are:** the largest difference of fkl_batch's tensor from each baseline, opencv_cuda and npp, per N. The header is
  `gpu,crops,baseline,values,values_not_equal,max_abs_diff_output_units,max_abs_diff_grey_levels`.
- **Where they come from:** the program's own in-memory comparison, as in the record. Every run's checks compare fkl_batch's tensor
  with those of opencv_cuda and npp. These are the "between" rows of `e19_batched_box_checks.csv`: the values, those not equal, and the largest
  absolute difference in output units and in grey levels (× 57.375). No dump is written, and the window script has no extra step.
- **Repeatability:** the script checks that a side's runs agree at every N (its warm-up and three quiet sweeps).
- **Tested on the record:** from the record's five sweeps, it gives the 32 rows of the record's `../e19-crop-sweep-checks/analysis/between.csv`
  exactly. At every N, the rows are the same in all five sweeps.

## Files

| path | content |
|---|---|
| `binaries.sha256` | A and B, the two binaries the window runs |
| `window-script.sh` | the window's script |
| `compile-B/` | B's build: the report, target lists, configure log, build time, console, and the sha256 of its six binaries |
| `resources/` | the kernels of A and B: `cuobjdump -res-usage`, the SASS comparison and its CSV, and the per-N table |
| `scripts-used/` | `check_fkl_tree_rt.sh`, `pr435_sass_identity.py`, `fkl_batch_resources.py`, `rt1_report.py` and `rt1_differences.py` |
| `window/` | the window: `window-console.txt`, the environment before and after, `gpu-info.csv`, `gpu_clocks.csv` (and `.err`), `load.log`, the copy of the script the window ran, and one folder per sweep with each run's two CSVs, stdout, stderr, exit code, load and the GPU state at its end |
| `analysis/` | `retime-report.txt` (with the ratios), `retime-summary.csv` and `retime-table.md`: `rt1_report.py`'s output, old against the window |
| `differences-A/`, `differences-B/` | `e19_crop_sweep_differences.csv` and `consistency.txt` for each side: `rt1_differences.py`'s output |
