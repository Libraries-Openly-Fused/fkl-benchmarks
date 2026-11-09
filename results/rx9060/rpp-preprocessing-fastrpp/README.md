# Pipeline (a) on the RX 9060 with FastRPP (2026-11-09)

These runs of `benchmark_rpp_preprocessing_hip` at fkl-benchmarks 84d0bfa add the three FastRPP variants to the program. FastRPP
(`external/FastRPP` at efbf009, the merge of FastRPP #1) is RPP's rppt tensor API written as FKL operations:
- `rppt_resize_mirror_normalize` returns a read IOp;
- `rppt_channel_permute` returns a compute IOp, `fk::VectorReorderRT`;
- `rppt_copy` returns the write;
- `fastrpp::executeOperations` runs the three as one kernel, on the stream of an RPP handle.

The code is the fastrpp session's handoff, as given. The header of `rpp/benchmark_rpp_preprocessing.h` describes every variant.

| variant | what is timed | calls |
|---|---|---|
| `fastrpp_shared` | The frame read in place as a shared-source batch (nStride = 0, one ROI per crop). The build is the ROIs and the three FastRPP calls; the compute part is `executeOperations` and the synchronisation. | 1 |
| `fastrpp_crop_batch` | The same on the crops copied into a dense batch, one `hipMemcpy2DAsync` per crop, as `rpp_crop_batch`. The build is the copies and a synchronisation. | N + 1 |
| `fastrpp_shared_once` | `fastrpp_shared` built once, as the other built-once variants: the ROIs and the three FastRPP calls run before the timed calls. It is valid only while the crops (their positions and sizes) and the parameter arrays do not change. | 1 |

**The rotation changed.**
- The program now has 13 variants instead of 10.
- The first variant of a point is variant ((point index + `E5_ROTATE_OFFSET`) mod 13), so the order of every point differs from the runs of
  `../rpp-preprocessing-built-once/` (49c71db).
- The ten earlier variants still keep their relative order, with the FastRPP variants between them, at 30 of the 90 noise points: run0 up to
  110 crops, run1 up to 100 and run2 up to 90.
- The ten variants of those runs are repeated here and compared with them below.

## What was run

**Versions** (`step0/versions.txt`):
- `benchmark_rpp_preprocessing_hip` at fkl-benchmarks 84d0bfa, the first commit of this branch, on main 5caa9d1;
- FKL 1da77899, the pin since #42;
- FastRPP efbf009;
- RPP 3.2.0, ROCm 10.1.0, AMD clang 24.0.0git, gfx1200.

**Build** (`step0/run_build.sh`; `step0/build-ref.txt`):
- The sources are a `git archive` of 84d0bfa, with clean checkouts of FKL 1da77899 and FastRPP efbf009.
- The binary's sha256 is 1b968c347228114b0f6a10c52b46ad4a6ccafdb7a2aafdc3483deaa9855f57c3. The build had 0 warnings.
- The CSV headers read `repo: fkl-benchmarks unknown` because the sources were an archive, not a worktree. `step0/build-ref.txt` records the
  commit with the binary's sha256, and every run in `runs.log` names that sha256.

**Runs** (`runs.log`, made with `step0/runs.sh`, the protocol of `../rpp-preprocessing-built-once/`):
- three noise runs, `noise/run0`, `run1` and `run2`, with `E5_ROTATE_OFFSET` 0, 1 and 2;
- the natural-image run, `natural-image`: kodim23 tiled over the frame, offset 0;
- each after the idle wait (no GitHub runner job, load average below 0.8), with the program's defaults (`E5_ITERS` 30, `E5_WARMUP` 3,
  `E5_WARMUP_SECONDS` 2.0) and `ulimit -v` 40 GB;
- 12:54:16 to 13:49:11 UTC, about 13 min 17 s per run. All four exited 0 with "all required numerical checks passed".

No GitHub runner job ran on the machine during the runs: the runner's diagnostics have no job between 12:28 and 13:49 UTC. The runs' environment
set no `HIP_`, `HSA_`, `ROC_` or `AMD_` variable and none naming a kernarg setting.

**Step 0.** Two runs with one timed call per variant came first. Their times are not used, and both passed every required check. Their commands
are in `step0/commands.txt`.
- **Correctness run** (`step0/correctness/`).
  - Settings: `E5_ITERS=1`, `E5_WARMUP=0` and `E5_WARMUP_SECONDS=0`. The steady-state rule still warmed each variant up for 0.50 to 1.25 s.
  - Binary: the first build, made from the worktree's files before they were committed as 84d0bfa (sha256 417a6748…a3bdf). Its `rpp/`,
    `common/` and `cmake/` are those of 84d0bfa, and its device code is that of the timing binary (247 of 247 functions identical).
- **Dry check** (`step0/dry-check/`): the timing binary with `E5_ITERS=1` under the runs' `ulimit -v` 40 GB.

**RPP's handle and its host buffer.** `rppCreate` returned success for FastRPP's second handle at every N up to 300 under that limit. RPP 3.2.0's
HIP handle also allocates a host buffer of 398,131,200 B x N for its host back-end with a `malloc` whose result it does not check
(`rpp::HandleImpl::PreInitializeBuffer` in the installed `librpp.so.3.2.0`).
- At 300 crops that buffer is 119 GB, more than the limit and more than RAM and swap, so at the larger batches it is not allocated.
- This holds for the benchmark's own handle as well, in these runs and in the earlier ones.
- The HIP back-end does not use the buffer.

**Kernel arguments** (the code-object metadata of the binary, `.kernarg_segment_size`, including the hidden arguments):

| kernel | N = 10 | N = 150 | N = 300 | per crop |
|---|---|---|---|---|
| FastRPP's (`fk::VectorReorderRT` in its chain) | 1,048 B | 11,128 B | 21,928 B | 72 B |
| FKL's fused kernel, either sampling | 832 B | 7,552 B | 14,752 B | 48 B |

The 24 B per crop that FastRPP adds are each image's mean and 1 / std.

## FastRPP against FKL and RPP

Each ratio is taken per run; the table gives the median of the three runs (min .. max).

| batch | fastrpp_shared_once / fkl_fused_halfpixel_once | fastrpp_shared / fkl_fused_halfpixel | rpp_crop_batch_unsplit / fastrpp_shared_once | fastrpp_crop_batch / rpp_crop_batch |
|---|---|---|---|---|
| 10 | 0.993 (0.987 .. 1.004) | 1.001 (0.997 .. 1.004) | 3.017 (2.990 .. 3.026) | 0.745 (0.741 .. 0.750) |
| 50 | 1.006 (0.996 .. 1.011) | 0.979 (0.975 .. 0.984) | 5.453 (5.421 .. 5.460) | 0.809 (0.809 .. 0.810) |
| 100 | 1.043 (1.026 .. 1.043) | 0.995 (0.990 .. 0.998) | 7.001 (6.988 .. 7.240) | 0.829 (0.828 .. 0.830) |
| 150 | 1.054 (1.045 .. 1.054) | 0.996 (0.988 .. 1.001) | 7.950 (7.901 .. 8.208) | 0.805 (0.804 .. 0.829) |
| 200 | 1.062 (1.061 .. 1.063) | 1.012 (1.008 .. 1.015) | 8.916 (8.879 .. 8.926) | 0.768 (0.728 .. 0.792) |
| 250 | 1.075 (1.065 .. 1.079) | 1.024 (1.018 .. 1.025) | 9.532 (9.523 .. 9.593) | 0.772 (0.766 .. 0.775) |
| 300 | 1.069 (1.062 .. 1.074) | 1.019 (1.018 .. 1.020) | 9.863 (9.759 .. 9.866) | 0.757 (0.725 .. 0.758) |

| ratio | median range, 10 .. 150 | median range, 10 .. 300 | batches with all three runs above 1 | batches with all three runs below 1 |
|---|---|---|---|---|
| fastrpp_shared_once / fkl_fused_halfpixel_once | 0.993 .. 1.061 | 0.965 .. 1.075 | 22 of 30 | 0 of 30 |
| fastrpp_shared / fkl_fused_halfpixel | 0.946 .. 1.004 | 0.946 .. 1.024 | 10 of 30 | 10 of 30 |
| rpp_crop_batch_unsplit / fastrpp_shared_once | 3.017 .. 7.950 | 3.017 .. 10.007 | 30 of 30 | 0 of 30 |
| fastrpp_crop_batch / rpp_crop_batch | 0.745 .. 0.835 | 0.728 .. 0.835 | 0 of 30 | 30 of 30 |
| rpp_crop_batch_unsplit / fkl_fused_halfpixel_once (Figure 15's FKL line, for reference) | 3.002 .. 8.328 | 3.002 .. 10.486 | 30 of 30 | 0 of 30 |
| hand_roi_halfpixel_once / fkl_fused_halfpixel_once (Figure 15's other line) | 0.966 .. 1.003 | 0.943 .. 1.003 | 0 of 30 | 27 of 30 |

`comparison.csv` has every batch.

**FastRPP's kernel against FKL's, both built once.**
- FastRPP's kernel is within 2 % of FKL's at 10, 20 and 40 to 60 crops, but 6.1 % slower at 30, the largest gap up to 150 crops.
- From 170 crops on it is 6.2 to 7.5 % slower at every batch but 210, a split point (below). At 300 crops it is +6.9 %.
- The handoff estimated about +15 % at 300 crops, from a probe of two of its components.
- The kernel differs from FKL's in three ways:
  - its permutation has run-time channel indices (`fk::VectorReorderRT`);
  - its mean and standard deviation are per image, so its parameters are 72 B per crop instead of 48;
  - it normalises inside each image's read, before the permutation, as (v - mean) * (1 / std). FKL permutes first, then multiplies, subtracts
    and divides with one set of constants.
- At 160, 210, 220 and 290 crops the three runs of `fastrpp_shared_once` split by about 10 %:
  - 160 crops: run0 0.0935 ms against 0.0851 and 0.0854;
  - 210 crops: run1 0.1125 against 0.1020 and 0.1017;
  - 220 crops: run1 0.1056 against 0.1167 and 0.1165;
  - 290 crops: run2 0.1300 against 0.1428 and 0.1429.

  `fkl_fused_halfpixel_once` does not split at those points. The cause is unknown.

**The whole calls.**
- The medians of `fastrpp_shared` / `fkl_fused_halfpixel` are within 5.4 % of 1 at every batch (0.946 .. 1.024).
- Single runs go down to 0.902: at 70, 80, 110, 170, 220, 230, 260 and 290 crops one run of `fastrpp_shared` is about 9 % faster than the other
  two. `fkl_fused_halfpixel` does not split.
- FastRPP's host-side build (its argument checks and the IOps) is shorter than FKL's (the crop views and the operations): 4.2 us against 11.2 us
  at 300 crops. That makes up for most of the slower kernel.

**Against RPP.**
- Timed as a user calls it (`rpp_crop_batch_unsplit`), RPP takes 3.0 to 10.0 times as long as FastRPP built once, and 3.0 to 10.5 times as long
  as FKL built once.
- On RPP's own input, the dense crop batch, FastRPP's one kernel makes the call 16 to 27 % shorter than RPP's two calls. The copies are the same
  in both: 0.96 ms at 300 crops.

**The times** (medians over the three runs of the per-run medians, ms):

| batch | fkl_fused_halfpixel_once | fastrpp_shared_once | fkl_fused_halfpixel | fastrpp_shared | rpp_crop_batch | fastrpp_crop_batch | rpp_crop_batch_unsplit |
|---|---|---|---|---|---|---|---|
| 10 | 0.0350 | 0.0347 | 0.0350 | 0.0349 | 0.1268 | 0.0944 | 0.1044 |
| 50 | 0.0505 | 0.0507 | 0.0528 | 0.0518 | 0.2997 | 0.2425 | 0.2764 |
| 100 | 0.0673 | 0.0701 | 0.0719 | 0.0715 | 0.5144 | 0.4264 | 0.4908 |
| 150 | 0.0850 | 0.0892 | 0.0913 | 0.0910 | 0.7578 | 0.6097 | 0.7091 |
| 200 | 0.1026 | 0.1090 | 0.1104 | 0.1117 | 0.9944 | 0.7638 | 0.9715 |
| 250 | 0.1195 | 0.1285 | 0.1291 | 0.1322 | 1.2125 | 0.9379 | 1.2247 |
| 300 | 0.1377 | 0.1468 | 0.1494 | 0.1520 | 1.4774 | 1.1200 | 1.4459 |

The build part of the built-once variants, `fastrpp_shared_once` included, is at most 0.03 us.

## Agreement

All four runs. A required comparison has 30 batches x 2 regions per run, 240 rows; a comparison of the interior only has 120.

| comparison | max over the runs and batches | required rows passed | bitwise equal |
|---|---|---|---|
| fastrpp_shared vs scalar reference HALF_PIXEL | 1.91e-06 | 240 of 240 | |
| fastrpp_shared vs fkl_fused_halfpixel | 7.63e-06 | 240 of 240 | |
| fastrpp_shared vs rpp_crop_batch, interior | 0 | 120 of 120 (interior only) | |
| fastrpp_crop_batch vs fastrpp_shared | 0 | 240 of 240 | |
| fastrpp_shared_once vs scalar reference HALF_PIXEL | 1.91e-06 | 240 of 240 | |
| fastrpp_shared_once vs fastrpp_shared | 0 | 240 of 240 | 120 of 120 batch-runs |
| fkl_fused_halfpixel_once vs fkl_fused_halfpixel | 0 | 240 of 240 | 120 of 120 batch-runs |
| hand_roi_halfpixel_once vs scalar reference HALF_PIXEL | 7.63e-06 | 240 of 240 | |
| fkl_fused, fkl_fused_halfpixel vs their scalar references | 7.63e-06 | 240 of 240 each | |
| fkl_perop_batch vs fkl_fused | 7.63e-06 | 240 of 240 | |
| rpp_crop_batch vs scalar reference HALF_PIXEL, interior | 1.91e-06 | 120 of 120 (interior only) | |
| rpp_crop_batch_unsplit vs rpp_crop_batch | 0 | 240 of 240 | |
| hand_batch, hand_padded, hand_roi vs scalar reference ASYMMETRIC | 7.63e-06 | 240 of 240 each | |
| hand_batch, hand_padded, hand_roi, hand_roi_halfpixel_once vs FKL | 0 | informational | 120 of 120 each |

- **RPP's edges.** RPP's last two columns and rows differ from the HALF_PIXEL reference by design (up to 126), as in the earlier runs. FastRPP's
  are within 1.91e-06 of the reference, so `fastrpp_shared` differs from `rpp_crop_batch` there by RPP's own error (informational).
- **FastRPP's intermediate tensor** is not written: stdout says "not written (one kernel)" in 120 of 120 batch-runs. Its contents are still the
  NaN they start from after one call of each FastRPP variant: the agreement calls, made before the timing. The timed calls are not checked.
- **RPP's arrays.** In every run and batch, stdout says that RPP's other pinned arrays are unchanged after its calls, and that RPP rewrote exactly
  256 * ceil(N / 256) - N padding ROIs.

## The ten variants of rpp-preprocessing-built-once, old against new

**Method.**
- **old:** `../rpp-preprocessing-built-once/noise/run0..2` (49c71db, FKL 64ad9195).
- **new:** these runs.
- **Value:** the median of the three per-run medians; the spread is max - min.
- **A change exceeds the spread** when |new - old| > max(spread old, spread new).
- **Flags:** 3 % for a GPU-bound variant, 5 % for a host-bound one (build at least half of the run).
- **What else changed:** the rotation (13 variants instead of 10), FastRPP's allocations (its frame, buffers and second RPP handle) and the FKL
  pin. The device code of these ten variants is the same at both pins (`../pin-move-devcode-64ad9195-1da77899/`).

| variant | class | change over the 30 batches, min .. median .. max | batches beyond the spread | flagged | beyond the spread and flagged |
|---|---|---|---|---|---|
| fkl_fused | GPU | -0.6 .. +0.3 .. +5.2 % | 7 of 30 | 1 of 30 | 1 of 30 |
| fkl_fused_halfpixel | GPU | -0.4 .. +0.4 .. +5.3 % | 8 of 30 | 1 of 30 | 1 of 30 |
| fkl_perop_batch | GPU | -1.1 .. +0.2 .. +1.9 % | 5 of 30 | 0 of 30 | 0 of 30 |
| rpp_crop_batch | host 28, GPU 2 | -2.8 .. +0.4 .. +2.8 % | 15 of 30 | 0 of 30 | 0 of 30 |
| rpp_crop_batch_unsplit | GPU | -3.1 .. +0.2 .. +1.4 % | 8 of 30 | 1 of 30 | 0 of 30 |
| hand_batch | GPU | +0.3 .. +1.7 .. +4.9 % | 17 of 30 | 4 of 30 | 3 of 30 |
| hand_padded | GPU | -0.3 .. +1.3 .. +5.1 % | 14 of 30 | 1 of 30 | 1 of 30 |
| hand_roi | GPU | -0.9 .. +0.9 .. +5.6 % | 13 of 30 | 1 of 30 | 0 of 30 |
| fkl_fused_halfpixel_once | GPU | -0.1 .. +0.7 .. +5.1 % | 8 of 30 | 1 of 30 | 0 of 30 |
| hand_roi_halfpixel_once | GPU | -0.6 .. +1.0 .. +5.4 % | 11 of 30 | 1 of 30 | 1 of 30 |

The rows that are beyond the spread and flagged:

| variant | batch | old ms (spread) | new ms (spread) | change | class |
|---|---|---|---|---|---|
| fkl_fused | 30 | 0.0436 (0.0018) | 0.0458 (0.0009) | +5.2 % | GPU |
| fkl_fused_halfpixel | 30 | 0.0442 (0.0016) | 0.0466 (0.0005) | +5.3 % | GPU |
| hand_batch | 10 | 0.0322 (0.0004) | 0.0337 (0.0003) | +4.9 % | GPU |
| hand_batch | 20 | 0.0371 (0.0006) | 0.0383 (0.0001) | +3.2 % | GPU |
| hand_batch | 130 | 0.0700 (0.0016) | 0.0731 (0.0022) | +4.4 % | GPU |
| hand_padded | 10 | 0.0330 (0.0009) | 0.0347 (0.0000) | +5.1 % | GPU |
| hand_roi_halfpixel_once | 10 | 0.0331 (0.0005) | 0.0349 (0.0005) | +5.4 % | GPU |

**What the comparison shows.**
- **The medians** move by +0.2 to +1.7 %. The new runs are slightly slower in most variants, the hand-written kernels most of all.
- **The seven rows beyond the spread and flagged** are mostly at small batches, 10 to 30 crops, where a call takes 32 to 47 us; there the change
  is 1.2 to 2.4 us. The one at 130 crops (`hand_batch`) is +3.1 us on 70 us. Four more rows are flagged but within the spread:
  - `rpp_crop_batch_unsplit` at 150 crops: -3.1 %;
  - `hand_batch` at 140: +4.4 %;
  - `hand_roi` at 10: +5.6 %;
  - `fkl_fused_halfpixel_once` at 10: +5.1 %.
- **The cause is unknown.**
  - For four of the seven rows the new rotation is not the cause: `hand_batch` at 10 and 20 crops, and `hand_padded` and
    `hand_roi_halfpixel_once` at 10. In all three runs each ran at the same position of its point, after the same variants, as in 49c71db. The
    other changes of the method remain.
  - For the other three rows these runs cannot separate the changes.
- **The class** is the analysis script's rule (`host_bound` in `scripts/rpp_fastrpp_compare.py`): GPU when the build is less than half of the
  run. `rpp_crop_batch_unsplit` has no parts and counts as GPU.
  - At these sizes a call is mostly launch and synchronisation, so "GPU-bound" here does not mean that the kernel's execution dominates.

## The PCIe links, the clocks and the fast state

**Links.** Every reading of every row is the same:
- host link (`host_link_*`): 16 GT/s x16;
- card-internal link (`card_link_*`): 32 GT/s x16.

That is 3,510 readings in the noise runs and 1,170 in the natural-image run, at three moments per row.

**Memory clock** at the start of the timed runs (noise runs): 1,124 MHz in 1,065 of the 1,170 rows, 875 MHz in 71 and 772 MHz in 34.

**The graphics clock** at the start of the timed runs was 2,820 to 3,433 MHz in the noise runs and 2,855 to 3,434 MHz in the natural-image run.
The warm-up took 2.00 to 2.25 s, with `warmup_steady` = 1 in every row.

**The fast state.** These are the points in which `fkl_fused` is more than 4 % below the median of the crop-batch runs, defined as in the earlier
runs. There are 3 of 90:

```
run0, N = 10 .. 300: .....................FF.......
run1, N = 10 .. 300: ................F.............
run2, N = 10 .. 300: ..............................
```

- In those points `fkl_fused` is 6.9 to 7.3 % faster than the median of its three runs.
- `fastrpp_shared` is 8.4 and 8.9 % faster in two of them (run0 at 220 and 230 crops), and not in the third. It is as fast at six other points:
  run0 at 70, 80 and 110, run1 at 260, and run2 at 170 and 290.
- Over the three points, the median change of every other variant is within 1 %. Single points move by up to +3.3 % (`rpp_crop_batch_unsplit`,
  run0 at 230).
- The cause is unknown.

## The natural-image run

One run, so there is no spread.

| ratio | min .. median .. max over the 30 batches | 10 .. 150 |
|---|---|---|
| fastrpp_shared_once / fkl_fused_halfpixel_once | 0.947 .. 1.011 .. 1.078 | 0.947 .. 1.056 |
| fastrpp_shared / fkl_fused_halfpixel | 0.979 .. 1.004 .. 1.028 | 0.979 .. 1.004 |
| rpp_crop_batch_unsplit / fastrpp_shared_once | 2.973 .. 8.158 .. 10.660 | 2.973 .. 8.223 |
| fastrpp_crop_batch / rpp_crop_batch | 0.719 .. 0.784 .. 0.832 | 0.747 .. 0.832 |
| rpp_crop_batch_unsplit / fkl_fused_halfpixel_once | 2.992 .. 8.633 .. 10.512 | 2.992 .. 8.682 |
| hand_roi_halfpixel_once / fkl_fused_halfpixel_once | 0.914 .. 0.964 .. 1.007 | 0.914 .. 1.007 |

Against the natural-image run of `../rpp-preprocessing-built-once/`, one run each, the medians of the ten variants' changes are +0.2 to +1.3 %.
`scripts/rpp_fastrpp_compare.py` prints each variant's range.

## Files

- **`noise/run0..2/`, `natural-image/`:** the two CSVs of each run and its stdout.
- **`runs.log`:** the start and end of each run, its idle wait and its exit.
- **`step0/`:**
  - `run_build.sh` and `runs.sh`, the scripts that built and ran;
  - `commands.txt`, the commands of the two step-0 runs, and `build.log`, the timing build's log;
  - `build-ref.txt` and `versions.txt`;
  - `correctness/` and `dry-check/`, the two short runs.
- **`comparison.csv`:** written by `python3 scripts/rpp_fastrpp_compare.py`, which also prints the tables of this README.
