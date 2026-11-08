# Pipeline (a) on the RX 9060, built once: the RX 9060 dataset of Figure 15 (2026-11-08)

These runs of `benchmark_rpp_preprocessing_hip` at fkl-benchmarks bdc7d19 (#38) give the RX 9060 series of Figure 15. The figure leaves out the
host-side build of our implementations, because the paper does not describe the builder API. So the two lines of the figure use the
**built-once** variants: their operations and parameter blocks are built once, before the timed calls, and each timed call holds only the launch
and the synchronisation.

| line | variant | against |
|---|---|---|
| FKL against RPP | `fkl_fused_halfpixel_once` | `rpp_crop_batch_unsplit`, timed as a user calls it: the N crop copies and RPP's two calls |
| FKL against a hand-written kernel | `hand_roi_halfpixel_once` | `fkl_fused_halfpixel_once` |

`hand_roi_halfpixel_once` is the hand-written kernel of E4 for crops of any location and size. It uses FKL's HALF_PIXEL sampling, and its CPU
function `handRoiParameters` runs once.

**Validity.** The built-once variants are valid only while the crops (their positions and sizes) do not change. What is built once holds every
crop's location and size, and for `hand_roi_halfpixel_once` also its pitch and resize factors. Crops that move or change size need a new build,
which these variants leave out on purpose.

The whole-call variants are timed in the same runs, with their build included. They are compared below with the whole-call record
`results/rx9060/rpp-preprocessing-hand-roi/`.

## What was run

- **Program:** `benchmark_rpp_preprocessing_hip` at fkl-benchmarks bdc7d19, FKL 64ad9195, RPP 3.2.0, ROCm 10.1.0, gfx1200 (`step0/versions.txt`).
- **Build:** from a clean worktree with `step0/rpp_run_build.sh`. sha256 ed953c9da3d39e7a7fbb33c3bf752feb632c76ea3e86b45559aaf69174dadf1e.
- **Runs (`runs.log`):**
  - three noise runs, `noise/run0`, `run1` and `run2`, with `E5_ROTATE_OFFSET` 0, 1 and 2;
  - the natural-image run, `natural-image` (kodim23 tiled over the frame, offset 0);
  - each after the idle wait of `step0/rpp_runs_once.sh` (no GitHub runner job, load average below 0.8), with the program's defaults (`E5_ITERS`
    30, `E5_WARMUP` 3, `E5_WARMUP_SECONDS` 2.0);
  - 13:29:28 to 14:11:49 UTC, all four exited 0 with "all required numerical checks passed".
- **Variants of a point**, ten in a rotated order (the first is variant (point index + offset) mod 10):
  - the eight of #36: `fkl_fused`, `fkl_fused_halfpixel`, `fkl_perop_batch`, `rpp_crop_batch`, `rpp_crop_batch_unsplit`, `hand_batch`,
    `hand_padded`, `hand_roi`;
  - the two built-once variants.

  The header of `rpp/benchmark_rpp_preprocessing.h` describes them.
- **Ratios in the figure:** both lines are ratios taken per run, then the median of the three runs. The rotation puts the two variants of a
  ratio at different positions of a point, and the warm-up rule of each variant is meant to make the position irrelevant.

**The machine's PCIe topology.** The GPU's sysfs path is `/sys/devices/pci0000:00/0000:00:03.1/0000:0b:00.0/0000:0c:00.0/0000:0d:00.0`.

| link | from | to | speed |
|---|---|---|---|
| host link (`host_link_*`) | the CPU root port 0000:00:03.1 | the card switch's upstream port 0000:0b:00.0 | x16, PCIe 4.0, 16 GT/s (the B550 limit) |
| card-internal link (`card_link_*`) | the card switch's downstream port 0000:0c:00.0 | the GPU 0000:0d:00.0 | 32 GT/s x16 |

The card hangs directly off the CPU root port, not behind the chipset.

## Step 0 (`step0/`)

- **Same configuration as the build of the whole-call record.** `step0-built-once.txt` (this build) and `step0-151d517.txt` (that build) differ
  only in the name and the worktree path once commits and paths are normalised. That covers the build information, the commit strings compiled in
  and the compile and link commands. `commands.txt` is the same as that build's.
- **CMake cache:** 177 entries, no differences (`cmake-cache-compare.txt`).
- **Device code:** `devcode-151d517-vs-bdc7d19.txt` and `functions/`. The 187 functions of the binary of the whole-call record are identical, and
  the 30 functions that exist only in this binary are `e1c::handPreprocRoiHalfPixel<N>`. `fkl_fused_halfpixel_once` runs the kernel of
  `fkl_fused_halfpixel`.
- **Dry check:** one run with `E5_ITERS=1`; its times are not used.
  - `dry-check/csv-header.txt`: repository bdc7d19 and FKL 64ad9195, no `-dirty`.
  - All required checks passed.
- **`aspm-kernarg-runner.txt`:**
  - the ASPM policy is `default`;
  - L1 ASPM is off on the host link (the entries of its downstream end, 0000:0b:00.0: `l1_aspm=0`, `l1_1_aspm=0`, `l1_1_pcipm=0`) and on the
    card-internal link (0000:0d:00.0: `clkpm=0`, `l1_aspm=0`);
  - the run's environment sets no HIP, HSA, ROC or AMD variable and none naming a kernarg setting;
  - no GitHub runner job was active.

## The series of Figure 15

Per run, then the median of the three runs (min .. max):

| batch | rpp_crop_batch_unsplit / fkl_fused_halfpixel_once | hand_roi_halfpixel_once / fkl_fused_halfpixel_once |
|---|---|---|
| 10 | 3.077 (3.039 .. 3.131) | 0.994 (0.951 .. 1.007) |
| 20 | 3.769 (3.768 .. 3.799) | 0.986 (0.978 .. 0.991) |
| 30 | 4.496 (4.337 .. 4.508) | 0.984 (0.981 .. 0.984) |
| 40 | 5.036 (5.020 .. 5.045) | 0.997 (0.973 .. 1.003) |
| 50 | 5.535 (5.518 .. 5.543) | 0.998 (0.989 .. 1.003) |
| 60 | 5.928 (5.914 .. 5.955) | 0.978 (0.977 .. 0.986) |
| 70 | 6.332 (6.288 .. 6.345) | 0.974 (0.971 .. 0.976) |
| 80 | 6.717 (6.666 .. 6.752) | 0.988 (0.986 .. 0.991) |
| 90 | 6.964 (6.737 .. 6.967) | 0.979 (0.940 .. 0.988) |
| 100 | 7.267 (7.244 .. 7.300) | 0.986 (0.976 .. 0.990) |
| 110 | 7.579 (7.554 .. 8.077) | 0.991 (0.979 .. 1.041) |
| 120 | 7.776 (7.728 .. 7.842) | 0.980 (0.972 .. 0.989) |
| 130 | 7.947 (7.903 .. 7.953) | 0.975 (0.968 .. 0.975) |
| 140 | 8.075 (8.038 .. 8.194) | 0.965 (0.954 .. 0.973) |
| 150 | 8.656 (8.641 .. 8.818) | 0.972 (0.968 .. 0.973) |
| 160 | 8.950 (8.875 .. 9.053) | 0.959 (0.957 .. 0.961) |
| 170 | 9.108 (9.027 .. 9.254) | 0.965 (0.965 .. 0.967) |
| 180 | 9.195 (9.195 .. 9.327) | 0.966 (0.963 .. 0.968) |
| 190 | 9.445 (9.370 .. 9.476) | 0.955 (0.949 .. 0.957) |
| 200 | 9.598 (9.589 .. 10.036) | 0.954 (0.950 .. 1.036) |
| 210 | 9.691 (9.684 .. 9.718) | 0.962 (0.962 .. 0.963) |
| 220 | 9.498 (9.475 .. 9.834) | 0.954 (0.952 .. 0.956) |
| 230 | 10.019 (9.676 .. 10.041) | 0.960 (0.955 .. 0.974) |
| 240 | 10.077 (9.768 .. 10.117) | 0.940 (0.940 .. 0.943) |
| 250 | 10.266 (10.228 .. 10.279) | 0.962 (0.959 .. 0.962) |
| 260 | 10.042 (9.952 .. 10.051) | 0.952 (0.947 .. 0.953) |
| 270 | 10.117 (10.112 .. 10.187) | 0.950 (0.945 .. 0.960) |
| 280 | 10.267 (10.236 .. 10.302) | 0.961 (0.956 .. 0.962) |
| 290 | 10.435 (10.352 .. 10.505) | 0.958 (0.952 .. 0.961) |
| 300 | 10.562 (10.554 .. 10.571) | 0.949 (0.947 .. 0.962) |

Range of the median ratio:

| series | 10 .. 150 | 10 .. 300 |
|---|---|---|
| rpp_crop_batch_unsplit / fkl_fused_halfpixel_once | 3.077 .. 8.656 | 3.077 .. 10.562 |
| hand_roi_halfpixel_once / fkl_fused_halfpixel_once | 0.965 .. 0.998 | 0.940 .. 0.998 |

**The hand-written kernel against FKL.** Built once and in the same sampling, the hand-written kernel is at most 3.5 percent faster than FKL up
to 150 crops, and at most 6.0 percent up to 300. At 25 of the 30 batch sizes all three runs are below 1. At 10, 40, 50, 110 and 200 crops the
three runs include 1.

The two kernels give the same bits (see the agreement). They differ in their parameter blocks: FKL carries 48 bytes per crop,
`hand_roi_halfpixel_once` 32 (14,752 and 9,880 bytes of kernel arguments at N = 300, in the code-object metadata). The kernel-only reasons for the remaining difference are not part of this run.

The times (medians over the three runs, ms):

| batch | fkl_fused_halfpixel_once | hand_roi_halfpixel_once | rpp_crop_batch_unsplit | fkl_fused_halfpixel (whole call) | whole call / built once |
|---|---|---|---|---|---|
| 10 | 0.0333 | 0.0331 | 0.1030 | 0.0346 | 1.037 (1.000 .. 1.053) |
| 50 | 0.0497 | 0.0497 | 0.2753 | 0.0524 | 1.054 (1.051 .. 1.055) |
| 100 | 0.0673 | 0.0660 | 0.4888 | 0.0720 | 1.070 (1.066 .. 1.076) |
| 150 | 0.0841 | 0.0814 | 0.7318 | 0.0908 | 1.090 (1.074 .. 1.091) |
| 200 | 0.1013 | 0.0966 | 0.9717 | 0.1102 | 1.089 (1.079 .. 1.178) |
| 250 | 0.1188 | 0.1144 | 1.2201 | 0.1295 | 1.086 (1.083 .. 1.091) |
| 300 | 0.1371 | 0.1300 | 1.4480 | 0.1491 | 1.087 (1.085 .. 1.093) |

- **The build part** of the built-once variants is at most 0.03 us.
- **What building in the call costs FKL:** the build and the release of the crop views add 3.7 to 9.0 percent, from `fkl_fused_halfpixel`
  against `fkl_fused_halfpixel_once`.

## Agreement and RPP's arrays

All four runs. A required comparison has 30 batches x 2 regions per run, 240 rows, unless stated otherwise.

| comparison | max over the runs and batches | required rows passed | bitwise equal |
|---|---|---|---|
| fkl_fused_halfpixel_once vs fkl_fused_halfpixel | 0 | 240 of 240 | 120 of 120 batch-runs |
| hand_roi_halfpixel_once vs scalar reference HALF_PIXEL | 7.63e-06 | 240 of 240 | |
| hand_roi_halfpixel_once vs fkl_fused_halfpixel | 0 | informational | 120 of 120 batch-runs |
| fkl_fused, fkl_fused_halfpixel vs their scalar references | 7.63e-06 | 240 of 240 each | |
| fkl_perop_batch vs fkl_fused | 7.63e-06 | 240 of 240 | |
| rpp_crop_batch vs scalar reference HALF_PIXEL, interior | 1.91e-06 | 120 of 120 (interior only) | |
| rpp_crop_batch_unsplit vs rpp_crop_batch | 0 | 240 of 240 | |
| hand_batch, hand_padded, hand_roi vs scalar reference ASYMMETRIC | 7.63e-06 | 240 of 240 each | |
| hand_batch, hand_padded, hand_roi vs fkl_fused | 0 | informational | 120 of 120 each |

RPP's last two columns and rows differ from the HALF_PIXEL reference by design (up to 126), as in the earlier runs.

**RPP's arrays.** In every run and batch, stdout says that RPP's other pinned arrays are unchanged after its calls, and that RPP rewrote exactly
256 * ceil(N / 256) - N padding ROIs (`scripts/rpp_built_once_compare.py` checks it).

## The PCIe links, the clocks and the fast state

**Links.** Every reading of every row (2,700 readings in the noise runs, at the three moments) is:
- host link (`host_link_*`): 16 GT/s x16;
- card-internal link (`card_link_*`): 32 GT/s x16.

The host link did not change at any reading. These are single readings at three moments, not during the timed runs.

**Memory clock** at the start of the timed runs: 1,124 MHz in 851 of the 900 rows, 875 MHz in 37, 772 MHz in 5 and 456 MHz in 7.

**The fast state** of the earlier runs, the points in which `fkl_fused` is more than 4 percent below the median of the crop-batch runs, is rare
here: 3 of 90 points (run0 at 100 and 170 crops, run1 at 10).

```
run0, N = 10 .. 300: .........F......F.............
run1, N = 10 .. 300: F.............................
run2, N = 10 .. 300: ..............................
```

- In those points only `fkl_fused` is faster (-8.0 percent). Every other variant, the built-once ones included, moves by 0.4 percent at most.
- The fast `fkl_fused` rows had the host link at 16 GT/s x16 and the memory clock at 1,124 MHz in 2 of the 3.
- The cause is unknown.

## The eight variants of the whole-call record, old against new

**Method.**
- **old:** `rpp-preprocessing-hand-roi/noise/run0..2`.
- **new:** these runs.
- **Value:** the median of the three per-run medians; the spread is max - min.
- **A change exceeds the spread** when |new - old| > max(spread old, spread new).
- **Flags:** 3 % for a GPU-bound variant, 5 % for a host-bound one.
- **Rotation:** it runs over ten variants instead of eight, and the comparison includes that change.

| variant | change over the 30 batches, min .. median .. max | batches beyond the spread | flagged |
|---|---|---|---|
| fkl_fused | -1.0 .. -0.2 .. +1.3 % | 0 of 30 | 0 of 30 |
| fkl_fused_halfpixel | -0.9 .. -0.1 .. +1.3 % | 1 of 30 | 0 of 30 |
| fkl_perop_batch | -1.7 .. +0.3 .. +11.2 % | 6 of 30 | 1 of 30 |
| rpp_crop_batch | -2.4 .. -0.1 .. +3.2 % | 2 of 30 | 0 of 30 |
| rpp_crop_batch_unsplit | -0.6 .. -0.0 .. +3.4 % | 4 of 30 | 3 of 30 |
| hand_batch | -3.8 .. -0.3 .. +1.4 % | 2 of 30 | 1 of 30 |
| hand_padded | -1.3 .. -0.2 .. +1.3 % | 0 of 30 | 0 of 30 |
| hand_roi | -1.8 .. -0.1 .. +2.0 % | 2 of 30 | 0 of 30 |

- **Medians:** the median changes are within 0.3 percent.
- **Beyond the spread and flagged:**
  - `fkl_perop_batch` at 180 crops: +11.2 %. The old value, 0.4328 ms, had a spread of 0.0449 ms, so it is a fast row of the old runs.
  - `rpp_crop_batch_unsplit` at 190 crops: +3.1 %, and at 250 crops: +3.0 %. The cause is unknown.

  `scripts/rpp_built_once_compare.py` lists them, and `comparison.csv` has every line.

## The natural-image run

One run, so there is no spread.
- `rpp_crop_batch_unsplit / fkl_fused_halfpixel_once`: 3.113 .. 8.684 .. 10.918 over the 30 batches (min .. median .. max); 3.113 .. 8.701 at 10
  .. 150.
- `hand_roi_halfpixel_once / fkl_fused_halfpixel_once`: 0.939 .. 0.970 .. 1.106; 0.960 .. 1.106 at 10 .. 150.

## Clocks and warm-up

| source | sclk at the start of the timed runs, MHz | warm-up, s | rows with warmup_steady = 0 |
|---|---|---|---|
| noise runs | 2830 .. 3413 | 2.00 .. 2.25 | 0 of 900 |
| natural image | 2866 .. 3416 | 2.00 .. 2.25 | 0 of 300 |

## Files

- **`noise/run0..2/`, `natural-image/`:** the two CSVs of each run and its stdout.
- **`runs.log`:** the start and end of each run, its idle wait and its exit.
- **`step0/`:** the step-0 records and the three scripts.
- **`comparison.csv`:** written by `python3 scripts/rpp_built_once_compare.py`, which also prints the tables of this README.
