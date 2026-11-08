# Pipeline (a) on the RX 9060 with the hand-written kernels hand_batch, hand_padded and hand_roi: the whole-call record (2026-11-08)

These runs of `benchmark_rpp_preprocessing_hip` at fkl-benchmarks 151d517 (#36) time every variant as the whole call, its host-side build included.

**This is the whole-call record, not the dataset of Figure 15.** Figure 15 leaves the host-side build of our implementations out. The paper does
not describe the builder API, so FKL and the hand-written kernels are timed with their operations and parameter blocks built once. That run, with
the variants `fkl_fused_once`, `fkl_fused_halfpixel_once` and `hand_roi_once`, comes in a later folder. The RPP baseline is timed as a user calls it
in both.

## What was run

- **Program:** `benchmark_rpp_preprocessing_hip` at fkl-benchmarks 151d517, FKL 64ad9195, RPP 3.2.0, ROCm 10.1.0, gfx1200 (`step0/versions.txt`).
- **Build:** from a clean worktree with `step0/rpp_run_build.sh`. sha256 435f35ff741260fbfcf23919d3c95e77751d6d6b4541041a27ebcac24e91e215.
- **Runs (`runs.log`):**
  - three noise runs, `noise/run0`, `run1` and `run2`, with `E5_ROTATE_OFFSET` 0, 1 and 2;
  - the natural-image run, `natural-image` (kodim23 tiled over the frame, offset 0);
  - each after the idle wait of `step0/rpp_runs_roi.sh`, with the program's defaults (`E5_ITERS` 30, `E5_WARMUP` 3, `E5_WARMUP_SECONDS` 2.0);
  - 12:01:21 to 12:35:38 UTC, all four exited 0 with "all required numerical checks passed".
- **Eight variants per point,** in a rotated order (the first is variant (point index + offset) mod 8):
  - `fkl_fused` and `fkl_fused_halfpixel`;
  - `fkl_perop_batch`;
  - `rpp_crop_batch` and `rpp_crop_batch_unsplit`;
  - the hand-written kernels `hand_batch` (E1c), `hand_padded` (E1c's diagnostic, with FKL's kernel-argument size) and `hand_roi` (E4's kernel for
    crops of any location and size, with its library-free CPU function `handRoiParameters`).

  The header of `rpp/benchmark_rpp_preprocessing.h` describes them.
- **New columns:** each timing row also records the memory clock and the PCIe link speed and width, at the start of the warm-up, at the start of
  the timed runs and after them.

**The FKL time of the hand ratios** is the `fkl_fused` of the same point. The rotation over eight variants puts it at another position than the
hand-written kernel, and the warm-up rule of each variant is meant to make the position irrelevant. E4 on the RTX 4090 does it differently: it
brackets its hand-written kernels with two adjacent FKL timings.

**The rotation changed again.** With eight variants instead of six, every variant runs at other positions than in the hand_batch runs
(`results/rx9060/rpp-preprocessing-hand-batch/`). Every comparison with those runs below includes that change.

## Step 0 (`step0/`)

- **Same configuration as the hand_batch build.** `step0-hand-roi.txt` (this build) and `step0-6dc0541.txt` (the build of the hand_batch runs)
  differ only in the name and the worktree path once commits and paths are normalised. That covers the build information, the commit strings
  compiled in and the compile and link commands. `commands.txt` is the same as the hand_batch runs' `commands.txt`.
- **CMake cache:** 177 entries, no differences once the paths are normalised (`cmake-cache-compare.txt`).
- **Device code:** `devcode-6dc0541-vs-151d517.txt` and `functions/`. The 127 functions of the hand_batch binary are identical, and the 60
  functions that exist only in this binary are 30 `e1c::handPreprocPadded<N>` and 30 `e1c::handPreprocRoi<N>`.
- **Dry check:** one run with `E5_ITERS=1` and no warm-up minimum; its times are not used.
  - `dry-check/csv-header.txt`: repository 151d517 and FKL 64ad9195, no `-dirty`.
  - All required checks passed.

## Agreement and RPP's arrays

All four runs: required rows are 30 batches x 2 regions per run, 240 per comparison, unless stated otherwise.

| comparison | max over the runs and batches | required rows passed |
|---|---|---|
| fkl_fused vs scalar reference ASYMMETRIC | 7.63e-06 | 240 of 240 |
| fkl_fused_halfpixel vs scalar reference HALF_PIXEL | 7.63e-06 | 240 of 240 |
| fkl_perop_batch vs fkl_fused | 7.63e-06 | 240 of 240 |
| rpp_crop_batch vs scalar reference HALF_PIXEL, interior | 1.91e-06 | 120 of 120 (interior only) |
| rpp_crop_batch_unsplit vs rpp_crop_batch | 0 | 240 of 240 |
| hand_batch, hand_padded and hand_roi vs scalar reference ASYMMETRIC | 7.63e-06 | 240 of 240 each |
| hand_batch, hand_padded and hand_roi vs fkl_fused | 0, **bitwise equal** in all 120 batch-runs, for each | informational |

RPP's last two columns and rows differ from the HALF_PIXEL reference by design (up to 126), as in the earlier runs.

**RPP 3.2.0's ROI overflow.** The ROI array has 256 * ceil(N / 256) entries.
- After the checked RPP calls of every batch, the program compared RPP's other pinned arrays (output sizes, mean, standard deviation, mirror,
  channel order) byte by byte with host copies. It also checked that none of them lies in the padding of the ROI array. It repeated both checks
  after the timing.
- Every run's stdout reports for all 30 batches that RPP's other pinned arrays are unchanged, and that RPP rewrote exactly
  256 * ceil(N / 256) - N padding ROIs (`scripts/rpp_hand_roi_compare.py` checks it).

## The hand-written kernels against fkl_fused

Per run, then the median of the three runs (min .. max). The compute part is the launch, the kernel and the synchronisation.

| batch | hand_roi / fkl_fused | compute parts | hand_batch / fkl_fused | compute parts | hand_padded / fkl_fused | compute parts |
|---|---|---|---|---|---|---|
| 10 | 0.966 (0.965 .. 1.062) | 0.974 | 0.941 (0.935 .. 0.958) | 0.949 | 0.968 (0.959 .. 1.035) | 0.977 |
| 50 | 0.964 (0.962 .. 0.971) | 0.994 | 0.908 (0.902 .. 0.913) | 0.939 | 0.942 (0.935 .. 0.943) | 0.973 |
| 100 | 0.945 (0.874 .. 0.996) | 0.992 | 0.896 (0.887 .. 0.903) | 0.946 | 0.929 (0.917 .. 0.984) | 0.979 |
| 150 | 0.916 (0.907 .. 0.918) | 0.976 | 0.816 (0.815 .. 0.819) | 0.873 | 0.910 (0.901 .. 0.912) | 0.973 |
| 200 | 0.891 (0.886 .. 0.903) | 0.957 | 0.792 (0.791 .. 0.820) | 0.855 | 0.846 (0.845 .. 0.865) | 0.912 |
| 250 | 0.890 (0.882 .. 0.891) | 0.957 | 0.795 (0.782 .. 0.797) | 0.859 | 0.841 (0.838 .. 0.846) | 0.913 |
| 300 | 0.894 (0.883 .. 0.955) | 0.968 | 0.791 (0.784 .. 0.848) | 0.862 | 0.837 (0.826 .. 0.890) | 0.911 |

Range of the median ratio:

| ratio | 10 .. 150 | 10 .. 300 |
|---|---|---|
| hand_roi / fkl_fused | 0.916 .. 0.980 | 0.885 .. 0.980 |
| the same, compute parts | 0.949 .. 1.010 | 0.949 .. 1.010 |
| hand_batch / fkl_fused | 0.816 .. 0.941 | 0.780 .. 0.941 |
| hand_padded / fkl_fused | 0.907 .. 0.968 | 0.837 .. 0.968 |

All 30 batches, with min .. max for each ratio, are printed by `scripts/rpp_hand_roi_compare.py` and are in `comparison.csv`.

The parts (medians over the three runs, ms):

| batch | fkl_fused total | build | compute | hand_roi total | build | compute | hand_batch compute | hand_padded compute |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.0344 | 0.0003 | 0.0340 | 0.0332 | 0.0001 | 0.0332 | 0.0322 | 0.0330 |
| 50 | 0.0518 | 0.0017 | 0.0501 | 0.0499 | 0.0002 | 0.0497 | 0.0469 | 0.0484 |
| 100 | 0.0699 | 0.0036 | 0.0663 | 0.0659 | 0.0003 | 0.0656 | 0.0630 | 0.0650 |
| 150 | 0.0898 | 0.0058 | 0.0838 | 0.0823 | 0.0005 | 0.0818 | 0.0734 | 0.0817 |
| 200 | 0.1084 | 0.0079 | 0.1003 | 0.0960 | 0.0006 | 0.0954 | 0.0856 | 0.0915 |
| 250 | 0.1264 | 0.0097 | 0.1167 | 0.1127 | 0.0008 | 0.1118 | 0.0997 | 0.1068 |
| 300 | 0.1457 | 0.0120 | 0.1337 | 0.1303 | 0.0009 | 0.1294 | 0.1152 | 0.1218 |

What the parts show at N = 300:
- **FKL's build** (crop views and operations) takes 12.0 us of FKL's 145.7 us. `handRoiParameters`, `hand_roi`'s build, takes 0.9 us.
- **FKL's compute part also holds the release of its N crop views** at the end of the call, which is host work. The hand-written kernels' compute
  part has no such release. The program does not time that release on its own.
- **The compute parts run in this order:**

  | variant | compute part |
  |---|---|
  | `hand_batch` (8 bytes of parameters per crop) | 115.2 us |
  | `hand_padded` (the same code, with FKL's kernel-argument size) | 121.8 us |
  | `hand_roi` (32 bytes per crop, read in the kernel) | 129.4 us |
  | `fkl_fused` | 133.7 us |

  So the kernel-argument size alone accounts for 6.6 us of the 18.5 us between `hand_batch` and `fkl_fused`, and `hand_roi`, which reads its
  per-crop parameters as FKL does, is within 4.3 us of FKL. The mechanism is not established.

## The fast state

The interim hand_batch runs found periods of minutes in which FKL's fused kernels were 5 to 9 percent faster. Here it occurs in 14 of the 90
points: the points in which `fkl_fused` is more than 4 percent below the median of the crop-batch runs at its batch, in blocks of consecutive
points.

```
run0, N = 10 .. 300: FF...............FFFFFFFFF....
run1, N = 10 .. 300: .......F.F...................F
run2, N = 10 .. 300: ..............................
```

| variant | in the fast points | in the other points |
|---|---|---|
| fkl_fused | -7.8 % | +0.0 % |
| fkl_fused_halfpixel | -7.1 % | +0.0 % |
| hand_padded | -6.0 % | +0.0 % |
| hand_roi | -6.1 % | +0.0 % |
| hand_batch | -4.4 % | +0.0 % |
| fkl_perop_batch | -0.6 % | +0.0 % |
| rpp_crop_batch | -0.1 % | +0.0 % |
| rpp_crop_batch_unsplit | -0.0 % | +0.0 % |

Each variant is compared with the median of its three runs at its batch.

- **It is not only FKL.** In this run the hand-written kernels are faster in the same points too. The fused pipeline kernels are all affected,
  and the variants that are bound by many launches or copies are not. The effect is largest for the kernels with the largest parameter blocks.
- **This revises the interim note.** In the hand_batch runs, `hand_batch` was not faster in FKL's fast points.
- **The new columns do not explain it.**
  - The PCIe link read 32 GT/s x16 at every reading of every row.
  - The memory clock was at 1,124 MHz in 12 of the 14 fast `fkl_fused` rows and in 71 of the 76 others.
  - The graphics clock at the start of the timed runs had a median of 3,113 MHz in the fast rows and 3,156 MHz in the others.
  - These are single readings at three moments, not during the timed runs.
- **Cause:** unknown.

## rpp_crop_batch_unsplit / fkl_fused

Per run, then the median of the three runs:

| batches | these runs | the hand_batch runs |
|---|---|---|
| 10 .. 150 | 2.990 .. 8.149 | 3.465 .. 7.908 |
| 10 .. 300 | 2.990 .. 9.922 | 3.465 .. 10.228 |

The value per batch, with min .. max, is in `comparison.csv`.

## The six variants of the hand_batch runs, old against new

**Method.**
- **old:** `rpp-preprocessing-hand-batch/noise/run0..2`.
- **new:** these runs.
- **Value:** the median of the three per-run medians; the spread is max - min.
- **A change exceeds the spread** when |new - old| > max(spread old, spread new).
- **Flags:** 3 % for a GPU-bound variant, 5 % for a host-bound one, where host-bound means the build part is at least half of the run.

| variant | change over the 30 batches, min .. median .. max | batches beyond the spread | flagged |
|---|---|---|---|
| fkl_fused | -0.9 .. +0.9 .. +10.4 % | 5 of 30 | 14 of 30 |
| fkl_fused_halfpixel | -4.8 .. +1.0 .. +8.3 % | 5 of 30 | 13 of 30 |
| fkl_perop_batch | -4.8 .. -0.1 .. +6.2 % | 2 of 30 | 4 of 30 |
| rpp_crop_batch | -6.4 .. +0.0 .. +3.9 % | 4 of 30 | 1 of 30 |
| rpp_crop_batch_unsplit | -5.9 .. +0.1 .. +3.9 % | 3 of 30 | 4 of 30 |
| hand_batch | -2.8 .. +0.2 .. +7.3 % | 2 of 30 | 6 of 30 |

- **Medians:** the median changes are within 1 percent.
- **FKL's larger changes are mostly slower rows here.** This run has fewer fast points than the hand_batch runs had (14 against 37 of 90), so the
  fast state explains them.
- **Rotation:** it changed (mod 8 instead of mod 6). The comparison includes that change.

## The natural-image run

One run, so there is no spread.
- `hand_roi / fkl_fused`: 0.878 .. 0.922 .. 0.977 over the 30 batches (min .. median .. max).
- `hand_batch / fkl_fused`: 0.779 .. 0.835 .. 0.944.
- `hand_padded / fkl_fused`: 0.798 .. 0.910 .. 0.973.
- The six old variants against the hand_batch natural-image run: -7.1 .. +0.3 .. +10.0 %.

## Clocks and warm-up

| source | sclk at the start of the timed runs, MHz | mclk at the start of the timed runs, MHz | PCIe | warm-up, s | rows with warmup_steady = 0 |
|---|---|---|---|---|---|
| noise runs | 2267 .. 3420 | 456 .. 1124 | 32 GT/s x16 | 2.00 .. 2.75 | 0 of 720 |
| natural image | 2855 .. 3413 | 772 .. 1124 | 32 GT/s x16 | 2.00 .. 2.00 | 0 of 240 |

## Files

- **`noise/run0..2/`, `natural-image/`:** the two CSVs of each run and its stdout.
- **`runs.log`:** the start and end of each run, its idle wait and its exit.
- **`step0/`:** the step-0 records and the three scripts.
- **`comparison.csv`:** written by `python3 scripts/rpp_hand_roi_compare.py`, which also prints the tables of this README.
