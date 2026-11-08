# Pipeline (a) on the RX 9060 with the hand-written kernel hand_batch (2026-11-08)

The first runs of `benchmark_rpp_preprocessing_hip` with its sixth variant, `hand_batch`: the hand-written fused kernel of E1c
(`handwritten-cuda/e1_preproc.cu`, `handPreproc`) ported to HIP (fkl-benchmarks #33). **These runs are interim.** A variant `hand_roi`, a
hand-written kernel that reads each crop's pointer, size, pitch and resize factors at run time, will be added, and the final RX 9060 dataset of
Figure 15 will come from a later run of the whole program. `hand_batch` then stays as a diagnostic.

## What was run

- **Program:** `benchmark_rpp_preprocessing_hip` at fkl-benchmarks 6dc0541 (the merge of #33), FKL 64ad9195, RPP 3.2.0, ROCm 10.1.0, gfx1200
  (`step0/versions.txt`).
- **Build:** from a clean worktree with `step0/rpp_run_build.sh` (the flags of the RX 9060 runs). sha256
  3a601c69029b7f64dcc5cd461609c594be44046c4282010fb16f8a2fef20f9cc.
- **Runs (`runs.log`):**
  - three noise runs, `noise/run0`, `run1` and `run2`, with `E5_ROTATE_OFFSET` 0, 1 and 2;
  - the natural-image run, `natural-image` (kodim23 tiled over the frame, offset 0);
  - each run after the idle wait of `step0/rpp_runs_hand.sh` (no GitHub runner job, load average below 0.8), with the program's defaults
    (`E5_ITERS` 30, `E5_WARMUP` 3, `E5_WARMUP_SECONDS` 2.0);
  - 10:59:35 to 11:26:12 UTC, all four exited 0 with "all required numerical checks passed".
- **Variants of a point**, timed in a rotated order (the first is variant (point index + offset) mod 6):
  - `fkl_fused` and `fkl_fused_halfpixel`;
  - `fkl_perop_batch`;
  - `rpp_crop_batch` and `rpp_crop_batch_unsplit`;
  - `hand_batch`.

  The header of `rpp/benchmark_rpp_preprocessing.h` describes them.

**The rotation changed.** With a sixth variant, each point starts at (index + offset) mod 6 instead of mod 5. The five variants of
`results/rx9060/rpp-preprocessing-crop-batch/` therefore run at other positions than in those runs, and every comparison with them below
includes that change.

## Step 0 (`step0/`)

- **The two builds have the same configuration.** `step0-hand-batch.txt` (this build) and `step0-9461389.txt` (the build of the crop-batch
  runs, read again the same day) differ only in the commit, the worktree path and the sha256. That holds for the build information, the commit
  strings compiled in and the compile and link commands. `commands.txt` is the same as the crop-batch runs' `commands-new.txt` once the paths
  are normalised.
- **CMake cache:** 177 entries, no differences once the paths are normalised (`cmake-cache-compare.txt`).
- **Device code:** `devcode-9461389-vs-6dc0541.txt` and `functions/` (scripts/devcode_compare.py). The 97 functions of the crop-batch binary
  are identical, and the 30 functions that exist only in this binary are `e1c::handPreproc<N>`.
- **Dry check:** one run with `E5_ITERS=1` and no warm-up minimum. It is a check, not a measurement, and its times are not used.
  - `dry-check/csv-header.txt`: repository 6dc0541 and FKL 64ad9195, no `-dirty`.
  - All required checks passed.

## Agreement

Every required row passes in all four runs (`scripts/rpp_hand_batch_compare.py` prints the table per run).

The required rows below are the rows of the four runs, 30 batches each, and both regions (interior, last two columns and rows), unless
stated otherwise.

| comparison | max over the runs and batches | required rows passed |
|---|---|---|
| fkl_fused vs scalar reference ASYMMETRIC | 7.63e-06 | 240 of 240 |
| fkl_fused_halfpixel vs scalar reference HALF_PIXEL | 7.63e-06 | 240 of 240 |
| fkl_perop_batch vs fkl_fused | 7.63e-06 | 240 of 240 |
| rpp_crop_batch vs scalar reference HALF_PIXEL, interior | 1.91e-06 | 120 of 120 (interior only) |
| rpp_crop_batch_unsplit vs rpp_crop_batch | 0 | 240 of 240 |
| hand_batch vs scalar reference ASYMMETRIC | 7.63e-06 | 240 of 240 |
| hand_batch vs fkl_fused | 0, **bitwise equal** in every run and batch | informational |

As in the crop-batch runs, RPP's last two columns and rows differ from the HALF_PIXEL reference by design (up to 126); those rows are
informational.

## hand_batch against fkl_fused

The two kernels give the same bits. The ratios are per run, then the median of the three runs (min .. max):

| batch | hand_batch / fkl_fused | the same, compute parts |
|---|---|---|
| 10 | 0.956 (0.950 .. 1.046) | 0.966 (0.959 .. 1.056) |
| 20 | 0.925 (0.814 .. 0.940) | 0.942 (0.828 .. 0.959) |
| 30 | 0.936 (0.928 .. 0.937) | 0.960 (0.954 .. 0.961) |
| 40 | 0.933 (0.925 .. 0.997) | 0.960 (0.953 .. 1.029) |
| 50 | 0.924 (0.913 .. 0.929) | 0.958 (0.944 .. 0.963) |
| 60 | 0.971 (0.909 .. 0.972) | 1.012 (0.946 .. 1.013) |
| 70 | 0.931 (0.908 .. 0.983) | 0.971 (0.944 .. 1.025) |
| 80 | 0.915 (0.914 .. 0.961) | 0.960 (0.958 .. 1.007) |
| 90 | 0.909 (0.898 .. 0.959) | 0.955 (0.941 .. 1.009) |
| 100 | 0.918 (0.899 .. 0.954) | 0.973 (0.947 .. 1.008) |
| 110 | 0.907 (0.903 .. 0.985) | 0.962 (0.954 .. 1.045) |
| 120 | 0.909 (0.886 .. 0.957) | 0.968 (0.937 .. 1.017) |
| 130 | 0.847 (0.842 .. 0.888) | 0.907 (0.895 .. 0.942) |
| 140 | 0.855 (0.840 .. 0.950) | 0.907 (0.892 .. 1.013) |
| 150 | 0.824 (0.821 .. 0.826) | 0.882 (0.875 .. 0.883) |
| 160 | 0.809 (0.809 .. 0.885) | 0.868 (0.866 .. 0.953) |
| 170 | 0.813 (0.803 .. 0.879) | 0.872 (0.870 .. 0.954) |
| 180 | 0.814 (0.775 .. 0.867) | 0.876 (0.836 .. 0.937) |
| 190 | 0.789 (0.775 .. 0.826) | 0.853 (0.831 .. 0.893) |
| 200 | 0.795 (0.793 .. 0.860) | 0.857 (0.856 .. 0.932) |
| 210 | 0.794 (0.793 .. 0.829) | 0.854 (0.854 .. 0.900) |
| 220 | 0.790 (0.782 .. 0.791) | 0.854 (0.848 .. 0.854) |
| 230 | 0.789 (0.787 .. 0.791) | 0.854 (0.850 .. 0.855) |
| 240 | 0.825 (0.794 .. 0.850) | 0.900 (0.859 .. 0.926) |
| 250 | 0.842 (0.794 .. 0.864) | 0.919 (0.858 .. 0.940) |
| 260 | 0.844 (0.791 .. 0.851) | 0.920 (0.858 .. 0.926) |
| 270 | 0.794 (0.785 .. 0.856) | 0.863 (0.857 .. 0.933) |
| 280 | 0.793 (0.791 .. 0.793) | 0.860 (0.859 .. 0.861) |
| 290 | 0.794 (0.788 .. 0.796) | 0.862 (0.855 .. 0.863) |
| 300 | 0.791 (0.765 .. 0.791) | 0.861 (0.831 .. 0.863) |

Range of the median ratio:

| batches | total | compute parts |
|---|---|---|
| 10 .. 150 | 0.824 .. 0.971 | 0.882 .. 1.012 |
| 10 .. 300 | 0.789 .. 0.971 | 0.853 .. 1.012 |

The parts (medians over the three runs, ms):

| batch | fkl_fused total | fkl_fused build | fkl_fused compute | hand_batch total | hand_batch build | hand_batch compute |
|---|---|---|---|---|---|---|
| 10 | 0.0312 | 0.0003 | 0.0308 | 0.0326 | 0.0000 | 0.0326 |
| 50 | 0.0476 | 0.0017 | 0.0459 | 0.0442 | 0.0000 | 0.0442 |
| 100 | 0.0660 | 0.0036 | 0.0624 | 0.0630 | 0.0000 | 0.0629 |
| 150 | 0.0895 | 0.0057 | 0.0837 | 0.0739 | 0.0001 | 0.0738 |
| 200 | 0.1076 | 0.0078 | 0.0998 | 0.0857 | 0.0001 | 0.0856 |
| 250 | 0.1171 | 0.0095 | 0.1076 | 0.1002 | 0.0001 | 0.1001 |
| 300 | 0.1456 | 0.0119 | 0.1334 | 0.1149 | 0.0001 | 0.1148 |

At N = 300, `fkl_fused` takes 30.7 us longer than `hand_batch`. Its build, the crop views and the operations, accounts for 11.8 us. The other
18.6 us are in the compute part. For `fkl_fused` that part holds the launch, the kernel and the synchronisation, and also the release of its N
crop views at the end of the call, which is host work. `hand_batch`'s compute part has no such release: its parameter block is a plain array.
The program does not time that release on its own.

The parameter blocks differ a lot. In the gfx1200 code-object metadata, FKL's fused kernel at N = 300 has 14,752 bytes of kernel arguments,
48 bytes per crop. `e1c::handPreproc<300>` has 2,696 bytes, 8 per crop. FKL's kernel uses 21 VGPRs and the hand-written kernel 24. Neither uses
scratch. The cause of the difference in the compute part is not established. A hand-written kernel with FKL's parameter-block size (E1c's
`hand_padded` diagnostic) would separate the size of the parameter block from the generated code; it was not run.

### A state in which only FKL's fused kernels are faster

`fkl_fused` and `fkl_fused_halfpixel` are 5 to 9 percent faster than in the crop-batch runs in 37 and 36 of the 90 rows of these runs. The
crop-batch runs have such rows too, but only 4 and 2 of 90 (rows more than 4 percent below the median of the six runs at their batch).
- **When.** These rows come in blocks of consecutive points. In run0 they are at N = 10 .. 20, 40 .. 120 and 240 .. 270; in run1 at N = 10,
  40 .. 80 and 240 .. 250; in run2 at N = 50 .. 120, 140 and 160 .. 210, with the halfpixel variant also at 250 .. 280. The two variants are fast
  in the same points in 32 cases. So this is a state of minutes, not noise of single rows.
- **Who.** `hand_batch` is not faster in those points (+0.0 percent against its median), the RPP variants are not either (0 to 2 rows), and
  `fkl_perop_batch` is faster in 6 of 90 rows.
- **The clock.** The graphics clock at the start of the timed runs does not explain it: its median is 3,129 MHz in the fast rows and 3,175 MHz
  in the others. The memory clock and the PCIe link state are not recorded.
- **Effect on the ratio.** `hand_batch / fkl_fused` is 0.915 in the points with the fast state and 0.796 in the others. The compute parts give
  0.959 and 0.863. At the 20 batches with both kinds of runs, the ratio is 0.906 against 0.828. The spread of the ratio at a batch, its
  min .. max, therefore often spans the two states.
- **Cause:** unknown.

## rpp_crop_batch_unsplit / fkl_fused

The ratio of the paper. Per run, then the median of the three runs:

| batches | these runs | the crop-batch runs |
|---|---|---|
| 10 .. 150 | 3.465 .. 7.908 | 3.171 .. 8.160 |
| 10 .. 300 | 3.465 .. 10.228 | 3.171 .. 9.957 |

`scripts/rpp_hand_batch_compare.py` prints the value per batch, and `comparison.csv` has it with min .. max.

## The five variants against the crop-batch runs

**Method.**
- **old:** `rpp-preprocessing-crop-batch/noise/new-run0..2` (9461389, the same day).
- **new:** these runs.
- **Value:** the median of the three per-run medians; the spread is max - min.
- **A change exceeds the spread** when |new - old| > max(spread old, spread new).
- **Flags:** 3 % for a GPU-bound variant, 5 % for a host-bound one, where host-bound means the build part is at least half of the run.

The rule is the one of `scripts/rpp_crop_batch_compare.py`. It also counts `rpp_crop_batch_unsplit` as GPU-bound, because that variant has no
parts. Every line of all 30 batches is in `comparison.csv`.

| variant | change over the 30 batches, min .. median .. max | batches beyond the spread | flagged | cause |
|---|---|---|---|---|
| fkl_fused | -8.4 .. -0.4 .. +0.8 % | 3 of 30 (N = 60, 70, 80: -7.7, -7.4, -6.7 %) | 14 of 30 | the fast state above; rotation |
| fkl_fused_halfpixel | -7.3 .. -0.7 .. +4.7 % | 6 of 30 (N = 50, 90, 110, 240, 250: -5.5 .. -7.2 %; 220: +0.9 %) | 13 of 30 | the fast state above; rotation |
| fkl_perop_batch | -9.9 .. -0.2 .. +4.8 % | 6 of 30 (N = 20: -6.0 %, 180: -9.9 %; four within 3 %) | 5 of 30 | unknown |
| rpp_crop_batch | -3.5 .. -0.1 .. +7.2 % | 4 of 30 (N = 10: +7.2 %; three within 3.5 %) | 1 of 30 | unknown |
| rpp_crop_batch_unsplit | -3.8 .. -0.1 .. +5.8 % | 9 of 30 (N = 10: +5.8 %; 150 .. 250: -3.0 .. -3.8 %) | 7 of 30 | unknown |

- **Medians:** the median changes are within 1 percent for all five variants.
- **FKL's two fused variants:** their larger changes are all faster rows, and the fast state explains them.
- **The RPP variants at N = 10:** +7.2 % and +5.8 %. This is not the rotation, because at N = 10 the RPP variants ran at the same positions as
  in the crop-batch runs. The cause is unknown.

## The natural-image run

One run each, so there is no spread. Against the crop-batch natural-image run, the changes at the batches of the tables are:
- `fkl_fused` and `fkl_fused_halfpixel`: -13.1 to +6.4 %;
- `fkl_perop_batch`: -1.8 to +0.7 %;
- `rpp_crop_batch` and `rpp_crop_batch_unsplit`: -5.5 to +7.5 %.

`hand_batch / fkl_fused` over the 30 batches is 0.782 .. 0.858 .. 1.016 (min .. median .. max).

## Clocks and warm-up

Min .. max over every row of a source:

| source | sclk at the start of the timed runs, MHz | sclk after them, MHz | warm-up, s | rows with warmup_steady = 0 |
|---|---|---|---|---|
| crop-batch runs (old) | 2821 .. 3414 | 3047 .. 3418 | 2.00 .. 2.25 | 0 of 450 |
| these runs (new) | 2212 .. 3419 | 2866 .. 3419 | 2.00 .. 2.75 | 0 of 540 |
| crop-batch natural image | 3050 .. 3416 | 3086 .. 3420 | 2.00 .. 2.25 | 0 of 150 |
| this natural image | 2256 .. 3419 | 2907 .. 3424 | 2.00 .. 2.50 | 0 of 180 |

Every warm-up reached the steady state. The graphics clock is a single reading at the start of the timed runs. Its lowest values in these
runs (2,212 and 2,256 MHz) are below the crop-batch runs' lowest, but the readings after the timed runs are at least 2,866 MHz.

## RPP facts behind the program's choices

These come from the RPP 3.2.0 source (ROCm/rocm-libraries, tag therock-10.1, projects/rpp; the installed headers are the same).
- **ROI overflow.** The kernel that converts the ROIs from XYWH to LTRB in `rppt_resize_mirror_normalize` has no bounds check. It runs
  256 * ceil(N / 256) threads that each rewrite one 16-byte ROI, so it writes (256 * ceil(N / 256) - N) * 16 bytes past the end of the ROI array:
  3,392 bytes at N = 300.
  - In these runs the array has N entries. It is harmless only when the array has pages of its own. The agreement rows, which need correct
    means, standard deviations and channel orders, passed in every run.
  - The next change of the program allocates 256 * ceil(N / 256) entries.
- **Output width.** Both RPP kernels write 8 pixels per thread and check only the first, so the output widths must be multiples of 8. The
  program keeps them so.
- **Pinned output sizes.** The header documents `dstImgSizes` as host memory, but the kernel reads it on the GPU, so it must be pinned. The
  program allocates it with `hipHostMalloc`.

## Files

- **`noise/run0..2/`, `natural-image/`:** the two CSVs of each run and its stdout.
- **`runs.log`:** the start and end of each run, its idle wait and its exit.
- **`step0/`:** the step-0 records (see above) and the three scripts.
- **`comparison.csv`:** written by `python3 scripts/rpp_hand_batch_compare.py`, which also prints the tables of this README.
