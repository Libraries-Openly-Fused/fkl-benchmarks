# The maximum speedup of vertical and horizontal fusion on the RX 9060 (2026-11-08)

This folder holds the RX 9060 data for the GPU-size figure (System 6), from `fkl-self/vfhf/benchmark_vfhf_max_speedup`. The program runs FKL
against itself on the 2025 chain: 50 images of 60x120 uchar, the conversion to float, then NUM_OPS operations that alternate a
multiplication and an addition by 0.54.
- **fused:** one `executeOperations` for the whole batch.
- **baseline:** one `executeOperations` per operation and per image, i.e. 50 (1 + NUM_OPS) calls per repetition.

The program's header has the full method.

## The answer

The speedup rises with NUM_OPS and flattens from about 9002 operations on. Its **maximum is about 4,290**:

| estimator | maximum of the per-point median of the three runs (NUM_OPS) | run0 | run1 | run2 |
|---|---|---|---|---|
| **block-paired mean of ratios** (`speedup_block_paired_mean_of_ratios`) | **4,294 (16002)** | 4,272 (14002) | 4,296 (19902) | 4,295 (16002) |
| ratio of means (`speedup_ratio_of_means`) | 4,290 (19902) | 4,290 (19902) | 4,298 (19902) | 4,294 (16002) |
| ratio of medians (`speedup_ratio_of_medians`) | 4,297 (19902) | 4,297 (19902) | 4,299 (19902) | 4,293 (17002, extra) |

All three estimators and all three runs put the maximum between 4,272 and 4,299, a range of 0.6 %.

- **Where the maximum is:** it lies on a plateau. From 10002 to 19902 operations, every point's median is within 1.5 % of 4,294, except 15002, 17002 and 18002. At each of those three points the slow state described below hit two of the three runs, so its median is a slow value; they are left as measured. So the noise does not resolve whether the maximum is at 16002 or at 19902.
- **Within 5 % of the maximum:** every point from 6002 to 14002, then 16002 and 19902.

**The comparable estimator.** The block-paired mean of ratios is the one to compare with systems 1 to 5 of the figure. Those systems plot the maximum over NUM_OPS of the 2025 harness's "Mean Speedup": the mean over the iterations of each iteration's ratio, baseline time over fused time (`opencv-cuda/testsCommon.cuh`).
- The 2025 harness paired the two sides within each iteration: it timed them back to back.
- This program pairs them by index within two separately timed blocks: the i-th timed repetition of the baseline over the i-th of the fused side, for the first min(reps) repetitions.

The ratio of means and the ratio of medians are given next to it.

## What was run

- **The program:** fkl-benchmarks main 7b96c47 (#39) with FKL 64ad9195.
  - Binary `benchmark_vfhf_max_speedup_hip`, sha256 d7a4e69532614ba826944256bf6cb664130fa14557c9e4ff3911f312988d3ce5.
  - ROCm 10.1.0, AMD clang 24.0.0git (d6f6cb69), Release, gfx1200.
- **The machine:** the RX 9060 (gfx1200). Host link PCIe 16 GT/s x16 (the B550's PCIe 4.0); the card's own link to its PCIe switch 32 GT/s x16; ASPM L1 off on both.
- **The points:** 2, 102, 502, 1002, 2002, every 1000 to 10002, then 12002, 14002, 16002, 18002 and 19902; 15002 and 17002 were added afterwards (below).
- **The timing:** the program's defaults.
  - Warm-up: at least 2 s and 2 repetitions, until steady, at most 10 s.
  - Repetitions: fused 100; baseline min(100, max(10, as many as fit in 30 s)).
- **Three runs**, each after the idle wait (no CI job and a 1-minute load below 0.8), with the machine facts before and after:

| run | order | start (UTC) | end (UTC) | idle wait |
|---|---|---|---|---|
| run0 | fused first | 15:35:00 | 15:44:04 | 0 s |
| run1 | baseline first (`VFHF_BASELINE_FIRST=1`) | 15:44:37 | 15:53:41 | 30 s |
| run2 | fused first | 15:54:14 | 16:03:18 | 30 s |

  No CI job ran on the machine during the runs: the runner's newest Worker log is from 14:13Z.
- **The extra points 15002 and 17002**, in `runs-extra/`.
  - They were added because the comparable estimator's maximum fell inside the range, at 16002, as agreed before the runs.
  - They come from a binary of their own: the same commit and options with `FKLB_VFHF_POINTS=15002;17002` (sha256 e49040df...). `runs-extra/step0/` shows that its build facts are those of the run build apart from that option, and that its device code is the same apart from the two new fused kernels.
  - They were measured in three runs of their own, in the same order and with the same protocol:

| run | order | start (UTC) | end (UTC) | idle wait |
|---|---|---|---|---|
| run0 | fused first | 16:18:01 | 16:19:18 | 0 s |
| run1 | baseline first | 16:19:51 | 16:21:09 | 30 s |
| run2 | fused first | 16:21:42 | 16:22:59 | 30 s |

  In the tables each extra run joins the run of the same name, and its points are marked "(extra)".
- **A check run** before them (15:30:42-15:31:22Z) covered the points up to 1002 operations: `check-run/`.

## The speedups per point

The median of the three runs (min .. max). The spread is (max - min) / median of the block-paired mean of ratios.

| NUM_OPS | block-paired mean of ratios | ratio of means | ratio of medians | spread | flag |
|---|---|---|---|---|---|
| 2 | 23 (23 .. 24) | 23 (23 .. 24) | 23 (23 .. 24) | 6.0 % | > 5 % |
| 102 | 692 (687 .. 695) | 692 (687 .. 695) | 693 (688 .. 697) | 1.1 % |  |
| 502 | 2260 (2228 .. 2326) | 2259 (2227 .. 2326) | 2266 (2238 .. 2330) | 4.4 % | > 3 % |
| 1002 | 3041 (3018 .. 3055) | 3040 (3018 .. 3055) | 3043 (3018 .. 3055) | 1.2 % |  |
| 2002 | 3610 (3595 .. 3611) | 3610 (3595 .. 3611) | 3611 (3602 .. 3615) | 0.4 % |  |
| 3002 | 3833 (3827 .. 3847) | 3837 (3831 .. 3848) | 3845 (3843 .. 3849) | 0.5 % |  |
| 4002 | 3966 (3965 .. 3972) | 3967 (3963 .. 3972) | 3975 (3975 .. 3976) | 0.2 % |  |
| 5002 | 4045 (4016 .. 4052) | 4053 (4024 .. 4057) | 4063 (4061 .. 4063) | 0.9 % |  |
| 6002 | 4085 (3578 .. 4115) | 4102 (3648 .. 4122) | 4117 (3747 .. 4125) | 13.1 % | > 5 % |
| 7002 | 4161 (4156 .. 4171) | 4160 (4156 .. 4160) | 4169 (4167 .. 4170) | 0.3 % |  |
| 8002 | 4138 (4134 .. 4143) | 4142 (4139 .. 4148) | 4143 (4143 .. 4147) | 0.2 % |  |
| 9002 | 4216 (4204 .. 4220) | 4221 (4220 .. 4230) | 4229 (4227 .. 4235) | 0.4 % |  |
| 10002 | 4244 (4243 .. 4258) | 4259 (4253 .. 4259) | 4262 (4258 .. 4263) | 0.4 % |  |
| 12002 | 4263 (4245 .. 4266) | 4266 (4258 .. 4270) | 4267 (4262 .. 4272) | 0.5 % |  |
| 14002 | 4270 (4260 .. 4272) | 4273 (4227 .. 4277) | 4279 (4275 .. 4280) | 0.3 % |  |
| 15002 (extra) | 3917 (3866 .. 4258) | 3976 (3937 .. 4285) | 3963 (3732 .. 4290) | 10.0 % | > 5 % |
| 16002 | 4294 (4248 .. 4295) | 4284 (4236 .. 4294) | 4292 (4291 .. 4292) | 1.1 % |  |
| 17002 (extra) | 3851 (3835 .. 4284) | 3859 (3817 .. 4293) | 3785 (3779 .. 4293) | 11.7 % | > 5 % |
| 18002 | 3949 (3805 .. 4241) | 3886 (3880 .. 4287) | 3815 (3799 .. 4294) | 11.0 % | > 5 % |
| 19902 | 4231 (3912 .. 4296) | 4290 (3917 .. 4298) | 4297 (3876 .. 4299) | 9.1 % | > 5 % |

- At 13 of the 20 points the three runs agree within 1.2 %.
- The seven flagged points (2, 502, 6002, 15002, 17002, 18002, 19902) are exactly the points where a run's fused block was in the slow state.

## The times

The mean of the timed repetitions of a side; the median of the three runs (min .. max).

| NUM_OPS | fused, ms | baseline, ms | baseline per call, us | baseline repetitions (rule) |
|---|---|---|---|---|
| 2 | 0.0193 (0.0182 .. 0.0194) | 0.44 (0.44 .. 0.44) | 2.913 | 100 (max) |
| 102 | 0.0212 (0.0211 .. 0.0213) | 14.66 (14.65 .. 14.67) | 2.847 | 100 (max) |
| 502 | 0.0317 (0.0307 .. 0.0321) | 71.57 (71.49 .. 71.62) | 2.846 | 100 (max) |
| 1002 | 0.0469 (0.0467 .. 0.0473) | 142.70 (142.67 .. 142.74) | 2.845 | 100 (max) |
| 2002 | 0.0789 (0.0789 .. 0.0792) | 284.89 (284.89 .. 285.03) | 2.845 | 100 (max) |
| 3002 | 0.1113 (0.1111 .. 0.1115) | 427.18 (427.15 .. 427.38) | 2.845 | 70 (budget) |
| 4002 | 0.1436 (0.1434 .. 0.1437) | 569.44 (569.42 .. 569.61) | 2.845 | 52 (budget) |
| 5002 | 0.1756 (0.1754 .. 0.1757) | 711.78 (707.18 .. 711.82) | 2.845 | 42 (budget) |
| 6002 | 0.2082 (0.2072 .. 0.2374) | 854.10 (854.00 .. 866.03) | 2.846 | 34 (budget), 35 (budget) |
| 7002 | 0.2395 (0.2395 .. 0.2397) | 996.54 (996.31 .. 996.55) | 2.846 | 30 (budget) |
| 8002 | 0.2749 (0.2745 .. 0.2752) | 1138.65 (1138.47 .. 1138.79) | 2.846 | 26 (budget) |
| 9002 | 0.3035 (0.3028 .. 0.3035) | 1281.14 (1280.78 .. 1281.17) | 2.846 | 23 (budget) |
| 10002 | 0.3342 (0.3342 .. 0.3347) | 1423.34 (1423.03 .. 1423.52) | 2.846 | 21 (budget) |
| 12002 | 0.4001 (0.4000 .. 0.4011) | 1707.68 (1706.77 .. 1707.74) | 2.845 | 17 (budget) |
| 14002 | 0.4663 (0.4657 .. 0.4714) | 1992.46 (1992.04 .. 1992.62) | 2.846 | 15 (budget) |
| 15002 (extra) | 0.5367 (0.4981 .. 0.5418) | 2133.75 (2133.18 .. 2134.23) | 2.844 | 14 (budget) |
| 16002 | 0.5316 (0.5302 .. 0.5376) | 2277.17 (2276.66 .. 2277.19) | 2.846 | 13 (budget) |
| 17002 (extra) | 0.6267 (0.5636 .. 0.6336) | 2418.29 (2418.27 .. 2419.39) | 2.845 | 12 (budget) |
| 18002 | 0.6593 (0.5976 .. 0.6605) | 2561.85 (2561.67 .. 2562.89) | 2.846 | 11 (budget) |
| 19902 | 0.6602 (0.6589 .. 0.7231) | 2831.96 (2831.83 .. 2832.57) | 2.846 | 10 (min) |

- **The baseline is launch-bound.** It takes 2.844 to 2.847 µs per call from 102 to 19902 operations. The three runs' means are within 0.2 % of each other, apart from 5002 (0.65 %) and 6002 (1.4 %), both from run2. Its time includes the host's building and launching of each call. The GPU events cannot tell whether the host's enqueue or the GPU's dispatch sets that rate.
- **The fused side is GPU-bound.** It grows by about 0.032 µs per operation from 2002 operations on.
- **One repetition count differs.** Run2 at 6002 timed 34 baseline repetitions instead of 35, because its last warm-up repetition was slower.

## The slow state of the fused kernel

In some blocks the fused kernel runs 3 to 15 % slower, measured as the block's median over the median of the fastest run at that point:
3.0 to 5.5 % at 2 and 502, and 8.2 to 14.9 % at 6002, 15002, 17002, 18002 and 19902. Most repetitions of such a block are slow (`comparison.csv`, column
`fused_slow_reps`). A repetition counts as slow when it is more than 6 % above the fastest fused repetition of the three runs at its point. The sclk and mclk readings around such a block are the same as around a fast one.

- **12 of the 60 blocks have at least half of their repetitions slow:** run1 and run2 at 2 and 502; run2 at 6002; run0 and run2 at 15002; run0 and run1 at 17002; run1 and run2 at 18002; run2 at 19902. These are the flagged points of the table above.
- **4 more blocks begin with slow repetitions,** the first of them 5.6 to 15.0 % above the block's median: run0 at 3002 (the first three), 16002, 18002 and 19902 (the first one each).
- **The temperatures:** run0 started at 42 C, run1 and run2 at 50 to 51 C, and all three ended at 53 to 58 C. The extra runs started at 44, 50 and 51 C and ended at 53 to 57 C. Whether temperature plays a part is not established.
- **The cause is not known.** A similar two-state behaviour of fused kernels was seen in the RPP runs (`../rpp-preprocessing-built-once/`, the "fast state"), whose cause was not found either.

Per block: the slow repetitions, and the first timed repetition over the median of its block.

| NUM_OPS | run0 | run1 | run2 |
|---|---|---|---|
| 2 | 1 slow, first 0.982 | 100 slow, first 1.006 | 100 slow, first 1.023 |
| 102 | 1 slow, first 1.003 | 1 slow, first 0.992 | 1 slow, first 0.989 |
| 502 | 2 slow, first 1.004 | 100 slow, first 0.997 | 99 slow, first 1.001 |
| 1002 | 1 slow, first 1.004 | 1 slow, first 0.999 | 0 slow, first 1.001 |
| 2002 | 1 slow, first 1.001 | 0 slow, first 0.995 | 0 slow, first 0.993 |
| 3002 | 3 slow, first 1.056 | 0 slow, first 1.005 | 0 slow, first 1.000 |
| 4002 | 0 slow, first 1.029 | 0 slow, first 1.001 | 0 slow, first 1.012 |
| 5002 | 0 slow, first 1.013 | 0 slow, first 1.036 | 0 slow, first 1.050 |
| 6002 | 0 slow, first 1.035 | 1 slow, first 1.015 | 94 slow, first 1.242 |
| 7002 | 0 slow, first 1.009 | 0 slow, first 1.025 | 0 slow, first 1.000 |
| 8002 | 0 slow, first 1.012 | 0 slow, first 1.015 | 0 slow, first 0.999 |
| 9002 | 0 slow, first 1.034 | 0 slow, first 1.000 | 0 slow, first 1.017 |
| 10002 | 0 slow, first 1.002 | 0 slow, first 1.014 | 0 slow, first 1.036 |
| 12002 | 0 slow, first 1.002 | 0 slow, first 1.000 | 0 slow, first 1.028 |
| 14002 | 0 slow, first 1.000 | 0 slow, first 1.034 | 7 slow, first 1.031 |
| 15002 (extra) | 58 slow, first 1.007 | 0 slow, first 1.038 | 50 slow, first 1.089 |
| 16002 | 1 slow, first 1.150 | 9 slow, first 0.999 | 0 slow, first 0.998 |
| 17002 (extra) | 93 slow, first 1.000 | 82 slow, first 0.884 | 0 slow, first 1.001 |
| 18002 | 1 slow, first 1.144 | 82 slow, first 1.127 | 80 slow, first 1.134 |
| 19902 | 1 slow, first 1.129 | 0 slow, first 1.003 | 89 slow, first 1.017 |

**The effect on the answer.** The slow blocks lower their points by up to 15 %, but they do not set the maximum: the maximum comes from points where the runs agree. One estimator detail matters at the top points:
- There the baseline has only 10 to 26 repetitions, so the block-paired mean pairs them with only the first 10 to 26 fused repetitions.
- A slow first repetition therefore weighs 4 to 10 %.
- For example, run0 at 19902 gives 4,231 by the block-paired mean against 4,290 by the ratio of means.

## The order

The order of the sides was alternated between the runs: fused first in run0 and run2, baseline first in run1.

| quantity | run1 over the mean of run0 and run2: median over the points | range |
|---|---|---|
| block-paired mean of ratios | -0.1 % | -5.1 .. +9.4 % |
| fused mean | +0.1 % | -7.6 .. +4.8 % |
| baseline mean | +0.0 % | -0.7 .. +0.3 % |

The large values all fall at the points with slow fused blocks (2, 502, 6002, 15002, 17002, 18002, 19902). Beyond that state, there is no order effect.

Per point, run1 over the mean of run0 and run2:

| NUM_OPS | block-paired mean of ratios | fused mean | baseline mean |
|---|---|---|---|
| 2 | -3.0 % | +3.3 % | +0.1 % |
| 102 | +0.8 % | -0.7 % | +0.0 % |
| 502 | -2.8 % | +2.9 % | +0.0 % |
| 1002 | -1.0 % | +1.0 % | -0.0 % |
| 2002 | -0.4 % | +0.4 % | -0.0 % |
| 3002 | -0.1 % | +0.0 % | -0.0 % |
| 4002 | +0.1 % | -0.2 % | -0.0 % |
| 5002 | +0.3 % | +0.0 % | +0.3 % |
| 6002 | +6.2 % | -6.3 % | -0.7 % |
| 7002 | -0.1 % | -0.0 % | +0.0 % |
| 8002 | -0.0 % | +0.2 % | +0.0 % |
| 9002 | +0.1 % | +0.1 % | +0.0 % |
| 10002 | -0.2 % | +0.2 % | +0.0 % |
| 12002 | -0.5 % | +0.3 % | +0.0 % |
| 14002 | -0.3 % | -0.5 % | +0.0 % |
| 15002 (extra) | +9.4 % | -7.6 % | +0.0 % |
| 16002 | +0.5 % | +1.3 % | +0.0 % |
| 17002 (extra) | -5.1 % | +4.7 % | -0.0 % |
| 18002 | -1.9 % | +4.8 % | -0.0 % |
| 19902 | +5.5 % | -4.7 % | -0.0 % |

## The checks

Tolerance 1e-4 absolute, once per point before the timing, with both sides' outputs filled with NaN before the checked repetition:

| comparison | maximum of max_abs_diff over the runs and points | required rows passed |
|---|---|---|
| fused vs the double-precision reference | 8.94e-07 | 60 of 60 |
| fused vs baseline | 1.91e-06 | 60 of 60 |
| baseline vs the reference | 1.01e-06 | informational |

From 102 operations on, fused and baseline are bit-equal at the chain's fixed point, 1.1739131.

**The chain's length is checked in the device code instead.** The chain is within the tolerance of its fixed point after about 42 operations, so these rows cannot see its length beyond NUM_OPS = 2. Each of the 18 fused kernels has NUM_OPS / 2 FMA operations, 1 at 2 up to 9951 at 19902: `step0/devcode-run-build.txt`, which reads all 19 offload bundles of the binary. The two of the extra points have 7501 and 8501 (`runs-extra/step0/devcode-extra-build.txt`).

On sm_89, the compile-only review of the CUDA build on #39 found the same (https://github.com/Libraries-Openly-Fused/fkl-benchmarks/pull/39#pullrequestreview-5818366660), read with cuobjdump and never run:
- at e4b28fe, all 18 fused kernels have NUM_OPS / 2 FFMA (1, 51, ..., 9951) in one straight-line chain, with the same mangled names and identical normalised SASS (18 of 18) as the kernels of the 2025 `benchmark_vertical_fusionCompareMulAdd_cu` at the same NUM_OPS;
- at cd5f8d1, 22 kernels were identical to e4b28fe's, and the 2 new setTo kernels had no floating-point instruction.

The merged code changed only strings after cd5f8d1.

## The clocks, the links and the warm-ups

Every timing row of the runs, the extra ones included:

- **fused:** sclk 3217 .. 3456 MHz at the start of the timed repetitions and after them. mclk at the start of the timed repetitions: 96 MHz in 47 of the 60 rows, 456 in 2, 772 in 5, 875 in 6. Warm-up 2.00 s, steady in all 60.
- **baseline:** sclk 3392 .. 3471 MHz. mclk 96 MHz in 56 of the 60 rows, 456 in 1, 772 in 2, 875 in 1. Warm-up 2.00 .. 5.67 s, steady in all 60.
- **The links:** host 16 GT/s x16 and card 32 GT/s x16 at all 360 readings.

**The memory clock.** It sat at its idle level, 96 MHz, at most readings from the check run's 102-operation baseline on.
- A plausible reason, not established: the working set (a few MB: the inputs, the fused tensor and the baseline's float images) stays in the GPU's last-level cache, so the DRAM is almost idle.
- The fused times grow smoothly with NUM_OPS, with no step where the memory clock changed.
- Nothing was changed for it: the runs measure the GPU as its power management runs it.

## Files

- **`runs/run0`, `run1`, `run2`:** the program's four CSVs (`benchmark_vfhf_max_speedup_*_GPU_AMD.csv`: timing, speedup, reps, agreement), its stdout, and the machine facts before and after the run. `runs/runs.log` has the times and the idle waits.
- **`runs-extra/run0`, `run1`, `run2`, `runs-extra/runs.log`:** the runs of the extra points 15002 and 17002, the same files.
- **`runs-extra/step0/`:**
  - the build script with its one difference, `FKLB_VFHF_POINTS`, and its per-TU compile times;
  - the provenance (`step0-extra-build.txt`), the CMake caches compared (`cmake-cache-compare.txt`), and the device code against the run build (`devcode-extra-build.txt`);
  - the machine facts and the run script.
- **`check-run/`:** the check run up to 1002 operations, the same files.
- **`comparison.csv`:** a row per run and point (the column `set` says `runs` or `runs-extra`): the three speedups, the side times, the slow fused repetitions, the first repetition over the median, and the clocks.
- **`step0/`:**
  - the build script (`vfhf_run_build.sh`) and its per-TU compile times (`run-build.tu.log`: seconds, peak resident KB, exit code);
  - the provenance (`step0-run-build.txt`, `versions.txt`);
  - the device code of all 19 bundles against the reviewed PR's build (`devcode-run-build.txt`, `vfhf_devcode.py`: 26 of 26 functions identical, the FMA counts);
  - the machine facts (`machine-step0.txt`, `vfhf_machine.sh`);
  - the run script (`vfhf_runs.sh`).
- **`scripts/vfhf_compare.py`:** the tables of this README and `comparison.csv`.

## How to reproduce

1. **Build:** `step0/vfhf_run_build.sh <source> <build dir>`, both absolute paths. It uses the RX 9060 flags and the repository's points.
2. **Run:** `step0/vfhf_runs.sh <out> runs`, the three runs, or `check` for the short run. It assumes the scratch paths of this machine; change `NS` and `BIN` first.
3. **The extra points:** `runs-extra/step0/vfhf_run_build_extra.sh <source> <build dir>`, then `runs-extra/step0/vfhf_runs_extra.sh <out>`.
4. **Tables:** `python3 -I scripts/vfhf_compare.py results/rx9060/vfhf-max-speedup`, with the results folder itself as the argument, not the repository root.
