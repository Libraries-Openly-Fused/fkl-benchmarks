# FKL (HIP) against AMD RPP, pipeline (a), with RPP's documented batched form: RX 9060 (gfx1200), ROCm 10.1, RPP 3.2.0, FKL 64ad9195

These are the runs of 2026-11-08, after the rewrite of `rpp/benchmark_rpp_preprocessing.h` (fkl-benchmarks #32, merged as 9461389). They replace
nothing yet. The published runs behind Figure 15 are in `../rpp-preprocessing/` (fkl-benchmarks 534fca7, FKL c8fcbe8).

## What changed in the program (#32)

The program follows the decisions of 2026-11-06 and 2026-11-07 (fkl-benchmarks #32).

- **RPP** uses its documented batched form. The N crops of the frame are copied into one dense batch, one `hipMemcpy2DAsync` per crop, and
  then one call each of `rppt_resize_mirror_normalize` and `rppt_channel_permute` processes that batch. The batch has an `RpptDesc` with a
  real batch stride and one ROI per crop, covering the whole crop. RPP owns its frame (dense `hipMalloc`, filled from the same host bytes as
  FKL's) and its own `hipStream_t`. The former form, the whole frame with a batch stride of 0 and one ROI per crop, is not used any more,
  because RPP does not document a batch stride of 0. Its output is bitwise the same (#32, one-off check (a)).
- **FKL** uses the form of E4: a BatchRead of `crop2D(fk::Rect(i, i, 60, 120))` views `.then(Resize)`, then VectorReorder, Mul, Sub, Div and
  TensorSplit. The crop views and every operation are built inside every timed call, because the location of the crops may change on every
  image. The FKL kernels are the same as in the old form: the device code is identical, 97 of 97 functions.
- **Variants.**
  - `fkl_fused`, `fkl_fused_halfpixel` and `fkl_perop_batch`, as before.
  - `rpp_crop_batch` (was `rpp_fused_api`): the copies plus a stream synchronisation (part "build"), then the ROIs, the two RPP calls and a
    stream synchronisation (part "compute").
  - `rpp_crop_batch_unsplit`: the same calls without the synchronisation between the copies and the RPP calls, which is the time a user pays.
    The split total therefore includes one stream synchronisation more.
  - FKL's parts: build is the crop views and the operations, on the host; compute is `executeOperations` and the synchronisation. There is no
    synchronisation between the two.
  - The `calls` column: 1, 1, 6, N + 2 and N + 2.

## The runs

Seven runs on 2026-11-08 from 09:34 to 10:11 UTC, in this order:
- `reference-run0`, `new-run0`, `new-run1`, `reference-run1`, `reference-run2`, `new-run2`, with `E5_ROTATE_OFFSET` 0, 1, 2 for the three
  runs of each program;
- then `natural-image`, the new program on the Kodak image kodim23, as in the published run.

Each run started after the runner was idle: no `Runner.Worker` and a 1-minute load average below 0.8. The waits are in `runs.log`. Nothing
else ran on the GPU. The defaults were used: 3 warm-up runs, then at least 2 s of warm-up and until the mean time of two consecutive windows
of 0.25 s differs by at most 2 percent (cap 10 s), then 30 timed runs.

- **new:** fkl-benchmarks 94613897beb8003a0e024236a1eb527bb903a4d6 (main after #32);
  `benchmark_rpp_preprocessing_hip` sha256 fe58e11e983f3073af07015f86c82db5d6111065f1a9028ebd77a83d0861384c.
- **reference:** the program before the rewrite, fkl-benchmarks 4acf63c36b82aae23135854dca4ca26398ba79cc;
  sha256 4e7ca971a72f11309c549b89d1271d58b091e43c53883d9c3946a425c07e7ffa.
- **Both builds:** made from clean worktrees with `external/fkl` checked out at 64ad9195cd785d0c93a70b35e02d5a6c7c4a5048, using the same
  `step0/rpp_run_build.sh`. Their CMake caches (380 entries) and their compile and link commands are equal apart from the paths
  (`step0/`). The CSV headers carry each commit and no `-dirty`.
- **cvGPUSpeedup and FastNPP:** the headers say "not checked out (pinned ...)", unlike the published 534fca7 runs, which had them checked out.
  This program uses neither.
- **Software:** RPP 3.2.0 (`librpp.so.3.2.0` sha256 c6d9bdc7800e105122632f01f408a709b38bea4775677040beda4d7e5614844d), ROCm 10.1.0, amdclang++
  24.0.0. The RX 9060 is the only AMD device of the machine, so the `sclk_*` columns are its own.

`../../../scripts/rpp_crop_batch_compare.py` computes the tables below, and writes `comparison.csv` with every batch.
- **Method.** Every value is the median of the three runs' medians; the spread is max - min of the three.
- **Exceeds the spread:** |a - b| > the larger of the two spreads.
- **Flag:** a change beyond 3 % for a GPU-bound variant, or 5 % for a host-bound one. A variant is host-bound when its build part is at least
  half of the run.

## New against reference: the effect of the rewrite

| what | batch | new ms (spread) | reference ms (spread) | change | exceeds spread | class | flag |
|---|---|---|---|---|---|---|---|
| fkl_fused | 10 | 0.0338 (0.0034) | 0.0329 (0.0002) | +2.7 % | no | GPU | no |
| fkl_fused | 50 | 0.0520 (0.0006) | 0.0493 (0.0009) | +5.4 % | yes | GPU | **yes** |
| fkl_fused | 100 | 0.0701 (0.0056) | 0.0661 (0.0005) | +6.1 % | no | GPU | **yes** |
| fkl_fused | 150 | 0.0895 (0.0010) | 0.0767 (0.0058) | +16.7 % | yes | GPU | **yes** |
| fkl_fused | 200 | 0.1072 (0.0016) | 0.0987 (0.0007) | +8.6 % | yes | GPU | **yes** |
| fkl_fused | 250 | 0.1260 (0.0008) | 0.1156 (0.0085) | +9.0 % | yes | GPU | **yes** |
| fkl_fused | 300 | 0.1449 (0.0014) | 0.1333 (0.0010) | +8.7 % | yes | GPU | **yes** |
| fkl_fused_halfpixel | 10 | 0.0347 (0.0001) | 0.0330 (0.0024) | +5.1 % | no | GPU | **yes** |
| fkl_fused_halfpixel | 50 | 0.0524 (0.0002) | 0.0501 (0.0007) | +4.5 % | yes | GPU | **yes** |
| fkl_fused_halfpixel | 100 | 0.0714 (0.0015) | 0.0674 (0.0007) | +5.8 % | yes | GPU | **yes** |
| fkl_fused_halfpixel | 150 | 0.0911 (0.0005) | 0.0840 (0.0012) | +8.5 % | yes | GPU | **yes** |
| fkl_fused_halfpixel | 200 | 0.1097 (0.0008) | 0.1014 (0.0004) | +8.2 % | yes | GPU | **yes** |
| fkl_fused_halfpixel | 250 | 0.1291 (0.0006) | 0.1185 (0.0083) | +9.0 % | yes | GPU | **yes** |
| fkl_fused_halfpixel | 300 | 0.1494 (0.0020) | 0.1371 (0.0007) | +9.0 % | yes | GPU | **yes** |
| fkl_perop_batch | 10 | 0.0571 (0.0009) | 0.0566 (0.0002) | +1.0 % | no | GPU | no |
| fkl_perop_batch | 50 | 0.1248 (0.0002) | 0.1230 (0.0056) | +1.5 % | no | GPU | no |
| fkl_perop_batch | 100 | 0.2048 (0.0082) | 0.2016 (0.0056) | +1.6 % | no | GPU | no |
| fkl_perop_batch | 150 | 0.3092 (0.0025) | 0.3026 (0.0008) | +2.2 % | yes | GPU | no |
| fkl_perop_batch | 200 | 0.5672 (0.0052) | 0.5582 (0.0038) | +1.6 % | yes | GPU | no |
| fkl_perop_batch | 250 | 0.7649 (0.0135) | 0.7588 (0.0147) | +0.8 % | no | GPU | no |
| fkl_perop_batch | 300 | 1.1333 (0.0069) | 1.1234 (0.0113) | +0.9 % | no | GPU | no |
| rpp_crop_batch_unsplit vs rpp_fused_api | 10 | 0.1032 (0.0044) | 0.0713 (0.0004) | +44.7 % | yes | GPU | **yes** |
| rpp_crop_batch_unsplit vs rpp_fused_api | 50 | 0.2761 (0.0011) | 0.1275 (0.0002) | +116.6 % | yes | GPU | **yes** |
| rpp_crop_batch_unsplit vs rpp_fused_api | 100 | 0.4893 (0.0172) | 0.1913 (0.0011) | +155.8 % | yes | GPU | **yes** |
| rpp_crop_batch_unsplit vs rpp_fused_api | 150 | 0.7284 (0.0159) | 0.2597 (0.0006) | +180.4 % | yes | GPU | **yes** |
| rpp_crop_batch_unsplit vs rpp_fused_api | 200 | 0.9712 (0.0023) | 0.3217 (0.0016) | +201.9 % | yes | GPU | **yes** |
| rpp_crop_batch_unsplit vs rpp_fused_api | 250 | 1.2210 (0.0008) | 0.3876 (0.0015) | +215.0 % | yes | GPU | **yes** |
| rpp_crop_batch_unsplit vs rpp_fused_api | 300 | 1.4430 (0.0027) | 0.4662 (0.0083) | +209.5 % | yes | GPU | **yes** |
| rpp_crop_batch vs rpp_fused_api | 10 | 0.1236 (0.0005) | 0.0713 (0.0004) | +73.4 % | yes | GPU | **yes** |
| rpp_crop_batch vs rpp_fused_api | 50 | 0.2986 (0.0004) | 0.1275 (0.0002) | +134.2 % | yes | host | **yes** |
| rpp_crop_batch vs rpp_fused_api | 100 | 0.5122 (0.0011) | 0.1913 (0.0011) | +167.8 % | yes | host | **yes** |
| rpp_crop_batch vs rpp_fused_api | 150 | 0.7571 (0.0069) | 0.2597 (0.0006) | +191.5 % | yes | host | **yes** |
| rpp_crop_batch vs rpp_fused_api | 200 | 0.9627 (0.0368) | 0.3217 (0.0016) | +199.2 % | yes | host | **yes** |
| rpp_crop_batch vs rpp_fused_api | 250 | 1.2481 (0.0415) | 0.3876 (0.0015) | +222.0 % | yes | host | **yes** |
| rpp_crop_batch vs rpp_fused_api | 300 | 1.4736 (0.0029) | 0.4662 (0.0083) | +216.1 % | yes | host | **yes** |
| rpp_crop_batch compute vs rpp_fused_api | 10 | 0.0719 (0.0001) | 0.0713 (0.0004) | +0.8 % | yes | GPU | no |
| rpp_crop_batch compute vs rpp_fused_api | 50 | 0.1255 (0.0002) | 0.1275 (0.0002) | -1.5 % | yes | GPU | no |
| rpp_crop_batch compute vs rpp_fused_api | 100 | 0.1888 (0.0009) | 0.1913 (0.0011) | -1.3 % | yes | GPU | no |
| rpp_crop_batch compute vs rpp_fused_api | 150 | 0.2580 (0.0021) | 0.2597 (0.0006) | -0.7 % | no | GPU | no |
| rpp_crop_batch compute vs rpp_fused_api | 200 | 0.3134 (0.0084) | 0.3217 (0.0016) | -2.6 % | no | GPU | no |
| rpp_crop_batch compute vs rpp_fused_api | 250 | 0.4081 (0.0076) | 0.3876 (0.0015) | +5.3 % | yes | GPU | **yes** |
| rpp_crop_batch compute vs rpp_fused_api | 300 | 0.5122 (0.0030) | 0.4662 (0.0083) | +9.9 % | yes | GPU | **yes** |

Over the 30 batches:

| what | change, min .. median .. max | batches beyond the spread | flagged |
|---|---|---|---|
| fkl_fused | +2.7 .. +8.2 .. +18.3 % | 25 of 30 | 29 of 30 |
| fkl_fused_halfpixel | +4.5 .. +8.0 .. +9.0 % | 25 of 30 | 30 of 30 |
| fkl_perop_batch | +0.8 .. +1.8 .. +3.1 % | 19 of 30 | 1 of 30 |
| rpp_crop_batch_unsplit vs rpp_fused_api | +44.7 .. +183.0 .. +215.0 % | 30 of 30 | 30 of 30 |
| rpp_crop_batch vs rpp_fused_api | +73.4 .. +193.4 .. +222.1 % | 30 of 30 | 30 of 30 |
| rpp_crop_batch compute vs rpp_fused_api | -4.2 .. -0.9 .. +9.9 % | 19 of 30 | 9 of 30 |

**Causes.**
- **fkl_fused and fkl_fused_halfpixel, +8 % (median).** The kernels are the same (device code 97 of 97). The difference is the batch building,
  now inside the timed call. At N = 300 the new run is 11.6 us slower and its build part is 11.1 us; at N = 10 the figures are 0.9 us and
  0.3 us. What is left is the release of the crop views at the end of the call, which is counted in compute, plus noise. Batch 150 is the
  exception: there the reference's own spread is 5.8 us.
- **fkl_perop_batch, +0.8 to +3.1 %.** The same batch building, against six kernels.
- **rpp_crop_batch_unsplit against rpp_fused_api, +45 % at N = 10 to +215 %.** The N copies into the dense batch: N `hipMemcpy2DAsync` of
  21.6 KB, mostly host API time. They are what RPP's documented batched form needs, and they were not done before.
- **rpp_crop_batch compute against rpp_fused_api, -4.2 .. +9.9 % (median -0.9 %).** These are the same two RPP calls with the same output,
  bitwise, on a dense batch instead of ROIs of the frame. The changes beyond 3 % are at the largest batches (N = 250 to 300: +5 to +10 %).

## Reference against published: everything else since the published runs

| what | batch | reference ms (spread) | published ms (spread) | change | exceeds spread | class | flag |
|---|---|---|---|---|---|---|---|
| fkl_fused | 10 | 0.0329 (0.0002) | 0.0310 (0.0006) | +6.1 % | yes | GPU | **yes** |
| fkl_fused | 50 | 0.0493 (0.0009) | 0.0454 (0.0007) | +8.8 % | yes | GPU | **yes** |
| fkl_fused | 100 | 0.0661 (0.0005) | 0.0608 (0.0013) | +8.6 % | yes | GPU | **yes** |
| fkl_fused | 150 | 0.0767 (0.0058) | 0.0752 (0.0007) | +2.1 % | no | GPU | no |
| fkl_fused | 200 | 0.0987 (0.0007) | 0.0910 (0.0007) | +8.5 % | yes | GPU | **yes** |
| fkl_fused | 250 | 0.1156 (0.0085) | 0.1078 (0.0004) | +7.2 % | no | GPU | **yes** |
| fkl_fused | 300 | 0.1333 (0.0010) | 0.1236 (0.0003) | +7.9 % | yes | GPU | **yes** |
| fkl_fused_halfpixel | 10 | 0.0330 (0.0024) | 0.0308 (0.0006) | +7.3 % | no | GPU | **yes** |
| fkl_fused_halfpixel | 50 | 0.0501 (0.0007) | 0.0461 (0.0044) | +8.6 % | no | GPU | **yes** |
| fkl_fused_halfpixel | 100 | 0.0674 (0.0007) | 0.0621 (0.0011) | +8.7 % | yes | GPU | **yes** |
| fkl_fused_halfpixel | 150 | 0.0840 (0.0012) | 0.0773 (0.0007) | +8.6 % | yes | GPU | **yes** |
| fkl_fused_halfpixel | 200 | 0.1014 (0.0004) | 0.0935 (0.0002) | +8.5 % | yes | GPU | **yes** |
| fkl_fused_halfpixel | 250 | 0.1185 (0.0083) | 0.1102 (0.0002) | +7.5 % | no | GPU | **yes** |
| fkl_fused_halfpixel | 300 | 0.1371 (0.0007) | 0.1276 (0.0004) | +7.5 % | yes | GPU | **yes** |
| fkl_perop_batch | 10 | 0.0566 (0.0002) | 0.0526 (0.0002) | +7.5 % | yes | GPU | **yes** |
| fkl_perop_batch | 50 | 0.1230 (0.0056) | 0.1188 (0.0059) | +3.5 % | no | GPU | **yes** |
| fkl_perop_batch | 100 | 0.2016 (0.0056) | 0.1950 (0.0019) | +3.4 % | yes | GPU | **yes** |
| fkl_perop_batch | 150 | 0.3026 (0.0008) | 0.2955 (0.0012) | +2.4 % | yes | GPU | no |
| fkl_perop_batch | 200 | 0.5582 (0.0038) | 0.5514 (0.0008) | +1.2 % | yes | GPU | no |
| fkl_perop_batch | 250 | 0.7588 (0.0147) | 0.7473 (0.0030) | +1.5 % | no | GPU | no |
| fkl_perop_batch | 300 | 1.1234 (0.0113) | 1.1083 (0.0060) | +1.4 % | yes | GPU | no |
| rpp_fused_api | 10 | 0.0713 (0.0004) | 0.0712 (0.0002) | +0.2 % | no | GPU | no |
| rpp_fused_api | 50 | 0.1275 (0.0002) | 0.1276 (0.0005) | -0.1 % | no | GPU | no |
| rpp_fused_api | 100 | 0.1913 (0.0011) | 0.1907 (0.0015) | +0.3 % | no | GPU | no |
| rpp_fused_api | 150 | 0.2597 (0.0006) | 0.2520 (0.0018) | +3.1 % | yes | GPU | **yes** |
| rpp_fused_api | 200 | 0.3217 (0.0016) | 0.3140 (0.0009) | +2.5 % | yes | GPU | no |
| rpp_fused_api | 250 | 0.3876 (0.0015) | 0.3810 (0.0016) | +1.8 % | yes | GPU | no |
| rpp_fused_api | 300 | 0.4662 (0.0083) | 0.4649 (0.0004) | +0.3 % | no | GPU | no |

Over the 30 batches:

| what | change, min .. median .. max | batches beyond the spread | flagged |
|---|---|---|---|
| fkl_fused | +0.4 .. +8.1 .. +9.6 % | 24 of 30 | 28 of 30 |
| fkl_fused_halfpixel | +3.9 .. +8.1 .. +9.1 % | 23 of 30 | 30 of 30 |
| fkl_perop_batch | +0.0 .. +2.4 .. +7.5 % | 23 of 30 | 13 of 30 |
| rpp_fused_api | -0.7 .. +0.3 .. +4.1 % | 13 of 30 | 3 of 30 |

**Cause: not identified, and an open item.**
- `fkl_fused` and `fkl_fused_halfpixel` of the reference are about 8 % slower than published, and `rpp_fused_api` agrees within 0.3 % (median).
- The device code of pipeline (a) is the same in the published configuration (534fca7 against FKL c8fcbe8, rebuilt here, compile only) and in
  the reference: 97 of 97 functions identical. The 16 more functions of 534fca7 are the NV12 pipeline (b), which the program no longer has.
- The graphics clock during the timed runs is higher today, not lower: the median `sclk_timed_start_mhz` of `fkl_fused` is 3112 MHz, against
  3056 MHz published.
- The difference grows with N (+1.9 us at N = 10, +9.7 us at N = 300), so it is not a fixed cost per launch.
- To separate the environment from the host code: run the rebuilt published binary alternately with the reference in one window; this has not
  been done.

## The parts of the new variants

Medians over the three runs of the per-run medians, in ms. FKL's build part is the cost of building the batch in the timed call.

| batch | fkl_fused total | fkl_fused build | build share | fkl_perop_batch build | rpp_crop_batch total | rpp_crop_batch build | rpp_crop_batch compute | rpp_crop_batch_unsplit |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.0338 | 0.0003 | 0.9 % | 0.0003 | 0.1236 | 0.0517 | 0.0719 | 0.1032 |
| 50 | 0.0520 | 0.0018 | 3.4 % | 0.0019 | 0.2986 | 0.1729 | 0.1255 | 0.2761 |
| 100 | 0.0701 | 0.0036 | 5.1 % | 0.0041 | 0.5122 | 0.3231 | 0.1888 | 0.4893 |
| 150 | 0.0895 | 0.0057 | 6.4 % | 0.0064 | 0.7571 | 0.4979 | 0.2580 | 0.7284 |
| 200 | 0.1072 | 0.0074 | 6.9 % | 0.0086 | 0.9627 | 0.6494 | 0.3134 | 0.9712 |
| 250 | 0.1260 | 0.0095 | 7.6 % | 0.0107 | 1.2481 | 0.8404 | 0.4081 | 1.2210 |
| 300 | 0.1449 | 0.0111 | 7.7 % | 0.0125 | 1.4736 | 0.9619 | 0.5122 | 1.4430 |

## The ratio of the paper: RPP / fkl_fused, per batch

The ratio is taken per run, with `rpp_crop_batch_unsplit` (new) or `rpp_fused_api` (reference, published) over `fkl_fused`. The table gives
the median of the three runs, with their min .. max in brackets.

| batch | new: rpp_crop_batch_unsplit / fkl_fused | reference: rpp_fused_api / fkl_fused | published: rpp_fused_api / fkl_fused |
|---|---|---|---|
| 10 | 3.17 (2.97 .. 3.31) | 2.16 (2.15 .. 2.17) | 2.30 (2.29 .. 2.33) |
| 50 | 5.32 (5.31 .. 5.36) | 2.58 (2.58 .. 2.63) | 2.81 (2.79 .. 2.84) |
| 100 | 6.98 (6.97 .. 7.84) | 2.88 (2.88 .. 2.90) | 3.13 (3.11 .. 3.16) |
| 150 | 8.16 (7.96 .. 8.21) | 3.39 (3.17 .. 3.42) | 3.34 (3.33 .. 3.36) |
| 200 | 9.06 (8.98 .. 9.10) | 3.26 (3.25 .. 3.26) | 3.45 (3.45 .. 3.48) |
| 250 | 9.69 (9.66 .. 9.73) | 3.35 (3.35 .. 3.61) | 3.54 (3.53 .. 3.55) |
| 300 | 9.96 (9.91 .. 10.02) | 3.50 (3.41 .. 3.50) | 3.76 (3.75 .. 3.77) |

Range over the 30 batches (of the median ratio): new 3.17 .. 9.96; reference 2.16 .. 3.51; published 2.30 .. 3.76

## Agreement

All required rows pass in all seven runs (tolerance 1e-4).

| comparison (region) | new runs (noise) | natural image | reference runs |
|---|---|---|---|
| fkl_fused vs scalar reference ASYMMETRIC (both regions) | 7.63e-6, required, 30 of 30 per run | 7.63e-6 | 7.63e-6 |
| fkl_fused_halfpixel vs scalar reference HALF_PIXEL (both regions) | 7.63e-6, required | 7.63e-6 | 7.63e-6 |
| fkl_perop_batch vs fkl_fused (both regions) | 7.63e-6, required | 7.63e-6 | 7.63e-6 |
| RPP vs scalar reference HALF_PIXEL, interior | 1.91e-6, required | 1.91e-6 | 1.91e-6 |
| RPP vs scalar reference HALF_PIXEL, last 2 columns and rows | 126, informational (RPP clamps the source coordinate to size - 2) | 73.8 | 126 |
| RPP vs fkl_fused_halfpixel, interior | 7.63e-6, informational | 7.63e-6 | 7.63e-6 |
| rpp_crop_batch_unsplit vs rpp_crop_batch (both regions) | 0, required | 0 | (no such variant) |

7.63e-6 is 1 ulp at about 120. The per-batch rows are in each run's `benchmark_rpp_preprocessing_a_agreement_GPU_AMD.csv`.

## Clock and warm-up columns

| source | sclk at the start of the timed runs, MHz | sclk after them, MHz | warm-up, s | rows with warmup_steady = 0 |
|---|---|---|---|---|
| published | 2600 .. 3351 | 2836 .. 3374 | 2.00 .. 2.25 | 0 of 360 |
| reference | 2745 .. 3361 | 3000 .. 3400 | 2.00 .. 2.25 | 0 of 360 |
| new | 2821 .. 3414 | 3047 .. 3418 | 2.00 .. 2.25 | 0 of 450 |

## Not measured

- One `hipMemcpy3DBatchAsync` (HIP, documented) instead of the N `hipMemcpy2DAsync` copies. The decision was one copy per crop.
- One RPP call per crop. It is not the batched API.

## Files

- `noise/new-run0..2/`, `noise/reference-run0..2/`, `natural-image/`: the timing CSV, the agreement CSV and `stdout.log` of each run.
- `runs.log`: the order, the idle-runner waits and the start and end of each run, with the sha256 of the binary.
- `step0/`:
  - the facts of the two binaries (`step0-new.txt`, `step0-reference.txt`), the versions (`versions.txt`) and the commands of the RPP target
    (`commands-new.txt`, equal for both apart from the paths);
  - the build script (`rpp_run_build.sh`) and the run script (`rpp_runs.sh`).
- `comparison.csv`: every batch of the comparisons and of the ratios, written by the script.
