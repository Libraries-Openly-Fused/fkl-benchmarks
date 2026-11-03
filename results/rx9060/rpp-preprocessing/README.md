# FKL (HIP) against AMD RPP on the pre-processing pipeline (a): RX 9060 (gfx1200), ROCm 10.1, RPP 3.2.0, FKL c8fcbe8

`benchmark_rpp_preprocessing_hip` (`rpp/benchmark_rpp_preprocessing.h`, build with `-DFKLB_BUILD_RPP=ON`) at fkl-benchmarks `534fca7` and FKL `c8fcbe8` (the final pin of
LTS-C++17). RPP is the ROCm package `amdrocm-rpp10.1` 10.1.0-3 (`librpp.so.3.2.0`), the `rppt_*` tensor API with the HIP back-end. Wall clock of one run including the
synchronisation of the stream; 3 warm-up runs, then a warm-up of at least 2 s and until the mean time of two consecutive windows of 0.25 s differs by at most 2 percent
(the GPU clock of the RX 9060 needs about 1 s of continuous work to reach its working value: see `../README.md`); 30 timed runs per variant, median/min/max/mean;
**three repetitions of the whole run with the variants in a rotated order** (`E5_ROTATE_OFFSET` = 0, 1, 2; the first variant of a point is variant
(index of the point + offset) mod the number of variants); the table gives the median of the three medians. Idle machine (no CI job on the runner), governor `powersave`.
These are the numbers to cite; the first run of this folder (warm-up 0.5 s, pin 38c4f32) is in the history of the branch.

- `noise/run0`, `run1`, `run2`: seeded noise as the image, the three repetitions. `natural-image/`: the Kodak image kodim23
  tiled over the image, one run whose timing rows are 5 runs with a warm-up of 0.2 s and are not meant, only the agreement.
- Pipeline (a): `benchmark_rpp_preprocessing_a_*.csv`, batch 10 to 300 step 10. Columns `warmup_s`, `warmup_steady`,
  `sclk_start_mhz`, `sclk_timed_start_mhz`, `sclk_end_mhz`: the warm-up that was run and the graphics clock at the start of the warm-up, at the start of the timed runs
  and after them.
- Variants: `fkl_fused`, `fkl_perop_batch` (FKL, ASYMMETRIC sampling, the default), `fkl_fused_halfpixel` (FKL fused with `ResizeHalfPixel`, the sampling of RPP),
  `rpp_fused_api` (RPP: `rppt_resize_mirror_normalize` + `rppt_channel_permute`, two calls).
- Removed: the NV12 video pipeline (pipeline (b): the `..._b_*.csv` files of every run, 1, 8, 16, 32 and 64 crops, with the RPP conversion `rppt_yuv_to_rgb`) is not in the paper; the benchmark does not run it any more.
  Its code and results are in the git history up to `341ecd1` (for example `git show 341ecd1:results/rx9060/rpp-preprocessing/noise/run0/benchmark_rpp_preprocessing_b_GPU_AMD.csv`). The `stdout.log` files are the console output of the runs and still show its lines.
  The `# notes:` header line of the pipeline (a) CSVs written before the removal (in `rpp-preprocessing/` and `clock-ramp/`) still names `rppt_yuv_to_rgb`, the RPP conversion that only pipeline (b) called; the benchmark does not write it any more.

Changes of the harness after the review of #6 (the committed runs were made before them): the interior of `rpp_fused_api vs scalar reference HALF_PIXEL` of pipeline (a) is a required check (`pass` 1 or 0 in the CSV, the edge stays informational), so a wrong RPP output fails the run (exit status and `NUMERICAL CHECK FAILED`) instead of being timed and published; the committed CSVs have -1 there, and their interior values (1.9e-06 at all 30 batches) would pass the tolerance of 1e-4. The maximum of a comparison keeps a NaN (it used to be overwritten by the next element). `E5_ITERS`, `E5_WARMUP`, `E5_WARMUP_SECONDS` (0 to 10 s: a warm-up ends after 10 s whatever the value) and `E5_ROTATE_OFFSET` are checked (a value that is not a number, zero iterations, a negative number, a NaN or an infinity stops the program with exit status 2).

The pipeline (a) agreement files (`..._a_agreement_*.csv`): the first committed version had the `note` column unquoted: 240 rows (60 in each of the four files) have the note "FKL samples ASYMMETRIC, RPP HALF_PIXEL", which a reader that splits on commas sees as a ninth field; the files were repaired afterwards by putting those notes in double quotes, which is the only change (`agreement-csv-quoting.txt` has the sha256 of every file before and after).

RPP calls: the crops of one image are a batch with a source batch stride of 0 and one ROI (XYWH) per crop; source `U8 NHWC`, destination `F32 NCHW`;
`mean = sub / 0.3`, `std = div / 0.3` per channel, mirror off; the channel
permutation (2, 1, 0) for BGR. RPP has no separate per-channel multiply, subtract and divide: its normalize calls cover the three (the
`rppt_*_scalar` functions take one scalar per image), so there is no one-call-per-operation RPP chain; `rpp_fused_api` is RPP's own pre-fused path.
`rppt_resize_mirror_normalize` changes the array of ROIs in place (XYWH to LTRB): the benchmark writes it again before each call and that is part of the
timed work. One RPP handle per call. Output widths are multiples of 8.

## Pipeline (a), median of the three repetitions (ms)

| batch | FKL fused | FKL fused half-pixel | FKL per-op (batch) | RPP fused API | RPP / FKL fused | RPP / FKL half-pixel | largest spread of the 3 repetitions |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 10 | 0.0310 | 0.0308 | 0.0526 | 0.0712 | 2.29x | 2.31x | 2.0 % |
| 20 | 0.0350 | 0.0351 | 0.0701 | 0.0859 | 2.45x | 2.45x | 0.9 % |
| 30 | 0.0402 | 0.0404 | 0.0871 | 0.1021 | 2.54x | 2.52x | 1.5 % |
| 40 | 0.0429 | 0.0435 | 0.1031 | 0.1152 | 2.68x | 2.65x | 1.8 % |
| 50 | 0.0454 | 0.0461 | 0.1188 | 0.1276 | 2.81x | 2.76x | 9.5 % |
| 60 | 0.0488 | 0.0494 | 0.1352 | 0.1406 | 2.88x | 2.84x | 2.3 % |
| 70 | 0.0519 | 0.0526 | 0.1483 | 0.1532 | 2.95x | 2.91x | 2.3 % |
| 80 | 0.0546 | 0.0561 | 0.1636 | 0.1657 | 3.04x | 2.95x | 1.3 % |
| 90 | 0.0578 | 0.0589 | 0.1787 | 0.1785 | 3.09x | 3.03x | 1.4 % |
| 100 | 0.0608 | 0.0621 | 0.1950 | 0.1907 | 3.13x | 3.07x | 2.1 % |
| 110 | 0.0637 | 0.0650 | 0.2111 | 0.2020 | 3.17x | 3.11x | 1.6 % |
| 120 | 0.0675 | 0.0688 | 0.2372 | 0.2149 | 3.19x | 3.12x | 2.9 % |
| 130 | 0.0702 | 0.0715 | 0.2597 | 0.2291 | 3.27x | 3.21x | 1.1 % |
| 140 | 0.0732 | 0.0748 | 0.2768 | 0.2400 | 3.28x | 3.21x | 1.0 % |
| 150 | 0.0752 | 0.0773 | 0.2955 | 0.2520 | 3.35x | 3.26x | 1.0 % |
| 160 | 0.0780 | 0.0806 | 0.3086 | 0.2651 | 3.40x | 3.29x | 9.9 % |
| 170 | 0.0817 | 0.0834 | 0.3241 | 0.2788 | 3.41x | 3.34x | 1.1 % |
| 180 | 0.0851 | 0.0867 | 0.4704 | 0.2901 | 3.41x | 3.34x | 1.4 % |
| 190 | 0.0875 | 0.0899 | 0.5241 | 0.3023 | 3.45x | 3.36x | 1.0 % |
| 200 | 0.0910 | 0.0935 | 0.5514 | 0.3140 | 3.45x | 3.36x | 0.7 % |
| 210 | 0.0942 | 0.0967 | 0.5776 | 0.3262 | 3.46x | 3.37x | 0.9 % |
| 220 | 0.0975 | 0.1000 | 0.6037 | 0.3371 | 3.46x | 3.37x | 0.7 % |
| 230 | 0.1007 | 0.1034 | 0.6362 | 0.3503 | 3.48x | 3.39x | 1.0 % |
| 240 | 0.1038 | 0.1070 | 0.6796 | 0.3656 | 3.52x | 3.42x | 0.9 % |
| 250 | 0.1078 | 0.1102 | 0.7473 | 0.3810 | 3.53x | 3.46x | 0.4 % |
| 260 | 0.1103 | 0.1138 | 0.8292 | 0.3969 | 3.60x | 3.49x | 1.5 % |
| 270 | 0.1139 | 0.1169 | 0.8926 | 0.4131 | 3.63x | 3.53x | 2.2 % |
| 280 | 0.1169 | 0.1203 | 0.9717 | 0.4287 | 3.67x | 3.56x | 0.7 % |
| 290 | 0.1202 | 0.1236 | 1.0298 | 0.4455 | 3.71x | 3.61x | 1.5 % |
| 300 | 0.1236 | 0.1276 | 1.1083 | 0.4649 | 3.76x | 3.64x | 0.5 % |

## Numerics

Agreement files `..._agreement_*.csv` (interior and last two columns and rows of the output planes reported separately). RPP follows the HALF_PIXEL sampling of
`common/resize_reference.h` in the interior to 1.9e-6, but it clamps the source coordinate to `size - 2` instead of `size - 1`, so the last two columns and rows of
each plane differ from that reference (a 60 pixel wide ROI resized to 64: the outputs 62 and 63 read the coordinate 58.0, the reference 58.09 and 59.0). FKL's `Resize`
samples ASYMMETRIC by default, so FKL against RPP differs by the sampling convention. With `ResizeHalfPixel` FKL and RPP use the same sampling and agree in the interior to rounding; what remains is the last two columns and rows. Maximum / mean /
fraction over 1e-4 of the absolute difference, `rpp_fused_api` against `fkl_fused_halfpixel`, and against `fkl_fused` in the interior:

| image | batch 100 interior | batch 100 last 2 columns and rows | vs ASYMMETRIC, interior |
|---|---|---|---|
| noise | 7.6e-06 / 8.9e-07 / 0 | 1.2e+02 / 7.8 / 1 | 7 / 0.64 / 1 |
| Kodak image | 7.6e-06 / 6.4e-07 / 0 | 67 / 0.32 / 0.95 | 2.9 / 0.019 / 0.95 |

The HALF_PIXEL variant of FKL costs the same as the
default: the table shows the two within a few percent.
