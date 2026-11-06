# E5, pipeline (a) on the CPU at FKL c8fcbe8 (the pin until 2026-11-07) against OpenCV, Ryzen 7 5800X3D

The rows of `../e5-preprocessing-a/` (FKL `fb9ac3a`, the merge of #406: before #407, #409, #405, #411, #412, #414, #416 and #417) run again with FKL `c8fcbe8`, the pin of this repository until 2026-11-07; the old rows stay as they were. Same benchmark
(`opencv-cpu/preprocessing/bench_preproc_a.cpp`, target `e5_preprocessing_a_cpp`; the source and `e5_common.h` are identical at fkl-benchmarks `f796f9f`, which made the old rows, and at `1be0f5e`; only the lines of the CSV header differ),
same protocol (3 warm-up and 30 timed runs per point and variant, wall clock, median/min/max/mean; seeded noise image; batch 10 to 300 step 10; FKL `ParArch::CPU` with 1 thread and `CPU_MULTITHREAD` with 16 threads; OpenCV 4.14.0 with IPP, 1 thread and 16
threads), built at fkl-benchmarks `1be0f5e` (clean) with g++ 13.4.0 and clang++ 23.1.5, `-O3 -march=native`. The FKL commit and the fkl-benchmarks commit are in the header of every CSV. Governor `powersave` (amd-pstate-epp, EPP balance_performance, boost on), as for the old
rows; the memory speed is not recorded (it needs root). The agreement with the natural image (`../e5-preprocessing-a/natural-image-g++-13/`) was not repeated.

## What is here

- `g++-13/`, `clang++-23/`: sweep 1 of each compiler, raw output with the file names of the old rows (`e5_preprocessing_a_CPU.csv`, `e5_preprocessing_a_agreement_CPU.csv`, `stdout.log`); in the agreement file the 60 comparison names with a comma were quoted afterwards, the only change (`../agreement-csv-quoting.txt` has the sha256 of the files before and after).
- `repeats/<compiler>/run2`, `repeats/<compiler>/run3`: two more sweeps of the same binary, raw output. `repeats/<compiler>/check-nothing-else-running`: one more sweep, see Conditions; it is not part of the median. The agreement file of each of these folders has the same quoting repair (the 60 comparison names with a comma).
- `g++-13/e5_preprocessing_a_CPU_median_of_3.csv`, `clang++-23/e5_preprocessing_a_CPU_median_of_3.csv`: **the numbers to cite: the numbers of the paper are the median of the three sweeps.** For every point and variant, `median_ms` is the median of the medians of sweeps 1, 2 and 3, and
  `medians_min_ms` and `medians_max_ms` are the smallest and the largest of the three medians. The 16-thread rows vary between sweeps, so one sweep alone is not a number to cite (last table).
- `scripts/e5_cpu_compare_runs.py` (in the root of this repository) writes the median files and prints every number of this file from the CSVs (command at the end).

## Conditions

The three sweeps of a compiler were run interleaved with those of the other (g++ 1, clang++ 1, g++ 2, clang++ 2, g++ 3, clang++ 3) on 2026-11-01 between 15:01 and 15:23, each started only with no `Runner.Worker` (CI job) on the machine and a 1-minute load average below 0.8,
and no CI job appeared while any of them ran (checked every 2 s). Wall time of a sweep: 120 to 122 s with g++ (the old sweep took 120.6 s) and 127 to 128 s with clang++ (the old one 127.1 s). The session that ran them was working while sweeps 1 to 3 ran (reading files,
writing scripts, and one commit and push of another pull request at 15:13, during sweep 2 of clang++). One more sweep per compiler, `repeats/<compiler>/check-nothing-else-running` (g++ 15:27 to 15:29, clang++ 15:31 to 15:33), ran while the session issued no command; it differs from
each of the three sweeps by 0.8 to 1.1 % (median over all points; 3.2 to 5.0 % at the 90th percentile) and the three sweeps from each other by 0.7 to 1.1 % (3.0 to 4.1 %), see the last table.

## Do the new rows match the old ones?

They match within the noise of this machine. Per group of rows, 330 points per compiler (11 variants times 30 batch sizes), the new row is the median of the three sweeps and D is its change from the old row: the single-thread FKL rows differ by 0.2 % (median) and at most 1.7 %;
the 16-thread FKL rows by 0.7 to 1.1 % (median) and at most 9.7 %, and the old 16-thread rows are wide themselves (their own minimum to maximum is 16.5 to 17.3 % of the median, as a median over the points); the OpenCV rows, which the FKL pin cannot change, differ from their old rows by
1.4 to 1.8 % (median) and by up to 15.3 %. S is the spread of the three new sweeps at the point, (largest minus smallest median) / median. "Outside the window" counts the points where the old median is outside the range of the three new medians widened by that range on both sides
(largest minus smallest, added above and below); "and outside the old range" those that also have the new median outside the minimum to maximum of the 30 timed runs of the old row.

| compiler | rows | points | change \|D\|: median / 90th percentile / max | spread S of the three sweeps: median / max | old (max-min)/median: median | outside the window of the three sweeps | and outside the old range |
|---|---|--:|--|--|--:|--:|--:|
| g++-13 | FKL, 1 thread | 90 | 0.22 % / 0.67 % / 0.97 % | 0.57 % / 2.54 % | 2.2 % | 11 | 3 |
| g++-13 | FKL, 16 threads | 90 | 0.66 % / 3.88 % / 9.71 % | 1.70 % / 21.28 % | 16.5 % | 12 | 0 |
| g++-13 | OpenCV, 1 thread | 60 | 1.35 % / 3.29 % / 11.76 % | 2.48 % / 9.37 % | 3.6 % | 9 | 7 |
| g++-13 | OpenCV, 16 threads | 90 | 1.55 % / 4.95 % / 10.04 % | 2.34 % / 14.76 % | 5.3 % | 27 | 16 |
| clang++-23 | FKL, 1 thread | 90 | 0.23 % / 0.85 % / 1.68 % | 0.43 % / 2.49 % | 2.0 % | 8 | 3 |
| clang++-23 | FKL, 16 threads | 90 | 1.06 % / 2.97 % / 6.06 % | 2.05 % / 12.87 % | 17.3 % | 13 | 0 |
| clang++-23 | OpenCV, 1 thread | 60 | 1.35 % / 3.96 % / 15.31 % | 2.45 % / 7.05 % | 3.5 % | 11 | 10 |
| clang++-23 | OpenCV, 16 threads | 90 | 1.76 % / 3.95 % / 14.80 % | 2.66 % / 13.30 % | 5.8 % | 19 | 8 |

g++-13: 59 of 330 points lie outside the window of the three sweeps (23 FKL, 36 OpenCV); outside the window and the old range: 3 FKL, 23 OpenCV.

clang++-23: 51 of 330 points lie outside the window of the three sweeps (21 FKL, 30 OpenCV); outside the window and the old range: 3 FKL, 18 OpenCV.

The FKL points that are outside both (old median [old min, old max] -> the three new medians; D; S), all single thread, at most 1.7 %:

- g++-13, batch 40, fkl_fused_cpu: 3.9689 [3.9615, 4.0587] -> 3.9345 3.9340 3.9332, D -0.88 %, S 0.03 %
- g++-13, batch 10, fkl_perop_batch_cpu: 1.1539 [1.1500, 1.1573] -> 1.1442 1.1438 1.1436, D -0.87 %, S 0.05 %
- g++-13, batch 240, fkl_perop_batch_cpu: 27.8674 [27.7684, 28.5098] -> 27.5957 27.5803 27.7222, D -0.97 %, S 0.51 %
- clang++-23, batch 10, fkl_perop_batch_cpu: 1.6307 [1.6247, 1.6736] -> 1.6054 1.6033 1.6028, D -1.68 %, S 0.16 %
- clang++-23, batch 140, fkl_perop_batch_cpu: 22.3999 [22.3295, 22.6379] -> 22.2888 22.2347 22.2825, D -0.52 %, S 0.24 %
- clang++-23, batch 200, fkl_perop_batch_cpu: 31.9666 [31.9200, 32.2284] -> 31.8780 31.8777 31.8108, D -0.28 %, S 0.21 %

The largest changes of each kind (same notation):

- g++-13, FKL, batch 90, fkl_perop_batch_cpu_mt: 2.4830 [2.2924, 2.9833] -> 2.5336 2.2419 2.2264, D -9.71 %, S 13.70 %
- g++-13, FKL, batch 110, fkl_fused_cpu_mt: 1.6664 [1.5018, 2.0123] -> 1.5106 1.5098 1.5115, D -9.35 %, S 0.11 %
- g++-13, FKL, batch 140, fkl_perop_crop_cpu_mt: 13.5848 [13.1252, 16.5152] -> 14.4556 14.7071 14.4324, D +6.41 %, S 1.90 %
- g++-13, FKL, batch 110, fkl_perop_crop_cpu_mt: 11.9722 [10.6392, 18.3916] -> 11.4862 11.3097 11.3174, D -5.47 %, S 1.56 %
- g++-13, OpenCV, batch 20, opencv_1thread: 1.2939 [1.1164, 1.5531] -> 1.4460 1.4558 1.3653, D +11.76 %, S 6.26 %
- g++-13, OpenCV, batch 160, opencv_percrop_parallel: 1.4748 [1.4360, 1.5667] -> 1.5593 1.6228 1.6373, D +10.04 %, S 4.81 %
- g++-13, OpenCV, batch 100, opencv_percrop_parallel: 0.9821 [0.9226, 1.0128] -> 1.0542 1.0526 1.0440, D +7.18 %, S 0.96 %
- g++-13, OpenCV, batch 120, opencv_percrop_parallel: 1.1402 [1.0716, 1.2256] -> 1.2074 1.2238 1.2169, D +6.73 %, S 1.35 %
- clang++-23, FKL, batch 270, fkl_fused_cpu_mt: 3.1936 [2.9842, 3.3203] -> 3.0139 3.0002 3.0000, D -6.06 %, S 0.46 %
- clang++-23, FKL, batch 210, fkl_perop_crop_cpu_mt: 20.2527 [19.6094, 24.0926] -> 21.8186 21.3841 21.1379, D +5.59 %, S 3.18 %
- clang++-23, FKL, batch 10, fkl_perop_crop_cpu_mt: 0.9824 [0.8773, 1.3072] -> 0.9931 1.0604 1.0270, D +4.54 %, S 6.55 %
- clang++-23, FKL, batch 270, fkl_perop_batch_cpu_mt: 6.8817 [6.5296, 7.5278] -> 6.5744 6.5990 6.5770, D -4.43 %, S 0.37 %
- clang++-23, OpenCV, batch 260, opencv_1thread: 17.4674 [17.2997, 17.7878] -> 15.4650 14.7937 14.7199, D -15.31 %, S 5.04 %
- clang++-23, OpenCV, batch 260, opencv_percrop_default: 16.8386 [16.7948, 17.5912] -> 14.9086 14.3468 14.2940, D -14.80 %, S 4.28 %
- clang++-23, OpenCV, batch 260, opencv_percrop_1thread: 16.7851 [16.6101, 18.0093] -> 14.8785 14.4018 14.2760, D -14.20 %, S 4.18 %
- clang++-23, OpenCV, batch 260, opencv_default: 17.7367 [17.5554, 18.2271] -> 15.7631 15.1417 15.4097, D -13.12 %, S 4.03 %

The numerical checks are identical to the old rows in all 360 comparisons of every sweep (maximum and mean absolute difference, fraction over 1e-4, pass flags), the required checks pass, and the log of every sweep says "all numerical checks passed":

| compiler | sweep | comparisons | differ from the old rows | required checks failed | log says all passed |
|---|---|--:|--:|--:|---|
| g++-13 | 1 | 360 | 0 | 0 | yes |
| g++-13 | 2 | 360 | 0 | 0 | yes |
| g++-13 | 3 | 360 | 0 | 0 | yes |
| g++-13 | check (nothing else running) | 360 | 0 | 0 | yes |
| clang++-23 | 1 | 360 | 0 | 0 | yes |
| clang++-23 | 2 | 360 | 0 | 0 | yes |
| clang++-23 | 3 | 360 | 0 | 0 | yes |
| clang++-23 | check (nothing else running) | 360 | 0 | 0 | yes |

The differences were not investigated and no cause is given.

## Old and new rows for four variants

Median ms of 30 runs; old (FKL `fb9ac3a`, one sweep) -> new (FKL `c8fcbe8`, median of the three sweeps), the change, and S.

#### g++-13, median ms of 30 runs
| batch | FKL fused, 1 thread | FKL fused, 16 threads | OpenCV, 1 thread | OpenCV, 16 threads |
|--:|--:|--:|--:|--:|
| 10 | 0.984 -> 0.985 (+0.1 %, S 0.3 %) | 0.145 -> 0.145 (+0.2 %, S 0.5 %) | 0.518 -> 0.514 (-0.9 %, S 0.4 %) | 0.514 -> 0.516 (+0.3 %, S 0.8 %) |
| 50 | 4.917 -> 4.916 (-0.0 %, S 0.2 %) | 0.760 -> 0.748 (-1.6 %, S 2.4 %) | 2.713 -> 2.725 (+0.4 %, S 2.0 %) | 2.650 -> 2.681 (+1.2 %, S 1.4 %) |
| 100 | 9.838 -> 9.887 (+0.5 %, S 0.0 %) | 1.383 -> 1.383 (+0.0 %, S 0.4 %) | 5.524 -> 5.635 (+2.0 %, S 2.1 %) | 5.768 -> 5.878 (+1.9 %, S 1.6 %) |
| 150 | 14.754 -> 14.835 (+0.6 %, S 0.5 %) | 2.009 -> 2.016 (+0.4 %, S 2.9 %) | 8.337 -> 8.195 (-1.7 %, S 1.2 %) | 8.597 -> 8.430 (-2.0 %, S 1.2 %) |
| 200 | 19.672 -> 19.667 (-0.0 %, S 0.2 %) | 2.712 -> 2.662 (-1.9 %, S 0.4 %) | 10.853 -> 10.911 (+0.5 %, S 1.8 %) | 11.121 -> 11.104 (-0.2 %, S 2.4 %) |
| 250 | 24.613 -> 24.645 (+0.1 %, S 0.5 %) | 3.296 -> 3.305 (+0.3 %, S 2.8 %) | 14.244 -> 14.117 (-0.9 %, S 1.8 %) | 14.290 -> 14.486 (+1.4 %, S 2.3 %) |
| 300 | 29.515 -> 29.511 (-0.0 %, S 2.3 %) | 3.913 -> 3.989 (+1.9 %, S 8.7 %) | 17.561 -> 17.844 (+1.6 %, S 6.7 %) | 17.838 -> 18.117 (+1.6 %, S 6.5 %) |

#### clang++-23, median ms of 30 runs
| batch | FKL fused, 1 thread | FKL fused, 16 threads | OpenCV, 1 thread | OpenCV, 16 threads |
|--:|--:|--:|--:|--:|
| 10 | 0.851 -> 0.850 (-0.1 %, S 0.1 %) | 0.128 -> 0.127 (-1.0 %, S 0.7 %) | 0.516 -> 0.517 (+0.3 %, S 0.2 %) | 0.514 -> 0.521 (+1.4 %, S 4.8 %) |
| 50 | 4.246 -> 4.247 (+0.0 %, S 0.0 %) | 0.658 -> 0.659 (+0.3 %, S 2.0 %) | 2.780 -> 2.771 (-0.3 %, S 1.0 %) | 2.742 -> 2.722 (-0.7 %, S 0.1 %) |
| 100 | 8.490 -> 8.490 (+0.0 %, S 0.1 %) | 1.201 -> 1.205 (+0.4 %, S 2.5 %) | 5.548 -> 5.411 (-2.5 %, S 1.0 %) | 5.785 -> 5.659 (-2.2 %, S 2.6 %) |
| 150 | 12.734 -> 12.736 (+0.0 %, S 0.1 %) | 1.755 -> 1.757 (+0.1 %, S 0.2 %) | 8.157 -> 8.058 (-1.2 %, S 3.8 %) | 8.299 -> 8.287 (-0.1 %, S 2.7 %) |
| 200 | 16.979 -> 16.978 (-0.0 %, S 0.1 %) | 2.301 -> 2.304 (+0.1 %, S 3.3 %) | 11.482 -> 11.027 (-4.0 %, S 0.9 %) | 11.578 -> 11.280 (-2.6 %, S 4.1 %) |
| 250 | 21.241 -> 21.235 (-0.0 %, S 0.1 %) | 2.908 -> 2.875 (-1.2 %, S 1.2 %) | 14.318 -> 14.254 (-0.4 %, S 2.4 %) | 14.490 -> 14.362 (-0.9 %, S 2.6 %) |
| 300 | 25.469 -> 25.468 (-0.0 %, S 0.0 %) | 3.419 -> 3.410 (-0.3 %, S 2.5 %) | 17.361 -> 17.596 (+1.4 %, S 5.3 %) | 17.752 -> 17.707 (-0.3 %, S 1.8 %) |

## FKL fused against OpenCV at 300 and 100 images

The values of the median-of-three files that the paper uses, with the variant names as they are in the CSV. `opencv_default` and `opencv_1thread` are the whole-batch loop of OpenCV (the "OCVBatch" loop) with `cv::getNumThreads()` = 16 threads and with 1 thread; `opencv_default` is not faster than
`opencv_1thread` (0.6 to 4.6 % slower in the first table). `opencv_percrop_parallel` processes the crops in parallel with `cv::parallel_for_`. FKL's `Resize` samples ASYMMETRIC and OpenCV HALF_PIXEL, so the outputs differ (the comparison of the two conventions in the agreement CSV is informational) and the comparison of the times
is informational too, as for the old rows.

### Time, ms: median of the three sweeps (smallest to largest of the three medians)
| images | compiler | variant in the CSV | what it is | time |
|--:|---|---|---|--:|
| 300 | g++-13 | fkl_fused_cpu | FKL ParArch::CPU, 1 thread, one executeOperations for the batch | 29.511 (29.507 to 30.187) |
| 300 | g++-13 | fkl_fused_cpu_mt | FKL ParArch::CPU_MULTITHREAD, 16 threads, one executeOperations for the batch | 3.989 (3.921 to 4.267) |
| 300 | g++-13 | opencv_1thread | OpenCV, whole-batch loop, 1 thread | 17.844 (17.334 to 18.528) |
| 300 | g++-13 | opencv_default | OpenCV, whole-batch loop, default threads (cv::getNumThreads() = 16) | 18.117 (17.705 to 18.875) |
| 300 | g++-13 | opencv_percrop_parallel | OpenCV, crops processed in parallel with cv::parallel_for_, 16 threads | 2.915 (2.912 to 2.970) |
| 300 | clang++-23 | fkl_fused_cpu | FKL ParArch::CPU, 1 thread, one executeOperations for the batch | 25.468 (25.464 to 25.469) |
| 300 | clang++-23 | fkl_fused_cpu_mt | FKL ParArch::CPU_MULTITHREAD, 16 threads, one executeOperations for the batch | 3.410 (3.406 to 3.490) |
| 300 | clang++-23 | opencv_1thread | OpenCV, whole-batch loop, 1 thread | 17.596 (17.396 to 18.332) |
| 300 | clang++-23 | opencv_default | OpenCV, whole-batch loop, default threads (cv::getNumThreads() = 16) | 17.707 (17.674 to 18.000) |
| 300 | clang++-23 | opencv_percrop_parallel | OpenCV, crops processed in parallel with cv::parallel_for_, 16 threads | 2.890 (2.816 to 2.935) |
| 100 | g++-13 | fkl_fused_cpu | FKL ParArch::CPU, 1 thread, one executeOperations for the batch | 9.887 (9.886 to 9.888) |
| 100 | g++-13 | fkl_fused_cpu_mt | FKL ParArch::CPU_MULTITHREAD, 16 threads, one executeOperations for the batch | 1.383 (1.379 to 1.386) |
| 100 | g++-13 | opencv_1thread | OpenCV, whole-batch loop, 1 thread | 5.635 (5.631 to 5.749) |
| 100 | g++-13 | opencv_default | OpenCV, whole-batch loop, default threads (cv::getNumThreads() = 16) | 5.878 (5.870 to 5.966) |
| 100 | g++-13 | opencv_percrop_parallel | OpenCV, crops processed in parallel with cv::parallel_for_, 16 threads | 1.053 (1.044 to 1.054) |
| 100 | clang++-23 | fkl_fused_cpu | FKL ParArch::CPU, 1 thread, one executeOperations for the batch | 8.490 (8.488 to 8.493) |
| 100 | clang++-23 | fkl_fused_cpu_mt | FKL ParArch::CPU_MULTITHREAD, 16 threads, one executeOperations for the batch | 1.205 (1.204 to 1.234) |
| 100 | clang++-23 | opencv_1thread | OpenCV, whole-batch loop, 1 thread | 5.411 (5.397 to 5.451) |
| 100 | clang++-23 | opencv_default | OpenCV, whole-batch loop, default threads (cv::getNumThreads() = 16) | 5.659 (5.603 to 5.751) |
| 100 | clang++-23 | opencv_percrop_parallel | OpenCV, crops processed in parallel with cv::parallel_for_, 16 threads | 1.037 (1.036 to 1.047) |

### Speedup = OpenCV time / FKL time, computed in each sweep: median over the three sweeps (smallest to largest); above 1 FKL is faster
| images | compiler | FKL variant | OpenCV variant | speedup |
|--:|---|---|---|--:|
| 300 | g++-13 | fkl_fused_cpu | opencv_1thread | 0.60 (0.59 to 0.61) |
| 300 | g++-13 | fkl_fused_cpu | opencv_default | 0.61 (0.60 to 0.63) |
| 300 | g++-13 | fkl_fused_cpu_mt | opencv_percrop_parallel | 0.73 (0.70 to 0.74) |
| 300 | g++-13 | fkl_fused_cpu_mt | opencv_default | 4.52 (4.42 to 4.54) |
| 300 | g++-13 | fkl_fused_cpu_mt | opencv_1thread | 4.42 (4.34 to 4.47) |
| 300 | g++-13 | fkl_fused_cpu | opencv_percrop_parallel | 0.10 (0.10 to 0.10) |
| 300 | clang++-23 | fkl_fused_cpu | opencv_1thread | 0.69 (0.68 to 0.72) |
| 300 | clang++-23 | fkl_fused_cpu | opencv_default | 0.70 (0.69 to 0.71) |
| 300 | clang++-23 | fkl_fused_cpu_mt | opencv_percrop_parallel | 0.85 (0.81 to 0.86) |
| 300 | clang++-23 | fkl_fused_cpu_mt | opencv_default | 5.18 (5.16 to 5.20) |
| 300 | clang++-23 | fkl_fused_cpu_mt | opencv_1thread | 5.17 (4.98 to 5.38) |
| 300 | clang++-23 | fkl_fused_cpu | opencv_percrop_parallel | 0.11 (0.11 to 0.12) |
| 100 | g++-13 | fkl_fused_cpu | opencv_1thread | 0.57 (0.57 to 0.58) |
| 100 | g++-13 | fkl_fused_cpu | opencv_default | 0.59 (0.59 to 0.60) |
| 100 | g++-13 | fkl_fused_cpu_mt | opencv_percrop_parallel | 0.76 (0.76 to 0.76) |
| 100 | g++-13 | fkl_fused_cpu_mt | opencv_default | 4.26 (4.24 to 4.31) |
| 100 | g++-13 | fkl_fused_cpu_mt | opencv_1thread | 4.08 (4.07 to 4.16) |
| 100 | g++-13 | fkl_fused_cpu | opencv_percrop_parallel | 0.11 (0.11 to 0.11) |
| 100 | clang++-23 | fkl_fused_cpu | opencv_1thread | 0.64 (0.64 to 0.64) |
| 100 | clang++-23 | fkl_fused_cpu | opencv_default | 0.67 (0.66 to 0.68) |
| 100 | clang++-23 | fkl_fused_cpu_mt | opencv_percrop_parallel | 0.86 (0.84 to 0.87) |
| 100 | clang++-23 | fkl_fused_cpu_mt | opencv_default | 4.66 (4.65 to 4.70) |
| 100 | clang++-23 | fkl_fused_cpu_mt | opencv_1thread | 4.48 (4.42 to 4.49) |
| 100 | clang++-23 | fkl_fused_cpu | opencv_percrop_parallel | 0.12 (0.12 to 0.12) |

## Sweep against sweep

|a / b - 1| of the medians of two sweeps over the points (median / 90th percentile; for the 16-thread FKL rows also the maximum). Sweeps 1 to 3 are the committed rows and repeats, "check" is `check-nothing-else-running`.

| compiler | sweeps | all rows | FKL, 1 thread | FKL, 16 threads |
|---|---|--|--|--|
| g++-13 | 1 - 2 | 0.75 % / 3.83 % | 0.35 % / 0.78 % | 0.98 % / 4.87 % / max 18.31 % |
| g++-13 | 1 - 3 | 0.76 % / 3.45 % | 0.46 % / 0.72 % | 0.90 % / 4.71 % / max 21.93 % |
| g++-13 | 1 - check | 1.03 % / 4.31 % | 0.21 % / 0.78 % | 1.14 % / 3.94 % / max 18.89 % |
| g++-13 | 2 - 3 | 0.69 % / 2.98 % | 0.25 % / 0.67 % | 0.94 % / 3.31 % / max 16.85 % |
| g++-13 | 2 - check | 0.84 % / 4.33 % | 0.32 % / 0.64 % | 1.31 % / 4.33 % / max 17.35 % |
| g++-13 | 3 - check | 0.90 % / 4.96 % | 0.23 % / 0.71 % | 1.22 % / 4.00 % / max 17.30 % |
| clang++-23 | 1 - 2 | 0.86 % / 3.77 % | 0.24 % / 1.15 % | 1.19 % / 4.41 % / max 8.01 % |
| clang++-23 | 1 - 3 | 0.86 % / 3.99 % | 0.15 % / 1.24 % | 0.73 % / 3.28 % / max 14.51 % |
| clang++-23 | 1 - check | 0.91 % / 3.19 % | 0.30 % / 1.27 % | 1.14 % / 4.02 % / max 17.07 % |
| clang++-23 | 2 - 3 | 1.11 % / 4.12 % | 0.21 % / 1.14 % | 1.38 % / 4.40 % / max 12.74 % |
| clang++-23 | 2 - check | 1.12 % / 4.09 % | 0.19 % / 1.09 % | 1.41 % / 5.59 % / max 16.62 % |
| clang++-23 | 3 - check | 0.92 % / 3.97 % | 0.16 % / 0.95 % | 1.40 % / 4.37 % / max 16.41 % |

## Reproduce

```
python3 scripts/e5_cpu_compare_runs.py a results/cpu-5800x3d/e5-preprocessing-a results/cpu-5800x3d/e5-preprocessing-a-c8fcbe8 --write     # prints this file's numbers, rewrites the median_of_3 files
```
Build and run, per compiler (the OpenCV prefix is the one of the old rows; `external/fkl` at `c8fcbe8` with its `thread-pool` submodule: `git submodule update --init external/fkl && git -C external/fkl submodule update --init thread-pool`):
```
cmake -S . -B build-g13 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-13 -DFKLB_ENABLE_CPU=ON -DFKLB_ENABLE_HIP=OFF -DFKLB_ENABLE_CUDA=OFF -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF -DFKLB_BUILD_HANDWRITTEN=OFF -DFKLB_BUILD_OPENCV_CPU=ON -DOPENCV_DIR=<opencv prefix>
ninja -C build-g13 e5_preprocessing_a_cpp
E5_OUT_DIR=<dir> build-g13/bin/e5_preprocessing_a_cpp
```
