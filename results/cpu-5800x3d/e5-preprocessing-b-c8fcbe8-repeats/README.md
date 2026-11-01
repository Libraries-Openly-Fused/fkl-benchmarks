# E5, pipeline (b) on the CPU at FKL c8fcbe8: three sweeps per compiler, Ryzen 7 5800X3D

Pull request #9 (branch `e5-cpu-b-c8fcbe8`, folder `results/cpu-5800x3d/e5-preprocessing-b-c8fcbe8/`) has one sweep per compiler of pipeline (b) (the NV12 pre-processing, FKL against OpenCV on the CPU) at the final pin, FKL `c8fcbe8`. The 16-thread rows of a
sweep vary between sweeps (see the pipeline (a) rows in `../e5-preprocessing-a-c8fcbe8/`), so two more sweeps per compiler were run here with the same binaries and the same protocol (3 warm-up and 30 timed runs per point and variant, wall clock, median/min/max/mean; seeded noise
NV12 frame 3840x2160, crops 1, 8, 16, 32, 64; FKL `ParArch::CPU` with 1 thread and `CPU_MULTITHREAD` with 16 threads; OpenCV 4.14.0 with IPP; `bench_preproc_b.cpp`, target `e5_preprocessing_b_cpp`; fkl-benchmarks `1be0f5e` clean, g++ 13.4.0 and clang++ 23.1.5, `-O3 -march=native`).
The binaries are the ones of pull request #9 (sha256 starting `33644f33` for g++ and `fbe2b42e` for clang++). The FKL commit and the fkl-benchmarks commit are in the header of every CSV. Governor `powersave` (amd-pstate-epp, EPP balance_performance, boost on); the memory
speed is not recorded (it needs root). The files of pull request #9 are not touched.

## What is here

- `<compiler>/run1/`: the three files of pull request #9 for that compiler, byte for byte (compared with the files of the branch `e5-cpu-b-c8fcbe8` when this was committed), copied so that this folder is complete; `run2/` and `run3/`: the two new sweeps, raw output
  (`e5_preprocessing_b_CPU.csv`, `e5_preprocessing_b_agreement_CPU.csv`, `stdout.log`).
- `g++-13/e5_preprocessing_b_CPU_median_of_3.csv`, `clang++-23/e5_preprocessing_b_CPU_median_of_3.csv`: **the numbers to cite.** For every point and variant `median_ms` is the median of the medians of run1, run2 and run3, and `medians_min_ms` and `medians_max_ms` the
  smallest and the largest of the three medians.
- `scripts/e5_cpu_compare_runs.py` writes the median files and prints every number of this file from the CSVs (command at the end).

## Conditions

Run2 and run3 were run interleaved (g++ run2, clang++ run2, g++ run3, clang++ run3) on 2026-11-01 between 15:35 and 15:44, each started only with no `Runner.Worker` (CI job) on the machine and a 1-minute load average below 0.8, with no CI job appearing while it ran (checked every
2 s), and with the session that ran them issuing no other command while they ran. Wall time of a sweep: 96 s with g++ and 35 s with clang++, as in pull request #9. Run1 is the sweep of pull request #9 (2026-11-01, about 00:29 to 00:32).

## Results

The numerical checks are identical in the three sweeps (all 50 comparisons), the required checks pass, and every log says "all numerical checks passed":

| compiler | sweep | comparisons | differ from run1 | required checks failed | log says all passed |
|---|---|--:|--:|--:|---|
| g++-13 | 1 | 50 | 0 | 0 | yes |
| g++-13 | 2 | 50 | 0 | 0 | yes |
| g++-13 | 3 | 50 | 0 | 0 | yes |
| clang++-23 | 1 | 50 | 0 | 0 | yes |
| clang++-23 | 2 | 50 | 0 | 0 | yes |
| clang++-23 | 3 | 50 | 0 | 0 | yes |

The median of the three sweeps against run1 (the rows of pull request #9), and the spread S = (largest minus smallest median) / median of the three sweeps at the point:

| compiler | points | change \|D\| = \|median of three / run1 - 1\|: median / 90th percentile / max | spread S = (max-min)/median of the three sweeps: median / 90th percentile / max |
|---|--:|--|--|
| g++-13 | 75 | 0.03 % / 0.85 % / 2.50 % | 0.43 % / 4.14 % / 28.99 % |
| clang++-23 | 75 | 0.05 % / 2.25 % / 13.66 % | 0.67 % / 5.21 % / 408.66 % |

The six largest differences per compiler, with the three sweeps in order run1, run2, run3:

- g++-13, 64 crops, opencv_pipeline_percrop_parallel: 1.0027 0.9746 0.9776, median of three 0.9776, D -2.50 %, S 2.87 %
- g++-13, 32 crops, fkl_pipeline_perop_crop_cpu_mt: 11.8157 11.5237 11.4884, median of three 11.5237, D -2.47 %, S 2.84 %
- g++-13, 64 crops, opencv_pipeline_percrop_1thread: 7.0975 7.3548 7.2016, median of three 7.2016, D +1.47 %, S 3.57 %
- g++-13, 1 crops, opencv_pipeline_percrop_1thread: 3.4596 3.4111 3.4135, median of three 3.4135, D -1.33 %, S 1.42 %
- g++-13, 8 crops, fkl_chain_fused_cpu_mt: 0.2028 0.2052 0.2113, median of three 0.2052, D +1.17 %, S 4.14 %
- g++-13, 16 crops, opencv_pipeline_percrop_parallel: 0.5856 0.5751 0.5801, median of three 0.5801, D -0.94 %, S 1.82 %
- clang++-23, 32 crops, opencv_pipeline_percrop_default: 2.4241 3.5264 2.7551, median of three 2.7551, D +13.66 %, S 40.01 %
- clang++-23, 16 crops, opencv_pipeline_percrop_default: 1.7405 1.9872 1.9707, median of three 1.9707, D +13.22 %, S 12.52 %
- clang++-23, 64 crops, fkl_pipeline_perop_crop_cpu_mt: 10.7901 11.3521 11.7625, median of three 11.3521, D +5.21 %, S 8.57 %
- clang++-23, 8 crops, fkl_chain_fused_cpu_mt: 0.2098 0.2004 0.1994, median of three 0.2004, D -4.49 %, S 5.21 %
- clang++-23, 8 crops, opencv_convert_default: 0.4034 0.3865 0.3864, median of three 0.3865, D -4.19 %, S 4.38 %
- clang++-23, 32 crops, opencv_convert_default: 0.3869 0.4065 0.3998, median of three 0.3998, D +3.34 %, S 4.92 %

The FKL conversion and whole-pipeline rows (the ones the paper quotes), the median of three against run1 over the five crop counts:

| compiler | variant | max \|D\| over the crop counts | median of three over the crop counts: smallest to largest (ms) | S: median / max |
|---|---|--:|--|--|
| g++-13 | fkl_convert_cpu | 0.04 % | 127.4 to 127.5 | 0.05 % / 0.18 % |
| g++-13 | fkl_convert_cpu_mt | 0.44 % | 8.23 to 8.33 | 1.37 % / 5.73 % |
| g++-13 | fkl_pipeline_fused_cpu | 0.07 % | 129.0 to 133.6 | 0.05 % / 0.10 % |
| g++-13 | fkl_pipeline_fused_cpu_mt | 0.59 % | 8.30 to 9.19 | 0.86 % / 3.20 % |
| clang++-23 | fkl_convert_cpu | 0.48 % | 41.2 to 41.4 | 0.50 % / 0.73 % |
| clang++-23 | fkl_convert_cpu_mt | 0.58 % | 4.78 to 4.80 | 0.74 % / 0.97 % |
| clang++-23 | fkl_pipeline_fused_cpu | 0.53 % | 41.3 to 46.5 | 0.53 % / 0.56 % |
| clang++-23 | fkl_pipeline_fused_cpu_mt | 2.25 % | 4.83 to 5.57 | 0.71 % / 2.36 % |

With 16 threads the conversion is 8.23 to 8.33 ms with g++ (8.33 ms at one crop, 8.30 to 8.34 over the three sweeps) and 4.78 to 4.80 ms with clang++ (4.80 ms at one crop, 4.79 to 4.83): the 8.3 and 4.8 ms of pull request #9 (8.30 and 4.83 at one crop) hold. The 16-thread chain of the crops with clang++
at one crop is bimodal: 0.0251 ms (run1), 0.0229 ms (run2) and 0.1253 ms (run3); the median of three is 0.0251 ms.

### Median of the three sweeps, ms

In each cell the median of the three sweeps and, in brackets, the smallest and the largest of the three medians; the columns are those of the tables of pull request #9.

### g++-13
| crops | FKL convert 1 thread | FKL convert 16 threads | OpenCV convert 1 thread | OpenCV convert 16 threads | FKL chain only (fused) 1 thread | FKL chain only (fused) 16 threads | OpenCV whole pipeline 1 thread | FKL whole pipeline (fused) 1 thread | FKL whole pipeline (fused) 16 threads | OpenCV whole pipeline, crops in parallel (16 threads) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 127.4 (127.4 to 127.5) | 8.33 (8.30 to 8.34) | 3.40 (3.40 to 3.40) | 0.382 (0.382 to 0.383) | 0.101 (0.101 to 0.101) | 0.126 (0.126 to 0.127) | 3.41 (3.41 to 3.46) | 129.2 (129.1 to 129.2) | 8.30 (8.29 to 8.31) | 0.487 (0.487 to 0.487) |
| 8 | 127.4 (127.4 to 127.6) | 8.23 (8.22 to 8.24) | 3.42 (3.41 to 3.42) | 0.386 (0.386 to 0.386) | 0.819 (0.818 to 0.822) | 0.205 (0.203 to 0.211) | 3.90 (3.87 to 3.93) | 129.9 (129.8 to 129.9) | 8.39 (8.36 to 8.43) | 0.539 (0.538 to 0.539) |
| 16 | 127.4 (127.4 to 127.5) | 8.25 (8.22 to 8.69) | 3.41 (3.40 to 3.41) | 0.389 (0.386 to 0.394) | 1.64 (1.64 to 1.64) | 0.324 (0.232 to 0.326) | 4.37 (4.30 to 4.38) | 129.0 (129.0 to 129.0) | 8.50 (8.50 to 8.77) | 0.580 (0.575 to 0.586) |
| 32 | 127.5 (127.4 to 127.5) | 8.25 (8.23 to 8.34) | 3.43 (3.41 to 3.43) | 0.385 (0.384 to 0.392) | 3.29 (3.29 to 3.31) | 0.477 (0.477 to 0.482) | 5.33 (5.29 to 5.34) | 130.5 (130.5 to 130.5) | 8.65 (8.64 to 8.67) | 0.716 (0.714 to 0.722) |
| 64 | 127.5 (127.4 to 127.5) | 8.24 (8.20 to 8.34) | 3.40 (3.39 to 3.40) | 0.384 (0.384 to 0.385) | 6.57 (6.57 to 6.60) | 0.965 (0.934 to 0.967) | 7.20 (7.10 to 7.35) | 133.6 (133.6 to 133.6) | 9.19 (9.13 to 9.27) | 0.978 (0.975 to 1.003) |

### clang++-23
| crops | FKL convert 1 thread | FKL convert 16 threads | OpenCV convert 1 thread | OpenCV convert 16 threads | FKL chain only (fused) 1 thread | FKL chain only (fused) 16 threads | OpenCV whole pipeline 1 thread | FKL whole pipeline (fused) 1 thread | FKL whole pipeline (fused) 16 threads | OpenCV whole pipeline, crops in parallel (16 threads) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 41.2 (41.2 to 41.2) | 4.80 (4.79 to 4.83) | 3.40 (3.40 to 3.40) | 0.382 (0.381 to 0.382) | 0.081 (0.081 to 0.081) | 0.025 (0.023 to 0.125) | 3.41 (3.39 to 3.46) | 41.3 (41.3 to 41.4) | 4.83 (4.83 to 4.86) | 0.487 (0.485 to 0.487) |
| 8 | 41.4 (41.2 to 41.4) | 4.78 (4.78 to 4.82) | 3.43 (3.42 to 3.44) | 0.386 (0.386 to 0.403) | 0.664 (0.663 to 0.665) | 0.200 (0.199 to 0.210) | 3.93 (3.92 to 3.93) | 42.1 (41.8 to 42.1) | 4.91 (4.90 to 4.93) | 0.537 (0.536 to 0.538) |
| 16 | 41.4 (41.2 to 41.4) | 4.79 (4.78 to 4.81) | 3.42 (3.41 to 3.43) | 0.386 (0.386 to 0.386) | 1.33 (1.33 to 1.35) | 0.290 (0.222 to 0.300) | 4.41 (4.40 to 4.41) | 42.7 (42.6 to 42.7) | 5.03 (5.02 to 5.04) | 0.590 (0.582 to 0.604) |
| 32 | 41.2 (41.2 to 41.5) | 4.79 (4.78 to 4.81) | 3.43 (3.40 to 3.43) | 0.400 (0.387 to 0.407) | 2.65 (2.65 to 2.65) | 0.484 (0.478 to 0.487) | 5.31 (5.26 to 5.39) | 44.0 (43.9 to 44.1) | 5.20 (5.19 to 5.32) | 0.725 (0.712 to 0.762) |
| 64 | 41.2 (41.2 to 41.4) | 4.80 (4.78 to 4.82) | 3.43 (3.42 to 3.43) | 0.386 (0.384 to 0.392) | 5.29 (5.29 to 5.30) | 0.854 (0.846 to 0.863) | 7.21 (7.10 to 7.34) | 46.5 (46.5 to 46.7) | 5.57 (5.56 to 5.65) | 0.985 (0.978 to 0.996) |

## Reproduce

```
python3 scripts/e5_cpu_compare_runs.py b results/cpu-5800x3d/e5-preprocessing-b-c8fcbe8-repeats --write     # prints this file's numbers, rewrites the median_of_3 files
```
