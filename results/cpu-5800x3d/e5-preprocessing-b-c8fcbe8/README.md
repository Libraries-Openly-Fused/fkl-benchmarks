# E5, pipeline (b) on the CPU at the final pin: FKL c8fcbe8 against OpenCV, Ryzen 7 5800X3D

The rows of `../e5-preprocessing-b/` (FKL `fb9ac3a`, before #414 and #416) run again with FKL `c8fcbe8`, the final pin of LTS-C++17 (it has #411, #412, #414 `cxp::nearbyint` at run time, #416 branch free host `cxp::saturate_cast`,
#417); the old rows stay as they were. Same benchmark (`opencv-cpu/preprocessing/bench_preproc_b.cpp`, target `e5_preprocessing_b_cpp`), same protocol (3 warm-up and 30 timed runs, wall clock, median/min/max/mean; seeded noise
NV12 frame 3840x2160, crops 1, 8, 16, 32, 64; FKL `ParArch::CPU` and `CPU_MULTITHREAD` with 16 threads; OpenCV 4.14.0 with IPP), built at fkl-benchmarks `1be0f5e` (clean) with g++ 13.4.0 and clang++ 23.1.5, `-O3 -march=native`.
All numerical checks pass (`stdout.log`). Idle machine (no CI job), **governor `powersave`** (amd-pstate-epp, EPP balance_performance, boost on), as for the old rows; memory speed not recorded (needs root). The FKL commit is in the
header of every CSV.

## What changed: the 8-bit conversion (and the rows that contain it)

The conversion NV12 to 8-bit RGB (`ReadYUV` + `ConvertYUVToRGB` with a `uchar3` output) was 226 ms (g++ 13) and 235 ms (clang++ 23) on one thread before #414, it is now **127.4 ms (g++) and 41.2 ms (clang++)**; with 16 threads 15.5 and 17.6 ms became
**8.30 and 4.83 ms**. For g++ 13 the history is 226 ms (fb9ac3a), 133 ms (after #414, as measured in its pull request), 127 ms (c8fcbe8, #416 included); for clang++ 23 235, 137 and 41 ms. The branch free host path
of #416 helps clang a lot (3.3 times on this noise frame; the cost no longer depends on the data) and g++ 13 little on noise (GCC turns the selects of the narrow types back into branches; on a natural image the gain is larger: see the
issue and the pull request of #416). `cv::cvtColor` with IPP takes 3.40 ms on one thread and 0.38 ms on 16 threads, so the conversion is still the dominant cost of the FKL CPU pipeline (the gap to OpenCV is
37 times with g++ and 12 times with clang++ on one thread). Every row without the conversion (the chain of the crops, OpenCV) is the same as in the old rows within the noise; only two medians outside the pipeline rows differ by more than 12 percent, both of the 16-thread chain of
the crops: g++ at 32 crops (0.545 to 0.478 ms) and clang++ at one crop (0.128 to 0.025 ms; the old row is bimodal, minimum 0.034 and maximum 0.143 ms).

## Results

### g++-13, median ms of 30 runs

| crops | FKL convert 1 thread | FKL convert 16 threads | OpenCV convert 1 thread / 16 threads | FKL chain only (fused) 1 thread / 16 threads | OpenCV whole pipeline 1 thread | FKL whole pipeline (fused) 1 thread | FKL whole pipeline (fused) 16 threads | OpenCV whole pipeline, crops in parallel (16 threads) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 127.4 | 8.30 | 3.40 / 0.38 | 0.101 / 0.126 | 3.46 | 129.2 | 8.29 | 0.487 |
| 8 | 127.4 | 8.23 | 3.42 / 0.39 | 0.818 / 0.203 | 3.93 | 129.8 | 8.36 | 0.539 |
| 16 | 127.4 | 8.22 | 3.41 / 0.39 | 1.641 / 0.326 | 4.37 | 129.0 | 8.50 | 0.586 |
| 32 | 127.5 | 8.25 | 3.41 / 0.38 | 3.291 / 0.477 | 5.29 | 130.5 | 8.64 | 0.722 |
| 64 | 127.5 | 8.20 | 3.40 / 0.38 | 6.565 / 0.965 | 7.10 | 133.6 | 9.13 | 1.003 |

### clang++-23, median ms of 30 runs

| crops | FKL convert 1 thread | FKL convert 16 threads | OpenCV convert 1 thread / 16 threads | FKL chain only (fused) 1 thread / 16 threads | OpenCV whole pipeline 1 thread | FKL whole pipeline (fused) 1 thread | FKL whole pipeline (fused) 16 threads | OpenCV whole pipeline, crops in parallel (16 threads) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 41.2 | 4.83 | 3.40 / 0.38 | 0.081 / 0.025 | 3.41 | 41.4 | 4.83 | 0.485 |
| 8 | 41.4 | 4.78 | 3.43 / 0.40 | 0.665 / 0.210 | 3.93 | 42.1 | 4.93 | 0.536 |
| 16 | 41.4 | 4.78 | 3.43 / 0.39 | 1.327 / 0.290 | 4.41 | 42.7 | 5.03 | 0.590 |
| 32 | 41.2 | 4.79 | 3.43 / 0.39 | 2.648 / 0.487 | 5.31 | 44.0 | 5.32 | 0.725 |
| 64 | 41.4 | 4.78 | 3.43 / 0.38 | 5.296 / 0.863 | 7.34 | 46.7 | 5.56 | 0.985 |
