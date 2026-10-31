# Data-size experiment: why OpenCV-CUDA alternates between two speeds

**Finding.** In `e0-data-size` the OpenCV-CUDA time of the one-channel pairs (8UC1 -> 32FC1 and 32FC1 -> 32FC1) is about 3x higher at every point whose element count N is not a multiple of 4: the 60 sizes of the benchmark are 100 + 282,270 k, which alternate between N mod 4 = 0 and N mod 4 = 2. cvGS (the fused kernel) is not affected, and neither are the 3- and 4-channel pairs. The 2025 data of the paper did not show it.

**Mechanism (confirmed in the kernels, Nsight Compute, RTX 4090, CUDA 13.4.3, OpenCV 4.14).** OpenCV-CUDA's `convertTo`, `multiply` and `add` on a one-channel matrix launch `cv::transformSmart<4, ...>`, a kernel that processes four elements per thread, when the element count is a multiple of 4, and the scalar `cv::transformSimple<...>` otherwise. All 30 sizes with N mod 4 = 0 of the benchmark run the vectorised kernel and all 30 with N mod 4 = 2 run the scalar one (`parity_ncu_sweep_32FC1.csv`). The residues 1, 2 and 3 are equally slow (N = 1,000,000 ... 1,000,007: 1.11 ms for N mod 4 = 0, 3.34 ms for 1, 2 and 3, mod 8 does not matter; `timing_selected_sizes.csv`). 3-channel matrices never use the vectorised kernel (`transformSimple` for `multiply` and `add` at every size), so they have no parity effect.

## What Nsight Compute measures (float -> float pair; `parity_ncu_selected_sizes.csv`)
One launch of each kernel, clocks locked and caches flushed by ncu between kernels (isolated cold launches, not the back-to-back launches of the benchmark: the figures describe the kernels, not the benchmark's run). `multiply` is the representative kernel of the 201 launched per OpenCV-CUDA run (`convertTo` and `add` are within 2 % of it). Peak DRAM throughput is `dram__throughput.avg.pct_of_peak_sustained_elapsed`.

| N | N mod 4 | OpenCV kernel | duration (us) | DRAM (% of peak) | cvGS fused kernel: duration (us), DRAM (%) |
|---:|---:|---|---:|---:|---|
| 100 | 0 | transformSmart<4> | 1.79 | 0.4 | 1.82, 0.3 |
| 282,370 | 2 | transformSimple | 6.72 | 17.2 | 3.42, 33.9 |
| 564,640 | 0 | transformSmart<4> | 4.51 | 51.7 | 4.67, 49.9 |
| 846,910 | 2 | transformSimple | 15.81 | 21.9 | 6.05, 57.5 |
| 9,032,740 | 0 | transformSmart<4> | 40.90 | 92.6 | 43.68, 87.0 |
| 16,654,030 | 2 | transformSimple | 274.69 | 36.2 | 97.44, 92.1 |

Totals of the whole OpenCV-CUDA sequence (1 convertTo + 100 x multiply + 100 x add, CUDA events, mean of 100 runs; `timing_benchmark_sizes.csv`, equal to the `e0-data-size` CSV to 0.05 %): 0.396 ms at N = 100, 1.195 ms at 282,370, 0.777 ms at 564,640, 7.520 ms at 9,032,740 and 53.54 ms at 16,654,030 (the cvGS fused kernel alone, without the wrapper: 0.0029, 0.0040, 0.0052, 0.0364 and 0.1454 ms).

Over the 60 sizes (float pair): the OpenCV `multiply` kernel exceeds 40 % of the peak DRAM bandwidth for the first time at N = 564,640 (51.7 %); it never exceeds 40 % at an N mod 4 = 2 size (maximum 36.3 %) and reaches 94.8 % at N mod 4 = 0 sizes. The fused kernel first exceeds 40 % at N = 564,640 too (50.0 %) and reaches 92.1 %. The 2025 text quotes 0.6 % (N = 100), 30 % (N = 282,370), "about 40 %" and "up to 90 %": the new values are 0.4 %, 17.2 % (a scalar-kernel size; 51.7 % at the next, vectorised size), 564,640 elements for the 40 % threshold, and up to 94.8 % (vectorised) or 36.3 % (scalar) at the large sizes.

## Consequence for the experiment
The speedup curve of the paper's figure (cvGS against OpenCV-CUDA) is the ratio of a smooth cvGS time to an OpenCV-CUDA time that alternates between the vectorised and the scalar kernel; the ratio at a size with N mod 4 = 2 is about 3x higher than at its neighbours (float pair, `Mean Speedup` of the `e0-data-size` CSV: 135x at N = 11,290,900 with N mod 4 = 0, 416x at 12,702,250 with N mod 4 = 2, 164x at 14,113,600 with N mod 4 = 0, 409x at 16,654,030 with N mod 4 = 2), so no speedup read at two end points of different parity should be quoted. The sweep with all sizes multiples of 4 (`../e0-data-size-mul4`, the benchmark `benchmark_image_resolution_MAD_loop_mul4`: increment 282,272, one-channel pairs only) is the best case for OpenCV-CUDA and the stronger comparison; the original sweep is kept unchanged.

Files: `opencv_parity_probe.csv`, `timing_*.csv` and `stdout_*` (event timing by element count, `opencv-cuda/parity/opencv_parity_probe.cu`); `ncu_*.csv` (raw Nsight Compute output) and the tables derived by `scripts/parity_tables.py`.
