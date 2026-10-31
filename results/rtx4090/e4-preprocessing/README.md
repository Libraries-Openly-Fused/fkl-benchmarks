# E4a / E4c: the DNN pre-processing chain, FKL against OpenCV-CUDA and NPP (RTX 4090, CUDA 13.4.3, library c8fcbe8)

Program: `opencv-cuda/batched-preprocessing/e4_preprocessing.cu` (header comment: the chain, the implementations, the sampling conventions). Chain: 8UC3 crop -> float, bilinear resize, multiply by 0.3, RGB to BGR, subtract {1, 4, 3.2},
divide by {3.2, 0.6, 11.8}, split into one NCHW float tensor. E4a: BATCH crops of 60 x 120 of a seeded 3840 x 2160 image, output 64 x 128, BATCH = 10, 20, ..., 300. E4c: one whole frame, 720 x 480, 1920 x 1080 and
3840 x 2160, output 768 x 512, 2048 x 1152, 4096 x 2304. Thread fusion off. Timing: CUDA events around one launch of the whole implementation (including the host calls that enqueue it), 100 launches after one warm-up (`gpu_mean_ms`).
Host time of one call, two ways (`host_enqueue_us`: the launch function returns, stream idle before each of 100 calls; `host_launch_us`: 100 calls back to back, wall time per call, the definition of the E2 script).
Built from fkl-benchmarks 51fa9e6 with the library at c8fcbe8 (header lines `# repo`, `# fkl`), FastNPP 9dfc4f4 (the stream-context swap of its tests is not used here: no measurable effect, `../npp-context/`).
`opencv_percrop`: 7 OpenCV-CUDA calls per crop; `opencv_batch`: the strongest OpenCV baseline (convertTo and resize per crop into a stacked buffer, one multiply/cvtColor/subtract/divide for the stack, split per crop);
`npp`: per-crop conversion, one batched resize (single-image resize for E4c), per-crop arithmetic; `fkl_fused`: one kernel (ASYMMETRIC sampling, FKL's default and OpenCV's); `fkl_fused_half_pixel`: the same with
`ResizeSampling::HALF_PIXEL`, the convention of NPP. Each output is checked against the scalar double-precision reference of its own convention (`quiet-run*/e4_preprocessing_agreement.csv`): 486 rows, ok_own = 1 in all of them,
largest error 5.4e-6 (tolerance 1e-4).

## Conditions of the numbers: `quiet-run1`, `quiet-run2`, `quiet-run3`
Three complete sweeps of the same binary, one after the other (about a minute each), pinned to cores 0-3,16-19, on a machine where nothing else ran: no compile job, no other measurement, no transfer (the window script
refuses to start otherwise and checked that); the 1-minute load average was 1.0 to 1.2 before and after every run (`load.txt`: the benchmark itself is the one runnable process). `e4_summary.csv` has, for every point and version, the
median over the three runs of `gpu_mean_ms`, the smallest and the largest of the three, `spread_pct` = (largest - smallest) / median, the median host times; `e4_speedups.csv` the median over the runs of the ratio of the baseline to the fused
kernel, with the smallest and largest of the three ratios. Run-to-run spread of `gpu_mean_ms` over the 162 (point, version) rows: median 0.9 %, 158 rows below 5 %; the four rows above 5 % are `fkl_fused_half_pixel` at batch 10 (47.9 %),
batch 20 (6.7 %) and batch 110 (47.1 %), and `opencv_batch` at batch 10 (13.4 %): in each of them one run is slower than the other two (the medians are those of the two agreeing runs).
The two earlier sweeps, taken while compile jobs ran on the other cores, are in `loaded-sweeps/` with the load noted; against the quiet medians the OpenCV-CUDA and NPP versions (bound by the host that enqueues their kernels) were slower by 3 to 6.7 % (sweep a) and by 12.6 to 16.2 % (sweep b), the FKL kernels by 0.9 to 1.3 % and 4.6 to 5.3 % (medians over the 30 batches of E4a); nothing of them is used.

## Results (median over the 3 runs of `gpu_mean_ms`, ms; run-to-run spread in brackets)
| point | opencv_percrop | opencv_batch | npp | fkl_fused | fkl_fused_half_pixel |
|---|---|---|---|---|---|
| E4a batch 10 | 0.1302 (1.5 %) | 0.05981 (13.4 %) | 0.1322 (0.9 %) | 0.00358 (1.9 %) | 0.003582 (47.9 %) |
| E4a batch 50 | 0.638 (0.1 %) | 0.2668 (0.5 %) | 0.6497 (0.9 %) | 0.005766 (1.0 %) | 0.005784 (0.6 %) |
| E4a batch 100 | 1.272 (1.1 %) | 0.5272 (0.3 %) | 1.297 (1.7 %) | 0.008172 (2.5 %) | 0.00825 (1.4 %) |
| E4a batch 150 | 1.907 (0.8 %) | 0.7893 (0.6 %) | 1.943 (0.8 %) | 0.01069 (1.4 %) | 0.01076 (0.8 %) |
| E4a batch 200 | 2.533 (0.5 %) | 1.052 (0.4 %) | 2.594 (1.6 %) | 0.01353 (1.1 %) | 0.01354 (0.9 %) |
| E4a batch 250 | 3.182 (0.7 %) | 1.32 (0.3 %) | 3.242 (1.2 %) | 0.0161 (0.6 %) | 0.01611 (1.1 %) |
| E4a batch 300 | 3.804 (1.3 %) | 1.594 (0.3 %) | 3.889 (2.1 %) | 0.01851 (0.1 %) | 0.01866 (0.5 %) |
| E4c 720 x 480 | 0.02759 (0.1 %) | - | 0.03603 (0.2 %) | 0.005221 (1.6 %) | 0.005309 (3.1 %) |
| E4c 1920 x 1080 | 0.1734 (0.1 %) | - | 0.1539 (0.1 %) | 0.0156 (0.5 %) | 0.01566 (0.3 %) |
| E4c 3840 x 2160 | 1.605 (0.0 %) | - | 1.849 (0.1 %) | 0.1567 (0.0 %) | 0.1566 (0.1 %) |

Speedup of the fused kernel (median over the runs of the per-run ratio; FKL with ASYMMETRIC sampling against the OpenCV versions, FKL with HALF_PIXEL against NPP): E4a, batch 10 to 300: against `opencv_percrop` 36.4x to 205.6x,
against `opencv_batch` (the strongest OpenCV baseline) 16.9x to 86.2x, against NPP 36.9x to 208.2x; at batch 100: 154.9x, 64.5x, 158.2x; at batch 300: 205.6x, 86.2x, 208.2x. E4c (720p, 1080p, 4K): against `opencv_percrop`
5.3x, 11.1x, 10.2x; against NPP 6.8x, 9.8x, 11.8x. The per-point ratios with their smallest and largest run are in `e4_speedups.csv`.

## Host time of one call (median over the runs, microseconds: `host_enqueue_us` / `host_launch_us`)
| point | opencv_percrop | opencv_batch | npp | fkl_fused | fkl_fused_half_pixel |
|---|---|---|---|---|---|
| E4a batch 10 | 127.4 / 127.7 | 58.9 / 58.4 | 132.5 / 131.3 | 1.7 / 1.7 | 1.7 / 1.8 |
| E4a batch 100 | 1276 / 1280 | 514 / 513.5 | 1295 / 1290 | 1.9 / 2.0 | 1.9 / 1.9 |
| E4a batch 300 | 3819 / 3847 | 1534 / 1524 | 3900 / 3888 | 2.8 / 2.7 | 2.7 / 2.7 |
| E4c 3840 x 2160 | 13.2 / 12.4 | - | 14.8 / 14.3 | 1.7 / 1.6 | 1.7 / 1.6 |

For the per-crop OpenCV-CUDA and the NPP versions in E4a the host time of one call equals the time measured with the events (batch 100: 1.276 ms of host enqueue against 1.272 ms of GPU event time): the call is bound by the
host that enqueues its 700 to 801 kernels, the GPU waits. The fused kernel's host time is 1.7 to 2.8 µs at every batch and is below its GPU time from batch 10 on. In E4c the host time is 12 to 15 µs for 7 to 9 kernels, below the GPU time at every frame size (27.6 µs at 720 x 480, 1.6 to 1.8 ms at 4K): GPU-bound.
Intermediate memory (what the implementation allocates besides the source and the output; FKL: none): E4a batch 100: `opencv_batch` 39.4 MB, NPP 28.3 MB, `opencv_percrop` 0.5 MB; batch 300: 118.1 MB, 84.9 MB, 0.5 MB; E4c 4K: `opencv_percrop` 439.3 MB, NPP 326.0 MB.

## Kernels and DRAM traffic (Nsight Compute, `ncu-warm/`, `ncu-cold/`; summary `e4_dram_bytes.csv` in each)
Kernels launched per call: E4a batch 100: `fkl_fused` 1, `opencv_batch` 304, `opencv_percrop` 700, `npp` 801; E4c 4K: `fkl_fused` 1, `opencv_percrop` 7, `npp` 9. These profile runs (`E4_PROFILE=1 E4_ONLY=...`, one implementation per
run, one launch of each kernel) were taken with the first build of the program (without the host-time columns): the kernels are the same.
- `ncu-cold/` (the profiler default: caches flushed before every kernel): DRAM bytes of the whole call, E4a batch 100: `fkl_fused` 0.09 MB, `opencv_batch` 62.4 MB, `opencv_percrop` 64.2 MB, `npp` 83.9 MB; E4c 4K: `fkl_fused` 90.3 MB, `opencv_percrop` 1,137 MB, `npp` 1,344 MB.
  Flushing before each kernel makes every intermediate of a multi-kernel implementation come from and go to DRAM, which a back-to-back run does not do while the data fits the 72 MB L2.
- `ncu-warm/` (`--cache-control none`, as the application runs): E4a batch 100: `fkl_fused` 0.07 MB, `opencv_batch` 0.32 MB, `opencv_percrop` 0.88 MB, `npp` 0.76 MB (everything stays in L2); E4c 4K (the working set is far above L2): `fkl_fused` 87.8 MB,
  `opencv_percrop` 1,240 MB, `npp` 1,473 MB. For the whole frame the DRAM traffic of the multi-kernel implementations is 14x (OpenCV) and 17x (NPP) that of the fused kernel under both conditions; for the batch of 100 crops the
  intermediates do not reach DRAM when the caches are not flushed.

Files: `quiet-run1..3/` (`e4_preprocessing.csv`, `e4_preprocessing_agreement.csv`, `stdout.txt`, `stderr.txt`, `exit_code.txt`, `load.txt`, `gpu-at-end.csv`), `e4_summary.csv`, `e4_speedups.csv`, `ncu-cold/`, `ncu-warm/`, `loaded-sweeps/` (not for the paper),
`gpu-at-start-quiet.csv`. Tools: `scripts/e4_summary.py`, `scripts/e4_ncu_summary.py`.
