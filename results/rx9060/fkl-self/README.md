# FKL-only benchmarks on the RX 9060 (gfx1200), ROCm 10.1: horizontal fusion, instructions per operation, latency hiding

`benchmark_horizontal_fusion_hip`, `benchmark_numinstructions_x_operation_hip` and `benchmark_latency_hiding_hip` (`fkl-self/`): horizontal fusion 120 points (batch 1, 11, ..., 1191; source 3840 wide x
2160 high, crops of 60x120), instructions per operation 100 points (1, 6, ..., 496; 500 instructions in total), latency hiding 60 points (1, 21, ..., 1181 operations). Thread fusion is not
measured. Each CSV column is the mean, variance, maximum and minimum time in ms of the benchmark (there is no median in these files). Horizontal fusion has three CSVs: with the CPU overhead of the
launches, without it, and `..._HIPGraphs_...` (the iterated batch as one HIP graph per batch, the HIP version of the CUDA Graphs variant). Machine, conditions and the clock of the GPU: `../README.md`
(governor `powersave`; idle machine, no CI job on the runner during any run).

## Which numbers to cite

| benchmark | folder | built from | status |
|---|---|---|---|
| latency hiding | `latency_hiding/` | FKL `c8fcbe8`, fkl-benchmarks `8390007`, `FKLB_GPU_SPIN_SECONDS=2` | **supersedes** `latency_hiding_without_spin_38c4f32/` (the first run, 15:10, in the afternoon) and `latency_hiding_without_spin_same_day/` |
| instructions per operation | `numinstructions_x_operation/`, `numinstructions_x_operation_repeat_22-25/` | FKL `38c4f32` (fkl-benchmarks `e0af273` = main `9afddfb` + #1 `7b68ff4`, which repairs the HIP part of the CSV header only); the repeat with FKL `c8fcbe8` | unchanged by the clock fix; the two runs agree within 2 percent |
| horizontal fusion | `horizontal_fusion/` (run 1), `horizontal_fusion_repeats/` (runs 2 to 4) | the same | unchanged by the clock fix, but the **absolute times depend on the state of the GPU clocks** (below); the speedup does not |

The device code of the three binaries is identical function by function at FKL `38c4f32` and `c8fcbe8` (372, 124 and 246 functions, 0 different: `../clock-ramp/`), so the results of the
older pin hold for the final pin.

## Latency hiding (the one that changed)

One kernel with N multiplications on 253 MiB of float in and out (details below). The buffers are allocated and filled between two points, which leaves the GPU idle, and a point times only 100
executions: without keeping the GPU busy the timed runs of most points are on the ramp of the GPU clock (2.0 GHz for the first 0.7 s of work, 3.0 GHz and more after 1 s), and the series was up to 52
percent slower than at the steady clock. With `FKLB_GPU_SPIN_SECONDS=2` (`fkBenchmarksCommon.h`, off by default, so nothing changes for a run that does not set it) the same work runs for 2 s before the
timed runs of each point: the clock is 3.16 to 3.43 GHz (median 3.17 GHz) at the end of the spin of every one of the 60 points (`latency_hiding/clock-per-point.txt`; the first three points: 3165, 3169,
3165 MHz). Two runs with the spin agree to 0.2 percent. Time in ms (mean of the 100 executions):

| operations | new: with the spin (c8fcbe8) | earlier run, no spin (38c4f32, 15:10) | same day, no spin (c8fcbe8, 21:09) | no spin / new (same day) |
|--:|--:|--:|--:|--:|
| 1 | 1.938 | 1.944 | 1.961 | 1.01 |
| 21 | 1.941 | 1.959 | 2.009 | 1.04 |
| 61 | 1.965 | 2.018 | 2.184 | 1.11 |
| 101 | 2.050 | 2.149 | 2.756 | 1.34 |
| 141 | 2.359 | 2.465 | 3.369 | 1.43 |
| 201 | 2.912 | 3.013 | 4.400 | 1.51 |
| 301 | 4.005 | 4.139 | 6.099 | 1.52 |
| 401 | 5.081 | 5.250 | 7.660 | 1.51 |
| 581 | 7.063 | 7.282 | 9.980 | 1.41 |
| 781 | 9.172 | 9.569 | 12.336 | 1.34 |
| 981 | 11.048 | 11.536 | 14.694 | 1.33 |
| 1181 | 13.149 | 13.164 | 17.048 | 1.30 |

The run of the afternoon (15:10) was 0 to 3 percent above the steady values by luck (its ramps were short); the run of the evening without the spin is 52 percent slower at 301 operations.

## Instructions per operation

The clock is 3.3 to 3.45 GHz at every point after the first second (`clock-evidence/summary.txt`); the first point (500 separated kernels per iteration, 97 s) has its first second at 2.8 GHz, which is
about 1 percent of that point and does not show in the mean of its fused part (5.897 ms; maximum 5.947 ms). Two runs, 15:17 and 22:25 (mean ms):

| instructions per operation | separated, run 1 | separated, repeat | all fused, run 1 | all fused, repeat |
|--:|--:|--:|--:|--:|
| 1 | 966.015 | 965.343 | 5.897 | 5.907 |
| 6 | 162.413 | 162.304 | 5.882 | 6.003 |
| 51 | 19.587 | 19.423 | 5.843 | 5.850 |
| 101 | 10.340 | 10.118 | 5.953 | 5.981 |
| 251 | 6.626 | 6.636 | 5.957 | 5.964 |
| 496 | 7.736 | 7.718 | 5.857 | 5.865 |

## Horizontal fusion: the speedup is stable, the absolute times are not

Four runs of the same code, with identical device code: run 1 (15:17), the three of the evening (the clock tracer, a pre-heat of 12 s of latency hiding, nothing else running). Mean ms; the speedup is
iterated over fused, and the last column is the iterated batch captured as a HIP graph over the iterated batch with plain launches (without the CPU overhead file):

| run | batch | iterated (launches) | fused | speedup | iterated as a HIP graph | graph / launches |
|---|--:|--:|--:|--:|--:|--:|
| 1 (15:17, built at 38c4f32) | 101 | 0.3386 | 0.0310 | 10.9 | 0.6538 | 1.93 |
|  | 301 | 0.9479 | 0.0643 | 14.7 | 1.8498 | 1.95 |
|  | 601 | 1.7972 | 0.1121 | 16.0 | 3.6995 | 2.06 |
|  | 1191 | 3.5537 | 0.2098 | 16.9 | 7.0019 | 1.97 |
| 2 (22:12, clock tracer 200 Hz) | 101 | 0.3967 | 0.0358 | 11.1 | 0.7445 | 1.88 |
|  | 301 | 1.1608 | 0.0781 | 14.9 | 2.1734 | 1.87 |
|  | 601 | 2.3099 | 0.1423 | 16.2 | 4.3244 | 1.87 |
|  | 1191 | 4.5571 | 0.2747 | 16.6 | 8.3905 | 1.84 |
| 3 (22:33, after 12 s of latency hiding, sampler 5 Hz) | 101 | 0.4157 | 0.0385 | 10.8 | 0.7982 | 1.92 |
|  | 301 | 1.2213 | 0.0846 | 14.4 | 2.3340 | 1.91 |
|  | 601 | 2.4383 | 0.1545 | 15.8 | 4.6418 | 1.90 |
|  | 1191 | 4.7731 | 0.2922 | 16.3 | 8.4754 | 1.78 |
| 4 (22:48, nothing else running) | 101 | 0.4047 | 0.0374 | 10.8 | 0.7840 | 1.94 |
|  | 301 | 1.1914 | 0.0831 | 14.3 | 2.2684 | 1.90 |
|  | 601 | 2.3615 | 0.1507 | 15.7 | 4.5784 | 1.94 |
|  | 1191 | 4.6822 | 0.2872 | 16.3 | 8.4771 | 1.81 |

The speedup agrees within 3 percent in the four runs (10.8 to 11.1 at 101 crops, 14.3 to 14.9 at 301, 15.7 to 16.2 at 601, 16.3 to 16.9 at 1191) and so does the HIP graph against plain launches (1.78 to 2.06 times
slower). The absolute times of the evening runs are 15 to 38 percent above run 1 (iterated at 1191 crops: 3.55 against 4.56 to 4.77 ms). The GPU does not boost for this benchmark: the kernels are tiny and the work is bound by
the launches, and the SMU keeps the clocks low: in the first 75 s of a run (`clock-evidence/hf-first-75s-sclk-mclk-busy-power-temp.txt`) the graphics clock is 1.6 to 1.95 GHz (maximum 2.09 GHz in the horizontal fusion sweeps and 2.7 GHz in the HIP graph sweep of the traced run, against
3.0 to 3.4 GHz in the other benchmarks), the memory clock mostly 772 to 875 MHz (456 and 1124 MHz also occur; the maximum of the GPU is 1124 MHz), and the power 22 to 29 W of 132 W. The clocks of run 1 were not
sampled; its times fit higher clocks (higher memory clock), which the SMU chose at that time and did not choose in the three runs of the evening, whatever the history of the GPU before the run (run 3 started right after
12 s of heavy work and ran at 1.82 GHz median). Without root the clocks cannot be fixed: `echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level` (as root; `auto` restores it) would fix the highest
levels and make the absolute times of this benchmark repeatable. Until then, cite the speedup and the graph ratio, and give the absolute times as a range of the four runs.

## What the latency hiding harness runs (`fkl-self/latencyhiding/benchmark_latency_hiding.h`)

- Data type `float`; one 1D buffer of `3840 * 2160 * 8 = 66 355 200` elements (265 420 800 bytes, 253.1 MiB) as input and another as output, so each launch reads 253 MiB and writes 253 MiB; the input is set to 1.0.
- The operation is a multiplication by 2 (`Mul<float>`), repeated N times (N = 1, 21, ..., 1181) in ONE fused kernel (`PerThreadRead` + N times `Mul` as nested `StaticLoop`s of 20 + the remainder +
  `PerThreadWrite`, `TransformDPP`, thread fusion at its default); a single `executeOperations` per point: one warm-up execution, `FKLB_GPU_SPIN_SECONDS` seconds of the same work, then 100 timed executions
  (`ITERS`), each timed with HIP events on the stream. The values overflow to infinity for large N, which does not change the time.
- At one operation the time is 1.940 ms for 506 MiB moved, i.e. 274 GB/s (decimal) of read plus write traffic; the time grows from 101 operations on.

## Reproduce

```
FKLB_GPU_SPIN_SECONDS=2 ./benchmark_latency_hiding_hip                       # latency hiding at the steady clock
python3 scripts/gpu_clock_trace.py run <dir> -- ./benchmark_numinstructions_x_operation_hip   # any of them with the clock of the GPU
```
