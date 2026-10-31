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
| horizontal fusion | `horizontal_fusion_with_flat_graph/run1` to `run3` (with the second HIP graph variant), `horizontal_fusion/` (run 1 of the earlier set), `horizontal_fusion_repeats/` (runs 2 to 4) | FKL `c8fcbe8`, fkl-benchmarks `132d2cc` for the three new runs; the earlier four with FKL `38c4f32` / `c8fcbe8` as above | unchanged by the clock fix, but the **absolute times depend on the state of the GPU clocks** (below); the speedup and the graph ratios do not |

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

## Horizontal fusion with the second HIP graph variant (fkl-benchmarks `132d2cc`, FKL `c8fcbe8`)

The benchmark now has two HIP graph variants (`benchmark_horizontal_fusion.h`): `..._HIPGraphs`, unchanged (ONE CHILD GRAPH NODE PER CROP, each captured from one launch, no dependencies between them: the structure of the CUDA Graphs
variant), and `..._HIPGraphs_OneCapturedGraph`, new (ONE GRAPH FROM ONE STREAM CAPTURE of all the launches, a chain of kernel nodes). Both time one `hipGraphLaunch` against the fused kernel, the CSV notes say how each graph
is built, and the output of the new graph is compared bit by bit with the output of the plain launches for every crop before its timing (all 120 batches of the three runs passed; a negative control with the launch removed reports
every value). Three full runs of the same binary, nothing else running, with a sampler at 5 Hz next to each (`clocks5hz.txt`: time, graphics clock, memory clock, GPU busy percent, power, temperature), 886 to 892 s each
(`horizontal_fusion_with_flat_graph/run1` to `run3`). Mean ms of the iterated batch (the "without CPU overhead" file) and of the fused kernel; median of the three runs and, in brackets, the smallest and the largest; the ratios
are computed in each run and then summarised the same way:

| batch | iterated, plain launches | iterated, graph of child nodes | iterated, graph of one capture | fused kernel | child graph / plain | one-capture graph / plain | plain / fused | one-capture graph / fused |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 11 | 0.0493 (0.0492 to 0.0493) | 0.0900 (0.0898 to 0.0905) | 0.0463 (0.0457 to 0.0501) | 0.0136 (0.0135 to 0.0136) | 1.83 (1.82 to 1.83) | 0.94 (0.93 to 1.02) | 3.63 (3.63 to 3.64) | 3.42 (3.37 to 3.70) |
| 51 | 0.2031 (0.2029 to 0.2035) | 0.3786 (0.3769 to 0.3886) | 0.1710 (0.1705 to 0.1861) | 0.0231 (0.0231 to 0.0231) | 1.86 (1.85 to 1.92) | 0.84 (0.84 to 0.91) | 8.78 (8.78 to 8.82) | 7.40 (7.38 to 8.06) |
| 101 | 0.3940 (0.3940 to 0.3951) | 0.7383 (0.7369 to 0.7423) | 0.3274 (0.3263 to 0.3568) | 0.0358 (0.0358 to 0.0359) | 1.87 (1.87 to 1.88) | 0.83 (0.83 to 0.91) | 11.00 (10.99 to 11.01) | 9.12 (9.10 to 9.97) |
| 201 | 0.7751 (0.7740 to 0.7754) | 1.4520 (1.4511 to 1.4547) | 0.6367 (0.6357 to 0.6918) | 0.0572 (0.0572 to 0.0572) | 1.87 (1.87 to 1.88) | 0.82 (0.82 to 0.89) | 13.54 (13.54 to 13.55) | 11.12 (11.12 to 12.09) |
| 301 | 1.1563 (1.1560 to 1.1578) | 2.1701 (2.1639 to 2.1735) | 0.9456 (0.9455 to 1.0276) | 0.0784 (0.0783 to 0.0784) | 1.88 (1.87 to 1.88) | 0.82 (0.82 to 0.89) | 14.75 (14.74 to 14.79) | 12.07 (12.05 to 13.13) |
| 601 | 2.3102 (2.3081 to 2.3147) | 4.4052 (4.3963 to 4.6965) | 1.9402 (1.9036 to 2.0391) | 0.1439 (0.1419 to 0.1442) | 1.91 (1.90 to 2.03) | 0.84 (0.82 to 0.88) | 16.04 (16.02 to 16.31) | 13.46 (13.23 to 14.37) |
| 1191 | 4.5010 (4.4979 to 4.5201) | 8.5238 (8.4516 to 8.5378) | 3.7912 (3.7154 to 4.1456) | 0.2734 (0.2726 to 0.2747) | 1.90 (1.87 to 1.90) | 0.84 (0.82 to 0.92) | 16.46 (16.46 to 16.50) | 13.91 (13.53 to 15.16) |

The ratios are the numbers to cite: the graph of child nodes is 1.83 to 1.91 times slower than plain launches (medians of the three runs for 11 to 1191 crops; the largest single value is 2.03; 1.87 to 1.90 at 1191 crops), the graph of one capture is 0.82 to
0.94 times the time of plain launches (medians; 0.82 to 0.92 at 1191 crops, i.e. faster; the largest single value is 1.02, at 11 crops), and the fused kernel is 16.5 times faster than plain launches and 13.5 to 15.2 times faster than the graph of one capture at 1191 crops. The graph of one capture is the series with the larger
spread between runs (3.72, 4.15 and 3.79 ms at 1191 crops); the other series stay within 1 percent. In the probe of `../hip-graphs/` (a busy GPU, five structures including a fork and join over one stream per crop) the graph
of one capture takes 2.89 ms against 3.64 ms for the plain launches at 1191 crops; here the plain launches take 4.50 ms because the clocks are low for this benchmark: in the first 75 s of each run the graphics clock is 1.85 GHz
on average (maximum 2.07 GHz), the memory clock mostly 772 to 875 MHz, the power 25 W of 132 W and the GPU busy 66 to 68 percent. Over the whole run the graphics clock averages 1.6 GHz and the memory clock is 96 or 456 MHz most of the
time (the GPU idles between points and sweeps). With the four earlier runs, seven runs of this benchmark agree on the speedup (iterated over fused, 16.3 to 16.9 at 1191 crops, 14.3 to 14.9 at 301) and on the graph ratios.

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
