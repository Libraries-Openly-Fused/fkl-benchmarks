# CUDA Graphs baselines of E0 on the RTX 4090: the structure of the graph and what the other structures cost

Question (fkl-bench-coordinator, 2026-10-31): the Graphs series of E0 times the baseline as a CUDA graph. Does the structure of that graph matter, and is it the fastest structure a user would write
without changing the kernels? Everything below was measured on one machine: RTX 4090 (AD102, 128 SMs, SM clock 2685 MHz in every measured window, driver 610.43.02), Ryzen 9 5950X, CUDA 13.4.3
(nvcc 13.4.92), FKL c8fcbe8 (LTS-C++17), GCC 15.2 as host compiler, `-std=c++17`. The programs ran in quiet windows: pinned to cores 0-3,16-19, nothing else running (a load logger on core 13 every 20 s and a
clock sampler on core 14 every 0.1 s wrote `load.log` and `gpu_clocks.csv` next to every result).

## Which E0 program builds which graph (from the code)

| experiment | program | how the graph is built | structure |
|---|---|---|---|
| VF+HF, `mul_add` (50 crops of 60x120 of a uchar image, 2 + 100 k operations per crop) | `opencv-cuda/` launcher and `external/cvGPUSpeedup` | one stream capture per crop, each captured graph added to the parent with `cudaGraphAddChildGraphNode`, no dependencies between them | child |
| VF, `mul_add1` and `mul1` (4096x2160 image, GPU bound) | the same launcher | one capture of the launches of all the crops on one stream | flat serial |
| HF | `fkl-self/fusion/benchmark_horizontal_fusion.h`, `..._CUDAGraphs` | one capture per crop, child node per crop (crops 1 + 10 k up to 1191, 60x120 of 3840x2160) | child |
| HF, added for this check | the same file, `..._CUDAGraphs_OneCapturedGraph` | one capture of all the launches on one stream | flat serial |
| instantiate times | `opencv-cuda/graphs/graph_instantiate_time.cu` | as the VF+HF launcher | child |

## The structures

The work of one crop is a callable `enqueue(crop, stream)` that launches its kernels; every structure is built from the same callable (`common/graph_structures.h`):

- `plain`: the launches of all the crops on one stream, no graph (the "Iterated Batch" of the plain series);
- `child`: the structure of the experiments: one capture per crop, one child graph node per crop, independent;
- `child_chained`: the same child nodes, each depending on the previous one;
- `flat_serial`: one capture of the launches of all the crops on one stream: only kernel nodes, one chain;
- `flat_parallel`: one capture with a fork and a join over one stream per crop: only kernel nodes, independent.

## Method

Probes: `opencv-cuda/graphs/graph_structure_probe.cu` (the OpenCV-CUDA work of VF+HF, 50 chains of 2 + 100 k operations) and `fkl-self/graphs/hf_graph_structure_probe.cu` (the FKL kernels of HF). Per structure and
point: warm-up of at least 2 s and until two consecutive windows of 0.25 s agree within 2 percent (cap 10 s); then 100 timed launches (30 from 10,002 operations); the time is the GPU time between two
events recorded around one launch on the stream (what E0 reports), next to the host time of the launch call and the total including the synchronisation. Every structure is compared bit by bit with the output
of `plain` (all passed). Capture and instantiate times are of one build per point (`cudaGraphInstantiate(&exec, graph, NULL, NULL, 0)` as in the experiments). Tables: `structure_summary_vf.md/.csv`,
`structure_summary_hf.md/.csv`; raw: `full/`, `quick/`, `hf120/`.

## VF+HF: the child-node graph is the fastest structure

GPU time per kernel node (kernel nodes = 50 x (operations + 1)), 102 to 19,902 operations: `plain` 1.96 us, `child` 0.241 to 0.257 us, `flat_parallel` 0.242 to 0.257 us, `child_chained` and `flat_serial` 1.016 to 1.021 us.
`child` is 7.6 to 8.1 times faster than `plain` and 3.95 to 4.22 times faster than `flat_serial`; `child` and `flat_parallel` agree to 0.3 percent and `child_chained` and `flat_serial` to 0.1 percent at every
point from 502 operations (`structure_summary_vf.md`). The probe reproduces the experiment: the `child` time over the `OpenCV MeanTime` of the E0 run is 0.996 to 1.005 from 102 operations
(`vf_speedups/graph_speedups.txt`).

Speedup of the fused kernel (ratio of the means, `cvGS MeanTime` of the E0 run in the denominator), at 17,702 operations: against `child` 2,550x (E0 column: 2,554x), against `flat_serial` 10,089x, against `plain`
19,432x. Whole sweep from 2,502 operations: plateau 2,380x and maximum 2,554x (17,702 operations) for the child graph of the experiment; the other structures are slower, so against the fastest structure the plateau
and the maximum are those values (`vf_speedups/`).

## Does CUDA flatten child graph nodes?

Measured, not read from documentation: `child` takes the time of `flat_parallel` (kernel nodes without dependencies) and `child_chained` the time of `flat_serial` (a chain of kernel nodes), at every point of
VF+HF to 0.3 and 0.1 percent. On the GPU timeline (Nsight Systems, `--cuda-graph-trace=node`, `nsys/`, `timeline_summary.csv`) the median gap between consecutive kernels in order of start is, for HF at 301 crops:
`child` 0.672 us, `flat_parallel` 0.704 us, `child_chained` 0.032 us, `flat_serial` 0.032 us, `plain` 2.048 us; for VF at 502 operations (node tracing slows the VF graphs, the numbers are of the timeline only): `child` 0.160,
`flat_parallel` 0.160, `child_chained` 0.192, `flat_serial` 0.192, `plain` 2.368 us. The largest number of kernels running at the same time on that timeline is 2 for the four graph structures of VF
(`plain` 1) and 1 for every structure of HF.

## Capture and instantiate (VF+HF, one build per point)

At 17,702 operations (885,150 kernel nodes): capture 1.15 s (`child`) against 0.79 s (`flat_serial`) and 0.80 s (`flat_parallel`); instantiate 2.67 s against 1.69 s and 1.66 s (`child_chained` 1.78 s). At 19,902
operations (995,150 nodes): instantiate 2.25 s against 1.88 s and 1.87 s. Per kernel node, instantiate takes about 1.9 us for the flat graphs at those two points and 2.3 to 3.6 us for the child graph over
the points from 5,002 operations (`structure_summary_vf.md`).

## HF: the graph of one capture is the fastest structure

`fkl-self/graphs/hf_graph_structure_probe.cu` runs the FKL kernels of the HF experiment (crops 60x120 of a 3840x2160 uchar image, SaturateCast, Mul, Sub, Div, one kernel per crop) in the five structures at all 120 batch sizes
(1, 11, ..., 1191; `hf120/run1`; `hf120/run2` and `run3` repeat `child` and `flat_serial`). At 1,191 crops (ms, GPU time of one launch): `plain` 1.976, `child` 1.221, `child_chained` 0.9926, `flat_serial` 0.9926, `flat_parallel` 1.209, that is
1.659, 1.025, 0.833, 0.833 and 1.015 us per crop (`hf120_speedups/hf_graph_speedups.csv`). `flat_serial` and `child_chained` agree to 0.15 percent at every batch size from 101 crops (2 percent at 1 crop). At 301 crops the host time of the launch call is 498 us for `plain` (GPU time 499 us),
306 us for `child` (GPU 308 us), 306 us for `flat_parallel` (GPU 308 us) and 2.1 us for `flat_serial` (GPU 252 us) (`hf120/interleave/`): `plain`, `child` and `flat_parallel` take as long as the host needs to submit them, the chain is bound by the GPU. At most one kernel
runs at a time in every structure of HF (Nsight Systems, 301 crops, `nsys/`). Run to run, the three host-bound structures spread: over the three sweeps `child` has a spread (largest over smallest time at a batch size) with median 0.4 percent, 90th percentile 7.9 percent and
largest 14.5 percent (181 crops; 22 of 120 batch sizes above 5 percent), `flat_serial` has median 0.02 percent and largest 2.0 percent (`hf120_speedups/hf_graph_repeats.txt`). Launching the work of one crop and synchronising between two timed launches (the loop of the experiment does that
with the fused kernel) changes none of the structures (the ratio of the time with the work of a crop in between to the time without it is 0.991 to 1.009 at 101, 301, 661, 831 and 1,191 crops, except `flat_parallel` at 1,191 crops, 1.086).

## HF: the E0 loop with the graph of one capture, and the definition of the speedup

`fkl-self/fusion/benchmark_horizontal_fusion.h` has a third baseline next to the plain launches and the graph of child nodes: `benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_OneCapturedGraph`, one graph from one stream capture of all the launches (the HIP version is in pull
request #7 of fkl-benchmarks; the CUDA version is the commit b984d57 on top of it, `fkl-self/fusion/benchmark_horizontal_fusion.h`: the same function with the CUDA calls, its output compared bit by bit with the plain launches for every crop before the timing: the comparison
passed at all 120 batch sizes in the three runs). Three runs of the whole benchmark (`e0-loop-hf-flat/run1` to `run3`, 14:45 to 14:50 on 2026-11-01, binary `benchmark_horizontal_fusion_cu` sha256 db2432b2..., FKL c8fcbe8, nvcc 13.4.92, pinned to cores 0-3,16-19; no other process on any core
in the load log; the SM clock is 2685 MHz in the measured sweeps, except for the first 1.4 to 1.8 s of the measured sweep of each graph file, when it is still at 2550 MHz after the warm-up sweep: the graph of one capture is 4.4 to 5.1 percent above the probe at 11 to 131 crops (8 percent at 1 crop) and 1.6 percent above at 141 crops; from 151 crops the ratio is 0.995 to 1.000).

**Definition of the speedup (B).** The speedup of a baseline is the mean time of the baseline in its own E0 loop over the mean time of the fused kernel in the plain-launch loop of the same run (`benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_GPU_NVIDIA.csv`), at every batch size, the same for the three baselines.
The reason is in the next section: the time of the fused kernel in a loop depends on the baseline that precedes it, and the plain-launch loop is the one that all HF speedups of the 2025 paper used. Median of the three runs, smallest and largest run in brackets (`e0-loop-hf-flat/analysis/hf_table_definition_b.md`, `.csv` for all 120 batch sizes):

| crops | fused kernel, plain loop (us) | plain launches | graph of child nodes | graph of one capture (fastest) |
|--:|--:|--:|--:|--:|
| 11 | 3.18 | 6.24 (6.19 to 6.26) | 4.34 (4.27 to 4.40) | 3.61 (3.58 to 3.64) |
| 101 | 5.41 | 31.27 (31.09 to 31.39) | 19.51 (19.38 to 19.56) | 16.61 (16.57 to 16.73) |
| 301 | 9.99 | 50.05 (49.63 to 50.93) | 31.11 (31.01 to 31.30) | 25.22 (25.11 to 25.32) |
| 601 | 17.33 | 57.75 (57.34 to 57.84) | 35.71 (35.65 to 35.74) | 28.93 (28.78 to 28.94) |
| 661 | 18.78 | 58.39 (57.68 to 59.07) | 36.22 (35.96 to 36.45) | 29.34 (29.24 to 29.48) |
| 831 | 23.19 | 59.40 (59.39 to 59.76) | 36.97 (36.94 to 37.08) | 29.85 (29.80 to 29.94) |
| 1001 | 30.66 | 54.27 (53.45 to 54.45) | 33.67 (33.39 to 33.68) | 27.12 (27.10 to 27.20) |
| 1101 | 35.21 | 51.83 (51.17 to 52.32) | 32.10 (31.79 to 32.36) | 26.00 (25.73 to 26.06) |
| 1191 | 39.00 | 50.86 (50.11 to 51.09) | 31.61 (31.35 to 32.48) | 25.33 (25.27 to 25.50) |

Over the 120 batch sizes (median of the runs at each batch size): plain launches maximum 59.67 at 821 crops (runs 58.77 to 59.96), median over the sweep 53.75; graph of child nodes maximum 36.98 at 821 (36.71 to 37.03), median 33.27; graph of one capture maximum 29.85 at 831 (29.80 to 29.94), median 26.93.
The graph of one capture is the fastest structure in 358 of the 360 (run, batch size) pairs (the other two, at 1 crop, are the graph of child nodes), so the maximum and the median of the fastest structure are those of the graph of one capture. The column "Mean Speedup" of each file (the mean over the iterations of the ratio of the iteration) is within 0.2 percent of the ratio of the means used here
(`e0-loop-hf-flat/analysis/hf_e0_structure_table.csv` has both, and the same-loop speedups, in which each baseline is divided by the fused-kernel time of its own file).

## The time of the fused kernel depends on the loop that times it

In the loop of the experiment (`TimeMarkerTwo`, `fkl-self/fkBenchmarksCommon.h`) the baseline is timed, the host synchronises, and the fused kernel is timed from a new start event: `record(start); baseline; record(stop); synchronize(stop); first = elapsed(start, stop); record(start); fused; record(stop); synchronize(stop); second = elapsed(start, stop)`.
The fused kernel is the same in every loop, but its mean time is not: at 1,191 crops it is 38.8 us in the plain-launch file, 38.9 us in the file of the graph of child nodes and 35.1 us in the file of the graph of one capture (the three runs agree to 1 percent). `fkl-self/graphs/hf_fused_time_probe.cu` runs that loop with each structure of the baseline
and the fused kernel on its own (`fused-time/run1` to `run3`, window of 15:03 to 15:07; `fused-time/nsys/` for the GPU duration; `fused-time/analysis/hf_fused_time_summary.txt`). Interval of the fused kernel, median of the three runs: "alone" in microseconds, the loops as percent over "alone", the last two columns in microseconds ("alone" is the loop with an empty baseline: the stream is idle after a host synchronisation, then record, launch, record, synchronise):

| crops | alone | after plain launches | after child nodes | after child nodes in a chain | after one capture | after fork and join | 10 launches in a row (per launch) | queued behind a 1 ms kernel |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 101 | 5.41 | +2 % | +6 % | +1 % | +1 % | +6 % | | |
| 301 | 9.99 | 0 % | +4 % | 0 % | 0 % | +4 % | | |
| 661 | 18.43 | +3 % | +7 % | +3 % | +3 % | +7 % | 15.39 | 17.54 |
| 831 | 22.76 | +3 % | +5 % | +3 % | +2 % | +5 % | | |
| 1001 | 26.85 | +14 % | +16 % | +6 % | +4 % | +16 % | 22.49 | 25.25 |
| 1101 | 29.24 | +20 % | +21 % | +7 % | +4 % | +21 % | | |
| 1191 | 31.39 | +21 % | +24 % | +8 % | +7 % | +24 % | 26.45 | 29.51 |

At 1,191 crops: GPU duration of the fused kernel from Nsight Systems 27.2 us alone, 26.8 us back to back, 28.9 to 29.3 us after the graph baselines (32.8 us after the plain launches, in the profiled run); host time of the call that launches it 4.0 us alone and 4.1 to 4.8 us in the loops
(10.5 us after a 64 MB host memory sweep). The extra time of the loops over "alone" is therefore neither the duration of the launch call nor the duration of the kernel: after the graph of child nodes and after the fork and join capture about 5 us of the interval lies between the call and the start of the kernel, after the chain, the graph of one capture and alone 0.2 to 0.7 us.
A 2 MB host memory sweep before the launch adds 0.5 to 1 us, a 64 MB sweep about 6 us; reading the parameters of the fused kernel on the host before the launch changes nothing (38.8 against 39.05 us). The component of the submission path is not identified.

Speedups with three fused-kernel times (baseline time in its own E0 loop; A: fused time of the same loop, B: of the plain-launch loop, C: of the fused kernel alone, measured at the seven crop counts of the probe), graph of one capture / graph of child nodes / plain launches:

| crops | A | B | C |
|--:|--:|--:|--:|
| 661 | 29.2 / 34.5 / 58.4 | 29.3 / 36.2 / 58.4 | 29.8 / 36.9 / 59.5 |
| 831 | 29.8 / 36.0 / 59.4 | 29.8 / 37.0 / 59.4 | 30.4 / 37.6 / 60.5 |
| 1001 | 28.9 / 33.4 / 54.3 | 27.1 / 33.7 / 54.3 | 31.0 / 38.4 / 61.9 |
| 1101 | 28.9 / 32.1 / 51.8 | 26.0 / 32.1 / 51.8 | 31.2 / 38.6 / 62.4 |
| 1191 | 28.2 / 31.5 / 50.9 | 25.3 / 31.6 / 50.9 | 31.5 / 39.1 / 63.2 |

Up to about 900 crops the three agree within 3 percent. C exists at 101, 301, 661, 831, 1001, 1101 and 1191 crops only; at those the maxima under C are at 1,191 crops (graph of one capture 31.48, child nodes 39.08, plain launches 63.18), under B at 831 crops (29.85, 36.97, 59.40).
B is the definition of the table above: it is the one reference that does not depend on the baseline, and it is at the cold-host end of the range: above about 1,000 crops the launch cost of the fused kernel depends on what the host and the graph did before by up to 20 percent, so the speedups there are conservative.

## VF+HF: the same dependence of the cvGS kernel, and the numbers under B

`opencv-cuda/graphs/vf_fused_time_probe.cu` does the same for the VF+HF experiment (50 crops of 60x120, the fused launch is the `VerticalFusion<...>::execute` of the experiment; `vf-fused-time/run1` to `run3`, window of 15:20 to 15:31). Interval of the cvGS kernel in microseconds, median of the three runs
(the loops of the probe reproduce the cvGS column of the E0 files to 1.6 percent: plain pass 6.75, 16.13, 51.93, 89.87; Graphs pass 7.10, 17.61, 53.74, 89.27):

| operations | alone | after plain calls | after the graph of child nodes | after one capture | 50 launches in a row (per launch) | after a 64 MB host sweep | GPU duration (Nsight Systems, alone) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 502 | 6.27 | 6.72 | 7.06 | 8.08 | 4.51 | 10.57 | |
| 2,502 | 15.37 | 16.39 | 17.47 | 18.32 | 13.53 | 20.19 | 13.28 |
| 10,002 | 49.49 | 51.86 | 53.28 | 55.33 | 49.01 | 54.92 | |
| 17,702 | 85.78 | 89.03 | 89.37 | 93.79 | 86.56 | 91.06 | 83.42 |

**Which cvGS time each figure uses.** In the E0 files every pass divides by the cvGS time of its own loop: the plain pass (`benchmark_vertical_fusion_loopMulAdd.csv`, the corrected series of `e0-vf-mul-add-corrected/`) divides the OpenCV-CUDA time of plain calls by the cvGS time measured in that loop; the Graphs pass (`..._CUDAGraphs.csv`) divides the OpenCV-CUDA
time of the graph of child nodes by the cvGS time measured in the Graphs loop (A). Under definition B the plain pass is the same (it is the plain-launch loop), and the Graphs pass divides the time of its graph by the cvGS time of the plain pass of the same run; the cvGS time of the Graphs pass over that of the plain pass has median 1.000 from 10,002 operations (mean 1.001, range 0.956 to 1.077)
and about 1.05 below it. All ratios here are ratios of the mean times ("OpenCV MeanTime" over "cvGS MeanTime"); the column "Mean Speedup" of the files (the mean of the ratio of each iteration) is the other statistic and is given where it differs in the fourth digit:

| | plain pass (A = B) | Graphs pass, A (cvGS of the Graphs pass) | Graphs pass, B (cvGS of the plain pass) |
|---|--:|--:|--:|
| plateau: mean over the 100 points from 10,002 operations | 19,086 (column 19,091; median 19,163) | 2,501 (column 2,502) | 2,504 |
| maximum | 19,525 at 16,602 operations (column 19,529) | 2,554 at 17,702 (column 2,554) | 2,567 at 16,602 |
| fused kernel alone (C), at 10,002 and 17,702 operations | 19,812 and 20,224 (B: 18,877 and 19,305) | | 2,602 and 2,658 (B: 2,479 and 2,537) |

19,525 and 19,529 are the same maximum, at 16,602 operations of the corrected plain-pass file, in the two statistics: the ratio of the mean times (19,525.3) and the largest value of the column "Mean Speedup" (19,529.3); the same two statistics give the plateau 19,086 and 19,091. The original plain-pass file has its maximum at 12,702 operations (19,767), a point that the quiet re-measurement did not confirm (18,614).
Under C the plateau and the maximum are 4.8 to 4.9 percent above B at the two operation counts measured. VF-only (`mul_add1`, `mul1`, 4096x2160, the cvGS kernel takes 66 us to 4.5 ms): the Graphs-pass over the plain-pass cvGS time is 1.00 to 1.006 from 2,502 operations (1.03 at 502); the probe was not run for them.

## Files

`e0-loop-hf-flat/` (three E0 runs and `analysis/`), `fused-time/` and `vf-fused-time/` (the probes, their `analysis/` and `nsys/`), `hf120/` and `hf120_speedups/` (the probe at the 120 batch sizes), `full/`, `quick/`, `nsys/` (the probes of VF+HF and HF), `vf_speedups/`, `structure_summary_{vf,hf}.md/.csv`; every window folder has
`load.log` (every 20 s), `gpu_clocks.csv` (every 0.1 s), `processes-before.txt`, `processes-after.txt` and `binary-sha256.txt`. Programs: `opencv-cuda/graphs/graph_structure_probe.cu`, `opencv-cuda/graphs/vf_fused_time_probe.cu`, `fkl-self/graphs/hf_graph_structure_probe.cu`, `fkl-self/graphs/hf_fused_time_probe.cu`, `common/graph_structures.h`;
scripts: `scripts/graph_structure_summary.py`, `graph_speedup_recompute.py`, `hf_graph_speedup_recompute.py`, `hf_graph_repeats.py`, `graph_timeline.py`, `graph_timeline_all.py`, `fused_kernel_timeline.py`, `hf_fused_time_summary.py`, `hf_e0_structure_table.py`, `hf_table_definition_b.py`, `plot_hf_b.py`.
