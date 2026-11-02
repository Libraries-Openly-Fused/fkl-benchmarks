# HIP Graphs on the RX 9060: why the graph of the horizontal fusion benchmark is twice as slow as plain launches

Question (fkl-bench-coordinator, 2026-10-31): the iterated batch of the horizontal fusion benchmark captured as a HIP graph (`benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_HIPGraphs`) is about twice as slow as the same launches
in the stream (1.78 to 2.06 times in four runs of `../fkl-self/`, 1.87 to 1.90 times at 1191 crops in the three runs with the new variant). Where does the time go? A bounded check (about 40 minutes of work, extended
with the parallel structure on 2026-11-01), no tuning, no change of the library. FKL c8fcbe8, ROCm 10.1.0, amdclang++ 24.0.0, gfx1200, idle machine, governor `powersave`.

## The graph of the benchmark

For each crop a `fk::Stream` captures ONE launch of the per-crop kernel (`hipStreamBeginCapture`, `hipStreamCaptureModeGlobal`), and `hipGraphAddChildGraphNode` adds that captured graph to the main graph with no
dependencies: the main graph has BATCH child graph nodes with one kernel node each (`graph_probe` prints the node types: `children: 301 nodes (kernel 0, child graph 301, other 0)`). The benchmark times `hipGraphLaunch` with
HIP events.

## The probe (`graph_probe.cpp`)

The same per-crop kernel (crop 60x120 of a 3840x2160 uchar image, saturate, multiply, subtract, divide, write float) launched five ways for BATCH crops: `plain` (BATCH launches in the stream), `children` (the graph of the
benchmark), `children_seq` (the same with every child depending on the previous one), `flat` (ONE stream capture of the BATCH launches: a graph of BATCH kernel nodes in a chain, serial) and `flat_parallel` (ONE stream
capture with a fork and a join over one stream per crop, with events: BATCH kernel nodes and no dependencies between them, so that they may run side by side; the NVIDIA session pointed out that the serial chain
has no concurrency, and its 4090 measurement has the same five structures; that measurement is not on main yet and will be at `results/rtx4090/e0-graph-structure/`). The probe checks that every strategy gives the output of the plain launches (a hash of the bits of all the outputs, which are cleared before
each strategy runs): 28 of 28 checks passed in each of the six runs. Per strategy and batch, the median of 100 repetitions after a warm-up of 1.5 s of: `submit` (the host time of the launch call alone; what it was in `probe5` is in the correction below), `total`
(from before the first event record to the end of the stream synchronisation) and `event` (HIP events around the submission, as in the benchmark). The graph building is not timed. Two runs (`probe5_run1.txt`, `probe5_run2.txt`) agree within 2 percent in total and event time, and the median of four later runs of the corrected program (`probe6_run1.txt` to `probe6_run4.txt`) agrees with `probe5_run1.txt` within 0.6 percent in the 40 total and event cells at 101, 301, 601 and 1191 crops. Total time in ms
(run 1):

| crops | plain launches | benchmark graph (child nodes) | child nodes in a chain | flat serial (one capture) | flat parallel (fork and join) | benchmark graph / plain | flat serial / plain | flat parallel / plain |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 11 | 0.0631 | 0.0966 | 0.0991 | 0.0573 | 0.0569 | 1.53 | 0.91 | 0.90 |
| 51 | 0.1838 | 0.3391 | 0.3446 | 0.1541 | 0.1557 | 1.84 | 0.84 | 0.85 |
| 101 | 0.3330 | 0.6446 | 0.6513 | 0.2759 | 0.2800 | 1.94 | 0.83 | 0.84 |
| 201 | 0.6279 | 1.2530 | 1.2570 | 0.5144 | 0.5229 | 2.00 | 0.82 | 0.83 |
| 301 | 0.9256 | 1.8584 | 1.8625 | 0.7506 | 0.7650 | 2.01 | 0.81 | 0.83 |
| 601 | 1.8186 | 3.6738 | 3.6768 | 1.4598 | 1.4871 | 2.02 | 0.80 | 0.82 |
| 1191 | 3.6360 | 7.3449 | 7.3451 | 2.8924 | 2.9828 | 2.02 | 0.80 | 0.82 |

Host time of the launch call alone (`submit`, ms, median of the four runs `probe6_run1.txt` to `probe6_run4.txt`; **this is the table to use**):

| crops | plain | benchmark graph | child nodes in a chain | flat serial | flat parallel |
|--:|--:|--:|--:|--:|--:|
| 101 | 0.1229 | 0.2725 | 0.2732 | 0.0033 | 0.0558 |
| 301 | 0.3739 | 0.7941 | 0.7893 | 0.0048 | 0.1651 |
| 601 | 0.7337 | 1.5897 | 1.5778 | 0.0068 | 0.3313 |
| 1191 | 1.3584 | 3.1351 | 3.1453 | 0.0103 | 0.6766 |

**Correction.** The first version of this folder (`probe5_run1.txt`, `probe5_run2.txt` and the `submit` table that stood here) took `submit` from before the first `hipEventRecord` of the harness to after the second one: it measured the
launch call plus the two event records, while `graph_probe.cpp` and this README said "the launch call". Two `hipEventRecord` calls cost 1.56 microseconds on an idle stream (first line of each `probe6` file; four runs: 1.56 to 1.57) and
2.1 to 2.8 microseconds inside the loop of the probe (7 for the plain launches at 1191 crops): in the same runs, the earlier interval exceeds the launch call by that much in every cell at 101 to 1191 crops (the line "old definition of submit"
that each batch of the `probe6` files prints below its main line, minus the main line). It matters for the flat serial graph: 5.6, 7.0, 9.2 and 12.7 microseconds earlier at 101, 301, 601 and 1191 crops, 3.3, 4.8, 6.8 and 10.3 now (a change of
19 to 41 percent of the earlier value; spread of the four runs 2 to 5 percent). The other cells change by 0.1 to 3.8 percent of the new value, which is above the spread of the four runs in 4 of those 16 cells and immaterial; the per-crop slopes
of the total time agree within 1 percent, and the total and event times do not change. The earlier table below, for the record (the launch call plus the two event records, ms), is `probe5_run1.txt`, a different run: cell by cell it differs
from the table above by the event records and by the run-to-run noise (up to 24 microseconds at 1191 crops), so the two tables do not subtract to 2.1 to 2.8 microseconds. The cells of the new table are medians of four values; ties at the fifth
decimal are as Python's round() prints them.

| crops | plain | benchmark graph | child nodes in a chain | flat serial | flat parallel |
|--:|--:|--:|--:|--:|--:|
| 101 | 0.1250 | 0.2728 | 0.2752 | 0.0056 | 0.0578 |
| 301 | 0.3824 | 0.7965 | 0.7956 | 0.0070 | 0.1679 |
| 601 | 0.7394 | 1.6094 | 1.5901 | 0.0092 | 0.3336 |
| 1191 | 1.3738 | 3.1487 | 3.1697 | 0.0127 | 0.6782 |

Least squares fit over 101 to 1191 crops (microseconds per crop, both runs of `probe5`; the host submit is the fit of the four runs of `probe6`): plain 3.02 to 3.03 (host submit 1.13 to 1.14), benchmark graph 6.07 to 6.15 (host submit 2.62 to 2.63), child nodes in a chain 6.07 to 6.14, flat serial 2.37 to
2.40 (host submit 6.4 nanoseconds per crop plus 2.8 microseconds: 10 microseconds for the whole launch at 1191 crops), flat parallel 2.43 to 2.48 (host submit 0.57); the intercepts of the total time are 0.01 to 0.04 ms for all of them.

## What the numbers say

- The benchmark graph reproduces the benchmark: 7.34 ms against 3.64 ms at 1191 crops (2.02 times).
- The cost is per child graph node and linear in the number of nodes: 6.1 microseconds per crop against 3.0 for a plain launch and 2.4 for a kernel node of a flat graph, i.e. about 3.7 microseconds more per child node than per
  kernel node. The kernels themselves take 1.0 to 1.6 microseconds (rocprofv3 `--kernel-trace`, `trace-summary.txt`); the rest is the runtime's work around each of them.
- Both the host and the GPU timeline pay it. `hipGraphLaunch` of the benchmark graph takes 2.6 microseconds per child on the host (3.14 ms at 1191 crops, 43 percent of the total: the launch is not O(1)); after it returns the GPU
  needs about 4.2 ms more. On the GPU timeline the profiler shows a gap of 7.2 microseconds (median; p90 17 microseconds) between the kernels of consecutive child graphs, against 3.8 microseconds between plain launches and kernels that
  overlap (median gap -0.9 microseconds) in the flat graph.
- The dependencies and the order do not matter: `children_seq` is the same as `children`. The number of kernels does not matter either: the flat serial graph runs the same BATCH kernels faster than plain launches (2.89 ms
  against 3.64 ms at 1191 crops, 0.80 times; its launch costs the host 3 microseconds at 11 crops and 10 at 1191, about 6.4 ns per node plus 2.8 microseconds; an earlier version of this README said 13 microseconds whatever the batch because the interval included the two event records).
- Concurrency does not help: the flat parallel graph (no dependencies between the crops) takes 2.98 ms at 1191 crops, 0.82 of the plain launches and 3 percent MORE than the serial chain; its launch costs the host 0.57 microseconds
  per crop (0.68 ms at 1191 crops) because the branches are launched on several streams. The kernels are tiny (1.0 to 1.6 microseconds) and the execution is bound by the dispatch of one kernel after another, not by what
  runs side by side.
- So on ROCm 10.1 a child graph node costs about 3.7 microseconds more than a kernel node, and a graph made of one child per launch is slower than plain launches. A graph of kernel nodes (here from one stream capture of all
  the launches, serial or with a fork and a join) is faster than plain launches by about 20 percent at 1191 crops. The time of the benchmark's graph is therefore a property of how it is built (child graph nodes), not of graphs.

## Build and run the probe

From the root of fkl-benchmarks, with the submodule `external/fkl` at FKL `c8fcbe8` (the pin of these results) and ROCm 10.1 (`amdclang++` of `/opt/rocm/core-10.1/bin`; no environment variable is needed):

```
/opt/rocm/core-10.1/bin/amdclang++ -x hip --offload-arch=gfx1200 -O3 -DNDEBUG -std=c++17 -D__HIP_PLATFORM_AMD__=1 -ftemplate-depth=1000 -DENABLE_BENCHMARK \
    -I external/fkl/include results/rx9060/hip-graphs/graph_probe.cpp -o graph_probe
./graph_probe [repetitions [batch]]
```

Without arguments it runs 100 repetitions of the batches 11, 51, 101, 201, 301, 601 and 1191 (the output of the `probe5` and `probe6` files); `batch` can be 101, 301 or 1191 only, any other value prints a message and exits with code 1.
`PROBE_VARIANT=plain|children|children_seq|flat|flat_parallel` runs one strategy only (what the profiler runs of `trace-summary.txt` used) and `PROBE_WARMUP=<seconds>` changes the warm-up (default 1.5 s).

`probe5_run1.txt` and `probe5_run2.txt` were made with the first version of `graph_probe.cpp`, in which `submit` included the two event records (see the correction above); the usage line and the check of the batch of `main()` were
added after them. `probe6_run1.txt` to `probe6_run4.txt` (four runs on the idle machine, no CI job, 2026-11-01) were made with the program of this folder before it got the two checks that run outside the timed code: the stop on a failed
output check (exit code 1, before the strategy is timed), the check that the repetitions are at least 1 and the check of the batch argument, which now comes before the measurement of the event records. The timed code (`timeIt`)
differs from the first version only by the time stamps around the launch callback and by the second interval that is printed as the old definition of `submit`.

## Not investigated

The capture mode (global as in the benchmark; the instantiated graph does not depend on it), other ROCm versions, the internal reason for the cost of a child graph node in the runtime, and the CUDA Graphs variant on NVIDIA GPUs
(see the NVIDIA session's results for the 4090, which are not on main yet and will be at `results/rtx4090/e0-graph-structure/`). Absolute times depend on the clocks of the GPU (see `../fkl-self/README.md`, horizontal fusion); the ratios do not (the benchmark: 1.78 to 2.06; this probe: 2.00 to 2.02 at 1191 crops).
