# HIP Graphs on the RX 9060: why the graph of the horizontal fusion benchmark is twice as slow as plain launches

Question (fkl-bench-coordinator, 2026-10-31): the iterated batch of the horizontal fusion benchmark captured as a HIP graph (`benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_HIPGraphs`) is about twice as slow as the same launches
in the stream (1.78 to 2.06 times in four runs of `../fkl-self/`). Where does the time go? A bounded check (about 40 minutes of work), no tuning, no change of the library. FKL c8fcbe8, ROCm 10.1.0, amdclang++ 24.0.0,
gfx1200, idle machine, governor `powersave`.

## The graph of the benchmark

For each crop a `fk::Stream` captures ONE launch of the per-crop kernel (`hipStreamBeginCapture`, `hipStreamCaptureModeGlobal`), and `hipGraphAddChildGraphNode` adds that captured graph to the main graph with no
dependencies: the main graph has BATCH child graph nodes with one kernel node each. The benchmark times `hipGraphLaunch` with HIP events.

## The probe (`graph_probe.cpp`)

The same per-crop kernel (crop 60x120 of a 3840x2160 uchar image, saturate, multiply, subtract, divide, write float) launched four ways for BATCH crops: `plain` (BATCH launches in the stream), `children` (the graph of the
benchmark), `children_seq` (the same with every child depending on the previous one) and `flat` (ONE stream capture of the BATCH launches: a graph of BATCH kernel nodes in a chain). Per strategy and batch, the median of 100
repetitions after a warm-up of 1.5 s of: `submit` (the host time until the launch call returns), `total` (submit + stream synchronisation) and `event` (HIP events around the submission, as in the benchmark). The graph
building is not timed. Two runs (`probe1.txt`, `probe2.txt`) agree within 1 percent. Total time in ms (run 1):

| crops | plain launches | benchmark graph (child nodes) | child nodes in a chain | flat graph (one capture) | benchmark graph / plain | flat / plain | host time of hipGraphLaunch, benchmark graph | host time of hipGraphLaunch, flat graph |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 11 | 0.0624 | 0.0963 | 0.0990 | 0.0572 | 1.54 | 0.92 | 0.0321 | 0.0053 |
| 51 | 0.1847 | 0.3415 | 0.3479 | 0.1561 | 1.85 | 0.85 | 0.1386 | 0.0053 |
| 101 | 0.3312 | 0.6450 | 0.6546 | 0.2785 | 1.95 | 0.84 | 0.2712 | 0.0055 |
| 201 | 0.6147 | 1.2490 | 1.2624 | 0.5157 | 2.03 | 0.84 | 0.5352 | 0.0061 |
| 301 | 0.9100 | 1.8638 | 1.8703 | 0.7524 | 2.05 | 0.83 | 0.7947 | 0.0070 |
| 601 | 1.7886 | 3.6848 | 3.6894 | 1.4591 | 2.06 | 0.82 | 1.5831 | 0.0091 |
| 1191 | 3.5491 | 7.2977 | 7.2892 | 2.8530 | 2.06 | 0.80 | 3.1097 | 0.0128 |

Least squares fit over 101 to 1191 crops (microseconds per crop, both runs): plain 2.96 (host submit 1.15), benchmark graph 6.10 to 6.12 (host submit 2.60), child nodes in a chain 6.09 to 6.14, flat graph 2.36 to 2.39 (host submit 0.01,
i.e. 13 microseconds for the whole launch); the intercepts are 0.02 to 0.04 ms for all of them.

## What the numbers say

- The benchmark graph reproduces the benchmark: 7.30 ms against 3.55 ms at 1191 crops (2.06 times).
- The cost is per child graph node and linear in the number of nodes: 6.1 microseconds per crop against 2.96 for a plain launch and 2.4 for a kernel node of a flat graph, i.e. about 3.7 microseconds more per child node than per
  kernel node. The kernels themselves take 1.0 to 1.6 microseconds (rocprofv3 `--kernel-trace`, `trace-summary.txt`); the rest is the runtime's work around each of them.
- Both the host and the GPU timeline pay it. `hipGraphLaunch` of the benchmark graph takes 2.6 microseconds per child on the host (3.11 ms at 1191 crops, 43 percent of the total: the launch is not O(1)); after it returns the GPU needs
  about 4.2 ms more. On the GPU timeline the profiler shows a gap of 7.2 microseconds (median; p90 17 microseconds) between the kernels of consecutive child graphs, against 3.8 microseconds between plain launches and kernels that overlap
  (median gap -0.9 microseconds) in the flat graph.
- The dependencies and the order do not matter: `children_seq` is the same as `children`. The number of kernels does not matter either: the flat graph runs the same BATCH kernels faster than plain launches (2.85 ms against 3.55 ms at
  1191 crops, 0.80 times; its launch costs the host 13 microseconds whatever the batch).
- So on ROCm 10.1 a child graph node costs about 3.7 microseconds more than a kernel node, and a graph made of one child per launch is slower than plain launches. A graph of kernel nodes (here from one stream capture of all the launches)
  is faster than plain launches by 20 percent at 1191 crops and has an O(1) host cost. The time of the benchmark's graph is therefore a property of how it is built (child graph nodes), not of graphs.

## Not investigated

The capture mode (global as in the benchmark; the instantiated graph does not depend on it), other ROCm versions, the internal reason for the cost of a child graph node in the runtime, and the CUDA Graphs variant on NVIDIA GPUs (where child
graph nodes may be flattened). Absolute times depend on the clocks of the GPU (see `../fkl-self/README.md`, horizontal fusion); the ratios do not (four runs of the benchmark: 1.78 to 2.06; this probe: 2.05 to 2.06 at 1191 crops).
