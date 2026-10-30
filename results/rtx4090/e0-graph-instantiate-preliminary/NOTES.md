# E0: how far can the CUDA-Graphs baseline of the VF + HF sweep go?

`opencv-cuda/graphs/graph_instantiate_time.cu` builds the graph that `benchmark_vertical_fusionCompareMulAdd` uses as its
OpenCV-CUDA CUDA-Graphs baseline (50 crops, each captured as `convertTo` + NumOps/2 x (multiply, add) into a child graph)
and times capture and `cudaGraphInstantiate`. Preliminary run, RTX 4090, CUDA 13.4.3, driver 610.43.02, **while 16 compile
jobs and a flash-attn build saturated the CPU** (instantiate is CPU bound), so the times are upper bounds; the final record
of E0 is taken on an idle machine.

| NumOps per crop | kernel nodes | capture (s) | instantiate (s) | one launch (ms) |
|---:|---:|---:|---:|---:|
| 502 | 25,150 | 0.27 | 0.82 | 65 (first-use effects) |
| 2,002 | 100,150 | 0.77 | 1.43 | 30 |
| 5,002 | 250,150 | 1.20 | 2.34 | 74 |
| 10,002 | 500,150 | 3.98 | 9.42 | 137 |
| 15,002 | 750,150 | 8.58 | 14.07 | 215 |
| 19,902 | 995,150 | 7.20 | 18.20 | 317 |

Decision rule (plan E0): extend the Graphs series only while instantiate stays under a minute. It does, up to the last point of the
sweep (19,902 operations, about one million kernel nodes: 18 s), so the Graphs series covers the whole range of the OpenCV series
(the 2025 figure stopped near 5.4k operations). The device memory column of the CSV is noisy (other processes, lazy allocation) and
not used.
