# FKL-only benchmarks on the RX 9060 (gfx1200), ROCm 10.1: horizontal fusion, instructions per operation, latency hiding

`benchmark_horizontal_fusion_hip`, `benchmark_numinstructions_x_operation_hip` and `benchmark_latency_hiding_hip` (`fkl-self/`), the harness of
fkl-benchmarks main (9afddfb): horizontal fusion 120 points (batch 1, 11, ..., 1191; source 3840 wide x 2160 high, crops of 60x120), instructions per
operation 100 points (1, 6, ..., 496; 500 instructions in total), latency hiding 60 points (1, 21, ..., 1181 operations; the fixed version).
Thread fusion is not measured. Each CSV column is the mean, variance, maximum and minimum time in ms of the benchmark (the columns the harness writes;
there is no median in these files). Horizontal fusion has three CSVs: with the CPU overhead of the launches, without it, and `..._HIPGraphs_...`
(the iterated batch as one HIP graph per batch, the HIP version of the CUDA Graphs variant).

## What was built and run

- Commit of the headers: `fkl-benchmarks e0af273` (a local merge, `-dirty` absent: the tree was clean), FKL `38c4f32` (LTS-C++17 with the ResizeSampling policy).
- `e0af273` is the merge of main `9afddfb` and the head of fkl-benchmarks #1 (`hip-system-info`, `7b68ff4`). The difference to main is only the repair of the HIP part of
  the CSV header (system information of ROCm 10.1 / gfx1200): the benchmarks and the timings are those of main.
- amdclang++ 24.0.0 (ROCm 10.1.0), Release (`-O3`), gfx1200, host `-march=native`; driver and runtime 7.16.26385, amdgpu 7.1.9.31600000.
- Conditions: idle machine (no CI job on the runner, load below 0.8, GPU activity 0 % before the start), one benchmark after the other, no other process.
  CPU governor `powersave` (amd-pstate-epp, EPP balance_performance, boost on) during the runs.
- Wall time of a run: latency hiding 1 min, instructions per operation 4 min 49 s, horizontal fusion 13 min 37 s.

## Headline values (mean time, ms)

| horizontal fusion, batch | 1 | 101 | 301 | 601 | 1191 |
|---|--:|--:|--:|--:|--:|
| iterated batch (launches) | 0.0093 | 0.339 | 0.947 | 1.798 | 3.555 |
| fused batch | 0.0095 | 0.031 | 0.064 | 0.112 | 0.210 |
| iterated batch as a HIP graph | 0.0162 | 0.654 | 1.850 | 3.700 | 7.002 |
| fused batch (graphs file) | 0.0094 | 0.031 | 0.063 | 0.115 | 0.209 |

| instructions per operation | 1 | 6 | 101 | 251 | 496 |
|---|--:|--:|--:|--:|--:|
| separated operations | 966.0 | 162.4 | 10.34 | 6.63 | 7.74 |
| all operations fused | 5.90 | 5.88 | 5.95 | 5.96 | 5.86 |

| latency hiding, operations fused | 1 | 101 | 301 | 581 | 1181 |
|---|--:|--:|--:|--:|--:|
| time | 1.944 | 2.149 | 4.139 | 7.282 | 13.164 |

(On this GPU and ROCm 10.1 the iterated batch as a HIP graph is slower than with plain launches, about twice from batch 101 on; the cause was not investigated.)

## What the latency hiding harness runs (`fkl-self/latencyhiding/benchmark_latency_hiding.h`)

- Data type `float`; one 1D buffer of `3840 * 2160 * 8 = 66 355 200` elements (265 420 800 bytes, 253.1 MiB) as input and another as output, so each launch reads
  253 MiB and writes 253 MiB; the input is set to 1.0.
- The operation is a multiplication by 2 (`Mul<float>`), repeated N times (N = 1, 21, ..., 1181) in ONE fused kernel (`PerThreadRead` + N times `Mul` as nested
  `StaticLoop`s of 20 + the remainder + `PerThreadWrite`, `TransformDPP`, thread fusion at its default); a single `executeOperations` per point: one warm-up
  execution, then 100 timed executions (`ITERS`), each timed with HIP events on the stream. The values overflow to infinity for large N, which does not change the time.
- At one operation the time is 1.944 ms for 506 MiB moved, i.e. 273 GB/s (decimal) of read plus write traffic; the time grows from 101 operations on.

