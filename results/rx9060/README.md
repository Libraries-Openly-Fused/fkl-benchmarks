# RX 9060 (gfx1200), ROCm 10.1: how the GPU results are measured

Machine: Ryzen 7 5800X3D, 60 GiB, AMD Radeon RX 9060 (gfx1200, 28 CUs, wave32, 8144 MiB), Linux 7.0.0-38-generic, ROCm 10.1.0 (amdclang++ 24.0.0), amdgpu 7.1.9;
CPU governor `powersave` (amd-pstate-epp, EPP balance_performance, boost on); the machine is also the self-hosted ROCm CI runner, so every run waits for the
runner to be idle (no `Runner.Worker`, load average below 0.8).

## The graphics clock of the GPU decides the result of a short run

A GPU that was at rest does not run at its working clock when work arrives. On this RX 9060 (sysfs `hwmon/freq1_input`, sampled at 200 Hz,
`scripts/gpu_clock_trace.py`) the clock goes from 31 MHz at rest to about **2.0 GHz, stays there for the first 0.7 s of continuous work, reaches 3.0 GHz
after about 1 s**, overshoots to 3.3 to 3.5 GHz for a moment, and settles at 3.0 to 3.2 GHz (the value depends on the kernel that is running). The warm-up of 0.5 s
that the first results of this folder used ended on the ramp. What it did to the numbers, measured:

- The fused chain of pipeline (a) with the default (ASYMMETRIC) resize, batch 300, is **22 percent slower at 2.0 GHz** than at 3.0 GHz: total 0.1550 ms (device
  0.1328 ms) after 3 s without GPU work and a warm-up of 0.5 s; 0.1265 ms (device 0.1050 ms) with a warm-up of 2 s or 5 s. The same kernel with the HALF_PIXEL
  sampling had been timed second, after the ramp, and looked unchanged. In the first RPP run after the 0.5 s warm-up the default chain read 0.1146 / 0.1577 ms
  at batch 200 / 300 against 0.0924 / 0.1241 ms in an earlier run of the same binary: the ramp had been faster in that run by chance.
- In the other direction, a warm-up that ends right after the ramp can be measured in the overshoot: the NV12 to RGB kernel of E4b at one crop read 0.2602 ms
  with a warm-up of 0.5 s and 0.2808 ms at the steady clock (median of three repetitions; 7 percent too fast), and it varied with the number of crops (0.2602 to 0.2827 ms) although the kernel
  does not depend on it; with the steady warm-up it is 0.2808 to 0.2840 ms for 1 to 64 crops.
- The latency hiding series of `fkl-self` (the buffers are allocated and filled between two points, which leaves the GPU idle) was up to 52 percent slower
  without keeping the GPU busy before the timed runs of each point (301 operations: 6.10 ms against 4.01 ms).
- Nothing in FKL is involved: the benchmark binaries have identical device code at the pins 38c4f32, #411, #414, #416 and #417 of LTS-C++17 (c8fcbe8), and the
  times are the same at the five pins within 1 percent (`clock-ramp/`).

## The rule

For the pipelines (E4b and `rpp/`, `nv12::measure` in `fkl-self/pipelines/nv12_pipeline.h`): a warm-up of at least 2 s and until the mean time of two consecutive
windows of 0.25 s differs by at most 2 percent (cap 10 s), then 30 timed runs; the variants of a point are timed in a rotated order (the first one is variant
(index of the point + E5_ROTATE_OFFSET) mod the number of variants) and the whole run is repeated three times with the offsets 0, 1 and 2; the result is the median of
the three medians and the CSV columns `warmup_s`, `warmup_steady`, `sclk_start_mhz`, `sclk_timed_start_mhz` and `sclk_end_mhz` give the warm-up that was run
and the graphics clock at the start of the warm-up, at the start of the timed runs and after them. For the benchmarks of `fkl-self` whose GPU work is not
continuous from one point to the next (latency hiding): `FKLB_GPU_SPIN_SECONDS=2` runs the work of the point for 2 s before its timed runs (`fkBenchmarksCommon.h`).

## Folders

- `rpp-preprocessing/`: FKL against RPP 3.2.0, pipelines (a) and (b), at the final pin c8fcbe8 with the rule above (replaces the first run, which used a warm-up of 0.5 s
  at the pin 38c4f32 and is in the history of the branch).
- `e4b-nv12-steady-clock/`: the E4b NV12 pipeline of the RX 9060 with the rule above (c8fcbe8). `e4b-nv12/` is the first run (warm-up 0.5 s, pin 38c4f32); do not cite it.
- `fkl-self/`: horizontal fusion, instructions per operation and latency hiding (FKL only).
- `clock-ramp/`: the evidence of this section.
