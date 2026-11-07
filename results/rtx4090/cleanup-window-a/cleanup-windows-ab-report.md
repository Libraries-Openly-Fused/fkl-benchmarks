# Cleanup windows A and B, 2026-11-07: published, reference now and cleaned

Windows A (23:11:59-23:14:25) and B (23:14:25-23:19:22) on 2026-11-07, alternating two builds:
- **reference now:** build/local-rtx4090, fkl-benchmarks a5dd9fa;
- **cleaned:** main 836a98e, built by step 0 from a clean checkout.
Both builds use FKL 64ad9195, cvGPUSpeedup 6d26c65 and FastNPP a1b5e245. Every run exited 0, with no INCOMPLETE line and no -nan.

**Published**:
- NPP and CPU-time NPP: e0-npp-after45 and e0-cpu-time-npp-after45 (3 runs each, window of 2026-11-06, FKL c8fcbe8);
- CPU-time OpenCV: e0-cpu-time-opencv (one run, 2026-10-30);
- E9: e9-cross-library (3 runs, 2026-11-06, FKL c8fcbe8);
- E4: e4-preprocessing quiet-run1..3 (FKL c8fcbe8).

**How each number is computed:**
- Values are medians over the runs. The spread is (largest - smallest) / median.
- "Exceeds" means the change is larger than both spreads involved.
- FLAG means it exceeds them and is also above 3 % (GPU-bound) or 5 % (host-bound).
- Bound:
  - Host-bound when the host time of the same pipeline is at least 80 % of the event time. For the NPP GPU program, the host time comes from the CPU-time NPP program, interpolated in batch.
  - CPU-time programs and host columns are host-bound by definition.
  - Speedups get the 3 % limit.

## Summary

**Drift (reference now against published: pins, time, environment): within the thresholds everywhere except single-run noise.**
- NPP GPU program, CPU-time NPP and E9: no flag.
- CPU-time OpenCV: cvGS at batches 12, 52, 112 and 122, and the speedup at 122 and 142, differ by 5 to 8 % from the published file. That file is one run. At batch 122 the reference now is 373.1x, inside the range of the ten repeats of 2026-11-01 (368.7x to 387.7x).
- E4: three host-time points of about 2.3 us (fkl at 130 and 170) are 5.4 to 7.9 % higher. Everything else is within.

**Cleanup effect, by design** (Oscar's answers 1 and 3: everything that depends on the crop locations is rebuilt inside the timed call). The paper's numbers change:
- **E4a, FKL fused** (both conventions): gpu_mean_ms +8 % (batch 10) to +50 % (batch 300).
  - FKL's crop views, read and operations are now built in the call. The reference built them once before timing (a5dd9fa e4_preprocessing.cu l.300-305).
  - This costs about 30 ns per crop on the host: host_enqueue_us 2.8 -> 12.2 us at batch 300, and the event interval grows by the same amount.
  - Speedup against opencv_percrop, published -> reference now -> cleaned: batch 10: 36.4x -> 37.0x -> 34.0x; 100: 154.9x -> 152.5x -> 111.7x; 300: 205.6x -> 207.3x -> 138.1x.
  - Speedup against npp: 36.9x -> 37.3x -> 34.8x; 158.2x -> 153.2x -> 113.4x; 208.2x -> 209.4x -> 141.0x.
  - opencv_percrop and npp themselves are within (-0.2 to +1.2 %).
  - E4c (whole frames) is unchanged: 4K speedups 10.2x and 11.7x.
- **CPU-time OpenCV against cvGPUSpeedup:** cvGS host time +5 % (batch 2) to +84 % (batch 152); the OpenCV side is within.
  - The GpuMat ROIs are now made in every iteration. The reference made them once before the timed loop (a5dd9fa l.86-87). This costs about 25 ns per crop.
  - Speedup, reference now -> cleaned: batch 12: 73.3x -> 64.4x; 52: 238.7x -> 160.9x; 102: 349.6x -> 199.0x; 152: 431.3x -> 233.8x. This is the figure of the CPU-time comparison.

**Cleanup effect, not by design (small):**
- FastNPP got faster:
  - by 0.6 to 3.6 % at every batch of the NPP GPU program (GPU-bound by the rule above; above 3 % at batches 30, 40, 50, 60, 110 and 150), so the speedup is up 2.8 to 3.5 % where flagged;
  - 2.9 to 6.0 % less host time in the CPU-time NPP program at batches 12 to 82, and 0 to 2.5 % above that, so the speedup is +5.5 % at 12, +6.7 % at 52 and +5.5 % at 62 (host, flagged).
- Probable cause, not isolated: FastNPP builds its batched resize on the host from the descriptor arrays. In the reference these were pinned host arrays (cudaMallocHost) filled once at setup; in the cleaned build they are stack arrays written in the same call, so they are in the host caches.
- In the No_CPU_Overhead test, where FastNPP is built once in both builds, the change is smaller and mixed (+2.1 to -2.9 %, mostly negative), so part of it may be on the GPU side.

**Unchanged (within the spread or under the threshold):** NPP everywhere; FastNPP No_CPU_Overhead; E9 (M1, M2, R1, R2 launch_ms and graph_ms within ±0.9 %, 1 kernel per call, nsys 6 launches in the range in all three sets); E4 opencv_percrop and npp; E4c.

**E4 DRAM bytes at 4K (Nsight Compute, cleaned build, the published metrics and columns):**
- Cold, the ncu default:
  - fkl_fused 86.13 -> 86.80 MiB (+0.8 %), npp 1,281.9 -> 1,306.3 MiB (+1.9 %), opencv_percrop 1,084.3 -> 1,073.4 MiB (-1.0 %);
  - kernel counts 1, 9 and 7, unchanged;
  - ratios against fkl_fused: 12.6x -> 12.4x (OpenCV) and 14.9x -> 15.0x (NPP).
- Warm (--cache-control none):
  - fkl_fused 83.72 -> 114.14 MiB (+36.3 %, kernel 108 -> 154 us), npp +9.9 %, opencv_percrop +9.9 %;
  - ratios 14.1x -> 11.4x (OpenCV) and 16.8x -> 13.5x (NPP).
- The timed 4K means did not change: fkl_fused 0.1581 -> 0.1579 ms, npp 1.846 -> 1.849, opencv_percrop 1.604 -> 1.605. The warm pass measures one launch, so its DRAM traffic depends on the L2 and memory state that the program's setup leaves before that launch, and the setup changed: each side now allocates and fills its own frame and output.
- The cause is not isolated (measure() clears the output before the launch in both versions). Warm passes of both builds in one window would separate it. The cold numbers are robust.

**Agreement:** every data row of the cleaned runs is identical to the published run of the same index:
- NPP GPU: 30 + 30 rows;
- CPU-time NPP: 32 rows;
- E9: 18 rows;
- E4: 132 rows, the opencv_batch rows being absent from the cleaned build.
The CPU-time OpenCV check passed in every run; the cleaned build reads noise, the reference a constant image.

**FastNPP's one-time construction** (No_CPU_Overhead, new column, cleaned only): 0.0012 ms at batch 10 to 0.0058 ms at batch 150.

Every number is in cleanup-windows-ab-report.csv next to this file. The detailed tables follow.

## Changes above the spread and above the 3 % (GPU) or 5 % (host) threshold

- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 30 FastNPP (GPU): cleanup -3.6 % (reference now 0.006382, cleaned 0.006155; spreads 0.7 %, 2.8 %): probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup
- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 40 FastNPP (GPU): cleanup -3.3 % (reference now 0.00754, cleaned 0.007294; spreads 1.4 %, 2.8 %): probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup
- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 40 speedup (GPU): cleanup +3.5 % (reference now 69.49, cleaned 71.96; spreads 2.4 %, 2.7 %): follows FastNPP's time (see that row)
- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 50 FastNPP (GPU): cleanup -3.2 % (reference now 0.008341, cleaned 0.008076; spreads 2.4 %, 0.7 %): probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup
- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 50 speedup (GPU): cleanup +3.2 % (reference now 78.53, cleaned 81.05; spreads 2.8 %, 0.7 %): follows FastNPP's time (see that row)
- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 60 FastNPP (GPU): cleanup -3.2 % (reference now 0.009123, cleaned 0.008832; spreads 1.2 %, 0.7 %): probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup
- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 90 speedup (GPU): cleanup +3.4 % (reference now 98.04, cleaned 101.4; spreads 0.2 %, 1.2 %): follows FastNPP's time (see that row)
- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 110 FastNPP (GPU): cleanup -3.6 % (reference now 0.01385, cleaned 0.01335; spreads 1.4 %, 0.9 %): probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup
- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 110 speedup (GPU): cleanup +3.5 % (reference now 103.5, cleaned 107.1; spreads 2.0 %, 0.8 %): follows FastNPP's time (see that row)
- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 140 speedup (GPU): cleanup +3.3 % (reference now 108.1, cleaned 111.6; spreads 1.0 %, 0.6 %): follows FastNPP's time (see that row)
- NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame) / 150 FastNPP (GPU): cleanup -3.1 % (reference now 0.01763, cleaned 0.01709; spreads 0.4 %, 0.5 %): probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup
- CPU-time program, NPP against FastNPP (ms of host time per frame) / 12 speedup (host): cleanup +5.5 % (reference now 58.11, cleaned 61.32; spreads 2.0 %, 1.5 %): follows FastNPP's time
- CPU-time program, NPP against FastNPP (ms of host time per frame) / 52 speedup (host): cleanup +6.7 % (reference now 181.2, cleaned 193.3; spreads 4.5 %, 0.6 %): follows FastNPP's time
- CPU-time program, NPP against FastNPP (ms of host time per frame) / 62 speedup (host): cleanup +5.5 % (reference now 201.2, cleaned 212.3; spreads 2.5 %, 0.8 %): follows FastNPP's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 2 cvGS (host): cleanup +5.1 % (reference now 0.001788, cleaned 0.00188; spreads 3.6 %, 4.1 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 12 cvGS (host): drift -6.4 % (published 0.002247, reference now 0.002104; spreads one run, 4.3 %)
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 12 cvGS (host): cleanup +17.9 % (reference now 0.002104, cleaned 0.00248; spreads 4.3 %, 0.6 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 12 speedup (host): cleanup -12.2 % (reference now 73.34, cleaned 64.42; spreads 1.3 %, 3.9 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 22 cvGS (host): cleanup +25.5 % (reference now 0.002308, cleaned 0.002897; spreads 1.2 %, 1.9 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 22 speedup (host): cleanup -18.8 % (reference now 122.2, cleaned 99.2; spreads 1.5 %, 2.2 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 32 cvGS (host): cleanup +31.3 % (reference now 0.00245, cleaned 0.003216; spreads 1.5 %, 2.2 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 32 speedup (host): cleanup -23.5 % (reference now 165.4, cleaned 126.6; spreads 0.9 %, 3.2 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 42 cvGS (host): cleanup +39.9 % (reference now 0.002617, cleaned 0.003663; spreads 0.7 %, 4.4 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 42 speedup (host): cleanup -27.9 % (reference now 202.6, cleaned 146.1; spreads 1.3 %, 5.7 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 52 cvGS (host): drift -5.9 % (published 0.002926, reference now 0.002752; spreads one run, 0.4 %)
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 52 cvGS (host): cleanup +50.0 % (reference now 0.002752, cleaned 0.004129; spreads 0.4 %, 7.6 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 52 speedup (host): cleanup -32.6 % (reference now 238.7, cleaned 160.9; spreads 1.0 %, 7.1 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 62 cvGS (host): cleanup +49.8 % (reference now 0.003002, cleaned 0.004497; spreads 3.2 %, 5.2 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 62 speedup (host): cleanup -33.4 % (reference now 263.3, cleaned 175.4; spreads 3.1 %, 4.2 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 72 cvGS (host): cleanup +59.6 % (reference now 0.003103, cleaned 0.004951; spreads 1.3 %, 6.4 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 72 speedup (host): cleanup -36.3 % (reference now 293, cleaned 186.5; spreads 0.6 %, 5.3 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 82 cvGS (host): cleanup +55.6 % (reference now 0.003426, cleaned 0.005331; spreads 1.9 %, 5.9 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 82 speedup (host): cleanup -35.9 % (reference now 305.5, cleaned 195.9; spreads 2.3 %, 5.1 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 92 cvGS (host): cleanup +64.1 % (reference now 0.003471, cleaned 0.005698; spreads 2.7 %, 6.7 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 92 speedup (host): cleanup -39.1 % (reference now 336.9, cleaned 205.2; spreads 3.0 %, 7.0 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 102 cvGS (host): cleanup +75.5 % (reference now 0.003702, cleaned 0.006497; spreads 2.2 %, 10.1 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 102 speedup (host): cleanup -43.1 % (reference now 349.6, cleaned 199; spreads 2.7 %, 8.6 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 112 cvGS (host): drift -5.4 % (published 0.00402, reference now 0.003803; spreads one run, 1.6 %)
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 112 cvGS (host): cleanup +73.9 % (reference now 0.003803, cleaned 0.006615; spreads 1.6 %, 6.8 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 112 speedup (host): cleanup -42.2 % (reference now 370.9, cleaned 214.3; spreads 1.2 %, 6.8 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 122 cvGS (host): drift +8.1 % (published 0.003879, reference now 0.004192; spreads one run, 2.4 %)
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 122 cvGS (host): cleanup +70.5 % (reference now 0.004192, cleaned 0.007148; spreads 2.4 %, 6.2 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 122 speedup (host): drift -6.8 % (published 400.3, reference now 373.1; spreads one run, 1.8 %)
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 122 speedup (host): cleanup -41.5 % (reference now 373.1, cleaned 218.2; spreads 1.8 %, 6.7 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 132 cvGS (host): cleanup +79.8 % (reference now 0.004148, cleaned 0.007459; spreads 1.7 %, 5.0 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 132 speedup (host): cleanup -44.1 % (reference now 402.3, cleaned 224.9; spreads 2.9 %, 5.5 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 142 cvGS (host): cleanup +78.2 % (reference now 0.004496, cleaned 0.008012; spreads 1.4 %, 4.0 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 142 speedup (host): drift -5.3 % (published 419.5, reference now 397.2; spreads one run, 3.1 %)
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 142 speedup (host): cleanup -43.3 % (reference now 397.2, cleaned 225.2; spreads 3.1 %, 4.0 %): follows cvGS's time
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 152 cvGS (host): cleanup +83.9 % (reference now 0.004468, cleaned 0.008217; spreads 3.6 %, 8.3 %): the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop
- CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame) / 152 speedup (host): cleanup -45.8 % (reference now 431.3, cleaned 233.8; spreads 3.5 %, 8.1 %): follows cvGS's time
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 10 fkl_fused (GPU): cleanup +8.1 % (reference now 0.003499, cleaned 0.003783; spreads 6.0 %, 3.1 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 10 fkl_fused_half_pixel (GPU): cleanup +7.0 % (reference now 0.003544, cleaned 0.003791; spreads 2.8 %, 0.6 %): as fkl_fused
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 50 fkl_fused (GPU): cleanup +31.9 % (reference now 0.005706, cleaned 0.007524; spreads 0.7 %, 1.1 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 50 fkl_fused_half_pixel (GPU): cleanup +30.1 % (reference now 0.005779, cleaned 0.00752; spreads 1.2 %, 0.9 %): as fkl_fused
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 100 fkl_fused (GPU): cleanup +36.8 % (reference now 0.008332, cleaned 0.0114; spreads 0.4 %, 0.2 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 100 fkl_fused_half_pixel (GPU): cleanup +36.8 % (reference now 0.008369, cleaned 0.01145; spreads 4.1 %, 0.3 %): as fkl_fused
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 150 fkl_fused (GPU): cleanup +42.8 % (reference now 0.01083, cleaned 0.01547; spreads 0.7 %, 0.8 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 150 fkl_fused_half_pixel (GPU): cleanup +41.9 % (reference now 0.01096, cleaned 0.01555; spreads 17.2 %, 1.3 %): as fkl_fused
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 200 fkl_fused (GPU): cleanup +42.3 % (reference now 0.0135, cleaned 0.0192; spreads 1.0 %, 0.4 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 200 fkl_fused_half_pixel (GPU): cleanup +42.6 % (reference now 0.01347, cleaned 0.01921; spreads 0.3 %, 0.7 %): as fkl_fused
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 250 fkl_fused (GPU): cleanup +45.6 % (reference now 0.01592, cleaned 0.02317; spreads 0.3 %, 0.4 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 250 fkl_fused_half_pixel (GPU): cleanup +45.0 % (reference now 0.016, cleaned 0.02319; spreads 0.6 %, 0.8 %): as fkl_fused
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 300 fkl_fused (GPU): cleanup +49.5 % (reference now 0.01843, cleaned 0.02754; spreads 0.7 %, 0.3 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all / E4a 300 fkl_fused_half_pixel (GPU): cleanup +49.0 % (reference now 0.01849, cleaned 0.02756; spreads 0.4 %, 1.6 %): as fkl_fused
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 10 fkl_fused (host): cleanup +18.1 % (reference now 1.728, cleaned 2.04; spreads 4.3 %, 2.3 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 10 fkl_fused_half_pixel (host): cleanup +19.6 % (reference now 1.691, cleaned 2.022; spreads 1.7 %, 3.9 %): as fkl_fused
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 50 fkl_fused (host): cleanup +85.5 % (reference now 1.803, cleaned 3.345; spreads 4.2 %, 2.1 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 50 fkl_fused_half_pixel (host): cleanup +81.5 % (reference now 1.828, cleaned 3.317; spreads 9.8 %, 1.8 %): as fkl_fused
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 100 fkl_fused (host): cleanup +150.5 % (reference now 1.99, cleaned 4.987; spreads 3.0 %, 0.2 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 100 fkl_fused_half_pixel (host): cleanup +148.1 % (reference now 1.997, cleaned 4.954; spreads 3.9 %, 1.1 %): as fkl_fused
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 150 fkl_fused (host): cleanup +217.8 % (reference now 2.178, cleaned 6.922; spreads 2.7 %, 0.9 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 150 fkl_fused_half_pixel (host): cleanup +217.5 % (reference now 2.166, cleaned 6.878; spreads 2.9 %, 0.6 %): as fkl_fused
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 200 fkl_fused (host): cleanup +246.7 % (reference now 2.385, cleaned 8.269; spreads 0.7 %, 1.4 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 200 fkl_fused_half_pixel (host): cleanup +240.6 % (reference now 2.409, cleaned 8.207; spreads 1.4 %, 0.3 %): as fkl_fused
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 250 fkl_fused (host): cleanup +287.7 % (reference now 2.561, cleaned 9.928; spreads 1.1 %, 1.5 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 250 fkl_fused_half_pixel (host): cleanup +281.3 % (reference now 2.608, cleaned 9.946; spreads 3.0 %, 1.0 %): as fkl_fused
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 300 fkl_fused (host): cleanup +336.4 % (reference now 2.796, cleaned 12.2; spreads 2.0 %, 1.1 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it) / E4a 300 fkl_fused_half_pixel (host): cleanup +346.4 % (reference now 2.742, cleaned 12.24; spreads 2.3 %, 1.4 %): as fkl_fused
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 10 fkl_fused (host): cleanup +19.2 % (reference now 1.716, cleaned 2.045; spreads 1.4 %, 1.8 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 10 fkl_fused_half_pixel (host): cleanup +21.6 % (reference now 1.721, cleaned 2.092; spreads 1.9 %, 2.2 %): as fkl_fused
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 50 fkl_fused (host): cleanup +89.1 % (reference now 1.763, cleaned 3.334; spreads 2.2 %, 0.5 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 50 fkl_fused_half_pixel (host): cleanup +91.0 % (reference now 1.763, cleaned 3.367; spreads 3.2 %, 0.8 %): as fkl_fused
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 100 fkl_fused (host): cleanup +154.1 % (reference now 1.95, cleaned 4.956; spreads 1.8 %, 1.6 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 100 fkl_fused_half_pixel (host): cleanup +155.4 % (reference now 1.948, cleaned 4.975; spreads 1.3 %, 0.9 %): as fkl_fused
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 150 fkl_fused (host): cleanup +218.0 % (reference now 2.178, cleaned 6.927; spreads 5.8 %, 1.6 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 150 fkl_fused_half_pixel (host): cleanup +226.1 % (reference now 2.124, cleaned 6.926; spreads 4.4 %, 0.2 %): as fkl_fused
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 200 fkl_fused (host): cleanup +249.8 % (reference now 2.351, cleaned 8.223; spreads 1.5 %, 1.0 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 200 fkl_fused_half_pixel (host): cleanup +248.1 % (reference now 2.37, cleaned 8.248; spreads 2.3 %, 0.9 %): as fkl_fused
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 250 fkl_fused (host): cleanup +295.0 % (reference now 2.516, cleaned 9.939; spreads 2.1 %, 2.3 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 250 fkl_fused_half_pixel (host): cleanup +291.7 % (reference now 2.534, cleaned 9.928; spreads 1.8 %, 0.3 %): as fkl_fused
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 300 fkl_fused (host): cleanup +345.1 % (reference now 2.733, cleaned 12.16; spreads 3.9 %, 0.2 %): FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop
- E4 pre-processing, host_launch_us (100 calls back to back, wall time per call) / E4a 300 fkl_fused_half_pixel (host): cleanup +351.1 % (reference now 2.699, cleaned 12.18; spreads 2.0 %, 0.9 %): as fkl_fused
- E4: 182 flagged (metric, point) changes over all batches, listed at the end of the E4 tables

### NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame)

| batch, side | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| 10 NPP | host | 0.1331 (2.6 %) | 0.1329 (1.6 %) | 0.1322 (0.5 %) | -0.1 % | within | -0.5 % | within |  |
| 10 FastNPP | GPU | 0.004348 (2.0 %) | 0.00442 (1.4 %) | 0.004316 (1.0 %) | +1.7 % | within | -2.4 % | exceeds |  |
| 10 speedup | GPU | 30.96 (4.1 %) | 30.11 (1.7 %) | 30.95 (0.7 %) | -2.7 % | within | +2.8 % | exceeds |  |
| 20 NPP | host | 0.2616 (1.8 %) | 0.2646 (1.8 %) | 0.2611 (1.1 %) | +1.1 % | within | -1.3 % | within |  |
| 20 FastNPP | GPU | 0.005365 (1.5 %) | 0.005449 (1.5 %) | 0.005416 (2.0 %) | +1.6 % | exceeds | -0.6 % | within |  |
| 20 speedup | GPU | 49.28 (1.5 %) | 49.33 (0.6 %) | 49.07 (1.0 %) | +0.1 % | within | -0.5 % | within |  |
| 30 NPP | host | 0.3893 (0.9 %) | 0.3935 (0.4 %) | 0.3898 (2.3 %) | +1.1 % | exceeds | -1.0 % | within |  |
| 30 FastNPP | GPU | 0.006323 (0.2 %) | 0.006382 (0.7 %) | 0.006155 (2.8 %) | +0.9 % | exceeds | -3.6 % | exceeds FLAG | probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup |
| 30 speedup | GPU | 61.91 (0.4 %) | 62.27 (0.6 %) | 63.17 (3.1 %) | +0.6 % | within | +1.4 % | within |  |
| 40 NPP | host | 0.5187 (1.1 %) | 0.5217 (1.1 %) | 0.5228 (0.2 %) | +0.6 % | within | +0.2 % | within |  |
| 40 FastNPP | GPU | 0.00746 (1.6 %) | 0.00754 (1.4 %) | 0.007294 (2.8 %) | +1.1 % | within | -3.3 % | exceeds FLAG | probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup |
| 40 speedup | GPU | 69.63 (1.9 %) | 69.49 (2.4 %) | 71.96 (2.7 %) | -0.2 % | within | +3.5 % | exceeds FLAG | follows FastNPP's time (see that row) |
| 50 NPP | host | 0.6471 (0.4 %) | 0.6522 (0.4 %) | 0.6481 (1.0 %) | +0.8 % | exceeds | -0.6 % | within |  |
| 50 FastNPP | GPU | 0.008151 (3.1 %) | 0.008341 (2.4 %) | 0.008076 (0.7 %) | +2.3 % | within | -3.2 % | exceeds FLAG | probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup |
| 50 speedup | GPU | 79.78 (3.3 %) | 78.53 (2.8 %) | 81.05 (0.7 %) | -1.6 % | within | +3.2 % | exceeds FLAG | follows FastNPP's time (see that row) |
| 60 NPP | host | 0.777 (0.9 %) | 0.7825 (1.0 %) | 0.78 (1.2 %) | +0.7 % | within | -0.3 % | within |  |
| 60 FastNPP | GPU | 0.009047 (1.4 %) | 0.009123 (1.2 %) | 0.008832 (0.7 %) | +0.8 % | within | -3.2 % | exceeds FLAG | probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup |
| 60 speedup | GPU | 86.76 (1.0 %) | 86.42 (0.6 %) | 88.57 (0.6 %) | -0.4 % | within | +2.5 % | exceeds |  |
| 70 NPP | host | 0.9107 (1.3 %) | 0.9113 (1.1 %) | 0.9083 (0.9 %) | +0.1 % | within | -0.3 % | within |  |
| 70 FastNPP | GPU | 0.01015 (0.2 %) | 0.01013 (0.9 %) | 0.009956 (0.9 %) | -0.2 % | within | -1.8 % | exceeds |  |
| 70 speedup | GPU | 90.03 (1.2 %) | 90.5 (1.6 %) | 91.73 (0.4 %) | +0.5 % | within | +1.4 % | within |  |
| 80 NPP | host | 1.038 (2.1 %) | 1.039 (0.8 %) | 1.034 (1.5 %) | +0.0 % | within | -0.5 % | within |  |
| 80 FastNPP | GPU | 0.01119 (0.6 %) | 0.01113 (0.6 %) | 0.01105 (1.0 %) | -0.5 % | within | -0.8 % | within |  |
| 80 speedup | GPU | 93.28 (1.5 %) | 93.31 (0.9 %) | 94.04 (0.4 %) | +0.0 % | within | +0.8 % | within |  |
| 90 NPP | host | 1.164 (1.2 %) | 1.168 (0.4 %) | 1.164 (1.4 %) | +0.4 % | within | -0.4 % | within |  |
| 90 FastNPP | GPU | 0.01187 (0.1 %) | 0.01194 (0.6 %) | 0.01166 (12.5 %) | +0.6 % | exceeds | -2.4 % | within |  |
| 90 speedup | GPU | 98.41 (1.2 %) | 98.04 (0.2 %) | 101.4 (1.2 %) | -0.4 % | within | +3.4 % | exceeds FLAG | follows FastNPP's time (see that row) |
| 100 NPP | host | 1.298 (1.4 %) | 1.295 (0.5 %) | 1.294 (1.2 %) | -0.2 % | within | -0.1 % | within |  |
| 100 FastNPP | GPU | 0.01287 (0.7 %) | 0.01277 (1.3 %) | 0.01249 (0.8 %) | -0.7 % | within | -2.2 % | exceeds |  |
| 100 speedup | GPU | 101.2 (1.0 %) | 101.4 (1.0 %) | 103.7 (1.9 %) | +0.2 % | within | +2.3 % | exceeds |  |
| 110 NPP | host | 1.425 (1.6 %) | 1.428 (0.6 %) | 1.427 (0.9 %) | +0.2 % | within | -0.0 % | within |  |
| 110 FastNPP | GPU | 0.01368 (1.2 %) | 0.01385 (1.4 %) | 0.01335 (0.9 %) | +1.3 % | within | -3.6 % | exceeds FLAG | probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup |
| 110 speedup | GPU | 104.6 (0.7 %) | 103.5 (2.0 %) | 107.1 (0.8 %) | -1.1 % | within | +3.5 % | exceeds FLAG | follows FastNPP's time (see that row) |
| 120 NPP | host | 1.556 (1.6 %) | 1.559 (0.6 %) | 1.554 (1.5 %) | +0.2 % | within | -0.3 % | within |  |
| 120 FastNPP | GPU | 0.01468 (1.1 %) | 0.01465 (0.4 %) | 0.01433 (0.6 %) | -0.2 % | within | -2.2 % | exceeds |  |
| 120 speedup | GPU | 105.9 (1.7 %) | 106.8 (0.6 %) | 109.3 (1.7 %) | +0.8 % | within | +2.4 % | exceeds |  |
| 130 NPP | host | 1.676 (1.8 %) | 1.692 (1.4 %) | 1.683 (0.5 %) | +0.9 % | within | -0.5 % | within |  |
| 130 FastNPP | GPU | 0.01562 (0.4 %) | 0.01575 (0.6 %) | 0.01533 (0.3 %) | +0.8 % | exceeds | -2.7 % | exceeds |  |
| 130 speedup | GPU | 107.7 (1.9 %) | 108.1 (1.4 %) | 110.2 (0.8 %) | +0.3 % | within | +2.0 % | exceeds |  |
| 140 NPP | host | 1.812 (1.7 %) | 1.813 (0.8 %) | 1.813 (0.2 %) | +0.0 % | within | +0.0 % | within |  |
| 140 FastNPP | GPU | 0.01675 (0.7 %) | 0.01678 (8.7 %) | 0.01632 (0.2 %) | +0.2 % | within | -2.7 % | within |  |
| 140 speedup | GPU | 108.6 (2.2 %) | 108.1 (1.0 %) | 111.6 (0.6 %) | -0.5 % | within | +3.3 % | exceeds FLAG | follows FastNPP's time (see that row) |
| 150 NPP | host | 1.944 (1.8 %) | 1.942 (0.6 %) | 1.936 (0.8 %) | -0.1 % | within | -0.3 % | within |  |
| 150 FastNPP | GPU | 0.01768 (1.1 %) | 0.01763 (0.4 %) | 0.01709 (0.5 %) | -0.3 % | within | -3.1 % | exceeds FLAG | probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup |
| 150 speedup | GPU | 109.7 (2.5 %) | 110.3 (0.8 %) | 113.6 (1.1 %) | +0.6 % | within | +3.0 % | exceeds |  |

### NPP GPU program, No_CPU_Overhead (ms; FastNPP built once, valid only for fixed crops)

| batch, side | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| 10 NPP | host | 0.1323 (1.5 %) | 0.1337 (2.2 %) | 0.1311 (2.0 %) | +1.1 % | within | -1.9 % | within |  |
| 10 FastNPP | GPU | 0.003737 (4.1 %) | 0.00378 (1.4 %) | 0.003726 (1.9 %) | +1.1 % | within | -1.4 % | within |  |
| 10 speedup | GPU | 36.15 (1.8 %) | 35.87 (3.6 %) | 36.16 (1.3 %) | -0.8 % | within | +0.8 % | within |  |
| 20 NPP | host | 0.2578 (1.4 %) | 0.2638 (3.5 %) | 0.2629 (2.9 %) | +2.3 % | within | -0.3 % | within |  |
| 20 FastNPP | GPU | 0.004631 (4.9 %) | 0.004646 (4.5 %) | 0.004669 (3.7 %) | +0.3 % | within | +0.5 % | within |  |
| 20 speedup | GPU | 56.93 (4.5 %) | 57.78 (1.1 %) | 57.07 (6.1 %) | +1.5 % | within | -1.2 % | within |  |
| 30 NPP | host | 0.3891 (1.1 %) | 0.3936 (1.7 %) | 0.39 (3.9 %) | +1.2 % | within | -0.9 % | within |  |
| 30 FastNPP | GPU | 0.0052 (4.0 %) | 0.005205 (1.4 %) | 0.005312 (1.4 %) | +0.1 % | within | +2.1 % | exceeds |  |
| 30 speedup | GPU | 75.41 (4.2 %) | 75.43 (1.0 %) | 74.76 (3.1 %) | +0.0 % | within | -0.9 % | within |  |
| 40 NPP | host | 0.519 (1.8 %) | 0.5189 (1.1 %) | 0.524 (0.9 %) | -0.0 % | within | +1.0 % | within |  |
| 40 FastNPP | GPU | 0.005926 (4.9 %) | 0.005887 (3.0 %) | 0.005833 (1.8 %) | -0.7 % | within | -0.9 % | within |  |
| 40 speedup | GPU | 87.76 (4.6 %) | 89.32 (2.9 %) | 89.92 (1.5 %) | +1.8 % | within | +0.7 % | within |  |
| 50 NPP | host | 0.6504 (3.5 %) | 0.6513 (2.2 %) | 0.6508 (2.0 %) | +0.1 % | within | -0.1 % | within |  |
| 50 FastNPP | GPU | 0.006509 (1.4 %) | 0.006573 (2.4 %) | 0.006566 (2.2 %) | +1.0 % | within | -0.1 % | within |  |
| 50 speedup | GPU | 100.6 (5.1 %) | 100.4 (3.2 %) | 100.3 (1.6 %) | -0.1 % | within | -0.1 % | within |  |
| 60 NPP | host | 0.7739 (2.5 %) | 0.7782 (0.9 %) | 0.7799 (1.0 %) | +0.6 % | within | +0.2 % | within |  |
| 60 FastNPP | GPU | 0.007214 (1.9 %) | 0.007167 (2.0 %) | 0.007188 (2.7 %) | -0.7 % | within | +0.3 % | within |  |
| 60 speedup | GPU | 107.7 (4.1 %) | 108.9 (1.1 %) | 108.6 (1.9 %) | +1.1 % | within | -0.2 % | within |  |
| 70 NPP | host | 0.91 (2.2 %) | 0.9161 (1.3 %) | 0.9056 (1.2 %) | +0.7 % | within | -1.2 % | within |  |
| 70 FastNPP | GPU | 0.008067 (4.0 %) | 0.008106 (1.8 %) | 0.008009 (0.9 %) | +0.5 % | within | -1.2 % | within |  |
| 70 speedup | GPU | 114.4 (4.9 %) | 113.7 (3.1 %) | 114.1 (1.1 %) | -0.6 % | within | +0.4 % | within |  |
| 80 NPP | host | 1.037 (2.0 %) | 1.043 (1.0 %) | 1.036 (1.0 %) | +0.6 % | within | -0.7 % | within |  |
| 80 FastNPP | GPU | 0.008782 (3.4 %) | 0.008785 (2.9 %) | 0.008816 (0.1 %) | +0.0 % | within | +0.3 % | within |  |
| 80 speedup | GPU | 119 (5.4 %) | 118.3 (3.0 %) | 118.1 (1.1 %) | -0.6 % | within | -0.2 % | within |  |
| 90 NPP | host | 1.162 (3.5 %) | 1.168 (1.6 %) | 1.166 (1.1 %) | +0.5 % | within | -0.1 % | within |  |
| 90 FastNPP | GPU | 0.009168 (1.2 %) | 0.009328 (1.3 %) | 0.009096 (1.9 %) | +1.7 % | exceeds | -2.5 % | exceeds |  |
| 90 speedup | GPU | 126.1 (3.2 %) | 125.8 (2.7 %) | 128.1 (0.9 %) | -0.2 % | within | +1.8 % | within |  |
| 100 NPP | host | 1.302 (1.1 %) | 1.29 (0.7 %) | 1.303 (0.8 %) | -0.9 % | within | +1.0 % | exceeds |  |
| 100 FastNPP | GPU | 0.009931 (1.1 %) | 0.009942 (1.4 %) | 0.009876 (1.5 %) | +0.1 % | within | -0.7 % | within |  |
| 100 speedup | GPU | 131.5 (2.3 %) | 130.1 (0.9 %) | 132.3 (2.1 %) | -1.0 % | within | +1.7 % | within |  |
| 110 NPP | host | 1.424 (1.8 %) | 1.425 (1.0 %) | 1.426 (0.5 %) | +0.1 % | within | +0.1 % | within |  |
| 110 FastNPP | GPU | 0.01057 (0.6 %) | 0.01077 (1.3 %) | 0.01055 (16.1 %) | +1.9 % | exceeds | -2.1 % | within |  |
| 110 speedup | GPU | 134.9 (1.3 %) | 133 (2.3 %) | 135.5 (1.6 %) | -1.5 % | within | +1.9 % | within |  |
| 120 NPP | host | 1.558 (1.7 %) | 1.556 (1.7 %) | 1.555 (0.3 %) | -0.1 % | within | -0.1 % | within |  |
| 120 FastNPP | GPU | 0.01129 (0.3 %) | 0.01141 (1.4 %) | 0.01122 (0.4 %) | +1.0 % | within | -1.6 % | exceeds |  |
| 120 speedup | GPU | 138.2 (1.4 %) | 137.4 (0.7 %) | 138.9 (0.6 %) | -0.6 % | within | +1.1 % | exceeds |  |
| 130 NPP | host | 1.682 (0.9 %) | 1.688 (0.9 %) | 1.685 (0.2 %) | +0.3 % | within | -0.2 % | within |  |
| 130 FastNPP | GPU | 0.01198 (0.7 %) | 0.01225 (2.1 %) | 0.01189 (0.9 %) | +2.2 % | exceeds | -2.9 % | exceeds |  |
| 130 speedup | GPU | 140.7 (0.2 %) | 138.8 (2.4 %) | 142 (0.4 %) | -1.4 % | within | +2.3 % | within |  |
| 140 NPP | host | 1.813 (1.6 %) | 1.81 (0.8 %) | 1.824 (1.7 %) | -0.2 % | within | +0.8 % | within |  |
| 140 FastNPP | GPU | 0.01282 (1.5 %) | 0.01287 (1.3 %) | 0.01291 (2.1 %) | +0.4 % | within | +0.3 % | within |  |
| 140 speedup | GPU | 141.5 (0.4 %) | 140.5 (0.7 %) | 142 (2.3 %) | -0.8 % | exceeds | +1.1 % | within |  |
| 150 NPP | host | 1.951 (1.0 %) | 1.945 (0.3 %) | 1.947 (1.0 %) | -0.3 % | within | +0.1 % | within |  |
| 150 FastNPP | GPU | 0.01369 (1.9 %) | 0.0137 (2.7 %) | 0.01341 (3.6 %) | +0.0 % | within | -2.1 % | within |  |
| 150 speedup | GPU | 141.5 (2.0 %) | 142.1 (2.8 %) | 144.6 (3.2 %) | +0.4 % | within | +1.8 % | within |  |

FastNPP's one-time construction (cleaned build only, host timer, ms; median over the three runs, spread):

| batch | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ms | 0.00119 (13.4 %) | 0.00225 (23.1 %) | 0.0022 (6.8 %) | 0.00232 (24.1 %) | 0.00251 (17.9 %) | 0.00244 (20.9 %) | 0.00291 (16.5 %) | 0.00339 (32.4 %) | 0.00391 (22.8 %) | 0.0042 (27.1 %) | 0.00488 (11.9 %) | 0.00486 (14.4 %) | 0.00602 (14.6 %) | 0.00663 (24.1 %) | 0.00584 (8.4 %) |

### CPU-time program, NPP against FastNPP (ms of host time per frame)

| batch, side | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| 2 NPP | host | 0.02753 (1.9 %) | 0.02746 (3.0 %) | 0.0281 (0.7 %) | -0.2 % | within | +2.3 % | within |  |
| 2 FastNPP | host | 0.00191 (2.0 %) | 0.001918 (1.8 %) | 0.001914 (1.5 %) | +0.4 % | within | -0.2 % | within |  |
| 2 speedup | host | 14.66 (3.2 %) | 14.56 (2.2 %) | 14.87 (0.9 %) | -0.6 % | within | +2.1 % | within |  |
| 12 NPP | host | 0.1581 (2.1 %) | 0.158 (1.7 %) | 0.1577 (1.7 %) | -0.1 % | within | -0.2 % | within |  |
| 12 FastNPP | host | 0.002729 (1.5 %) | 0.002742 (1.4 %) | 0.002609 (0.5 %) | +0.5 % | within | -4.8 % | exceeds |  |
| 12 speedup | host | 58.41 (2.8 %) | 58.11 (2.0 %) | 61.32 (1.5 %) | -0.5 % | within | +5.5 % | exceeds FLAG | follows FastNPP's time |
| 22 NPP | host | 0.2852 (3.6 %) | 0.2845 (0.3 %) | 0.2861 (1.2 %) | -0.3 % | within | +0.6 % | within |  |
| 22 FastNPP | host | 0.002727 (2.4 %) | 0.002725 (1.7 %) | 0.002619 (1.4 %) | -0.1 % | within | -3.9 % | exceeds |  |
| 22 speedup | host | 107.2 (2.3 %) | 104.8 (1.6 %) | 109.2 (1.6 %) | -2.2 % | within | +4.2 % | exceeds |  |
| 32 NPP | host | 0.4131 (1.4 %) | 0.4142 (0.2 %) | 0.4125 (0.7 %) | +0.3 % | within | -0.4 % | within |  |
| 32 FastNPP | host | 0.002995 (1.8 %) | 0.003012 (0.8 %) | 0.002876 (1.2 %) | +0.6 % | within | -4.5 % | exceeds |  |
| 32 speedup | host | 139.3 (2.7 %) | 139.2 (1.3 %) | 145 (2.2 %) | -0.0 % | within | +4.2 % | exceeds |  |
| 42 NPP | host | 0.5414 (0.7 %) | 0.5433 (0.1 %) | 0.5422 (0.7 %) | +0.4 % | within | -0.2 % | within |  |
| 42 FastNPP | host | 0.003416 (6.1 %) | 0.00334 (2.8 %) | 0.003224 (4.0 %) | -2.2 % | within | -3.5 % | within |  |
| 42 speedup | host | 161.8 (4.9 %) | 163.5 (2.2 %) | 171.4 (2.3 %) | +1.0 % | within | +4.8 % | exceeds |  |
| 52 NPP | host | 0.6711 (1.0 %) | 0.6736 (0.5 %) | 0.6729 (0.4 %) | +0.4 % | within | -0.1 % | within |  |
| 52 FastNPP | host | 0.003748 (6.9 %) | 0.00372 (6.7 %) | 0.003498 (1.7 %) | -0.8 % | within | -6.0 % | within FLAG |  |
| 52 speedup | host | 180 (4.8 %) | 181.2 (4.5 %) | 193.3 (0.6 %) | +0.7 % | within | +6.7 % | exceeds FLAG | follows FastNPP's time |
| 62 NPP | host | 0.7999 (0.6 %) | 0.7989 (0.9 %) | 0.8032 (0.7 %) | -0.1 % | within | +0.5 % | within |  |
| 62 FastNPP | host | 0.003932 (1.8 %) | 0.003963 (2.5 %) | 0.003809 (0.9 %) | +0.8 % | within | -3.9 % | exceeds |  |
| 62 speedup | host | 206.1 (1.6 %) | 201.2 (2.5 %) | 212.3 (0.8 %) | -2.4 % | within | +5.5 % | exceeds FLAG | follows FastNPP's time |
| 72 NPP | host | 0.9325 (1.0 %) | 0.923 (0.6 %) | 0.9339 (0.2 %) | -1.0 % | exceeds | +1.2 % | exceeds |  |
| 72 FastNPP | host | 0.004275 (2.4 %) | 0.004321 (3.6 %) | 0.004197 (3.1 %) | +1.1 % | within | -2.9 % | within |  |
| 72 speedup | host | 221.4 (1.2 %) | 217.4 (2.3 %) | 226 (1.5 %) | -1.8 % | within | +3.9 % | exceeds |  |
| 82 NPP | host | 1.058 (2.7 %) | 1.052 (0.0 %) | 1.058 (1.4 %) | -0.6 % | within | +0.5 % | within |  |
| 82 FastNPP | host | 0.004551 (1.2 %) | 0.004523 (0.9 %) | 0.004379 (1.4 %) | -0.6 % | within | -3.2 % | exceeds |  |
| 82 speedup | host | 236.7 (2.0 %) | 234.3 (0.9 %) | 243.4 (2.1 %) | -1.0 % | within | +3.9 % | exceeds |  |
| 92 NPP | host | 1.19 (0.1 %) | 1.176 (1.3 %) | 1.19 (0.5 %) | -1.1 % | within | +1.1 % | within |  |
| 92 FastNPP | host | 0.004892 (1.6 %) | 0.004729 (3.0 %) | 0.00474 (3.6 %) | -3.3 % | exceeds | +0.2 % | within |  |
| 92 speedup | host | 245.6 (1.9 %) | 248.9 (2.1 %) | 254.3 (2.2 %) | +1.3 % | within | +2.2 % | exceeds |  |
| 102 NPP | host | 1.311 (1.8 %) | 1.299 (0.6 %) | 1.314 (0.6 %) | -0.9 % | within | +1.2 % | exceeds |  |
| 102 FastNPP | host | 0.005102 (0.3 %) | 0.005101 (10.2 %) | 0.005062 (0.7 %) | -0.0 % | within | -0.8 % | within |  |
| 102 speedup | host | 258.6 (1.8 %) | 256.1 (9.8 %) | 262.6 (1.4 %) | -0.9 % | within | +2.5 % | within |  |
| 112 NPP | host | 1.438 (3.3 %) | 1.439 (1.6 %) | 1.438 (1.1 %) | +0.1 % | within | -0.1 % | within |  |
| 112 FastNPP | host | 0.005484 (8.6 %) | 0.005435 (1.4 %) | 0.005302 (2.7 %) | -0.9 % | within | -2.5 % | within |  |
| 112 speedup | host | 263.6 (3.1 %) | 265.5 (0.8 %) | 272.9 (3.5 %) | +0.7 % | within | +2.8 % | within |  |
| 122 NPP | host | 1.561 (1.7 %) | 1.553 (1.2 %) | 1.568 (1.5 %) | -0.5 % | within | +0.9 % | within |  |
| 122 FastNPP | host | 0.005722 (0.9 %) | 0.00573 (1.7 %) | 0.005653 (2.5 %) | +0.1 % | within | -1.3 % | within |  |
| 122 speedup | host | 275.4 (2.3 %) | 277.3 (1.6 %) | 275.5 (2.3 %) | +0.7 % | within | -0.6 % | within |  |
| 132 NPP | host | 1.694 (0.5 %) | 1.68 (1.0 %) | 1.697 (1.0 %) | -0.8 % | within | +1.0 % | within |  |
| 132 FastNPP | host | 0.006115 (1.0 %) | 0.006079 (0.9 %) | 0.006035 (0.9 %) | -0.6 % | within | -0.7 % | within |  |
| 132 speedup | host | 279.1 (0.7 %) | 280.2 (0.3 %) | 282.6 (1.6 %) | +0.4 % | within | +0.8 % | within |  |
| 142 NPP | host | 1.82 (1.2 %) | 1.807 (0.5 %) | 1.819 (0.5 %) | -0.7 % | within | +0.7 % | exceeds |  |
| 142 FastNPP | host | 0.006406 (1.4 %) | 0.006416 (1.7 %) | 0.006293 (0.5 %) | +0.1 % | within | -1.9 % | exceeds |  |
| 142 speedup | host | 287.4 (0.7 %) | 284.5 (1.4 %) | 291.1 (0.7 %) | -1.0 % | within | +2.3 % | exceeds |  |
| 152 NPP | host | 1.958 (2.3 %) | 1.936 (0.2 %) | 1.947 (0.7 %) | -1.1 % | within | +0.6 % | within |  |
| 152 FastNPP | host | 0.006827 (2.1 %) | 0.006744 (0.7 %) | 0.006721 (2.7 %) | -1.2 % | within | -0.3 % | within |  |
| 152 speedup | host | 290.7 (1.3 %) | 289.6 (0.7 %) | 291.8 (2.7 %) | -0.3 % | within | +0.7 % | within |  |

### CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame)

| batch, side | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| 2 OpenCV | host | 0.0259 (one run) | 0.02577 (1.0 %) | 0.02569 (1.0 %) | -0.5 % | within | -0.3 % | within |  |
| 2 cvGS | host | 0.001819 (one run) | 0.001788 (3.6 %) | 0.00188 (4.1 %) | -1.7 % | within | +5.1 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 2 speedup | host | 14.3 (one run) | 14.44 (2.4 %) | 13.87 (1.8 %) | +0.9 % | within | -4.0 % | exceeds |  |
| 12 OpenCV | host | 0.1588 (one run) | 0.1533 (2.1 %) | 0.156 (3.8 %) | -3.4 % | exceeds | +1.8 % | within |  |
| 12 cvGS | host | 0.002247 (one run) | 0.002104 (4.3 %) | 0.00248 (0.6 %) | -6.4 % | exceeds FLAG | +17.9 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 12 speedup | host | 72.58 (one run) | 73.34 (1.3 %) | 64.42 (3.9 %) | +1.0 % | within | -12.2 % | exceeds FLAG | follows cvGS's time |
| 22 OpenCV | host | 0.2863 (one run) | 0.279 (0.8 %) | 0.28 (1.0 %) | -2.5 % | exceeds | +0.4 % | within |  |
| 22 cvGS | host | 0.002299 (one run) | 0.002308 (1.2 %) | 0.002897 (1.9 %) | +0.4 % | within | +25.5 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 22 speedup | host | 125.6 (one run) | 122.2 (1.5 %) | 99.2 (2.2 %) | -2.7 % | exceeds | -18.8 % | exceeds FLAG | follows cvGS's time |
| 32 OpenCV | host | 0.4073 (one run) | 0.4036 (0.3 %) | 0.4036 (0.8 %) | -0.9 % | exceeds | +0.0 % | within |  |
| 32 cvGS | host | 0.002523 (one run) | 0.00245 (1.5 %) | 0.003216 (2.2 %) | -2.9 % | exceeds | +31.3 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 32 speedup | host | 164.5 (one run) | 165.4 (0.9 %) | 126.6 (3.2 %) | +0.6 % | within | -23.5 % | exceeds FLAG | follows cvGS's time |
| 42 OpenCV | host | 0.5318 (one run) | 0.5288 (1.0 %) | 0.5307 (1.4 %) | -0.6 % | within | +0.3 % | within |  |
| 42 cvGS | host | 0.002575 (one run) | 0.002617 (0.7 %) | 0.003663 (4.4 %) | +1.6 % | exceeds | +39.9 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 42 speedup | host | 207.6 (one run) | 202.6 (1.3 %) | 146.1 (5.7 %) | -2.4 % | exceeds | -27.9 % | exceeds FLAG | follows cvGS's time |
| 52 OpenCV | host | 0.6751 (one run) | 0.6532 (0.7 %) | 0.6585 (0.7 %) | -3.2 % | exceeds | +0.8 % | exceeds |  |
| 52 cvGS | host | 0.002926 (one run) | 0.002752 (0.4 %) | 0.004129 (7.6 %) | -5.9 % | exceeds FLAG | +50.0 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 52 speedup | host | 235.3 (one run) | 238.7 (1.0 %) | 160.9 (7.1 %) | +1.4 % | exceeds | -32.6 % | exceeds FLAG | follows cvGS's time |
| 62 OpenCV | host | 0.7857 (one run) | 0.7796 (1.4 %) | 0.7844 (1.3 %) | -0.8 % | within | +0.6 % | within |  |
| 62 cvGS | host | 0.002914 (one run) | 0.003002 (3.2 %) | 0.004497 (5.2 %) | +3.0 % | within | +49.8 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 62 speedup | host | 273.9 (one run) | 263.3 (3.1 %) | 175.4 (4.2 %) | -3.9 % | exceeds | -33.4 % | exceeds FLAG | follows cvGS's time |
| 72 OpenCV | host | 0.9107 (one run) | 0.9036 (0.9 %) | 0.9114 (0.5 %) | -0.8 % | within | +0.9 % | within |  |
| 72 cvGS | host | 0.003172 (one run) | 0.003103 (1.3 %) | 0.004951 (6.4 %) | -2.2 % | exceeds | +59.6 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 72 speedup | host | 289.7 (one run) | 293 (0.6 %) | 186.5 (5.3 %) | +1.1 % | exceeds | -36.3 % | exceeds FLAG | follows cvGS's time |
| 82 OpenCV | host | 1.041 (one run) | 1.034 (0.6 %) | 1.04 (0.5 %) | -0.7 % | exceeds | +0.6 % | within |  |
| 82 cvGS | host | 0.003273 (one run) | 0.003426 (1.9 %) | 0.005331 (5.9 %) | +4.7 % | exceeds | +55.6 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 82 speedup | host | 320.7 (one run) | 305.5 (2.3 %) | 195.9 (5.1 %) | -4.7 % | exceeds | -35.9 % | exceeds FLAG | follows cvGS's time |
| 92 OpenCV | host | 1.153 (one run) | 1.16 (0.0 %) | 1.164 (0.3 %) | +0.6 % | exceeds | +0.3 % | exceeds |  |
| 92 cvGS | host | 0.003547 (one run) | 0.003471 (2.7 %) | 0.005698 (6.7 %) | -2.1 % | within | +64.1 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 92 speedup | host | 329.7 (one run) | 336.9 (3.0 %) | 205.2 (7.0 %) | +2.2 % | within | -39.1 % | exceeds FLAG | follows cvGS's time |
| 102 OpenCV | host | 1.295 (one run) | 1.281 (1.8 %) | 1.287 (1.0 %) | -1.1 % | within | +0.5 % | within |  |
| 102 cvGS | host | 0.003594 (one run) | 0.003702 (2.2 %) | 0.006497 (10.1 %) | +3.0 % | exceeds | +75.5 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 102 speedup | host | 363.4 (one run) | 349.6 (2.7 %) | 199 (8.6 %) | -3.8 % | exceeds | -43.1 % | exceeds FLAG | follows cvGS's time |
| 112 OpenCV | host | 1.425 (one run) | 1.408 (1.4 %) | 1.414 (0.7 %) | -1.2 % | within | +0.4 % | within |  |
| 112 cvGS | host | 0.00402 (one run) | 0.003803 (1.6 %) | 0.006615 (6.8 %) | -5.4 % | exceeds FLAG | +73.9 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 112 speedup | host | 357.6 (one run) | 370.9 (1.2 %) | 214.3 (6.8 %) | +3.7 % | exceeds | -42.2 % | exceeds FLAG | follows cvGS's time |
| 122 OpenCV | host | 1.537 (one run) | 1.522 (1.6 %) | 1.539 (1.1 %) | -1.0 % | within | +1.2 % | within |  |
| 122 cvGS | host | 0.003879 (one run) | 0.004192 (2.4 %) | 0.007148 (6.2 %) | +8.1 % | exceeds FLAG | +70.5 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 122 speedup | host | 400.3 (one run) | 373.1 (1.8 %) | 218.2 (6.7 %) | -6.8 % | exceeds FLAG | -41.5 % | exceeds FLAG | follows cvGS's time |
| 132 OpenCV | host | 1.662 (one run) | 1.659 (1.8 %) | 1.667 (0.3 %) | -0.2 % | within | +0.5 % | within |  |
| 132 cvGS | host | 0.004267 (one run) | 0.004148 (1.7 %) | 0.007459 (5.0 %) | -2.8 % | exceeds | +79.8 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 132 speedup | host | 394.4 (one run) | 402.3 (2.9 %) | 224.9 (5.5 %) | +2.0 % | within | -44.1 % | exceeds FLAG | follows cvGS's time |
| 142 OpenCV | host | 1.779 (one run) | 1.773 (2.1 %) | 1.791 (0.5 %) | -0.4 % | within | +1.0 % | within |  |
| 142 cvGS | host | 0.004298 (one run) | 0.004496 (1.4 %) | 0.008012 (4.0 %) | +4.6 % | exceeds | +78.2 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 142 speedup | host | 419.5 (one run) | 397.2 (3.1 %) | 225.2 (4.0 %) | -5.3 % | exceeds FLAG | -43.3 % | exceeds FLAG | follows cvGS's time |
| 152 OpenCV | host | 1.909 (one run) | 1.911 (1.2 %) | 1.918 (0.2 %) | +0.1 % | within | +0.4 % | within |  |
| 152 cvGS | host | 0.004694 (one run) | 0.004468 (3.6 %) | 0.008217 (8.3 %) | -4.8 % | exceeds | +83.9 % | exceeds FLAG | the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop |
| 152 speedup | host | 411.5 (one run) | 431.3 (3.5 %) | 233.8 (8.1 %) | +4.8 % | exceeds | -45.8 % | exceeds FLAG | follows cvGS's time |

### E9 cross-library kernels (ms per call; median of each run, then over the runs)

| variant, column | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| M1 launch_ms | GPU | 0.006393 (0.4 %) | 0.006384 (0.3 %) | 0.006388 (0.1 %) | -0.2 % | within | +0.1 % | within |  |
| M1 graph_ms | GPU | 0.00594 (0.8 %) | 0.005914 (0.7 %) | 0.005904 (0.5 %) | -0.4 % | within | -0.2 % | within |  |
| M2 launch_ms | GPU | 0.006439 (0.5 %) | 0.006427 (0.2 %) | 0.006444 (0.5 %) | -0.2 % | within | +0.3 % | within |  |
| M2 graph_ms | GPU | 0.005953 (1.0 %) | 0.005948 (1.0 %) | 0.005895 (0.4 %) | -0.1 % | within | -0.9 % | within |  |
| R1 launch_ms | GPU | 0.006241 (0.1 %) | 0.006239 (0.0 %) | 0.006241 (0.0 %) | -0.0 % | within | +0.0 % | exceeds |  |
| R1 graph_ms | GPU | 0.005744 (1.3 %) | 0.005724 (0.2 %) | 0.00574 (0.3 %) | -0.3 % | within | +0.3 % | exceeds |  |
| R2 launch_ms | GPU | 0.006648 (0.1 %) | 0.006646 (0.0 %) | 0.006648 (0.1 %) | -0.0 % | within | +0.0 % | within |  |
| R2 graph_ms | GPU | 0.006061 (0.9 %) | 0.006041 (0.6 %) | 0.006055 (0.1 %) | -0.3 % | within | +0.2 % | within |  |

kernels_per_call: M1 published [1.0], reference [1.0], cleaned [1.0], M2 published [1.0], reference [1.0], cleaned [1.0], R1 published [1.0], reference [1.0], cleaned [1.0], R2 published [1.0], reference [1.0], cleaned [1.0]

### E4 pre-processing, gpu_mean_ms (CUDA events around one call); the table shows the batches of the published README, the CSV has all

| point, version | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| E4a 10 fkl_fused | GPU | 0.00358 (1.9 %) | 0.003499 (6.0 %) | 0.003783 (3.1 %) | -2.3 % | within | +8.1 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 10 fkl_fused_half_pixel | GPU | 0.003582 (47.9 %) | 0.003544 (2.8 %) | 0.003791 (0.6 %) | -1.1 % | within | +7.0 % | exceeds FLAG | as fkl_fused |
| E4a 10 npp | host | 0.1322 (0.9 %) | 0.1326 (3.8 %) | 0.1319 (1.5 %) | +0.3 % | within | -0.5 % | within |  |
| E4a 10 opencv_percrop | host | 0.1302 (1.5 %) | 0.1293 (2.8 %) | 0.1297 (0.7 %) | -0.7 % | within | +0.3 % | within |  |
| E4a 50 fkl_fused | GPU | 0.005766 (1.0 %) | 0.005706 (0.7 %) | 0.007524 (1.1 %) | -1.0 % | exceeds | +31.9 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 50 fkl_fused_half_pixel | GPU | 0.005784 (0.6 %) | 0.005779 (1.2 %) | 0.00752 (0.9 %) | -0.1 % | within | +30.1 % | exceeds FLAG | as fkl_fused |
| E4a 50 npp | host | 0.6497 (0.9 %) | 0.6466 (1.8 %) | 0.653 (1.4 %) | -0.5 % | within | +1.0 % | within |  |
| E4a 50 opencv_percrop | host | 0.638 (0.1 %) | 0.6376 (0.5 %) | 0.636 (1.2 %) | -0.1 % | within | -0.3 % | within |  |
| E4a 100 fkl_fused | GPU | 0.008172 (2.5 %) | 0.008332 (0.4 %) | 0.0114 (0.2 %) | +2.0 % | within | +36.8 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 100 fkl_fused_half_pixel | GPU | 0.00825 (1.4 %) | 0.008369 (4.1 %) | 0.01145 (0.3 %) | +1.4 % | within | +36.8 % | exceeds FLAG | as fkl_fused |
| E4a 100 npp | host | 1.297 (1.7 %) | 1.283 (1.5 %) | 1.299 (1.1 %) | -1.1 % | within | +1.2 % | within |  |
| E4a 100 opencv_percrop | host | 1.272 (1.1 %) | 1.27 (0.8 %) | 1.273 (1.3 %) | -0.1 % | within | +0.2 % | within |  |
| E4a 150 fkl_fused | GPU | 0.01069 (1.4 %) | 0.01083 (0.7 %) | 0.01547 (0.8 %) | +1.3 % | within | +42.8 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 150 fkl_fused_half_pixel | GPU | 0.01076 (0.8 %) | 0.01096 (17.2 %) | 0.01555 (1.3 %) | +1.8 % | within | +41.9 % | exceeds FLAG | as fkl_fused |
| E4a 150 npp | host | 1.943 (0.8 %) | 1.941 (1.0 %) | 1.949 (1.8 %) | -0.1 % | within | +0.4 % | within |  |
| E4a 150 opencv_percrop | host | 1.907 (0.8 %) | 1.889 (1.1 %) | 1.896 (1.3 %) | -0.9 % | within | +0.4 % | within |  |
| E4a 200 fkl_fused | GPU | 0.01353 (1.1 %) | 0.0135 (1.0 %) | 0.0192 (0.4 %) | -0.2 % | within | +42.3 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 200 fkl_fused_half_pixel | GPU | 0.01354 (0.9 %) | 0.01347 (0.3 %) | 0.01921 (0.7 %) | -0.5 % | within | +42.6 % | exceeds FLAG | as fkl_fused |
| E4a 200 npp | host | 2.594 (1.6 %) | 2.593 (0.8 %) | 2.59 (1.3 %) | -0.0 % | within | -0.1 % | within |  |
| E4a 200 opencv_percrop | host | 2.533 (0.5 %) | 2.539 (0.5 %) | 2.527 (1.4 %) | +0.3 % | within | -0.5 % | within |  |
| E4a 250 fkl_fused | GPU | 0.0161 (0.6 %) | 0.01592 (0.3 %) | 0.02317 (0.4 %) | -1.2 % | exceeds | +45.6 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 250 fkl_fused_half_pixel | GPU | 0.01611 (1.1 %) | 0.016 (0.6 %) | 0.02319 (0.8 %) | -0.7 % | within | +45.0 % | exceeds FLAG | as fkl_fused |
| E4a 250 npp | host | 3.242 (1.2 %) | 3.221 (1.2 %) | 3.233 (2.0 %) | -0.7 % | within | +0.4 % | within |  |
| E4a 250 opencv_percrop | host | 3.182 (0.7 %) | 3.175 (1.3 %) | 3.186 (1.3 %) | -0.2 % | within | +0.4 % | within |  |
| E4a 300 fkl_fused | GPU | 0.01851 (0.1 %) | 0.01843 (0.7 %) | 0.02754 (0.3 %) | -0.4 % | within | +49.5 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 300 fkl_fused_half_pixel | GPU | 0.01865 (0.5 %) | 0.01849 (0.4 %) | 0.02756 (1.6 %) | -0.9 % | exceeds | +49.0 % | exceeds FLAG | as fkl_fused |
| E4a 300 npp | host | 3.889 (2.1 %) | 3.862 (0.8 %) | 3.884 (2.2 %) | -0.7 % | within | +0.6 % | within |  |
| E4a 300 opencv_percrop | host | 3.804 (1.3 %) | 3.821 (0.9 %) | 3.814 (1.2 %) | +0.4 % | within | -0.2 % | within |  |
| E4c 720x480 fkl_fused | GPU | 0.005221 (1.6 %) | 0.0053 (1.5 %) | 0.005248 (1.9 %) | +1.5 % | within | -1.0 % | within |  |
| E4c 720x480 fkl_fused_half_pixel | GPU | 0.005309 (3.1 %) | 0.005249 (0.8 %) | 0.005243 (1.4 %) | -1.1 % | within | -0.1 % | within |  |
| E4c 720x480 npp | GPU | 0.03603 (0.2 %) | 0.03642 (0.1 %) | 0.0362 (1.3 %) | +1.1 % | exceeds | -0.6 % | within |  |
| E4c 720x480 opencv_percrop | GPU | 0.02759 (0.1 %) | 0.02773 (2.2 %) | 0.02775 (1.1 %) | +0.5 % | within | +0.1 % | within |  |
| E4c 1920x1080 fkl_fused | GPU | 0.0156 (0.5 %) | 0.01564 (0.4 %) | 0.01551 (0.3 %) | +0.3 % | within | -0.8 % | exceeds |  |
| E4c 1920x1080 fkl_fused_half_pixel | GPU | 0.01567 (0.3 %) | 0.01572 (0.5 %) | 0.01564 (0.8 %) | +0.4 % | within | -0.5 % | within |  |
| E4c 1920x1080 npp | GPU | 0.1539 (0.1 %) | 0.1548 (0.1 %) | 0.1556 (1.4 %) | +0.5 % | exceeds | +0.6 % | within |  |
| E4c 1920x1080 opencv_percrop | GPU | 0.1734 (0.1 %) | 0.1745 (0.4 %) | 0.1744 (0.2 %) | +0.7 % | exceeds | -0.1 % | within |  |
| E4c 3840x2160 fkl_fused | GPU | 0.1567 (0.0 %) | 0.1581 (0.1 %) | 0.1579 (0.1 %) | +0.9 % | exceeds | -0.2 % | exceeds |  |
| E4c 3840x2160 fkl_fused_half_pixel | GPU | 0.1566 (0.1 %) | 0.1579 (0.2 %) | 0.1577 (0.2 %) | +0.8 % | exceeds | -0.1 % | within |  |
| E4c 3840x2160 npp | GPU | 1.849 (0.1 %) | 1.846 (0.1 %) | 1.849 (0.0 %) | -0.2 % | exceeds | +0.2 % | exceeds |  |
| E4c 3840x2160 opencv_percrop | GPU | 1.605 (0.0 %) | 1.604 (0.0 %) | 1.605 (0.1 %) | -0.0 % | within | +0.1 % | within |  |

### E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it)

| point, version | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| E4a 10 fkl_fused | host | 1.73 (8.8 %) | 1.728 (4.3 %) | 2.04 (2.3 %) | -0.1 % | within | +18.1 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 10 fkl_fused_half_pixel | host | 1.697 (2.8 %) | 1.691 (1.7 %) | 2.022 (3.9 %) | -0.4 % | within | +19.6 % | exceeds FLAG | as fkl_fused |
| E4a 10 npp | host | 132.5 (2.8 %) | 129.7 (2.1 %) | 130.9 (1.2 %) | -2.1 % | within | +0.9 % | within |  |
| E4a 10 opencv_percrop | host | 127.4 (0.8 %) | 126.6 (4.6 %) | 126.7 (2.4 %) | -0.6 % | within | +0.1 % | within |  |
| E4a 50 fkl_fused | host | 1.774 (2.9 %) | 1.803 (4.2 %) | 3.345 (2.1 %) | +1.6 % | within | +85.5 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 50 fkl_fused_half_pixel | host | 1.777 (3.6 %) | 1.828 (9.8 %) | 3.317 (1.8 %) | +2.8 % | within | +81.5 % | exceeds FLAG | as fkl_fused |
| E4a 50 npp | host | 649.3 (1.6 %) | 646.9 (1.0 %) | 642.6 (1.9 %) | -0.4 % | within | -0.7 % | within |  |
| E4a 50 opencv_percrop | host | 637.8 (0.7 %) | 637.6 (0.8 %) | 638.1 (1.1 %) | -0.0 % | within | +0.1 % | within |  |
| E4a 100 fkl_fused | host | 1.949 (2.4 %) | 1.99 (3.0 %) | 4.987 (0.2 %) | +2.2 % | within | +150.5 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 100 fkl_fused_half_pixel | host | 1.924 (7.3 %) | 1.997 (3.9 %) | 4.954 (1.1 %) | +3.8 % | within | +148.1 % | exceeds FLAG | as fkl_fused |
| E4a 100 npp | host | 1,295 (2.2 %) | 1,288 (2.4 %) | 1,290 (1.8 %) | -0.5 % | within | +0.1 % | within |  |
| E4a 100 opencv_percrop | host | 1,275 (1.3 %) | 1,277 (1.4 %) | 1,271 (1.6 %) | +0.1 % | within | -0.5 % | within |  |
| E4a 150 fkl_fused | host | 2.199 (3.2 %) | 2.178 (2.7 %) | 6.922 (0.9 %) | -1.0 % | within | +217.8 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 150 fkl_fused_half_pixel | host | 2.143 (1.7 %) | 2.166 (2.9 %) | 6.878 (0.6 %) | +1.1 % | within | +217.5 % | exceeds FLAG | as fkl_fused |
| E4a 150 npp | host | 1,947 (1.0 %) | 1,944 (1.9 %) | 1,929 (1.8 %) | -0.1 % | within | -0.8 % | within |  |
| E4a 150 opencv_percrop | host | 1,915 (0.8 %) | 1,906 (0.5 %) | 1,900 (0.7 %) | -0.5 % | within | -0.3 % | within |  |
| E4a 200 fkl_fused | host | 2.346 (2.5 %) | 2.385 (0.7 %) | 8.269 (1.4 %) | +1.7 % | within | +246.7 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 200 fkl_fused_half_pixel | host | 2.368 (3.5 %) | 2.409 (1.4 %) | 8.207 (0.3 %) | +1.7 % | within | +240.6 % | exceeds FLAG | as fkl_fused |
| E4a 200 npp | host | 2,616 (1.5 %) | 2,597 (0.8 %) | 2,571 (1.7 %) | -0.7 % | within | -1.0 % | within |  |
| E4a 200 opencv_percrop | host | 2,552 (1.0 %) | 2,556 (1.8 %) | 2,541 (1.0 %) | +0.2 % | within | -0.6 % | within |  |
| E4a 250 fkl_fused | host | 2.528 (3.7 %) | 2.561 (1.1 %) | 9.928 (1.5 %) | +1.3 % | within | +287.7 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 250 fkl_fused_half_pixel | host | 2.597 (0.5 %) | 2.608 (3.0 %) | 9.946 (1.0 %) | +0.4 % | within | +281.3 % | exceeds FLAG | as fkl_fused |
| E4a 250 npp | host | 3,251 (1.6 %) | 3,248 (0.9 %) | 3,213 (2.0 %) | -0.1 % | within | -1.1 % | within |  |
| E4a 250 opencv_percrop | host | 3,205 (1.0 %) | 3,182 (0.5 %) | 3,171 (0.7 %) | -0.7 % | within | -0.3 % | within |  |
| E4a 300 fkl_fused | host | 2.785 (3.4 %) | 2.796 (2.0 %) | 12.2 (1.1 %) | +0.4 % | within | +336.4 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 300 fkl_fused_half_pixel | host | 2.707 (1.7 %) | 2.742 (2.3 %) | 12.24 (1.4 %) | +1.3 % | within | +346.4 % | exceeds FLAG | as fkl_fused |
| E4a 300 npp | host | 3,900 (1.0 %) | 3,892 (1.1 %) | 3,874 (1.6 %) | -0.2 % | within | -0.5 % | within |  |
| E4a 300 opencv_percrop | host | 3,819 (0.9 %) | 3,821 (0.4 %) | 3,811 (1.2 %) | +0.1 % | within | -0.3 % | within |  |
| E4c 720x480 fkl_fused | host | 1.602 (1.3 %) | 1.636 (1.2 %) | 1.6 (1.7 %) | +2.2 % | exceeds | -2.2 % | exceeds |  |
| E4c 720x480 fkl_fused_half_pixel | host | 1.618 (1.9 %) | 1.659 (2.0 %) | 1.616 (0.9 %) | +2.5 % | exceeds | -2.6 % | exceeds |  |
| E4c 720x480 npp | host | 14.53 (1.7 %) | 14.5 (0.6 %) | 14.45 (2.2 %) | -0.2 % | within | -0.4 % | within |  |
| E4c 720x480 opencv_percrop | host | 12.72 (1.2 %) | 12.72 (2.0 %) | 12.83 (4.6 %) | -0.0 % | within | +0.8 % | within |  |
| E4c 1920x1080 fkl_fused | host | 1.631 (0.6 %) | 1.626 (5.2 %) | 1.628 (3.4 %) | -0.3 % | within | +0.1 % | within |  |
| E4c 1920x1080 fkl_fused_half_pixel | host | 1.624 (1.9 %) | 1.635 (1.5 %) | 1.617 (3.2 %) | +0.7 % | within | -1.1 % | within |  |
| E4c 1920x1080 npp | host | 14.6 (1.4 %) | 14.58 (2.0 %) | 14.44 (2.7 %) | -0.2 % | within | -0.9 % | within |  |
| E4c 1920x1080 opencv_percrop | host | 12.78 (0.1 %) | 12.7 (0.5 %) | 12.72 (1.8 %) | -0.6 % | exceeds | +0.1 % | within |  |
| E4c 3840x2160 fkl_fused | host | 1.67 (1.9 %) | 1.671 (2.3 %) | 1.649 (3.8 %) | +0.1 % | within | -1.3 % | within |  |
| E4c 3840x2160 fkl_fused_half_pixel | host | 1.654 (1.4 %) | 1.68 (5.6 %) | 1.681 (4.3 %) | +1.6 % | within | +0.1 % | within |  |
| E4c 3840x2160 npp | host | 14.84 (1.2 %) | 14.92 (2.3 %) | 14.85 (2.1 %) | +0.5 % | within | -0.5 % | within |  |
| E4c 3840x2160 opencv_percrop | host | 13.2 (0.5 %) | 13.09 (1.1 %) | 13.02 (1.2 %) | -0.8 % | within | -0.5 % | within |  |

### E4 pre-processing, host_launch_us (100 calls back to back, wall time per call)

| point, version | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| E4a 10 fkl_fused | host | 1.716 (4.3 %) | 1.716 (1.4 %) | 2.045 (1.8 %) | -0.0 % | within | +19.2 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 10 fkl_fused_half_pixel | host | 1.774 (5.3 %) | 1.721 (1.9 %) | 2.092 (2.2 %) | -3.0 % | within | +21.6 % | exceeds FLAG | as fkl_fused |
| E4a 10 npp | host | 131.3 (1.7 %) | 129.6 (1.8 %) | 130.2 (1.9 %) | -1.2 % | within | +0.4 % | within |  |
| E4a 10 opencv_percrop | host | 127.7 (1.4 %) | 127 (2.2 %) | 126.5 (0.8 %) | -0.6 % | within | -0.4 % | within |  |
| E4a 50 fkl_fused | host | 1.782 (2.9 %) | 1.763 (2.2 %) | 3.334 (0.5 %) | -1.1 % | within | +89.1 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 50 fkl_fused_half_pixel | host | 1.775 (4.2 %) | 1.763 (3.2 %) | 3.367 (0.8 %) | -0.7 % | within | +91.0 % | exceeds FLAG | as fkl_fused |
| E4a 50 npp | host | 646.2 (1.9 %) | 644.3 (1.2 %) | 645.4 (1.5 %) | -0.3 % | within | +0.2 % | within |  |
| E4a 50 opencv_percrop | host | 635.9 (0.6 %) | 635.4 (0.4 %) | 634 (1.3 %) | -0.1 % | within | -0.2 % | within |  |
| E4a 100 fkl_fused | host | 1.969 (1.4 %) | 1.95 (1.8 %) | 4.956 (1.6 %) | -1.0 % | within | +154.1 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 100 fkl_fused_half_pixel | host | 1.946 (2.1 %) | 1.948 (1.3 %) | 4.975 (0.9 %) | +0.1 % | within | +155.4 % | exceeds FLAG | as fkl_fused |
| E4a 100 npp | host | 1,290 (1.7 %) | 1,283 (2.7 %) | 1,287 (1.8 %) | -0.5 % | within | +0.3 % | within |  |
| E4a 100 opencv_percrop | host | 1,280 (1.6 %) | 1,266 (1.1 %) | 1,271 (1.6 %) | -1.1 % | within | +0.4 % | within |  |
| E4a 150 fkl_fused | host | 2.136 (0.9 %) | 2.178 (5.8 %) | 6.927 (1.6 %) | +2.0 % | within | +218.0 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 150 fkl_fused_half_pixel | host | 2.161 (2.2 %) | 2.124 (4.4 %) | 6.926 (0.2 %) | -1.7 % | within | +226.1 % | exceeds FLAG | as fkl_fused |
| E4a 150 npp | host | 1,944 (1.5 %) | 1,934 (0.9 %) | 1,924 (2.0 %) | -0.5 % | within | -0.5 % | within |  |
| E4a 150 opencv_percrop | host | 1,912 (0.9 %) | 1,903 (0.6 %) | 1,904 (0.6 %) | -0.5 % | within | +0.1 % | within |  |
| E4a 200 fkl_fused | host | 2.295 (2.3 %) | 2.351 (1.5 %) | 8.223 (1.0 %) | +2.4 % | exceeds | +249.8 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 200 fkl_fused_half_pixel | host | 2.349 (4.1 %) | 2.37 (2.3 %) | 8.248 (0.9 %) | +0.9 % | within | +248.1 % | exceeds FLAG | as fkl_fused |
| E4a 200 npp | host | 2,595 (1.3 %) | 2,597 (0.2 %) | 2,573 (2.0 %) | +0.1 % | within | -0.9 % | within |  |
| E4a 200 opencv_percrop | host | 2,539 (0.6 %) | 2,557 (1.7 %) | 2,538 (1.0 %) | +0.7 % | within | -0.7 % | within |  |
| E4a 250 fkl_fused | host | 2.5 (1.0 %) | 2.516 (2.1 %) | 9.939 (2.3 %) | +0.6 % | within | +295.0 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 250 fkl_fused_half_pixel | host | 2.559 (1.6 %) | 2.534 (1.8 %) | 9.928 (0.3 %) | -1.0 % | within | +291.7 % | exceeds FLAG | as fkl_fused |
| E4a 250 npp | host | 3,266 (1.9 %) | 3,224 (1.1 %) | 3,209 (2.0 %) | -1.3 % | within | -0.5 % | within |  |
| E4a 250 opencv_percrop | host | 3,171 (1.4 %) | 3,175 (0.4 %) | 3,165 (0.7 %) | +0.1 % | within | -0.3 % | within |  |
| E4a 300 fkl_fused | host | 2.736 (2.6 %) | 2.733 (3.9 %) | 12.16 (0.2 %) | -0.1 % | within | +345.1 % | exceeds FLAG | FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop |
| E4a 300 fkl_fused_half_pixel | host | 2.664 (0.5 %) | 2.699 (2.0 %) | 12.18 (0.9 %) | +1.3 % | within | +351.1 % | exceeds FLAG | as fkl_fused |
| E4a 300 npp | host | 3,888 (1.0 %) | 3,888 (0.8 %) | 3,873 (1.7 %) | -0.0 % | within | -0.4 % | within |  |
| E4a 300 opencv_percrop | host | 3,847 (1.4 %) | 3,817 (0.5 %) | 3,795 (0.8 %) | -0.8 % | within | -0.6 % | within |  |
| E4c 720x480 fkl_fused | host | 1.613 (1.8 %) | 1.668 (2.1 %) | 1.612 (1.6 %) | +3.4 % | exceeds | -3.4 % | exceeds |  |
| E4c 720x480 fkl_fused_half_pixel | host | 1.624 (0.7 %) | 1.632 (2.1 %) | 1.613 (1.1 %) | +0.5 % | within | -1.2 % | within |  |
| E4c 720x480 npp | host | 14.51 (1.6 %) | 14.44 (1.0 %) | 14.41 (2.8 %) | -0.5 % | within | -0.2 % | within |  |
| E4c 720x480 opencv_percrop | host | 12.59 (1.0 %) | 12.47 (0.4 %) | 12.57 (1.8 %) | -0.9 % | within | +0.8 % | within |  |
| E4c 1920x1080 fkl_fused | host | 1.64 (1.1 %) | 1.611 (1.6 %) | 1.615 (1.7 %) | -1.8 % | exceeds | +0.3 % | within |  |
| E4c 1920x1080 fkl_fused_half_pixel | host | 1.603 (4.9 %) | 1.622 (3.4 %) | 1.634 (2.4 %) | +1.2 % | within | +0.8 % | within |  |
| E4c 1920x1080 npp | host | 14.33 (1.9 %) | 14.41 (1.5 %) | 14.25 (2.7 %) | +0.6 % | within | -1.1 % | within |  |
| E4c 1920x1080 opencv_percrop | host | 12.41 (1.8 %) | 12.46 (3.7 %) | 12.51 (0.4 %) | +0.4 % | within | +0.4 % | within |  |
| E4c 3840x2160 fkl_fused | host | 1.616 (3.9 %) | 1.61 (2.2 %) | 1.616 (2.0 %) | -0.4 % | within | +0.4 % | within |  |
| E4c 3840x2160 fkl_fused_half_pixel | host | 1.606 (5.9 %) | 1.589 (1.2 %) | 1.616 (3.5 %) | -1.1 % | within | +1.7 % | within |  |
| E4c 3840x2160 npp | host | 14.29 (1.3 %) | 14.26 (0.8 %) | 14.31 (1.4 %) | -0.2 % | within | +0.3 % | within |  |
| E4c 3840x2160 opencv_percrop | host | 12.44 (0.3 %) | 12.34 (0.8 %) | 12.32 (1.5 %) | -0.8 % | within | -0.1 % | within |  |

E4: flagged changes over all 30 batches and 3 frames (metric, drift or cleanup, point, version, change, bound):

- enq cleanup E4a fkl_fused: 10: +18.1 % (host), 20: +42.1 % (host), 30: +54.5 % (host), 40: +78.7 % (host), 50: +85.5 % (host), 60: +103.3 % (host), 70: +117.1 % (host), 80: +129.2 % (host), 90: +138.7 % (host), 100: +150.5 % (host), 110: +170.4 % (host), 120: +177.0 % (host), 130: +179.9 % (host), 140: +198.4 % (host), 150: +217.8 % (host), 160: +218.4 % (host), 170: +209.7 % (host), 180: +233.0 % (host), 190: +243.2 % (host), 200: +246.7 % (host), 210: +262.9 % (host), 220: +266.6 % (host), 230: +274.0 % (host), 240: +284.0 % (host), 250: +287.7 % (host), 260: +303.6 % (host), 270: +307.5 % (host), 280: +316.3 % (host), 290: +317.6 % (host), 300: +336.4 % (host)
- enq cleanup E4a fkl_fused_half_pixel: 10: +19.6 % (host), 20: +45.4 % (host), 30: +60.0 % (host), 40: +77.2 % (host), 50: +81.5 % (host), 60: +102.8 % (host), 70: +122.8 % (host), 80: +132.8 % (host), 90: +143.1 % (host), 100: +148.1 % (host), 110: +173.1 % (host), 120: +177.8 % (host), 130: +185.0 % (host), 140: +199.4 % (host), 150: +217.5 % (host), 160: +215.9 % (host), 170: +208.0 % (host), 180: +232.8 % (host), 190: +234.8 % (host), 200: +240.6 % (host), 210: +256.8 % (host), 220: +265.7 % (host), 230: +274.5 % (host), 240: +279.2 % (host), 250: +281.3 % (host), 260: +303.9 % (host), 270: +303.3 % (host), 280: +308.9 % (host), 290: +316.3 % (host), 300: +346.4 % (host)
- enq drift E4a fkl_fused_half_pixel: 170: +6.6 % (host)
- gpu cleanup E4a fkl_fused: 10: +8.1 % (GPU), 20: +17.4 % (GPU), 30: +19.7 % (GPU), 40: +25.1 % (GPU), 50: +31.9 % (GPU), 60: +32.8 % (GPU), 70: +36.1 % (GPU), 90: +34.7 % (GPU), 100: +36.8 % (GPU), 110: +39.6 % (GPU), 120: +39.4 % (GPU), 130: +41.4 % (GPU), 140: +39.8 % (GPU), 150: +42.8 % (GPU), 160: +41.7 % (GPU), 170: +42.5 % (GPU), 180: +42.0 % (GPU), 190: +42.9 % (GPU), 200: +42.3 % (GPU), 210: +43.0 % (GPU), 220: +43.3 % (GPU), 230: +43.9 % (GPU), 240: +44.2 % (GPU), 250: +45.6 % (GPU), 260: +45.6 % (GPU), 270: +45.0 % (GPU), 280: +45.7 % (GPU), 290: +46.1 % (GPU), 300: +49.5 % (GPU)
- gpu cleanup E4a fkl_fused_half_pixel: 10: +7.0 % (GPU), 20: +16.8 % (GPU), 30: +19.8 % (GPU), 40: +27.6 % (GPU), 50: +30.1 % (GPU), 60: +32.4 % (GPU), 70: +36.0 % (GPU), 80: +35.6 % (GPU), 90: +37.1 % (GPU), 100: +36.8 % (GPU), 110: +39.3 % (GPU), 120: +38.7 % (GPU), 130: +41.1 % (GPU), 140: +39.8 % (GPU), 150: +41.9 % (GPU), 160: +42.0 % (GPU), 170: +42.8 % (GPU), 180: +41.2 % (GPU), 190: +42.1 % (GPU), 200: +42.6 % (GPU), 210: +43.3 % (GPU), 220: +43.6 % (GPU), 230: +43.5 % (GPU), 240: +45.1 % (GPU), 250: +45.0 % (GPU), 260: +45.3 % (GPU), 270: +44.5 % (GPU), 280: +46.3 % (GPU), 290: +46.7 % (GPU), 300: +49.0 % (GPU)
- lau cleanup E4a fkl_fused: 10: +19.2 % (host), 20: +39.4 % (host), 30: +60.9 % (host), 40: +75.5 % (host), 50: +89.1 % (host), 60: +103.0 % (host), 70: +125.2 % (host), 80: +129.0 % (host), 90: +139.2 % (host), 100: +154.1 % (host), 110: +176.1 % (host), 120: +178.0 % (host), 130: +184.4 % (host), 140: +195.4 % (host), 150: +218.0 % (host), 160: +223.6 % (host), 170: +211.1 % (host), 180: +237.3 % (host), 190: +246.5 % (host), 200: +249.8 % (host), 210: +268.5 % (host), 220: +265.8 % (host), 230: +278.7 % (host), 240: +285.7 % (host), 250: +295.0 % (host), 260: +306.2 % (host), 270: +302.5 % (host), 280: +324.7 % (host), 290: +324.2 % (host), 300: +345.1 % (host)
- lau cleanup E4a fkl_fused_half_pixel: 10: +21.6 % (host), 20: +46.6 % (host), 30: +59.0 % (host), 40: +79.5 % (host), 50: +91.0 % (host), 60: +106.2 % (host), 70: +124.3 % (host), 80: +135.6 % (host), 90: +142.2 % (host), 100: +155.4 % (host), 110: +174.9 % (host), 120: +177.7 % (host), 130: +186.7 % (host), 140: +200.2 % (host), 150: +226.1 % (host), 160: +221.8 % (host), 170: +213.1 % (host), 180: +239.3 % (host), 190: +239.5 % (host), 200: +248.1 % (host), 210: +263.1 % (host), 220: +275.4 % (host), 230: +278.0 % (host), 240: +290.3 % (host), 250: +291.7 % (host), 260: +306.2 % (host), 270: +299.8 % (host), 280: +317.5 % (host), 290: +313.9 % (host), 300: +351.1 % (host)
- lau drift E4a fkl_fused: 130: +5.4 % (host), 170: +7.9 % (host)

## Agreement rows (data rows, comment lines left out)

- test_npp_batchresize_x_split3D_agreement.csv, reference: 30 data rows per run; identical to the published run of the same index: [True, True, True]
- test_npp_batchresize_x_split3D_agreement.csv, cleaned: 30 data rows per run; identical to the published run of the same index: [True, True, True]
- test_npp_batchresize_x_split3D_No_CPU_Overhead_agreement.csv, reference: 30 data rows per run; identical to the published run of the same index: [True, True, True]
- test_npp_batchresize_x_split3D_No_CPU_Overhead_agreement.csv, cleaned: 30 data rows per run; identical to the published run of the same index: [True, True, True]
- test_NPP_cpu_batchresize_x_split3D_agreement.csv, reference: 32 data rows per run; identical to the published run of the same index: [True, True, True]
- test_NPP_cpu_batchresize_x_split3D_agreement.csv, cleaned: 32 data rows per run; identical to the published run of the same index: [True, True, True]
- e9_agreement.csv, reference: 18 data rows per run; identical to the published run of the same index: [True, True, True]
- e9_agreement.csv, cleaned: 18 data rows per run; identical to the published run of the same index: [True, True, True]
- e4_preprocessing_agreement.csv, reference (opencv_batch rows left out of both sides): (rows, same header, rows identical) per run: [(132, True, True), (132, True, True), (132, True, True)]
- e4_preprocessing_agreement.csv, cleaned (opencv_batch rows left out of both sides): (rows, same header, rows identical) per run: [(132, True, True), (132, True, True), (132, True, True)]
- CPU-time OpenCV program, reference: 1 passed, 1 passed, 1 passed (pass/fail check only; the cleaned build reads seeded noise, the reference a constant image)
- CPU-time OpenCV program, cleaned: 1 passed, 1 passed, 1 passed (pass/fail check only; the cleaned build reads seeded noise, the reference a constant image)


## Warm-pass diagnostic (window 2026-11-07 23:33:08-23:34:09)

Nsight Compute at E4c 3840 x 2160 on both builds: warm (--cache-control none) and cold (ncu default), two repetitions. The build order flipped between the repetitions, and the settings were those of window B. Output: results/rtx4090/cleanup-window-b-warm-diagnostic/. All 24 passes exited 0, with kernel counts 1, 7 and 9.

| condition | version | published | reference today (2 passes) | cleaned today (window B + 2 passes) | all 5 today: median, range | median vs published |
|---|---|---|---|---|---|---|
| warm | fkl_fused | 83.72 MiB | 110.19-110.36 | 109.53-114.14 | 110.19, 4.2 % | +31.6 % |
| warm | opencv_percrop | 1,182.73 | 1,290.12-1,290.49 | 1,288.46-1,299.64 | 1,290.12, 0.9 % | +9.1 % |
| warm | npp | 1,404.87 | 1,553.77-1,565.36 | 1,544.33-1,563.83 | 1,553.77, 1.4 % | +10.6 % |
| cold | fkl_fused | 86.13 | 100.76-107.11 | 86.47-101.59 | 100.76, 20.5 % | +17.0 % |
| cold | opencv_percrop | 1,084.30 | 1,070.79-1,089.36 | 1,072.49-1,086.64 | 1,073.37, 1.7 % | -1.0 % |
| cold | npp | 1,281.89 | 1,289.60-1,290.23 | 1,285.12-1,331.97 | 1,290.23, 3.6 % | +0.7 % |

**Findings:**
- **The reference build today does not reproduce the published warm value.** It gives 110.19 and 110.36 MiB for fkl_fused; the cleaned build gives 110.14 and 109.53. The cleanup does not change the warm numbers. The difference from the published 83.72 MiB is drift since the published passes, which were made on 2026-10-31 with the first build of the program at FKL c8fcbe8. That binary no longer exists: build/e4 was rebuilt at 51fa9e6 at 20:34 the same day.
- **Window B's 114.14 MiB is a repetition outlier, not the setup.** Its DRAM writes (86.46 MiB) equal those of the other warm passes (85.8-86.6). It has 4 MiB more reads (27.68 MiB instead of the frame's 23.7).
- **Reads are the frame; writes vary.**
  - fkl_fused reads 23.7 MiB (the frame; 24.4 to 27.7 in three passes) and writes 60 to 87 MiB of its 108 MiB output during the kernel. The rest of the output is still dirty in the 68.7 MiB L2 when the kernel ends, and is written back after it, where ncu does not count it.
  - How much is written back during the kernel varies between passes, and the kernel time with it. Writes: published 62.4 (cold) and 60.0 (warm); today warm 85.8-86.6, cold 62.7-79.9.
  - This makes the fused kernel's DRAM bytes at 4K the least repeatable number of the table (cold range 20.5 % today).
  - The multi-kernel versions move 10 to 15 times more, so the same effect is a few percent for them.
  - What must reach DRAM sooner or later for fkl_fused is 23.7 + 108.0 = 131.7 MiB.
- **Ratios against fkl_fused today:**
  - warm: OpenCV 11.4x to 11.8x (published 14.1x), NPP 13.5x to 14.3x (published 16.8x);
  - cold: OpenCV 10.2x to 12.4x (published 12.6x), NPP 12.0x to 15.4x (published 14.9x).
- **A further diagnostic is possible:** running the binary of the published quiet runs, build/e4 at 51fa9e6 with FKL c8fcbe8, would separate the old program and library from today's environment. Its kernels are the same as those of the published passes, according to the published README.
