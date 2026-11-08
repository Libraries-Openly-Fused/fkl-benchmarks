## Findings

1. **hand_roi / FKL, the hand line of the figure.** It is 0.960 at batch 10, 0.734 at 50, 0.692 at 100, 0.674 at 150 and 0.659 at 300.
   - The FKL time is the mean of fkl_fused_late and fkl_fused_after, the two FKL calls next to the hand-written kernels. Against fkl_fused the ratio is 0.945, 0.738, 0.697, 0.671 and 0.656.
   - This is a whole-call ratio (build + dispatch + kernel), E4's protocol; E1c remains the source for the kernels themselves.
   - The host part explains most of the gap. hand_roi's call, including its CPU function over every crop, takes host_enqueue_us 1.60, 2.12 and 2.77 at batch 10, 150 and 300. FKL's call takes 2.03, 7.14 and 12.31: it builds its crop views and operations on the host, and the GPU waits for that inside the interval.
2. **hand_roi / hand_batch is 0.991 to 1.092, growing with the batch.** At 300, hand_roi's call costs about 1 us more on the host than hand_batch's (2.77 against 1.78 us). Two things add to it:
   - its CPU function computes each crop's pointer, size, pitch and resize factors;
   - its parameter block is larger: 9,624 B at 300, against 2,440 B.
3. **hand_roi is bitwise equal to fkl_fused** at all 30 batches in all three runs, as are hand_batch, hand_padded, cvgs_fused and both FKL controls.
   - cvgs_fused against opencv_percrop: the largest |difference| is 7.6e-6.
   - All 936 ok_own rows pass.
4. **The position has no visible effect.**
   - From batch 20 on, fkl_fused_late / fkl_fused is 0.992 to 1.005 and fkl_fused_after / fkl_fused 0.993 to 1.007. At batch 10, the first point of a run, they are 0.973 and 0.986.
   - hand_batch / FKL repeats the window e4-hand-cvgs within 0.004: 0.926, 0.735, 0.669, 0.639 and 0.603 at 10, 50, 100, 150 and 300.
5. **cvGPUSpeedup against OpenCV-CUDA per crop: 33.4x at batch 10, 79.5x at 50, 105.3x at 100, 113.8x at 150 and 131.0x at 300.** cvgs_fused is 0.6 to 7.8 % slower than fkl_fused.
6. **Every version against the window e4-hand-cvgs** (the medians of the changes over batches 50 to 300 are within +-0.7 % for every version and metric).
   - FKL's host times are unchanged this time: -0.2 % (host_enqueue_us) and -0.4 % (host_launch_us). The 30 kernels of hand_roi did not move FKL's host code, unlike the change of #34 (`../e4-hand-cvgs-host-diagnostic/`).
   - The 11 flags are all cvgs_fused, at scattered batches and in both directions: gpu_mean_ms -3.5 to +3.1 %, host times -5.1 to +6.5 %.
   - It is a run-level effect. At those batches, cvgs_fused's host times in runs 1 and 3 are 5 to 6 % above run 2, and run 2 equals e4-hand-cvgs. The binary is the same in all three runs.
   - Probable cause, not isolated: run-to-run differences in the placement of cvGPUSpeedup's per-call host data (its array of GpuMat ROIs and their headers).
   - The speedups against OpenCV move by at most about 3 % at those batches.
7. **The clock.** The warm-up started at 210 MHz (P8). The three runs ran at 2,685 to 2,700 MHz, with no sample below 2,685 MHz.
