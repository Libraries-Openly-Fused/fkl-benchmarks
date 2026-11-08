## Findings

1. **hand_batch / FKL is 0.92 at batch 10, 0.74 at 50, 0.67 at 100, 0.64 at 150 and 0.60 at 300.** The FKL time is the mean of fkl_fused_late and fkl_fused_after, the two FKL calls next to the hand-written kernels; against fkl_fused the ratio is 0.90, 0.73, 0.67, 0.64 and 0.60.
   - The hand-written call is faster, and the gap grows with the batch, because of the host work inside the interval. FKL's call builds its crop views and operations on the host: host_enqueue_us is 2.1 at batch 10 and 12.3 at 300. The hand-written call writes 8 bytes per crop and launches: 1.6 to 1.8 us at every batch. The GPU waits for that host work inside the interval.
   - This is a whole-call ratio (build + dispatch + kernel), E4's protocol. For the kernels themselves, E1c (Table III) remains the source.
2. **The position has no visible effect.**
   - From batch 20 on, fkl_fused_late / fkl_fused is 0.990 to 1.014 and fkl_fused_after / fkl_fused 0.992 to 1.004 (medians). At batch 10, the first point of a run, they are 0.982 and 0.977.
   - The hand ratio against fkl_fused and against the adjacent FKL time differ by at most 0.018, at batch 10.
3. **hand_padded / hand_batch is 1.007 to 1.097, growing with the batch: +1.6 us at 300.**
   - This is the cost of the larger launch-parameter block, 14,448 B instead of 2,440 B at 300. The hand build writes only the descriptors, not the padding, so it is not FKL's build cost.
   - hand_padded is 0.66 to 0.97 of the adjacent FKL time.
4. **cvGPUSpeedup against OpenCV-CUDA per crop: 33x at batch 10, 81x at 50, 104x at 100, 116x at 150, 128x at 300.**
   - cvgs_fused is 1.6 to 8.2 % slower than fkl_fused. Its host part is larger: the GpuMat ROIs, their conversion to raw pointers, and the conditional batch read with usedPlanes. Its host_enqueue_us is 2.2 at batch 10 and 14.4 at 300, against FKL's 2.1 and 12.3.
   - cvGS::executeOperations is called with its default arguments, which request thread fusion. Thread fusion cannot take effect in this chain: the SASS is identical with executeOperations<false> (tf-check/).
5. **Agreement.**
   - All 846 ok_own rows pass.
   - hand_batch, hand_padded, cvgs_fused, fkl_fused_late and fkl_fused_after are bitwise equal to fkl_fused at all 30 batches in all three runs.
   - cvgs_fused against opencv_percrop: the largest |difference| is 7.6e-6, within 1e-4 everywhere.
6. **The existing series against window B.**
   - opencv_percrop, npp and every E4c frame are within (GPU -1.3 to +1.4 %).
   - fkl_fused and fkl_fused_half_pixel move by a median of +1.9 % and +1.6 % in gpu_mean_ms, flagged at E4a 40 (+4.0 %) and 90 (+3.0 %).
   - Their host times are higher from batch 50 on: host_enqueue_us +0.1 to +7.0 % (median +5.0 %), host_launch_us -0.4 to +7.2 % (median +5.4 %). These are 54 of the 56 flags.
   - The GPU times follow the host times, because the GPU waits for FKL's host work inside the interval.
   - Probable cause, not isolated: the host code of the same FKL call is compiled differently. The device code is identical (SASS), but the translation unit now also holds cvGPUSpeedup and 90 more kernels, which can change the inlining and the placement of FKL's host path.
   - Against a position effect:
     - fkl_fused_half_pixel follows fkl_fused, as in window B;
     - fkl_fused_late and fkl_fused_after, at other places in the sequence, have the same host times as fkl_fused.
   - Drift between the two days is not excluded.
   - A diagnostic would isolate it: window B's binary (6e64d7e9) and this one, alternating three times, E4a with E4_ONLY=fkl_fused,fkl_fused_half_pixel (about a minute).
7. **The clock.** The warm-up started at 210 MHz (P8). The three runs ran at a median of 2,685 MHz, with no sample below 2,550 MHz.
