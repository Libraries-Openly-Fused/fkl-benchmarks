## Findings

1. **The build-once series of Figure 15.** Medians of the three per-run ratios:

   | batch | 10 | 50 | 100 | 150 | 200 | 250 | 300 |
   |---|---|---|---|---|---|---|---|
   | opencv_percrop / fkl_fused_once | 37.7x | 111.0x | 153.3x | 176.4x | 188.9x | 198.5x | 206.4x |
   | npp / fkl_fused_half_pixel_once | 37.4x | 113.5x | 156.7x | 179.5x | 192.6x | 203.6x | 210.8x |
   | opencv_percrop / cvgs_fused_once | 37.4x | 108.2x | 150.1x | 175.2x | 189.0x | 196.1x | 205.4x |

   - The baselines are timed as a user calls them, and the build-once variants without their build.
   - The run-to-run spread of these ratios is 0.1 to 5 % at most batches.
   - The largest spreads come from a single run whose mean of 100 calls was raised by a few slow calls; its fastest call equals the other runs'. They are 12.6 % (opencv_percrop / fkl_fused_once at batch 230: fkl_fused_once 16.90 us in run 1 against 14.86 and 14.87 us) and 6.3 % (npp / fkl_fused_half_pixel_once at batch 20).
2. **hand_roi_once / FKL, both built once (the hand line).** The FKL time is the mean of fkl_fused_once and fkl_fused_once_after, the two FKL calls next to it.
   - The ratio is 1.003 at batch 10, then 0.941 to 0.977 from batch 20 on: 0.954 at 50 and 100, 0.964 at 150, 0.948 at 200, 0.967 at 250, 0.954 at 300.
   - Against fkl_fused_once it is 1.000, 0.956, 0.952, 0.962 and 0.952 at 10, 50, 100, 150 and 300.
   - At batch 30 one run gives 0.865, a spread of 9.0 %: fkl_fused_once_after averaged 5.42 us in run 1, against 4.69 and 4.70 us.
   - With the build excluded, FKL's call is within 2 to 6 % of the hand-written kernel's from batch 20 on. With the build inside the call (e4-hand-roi), the ratio was 0.96 to 0.66.
   - What remains is in the launch path. host_enqueue_us is 1.64, 2.16 and 2.71 us for fkl_fused_once at batch 10, 150 and 300, against 1.58, 1.83 and 2.19 us for hand_roi_once. Their parameter blocks at 300 are 14,496 B and 9,624 B.
3. **Build once against the whole call, the same kernels:**
   - fkl_fused_once / fkl_fused is 0.907 at batch 10 and 0.680 at 300;
   - fkl_fused_half_pixel_once / fkl_fused_half_pixel is 0.921 and 0.682;
   - cvgs_fused_once / cvgs_fused is 0.895 and 0.617;
   - hand_roi_once / hand_roi is 0.961 to 1.020.
   - So the build is 9 to 32 % of FKL's whole-call interval, and at most about 4 % of hand_roi's.
4. **The position has no visible effect.**
   - fkl_fused_once_after / fkl_fused_once is 0.986 to 1.014 from batch 20 on, and 1.015 at batch 10.
   - cvgs_fused_once / fkl_fused_once is 0.987 to 1.030.
5. **Agreement.**
   - The five build-once pairs are bitwise equal in 90 of 90 (30 batches x 3 runs): fkl_fused_once, fkl_fused_once_after, hand_roi_once and cvgs_fused_once against fkl_fused, and fkl_fused_half_pixel_once against fkl_fused_half_pixel.
   - cvgs_fused_once against opencv_percrop: the largest |difference| is 7.63e-6, as for cvgs_fused.
   - The seven pairs of e4-hand-roi give the same results as there. All 1,386 ok_own rows pass.
6. **Every existing version against the window e4-hand-roi.**
   - The versions that are not FKL stay within +-0.5 %, as medians over batches 50 to 300 of every metric. The one exception is hand_roi's host_enqueue_us, at +1.0 %.
   - FKL's four whole-call versions (fkl_fused, fkl_fused_half_pixel, fkl_fused_late, fkl_fused_after) are faster: gpu_mean_ms by 1.5 to 1.8 %, host_enqueue_us by 3.8 % and host_launch_us by 4.3 to 4.7 %.
   - That gives 63 flags. 56 host-time and 6 GPU-time flags are on those four versions. All are negative (host -5.0 to -7.7 %, GPU -3.0 to -3.3 % at batches 110 and 150), except fkl_fused_half_pixel at batch 70 (+3.0 % GPU, +6.4 % host). The last flag is cvgs_fused's host_enqueue_us at batch 300, +6.7 % with spreads of 5.5 and 6.5 %: the run-level variation of e4-hand-roi.
   - Probable cause, not isolated: the binary composition, as in the diagnostic F2 (`../e4-hand-cvgs-host-diagnostic/`). The kernels are the same (182 of 182 identical in SASS), but the host code of the FKL call is not.
     - In window G's binary, fk::executeOperations of the FKL call at batch 300 is inlined into the timed lambda; it has no symbol of its own.
     - In this binary it is an out-of-line function of 2,031 B, shared by the whole-call and the build-once calls of the same operation types.
     - The measure<> instantiations of the batch-300 lambdas also changed size, for example from 6,325 to 5,531 B.
     - Running both binaries alternately in one window (F2's method) would isolate the cause; that was not done.
   - These whole-call rows do not replace the whole-call record of e4-hand-roi.
7. **The clock.** The warm-up started at 210 MHz. The three runs ran at 2,685 MHz, with no sample below 2,685 MHz.
