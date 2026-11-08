# E4 window H (e4-build-once), RTX 4090: the build-once series of Figure 15 (report)

Written by `e4_build_once_report.py` from the result folders only, 2026-11-08. Times: gpu_mean_ms, CUDA events around one call (E4's protocol), 100 calls after one warm-up; per point the median of the three runs. Ratios: the median of the three per-run ratios, with the smallest and the largest run. The build-once variants (`*_once`) build their crop views, ROIs or parameter block and every operation once, before the timing: their interval holds only the launch path and the kernel, and they are valid only while the crops (their positions and sizes) do not change. The baselines (opencv_percrop, npp) are timed as a user calls them.

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

## The build-once series of Figure 15 (E4a, every batch)

### opencv_percrop / fkl_fused_once (speedup of FKL, build once, over OpenCV-CUDA per crop)

| batch | opencv_percrop ms | fkl_fused_once ms | ratio, median of the runs | smallest run | largest run | spread |
|---|---|---|---|---|---|---|
| 10 | 0.1286 | 0.003414 | 37.7 | 37.1 | 37.9 | 2.1 % |
| 20 | 0.2535 | 0.004198 | 60.4 | 59.7 | 60.5 | 1.4 % |
| 30 | 0.3793 | 0.004683 | 80.5 | 80.5 | 82.3 | 2.2 % |
| 40 | 0.5033 | 0.005185 | 97.1 | 96.3 | 99.0 | 2.8 % |
| 50 | 0.6329 | 0.005699 | 111.0 | 110.6 | 111.1 | 0.4 % |
| 60 | 0.7608 | 0.006215 | 122.0 | 121.0 | 122.4 | 1.1 % |
| 70 | 0.8819 | 0.006856 | 128.6 | 121.7 | 131.6 | 7.6 % |
| 80 | 1.011 | 0.007249 | 139.2 | 139.1 | 140.2 | 0.8 % |
| 90 | 1.135 | 0.007812 | 145.3 | 144.0 | 145.7 | 1.2 % |
| 100 | 1.263 | 0.008237 | 153.3 | 152.5 | 159.8 | 4.7 % |
| 110 | 1.386 | 0.008755 | 158.7 | 156.7 | 158.8 | 1.3 % |
| 120 | 1.509 | 0.009333 | 161.6 | 161.1 | 164.9 | 2.3 % |
| 130 | 1.636 | 0.009787 | 167.2 | 166.0 | 167.7 | 1.1 % |
| 140 | 1.768 | 0.0103 | 171.6 | 170.5 | 172.6 | 1.2 % |
| 150 | 1.896 | 0.01075 | 176.4 | 175.9 | 177.6 | 0.9 % |
| 160 | 2.035 | 0.01141 | 178.1 | 177.1 | 178.5 | 0.8 % |
| 170 | 2.149 | 0.01191 | 180.8 | 179.1 | 181.6 | 1.4 % |
| 180 | 2.268 | 0.01234 | 183.5 | 182.5 | 183.9 | 0.7 % |
| 190 | 2.377 | 0.01287 | 185.0 | 184.6 | 186.8 | 1.2 % |
| 200 | 2.512 | 0.01335 | 188.9 | 188.1 | 189.7 | 0.9 % |
| 210 | 2.64 | 0.0139 | 190.0 | 189.1 | 191.0 | 1.0 % |
| 220 | 2.774 | 0.01441 | 193.2 | 191.1 | 193.9 | 1.5 % |
| 230 | 2.906 | 0.01487 | 195.5 | 171.0 | 195.5 | 12.6 % |
| 240 | 3.036 | 0.0154 | 197.1 | 196.3 | 199.3 | 1.6 % |
| 250 | 3.143 | 0.01584 | 198.5 | 198.4 | 198.6 | 0.1 % |
| 260 | 3.288 | 0.01639 | 201.5 | 199.3 | 201.7 | 1.2 % |
| 270 | 3.399 | 0.01687 | 201.5 | 201.1 | 201.9 | 0.4 % |
| 280 | 3.522 | 0.01735 | 203.0 | 202.2 | 203.8 | 0.8 % |
| 290 | 3.658 | 0.0179 | 204.4 | 203.4 | 206.5 | 1.5 % |
| 300 | 3.787 | 0.01835 | 206.4 | 204.7 | 208.0 | 1.6 % |

### npp / fkl_fused_half_pixel_once (speedup of FKL with HALF_PIXEL, build once, over NPP)

| batch | npp ms | fkl_fused_half_pixel_once ms | ratio, median of the runs | smallest run | largest run | spread |
|---|---|---|---|---|---|---|
| 10 | 0.132 | 0.003515 | 37.4 | 37.2 | 38.1 | 2.3 % |
| 20 | 0.2625 | 0.004187 | 63.0 | 61.0 | 65.0 | 6.3 % |
| 30 | 0.386 | 0.004716 | 81.8 | 81.4 | 82.0 | 0.8 % |
| 40 | 0.5159 | 0.00521 | 99.7 | 98.6 | 100.6 | 2.1 % |
| 50 | 0.6454 | 0.005714 | 113.5 | 113.0 | 114.8 | 1.6 % |
| 60 | 0.7711 | 0.006271 | 122.7 | 122.7 | 126.1 | 2.8 % |
| 70 | 0.8944 | 0.006767 | 132.9 | 131.9 | 135.0 | 2.3 % |
| 80 | 1.032 | 0.007302 | 142.1 | 140.7 | 142.3 | 1.1 % |
| 90 | 1.162 | 0.007794 | 148.8 | 148.2 | 149.4 | 0.8 % |
| 100 | 1.298 | 0.008282 | 156.7 | 153.9 | 158.9 | 3.1 % |
| 110 | 1.42 | 0.008833 | 161.2 | 157.5 | 161.7 | 2.6 % |
| 120 | 1.549 | 0.009355 | 165.5 | 164.0 | 166.2 | 1.3 % |
| 130 | 1.664 | 0.009845 | 169.6 | 168.9 | 171.1 | 1.3 % |
| 140 | 1.805 | 0.01032 | 175.6 | 172.0 | 176.4 | 2.5 % |
| 150 | 1.94 | 0.01084 | 179.5 | 176.7 | 179.8 | 1.7 % |
| 160 | 2.049 | 0.01142 | 179.1 | 179.0 | 181.1 | 1.1 % |
| 170 | 2.185 | 0.01199 | 181.8 | 181.7 | 182.2 | 0.3 % |
| 180 | 2.321 | 0.01246 | 186.3 | 184.4 | 188.1 | 2.0 % |
| 190 | 2.445 | 0.01292 | 189.7 | 188.2 | 189.7 | 0.8 % |
| 200 | 2.579 | 0.01343 | 192.6 | 190.6 | 192.9 | 1.2 % |
| 210 | 2.709 | 0.01397 | 193.5 | 192.3 | 195.8 | 1.8 % |
| 220 | 2.845 | 0.01449 | 196.3 | 193.9 | 196.8 | 1.4 % |
| 230 | 2.982 | 0.01491 | 199.8 | 198.5 | 200.1 | 0.8 % |
| 240 | 3.1 | 0.01537 | 201.5 | 199.8 | 202.8 | 1.5 % |
| 250 | 3.243 | 0.01594 | 203.6 | 200.6 | 205.1 | 2.2 % |
| 260 | 3.362 | 0.01645 | 204.1 | 202.2 | 205.0 | 1.4 % |
| 270 | 3.496 | 0.01694 | 206.4 | 204.5 | 207.7 | 1.5 % |
| 280 | 3.623 | 0.01741 | 208.1 | 206.0 | 209.2 | 1.6 % |
| 290 | 3.754 | 0.0179 | 208.2 | 207.9 | 210.8 | 1.4 % |
| 300 | 3.883 | 0.01844 | 210.8 | 208.1 | 215.0 | 3.3 % |

### opencv_percrop / cvgs_fused_once (speedup of cvGPUSpeedup, build once, over OpenCV-CUDA per crop)

| batch | opencv_percrop ms | cvgs_fused_once ms | ratio, median of the runs | smallest run | largest run | spread |
|---|---|---|---|---|---|---|
| 10 | 0.1286 | 0.00345 | 37.4 | 37.2 | 37.7 | 1.4 % |
| 20 | 0.2535 | 0.004281 | 59.0 | 58.9 | 59.6 | 1.1 % |
| 30 | 0.3793 | 0.004716 | 80.4 | 79.9 | 80.7 | 0.9 % |
| 40 | 0.5033 | 0.005311 | 94.8 | 93.9 | 96.8 | 3.1 % |
| 50 | 0.6329 | 0.005847 | 108.2 | 106.6 | 109.6 | 2.7 % |
| 60 | 0.7608 | 0.006293 | 121.2 | 120.9 | 121.2 | 0.3 % |
| 70 | 0.8819 | 0.006695 | 130.8 | 129.1 | 133.2 | 3.2 % |
| 80 | 1.011 | 0.007299 | 138.8 | 135.5 | 140.6 | 3.6 % |
| 90 | 1.135 | 0.007822 | 145.1 | 144.3 | 146.0 | 1.2 % |
| 100 | 1.263 | 0.008402 | 150.1 | 150.0 | 157.5 | 5.0 % |
| 110 | 1.386 | 0.009073 | 153.2 | 152.2 | 154.0 | 1.2 % |
| 120 | 1.509 | 0.009419 | 160.7 | 159.5 | 161.7 | 1.4 % |
| 130 | 1.636 | 0.009799 | 165.9 | 164.0 | 167.7 | 2.2 % |
| 140 | 1.768 | 0.01038 | 170.3 | 169.1 | 170.3 | 0.7 % |
| 150 | 1.896 | 0.01086 | 175.2 | 174.1 | 175.6 | 0.8 % |
| 160 | 2.035 | 0.01143 | 177.5 | 176.5 | 178.1 | 0.9 % |
| 170 | 2.149 | 0.01198 | 179.5 | 178.9 | 179.8 | 0.5 % |
| 180 | 2.268 | 0.01244 | 181.5 | 180.9 | 183.6 | 1.5 % |
| 190 | 2.377 | 0.01293 | 183.8 | 183.5 | 186.1 | 1.4 % |
| 200 | 2.512 | 0.01334 | 189.0 | 187.7 | 190.3 | 1.4 % |
| 210 | 2.64 | 0.01393 | 189.4 | 189.0 | 190.6 | 0.9 % |
| 220 | 2.774 | 0.01446 | 191.9 | 191.8 | 192.5 | 0.4 % |
| 230 | 2.906 | 0.01489 | 194.5 | 194.0 | 195.2 | 0.6 % |
| 240 | 3.036 | 0.01553 | 196.2 | 195.1 | 196.6 | 0.7 % |
| 250 | 3.143 | 0.0161 | 196.1 | 195.1 | 196.5 | 0.7 % |
| 260 | 3.288 | 0.01637 | 201.5 | 199.1 | 202.0 | 1.4 % |
| 270 | 3.399 | 0.01684 | 201.8 | 200.1 | 202.9 | 1.4 % |
| 280 | 3.522 | 0.01733 | 203.5 | 202.4 | 203.5 | 0.6 % |
| 290 | 3.658 | 0.01792 | 204.1 | 203.2 | 205.4 | 1.1 % |
| 300 | 3.787 | 0.01844 | 205.4 | 203.7 | 205.9 | 1.1 % |

### hand_roi_once / FKL, build once (the hand line)

Against the mean of fkl_fused_once (timed just before hand_roi_once) and fkl_fused_once_after (timed just after it):

| batch | hand_roi_once ms | mean(fkl_fused_once, fkl_fused_once_after) ms | ratio, median of the runs | smallest run | largest run | spread |
|---|---|---|---|---|---|---|
| 10 | 0.003461 | 0.003452 | 1.003 | 0.993 | 1.015 | 2.2 % |
| 20 | 0.00404 | 0.00422 | 0.968 | 0.948 | 0.980 | 3.3 % |
| 30 | 0.004427 | 0.004703 | 0.949 | 0.865 | 0.951 | 9.0 % |
| 40 | 0.005059 | 0.005179 | 0.977 | 0.976 | 0.983 | 0.7 % |
| 50 | 0.00545 | 0.005713 | 0.954 | 0.950 | 0.965 | 1.6 % |
| 60 | 0.005982 | 0.006197 | 0.967 | 0.947 | 0.967 | 2.1 % |
| 70 | 0.006467 | 0.00681 | 0.952 | 0.923 | 0.960 | 4.0 % |
| 80 | 0.006968 | 0.007245 | 0.967 | 0.958 | 0.973 | 1.6 % |
| 90 | 0.007449 | 0.007787 | 0.953 | 0.944 | 0.960 | 1.7 % |
| 100 | 0.00787 | 0.008248 | 0.954 | 0.946 | 0.962 | 1.7 % |
| 110 | 0.00837 | 0.008759 | 0.953 | 0.949 | 0.956 | 0.6 % |
| 120 | 0.008857 | 0.009309 | 0.951 | 0.949 | 0.953 | 0.4 % |
| 130 | 0.00938 | 0.009781 | 0.959 | 0.956 | 0.961 | 0.5 % |
| 140 | 0.009923 | 0.01029 | 0.964 | 0.960 | 0.966 | 0.7 % |
| 150 | 0.01036 | 0.01073 | 0.964 | 0.960 | 0.973 | 1.3 % |
| 160 | 0.01089 | 0.01142 | 0.952 | 0.951 | 0.959 | 0.9 % |
| 170 | 0.01139 | 0.01189 | 0.959 | 0.956 | 0.959 | 0.2 % |
| 180 | 0.01173 | 0.01234 | 0.950 | 0.948 | 0.951 | 0.3 % |
| 190 | 0.01226 | 0.0129 | 0.950 | 0.947 | 0.963 | 1.6 % |
| 200 | 0.0127 | 0.01334 | 0.948 | 0.948 | 0.956 | 0.9 % |
| 210 | 0.01321 | 0.01391 | 0.950 | 0.944 | 0.951 | 0.8 % |
| 220 | 0.0136 | 0.01439 | 0.945 | 0.942 | 0.948 | 0.7 % |
| 230 | 0.01402 | 0.0149 | 0.941 | 0.883 | 0.943 | 6.4 % |
| 240 | 0.01486 | 0.01538 | 0.969 | 0.966 | 0.969 | 0.4 % |
| 250 | 0.01533 | 0.01585 | 0.967 | 0.964 | 0.973 | 1.0 % |
| 260 | 0.01576 | 0.01636 | 0.965 | 0.963 | 0.967 | 0.4 % |
| 270 | 0.0162 | 0.01687 | 0.960 | 0.955 | 0.963 | 0.8 % |
| 280 | 0.01668 | 0.01736 | 0.961 | 0.958 | 0.967 | 0.9 % |
| 290 | 0.01714 | 0.01782 | 0.963 | 0.960 | 0.965 | 0.5 % |
| 300 | 0.0175 | 0.0183 | 0.954 | 0.952 | 0.958 | 0.6 % |

Against fkl_fused_once:

| batch | hand_roi_once ms | fkl_fused_once ms | ratio, median of the runs | smallest run | largest run | spread |
|---|---|---|---|---|---|---|
| 10 | 0.003461 | 0.003414 | 1.000 | 1.000 | 1.029 | 2.9 % |
| 20 | 0.00404 | 0.004198 | 0.971 | 0.941 | 0.985 | 4.5 % |
| 30 | 0.004427 | 0.004683 | 0.949 | 0.933 | 0.956 | 2.4 % |
| 40 | 0.005059 | 0.005185 | 0.976 | 0.975 | 0.979 | 0.3 % |
| 50 | 0.00545 | 0.005699 | 0.956 | 0.948 | 0.965 | 1.8 % |
| 60 | 0.005982 | 0.006215 | 0.963 | 0.943 | 0.965 | 2.3 % |
| 70 | 0.006467 | 0.006856 | 0.945 | 0.892 | 0.960 | 7.2 % |
| 80 | 0.006968 | 0.007249 | 0.965 | 0.956 | 0.973 | 1.7 % |
| 90 | 0.007449 | 0.007812 | 0.954 | 0.940 | 0.957 | 1.9 % |
| 100 | 0.00787 | 0.008237 | 0.952 | 0.949 | 0.961 | 1.3 % |
| 110 | 0.00837 | 0.008755 | 0.952 | 0.948 | 0.958 | 1.1 % |
| 120 | 0.008857 | 0.009333 | 0.955 | 0.949 | 0.955 | 0.7 % |
| 130 | 0.00938 | 0.009787 | 0.958 | 0.956 | 0.961 | 0.5 % |
| 140 | 0.009923 | 0.0103 | 0.964 | 0.964 | 0.964 | 0.1 % |
| 150 | 0.01036 | 0.01075 | 0.962 | 0.958 | 0.971 | 1.3 % |
| 160 | 0.01089 | 0.01141 | 0.957 | 0.953 | 0.958 | 0.6 % |
| 170 | 0.01139 | 0.01191 | 0.958 | 0.955 | 0.959 | 0.4 % |
| 180 | 0.01173 | 0.01234 | 0.951 | 0.949 | 0.951 | 0.2 % |
| 190 | 0.01226 | 0.01287 | 0.952 | 0.949 | 0.962 | 1.4 % |
| 200 | 0.0127 | 0.01335 | 0.948 | 0.948 | 0.957 | 0.9 % |
| 210 | 0.01321 | 0.0139 | 0.949 | 0.946 | 0.952 | 0.7 % |
| 220 | 0.0136 | 0.01441 | 0.944 | 0.938 | 0.951 | 1.4 % |
| 230 | 0.01402 | 0.01487 | 0.942 | 0.830 | 0.943 | 11.9 % |
| 240 | 0.01486 | 0.0154 | 0.968 | 0.962 | 0.968 | 0.6 % |
| 250 | 0.01533 | 0.01584 | 0.968 | 0.962 | 0.973 | 1.1 % |
| 260 | 0.01576 | 0.01639 | 0.964 | 0.961 | 0.967 | 0.6 % |
| 270 | 0.0162 | 0.01687 | 0.959 | 0.955 | 0.964 | 1.0 % |
| 280 | 0.01668 | 0.01735 | 0.962 | 0.959 | 0.965 | 0.7 % |
| 290 | 0.01714 | 0.0179 | 0.959 | 0.957 | 0.965 | 0.9 % |
| 300 | 0.0175 | 0.01835 | 0.952 | 0.951 | 0.960 | 1.0 % |

## The build-once variants against their whole-call versions, and the position

| batch | fkl_fused_once / fkl_fused | fkl_fused_half_pixel_once / fkl_fused_half_pixel | cvgs_fused_once / cvgs_fused | hand_roi_once / hand_roi | fkl_fused_once_after / fkl_fused_once (min - max) | cvgs_fused_once / fkl_fused_once |
|---|---|---|---|---|---|---|
| 10 | 0.907 | 0.921 | 0.895 | 0.970 | 1.016 (0.994 - 1.028) | 1.013 |
| 20 | 0.855 | 0.860 | 0.859 | 1.020 | 1.005 (0.987 - 1.010) | 1.025 |
| 30 | 0.840 | 0.838 | 0.805 | 0.982 | 1.014 (0.996 - 1.158) | 1.007 |
| 40 | 0.761 | 0.762 | 0.751 | 1.000 | 0.998 (0.990 - 0.998) | 1.024 |
| 50 | 0.754 | 0.758 | 0.732 | 0.989 | 1.001 (0.996 - 1.005) | 1.026 |
| 60 | 0.755 | 0.760 | 0.711 | 0.986 | 0.992 (0.991 - 0.994) | 1.007 |
| 70 | 0.734 | 0.718 | 0.702 | 0.981 | 0.986 (0.933 - 1.000) | 0.987 |
| 80 | 0.712 | 0.710 | 0.707 | 0.996 | 0.997 (0.995 - 0.999) | 1.002 |
| 90 | 0.718 | 0.712 | 0.696 | 0.986 | 0.994 (0.991 - 1.002) | 0.998 |
| 100 | 0.705 | 0.698 | 0.693 | 0.973 | 0.998 (0.996 - 1.006) | 1.017 |
| 110 | 0.724 | 0.727 | 0.693 | 0.966 | 0.999 (0.997 - 1.005) | 1.030 |
| 120 | 0.717 | 0.714 | 0.668 | 0.961 | 1.005 (0.994 - 1.012) | 1.010 |
| 130 | 0.711 | 0.710 | 0.673 | 0.979 | 0.999 (0.993 - 1.004) | 1.001 |
| 140 | 0.709 | 0.703 | 0.676 | 0.977 | 1.001 (0.995 - 1.009) | 1.007 |
| 150 | 0.707 | 0.710 | 0.653 | 0.972 | 0.995 (0.995 - 0.997) | 1.010 |
| 160 | 0.702 | 0.704 | 0.651 | 0.973 | 1.001 (0.998 - 1.014) | 1.003 |
| 170 | 0.699 | 0.704 | 0.662 | 0.986 | 0.999 (0.998 - 1.000) | 1.007 |
| 180 | 0.699 | 0.704 | 0.650 | 0.967 | 1.002 (1.001 - 1.002) | 1.009 |
| 190 | 0.698 | 0.701 | 0.643 | 0.972 | 1.004 (0.999 - 1.004) | 1.006 |
| 200 | 0.695 | 0.697 | 0.653 | 0.978 | 1.000 (0.999 - 1.002) | 0.999 |
| 210 | 0.697 | 0.698 | 0.654 | 0.978 | 1.001 (0.998 - 1.004) | 1.002 |
| 220 | 0.694 | 0.696 | 0.636 | 0.967 | 0.998 (0.993 - 1.006) | 1.004 |
| 230 | 0.693 | 0.693 | 0.635 | 0.967 | 0.998 (0.881 - 1.004) | 1.002 |
| 240 | 0.693 | 0.687 | 0.646 | 0.976 | 0.997 (0.993 - 0.997) | 1.006 |
| 250 | 0.686 | 0.688 | 0.638 | 0.971 | 1.000 (0.997 - 1.003) | 1.013 |
| 260 | 0.685 | 0.687 | 0.625 | 0.978 | 0.998 (0.996 - 1.000) | 1.000 |
| 270 | 0.684 | 0.685 | 0.623 | 0.975 | 0.999 (0.997 - 1.002) | 0.998 |
| 280 | 0.682 | 0.682 | 0.640 | 0.976 | 1.001 (0.995 - 1.002) | 0.999 |
| 290 | 0.680 | 0.679 | 0.637 | 0.974 | 0.994 (0.991 - 1.002) | 1.001 |
| 300 | 0.680 | 0.683 | 0.618 | 0.964 | 0.997 (0.995 - 1.003) | 1.005 |

### Host time of one call (host_enqueue_us: the call returns, the stream idle before it; median of the three runs, us)

| batch | fkl_fused | fkl_fused_once | fkl_fused_once_after | fkl_fused_half_pixel_once | hand_roi | hand_roi_once | cvgs_fused | cvgs_fused_once |
|---|---|---|---|---|---|---|---|---|
| 10 | 2.014 | 1.641 | 1.629 | 1.638 | 1.605 | 1.583 | 2.114 | 1.667 |
| 20 | 2.461 | 1.713 | 1.655 | 1.67 | 1.641 | 1.594 | 2.591 | 1.722 |
| 30 | 2.695 | 1.672 | 1.699 | 1.694 | 1.695 | 1.619 | 2.885 | 1.72 |
| 40 | 3.145 | 1.703 | 1.711 | 1.707 | 1.689 | 1.613 | 3.308 | 1.76 |
| 50 | 3.421 | 1.788 | 1.8 | 1.84 | 1.733 | 1.634 | 3.77 | 1.803 |
| 60 | 3.728 | 1.818 | 1.796 | 1.797 | 1.772 | 1.676 | 4.182 | 1.83 |
| 70 | 4.179 | 1.839 | 1.829 | 1.843 | 1.851 | 1.686 | 4.543 | 1.869 |
| 80 | 4.587 | 1.891 | 1.871 | 1.878 | 1.88 | 1.711 | 4.888 | 1.951 |
| 90 | 4.911 | 1.898 | 1.902 | 1.893 | 1.928 | 1.765 | 5.211 | 1.954 |
| 100 | 5.274 | 1.95 | 1.954 | 1.951 | 1.974 | 1.724 | 5.765 | 2.002 |
| 110 | 5.308 | 1.994 | 1.977 | 2.025 | 1.98 | 1.74 | 6.29 | 2.018 |
| 120 | 5.623 | 2.026 | 2.044 | 2.045 | 2.015 | 1.772 | 6.731 | 2.048 |
| 130 | 6.061 | 2.13 | 2.119 | 2.107 | 2.103 | 1.808 | 6.825 | 2.104 |
| 140 | 6.301 | 2.107 | 2.116 | 2.097 | 2.095 | 1.847 | 7.279 | 2.168 |
| 150 | 6.69 | 2.163 | 2.159 | 2.162 | 2.185 | 1.827 | 7.972 | 2.171 |
| 160 | 7.097 | 2.251 | 2.236 | 2.195 | 2.219 | 1.881 | 8.443 | 2.236 |
| 170 | 7.44 | 2.282 | 2.285 | 2.291 | 2.222 | 1.89 | 8.456 | 2.29 |
| 180 | 7.561 | 2.291 | 2.28 | 2.236 | 2.257 | 1.965 | 9.268 | 2.287 |
| 190 | 7.923 | 2.312 | 2.316 | 2.305 | 2.327 | 1.927 | 9.664 | 2.352 |
| 200 | 8.24 | 2.328 | 2.344 | 2.37 | 2.362 | 1.958 | 9.654 | 2.395 |
| 210 | 8.586 | 2.349 | 2.353 | 2.359 | 2.415 | 2.025 | 10.03 | 2.384 |
| 220 | 8.972 | 2.409 | 2.388 | 2.419 | 2.433 | 2.068 | 11.03 | 2.469 |
| 230 | 9.321 | 2.449 | 2.47 | 2.461 | 2.475 | 2.022 | 11.41 | 2.498 |
| 240 | 9.535 | 2.485 | 2.479 | 2.506 | 2.538 | 2.043 | 11.25 | 2.538 |
| 250 | 10.09 | 2.518 | 2.525 | 2.551 | 2.586 | 2.112 | 11.71 | 2.558 |
| 260 | 10.41 | 2.586 | 2.582 | 2.616 | 2.639 | 2.084 | 12.87 | 2.632 |
| 270 | 10.68 | 2.63 | 2.636 | 2.651 | 2.679 | 2.088 | 13.15 | 2.696 |
| 280 | 11.06 | 2.685 | 2.696 | 2.648 | 2.693 | 2.137 | 12.98 | 2.755 |
| 290 | 11.37 | 2.644 | 2.708 | 2.689 | 2.804 | 2.156 | 13.34 | 2.738 |
| 300 | 11.67 | 2.714 | 2.734 | 2.782 | 2.832 | 2.193 | 14.55 | 2.764 |

## Every existing version against the window e4-hand-roi (standing form)

old = the window e4-hand-roi (main 469aa6d), new = this window (the same 182 kernels; the build-once variants are timed after all the existing ones, which keep their order, code and buffers); median of three runs, spread = (largest - smallest) / median. The diagnostic F2 showed that the host code of the same call can change with the content of the translation unit.

### gpu_mean_ms

| point | version | bound | old (spread) | new (spread) | change | |
|---|---|---|---|---|---|---|
| E4a 10 | fkl_fused | GPU | 0.003812 (2.0 %) | 0.003768 (1.7 %) | -1.2 % | within |
| E4a 10 | fkl_fused_half_pixel | GPU | 0.003814 (2.1 %) | 0.003813 (2.3 %) | -0.0 % | within |
| E4a 10 | opencv_percrop | host | 0.1286 (1.3 %) | 0.1286 (0.9 %) | -0.1 % | within |
| E4a 10 | npp | host | 0.1313 (0.8 %) | 0.132 (2.2 %) | +0.5 % | within |
| E4a 10 | fkl_fused_late | GPU | 0.003736 (2.0 %) | 0.003697 (1.2 %) | -1.0 % | within |
| E4a 10 | hand_batch | GPU | 0.003442 (2.9 %) | 0.003509 (5.4 %) | +1.9 % | within |
| E4a 10 | hand_padded | GPU | 0.003621 (2.2 %) | 0.003561 (2.0 %) | -1.7 % | within |
| E4a 10 | hand_roi | GPU | 0.003579 (2.5 %) | 0.003521 (4.7 %) | -1.6 % | within |
| E4a 10 | fkl_fused_after | GPU | 0.003762 (2.2 %) | 0.003703 (1.3 %) | -1.6 % | within |
| E4a 10 | cvgs_fused | GPU | 0.00385 (2.0 %) | 0.00382 (3.2 %) | -0.8 % | within |
| E4a 50 | fkl_fused | GPU | 0.00755 (0.7 %) | 0.007579 (1.9 %) | +0.4 % | within |
| E4a 50 | fkl_fused_half_pixel | GPU | 0.007601 (1.4 %) | 0.00754 (0.9 %) | -0.8 % | within |
| E4a 50 | opencv_percrop | host | 0.6309 (0.4 %) | 0.6329 (0.3 %) | +0.3 % | within |
| E4a 50 | npp | host | 0.6468 (0.7 %) | 0.6454 (2.1 %) | -0.2 % | within |
| E4a 50 | fkl_fused_late | GPU | 0.007524 (1.4 %) | 0.007494 (1.0 %) | -0.4 % | within |
| E4a 50 | hand_batch | GPU | 0.005554 (1.4 %) | 0.005572 (2.6 %) | +0.3 % | within |
| E4a 50 | hand_padded | GPU | 0.005642 (1.1 %) | 0.005679 (0.7 %) | +0.7 % | within |
| E4a 50 | hand_roi | GPU | 0.005532 (1.1 %) | 0.005533 (1.9 %) | +0.0 % | within |
| E4a 50 | fkl_fused_after | GPU | 0.007564 (0.6 %) | 0.007486 (1.0 %) | -1.0 % | within |
| E4a 50 | cvgs_fused | GPU | 0.007951 (6.8 %) | 0.007988 (2.2 %) | +0.5 % | within |
| E4a 100 | fkl_fused | GPU | 0.01161 (1.5 %) | 0.01168 (2.4 %) | +0.6 % | within |
| E4a 100 | fkl_fused_half_pixel | GPU | 0.01164 (2.4 %) | 0.01185 (2.9 %) | +1.8 % | within |
| E4a 100 | opencv_percrop | host | 1.261 (2.8 %) | 1.263 (4.4 %) | +0.1 % | within |
| E4a 100 | npp | host | 1.293 (3.4 %) | 1.298 (2.4 %) | +0.4 % | within |
| E4a 100 | fkl_fused_late | GPU | 0.01162 (1.0 %) | 0.01165 (3.1 %) | +0.2 % | within |
| E4a 100 | hand_batch | GPU | 0.007784 (0.5 %) | 0.007807 (0.9 %) | +0.3 % | within |
| E4a 100 | hand_padded | GPU | 0.0081 (1.3 %) | 0.008222 (0.7 %) | +1.5 % | exceeds |
| E4a 100 | hand_roi | GPU | 0.00805 (1.8 %) | 0.008085 (1.9 %) | +0.4 % | within |
| E4a 100 | fkl_fused_after | GPU | 0.0116 (0.8 %) | 0.01166 (4.7 %) | +0.5 % | within |
| E4a 100 | cvgs_fused | GPU | 0.01194 (2.6 %) | 0.01211 (2.0 %) | +1.4 % | within |
| E4a 150 | fkl_fused | GPU | 0.01567 (0.5 %) | 0.01522 (0.3 %) | -2.9 % | exceeds |
| E4a 150 | fkl_fused_half_pixel | GPU | 0.01579 (0.8 %) | 0.01528 (0.6 %) | -3.2 % | exceeds FLAG |
| E4a 150 | opencv_percrop | host | 1.885 (0.6 %) | 1.896 (0.6 %) | +0.6 % | within |
| E4a 150 | npp | host | 1.924 (1.0 %) | 1.94 (1.7 %) | +0.9 % | within |
| E4a 150 | fkl_fused_late | GPU | 0.01563 (0.7 %) | 0.01521 (1.3 %) | -2.7 % | exceeds |
| E4a 150 | hand_batch | GPU | 0.009971 (0.6 %) | 0.01 (0.4 %) | +0.3 % | within |
| E4a 150 | hand_padded | GPU | 0.01065 (0.5 %) | 0.01069 (1.7 %) | +0.3 % | within |
| E4a 150 | hand_roi | GPU | 0.01052 (0.8 %) | 0.01064 (0.7 %) | +1.1 % | exceeds |
| E4a 150 | fkl_fused_after | GPU | 0.01565 (0.5 %) | 0.01518 (1.2 %) | -3.0 % | exceeds FLAG |
| E4a 150 | cvgs_fused | GPU | 0.01652 (3.1 %) | 0.01664 (3.2 %) | +0.8 % | within |
| E4a 200 | fkl_fused | GPU | 0.01955 (1.5 %) | 0.01921 (0.3 %) | -1.7 % | exceeds |
| E4a 200 | fkl_fused_half_pixel | GPU | 0.01954 (1.3 %) | 0.01925 (0.3 %) | -1.5 % | exceeds |
| E4a 200 | opencv_percrop | host | 2.52 (0.6 %) | 2.512 (1.1 %) | -0.3 % | within |
| E4a 200 | npp | host | 2.576 (1.2 %) | 2.579 (1.2 %) | +0.1 % | within |
| E4a 200 | fkl_fused_late | GPU | 0.01948 (0.8 %) | 0.01923 (0.7 %) | -1.3 % | exceeds |
| E4a 200 | hand_batch | GPU | 0.01212 (0.3 %) | 0.01218 (0.6 %) | +0.5 % | within |
| E4a 200 | hand_padded | GPU | 0.01319 (0.4 %) | 0.01324 (0.3 %) | +0.4 % | within |
| E4a 200 | hand_roi | GPU | 0.01297 (0.7 %) | 0.01301 (0.5 %) | +0.3 % | within |
| E4a 200 | fkl_fused_after | GPU | 0.01957 (1.7 %) | 0.01929 (1.1 %) | -1.4 % | within |
| E4a 200 | cvgs_fused | GPU | 0.02063 (2.5 %) | 0.02042 (2.9 %) | -1.0 % | within |
| E4a 250 | fkl_fused | GPU | 0.02349 (0.5 %) | 0.02317 (1.2 %) | -1.4 % | exceeds |
| E4a 250 | fkl_fused_half_pixel | GPU | 0.02345 (0.3 %) | 0.02325 (1.8 %) | -0.9 % | within |
| E4a 250 | opencv_percrop | host | 3.149 (0.4 %) | 3.143 (0.6 %) | -0.2 % | within |
| E4a 250 | npp | host | 3.213 (1.1 %) | 3.243 (1.9 %) | +1.0 % | within |
| E4a 250 | fkl_fused_late | GPU | 0.02346 (0.9 %) | 0.02307 (0.9 %) | -1.7 % | exceeds |
| E4a 250 | hand_batch | GPU | 0.01457 (0.3 %) | 0.0146 (1.0 %) | +0.2 % | within |
| E4a 250 | hand_padded | GPU | 0.0159 (0.4 %) | 0.01589 (0.6 %) | -0.1 % | within |
| E4a 250 | hand_roi | GPU | 0.01578 (0.1 %) | 0.01579 (0.2 %) | +0.0 % | within |
| E4a 250 | fkl_fused_after | GPU | 0.02351 (0.2 %) | 0.02315 (1.5 %) | -1.5 % | exceeds |
| E4a 250 | cvgs_fused | GPU | 0.02533 (2.7 %) | 0.02507 (3.1 %) | -1.0 % | within |
| E4a 300 | fkl_fused | GPU | 0.02766 (3.6 %) | 0.02706 (0.5 %) | -2.2 % | within |
| E4a 300 | fkl_fused_half_pixel | GPU | 0.02764 (4.1 %) | 0.02699 (0.4 %) | -2.4 % | within |
| E4a 300 | opencv_percrop | host | 3.78 (0.5 %) | 3.787 (0.5 %) | +0.2 % | within |
| E4a 300 | npp | host | 3.856 (1.5 %) | 3.883 (3.5 %) | +0.7 % | within |
| E4a 300 | fkl_fused_late | GPU | 0.02755 (3.3 %) | 0.02696 (0.4 %) | -2.1 % | within |
| E4a 300 | hand_batch | GPU | 0.01661 (0.2 %) | 0.01659 (0.4 %) | -0.1 % | within |
| E4a 300 | hand_padded | GPU | 0.01812 (0.3 %) | 0.01812 (1.0 %) | -0.0 % | within |
| E4a 300 | hand_roi | GPU | 0.01814 (0.2 %) | 0.01817 (0.4 %) | +0.1 % | within |
| E4a 300 | fkl_fused_after | GPU | 0.02755 (3.8 %) | 0.02701 (1.6 %) | -2.0 % | within |
| E4a 300 | cvgs_fused | GPU | 0.02874 (3.9 %) | 0.0298 (2.9 %) | +3.7 % | within FLAG |
| E4c 720 | fkl_fused | GPU | 0.005207 (1.2 %) | 0.005256 (2.2 %) | +0.9 % | within |
| E4c 720 | fkl_fused_half_pixel | GPU | 0.005253 (1.2 %) | 0.005232 (2.4 %) | -0.4 % | within |
| E4c 720 | opencv_percrop | GPU | 0.02769 (0.3 %) | 0.02761 (0.2 %) | -0.3 % | within |
| E4c 720 | npp | GPU | 0.03616 (0.2 %) | 0.03614 (0.1 %) | -0.1 % | within |
| E4c 1920 | fkl_fused | GPU | 0.0155 (0.4 %) | 0.01551 (0.3 %) | +0.1 % | within |
| E4c 1920 | fkl_fused_half_pixel | GPU | 0.01563 (0.5 %) | 0.01561 (0.3 %) | -0.1 % | within |
| E4c 1920 | opencv_percrop | GPU | 0.1744 (0.2 %) | 0.1745 (0.2 %) | +0.1 % | within |
| E4c 1920 | npp | GPU | 0.1553 (0.1 %) | 0.1553 (0.1 %) | -0.0 % | within |
| E4c 3840 | fkl_fused | GPU | 0.1578 (0.1 %) | 0.1579 (0.1 %) | +0.1 % | within |
| E4c 3840 | fkl_fused_half_pixel | GPU | 0.1578 (0.1 %) | 0.1578 (0.1 %) | +0.0 % | within |
| E4c 3840 | opencv_percrop | GPU | 1.605 (0.0 %) | 1.605 (0.0 %) | +0.0 % | within |
| E4c 3840 | npp | GPU | 1.846 (0.1 %) | 1.847 (0.1 %) | +0.0 % | within |

Median change over E4a batches 50 to 300, per version (gpu_mean_ms): fkl_fused -1.8 %, fkl_fused_half_pixel -1.5 %, opencv_percrop +0.2 %, npp +0.5 %, fkl_fused_late -1.7 %, hand_batch +0.2 %, hand_padded +0.2 %, hand_roi +0.1 %, fkl_fused_after -1.8 %, cvgs_fused +0.5 %

### host_enqueue_us

| point | version | bound | old (spread) | new (spread) | change | |
|---|---|---|---|---|---|---|
| E4a 10 | fkl_fused | host | 2.034 (3.8 %) | 2.014 (2.1 %) | -1.0 % | within |
| E4a 10 | fkl_fused_half_pixel | host | 2.074 (3.4 %) | 2.042 (1.7 %) | -1.5 % | within |
| E4a 10 | opencv_percrop | host | 126.5 (1.1 %) | 126.6 (0.8 %) | +0.1 % | within |
| E4a 10 | npp | host | 129.8 (1.0 %) | 129.7 (1.8 %) | -0.0 % | within |
| E4a 10 | fkl_fused_late | host | 2.025 (25.1 %) | 1.965 (2.1 %) | -3.0 % | within |
| E4a 10 | hand_batch | host | 1.588 (2.1 %) | 1.652 (4.6 %) | +4.0 % | within |
| E4a 10 | hand_padded | host | 1.611 (0.4 %) | 1.625 (3.6 %) | +0.8 % | within |
| E4a 10 | hand_roi | host | 1.601 (1.9 %) | 1.605 (1.5 %) | +0.3 % | within |
| E4a 10 | fkl_fused_after | host | 2.012 (0.9 %) | 1.986 (2.0 %) | -1.3 % | within |
| E4a 10 | cvgs_fused | host | 2.085 (2.2 %) | 2.114 (2.6 %) | +1.4 % | within |
| E4a 50 | fkl_fused | host | 3.39 (1.8 %) | 3.421 (6.1 %) | +0.9 % | within |
| E4a 50 | fkl_fused_half_pixel | host | 3.43 (0.2 %) | 3.314 (6.7 %) | -3.4 % | within |
| E4a 50 | opencv_percrop | host | 634.2 (1.4 %) | 633.4 (1.2 %) | -0.1 % | within |
| E4a 50 | npp | host | 643.9 (1.5 %) | 639 (1.8 %) | -0.8 % | within |
| E4a 50 | fkl_fused_late | host | 3.423 (1.4 %) | 3.314 (5.3 %) | -3.2 % | within |
| E4a 50 | hand_batch | host | 1.631 (2.4 %) | 1.652 (0.5 %) | +1.3 % | within |
| E4a 50 | hand_padded | host | 1.744 (0.9 %) | 1.75 (2.5 %) | +0.3 % | within |
| E4a 50 | hand_roi | host | 1.738 (3.1 %) | 1.733 (0.7 %) | -0.3 % | within |
| E4a 50 | fkl_fused_after | host | 3.435 (0.5 %) | 3.314 (7.5 %) | -3.5 % | within |
| E4a 50 | cvgs_fused | host | 3.72 (2.8 %) | 3.77 (1.2 %) | +1.3 % | within |
| E4a 100 | fkl_fused | host | 5.1 (3.9 %) | 5.274 (1.4 %) | +3.4 % | within |
| E4a 100 | fkl_fused_half_pixel | host | 5.15 (7.7 %) | 5.316 (2.6 %) | +3.2 % | within |
| E4a 100 | opencv_percrop | host | 1,266 (0.8 %) | 1,266 (1.3 %) | -0.0 % | within |
| E4a 100 | npp | host | 1,281 (0.8 %) | 1,281 (1.2 %) | +0.0 % | within |
| E4a 100 | fkl_fused_late | host | 5.077 (1.8 %) | 5.262 (1.4 %) | +3.6 % | exceeds |
| E4a 100 | hand_batch | host | 1.677 (3.5 %) | 1.655 (1.6 %) | -1.3 % | within |
| E4a 100 | hand_padded | host | 1.884 (4.1 %) | 1.945 (1.2 %) | +3.3 % | within |
| E4a 100 | hand_roi | host | 1.95 (0.6 %) | 1.974 (2.1 %) | +1.2 % | within |
| E4a 100 | fkl_fused_after | host | 5.122 (0.9 %) | 5.294 (0.9 %) | +3.4 % | exceeds |
| E4a 100 | cvgs_fused | host | 5.682 (4.9 %) | 5.765 (3.3 %) | +1.5 % | within |
| E4a 150 | fkl_fused | host | 7.137 (1.2 %) | 6.69 (2.5 %) | -6.3 % | exceeds FLAG |
| E4a 150 | fkl_fused_half_pixel | host | 7.126 (1.4 %) | 6.667 (1.0 %) | -6.4 % | exceeds FLAG |
| E4a 150 | opencv_percrop | host | 1,891 (0.4 %) | 1,901 (1.4 %) | +0.5 % | within |
| E4a 150 | npp | host | 1,913 (1.2 %) | 1,914 (1.3 %) | +0.1 % | within |
| E4a 150 | fkl_fused_late | host | 7.127 (0.3 %) | 6.678 (1.5 %) | -6.3 % | exceeds FLAG |
| E4a 150 | hand_batch | host | 1.68 (4.1 %) | 1.68 (1.3 %) | -0.0 % | within |
| E4a 150 | hand_padded | host | 2.131 (1.2 %) | 2.159 (1.5 %) | +1.3 % | within |
| E4a 150 | hand_roi | host | 2.117 (3.9 %) | 2.185 (1.8 %) | +3.2 % | within |
| E4a 150 | fkl_fused_after | host | 7.124 (0.3 %) | 6.706 (1.6 %) | -5.9 % | exceeds FLAG |
| E4a 150 | cvgs_fused | host | 7.955 (5.1 %) | 7.972 (7.2 %) | +0.2 % | within |
| E4a 200 | fkl_fused | host | 8.627 (1.0 %) | 8.24 (0.7 %) | -4.5 % | exceeds |
| E4a 200 | fkl_fused_half_pixel | host | 8.585 (1.3 %) | 8.292 (0.5 %) | -3.4 % | exceeds |
| E4a 200 | opencv_percrop | host | 2,534 (1.2 %) | 2,536 (0.9 %) | +0.1 % | within |
| E4a 200 | npp | host | 2,567 (1.1 %) | 2,555 (1.3 %) | -0.4 % | within |
| E4a 200 | fkl_fused_late | host | 8.595 (0.9 %) | 8.246 (0.7 %) | -4.1 % | exceeds |
| E4a 200 | hand_batch | host | 1.702 (0.9 %) | 1.687 (1.6 %) | -0.9 % | within |
| E4a 200 | hand_padded | host | 2.313 (4.1 %) | 2.347 (1.7 %) | +1.5 % | within |
| E4a 200 | hand_roi | host | 2.339 (1.6 %) | 2.362 (0.9 %) | +1.0 % | within |
| E4a 200 | fkl_fused_after | host | 8.526 (1.0 %) | 8.23 (0.8 %) | -3.5 % | exceeds |
| E4a 200 | cvgs_fused | host | 9.606 (6.3 %) | 9.654 (5.1 %) | +0.5 % | within |
| E4a 250 | fkl_fused | host | 10.48 (1.0 %) | 10.09 (3.1 %) | -3.7 % | exceeds |
| E4a 250 | fkl_fused_half_pixel | host | 10.45 (1.0 %) | 10.05 (3.2 %) | -3.9 % | exceeds |
| E4a 250 | opencv_percrop | host | 3,157 (0.4 %) | 3,170 (0.8 %) | +0.4 % | within |
| E4a 250 | npp | host | 3,201 (1.2 %) | 3,217 (2.2 %) | +0.5 % | within |
| E4a 250 | fkl_fused_late | host | 10.43 (2.5 %) | 10.04 (2.8 %) | -3.7 % | exceeds |
| E4a 250 | hand_batch | host | 1.739 (1.5 %) | 1.742 (1.5 %) | +0.2 % | within |
| E4a 250 | hand_padded | host | 2.557 (1.4 %) | 2.533 (1.5 %) | -0.9 % | within |
| E4a 250 | hand_roi | host | 2.557 (1.8 %) | 2.586 (0.4 %) | +1.1 % | within |
| E4a 250 | fkl_fused_after | host | 10.43 (1.6 %) | 10 (3.0 %) | -4.1 % | exceeds |
| E4a 250 | cvgs_fused | host | 12.21 (8.2 %) | 11.71 (5.5 %) | -4.1 % | within |
| E4a 300 | fkl_fused | host | 12.31 (1.1 %) | 11.67 (1.8 %) | -5.2 % | exceeds FLAG |
| E4a 300 | fkl_fused_half_pixel | host | 12.33 (2.6 %) | 11.73 (2.5 %) | -4.9 % | exceeds |
| E4a 300 | opencv_percrop | host | 3,797 (0.4 %) | 3,792 (0.2 %) | -0.1 % | within |
| E4a 300 | npp | host | 3,846 (1.3 %) | 3,829 (0.9 %) | -0.4 % | within |
| E4a 300 | fkl_fused_late | host | 12.32 (2.0 %) | 11.74 (0.9 %) | -4.7 % | exceeds |
| E4a 300 | hand_batch | host | 1.784 (0.8 %) | 1.841 (2.7 %) | +3.2 % | exceeds |
| E4a 300 | hand_padded | host | 2.741 (4.2 %) | 2.714 (1.6 %) | -1.0 % | within |
| E4a 300 | hand_roi | host | 2.768 (1.4 %) | 2.832 (2.4 %) | +2.3 % | within |
| E4a 300 | fkl_fused_after | host | 12.37 (1.2 %) | 11.71 (1.3 %) | -5.4 % | exceeds FLAG |
| E4a 300 | cvgs_fused | host | 13.64 (5.5 %) | 14.55 (6.5 %) | +6.7 % | exceeds FLAG |
| E4c 720 | fkl_fused | host | 1.591 (2.8 %) | 1.61 (3.5 %) | +1.2 % | within |
| E4c 720 | fkl_fused_half_pixel | host | 1.619 (1.1 %) | 1.624 (1.3 %) | +0.3 % | within |
| E4c 720 | opencv_percrop | host | 12.43 (3.5 %) | 12.52 (1.9 %) | +0.7 % | within |
| E4c 720 | npp | host | 14.38 (0.4 %) | 14.38 (2.6 %) | -0.0 % | within |
| E4c 1920 | fkl_fused | host | 1.652 (2.0 %) | 1.631 (1.1 %) | -1.3 % | within |
| E4c 1920 | fkl_fused_half_pixel | host | 1.6 (1.5 %) | 1.629 (1.7 %) | +1.8 % | exceeds |
| E4c 1920 | opencv_percrop | host | 12.54 (0.6 %) | 12.65 (1.1 %) | +0.9 % | within |
| E4c 1920 | npp | host | 14.4 (0.7 %) | 14.42 (2.6 %) | +0.2 % | within |
| E4c 3840 | fkl_fused | host | 1.663 (2.3 %) | 1.654 (3.6 %) | -0.6 % | within |
| E4c 3840 | fkl_fused_half_pixel | host | 1.674 (5.6 %) | 1.643 (3.1 %) | -1.9 % | within |
| E4c 3840 | opencv_percrop | host | 12.95 (2.3 %) | 13.06 (2.8 %) | +0.9 % | within |
| E4c 3840 | npp | host | 14.71 (1.2 %) | 14.76 (3.4 %) | +0.3 % | within |

Median change over E4a batches 50 to 300, per version (host_enqueue_us): fkl_fused -3.8 %, fkl_fused_half_pixel -3.8 %, opencv_percrop +0.2 %, npp +0.0 %, fkl_fused_late -3.8 %, hand_batch -0.1 %, hand_padded +0.4 %, hand_roi +1.0 %, fkl_fused_after -3.8 %, cvgs_fused +0.4 %

### host_launch_us

| point | version | bound | old (spread) | new (spread) | change | |
|---|---|---|---|---|---|---|
| E4a 10 | fkl_fused | host | 2.066 (1.8 %) | 2.056 (2.5 %) | -0.5 % | within |
| E4a 10 | fkl_fused_half_pixel | host | 2.081 (3.5 %) | 2.044 (1.0 %) | -1.8 % | within |
| E4a 10 | opencv_percrop | host | 126.3 (0.0 %) | 125.9 (0.8 %) | -0.3 % | within |
| E4a 10 | npp | host | 129.8 (0.2 %) | 130 (1.3 %) | +0.1 % | within |
| E4a 10 | fkl_fused_late | host | 2.032 (2.7 %) | 2.013 (2.7 %) | -0.9 % | within |
| E4a 10 | hand_batch | host | 1.612 (3.9 %) | 1.628 (2.1 %) | +1.0 % | within |
| E4a 10 | hand_padded | host | 1.621 (1.1 %) | 1.62 (2.0 %) | -0.1 % | within |
| E4a 10 | hand_roi | host | 1.621 (4.1 %) | 1.643 (2.7 %) | +1.4 % | within |
| E4a 10 | fkl_fused_after | host | 2.057 (2.4 %) | 1.997 (2.4 %) | -2.9 % | exceeds |
| E4a 10 | cvgs_fused | host | 2.103 (2.3 %) | 2.099 (6.0 %) | -0.2 % | within |
| E4a 50 | fkl_fused | host | 3.442 (1.0 %) | 3.357 (4.6 %) | -2.5 % | within |
| E4a 50 | fkl_fused_half_pixel | host | 3.444 (3.5 %) | 3.317 (6.4 %) | -3.7 % | within |
| E4a 50 | opencv_percrop | host | 632.3 (0.8 %) | 633 (0.7 %) | +0.1 % | within |
| E4a 50 | npp | host | 638.2 (1.0 %) | 640.1 (1.6 %) | +0.3 % | within |
| E4a 50 | fkl_fused_late | host | 3.429 (3.2 %) | 3.334 (6.4 %) | -2.8 % | within |
| E4a 50 | hand_batch | host | 1.622 (0.8 %) | 1.682 (1.1 %) | +3.7 % | exceeds |
| E4a 50 | hand_padded | host | 1.74 (4.4 %) | 1.73 (1.6 %) | -0.6 % | within |
| E4a 50 | hand_roi | host | 1.734 (6.0 %) | 1.732 (1.8 %) | -0.1 % | within |
| E4a 50 | fkl_fused_after | host | 3.432 (1.3 %) | 3.422 (6.2 %) | -0.3 % | within |
| E4a 50 | cvgs_fused | host | 3.795 (4.1 %) | 3.808 (7.3 %) | +0.4 % | within |
| E4a 100 | fkl_fused | host | 5.138 (2.7 %) | 5.27 (3.4 %) | +2.6 % | within |
| E4a 100 | fkl_fused_half_pixel | host | 5.125 (6.6 %) | 5.318 (2.5 %) | +3.8 % | within |
| E4a 100 | opencv_percrop | host | 1,264 (1.8 %) | 1,265 (1.0 %) | +0.0 % | within |
| E4a 100 | npp | host | 1,275 (0.9 %) | 1,281 (1.6 %) | +0.5 % | within |
| E4a 100 | fkl_fused_late | host | 5.107 (2.0 %) | 5.241 (2.0 %) | +2.6 % | exceeds |
| E4a 100 | hand_batch | host | 1.626 (1.8 %) | 1.637 (1.0 %) | +0.6 % | within |
| E4a 100 | hand_padded | host | 1.917 (3.2 %) | 1.935 (1.3 %) | +1.0 % | within |
| E4a 100 | hand_roi | host | 1.956 (1.4 %) | 1.949 (2.4 %) | -0.4 % | within |
| E4a 100 | fkl_fused_after | host | 5.158 (1.7 %) | 5.28 (2.8 %) | +2.4 % | within |
| E4a 100 | cvgs_fused | host | 5.658 (5.3 %) | 5.689 (4.8 %) | +0.5 % | within |
| E4a 150 | fkl_fused | host | 7.119 (1.6 %) | 6.72 (1.3 %) | -5.6 % | exceeds FLAG |
| E4a 150 | fkl_fused_half_pixel | host | 7.209 (3.3 %) | 6.659 (1.2 %) | -7.6 % | exceeds FLAG |
| E4a 150 | opencv_percrop | host | 1,888 (0.8 %) | 1,897 (1.4 %) | +0.4 % | within |
| E4a 150 | npp | host | 1,913 (1.1 %) | 1,918 (1.2 %) | +0.3 % | within |
| E4a 150 | fkl_fused_late | host | 7.17 (0.7 %) | 6.616 (1.1 %) | -7.7 % | exceeds FLAG |
| E4a 150 | hand_batch | host | 1.671 (2.6 %) | 1.682 (1.8 %) | +0.6 % | within |
| E4a 150 | hand_padded | host | 2.165 (2.9 %) | 2.167 (1.1 %) | +0.1 % | within |
| E4a 150 | hand_roi | host | 2.123 (1.4 %) | 2.133 (3.3 %) | +0.5 % | within |
| E4a 150 | fkl_fused_after | host | 7.164 (0.8 %) | 6.671 (0.7 %) | -6.9 % | exceeds FLAG |
| E4a 150 | cvgs_fused | host | 7.967 (4.6 %) | 8.028 (6.0 %) | +0.8 % | within |
| E4a 200 | fkl_fused | host | 8.624 (1.3 %) | 8.258 (5.1 %) | -4.2 % | within |
| E4a 200 | fkl_fused_half_pixel | host | 8.631 (1.0 %) | 8.253 (2.0 %) | -4.4 % | exceeds |
| E4a 200 | opencv_percrop | host | 2,529 (1.5 %) | 2,525 (0.4 %) | -0.2 % | within |
| E4a 200 | npp | host | 2,563 (1.2 %) | 2,577 (2.4 %) | +0.5 % | within |
| E4a 200 | fkl_fused_late | host | 8.617 (1.2 %) | 8.163 (1.2 %) | -5.3 % | exceeds FLAG |
| E4a 200 | hand_batch | host | 1.668 (0.5 %) | 1.701 (2.0 %) | +2.0 % | within |
| E4a 200 | hand_padded | host | 2.335 (0.7 %) | 2.285 (1.8 %) | -2.1 % | exceeds |
| E4a 200 | hand_roi | host | 2.33 (1.2 %) | 2.333 (1.0 %) | +0.1 % | within |
| E4a 200 | fkl_fused_after | host | 8.618 (1.3 %) | 8.212 (0.5 %) | -4.7 % | exceeds |
| E4a 200 | cvgs_fused | host | 9.593 (6.8 %) | 9.614 (5.4 %) | +0.2 % | within |
| E4a 250 | fkl_fused | host | 10.47 (1.5 %) | 10.01 (3.2 %) | -4.4 % | exceeds |
| E4a 250 | fkl_fused_half_pixel | host | 10.42 (1.6 %) | 9.972 (2.5 %) | -4.3 % | exceeds |
| E4a 250 | opencv_percrop | host | 3,153 (0.6 %) | 3,165 (0.9 %) | +0.4 % | within |
| E4a 250 | npp | host | 3,206 (1.2 %) | 3,216 (2.0 %) | +0.3 % | within |
| E4a 250 | fkl_fused_late | host | 10.58 (2.5 %) | 10.01 (2.0 %) | -5.3 % | exceeds FLAG |
| E4a 250 | hand_batch | host | 1.708 (0.6 %) | 1.719 (1.2 %) | +0.6 % | within |
| E4a 250 | hand_padded | host | 2.53 (10.0 %) | 2.501 (5.1 %) | -1.1 % | within |
| E4a 250 | hand_roi | host | 2.565 (0.6 %) | 2.591 (2.2 %) | +1.0 % | within |
| E4a 250 | fkl_fused_after | host | 10.45 (1.4 %) | 10.02 (2.7 %) | -4.1 % | exceeds |
| E4a 250 | cvgs_fused | host | 12.26 (6.4 %) | 11.65 (6.7 %) | -5.0 % | within |
| E4a 300 | fkl_fused | host | 12.38 (1.2 %) | 11.71 (0.7 %) | -5.4 % | exceeds FLAG |
| E4a 300 | fkl_fused_half_pixel | host | 12.34 (0.8 %) | 11.7 (0.1 %) | -5.1 % | exceeds FLAG |
| E4a 300 | opencv_percrop | host | 3,783 (0.2 %) | 3,780 (0.3 %) | -0.1 % | within |
| E4a 300 | npp | host | 3,847 (0.9 %) | 3,850 (0.9 %) | +0.1 % | within |
| E4a 300 | fkl_fused_late | host | 12.33 (1.8 %) | 11.69 (0.6 %) | -5.2 % | exceeds FLAG |
| E4a 300 | hand_batch | host | 1.761 (2.1 %) | 1.771 (0.1 %) | +0.6 % | within |
| E4a 300 | hand_padded | host | 2.689 (4.0 %) | 2.652 (2.0 %) | -1.4 % | within |
| E4a 300 | hand_roi | host | 2.734 (3.0 %) | 2.766 (0.7 %) | +1.2 % | within |
| E4a 300 | fkl_fused_after | host | 12.4 (1.3 %) | 11.75 (0.4 %) | -5.2 % | exceeds FLAG |
| E4a 300 | cvgs_fused | host | 13.64 (6.2 %) | 14.54 (7.2 %) | +6.7 % | within FLAG |
| E4c 720 | fkl_fused | host | 1.626 (2.9 %) | 1.603 (4.6 %) | -1.5 % | within |
| E4c 720 | fkl_fused_half_pixel | host | 1.606 (1.9 %) | 1.623 (2.2 %) | +1.0 % | within |
| E4c 720 | opencv_percrop | host | 12.42 (1.4 %) | 12.45 (0.5 %) | +0.2 % | within |
| E4c 720 | npp | host | 14.35 (1.7 %) | 14.31 (2.2 %) | -0.3 % | within |
| E4c 1920 | fkl_fused | host | 1.619 (1.7 %) | 1.598 (0.8 %) | -1.3 % | within |
| E4c 1920 | fkl_fused_half_pixel | host | 1.624 (5.0 %) | 1.593 (1.4 %) | -1.9 % | within |
| E4c 1920 | opencv_percrop | host | 12.26 (0.4 %) | 12.25 (1.8 %) | -0.1 % | within |
| E4c 1920 | npp | host | 14.27 (1.1 %) | 14.21 (1.9 %) | -0.4 % | within |
| E4c 3840 | fkl_fused | host | 1.583 (1.9 %) | 1.584 (4.4 %) | +0.1 % | within |
| E4c 3840 | fkl_fused_half_pixel | host | 1.633 (1.1 %) | 1.613 (3.2 %) | -1.2 % | within |
| E4c 3840 | opencv_percrop | host | 12.25 (0.8 %) | 12.25 (1.2 %) | +0.1 % | within |
| E4c 3840 | npp | host | 14.26 (0.5 %) | 14.18 (1.8 %) | -0.5 % | within |

Median change over E4a batches 50 to 300, per version (host_launch_us): fkl_fused -4.4 %, fkl_fused_half_pixel -4.3 %, opencv_percrop +0.2 %, npp +0.2 %, fkl_fused_late -4.7 %, hand_batch +0.3 %, hand_padded +0.3 %, hand_roi +0.4 %, fkl_fused_after -4.3 %, cvgs_fused +0.6 %

Flagged over all 30 batches and 3 frames (|change| above the flag and above both spreads): 63

- gpu_mean_ms E4a 70 fkl_fused_half_pixel (GPU): +3.0 % (old 0.009159, spread 0.3 %; new 0.009436, spread 0.8 %)
- gpu_mean_ms E4a 110 fkl_fused (GPU): -3.3 % (old 0.01254, spread 0.5 %; new 0.01213, spread 0.9 %)
- gpu_mean_ms E4a 110 fkl_fused_half_pixel (GPU): -3.1 % (old 0.01258, spread 0.5 %; new 0.01219, spread 1.2 %)
- gpu_mean_ms E4a 110 fkl_fused_after (GPU): -3.0 % (old 0.01251, spread 0.7 %; new 0.01213, spread 0.8 %)
- gpu_mean_ms E4a 150 fkl_fused_half_pixel (GPU): -3.2 % (old 0.01579, spread 0.8 %; new 0.01528, spread 0.6 %)
- gpu_mean_ms E4a 150 fkl_fused_after (GPU): -3.0 % (old 0.01565, spread 0.5 %; new 0.01518, spread 1.2 %)
- host_enqueue_us E4a 70 fkl_fused_half_pixel (host): +6.4 % (old 4.101, spread 1.9 %; new 4.362, spread 1.3 %)
- host_enqueue_us E4a 110 fkl_fused (host): -6.1 % (old 5.651, spread 0.8 %; new 5.308, spread 1.4 %)
- host_enqueue_us E4a 110 fkl_fused_late (host): -5.8 % (old 5.606, spread 0.2 %; new 5.283, spread 0.5 %)
- host_enqueue_us E4a 130 fkl_fused (host): -7.1 % (old 6.525, spread 1.3 %; new 6.061, spread 1.4 %)
- host_enqueue_us E4a 130 fkl_fused_half_pixel (host): -6.9 % (old 6.521, spread 1.1 %; new 6.072, spread 2.2 %)
- host_enqueue_us E4a 130 fkl_fused_late (host): -6.9 % (old 6.507, spread 1.1 %; new 6.06, spread 0.8 %)
- host_enqueue_us E4a 130 fkl_fused_after (host): -6.7 % (old 6.474, spread 1.9 %; new 6.041, spread 3.7 %)
- host_enqueue_us E4a 150 fkl_fused (host): -6.3 % (old 7.137, spread 1.2 %; new 6.69, spread 2.5 %)
- host_enqueue_us E4a 150 fkl_fused_half_pixel (host): -6.4 % (old 7.126, spread 1.4 %; new 6.667, spread 1.0 %)
- host_enqueue_us E4a 150 fkl_fused_late (host): -6.3 % (old 7.127, spread 0.3 %; new 6.678, spread 1.5 %)
- host_enqueue_us E4a 150 fkl_fused_after (host): -5.9 % (old 7.124, spread 0.3 %; new 6.706, spread 1.6 %)
- host_enqueue_us E4a 170 fkl_fused (host): -5.8 % (old 7.895, spread 2.1 %; new 7.44, spread 1.9 %)
- host_enqueue_us E4a 170 fkl_fused_half_pixel (host): -6.2 % (old 7.878, spread 1.7 %; new 7.388, spread 2.4 %)
- host_enqueue_us E4a 170 fkl_fused_late (host): -6.2 % (old 7.905, spread 2.1 %; new 7.417, spread 2.0 %)
- host_enqueue_us E4a 170 fkl_fused_after (host): -6.3 % (old 7.899, spread 1.2 %; new 7.4, spread 1.2 %)
- host_enqueue_us E4a 220 fkl_fused_half_pixel (host): -5.2 % (old 9.421, spread 1.6 %; new 8.928, spread 1.2 %)
- host_enqueue_us E4a 220 fkl_fused_late (host): -5.6 % (old 9.427, spread 1.6 %; new 8.897, spread 1.1 %)
- host_enqueue_us E4a 230 fkl_fused_half_pixel (host): -5.7 % (old 9.776, spread 1.3 %; new 9.219, spread 3.1 %)
- host_enqueue_us E4a 230 fkl_fused_late (host): -5.3 % (old 9.775, spread 0.2 %; new 9.255, spread 2.0 %)
- host_enqueue_us E4a 240 fkl_fused (host): -5.0 % (old 10.04, spread 3.3 %; new 9.535, spread 1.4 %)
- host_enqueue_us E4a 240 fkl_fused_half_pixel (host): -5.1 % (old 10.06, spread 2.5 %; new 9.554, spread 1.4 %)
- host_enqueue_us E4a 240 fkl_fused_late (host): -5.1 % (old 10.08, spread 2.9 %; new 9.568, spread 1.0 %)
- host_enqueue_us E4a 300 fkl_fused (host): -5.2 % (old 12.31, spread 1.1 %; new 11.67, spread 1.8 %)
- host_enqueue_us E4a 300 fkl_fused_after (host): -5.4 % (old 12.37, spread 1.2 %; new 11.71, spread 1.3 %)
- host_enqueue_us E4a 300 cvgs_fused (host): +6.7 % (old 13.64, spread 5.5 %; new 14.55, spread 6.5 %)
- host_launch_us E4a 60 fkl_fused (host): -5.4 % (old 3.862, spread 4.0 %; new 3.652, spread 4.8 %)
- host_launch_us E4a 110 fkl_fused (host): -5.9 % (old 5.63, spread 2.4 %; new 5.3, spread 0.6 %)
- host_launch_us E4a 110 fkl_fused_half_pixel (host): -5.9 % (old 5.633, spread 1.3 %; new 5.302, spread 0.9 %)
- host_launch_us E4a 110 fkl_fused_late (host): -6.0 % (old 5.646, spread 1.4 %; new 5.306, spread 2.3 %)
- host_launch_us E4a 130 fkl_fused (host): -7.7 % (old 6.55, spread 0.7 %; new 6.048, spread 1.6 %)
- host_launch_us E4a 130 fkl_fused_half_pixel (host): -7.0 % (old 6.501, spread 1.0 %; new 6.044, spread 1.7 %)
- host_launch_us E4a 130 fkl_fused_late (host): -7.3 % (old 6.512, spread 0.1 %; new 6.036, spread 1.9 %)
- host_launch_us E4a 130 fkl_fused_after (host): -7.5 % (old 6.53, spread 0.9 %; new 6.044, spread 0.5 %)
- host_launch_us E4a 140 fkl_fused (host): -5.4 % (old 6.652, spread 1.1 %; new 6.293, spread 1.4 %)
- host_launch_us E4a 150 fkl_fused (host): -5.6 % (old 7.119, spread 1.6 %; new 6.72, spread 1.3 %)
- host_launch_us E4a 150 fkl_fused_half_pixel (host): -7.6 % (old 7.209, spread 3.3 %; new 6.659, spread 1.2 %)
- host_launch_us E4a 150 fkl_fused_late (host): -7.7 % (old 7.17, spread 0.7 %; new 6.616, spread 1.1 %)
- host_launch_us E4a 150 fkl_fused_after (host): -6.9 % (old 7.164, spread 0.8 %; new 6.671, spread 0.7 %)
- host_launch_us E4a 170 fkl_fused (host): -6.5 % (old 7.877, spread 1.5 %; new 7.364, spread 1.5 %)
- host_launch_us E4a 170 fkl_fused_late (host): -6.8 % (old 7.926, spread 1.1 %; new 7.383, spread 2.3 %)
- host_launch_us E4a 170 fkl_fused_after (host): -7.4 % (old 7.96, spread 1.4 %; new 7.37, spread 2.0 %)
- host_launch_us E4a 190 fkl_fused_after (host): -5.2 % (old 8.322, spread 1.3 %; new 7.893, spread 0.9 %)
- host_launch_us E4a 200 fkl_fused_late (host): -5.3 % (old 8.617, spread 1.2 %; new 8.163, spread 1.2 %)
- host_launch_us E4a 220 fkl_fused (host): -5.3 % (old 9.412, spread 1.4 %; new 8.916, spread 1.5 %)
- host_launch_us E4a 220 fkl_fused_half_pixel (host): -6.1 % (old 9.461, spread 1.6 %; new 8.885, spread 0.3 %)
- host_launch_us E4a 220 fkl_fused_late (host): -5.3 % (old 9.426, spread 1.4 %; new 8.93, spread 0.7 %)
- host_launch_us E4a 230 fkl_fused (host): -5.5 % (old 9.785, spread 3.3 %; new 9.244, spread 2.0 %)
- host_launch_us E4a 230 fkl_fused_after (host): -5.5 % (old 9.778, spread 1.5 %; new 9.244, spread 0.9 %)
- host_launch_us E4a 240 fkl_fused (host): -6.2 % (old 10.15, spread 3.2 %; new 9.522, spread 1.0 %)
- host_launch_us E4a 240 fkl_fused_late (host): -6.0 % (old 10.14, spread 2.5 %; new 9.529, spread 1.9 %)
- host_launch_us E4a 240 fkl_fused_after (host): -6.9 % (old 10.27, spread 2.8 %; new 9.556, spread 2.0 %)
- host_launch_us E4a 250 fkl_fused_late (host): -5.3 % (old 10.58, spread 2.5 %; new 10.01, spread 2.0 %)
- host_launch_us E4a 280 fkl_fused (host): -5.3 % (old 11.56, spread 1.9 %; new 10.95, spread 0.9 %)
- host_launch_us E4a 300 fkl_fused (host): -5.4 % (old 12.38, spread 1.2 %; new 11.71, spread 0.7 %)
- host_launch_us E4a 300 fkl_fused_half_pixel (host): -5.1 % (old 12.34, spread 0.8 %; new 11.7, spread 0.1 %)
- host_launch_us E4a 300 fkl_fused_late (host): -5.2 % (old 12.33, spread 1.8 %; new 11.69, spread 0.6 %)
- host_launch_us E4a 300 fkl_fused_after (host): -5.2 % (old 12.4, spread 1.3 %; new 11.75, spread 0.4 %)

## Agreement

- ok_own (against the scalar reference of the version's own convention, 1e-4): 1386 of 1386 rows pass
- pairs (the outputs of the checked calls; 30 batches x 3 runs per pair):
  - cvgs_fused vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - cvgs_fused vs opencv_percrop: bitwise equal in 0 of 90, largest |difference| 7.63e-06, within 1e-4 in 90 of 90
  - cvgs_fused_once vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - cvgs_fused_once vs opencv_percrop: bitwise equal in 0 of 90, largest |difference| 7.63e-06, within 1e-4 in 90 of 90
  - fkl_fused_after vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - fkl_fused_half_pixel_once vs fkl_fused_half_pixel: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - fkl_fused_late vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - fkl_fused_once vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - fkl_fused_once_after vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - hand_batch vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - hand_padded vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - hand_roi vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - hand_roi_once vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90

## The runs and the SM clock

Sampler: 1526 samples of nvidia-smi every 100 ms (gpu_clocks.csv).

| run | start | end | SM clock median MHz | min MHz | samples | samples below 2,500 MHz |
|---|---|---|---|---|---|---|
| warmup | 14:18:43 | 14:18:47 | 2,700 | 210 | 35 | 9 |
| quiet-run1 | 14:18:47 | 14:19:36 | 2,685 | 2,685 | 490 | 0 |
| quiet-run2 | 14:19:36 | 14:20:26 | 2,685 | 2,685 | 499 | 0 |
| quiet-run3 | 14:20:26 | 14:21:16 | 2,685 | 2,685 | 500 | 0 |

