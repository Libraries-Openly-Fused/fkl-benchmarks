# E4 window e4-hand-cvgs, RTX 4090: the hand-written kernel of E1c and cvGPUSpeedup (report)

Written by `e4_hand_cvgs_report.py` from the result folders only, 2026-11-08. Times: gpu_mean_ms, CUDA events around one call of the whole implementation (E4's protocol), 100 calls after one warm-up; per point the median of the three runs. Ratios: the median of the three per-run ratios, with the smallest and the largest run.

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

## The new series (E4a)

### hand_batch / FKL (the series of the figure: the hand-written kernel's time over FKL's)

FKL time `adjacent` = the mean of fkl_fused_late (timed just before hand_batch) and fkl_fused_after (timed just after hand_padded).

| batch | hand_batch ms | fkl_fused ms | adjacent FKL ms | hand / fkl_fused | hand / adjacent (min - max) | hand / late | hand / after |
|---|---|---|---|---|---|---|---|
| 10 | 0.00344 | 0.003797 | 0.003731 | 0.904 | 0.922 (0.908 - 0.932) | 0.924 | 0.920 |
| 20 | 0.004067 | 0.004877 | 0.004922 | 0.834 | 0.824 (0.819 - 0.831) | 0.817 | 0.831 |
| 30 | 0.004492 | 0.005668 | 0.005688 | 0.793 | 0.791 (0.782 - 0.799) | 0.792 | 0.789 |
| 40 | 0.004947 | 0.006787 | 0.00677 | 0.729 | 0.730 (0.729 - 0.735) | 0.732 | 0.730 |
| 50 | 0.00555 | 0.00758 | 0.007556 | 0.734 | 0.735 (0.728 - 0.737) | 0.732 | 0.738 |
| 60 | 0.005908 | 0.00838 | 0.008389 | 0.710 | 0.704 (0.694 - 0.752) | 0.705 | 0.703 |
| 70 | 0.006346 | 0.009231 | 0.009152 | 0.689 | 0.695 (0.686 - 0.695) | 0.698 | 0.692 |
| 80 | 0.006863 | 0.01002 | 0.01001 | 0.688 | 0.688 (0.685 - 0.690) | 0.687 | 0.687 |
| 90 | 0.007365 | 0.01098 | 0.01105 | 0.669 | 0.666 (0.664 - 0.681) | 0.665 | 0.667 |
| 100 | 0.007771 | 0.0116 | 0.01156 | 0.671 | 0.673 (0.671 - 0.674) | 0.673 | 0.673 |
| 110 | 0.008313 | 0.0126 | 0.01247 | 0.660 | 0.666 (0.659 - 0.668) | 0.666 | 0.666 |
| 120 | 0.008794 | 0.01331 | 0.01322 | 0.661 | 0.667 (0.665 - 0.667) | 0.665 | 0.665 |
| 130 | 0.0091 | 0.01419 | 0.01406 | 0.641 | 0.647 (0.644 - 0.649) | 0.648 | 0.646 |
| 140 | 0.009573 | 0.01482 | 0.01472 | 0.646 | 0.651 (0.649 - 0.652) | 0.652 | 0.649 |
| 150 | 0.00999 | 0.0157 | 0.01565 | 0.635 | 0.638 (0.634 - 0.643) | 0.637 | 0.640 |
| 200 | 0.01219 | 0.01951 | 0.01942 | 0.627 | 0.628 (0.616 - 0.632) | 0.628 | 0.627 |
| 250 | 0.01461 | 0.02357 | 0.02361 | 0.622 | 0.619 (0.599 - 0.621) | 0.617 | 0.623 |
| 300 | 0.01657 | 0.02776 | 0.02757 | 0.597 | 0.600 (0.599 - 0.602) | 0.601 | 0.601 |

### hand_padded (the larger launch-parameter block) and the position of FKL

| batch | hand_padded ms | hand_padded / adjacent FKL | hand_padded / hand_batch | fkl_fused_late / fkl_fused | fkl_fused_after / fkl_fused |
|---|---|---|---|---|---|
| 10 | 0.00361 | 0.967 | 1.049 | 0.982 (0.965 - 0.999) | 0.977 (0.969 - 0.991) |
| 20 | 0.004095 | 0.832 | 1.007 | 1.014 (0.994 - 1.020) | 1.003 (0.987 - 1.004) |
| 30 | 0.004682 | 0.823 | 1.041 | 1.001 (0.992 - 1.005) | 0.996 (0.992 - 1.006) |
| 40 | 0.00518 | 0.765 | 1.042 | 0.995 (0.993 - 0.998) | 1.002 (0.997 - 1.005) |
| 50 | 0.005644 | 0.747 | 1.017 | 1.000 (0.995 - 1.007) | 0.994 (0.987 - 1.010) |
| 60 | 0.006133 | 0.728 | 1.038 | 1.006 (1.001 - 1.006) | 0.997 (0.996 - 1.010) |
| 70 | 0.006603 | 0.720 | 1.040 | 0.992 (0.987 - 0.998) | 0.997 (0.995 - 0.999) |
| 80 | 0.0071 | 0.709 | 1.035 | 0.999 (0.995 - 1.001) | 0.999 (0.998 - 1.006) |
| 90 | 0.007657 | 0.692 | 1.041 | 1.007 (0.987 - 1.009) | 1.004 (0.995 - 1.006) |
| 100 | 0.008135 | 0.704 | 1.046 | 0.995 (0.993 - 0.999) | 0.996 (0.996 - 1.001) |
| 110 | 0.008799 | 0.707 | 1.062 | 0.990 (0.974 - 0.998) | 0.996 (0.980 - 1.013) |
| 120 | 0.009223 | 0.696 | 1.046 | 0.997 (0.993 - 0.999) | 0.993 (0.974 - 1.001) |
| 130 | 0.009712 | 0.688 | 1.065 | 0.990 (0.977 - 0.996) | 0.992 (0.976 - 1.003) |
| 140 | 0.01013 | 0.688 | 1.059 | 0.990 (0.990 - 0.992) | 0.994 (0.990 - 0.995) |
| 150 | 0.01069 | 0.682 | 1.071 | 0.996 (0.994 - 1.002) | 0.992 (0.989 - 1.000) |
| 200 | 0.01315 | 0.676 | 1.082 | 0.999 (0.992 - 1.001) | 0.997 (0.990 - 1.002) |
| 250 | 0.01587 | 0.672 | 1.086 | 1.012 (0.996 - 1.078) | 0.999 (0.998 - 1.000) |
| 300 | 0.01813 | 0.658 | 1.097 | 0.997 (0.989 - 0.998) | 0.997 (0.992 - 1.001) |

### cvGPUSpeedup against OpenCV-CUDA per crop

| batch | opencv_percrop ms | cvgs_fused ms | speedup opencv_percrop / cvgs_fused (min - max) | cvgs_fused / fkl_fused |
|---|---|---|---|---|
| 10 | 0.1299 | 0.003942 | 33.0 (30.9 - 33.2) | 1.049 |
| 20 | 0.2549 | 0.005161 | 48.7 (47.4 - 50.7) | 1.058 |
| 30 | 0.3826 | 0.005764 | 66.1 (66.1 - 66.5) | 1.016 |
| 40 | 0.5105 | 0.007189 | 71.0 (70.4 - 71.2) | 1.055 |
| 50 | 0.6364 | 0.007898 | 80.9 (79.4 - 83.3) | 1.042 |
| 60 | 0.7594 | 0.008839 | 85.8 (85.7 - 86.2) | 1.055 |
| 70 | 0.8837 | 0.009406 | 94.1 (93.3 - 95.2) | 1.024 |
| 80 | 1.012 | 0.01047 | 97.0 (94.3 - 97.2) | 1.050 |
| 90 | 1.141 | 0.01135 | 100.5 (99.0 - 102.6) | 1.032 |
| 100 | 1.272 | 0.01227 | 103.7 (94.1 - 104.0) | 1.059 |
| 110 | 1.396 | 0.01286 | 107.2 (105.5 - 108.6) | 1.030 |
| 120 | 1.525 | 0.01407 | 108.4 (106.4 - 108.6) | 1.055 |
| 130 | 1.645 | 0.0146 | 112.6 (111.8 - 116.4) | 1.029 |
| 140 | 1.775 | 0.01584 | 112.0 (110.5 - 113.0) | 1.069 |
| 150 | 1.902 | 0.01622 | 115.9 (105.4 - 118.2) | 1.036 |
| 200 | 2.535 | 0.02088 | 121.6 (119.7 - 121.8) | 1.073 |
| 250 | 3.156 | 0.02466 | 128.0 (126.7 - 129.3) | 1.046 |
| 300 | 3.808 | 0.02969 | 128.2 (126.9 - 128.8) | 1.069 |

### Host time of one call (host_enqueue_us: the call returns, the stream idle before it; median of the three runs, us)

| batch | fkl_fused | fkl_fused_late | fkl_fused_after | hand_batch | hand_padded | cvgs_fused | opencv_percrop |
|---|---|---|---|---|---|---|---|
| 10 | 2.083 | 2.037 | 2.022 | 1.611 | 1.638 | 2.193 | 126.5 |
| 20 | 2.414 | 2.453 | 2.414 | 1.598 | 1.637 | 2.54 | 255.4 |
| 30 | 2.757 | 2.777 | 2.806 | 1.607 | 1.682 | 2.853 | 382.1 |
| 40 | 3.072 | 3.106 | 3.076 | 1.632 | 1.675 | 3.376 | 510.7 |
| 50 | 3.516 | 3.454 | 3.463 | 1.655 | 1.729 | 3.67 | 638.8 |
| 60 | 3.74 | 3.753 | 3.727 | 1.702 | 1.792 | 4.212 | 767.8 |
| 70 | 4.131 | 4.139 | 4.125 | 1.631 | 1.8 | 4.471 | 891.1 |
| 80 | 4.481 | 4.442 | 4.461 | 1.626 | 1.842 | 5.039 | 1,016 |
| 90 | 4.917 | 4.981 | 4.953 | 1.655 | 1.876 | 5.173 | 1,146 |
| 100 | 5.2 | 5.15 | 5.162 | 1.648 | 1.913 | 5.895 | 1,270 |
| 110 | 5.618 | 5.657 | 5.63 | 1.68 | 1.991 | 6.092 | 1,399 |
| 120 | 5.947 | 5.945 | 5.848 | 1.688 | 2.024 | 6.748 | 1,524 |
| 130 | 6.495 | 6.434 | 6.502 | 1.672 | 2.067 | 6.795 | 1,661 |
| 140 | 6.58 | 6.58 | 6.571 | 1.674 | 2.087 | 7.579 | 1,780 |
| 150 | 7.15 | 7.153 | 7.09 | 1.676 | 2.139 | 7.604 | 1,905 |
| 200 | 8.741 | 8.68 | 8.762 | 1.685 | 2.357 | 10.18 | 2,534 |
| 250 | 10.44 | 10.3 | 10.37 | 1.727 | 2.547 | 11.65 | 3,189 |
| 300 | 12.33 | 12.37 | 12.32 | 1.781 | 2.694 | 14.37 | 3,815 |

## The existing series against window B (standing form)

old = window B's cleaned build (main 836a98e, 2026-11-07), new = this window (main 2dac1c8; the same kernels, SASS identical); median of three runs, spread = (largest - smallest) / median. In this program every batch now ends with the five new versions, so the next batch's fkl_fused follows cvgs_fused (in window B it followed npp), and E4c's first frame follows batch 300's cvgs_fused.

### gpu_mean_ms

| point | version | bound | old (spread) | new (spread) | change | |
|---|---|---|---|---|---|---|
| E4a 10 | fkl_fused | GPU | 0.003783 (3.1 %) | 0.003797 (2.7 %) | +0.4 % | within |
| E4a 10 | fkl_fused_half_pixel | GPU | 0.003791 (0.6 %) | 0.003838 (9.1 %) | +1.2 % | within |
| E4a 10 | opencv_percrop | host | 0.1297 (0.7 %) | 0.1299 (1.7 %) | +0.2 % | within |
| E4a 10 | npp | host | 0.1319 (1.5 %) | 0.1337 (1.8 %) | +1.4 % | within |
| E4a 50 | fkl_fused | GPU | 0.007524 (1.1 %) | 0.00758 (1.0 %) | +0.7 % | within |
| E4a 50 | fkl_fused_half_pixel | GPU | 0.00752 (0.9 %) | 0.007625 (0.8 %) | +1.4 % | exceeds |
| E4a 50 | opencv_percrop | host | 0.636 (1.2 %) | 0.6364 (4.3 %) | +0.1 % | within |
| E4a 50 | npp | host | 0.653 (1.4 %) | 0.6517 (0.5 %) | -0.2 % | within |
| E4a 100 | fkl_fused | GPU | 0.0114 (0.2 %) | 0.0116 (0.4 %) | +1.8 % | exceeds |
| E4a 100 | fkl_fused_half_pixel | GPU | 0.01145 (0.3 %) | 0.01164 (0.6 %) | +1.6 % | exceeds |
| E4a 100 | opencv_percrop | host | 1.273 (1.3 %) | 1.272 (1.8 %) | -0.0 % | within |
| E4a 100 | npp | host | 1.299 (1.1 %) | 1.295 (1.1 %) | -0.3 % | within |
| E4a 150 | fkl_fused | GPU | 0.01547 (0.8 %) | 0.0157 (0.5 %) | +1.5 % | exceeds |
| E4a 150 | fkl_fused_half_pixel | GPU | 0.01555 (1.3 %) | 0.01581 (1.3 %) | +1.7 % | exceeds |
| E4a 150 | opencv_percrop | host | 1.896 (1.3 %) | 1.902 (1.7 %) | +0.3 % | within |
| E4a 150 | npp | host | 1.949 (1.8 %) | 1.934 (1.3 %) | -0.8 % | within |
| E4a 200 | fkl_fused | GPU | 0.0192 (0.4 %) | 0.01951 (1.3 %) | +1.6 % | exceeds |
| E4a 200 | fkl_fused_half_pixel | GPU | 0.01921 (0.7 %) | 0.01943 (1.8 %) | +1.1 % | within |
| E4a 200 | opencv_percrop | host | 2.527 (1.4 %) | 2.535 (1.2 %) | +0.3 % | within |
| E4a 200 | npp | host | 2.59 (1.3 %) | 2.582 (1.3 %) | -0.3 % | within |
| E4a 250 | fkl_fused | GPU | 0.02317 (0.4 %) | 0.02357 (1.5 %) | +1.7 % | exceeds |
| E4a 250 | fkl_fused_half_pixel | GPU | 0.02319 (0.8 %) | 0.0236 (2.1 %) | +1.8 % | within |
| E4a 250 | opencv_percrop | host | 3.186 (1.3 %) | 3.156 (1.2 %) | -1.0 % | within |
| E4a 250 | npp | host | 3.233 (2.0 %) | 3.228 (0.5 %) | -0.1 % | within |
| E4a 300 | fkl_fused | GPU | 0.02754 (0.3 %) | 0.02776 (0.7 %) | +0.8 % | exceeds |
| E4a 300 | fkl_fused_half_pixel | GPU | 0.02756 (1.6 %) | 0.02756 (0.5 %) | -0.0 % | within |
| E4a 300 | opencv_percrop | host | 3.814 (1.2 %) | 3.808 (1.0 %) | -0.2 % | within |
| E4a 300 | npp | host | 3.884 (2.2 %) | 3.88 (1.0 %) | -0.1 % | within |
| E4c 720 | fkl_fused | GPU | 0.005248 (1.9 %) | 0.005174 (1.0 %) | -1.4 % | within |
| E4c 720 | fkl_fused_half_pixel | GPU | 0.005243 (1.4 %) | 0.005193 (0.1 %) | -1.0 % | within |
| E4c 720 | opencv_percrop | GPU | 0.02775 (1.1 %) | 0.02762 (0.2 %) | -0.5 % | within |
| E4c 720 | npp | GPU | 0.0362 (1.3 %) | 0.03612 (0.2 %) | -0.2 % | within |
| E4c 1920 | fkl_fused | GPU | 0.01551 (0.3 %) | 0.0155 (0.3 %) | -0.1 % | within |
| E4c 1920 | fkl_fused_half_pixel | GPU | 0.01564 (0.8 %) | 0.01562 (0.2 %) | -0.1 % | within |
| E4c 1920 | opencv_percrop | GPU | 0.1744 (0.2 %) | 0.1745 (0.1 %) | +0.1 % | within |
| E4c 1920 | npp | GPU | 0.1556 (1.4 %) | 0.155 (0.5 %) | -0.4 % | within |
| E4c 3840 | fkl_fused | GPU | 0.1579 (0.1 %) | 0.1577 (0.2 %) | -0.1 % | within |
| E4c 3840 | fkl_fused_half_pixel | GPU | 0.1577 (0.2 %) | 0.1579 (0.1 %) | +0.1 % | within |
| E4c 3840 | opencv_percrop | GPU | 1.605 (0.1 %) | 1.606 (0.0 %) | +0.0 % | within |
| E4c 3840 | npp | GPU | 1.849 (0.0 %) | 1.847 (0.1 %) | -0.1 % | exceeds |

### host_enqueue_us

| point | version | bound | old (spread) | new (spread) | change | |
|---|---|---|---|---|---|---|
| E4a 10 | fkl_fused | host | 2.04 (2.3 %) | 2.083 (8.3 %) | +2.1 % | within |
| E4a 10 | fkl_fused_half_pixel | host | 2.022 (3.9 %) | 2.099 (3.0 %) | +3.8 % | within |
| E4a 10 | opencv_percrop | host | 126.7 (2.4 %) | 126.5 (1.0 %) | -0.2 % | within |
| E4a 10 | npp | host | 130.9 (1.2 %) | 129.8 (1.8 %) | -0.8 % | within |
| E4a 50 | fkl_fused | host | 3.345 (2.1 %) | 3.516 (1.7 %) | +5.1 % | exceeds FLAG |
| E4a 50 | fkl_fused_half_pixel | host | 3.317 (1.8 %) | 3.52 (3.0 %) | +6.1 % | exceeds FLAG |
| E4a 50 | opencv_percrop | host | 638.1 (1.1 %) | 638.8 (2.8 %) | +0.1 % | within |
| E4a 50 | npp | host | 642.6 (1.9 %) | 644.1 (1.3 %) | +0.2 % | within |
| E4a 100 | fkl_fused | host | 4.987 (0.2 %) | 5.2 (1.3 %) | +4.3 % | exceeds |
| E4a 100 | fkl_fused_half_pixel | host | 4.954 (1.1 %) | 5.214 (2.3 %) | +5.2 % | exceeds FLAG |
| E4a 100 | opencv_percrop | host | 1,271 (1.6 %) | 1,270 (1.5 %) | -0.1 % | within |
| E4a 100 | npp | host | 1,290 (1.8 %) | 1,285 (1.4 %) | -0.4 % | within |
| E4a 150 | fkl_fused | host | 6.922 (0.9 %) | 7.15 (0.7 %) | +3.3 % | exceeds |
| E4a 150 | fkl_fused_half_pixel | host | 6.878 (0.6 %) | 7.152 (2.6 %) | +4.0 % | exceeds |
| E4a 150 | opencv_percrop | host | 1,900 (0.7 %) | 1,905 (1.4 %) | +0.3 % | within |
| E4a 150 | npp | host | 1,929 (1.8 %) | 1,923 (1.2 %) | -0.3 % | within |
| E4a 200 | fkl_fused | host | 8.269 (1.4 %) | 8.741 (2.2 %) | +5.7 % | exceeds FLAG |
| E4a 200 | fkl_fused_half_pixel | host | 8.207 (0.3 %) | 8.708 (2.0 %) | +6.1 % | exceeds FLAG |
| E4a 200 | opencv_percrop | host | 2,541 (1.0 %) | 2,534 (0.6 %) | -0.3 % | within |
| E4a 200 | npp | host | 2,571 (1.7 %) | 2,578 (0.7 %) | +0.2 % | within |
| E4a 250 | fkl_fused | host | 9.928 (1.5 %) | 10.44 (2.9 %) | +5.2 % | exceeds FLAG |
| E4a 250 | fkl_fused_half_pixel | host | 9.946 (1.0 %) | 10.37 (3.5 %) | +4.3 % | exceeds |
| E4a 250 | opencv_percrop | host | 3,171 (0.7 %) | 3,189 (1.1 %) | +0.6 % | within |
| E4a 250 | npp | host | 3,213 (2.0 %) | 3,214 (1.3 %) | +0.0 % | within |
| E4a 300 | fkl_fused | host | 12.2 (1.1 %) | 12.33 (0.3 %) | +1.1 % | exceeds |
| E4a 300 | fkl_fused_half_pixel | host | 12.24 (1.4 %) | 12.32 (1.7 %) | +0.6 % | within |
| E4a 300 | opencv_percrop | host | 3,811 (1.2 %) | 3,815 (1.1 %) | +0.1 % | within |
| E4a 300 | npp | host | 3,874 (1.6 %) | 3,849 (1.0 %) | -0.7 % | within |
| E4c 720 | fkl_fused | host | 1.6 (1.7 %) | 1.61 (0.9 %) | +0.6 % | within |
| E4c 720 | fkl_fused_half_pixel | host | 1.616 (0.9 %) | 1.609 (2.2 %) | -0.4 % | within |
| E4c 720 | opencv_percrop | host | 12.83 (4.6 %) | 12.52 (2.7 %) | -2.4 % | within |
| E4c 720 | npp | host | 14.45 (2.2 %) | 14.44 (0.3 %) | -0.1 % | within |
| E4c 1920 | fkl_fused | host | 1.628 (3.4 %) | 1.608 (1.5 %) | -1.2 % | within |
| E4c 1920 | fkl_fused_half_pixel | host | 1.617 (3.2 %) | 1.6 (6.1 %) | -1.1 % | within |
| E4c 1920 | opencv_percrop | host | 12.72 (1.8 %) | 12.75 (3.0 %) | +0.2 % | within |
| E4c 1920 | npp | host | 14.44 (2.7 %) | 14.44 (1.4 %) | -0.1 % | within |
| E4c 3840 | fkl_fused | host | 1.649 (3.8 %) | 1.635 (4.0 %) | -0.9 % | within |
| E4c 3840 | fkl_fused_half_pixel | host | 1.681 (4.3 %) | 1.696 (4.3 %) | +0.9 % | within |
| E4c 3840 | opencv_percrop | host | 13.02 (1.2 %) | 13.03 (2.4 %) | +0.0 % | within |
| E4c 3840 | npp | host | 14.85 (2.1 %) | 14.76 (1.6 %) | -0.6 % | within |

### host_launch_us

| point | version | bound | old (spread) | new (spread) | change | |
|---|---|---|---|---|---|---|
| E4a 10 | fkl_fused | host | 2.045 (1.8 %) | 2.098 (2.8 %) | +2.6 % | within |
| E4a 10 | fkl_fused_half_pixel | host | 2.092 (2.2 %) | 2.099 (4.5 %) | +0.3 % | within |
| E4a 10 | opencv_percrop | host | 126.5 (0.8 %) | 126.3 (1.3 %) | -0.2 % | within |
| E4a 10 | npp | host | 130.2 (1.9 %) | 129.5 (1.9 %) | -0.5 % | within |
| E4a 50 | fkl_fused | host | 3.334 (0.5 %) | 3.567 (4.2 %) | +7.0 % | exceeds FLAG |
| E4a 50 | fkl_fused_half_pixel | host | 3.367 (0.8 %) | 3.552 (1.3 %) | +5.5 % | exceeds FLAG |
| E4a 50 | opencv_percrop | host | 634 (1.3 %) | 635.9 (1.6 %) | +0.3 % | within |
| E4a 50 | npp | host | 645.4 (1.5 %) | 641.4 (1.2 %) | -0.6 % | within |
| E4a 100 | fkl_fused | host | 4.956 (1.6 %) | 5.246 (19.2 %) | +5.9 % | within FLAG |
| E4a 100 | fkl_fused_half_pixel | host | 4.975 (0.9 %) | 5.215 (1.0 %) | +4.8 % | exceeds |
| E4a 100 | opencv_percrop | host | 1,271 (1.6 %) | 1,268 (1.1 %) | -0.2 % | within |
| E4a 100 | npp | host | 1,287 (1.8 %) | 1,297 (3.3 %) | +0.7 % | within |
| E4a 150 | fkl_fused | host | 6.927 (1.6 %) | 7.156 (0.1 %) | +3.3 % | exceeds |
| E4a 150 | fkl_fused_half_pixel | host | 6.926 (0.2 %) | 7.155 (1.0 %) | +3.3 % | exceeds |
| E4a 150 | opencv_percrop | host | 1,904 (0.6 %) | 1,902 (1.3 %) | -0.1 % | within |
| E4a 150 | npp | host | 1,924 (2.0 %) | 1,922 (1.0 %) | -0.1 % | within |
| E4a 200 | fkl_fused | host | 8.223 (1.0 %) | 8.703 (1.6 %) | +5.8 % | exceeds FLAG |
| E4a 200 | fkl_fused_half_pixel | host | 8.248 (0.9 %) | 8.704 (2.1 %) | +5.5 % | exceeds FLAG |
| E4a 200 | opencv_percrop | host | 2,538 (1.0 %) | 2,533 (1.0 %) | -0.2 % | within |
| E4a 200 | npp | host | 2,573 (2.0 %) | 2,575 (1.0 %) | +0.1 % | within |
| E4a 250 | fkl_fused | host | 9.939 (2.3 %) | 10.35 (4.2 %) | +4.1 % | within |
| E4a 250 | fkl_fused_half_pixel | host | 9.928 (0.3 %) | 10.28 (3.0 %) | +3.6 % | exceeds |
| E4a 250 | opencv_percrop | host | 3,165 (0.7 %) | 3,169 (3.1 %) | +0.1 % | within |
| E4a 250 | npp | host | 3,209 (2.0 %) | 3,217 (1.4 %) | +0.2 % | within |
| E4a 300 | fkl_fused | host | 12.16 (0.2 %) | 12.33 (0.5 %) | +1.4 % | exceeds |
| E4a 300 | fkl_fused_half_pixel | host | 12.18 (0.9 %) | 12.3 (0.8 %) | +1.0 % | exceeds |
| E4a 300 | opencv_percrop | host | 3,795 (0.8 %) | 3,805 (0.6 %) | +0.3 % | within |
| E4a 300 | npp | host | 3,873 (1.7 %) | 3,855 (0.9 %) | -0.5 % | within |
| E4c 720 | fkl_fused | host | 1.612 (1.6 %) | 1.608 (0.4 %) | -0.2 % | within |
| E4c 720 | fkl_fused_half_pixel | host | 1.613 (1.1 %) | 1.642 (1.1 %) | +1.9 % | exceeds |
| E4c 720 | opencv_percrop | host | 12.57 (1.8 %) | 12.41 (0.8 %) | -1.3 % | within |
| E4c 720 | npp | host | 14.41 (2.8 %) | 14.3 (1.4 %) | -0.8 % | within |
| E4c 1920 | fkl_fused | host | 1.615 (1.7 %) | 1.593 (0.4 %) | -1.4 % | within |
| E4c 1920 | fkl_fused_half_pixel | host | 1.634 (2.4 %) | 1.593 (4.5 %) | -2.5 % | within |
| E4c 1920 | opencv_percrop | host | 12.51 (0.4 %) | 12.32 (1.2 %) | -1.5 % | exceeds |
| E4c 1920 | npp | host | 14.25 (2.7 %) | 14.26 (0.4 %) | +0.0 % | within |
| E4c 3840 | fkl_fused | host | 1.616 (2.0 %) | 1.621 (1.6 %) | +0.3 % | within |
| E4c 3840 | fkl_fused_half_pixel | host | 1.616 (3.5 %) | 1.585 (4.0 %) | -1.9 % | within |
| E4c 3840 | opencv_percrop | host | 12.32 (1.5 %) | 12.26 (1.1 %) | -0.6 % | within |
| E4c 3840 | npp | host | 14.31 (1.4 %) | 14.26 (2.5 %) | -0.4 % | within |

Flagged over all 30 batches and 3 frames (|change| above the flag and above both spreads): 56

- gpu_mean_ms E4a 40 fkl_fused (GPU): +4.0 % (old 0.006528, spread 3.4 %; new 0.006787, spread 1.1 %)
- gpu_mean_ms E4a 90 fkl_fused (GPU): +3.0 % (old 0.01066, spread 0.6 %; new 0.01098, spread 0.5 %)
- host_enqueue_us E4a 50 fkl_fused (host): +5.1 % (old 3.345, spread 2.1 %; new 3.516, spread 1.7 %)
- host_enqueue_us E4a 50 fkl_fused_half_pixel (host): +6.1 % (old 3.317, spread 1.8 %; new 3.52, spread 3.0 %)
- host_enqueue_us E4a 90 fkl_fused (host): +6.2 % (old 4.629, spread 1.1 %; new 4.917, spread 2.2 %)
- host_enqueue_us E4a 90 fkl_fused_half_pixel (host): +6.5 % (old 4.618, spread 0.5 %; new 4.92, spread 5.2 %)
- host_enqueue_us E4a 100 fkl_fused_half_pixel (host): +5.2 % (old 4.954, spread 1.1 %; new 5.214, spread 2.3 %)
- host_enqueue_us E4a 120 fkl_fused (host): +5.5 % (old 5.634, spread 4.4 %; new 5.947, spread 3.9 %)
- host_enqueue_us E4a 130 fkl_fused (host): +5.1 % (old 6.179, spread 0.9 %; new 6.495, spread 3.6 %)
- host_enqueue_us E4a 170 fkl_fused (host): +5.2 % (old 7.522, spread 0.4 %; new 7.91, spread 2.4 %)
- host_enqueue_us E4a 170 fkl_fused_half_pixel (host): +5.0 % (old 7.537, spread 1.0 %; new 7.917, spread 1.2 %)
- host_enqueue_us E4a 180 fkl_fused_half_pixel (host): +5.4 % (old 7.515, spread 0.3 %; new 7.922, spread 1.2 %)
- host_enqueue_us E4a 190 fkl_fused (host): +5.7 % (old 7.836, spread 0.4 %; new 8.28, spread 1.1 %)
- host_enqueue_us E4a 190 fkl_fused_half_pixel (host): +5.7 % (old 7.851, spread 0.7 %; new 8.295, spread 2.0 %)
- host_enqueue_us E4a 200 fkl_fused (host): +5.7 % (old 8.269, spread 1.4 %; new 8.741, spread 2.2 %)
- host_enqueue_us E4a 200 fkl_fused_half_pixel (host): +6.1 % (old 8.207, spread 0.3 %; new 8.708, spread 2.0 %)
- host_enqueue_us E4a 210 fkl_fused (host): +6.0 % (old 8.591, spread 0.9 %; new 9.108, spread 3.1 %)
- host_enqueue_us E4a 210 fkl_fused_half_pixel (host): +5.6 % (old 8.572, spread 0.2 %; new 9.05, spread 1.9 %)
- host_enqueue_us E4a 220 fkl_fused (host): +5.7 % (old 8.868, spread 1.1 %; new 9.374, spread 1.2 %)
- host_enqueue_us E4a 220 fkl_fused_half_pixel (host): +5.6 % (old 8.85, spread 2.1 %; new 9.341, spread 0.3 %)
- host_enqueue_us E4a 230 fkl_fused (host): +5.0 % (old 9.193, spread 2.0 %; new 9.657, spread 1.7 %)
- host_enqueue_us E4a 230 fkl_fused_half_pixel (host): +5.1 % (old 9.201, spread 1.6 %; new 9.672, spread 0.2 %)
- host_enqueue_us E4a 250 fkl_fused (host): +5.2 % (old 9.928, spread 1.5 %; new 10.44, spread 2.9 %)
- host_enqueue_us E4a 270 fkl_fused_half_pixel (host): +5.8 % (old 10.56, spread 0.7 %; new 11.18, spread 1.3 %)
- host_enqueue_us E4a 280 fkl_fused (host): +6.4 % (old 10.99, spread 0.8 %; new 11.69, spread 2.5 %)
- host_enqueue_us E4a 280 fkl_fused_half_pixel (host): +6.9 % (old 10.93, spread 1.0 %; new 11.69, spread 3.1 %)
- host_enqueue_us E4a 290 fkl_fused (host): +6.7 % (old 11.28, spread 0.3 %; new 12.03, spread 2.6 %)
- host_enqueue_us E4a 290 fkl_fused_half_pixel (host): +7.0 % (old 11.33, spread 1.6 %; new 12.12, spread 3.3 %)
- host_launch_us E4a 50 fkl_fused (host): +7.0 % (old 3.334, spread 0.5 %; new 3.567, spread 4.2 %)
- host_launch_us E4a 50 fkl_fused_half_pixel (host): +5.5 % (old 3.367, spread 0.8 %; new 3.552, spread 1.3 %)
- host_launch_us E4a 90 fkl_fused (host): +6.9 % (old 4.621, spread 0.9 %; new 4.94, spread 2.0 %)
- host_launch_us E4a 90 fkl_fused_half_pixel (host): +6.8 % (old 4.633, spread 0.8 %; new 4.948, spread 6.0 %)
- host_launch_us E4a 120 fkl_fused (host): +5.1 % (old 5.644, spread 0.3 %; new 5.933, spread 4.7 %)
- host_launch_us E4a 120 fkl_fused_half_pixel (host): +6.4 % (old 5.651, spread 1.0 %; new 6.01, spread 2.8 %)
- host_launch_us E4a 130 fkl_fused (host): +6.2 % (old 6.172, spread 0.2 %; new 6.552, spread 3.8 %)
- host_launch_us E4a 130 fkl_fused_half_pixel (host): +5.4 % (old 6.196, spread 1.3 %; new 6.527, spread 3.2 %)
- host_launch_us E4a 140 fkl_fused (host): +5.1 % (old 6.288, spread 0.6 %; new 6.608, spread 0.7 %)
- host_launch_us E4a 140 fkl_fused_half_pixel (host): +6.1 % (old 6.256, spread 0.7 %; new 6.639, spread 1.0 %)
- host_launch_us E4a 180 fkl_fused (host): +5.7 % (old 7.535, spread 1.0 %; new 7.965, spread 2.7 %)
- host_launch_us E4a 190 fkl_fused (host): +6.5 % (old 7.806, spread 0.5 %; new 8.313, spread 0.6 %)
- host_launch_us E4a 190 fkl_fused_half_pixel (host): +6.2 % (old 7.843, spread 1.4 %; new 8.326, spread 2.3 %)
- host_launch_us E4a 200 fkl_fused (host): +5.8 % (old 8.223, spread 1.0 %; new 8.703, spread 1.6 %)
- host_launch_us E4a 200 fkl_fused_half_pixel (host): +5.5 % (old 8.248, spread 0.9 %; new 8.704, spread 2.1 %)
- host_launch_us E4a 210 fkl_fused (host): +6.0 % (old 8.572, spread 0.4 %; new 9.084, spread 3.2 %)
- host_launch_us E4a 210 fkl_fused_half_pixel (host): +6.2 % (old 8.55, spread 0.6 %; new 9.078, spread 0.7 %)
- host_launch_us E4a 220 fkl_fused (host): +5.9 % (old 8.841, spread 5.6 %; new 9.365, spread 1.3 %)
- host_launch_us E4a 220 fkl_fused_half_pixel (host): +5.6 % (old 8.875, spread 0.8 %; new 9.375, spread 1.3 %)
- host_launch_us E4a 230 fkl_fused (host): +6.2 % (old 9.144, spread 0.4 %; new 9.714, spread 0.8 %)
- host_launch_us E4a 230 fkl_fused_half_pixel (host): +6.2 % (old 9.148, spread 0.6 %; new 9.719, spread 0.6 %)
- host_launch_us E4a 260 fkl_fused_half_pixel (host): +5.5 % (old 10.3, spread 1.7 %; new 10.87, spread 1.0 %)
- host_launch_us E4a 270 fkl_fused (host): +7.2 % (old 10.52, spread 0.6 %; new 11.28, spread 1.1 %)
- host_launch_us E4a 270 fkl_fused_half_pixel (host): +6.2 % (old 10.54, spread 0.4 %; new 11.2, spread 2.2 %)
- host_launch_us E4a 280 fkl_fused (host): +6.1 % (old 11.03, spread 1.6 %; new 11.71, spread 3.1 %)
- host_launch_us E4a 280 fkl_fused_half_pixel (host): +6.6 % (old 11, spread 0.6 %; new 11.73, spread 2.9 %)
- host_launch_us E4a 290 fkl_fused (host): +6.5 % (old 11.32, spread 1.0 %; new 12.06, spread 2.5 %)
- host_launch_us E4a 290 fkl_fused_half_pixel (host): +6.5 % (old 11.3, spread 2.3 %; new 12.03, spread 2.6 %)

## Agreement

- ok_own (against the scalar reference of the version's own convention, 1e-4): 846 of 846 rows pass
- pairs (the outputs of the checked calls; 30 batches x 3 runs per pair):
  - cvgs_fused vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - cvgs_fused vs opencv_percrop: bitwise equal in 0 of 90, largest |difference| 7.63e-06, within 1e-4 in 90 of 90
  - fkl_fused_after vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - fkl_fused_late vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - hand_batch vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - hand_padded vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90

## The runs and the SM clock

Sampler: 1534 samples of nvidia-smi every 100 ms (gpu_clocks.csv).

| run | start | end | SM clock median MHz | min MHz | samples | samples below 2,500 MHz |
|---|---|---|---|---|---|---|
| warmup | 12:13:16 | 12:13:20 | 2,700 | 210 | 32 | 6 |
| quiet-run1 | 12:13:20 | 12:14:10 | 2,685 | 2,550 | 499 | 0 |
| quiet-run2 | 12:14:10 | 12:15:00 | 2,685 | 2,550 | 500 | 0 |
| quiet-run3 | 12:15:00 | 12:15:50 | 2,685 | 2,550 | 499 | 0 |

