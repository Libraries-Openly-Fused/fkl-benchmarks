# E4 window G (e4-hand-roi), RTX 4090: the final data of Figure 15 (report)

Written by `e4_hand_roi_report.py` from the result folders only, 2026-11-08. Times: gpu_mean_ms, CUDA events around one call of the whole implementation (E4's protocol), 100 calls after one warm-up; per point the median of the three runs. Ratios: the median of the three per-run ratios, with the smallest and the largest run.

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

## The new series (E4a)

### hand_roi / FKL (the hand line of the figure: the hand-written kernel with its CPU function, over FKL)

FKL time `adjacent` = the mean of fkl_fused_late (timed just before hand_batch) and fkl_fused_after (timed just after hand_roi).

| batch | hand_roi ms | fkl_fused ms | adjacent FKL ms | hand_roi / adjacent (min - max) | hand_roi / fkl_fused | hand_roi / hand_batch |
|---|---|---|---|---|---|---|
| 10 | 0.003579 | 0.003812 | 0.003725 | 0.960 (0.949 - 0.981) | 0.945 | 1.036 |
| 20 | 0.003985 | 0.004909 | 0.004885 | 0.818 (0.807 - 0.826) | 0.812 | 0.991 |
| 30 | 0.004487 | 0.005635 | 0.005642 | 0.795 (0.790 - 0.802) | 0.795 | 1.005 |
| 40 | 0.005103 | 0.006786 | 0.006802 | 0.759 (0.744 - 0.767) | 0.748 | 1.032 |
| 50 | 0.005532 | 0.00755 | 0.007561 | 0.734 (0.726 - 0.737) | 0.738 | 0.996 |
| 60 | 0.006052 | 0.008377 | 0.008337 | 0.726 (0.719 - 0.734) | 0.718 | 1.026 |
| 70 | 0.006509 | 0.009134 | 0.009199 | 0.708 (0.701 - 0.713) | 0.715 | 1.025 |
| 80 | 0.007075 | 0.009985 | 0.009926 | 0.712 (0.711 - 0.713) | 0.709 | 1.024 |
| 90 | 0.007538 | 0.01087 | 0.01091 | 0.690 (0.686 - 0.694) | 0.693 | 1.024 |
| 100 | 0.00805 | 0.01161 | 0.01161 | 0.692 (0.690 - 0.700) | 0.697 | 1.034 |
| 110 | 0.008666 | 0.01254 | 0.01248 | 0.694 (0.694 - 0.695) | 0.691 | 1.040 |
| 120 | 0.009183 | 0.01319 | 0.01318 | 0.697 (0.693 - 0.704) | 0.696 | 1.035 |
| 130 | 0.009549 | 0.01416 | 0.01413 | 0.676 (0.674 - 0.682) | 0.675 | 1.046 |
| 140 | 0.01016 | 0.01472 | 0.01465 | 0.692 (0.691 - 0.693) | 0.689 | 1.057 |
| 150 | 0.01052 | 0.01567 | 0.01564 | 0.674 (0.671 - 0.677) | 0.671 | 1.055 |
| 200 | 0.01297 | 0.01955 | 0.01953 | 0.664 (0.660 - 0.672) | 0.663 | 1.072 |
| 250 | 0.01578 | 0.02349 | 0.02348 | 0.672 (0.671 - 0.673) | 0.672 | 1.083 |
| 300 | 0.01814 | 0.02766 | 0.02755 | 0.659 (0.641 - 0.662) | 0.656 | 1.092 |

### hand_batch / FKL (E1c's kernel with compile-time crop sizes, a repository diagnostic)


| batch | hand_batch ms | fkl_fused ms | adjacent FKL ms | hand / fkl_fused | hand / adjacent (min - max) | hand / late | hand / after |
|---|---|---|---|---|---|---|---|
| 10 | 0.003442 | 0.003812 | 0.003725 | 0.908 | 0.926 (0.911 - 0.927) | 0.933 | 0.915 |
| 20 | 0.004063 | 0.004909 | 0.004885 | 0.828 | 0.830 (0.826 - 0.832) | 0.827 | 0.833 |
| 30 | 0.00447 | 0.005635 | 0.005642 | 0.788 | 0.796 (0.783 - 0.798) | 0.797 | 0.795 |
| 40 | 0.004946 | 0.006786 | 0.006802 | 0.727 | 0.728 (0.721 - 0.738) | 0.725 | 0.731 |
| 50 | 0.005554 | 0.00755 | 0.007561 | 0.736 | 0.735 (0.729 - 0.744) | 0.739 | 0.731 |
| 60 | 0.005861 | 0.008377 | 0.008337 | 0.700 | 0.706 (0.701 - 0.708) | 0.706 | 0.705 |
| 70 | 0.006358 | 0.009134 | 0.009199 | 0.696 | 0.691 (0.684 - 0.693) | 0.684 | 0.694 |
| 80 | 0.006907 | 0.009985 | 0.009926 | 0.695 | 0.696 (0.693 - 0.703) | 0.693 | 0.699 |
| 90 | 0.007328 | 0.01087 | 0.01091 | 0.675 | 0.671 (0.671 - 0.674) | 0.671 | 0.675 |
| 100 | 0.007784 | 0.01161 | 0.01161 | 0.667 | 0.669 (0.667 - 0.673) | 0.670 | 0.669 |
| 110 | 0.008327 | 0.01254 | 0.01248 | 0.664 | 0.668 (0.662 - 0.669) | 0.668 | 0.667 |
| 120 | 0.008904 | 0.01319 | 0.01318 | 0.675 | 0.675 (0.663 - 0.680) | 0.675 | 0.675 |
| 130 | 0.009135 | 0.01416 | 0.01413 | 0.645 | 0.646 (0.644 - 0.652) | 0.644 | 0.649 |
| 140 | 0.009575 | 0.01472 | 0.01465 | 0.651 | 0.656 (0.652 - 0.656) | 0.654 | 0.657 |
| 150 | 0.009971 | 0.01567 | 0.01564 | 0.637 | 0.639 (0.636 - 0.641) | 0.639 | 0.640 |
| 200 | 0.01212 | 0.01955 | 0.01953 | 0.621 | 0.621 (0.615 - 0.624) | 0.623 | 0.620 |
| 250 | 0.01457 | 0.02349 | 0.02348 | 0.620 | 0.620 (0.619 - 0.621) | 0.620 | 0.619 |
| 300 | 0.01661 | 0.02766 | 0.02755 | 0.600 | 0.603 (0.585 - 0.607) | 0.603 | 0.603 |

### hand_padded (the larger launch-parameter block) and the position of FKL

| batch | hand_padded ms | hand_padded / adjacent FKL | hand_padded / hand_batch | fkl_fused_late / fkl_fused | fkl_fused_after / fkl_fused |
|---|---|---|---|---|---|
| 10 | 0.003621 | 0.972 | 1.050 | 0.973 (0.973 - 0.980) | 0.986 (0.974 - 0.997) |
| 20 | 0.00408 | 0.835 | 1.004 | 1.001 (0.982 - 1.018) | 1.003 (0.993 - 1.007) |
| 30 | 0.004693 | 0.834 | 1.047 | 0.993 (0.986 - 1.011) | 0.998 (0.988 - 1.002) |
| 40 | 0.005186 | 0.767 | 1.051 | 0.994 (0.976 - 1.010) | 1.000 (0.995 - 1.003) |
| 50 | 0.005642 | 0.748 | 1.017 | 1.004 (0.996 - 1.009) | 1.007 (1.002 - 1.008) |
| 60 | 0.006153 | 0.742 | 1.050 | 0.992 (0.948 - 0.993) | 0.999 (0.944 - 1.005) |
| 70 | 0.006587 | 0.716 | 1.036 | 1.004 (0.992 - 1.022) | 1.002 (0.991 - 1.003) |
| 80 | 0.007117 | 0.716 | 1.030 | 0.998 (0.996 - 1.002) | 0.993 (0.989 - 0.994) |
| 90 | 0.007659 | 0.702 | 1.041 | 1.005 (0.987 - 1.006) | 1.002 (0.977 - 1.006) |
| 100 | 0.0081 | 0.698 | 1.046 | 1.003 (0.983 - 1.006) | 0.996 (0.989 - 1.010) |
| 110 | 0.008729 | 0.700 | 1.048 | 0.993 (0.988 - 0.994) | 0.996 (0.994 - 1.004) |
| 120 | 0.009264 | 0.703 | 1.040 | 0.996 (0.996 - 1.000) | 1.004 (0.998 - 1.010) |
| 130 | 0.009744 | 0.690 | 1.067 | 1.001 (0.991 - 1.002) | 0.994 (0.989 - 1.003) |
| 140 | 0.01012 | 0.693 | 1.058 | 0.996 (0.995 - 0.997) | 0.993 (0.992 - 0.996) |
| 150 | 0.01065 | 0.683 | 1.068 | 0.997 (0.994 - 1.002) | 0.996 (0.995 - 1.000) |
| 200 | 0.01319 | 0.677 | 1.089 | 0.996 (0.991 - 0.997) | 1.000 (0.998 - 1.001) |
| 250 | 0.0159 | 0.677 | 1.092 | 0.998 (0.997 - 1.001) | 1.001 (0.998 - 1.004) |
| 300 | 0.01812 | 0.658 | 1.092 | 0.996 (0.996 - 0.998) | 0.996 (0.996 - 0.998) |

### cvGPUSpeedup against OpenCV-CUDA per crop

| batch | opencv_percrop ms | cvgs_fused ms | speedup opencv_percrop / cvgs_fused (min - max) | cvgs_fused / fkl_fused |
|---|---|---|---|---|
| 10 | 0.1286 | 0.00385 | 33.4 (32.9 - 34.1) | 1.006 |
| 20 | 0.253 | 0.004949 | 50.9 (49.8 - 51.9) | 1.008 |
| 30 | 0.3793 | 0.005823 | 65.1 (64.0 - 65.4) | 1.030 |
| 40 | 0.5045 | 0.007019 | 71.8 (68.8 - 72.4) | 1.034 |
| 50 | 0.6309 | 0.007951 | 79.5 (75.2 - 80.3) | 1.061 |
| 60 | 0.7575 | 0.008788 | 85.8 (82.8 - 87.0) | 1.043 |
| 70 | 0.8803 | 0.009496 | 92.7 (92.5 - 93.1) | 1.044 |
| 80 | 1.007 | 0.01028 | 98.3 (90.0 - 98.6) | 1.030 |
| 90 | 1.133 | 0.01132 | 100.1 (99.7 - 100.1) | 1.040 |
| 100 | 1.261 | 0.01194 | 105.3 (105.0 - 105.7) | 1.034 |
| 110 | 1.384 | 0.01304 | 106.0 (105.1 - 107.5) | 1.041 |
| 120 | 1.509 | 0.01396 | 108.1 (107.8 - 110.2) | 1.059 |
| 130 | 1.632 | 0.01487 | 110.0 (109.4 - 112.9) | 1.049 |
| 140 | 1.755 | 0.01533 | 114.7 (111.2 - 115.2) | 1.043 |
| 150 | 1.885 | 0.01652 | 113.8 (113.2 - 116.5) | 1.056 |
| 200 | 2.52 | 0.02063 | 121.5 (121.2 - 124.3) | 1.053 |
| 250 | 3.149 | 0.02533 | 124.1 (124.0 - 127.8) | 1.078 |
| 300 | 3.78 | 0.02874 | 131.0 (126.6 - 131.5) | 1.048 |

### Host time of one call (host_enqueue_us: the call returns, the stream idle before it; median of the three runs, us)

| batch | fkl_fused | fkl_fused_late | fkl_fused_after | hand_roi | hand_batch | hand_padded | cvgs_fused | opencv_percrop |
|---|---|---|---|---|---|---|---|---|
| 10 | 2.034 | 2.025 | 2.012 | 1.601 | 1.588 | 1.611 | 2.085 | 126.5 |
| 20 | 2.396 | 2.445 | 2.48 | 1.636 | 1.632 | 1.662 | 2.522 | 253.3 |
| 30 | 2.696 | 2.709 | 2.725 | 1.651 | 1.653 | 1.661 | 2.962 | 379.3 |
| 40 | 3.083 | 3.056 | 3.062 | 1.695 | 1.615 | 1.673 | 3.25 | 505.6 |
| 50 | 3.39 | 3.423 | 3.435 | 1.738 | 1.631 | 1.744 | 3.72 | 634.2 |
| 60 | 3.77 | 3.766 | 3.717 | 1.794 | 1.689 | 1.769 | 4.091 | 758.8 |
| 70 | 4.098 | 4.158 | 4.124 | 1.797 | 1.634 | 1.805 | 4.54 | 884.4 |
| 80 | 4.481 | 4.44 | 4.433 | 1.853 | 1.651 | 1.851 | 5.012 | 1,012 |
| 90 | 4.915 | 4.915 | 4.897 | 1.913 | 1.662 | 1.888 | 5.384 | 1,139 |
| 100 | 5.1 | 5.077 | 5.122 | 1.95 | 1.677 | 1.884 | 5.682 | 1,266 |
| 110 | 5.651 | 5.606 | 5.605 | 1.958 | 1.679 | 1.984 | 6.18 | 1,388 |
| 120 | 5.773 | 5.789 | 5.917 | 2.013 | 1.693 | 2.018 | 6.733 | 1,520 |
| 130 | 6.525 | 6.507 | 6.474 | 2.068 | 1.665 | 2.1 | 7.152 | 1,644 |
| 140 | 6.605 | 6.663 | 6.579 | 2.083 | 1.67 | 2.09 | 7.227 | 1,771 |
| 150 | 7.137 | 7.127 | 7.124 | 2.117 | 1.68 | 2.131 | 7.955 | 1,891 |
| 200 | 8.627 | 8.595 | 8.526 | 2.339 | 1.702 | 2.313 | 9.606 | 2,534 |
| 250 | 10.48 | 10.43 | 10.43 | 2.557 | 1.739 | 2.557 | 12.21 | 3,157 |
| 300 | 12.31 | 12.32 | 12.37 | 2.768 | 1.784 | 2.741 | 13.64 | 3,797 |

## Every version against the window e4-hand-cvgs (standing form)

old = the window e4-hand-cvgs (main 2dac1c8, 2026-11-08 12:13), new = this window (main 469aa6d; the kernels of 2dac1c8 identical, plus hand_roi's); median of three runs, spread = (largest - smallest) / median. In every batch hand_roi now runs between hand_padded and fkl_fused_after; the diagnostic F2 showed that the host code of the same FKL call can change with the content of the translation unit (here: hand_roi's 30 kernels).

### gpu_mean_ms

| point | version | bound | old (spread) | new (spread) | change | |
|---|---|---|---|---|---|---|
| E4a 10 | fkl_fused | GPU | 0.003797 (2.7 %) | 0.003812 (2.0 %) | +0.4 % | within |
| E4a 10 | fkl_fused_half_pixel | GPU | 0.003838 (9.1 %) | 0.003814 (2.1 %) | -0.6 % | within |
| E4a 10 | opencv_percrop | host | 0.1299 (1.7 %) | 0.1286 (1.3 %) | -1.0 % | within |
| E4a 10 | npp | host | 0.1337 (1.8 %) | 0.1313 (0.8 %) | -1.8 % | within |
| E4a 10 | fkl_fused_late | GPU | 0.003728 (0.8 %) | 0.003736 (2.0 %) | +0.2 % | within |
| E4a 10 | hand_batch | GPU | 0.00344 (2.1 %) | 0.003442 (2.9 %) | +0.1 % | within |
| E4a 10 | hand_padded | GPU | 0.00361 (3.0 %) | 0.003621 (2.2 %) | +0.3 % | within |
| E4a 10 | fkl_fused_after | GPU | 0.003723 (0.7 %) | 0.003762 (2.2 %) | +1.1 % | within |
| E4a 10 | cvgs_fused | GPU | 0.003942 (9.0 %) | 0.00385 (2.0 %) | -2.3 % | within |
| E4a 50 | fkl_fused | GPU | 0.00758 (1.0 %) | 0.00755 (0.7 %) | -0.4 % | within |
| E4a 50 | fkl_fused_half_pixel | GPU | 0.007625 (0.8 %) | 0.007601 (1.4 %) | -0.3 % | within |
| E4a 50 | opencv_percrop | host | 0.6364 (4.3 %) | 0.6309 (0.4 %) | -0.9 % | within |
| E4a 50 | npp | host | 0.6517 (0.5 %) | 0.6468 (0.7 %) | -0.8 % | exceeds |
| E4a 50 | fkl_fused_late | GPU | 0.007587 (0.3 %) | 0.007524 (1.4 %) | -0.8 % | within |
| E4a 50 | hand_batch | GPU | 0.00555 (0.5 %) | 0.005554 (1.4 %) | +0.1 % | within |
| E4a 50 | hand_padded | GPU | 0.005644 (0.3 %) | 0.005642 (1.1 %) | -0.0 % | within |
| E4a 50 | fkl_fused_after | GPU | 0.007532 (1.4 %) | 0.007564 (0.6 %) | +0.4 % | within |
| E4a 50 | cvgs_fused | GPU | 0.007898 (0.8 %) | 0.007951 (6.8 %) | +0.7 % | within |
| E4a 100 | fkl_fused | GPU | 0.0116 (0.4 %) | 0.01161 (1.5 %) | +0.1 % | within |
| E4a 100 | fkl_fused_half_pixel | GPU | 0.01164 (0.6 %) | 0.01164 (2.4 %) | +0.0 % | within |
| E4a 100 | opencv_percrop | host | 1.272 (1.8 %) | 1.261 (2.8 %) | -0.8 % | within |
| E4a 100 | npp | host | 1.295 (1.1 %) | 1.293 (3.4 %) | -0.2 % | within |
| E4a 100 | fkl_fused_late | GPU | 0.01154 (0.2 %) | 0.01162 (1.0 %) | +0.7 % | within |
| E4a 100 | hand_batch | GPU | 0.007771 (0.3 %) | 0.007784 (0.5 %) | +0.2 % | within |
| E4a 100 | hand_padded | GPU | 0.008135 (1.8 %) | 0.0081 (1.3 %) | -0.4 % | within |
| E4a 100 | fkl_fused_after | GPU | 0.01157 (0.7 %) | 0.0116 (0.8 %) | +0.2 % | within |
| E4a 100 | cvgs_fused | GPU | 0.01227 (8.5 %) | 0.01194 (2.6 %) | -2.7 % | within |
| E4a 150 | fkl_fused | GPU | 0.0157 (0.5 %) | 0.01567 (0.5 %) | -0.2 % | within |
| E4a 150 | fkl_fused_half_pixel | GPU | 0.01581 (1.3 %) | 0.01579 (0.8 %) | -0.1 % | within |
| E4a 150 | opencv_percrop | host | 1.902 (1.7 %) | 1.885 (0.6 %) | -0.9 % | within |
| E4a 150 | npp | host | 1.934 (1.3 %) | 1.924 (1.0 %) | -0.5 % | within |
| E4a 150 | fkl_fused_late | GPU | 0.01568 (0.7 %) | 0.01563 (0.7 %) | -0.3 % | within |
| E4a 150 | hand_batch | GPU | 0.00999 (0.6 %) | 0.009971 (0.6 %) | -0.2 % | within |
| E4a 150 | hand_padded | GPU | 0.01069 (0.3 %) | 0.01065 (0.5 %) | -0.4 % | within |
| E4a 150 | fkl_fused_after | GPU | 0.01562 (0.9 %) | 0.01565 (0.5 %) | +0.2 % | within |
| E4a 150 | cvgs_fused | GPU | 0.01622 (12.7 %) | 0.01652 (3.1 %) | +1.8 % | within |
| E4a 200 | fkl_fused | GPU | 0.01951 (1.3 %) | 0.01955 (1.5 %) | +0.2 % | within |
| E4a 200 | fkl_fused_half_pixel | GPU | 0.01943 (1.8 %) | 0.01954 (1.3 %) | +0.5 % | within |
| E4a 200 | opencv_percrop | host | 2.535 (1.2 %) | 2.52 (0.6 %) | -0.6 % | within |
| E4a 200 | npp | host | 2.582 (1.3 %) | 2.576 (1.2 %) | -0.2 % | within |
| E4a 200 | fkl_fused_late | GPU | 0.01939 (1.7 %) | 0.01948 (0.8 %) | +0.5 % | within |
| E4a 200 | hand_batch | GPU | 0.01219 (1.1 %) | 0.01212 (0.3 %) | -0.6 % | within |
| E4a 200 | hand_padded | GPU | 0.01315 (0.8 %) | 0.01319 (0.4 %) | +0.4 % | within |
| E4a 200 | fkl_fused_after | GPU | 0.01945 (1.5 %) | 0.01957 (1.7 %) | +0.6 % | within |
| E4a 200 | cvgs_fused | GPU | 0.02088 (0.6 %) | 0.02063 (2.5 %) | -1.2 % | within |
| E4a 250 | fkl_fused | GPU | 0.02357 (1.5 %) | 0.02349 (0.5 %) | -0.3 % | within |
| E4a 250 | fkl_fused_half_pixel | GPU | 0.0236 (2.1 %) | 0.02345 (0.3 %) | -0.6 % | within |
| E4a 250 | opencv_percrop | host | 3.156 (1.2 %) | 3.149 (0.4 %) | -0.2 % | within |
| E4a 250 | npp | host | 3.228 (0.5 %) | 3.213 (1.1 %) | -0.5 % | within |
| E4a 250 | fkl_fused_late | GPU | 0.02359 (7.7 %) | 0.02346 (0.9 %) | -0.6 % | within |
| E4a 250 | hand_batch | GPU | 0.01461 (0.7 %) | 0.01457 (0.3 %) | -0.3 % | within |
| E4a 250 | hand_padded | GPU | 0.01587 (0.1 %) | 0.0159 (0.4 %) | +0.2 % | within |
| E4a 250 | fkl_fused_after | GPU | 0.02352 (1.4 %) | 0.02351 (0.2 %) | -0.1 % | within |
| E4a 250 | cvgs_fused | GPU | 0.02466 (0.9 %) | 0.02533 (2.7 %) | +2.7 % | exceeds |
| E4a 300 | fkl_fused | GPU | 0.02776 (0.7 %) | 0.02766 (3.6 %) | -0.4 % | within |
| E4a 300 | fkl_fused_half_pixel | GPU | 0.02756 (0.5 %) | 0.02764 (4.1 %) | +0.3 % | within |
| E4a 300 | opencv_percrop | host | 3.808 (1.0 %) | 3.78 (0.5 %) | -0.7 % | within |
| E4a 300 | npp | host | 3.88 (1.0 %) | 3.856 (1.5 %) | -0.6 % | within |
| E4a 300 | fkl_fused_late | GPU | 0.02749 (0.9 %) | 0.02755 (3.3 %) | +0.2 % | within |
| E4a 300 | hand_batch | GPU | 0.01657 (0.5 %) | 0.01661 (0.2 %) | +0.2 % | within |
| E4a 300 | hand_padded | GPU | 0.01813 (0.4 %) | 0.01812 (0.3 %) | -0.0 % | within |
| E4a 300 | fkl_fused_after | GPU | 0.0276 (0.4 %) | 0.02755 (3.8 %) | -0.2 % | within |
| E4a 300 | cvgs_fused | GPU | 0.02969 (0.5 %) | 0.02874 (3.9 %) | -3.2 % | within FLAG |
| E4c 720 | fkl_fused | GPU | 0.005174 (1.0 %) | 0.005207 (1.2 %) | +0.6 % | within |
| E4c 720 | fkl_fused_half_pixel | GPU | 0.005193 (0.1 %) | 0.005253 (1.2 %) | +1.2 % | within |
| E4c 720 | opencv_percrop | GPU | 0.02762 (0.2 %) | 0.02769 (0.3 %) | +0.3 % | within |
| E4c 720 | npp | GPU | 0.03612 (0.2 %) | 0.03616 (0.2 %) | +0.1 % | within |
| E4c 1920 | fkl_fused | GPU | 0.0155 (0.3 %) | 0.0155 (0.4 %) | -0.0 % | within |
| E4c 1920 | fkl_fused_half_pixel | GPU | 0.01562 (0.2 %) | 0.01563 (0.5 %) | +0.0 % | within |
| E4c 1920 | opencv_percrop | GPU | 0.1745 (0.1 %) | 0.1744 (0.2 %) | -0.1 % | within |
| E4c 1920 | npp | GPU | 0.155 (0.5 %) | 0.1553 (0.1 %) | +0.2 % | within |
| E4c 3840 | fkl_fused | GPU | 0.1577 (0.2 %) | 0.1578 (0.1 %) | +0.1 % | within |
| E4c 3840 | fkl_fused_half_pixel | GPU | 0.1579 (0.1 %) | 0.1578 (0.1 %) | -0.1 % | within |
| E4c 3840 | opencv_percrop | GPU | 1.606 (0.0 %) | 1.605 (0.0 %) | -0.0 % | exceeds |
| E4c 3840 | npp | GPU | 1.847 (0.1 %) | 1.846 (0.1 %) | -0.0 % | within |

### host_enqueue_us

| point | version | bound | old (spread) | new (spread) | change | |
|---|---|---|---|---|---|---|
| E4a 10 | fkl_fused | host | 2.083 (8.3 %) | 2.034 (3.8 %) | -2.3 % | within |
| E4a 10 | fkl_fused_half_pixel | host | 2.099 (3.0 %) | 2.074 (3.4 %) | -1.2 % | within |
| E4a 10 | opencv_percrop | host | 126.5 (1.0 %) | 126.5 (1.1 %) | -0.0 % | within |
| E4a 10 | npp | host | 129.8 (1.8 %) | 129.8 (1.0 %) | -0.0 % | within |
| E4a 10 | fkl_fused_late | host | 2.037 (7.4 %) | 2.025 (25.1 %) | -0.6 % | within |
| E4a 10 | hand_batch | host | 1.611 (0.4 %) | 1.588 (2.1 %) | -1.4 % | within |
| E4a 10 | hand_padded | host | 1.638 (8.6 %) | 1.611 (0.4 %) | -1.6 % | within |
| E4a 10 | fkl_fused_after | host | 2.022 (1.8 %) | 2.012 (0.9 %) | -0.5 % | within |
| E4a 10 | cvgs_fused | host | 2.193 (8.1 %) | 2.085 (2.2 %) | -4.9 % | within |
| E4a 50 | fkl_fused | host | 3.516 (1.7 %) | 3.39 (1.8 %) | -3.6 % | exceeds |
| E4a 50 | fkl_fused_half_pixel | host | 3.52 (3.0 %) | 3.43 (0.2 %) | -2.6 % | within |
| E4a 50 | opencv_percrop | host | 638.8 (2.8 %) | 634.2 (1.4 %) | -0.7 % | within |
| E4a 50 | npp | host | 644.1 (1.3 %) | 643.9 (1.5 %) | -0.0 % | within |
| E4a 50 | fkl_fused_late | host | 3.454 (6.3 %) | 3.423 (1.4 %) | -0.9 % | within |
| E4a 50 | hand_batch | host | 1.655 (2.6 %) | 1.631 (2.4 %) | -1.4 % | within |
| E4a 50 | hand_padded | host | 1.729 (3.4 %) | 1.744 (0.9 %) | +0.8 % | within |
| E4a 50 | fkl_fused_after | host | 3.463 (4.2 %) | 3.435 (0.5 %) | -0.8 % | within |
| E4a 50 | cvgs_fused | host | 3.67 (4.3 %) | 3.72 (2.8 %) | +1.4 % | within |
| E4a 100 | fkl_fused | host | 5.2 (1.3 %) | 5.1 (3.9 %) | -1.9 % | within |
| E4a 100 | fkl_fused_half_pixel | host | 5.214 (2.3 %) | 5.15 (7.7 %) | -1.2 % | within |
| E4a 100 | opencv_percrop | host | 1,270 (1.5 %) | 1,266 (0.8 %) | -0.3 % | within |
| E4a 100 | npp | host | 1,285 (1.4 %) | 1,281 (0.8 %) | -0.3 % | within |
| E4a 100 | fkl_fused_late | host | 5.15 (0.5 %) | 5.077 (1.8 %) | -1.4 % | within |
| E4a 100 | hand_batch | host | 1.648 (2.7 %) | 1.677 (3.5 %) | +1.7 % | within |
| E4a 100 | hand_padded | host | 1.913 (2.8 %) | 1.884 (4.1 %) | -1.5 % | within |
| E4a 100 | fkl_fused_after | host | 5.162 (1.1 %) | 5.122 (0.9 %) | -0.8 % | within |
| E4a 100 | cvgs_fused | host | 5.895 (0.3 %) | 5.682 (4.9 %) | -3.6 % | within |
| E4a 150 | fkl_fused | host | 7.15 (0.7 %) | 7.137 (1.2 %) | -0.2 % | within |
| E4a 150 | fkl_fused_half_pixel | host | 7.152 (2.6 %) | 7.126 (1.4 %) | -0.4 % | within |
| E4a 150 | opencv_percrop | host | 1,905 (1.4 %) | 1,891 (0.4 %) | -0.7 % | within |
| E4a 150 | npp | host | 1,923 (1.2 %) | 1,913 (1.2 %) | -0.5 % | within |
| E4a 150 | fkl_fused_late | host | 7.153 (0.6 %) | 7.127 (0.3 %) | -0.4 % | within |
| E4a 150 | hand_batch | host | 1.676 (0.5 %) | 1.68 (4.1 %) | +0.3 % | within |
| E4a 150 | hand_padded | host | 2.139 (1.8 %) | 2.131 (1.2 %) | -0.4 % | within |
| E4a 150 | fkl_fused_after | host | 7.09 (1.9 %) | 7.124 (0.3 %) | +0.5 % | within |
| E4a 150 | cvgs_fused | host | 7.604 (0.8 %) | 7.955 (5.1 %) | +4.6 % | within |
| E4a 200 | fkl_fused | host | 8.741 (2.2 %) | 8.627 (1.0 %) | -1.3 % | within |
| E4a 200 | fkl_fused_half_pixel | host | 8.708 (2.0 %) | 8.585 (1.3 %) | -1.4 % | within |
| E4a 200 | opencv_percrop | host | 2,534 (0.6 %) | 2,534 (1.2 %) | +0.0 % | within |
| E4a 200 | npp | host | 2,578 (0.7 %) | 2,567 (1.1 %) | -0.4 % | within |
| E4a 200 | fkl_fused_late | host | 8.68 (2.8 %) | 8.595 (0.9 %) | -1.0 % | within |
| E4a 200 | hand_batch | host | 1.685 (2.7 %) | 1.702 (0.9 %) | +1.0 % | within |
| E4a 200 | hand_padded | host | 2.357 (17.8 %) | 2.313 (4.1 %) | -1.9 % | within |
| E4a 200 | fkl_fused_after | host | 8.762 (2.9 %) | 8.526 (1.0 %) | -2.7 % | within |
| E4a 200 | cvgs_fused | host | 10.18 (2.1 %) | 9.606 (6.3 %) | -5.6 % | within FLAG |
| E4a 250 | fkl_fused | host | 10.44 (2.9 %) | 10.48 (1.0 %) | +0.3 % | within |
| E4a 250 | fkl_fused_half_pixel | host | 10.37 (3.5 %) | 10.45 (1.0 %) | +0.8 % | within |
| E4a 250 | opencv_percrop | host | 3,189 (1.1 %) | 3,157 (0.4 %) | -1.0 % | within |
| E4a 250 | npp | host | 3,214 (1.3 %) | 3,201 (1.2 %) | -0.4 % | within |
| E4a 250 | fkl_fused_late | host | 10.3 (2.9 %) | 10.43 (2.5 %) | +1.3 % | within |
| E4a 250 | hand_batch | host | 1.727 (2.3 %) | 1.739 (1.5 %) | +0.7 % | within |
| E4a 250 | hand_padded | host | 2.547 (1.9 %) | 2.557 (1.4 %) | +0.4 % | within |
| E4a 250 | fkl_fused_after | host | 10.37 (2.6 %) | 10.43 (1.6 %) | +0.6 % | within |
| E4a 250 | cvgs_fused | host | 11.65 (1.6 %) | 12.21 (8.2 %) | +4.8 % | within |
| E4a 300 | fkl_fused | host | 12.33 (0.3 %) | 12.31 (1.1 %) | -0.2 % | within |
| E4a 300 | fkl_fused_half_pixel | host | 12.32 (1.7 %) | 12.33 (2.6 %) | +0.1 % | within |
| E4a 300 | opencv_percrop | host | 3,815 (1.1 %) | 3,797 (0.4 %) | -0.5 % | within |
| E4a 300 | npp | host | 3,849 (1.0 %) | 3,846 (1.3 %) | -0.1 % | within |
| E4a 300 | fkl_fused_late | host | 12.37 (1.5 %) | 12.32 (2.0 %) | -0.4 % | within |
| E4a 300 | hand_batch | host | 1.781 (4.8 %) | 1.784 (0.8 %) | +0.2 % | within |
| E4a 300 | hand_padded | host | 2.694 (0.6 %) | 2.741 (4.2 %) | +1.7 % | within |
| E4a 300 | fkl_fused_after | host | 12.32 (1.3 %) | 12.37 (1.2 %) | +0.4 % | within |
| E4a 300 | cvgs_fused | host | 14.37 (0.8 %) | 13.64 (5.5 %) | -5.1 % | within FLAG |
| E4c 720 | fkl_fused | host | 1.61 (0.9 %) | 1.591 (2.8 %) | -1.2 % | within |
| E4c 720 | fkl_fused_half_pixel | host | 1.609 (2.2 %) | 1.619 (1.1 %) | +0.6 % | within |
| E4c 720 | opencv_percrop | host | 12.52 (2.7 %) | 12.43 (3.5 %) | -0.7 % | within |
| E4c 720 | npp | host | 14.44 (0.3 %) | 14.38 (0.4 %) | -0.4 % | exceeds |
| E4c 1920 | fkl_fused | host | 1.608 (1.5 %) | 1.652 (2.0 %) | +2.8 % | exceeds |
| E4c 1920 | fkl_fused_half_pixel | host | 1.6 (6.1 %) | 1.6 (1.5 %) | +0.0 % | within |
| E4c 1920 | opencv_percrop | host | 12.75 (3.0 %) | 12.54 (0.6 %) | -1.6 % | within |
| E4c 1920 | npp | host | 14.44 (1.4 %) | 14.4 (0.7 %) | -0.3 % | within |
| E4c 3840 | fkl_fused | host | 1.635 (4.0 %) | 1.663 (2.3 %) | +1.7 % | within |
| E4c 3840 | fkl_fused_half_pixel | host | 1.696 (4.3 %) | 1.674 (5.6 %) | -1.3 % | within |
| E4c 3840 | opencv_percrop | host | 13.03 (2.4 %) | 12.95 (2.3 %) | -0.6 % | within |
| E4c 3840 | npp | host | 14.76 (1.6 %) | 14.71 (1.2 %) | -0.3 % | within |

### host_launch_us

| point | version | bound | old (spread) | new (spread) | change | |
|---|---|---|---|---|---|---|
| E4a 10 | fkl_fused | host | 2.098 (2.8 %) | 2.066 (1.8 %) | -1.5 % | within |
| E4a 10 | fkl_fused_half_pixel | host | 2.099 (4.5 %) | 2.081 (3.5 %) | -0.9 % | within |
| E4a 10 | opencv_percrop | host | 126.3 (1.3 %) | 126.3 (0.0 %) | +0.0 % | within |
| E4a 10 | npp | host | 129.5 (1.9 %) | 129.8 (0.2 %) | +0.3 % | within |
| E4a 10 | fkl_fused_late | host | 2.066 (1.8 %) | 2.032 (2.7 %) | -1.7 % | within |
| E4a 10 | hand_batch | host | 1.644 (1.7 %) | 1.612 (3.9 %) | -2.0 % | within |
| E4a 10 | hand_padded | host | 1.636 (3.1 %) | 1.621 (1.1 %) | -0.9 % | within |
| E4a 10 | fkl_fused_after | host | 2.067 (0.7 %) | 2.057 (2.4 %) | -0.5 % | within |
| E4a 10 | cvgs_fused | host | 2.217 (4.5 %) | 2.103 (2.3 %) | -5.1 % | exceeds FLAG |
| E4a 50 | fkl_fused | host | 3.567 (4.2 %) | 3.442 (1.0 %) | -3.5 % | within |
| E4a 50 | fkl_fused_half_pixel | host | 3.552 (1.3 %) | 3.444 (3.5 %) | -3.1 % | within |
| E4a 50 | opencv_percrop | host | 635.9 (1.6 %) | 632.3 (0.8 %) | -0.6 % | within |
| E4a 50 | npp | host | 641.4 (1.2 %) | 638.2 (1.0 %) | -0.5 % | within |
| E4a 50 | fkl_fused_late | host | 3.475 (7.0 %) | 3.429 (3.2 %) | -1.3 % | within |
| E4a 50 | hand_batch | host | 1.627 (2.9 %) | 1.622 (0.8 %) | -0.4 % | within |
| E4a 50 | hand_padded | host | 1.729 (0.7 %) | 1.74 (4.4 %) | +0.6 % | within |
| E4a 50 | fkl_fused_after | host | 3.423 (6.1 %) | 3.432 (1.3 %) | +0.3 % | within |
| E4a 50 | cvgs_fused | host | 3.663 (4.0 %) | 3.795 (4.1 %) | +3.6 % | within |
| E4a 100 | fkl_fused | host | 5.246 (19.2 %) | 5.138 (2.7 %) | -2.1 % | within |
| E4a 100 | fkl_fused_half_pixel | host | 5.215 (1.0 %) | 5.125 (6.6 %) | -1.7 % | within |
| E4a 100 | opencv_percrop | host | 1,268 (1.1 %) | 1,264 (1.8 %) | -0.3 % | within |
| E4a 100 | npp | host | 1,297 (3.3 %) | 1,275 (0.9 %) | -1.7 % | within |
| E4a 100 | fkl_fused_late | host | 5.182 (1.2 %) | 5.107 (2.0 %) | -1.4 % | within |
| E4a 100 | hand_batch | host | 1.649 (2.0 %) | 1.626 (1.8 %) | -1.4 % | within |
| E4a 100 | hand_padded | host | 1.911 (5.4 %) | 1.917 (3.2 %) | +0.3 % | within |
| E4a 100 | fkl_fused_after | host | 5.19 (0.5 %) | 5.158 (1.7 %) | -0.6 % | within |
| E4a 100 | cvgs_fused | host | 5.9 (1.6 %) | 5.658 (5.3 %) | -4.1 % | within |
| E4a 150 | fkl_fused | host | 7.156 (0.1 %) | 7.119 (1.6 %) | -0.5 % | within |
| E4a 150 | fkl_fused_half_pixel | host | 7.155 (1.0 %) | 7.209 (3.3 %) | +0.8 % | within |
| E4a 150 | opencv_percrop | host | 1,902 (1.3 %) | 1,888 (0.8 %) | -0.7 % | within |
| E4a 150 | npp | host | 1,922 (1.0 %) | 1,913 (1.1 %) | -0.5 % | within |
| E4a 150 | fkl_fused_late | host | 7.143 (1.4 %) | 7.17 (0.7 %) | +0.4 % | within |
| E4a 150 | hand_batch | host | 1.661 (0.7 %) | 1.671 (2.6 %) | +0.6 % | within |
| E4a 150 | hand_padded | host | 2.115 (1.7 %) | 2.165 (2.9 %) | +2.4 % | within |
| E4a 150 | fkl_fused_after | host | 7.13 (0.5 %) | 7.164 (0.8 %) | +0.5 % | within |
| E4a 150 | cvgs_fused | host | 7.662 (0.5 %) | 7.967 (4.6 %) | +4.0 % | within |
| E4a 200 | fkl_fused | host | 8.703 (1.6 %) | 8.624 (1.3 %) | -0.9 % | within |
| E4a 200 | fkl_fused_half_pixel | host | 8.704 (2.1 %) | 8.631 (1.0 %) | -0.8 % | within |
| E4a 200 | opencv_percrop | host | 2,533 (1.0 %) | 2,529 (1.5 %) | -0.1 % | within |
| E4a 200 | npp | host | 2,575 (1.0 %) | 2,563 (1.2 %) | -0.5 % | within |
| E4a 200 | fkl_fused_late | host | 8.704 (2.1 %) | 8.617 (1.2 %) | -1.0 % | within |
| E4a 200 | hand_batch | host | 1.687 (2.8 %) | 1.668 (0.5 %) | -1.1 % | within |
| E4a 200 | hand_padded | host | 2.273 (2.3 %) | 2.335 (0.7 %) | +2.7 % | exceeds |
| E4a 200 | fkl_fused_after | host | 8.771 (3.1 %) | 8.618 (1.3 %) | -1.7 % | within |
| E4a 200 | cvgs_fused | host | 10.07 (1.5 %) | 9.593 (6.8 %) | -4.7 % | within |
| E4a 250 | fkl_fused | host | 10.35 (4.2 %) | 10.47 (1.5 %) | +1.2 % | within |
| E4a 250 | fkl_fused_half_pixel | host | 10.28 (3.0 %) | 10.42 (1.6 %) | +1.4 % | within |
| E4a 250 | opencv_percrop | host | 3,169 (3.1 %) | 3,153 (0.6 %) | -0.5 % | within |
| E4a 250 | npp | host | 3,217 (1.4 %) | 3,206 (1.2 %) | -0.4 % | within |
| E4a 250 | fkl_fused_late | host | 10.28 (3.5 %) | 10.58 (2.5 %) | +2.9 % | within |
| E4a 250 | hand_batch | host | 1.721 (0.8 %) | 1.708 (0.6 %) | -0.7 % | within |
| E4a 250 | hand_padded | host | 2.483 (0.2 %) | 2.53 (10.0 %) | +1.9 % | within |
| E4a 250 | fkl_fused_after | host | 10.42 (3.5 %) | 10.45 (1.4 %) | +0.3 % | within |
| E4a 250 | cvgs_fused | host | 11.55 (1.3 %) | 12.26 (6.4 %) | +6.1 % | within FLAG |
| E4a 300 | fkl_fused | host | 12.33 (0.5 %) | 12.38 (1.2 %) | +0.3 % | within |
| E4a 300 | fkl_fused_half_pixel | host | 12.3 (0.8 %) | 12.34 (0.8 %) | +0.3 % | within |
| E4a 300 | opencv_percrop | host | 3,805 (0.6 %) | 3,783 (0.2 %) | -0.6 % | within |
| E4a 300 | npp | host | 3,855 (0.9 %) | 3,847 (0.9 %) | -0.2 % | within |
| E4a 300 | fkl_fused_late | host | 12.34 (0.9 %) | 12.33 (1.8 %) | -0.1 % | within |
| E4a 300 | hand_batch | host | 1.756 (2.5 %) | 1.761 (2.1 %) | +0.3 % | within |
| E4a 300 | hand_padded | host | 2.653 (0.8 %) | 2.689 (4.0 %) | +1.4 % | within |
| E4a 300 | fkl_fused_after | host | 12.35 (1.0 %) | 12.4 (1.3 %) | +0.4 % | within |
| E4a 300 | cvgs_fused | host | 14.45 (1.1 %) | 13.64 (6.2 %) | -5.6 % | within FLAG |
| E4c 720 | fkl_fused | host | 1.608 (0.4 %) | 1.626 (2.9 %) | +1.1 % | within |
| E4c 720 | fkl_fused_half_pixel | host | 1.642 (1.1 %) | 1.606 (1.9 %) | -2.2 % | exceeds |
| E4c 720 | opencv_percrop | host | 12.41 (0.8 %) | 12.42 (1.4 %) | +0.1 % | within |
| E4c 720 | npp | host | 14.3 (1.4 %) | 14.35 (1.7 %) | +0.4 % | within |
| E4c 1920 | fkl_fused | host | 1.593 (0.4 %) | 1.619 (1.7 %) | +1.7 % | within |
| E4c 1920 | fkl_fused_half_pixel | host | 1.593 (4.5 %) | 1.624 (5.0 %) | +1.9 % | within |
| E4c 1920 | opencv_percrop | host | 12.32 (1.2 %) | 12.26 (0.4 %) | -0.5 % | within |
| E4c 1920 | npp | host | 14.26 (0.4 %) | 14.27 (1.1 %) | +0.1 % | within |
| E4c 3840 | fkl_fused | host | 1.621 (1.6 %) | 1.583 (1.9 %) | -2.4 % | exceeds |
| E4c 3840 | fkl_fused_half_pixel | host | 1.585 (4.0 %) | 1.633 (1.1 %) | +3.0 % | within |
| E4c 3840 | opencv_percrop | host | 12.26 (1.1 %) | 12.25 (0.8 %) | -0.1 % | within |
| E4c 3840 | npp | host | 14.26 (2.5 %) | 14.26 (0.5 %) | -0.0 % | within |

Flagged over all 30 batches and 3 frames (|change| above the flag and above both spreads): 11

- gpu_mean_ms E4a 140 cvgs_fused (GPU): -3.2 % (old 0.01584, spread 0.9 %; new 0.01533, spread 2.9 %)
- gpu_mean_ms E4a 220 cvgs_fused (GPU): -3.5 % (old 0.02281, spread 3.1 %; new 0.02202, spread 3.5 %)
- gpu_mean_ms E4a 290 cvgs_fused (GPU): +3.1 % (old 0.02784, spread 0.1 %; new 0.0287, spread 3.0 %)
- host_enqueue_us E4a 170 cvgs_fused (host): +5.4 % (old 8.405, spread 0.4 %; new 8.86, spread 5.3 %)
- host_enqueue_us E4a 210 cvgs_fused (host): +6.5 % (old 9.895, spread 0.9 %; new 10.53, spread 5.7 %)
- host_enqueue_us E4a 270 cvgs_fused (host): +6.5 % (old 12.38, spread 0.5 %; new 13.18, spread 6.1 %)
- host_launch_us E4a 10 cvgs_fused (host): -5.1 % (old 2.217, spread 4.5 %; new 2.103, spread 2.3 %)
- host_launch_us E4a 130 cvgs_fused (host): +5.1 % (old 6.798, spread 0.5 %; new 7.145, spread 4.7 %)
- host_launch_us E4a 210 cvgs_fused (host): +6.2 % (old 9.892, spread 0.5 %; new 10.51, spread 5.6 %)
- host_launch_us E4a 230 cvgs_fused (host): +5.4 % (old 10.71, spread 1.3 %; new 11.29, spread 4.9 %)
- host_launch_us E4a 290 cvgs_fused (host): +6.0 % (old 13.15, spread 0.7 %; new 13.95, spread 6.0 %)

## Agreement

- ok_own (against the scalar reference of the version's own convention, 1e-4): 936 of 936 rows pass
- pairs (the outputs of the checked calls; 30 batches x 3 runs per pair):
  - cvgs_fused vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - cvgs_fused vs opencv_percrop: bitwise equal in 0 of 90, largest |difference| 7.63e-06, within 1e-4 in 90 of 90
  - fkl_fused_after vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - fkl_fused_late vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - hand_batch vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - hand_padded vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90
  - hand_roi vs fkl_fused: bitwise equal in 90 of 90, largest |difference| 0, within 1e-4 in 90 of 90

## The runs and the SM clock

Sampler: 1559 samples of nvidia-smi every 100 ms (gpu_clocks.csv).

| run | start | end | SM clock median MHz | min MHz | samples | samples below 2,500 MHz |
|---|---|---|---|---|---|---|
| warmup | 12:40:20 | 12:40:24 | 2,700 | 210 | 36 | 6 |
| quiet-run1 | 12:40:24 | 12:41:14 | 2,685 | 2,685 | 500 | 0 |
| quiet-run2 | 12:41:14 | 12:42:05 | 2,685 | 2,685 | 509 | 0 |
| quiet-run3 | 12:42:05 | 12:42:56 | 2,700 | 2,685 | 510 | 0 |

