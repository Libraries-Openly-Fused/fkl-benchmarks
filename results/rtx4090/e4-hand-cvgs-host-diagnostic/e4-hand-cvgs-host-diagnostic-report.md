# Diagnostic window F2 (e4-hand-cvgs-host-diagnostic), RTX 4090: FKL's host time, binary or drift (report)

Written by `e4_host_diagnostic_report.py` from the result folders only, 2026-11-08. old = 836a98e (window B's binary), new = 2dac1c8 (the binary of the window e4-hand-cvgs), alternating in this window; E4a, fkl_fused and fkl_fused_half_pixel only; median of three runs.

## Findings

**Verdict: binary composition, not drift.** FKL's higher host time in the window e4-hand-cvgs belongs to its binary (2dac1c8). Window B's binary (836a98e), run in the same window, alternating, still has window B's host times.

1. **New against old, in this window** (medians of the per-batch changes, batches 50 to 300):
   - fkl_fused: host_enqueue_us +4.9 % (-0.6 to +10.3), host_launch_us +5.0 % (+0.0 to +9.8);
   - fkl_fused_half_pixel: +4.6 % and +5.3 %.
   - This is the size of the change that the window e4-hand-cvgs showed against window B (+5.0 % and +5.4 %).
   - gpu_mean_ms follows with +2.0 % for both versions, because the GPU waits for FKL's host work inside the interval.
2. **No drift.**
   - The old binary today against window B (2026-11-07): host_enqueue_us -1.4 %, host_launch_us -1.2 % (fkl_fused; half pixel -1.1 % and -1.5 %).
   - The new binary today against the window e4-hand-cvgs: -1.3 % and -1.7 % (half pixel -1.1 % and -1.1 %).
   - Each binary keeps its own host time from one window to the next.
3. **The device code is the same in both binaries** (SASS identical, the step 0 of e4-hand-cvgs). So is the source of the FKL call: the launch lambdas are unchanged since 836a98e.
   - What differs is the compiled host code around the call. At batch 300, `fklb::timeIt` of a launch is a function of its own in 836a98e (3,126 bytes) and is inlined in 2dac1c8, and the `measure<...>` functions grow from 5,646 to 6,317 bytes.
   - Part of that growth is #34's own change to `measure()`, which now returns the checked output. The translation unit also holds cvGPUSpeedup and 90 more kernels. Which of the two changes FKL's host path is not separated here; either way it is the binary, not the machine or the day.
4. **The GPU-time drift columns** (old against window B +3.0 %, new against e4-hand-cvgs +3.1 %) are this window's lower SM clock, not the binaries.
   - With only the two FKL versions, a run is mostly tiny kernels with gaps, and the GPU ran at 2,550 MHz in five of the six runs; the full sweeps ran at 2,685 to 2,700 MHz.
   - Both binaries ran under the same conditions here, so the new-against-old comparison is not affected.
5. **For the window e4-hand-cvgs:** its fkl_fused and fkl_fused_half_pixel calls, and so its hand / FKL ratios, carry this host-code effect. With 836a98e's host code, FKL's call takes about 5 % less host time and about 2 % less time in the interval. hand_batch was compiled in the same binary as the FKL versions it is compared with.

## Summary over batches 50 to 300 (the median of the per-batch changes, with the smallest and the largest)

| version | metric | new against old, this window | old this window against window B | new this window against e4-hand-cvgs |
|---|---|---|---|---|
| fkl_fused | host_enqueue_us | +4.9 % (-0.6 to +10.3) | -1.4 % (-2.7 to +0.5) | -1.3 % (-3.6 to +3.9) |
| fkl_fused | host_launch_us | +5.0 % (+0.0 to +9.8) | -1.2 % (-3.4 to +0.3) | -1.7 % (-3.9 to +3.8) |
| fkl_fused | gpu_mean_ms | +2.0 % (+0.6 to +3.4) | +3.0 % (+1.7 to +4.0) | +3.1 % (+2.5 to +3.8) |
| fkl_fused_half_pixel | host_enqueue_us | +4.6 % (+1.5 to +7.5) | -1.1 % (-2.6 to +1.0) | -1.1 % (-3.6 to +0.5) |
| fkl_fused_half_pixel | host_launch_us | +5.3 % (-0.2 to +8.4) | -1.5 % (-2.9 to +1.1) | -1.1 % (-5.3 to +1.9) |
| fkl_fused_half_pixel | gpu_mean_ms | +2.0 % (+0.4 to +8.6) | +2.8 % (+1.1 to +4.8) | +2.7 % (+1.9 to +9.9) |

## host_enqueue_us, every batch (standing form: new against old in this window; then the drift columns)

| batch | version | bound | old (spread) | new (spread) | change | | old against window B | new against e4-hand-cvgs |
|---|---|---|---|---|---|---|---|---|
| 10 | fkl_fused | host | 2.017 (3.0 %) | 2.052 (0.8 %) | +1.8 % | within | -1.2 % | -1.5 % |
| 10 | fkl_fused_half_pixel | host | 2.042 (0.7 %) | 2.049 (1.8 %) | +0.3 % | within | +1.0 % | -2.4 % |
| 20 | fkl_fused | host | 2.418 (1.3 %) | 2.442 (0.7 %) | +1.0 % | within | +1.4 % | +1.2 % |
| 20 | fkl_fused_half_pixel | host | 2.438 (0.6 %) | 2.453 (0.8 %) | +0.6 % | within | +0.8 % | +1.2 % |
| 30 | fkl_fused | host | 2.729 (5.4 %) | 2.773 (2.5 %) | +1.6 % | within | +1.7 % | +0.6 % |
| 30 | fkl_fused_half_pixel | host | 2.715 (6.8 %) | 2.737 (3.1 %) | +0.8 % | within | +0.5 % | -0.5 % |
| 40 | fkl_fused | host | 3.046 (4.9 %) | 3.057 (2.0 %) | +0.4 % | within | -0.1 % | -0.5 % |
| 40 | fkl_fused_half_pixel | host | 3.052 (1.5 %) | 3.035 (1.3 %) | -0.6 % | within | -0.2 % | -2.6 % |
| 50 | fkl_fused | host | 3.327 (2.1 %) | 3.415 (1.5 %) | +2.6 % | exceeds | -0.5 % | -2.9 % |
| 50 | fkl_fused_half_pixel | host | 3.311 (2.6 %) | 3.394 (1.5 %) | +2.5 % | within | -0.2 % | -3.6 % |
| 60 | fkl_fused | host | 3.544 (2.9 %) | 3.712 (5.0 %) | +4.7 % | within | -2.7 % | -0.8 % |
| 60 | fkl_fused_half_pixel | host | 3.647 (3.2 %) | 3.752 (2.8 %) | +2.9 % | within | +0.6 % | -0.7 % |
| 70 | fkl_fused | host | 4.082 (3.7 %) | 4.105 (3.5 %) | +0.6 % | within | -1.0 % | -0.6 % |
| 70 | fkl_fused_half_pixel | host | 4.056 (3.1 %) | 4.119 (1.2 %) | +1.5 % | within | -1.6 % | -0.2 % |
| 80 | fkl_fused | host | 4.406 (4.0 %) | 4.378 (5.1 %) | -0.6 % | within | +0.5 % | -2.3 % |
| 80 | fkl_fused_half_pixel | host | 4.379 (5.8 %) | 4.484 (2.4 %) | +2.4 % | within | -0.1 % | +0.5 % |
| 90 | fkl_fused | host | 4.565 (2.0 %) | 4.891 (3.1 %) | +7.1 % | exceeds FLAG | -1.4 % | -0.5 % |
| 90 | fkl_fused_half_pixel | host | 4.6 (2.1 %) | 4.878 (1.0 %) | +6.0 % | exceeds FLAG | -0.4 % | -0.9 % |
| 100 | fkl_fused | host | 4.94 (2.5 %) | 5.015 (1.8 %) | +1.5 % | within | -0.9 % | -3.6 % |
| 100 | fkl_fused_half_pixel | host | 4.88 (0.9 %) | 5.103 (1.8 %) | +4.6 % | exceeds | -1.5 % | -2.1 % |
| 110 | fkl_fused | host | 5.373 (0.9 %) | 5.545 (0.4 %) | +3.2 % | exceeds | -0.9 % | -1.3 % |
| 110 | fkl_fused_half_pixel | host | 5.403 (2.3 %) | 5.594 (3.8 %) | +3.5 % | within | -1.3 % | -0.7 % |
| 120 | fkl_fused | host | 5.6 (4.5 %) | 5.774 (0.9 %) | +3.1 % | within | -0.6 % | -2.9 % |
| 120 | fkl_fused_half_pixel | host | 5.629 (2.5 %) | 5.836 (1.5 %) | +3.7 % | exceeds | -0.5 % | -1.7 % |
| 130 | fkl_fused | host | 6.195 (3.3 %) | 6.415 (0.4 %) | +3.5 % | exceeds | +0.3 % | -1.2 % |
| 130 | fkl_fused_half_pixel | host | 6.188 (1.3 %) | 6.407 (2.1 %) | +3.5 % | exceeds | -0.6 % | -1.9 % |
| 140 | fkl_fused | host | 6.187 (3.9 %) | 6.577 (10.5 %) | +6.3 % | within FLAG | -1.8 % | -0.0 % |
| 140 | fkl_fused_half_pixel | host | 6.208 (0.8 %) | 6.539 (10.7 %) | +5.3 % | within FLAG | -1.7 % | -0.6 % |
| 150 | fkl_fused | host | 6.814 (2.7 %) | 7.15 (9.5 %) | +4.9 % | within | -1.6 % | -0.0 % |
| 150 | fkl_fused_half_pixel | host | 6.791 (0.4 %) | 7.043 (10.0 %) | +3.7 % | within | -1.3 % | -1.5 % |
| 160 | fkl_fused | host | 6.853 (4.6 %) | 7.285 (0.6 %) | +6.3 % | exceeds FLAG | -1.6 % | +1.0 % |
| 160 | fkl_fused_half_pixel | host | 6.92 (1.8 %) | 7.275 (1.8 %) | +5.1 % | exceeds FLAG | -1.8 % | -0.1 % |
| 170 | fkl_fused | host | 7.513 (3.4 %) | 7.794 (0.9 %) | +3.7 % | exceeds | -0.1 % | -1.5 % |
| 170 | fkl_fused_half_pixel | host | 7.615 (2.8 %) | 7.813 (2.6 %) | +2.6 % | within | +1.0 % | -1.3 % |
| 180 | fkl_fused | host | 7.477 (2.9 %) | 7.795 (2.0 %) | +4.3 % | exceeds | -0.9 % | -1.4 % |
| 180 | fkl_fused_half_pixel | host | 7.429 (2.6 %) | 7.805 (4.7 %) | +5.1 % | exceeds FLAG | -1.1 % | -1.5 % |
| 190 | fkl_fused | host | 7.744 (1.6 %) | 8.126 (0.7 %) | +4.9 % | exceeds | -1.2 % | -1.9 % |
| 190 | fkl_fused_half_pixel | host | 7.787 (1.2 %) | 8.111 (1.7 %) | +4.2 % | exceeds | -0.8 % | -2.2 % |
| 200 | fkl_fused | host | 8.068 (1.3 %) | 8.508 (0.8 %) | +5.5 % | exceeds FLAG | -2.4 % | -2.7 % |
| 200 | fkl_fused_half_pixel | host | 8.194 (1.0 %) | 8.538 (0.8 %) | +4.2 % | exceeds | -0.2 % | -2.0 % |
| 210 | fkl_fused | host | 8.475 (2.3 %) | 8.945 (0.9 %) | +5.5 % | exceeds FLAG | -1.3 % | -1.8 % |
| 210 | fkl_fused_half_pixel | host | 8.449 (1.4 %) | 8.886 (1.3 %) | +5.2 % | exceeds FLAG | -1.4 % | -1.8 % |
| 220 | fkl_fused | host | 8.765 (0.8 %) | 9.469 (1.7 %) | +8.0 % | exceeds FLAG | -1.2 % | +1.0 % |
| 220 | fkl_fused_half_pixel | host | 8.88 (18.5 %) | 9.291 (0.7 %) | +4.6 % | within | +0.3 % | -0.5 % |
| 230 | fkl_fused | host | 9.193 (1.5 %) | 9.695 (1.6 %) | +5.5 % | exceeds FLAG | +0.0 % | +0.4 % |
| 230 | fkl_fused_half_pixel | host | 9.099 (2.2 %) | 9.696 (1.6 %) | +6.6 % | exceeds FLAG | -1.1 % | +0.3 % |
| 240 | fkl_fused | host | 9.352 (2.2 %) | 10.31 (11.2 %) | +10.3 % | within FLAG | -2.0 % | +3.9 % |
| 240 | fkl_fused_half_pixel | host | 9.374 (4.1 %) | 10.08 (12.2 %) | +7.5 % | within FLAG | -2.0 % | +0.5 % |
| 250 | fkl_fused | host | 9.739 (1.2 %) | 10.34 (1.4 %) | +6.2 % | exceeds FLAG | -1.9 % | -1.0 % |
| 250 | fkl_fused_half_pixel | host | 9.739 (1.0 %) | 10.42 (1.6 %) | +7.0 % | exceeds FLAG | -2.1 % | +0.5 % |
| 260 | fkl_fused | host | 10.16 (1.2 %) | 10.67 (0.5 %) | +5.0 % | exceeds FLAG | -2.0 % | -1.5 % |
| 260 | fkl_fused_half_pixel | host | 10.19 (0.9 %) | 10.74 (2.1 %) | +5.4 % | exceeds FLAG | -1.4 % | -1.0 % |
| 270 | fkl_fused | host | 10.46 (0.9 %) | 10.96 (0.6 %) | +4.7 % | exceeds | -2.3 % | -2.2 % |
| 270 | fkl_fused_half_pixel | host | 10.49 (1.2 %) | 11.04 (1.1 %) | +5.2 % | exceeds FLAG | -0.7 % | -1.2 % |
| 280 | fkl_fused | host | 10.81 (2.1 %) | 11.42 (1.4 %) | +5.6 % | exceeds FLAG | -1.6 % | -2.3 % |
| 280 | fkl_fused_half_pixel | host | 10.86 (0.5 %) | 11.42 (0.9 %) | +5.1 % | exceeds FLAG | -0.6 % | -2.3 % |
| 290 | fkl_fused | host | 11.04 (2.7 %) | 11.76 (2.3 %) | +6.5 % | exceeds FLAG | -2.1 % | -2.2 % |
| 290 | fkl_fused_half_pixel | host | 11.14 (2.0 %) | 11.77 (1.9 %) | +5.6 % | exceeds FLAG | -1.7 % | -3.0 % |
| 300 | fkl_fused | host | 11.93 (3.7 %) | 12.26 (0.9 %) | +2.8 % | within | -2.2 % | -0.6 % |
| 300 | fkl_fused_half_pixel | host | 11.92 (2.5 %) | 12.24 (3.5 %) | +2.7 % | within | -2.6 % | -0.6 % |

## host_launch_us, every batch (standing form: new against old in this window; then the drift columns)

| batch | version | bound | old (spread) | new (spread) | change | | old against window B | new against e4-hand-cvgs |
|---|---|---|---|---|---|---|---|---|
| 10 | fkl_fused | host | 2.035 (2.6 %) | 2.053 (0.9 %) | +0.9 % | within | -0.5 % | -2.2 % |
| 10 | fkl_fused_half_pixel | host | 2.037 (3.8 %) | 2.099 (2.3 %) | +3.0 % | within | -2.6 % | +0.0 % |
| 20 | fkl_fused | host | 2.452 (1.7 %) | 2.447 (1.7 %) | -0.2 % | within | +2.0 % | +0.3 % |
| 20 | fkl_fused_half_pixel | host | 2.45 (3.1 %) | 2.522 (2.3 %) | +2.9 % | within | -0.1 % | +3.2 % |
| 30 | fkl_fused | host | 2.604 (4.2 %) | 2.759 (1.9 %) | +6.0 % | exceeds FLAG | -3.8 % | -0.1 % |
| 30 | fkl_fused_half_pixel | host | 2.701 (5.1 %) | 2.718 (10.3 %) | +0.6 % | within | -0.9 % | -1.5 % |
| 40 | fkl_fused | host | 3.044 (12.2 %) | 3.025 (1.3 %) | -0.6 % | within | -0.7 % | -2.7 % |
| 40 | fkl_fused_half_pixel | host | 3.061 (2.4 %) | 3.023 (1.0 %) | -1.2 % | within | -1.1 % | -2.1 % |
| 50 | fkl_fused | host | 3.297 (1.9 %) | 3.43 (3.9 %) | +4.0 % | exceeds | -1.1 % | -3.9 % |
| 50 | fkl_fused_half_pixel | host | 3.31 (1.0 %) | 3.366 (1.5 %) | +1.7 % | exceeds | -1.7 % | -5.3 % |
| 60 | fkl_fused | host | 3.568 (0.9 %) | 3.735 (1.5 %) | +4.7 % | exceeds | -2.3 % | -2.3 % |
| 60 | fkl_fused_half_pixel | host | 3.611 (7.0 %) | 3.844 (5.6 %) | +6.5 % | within FLAG | -1.6 % | +1.6 % |
| 70 | fkl_fused | host | 4.012 (3.8 %) | 4.095 (4.2 %) | +2.1 % | within | -3.4 % | -1.0 % |
| 70 | fkl_fused_half_pixel | host | 4.006 (5.3 %) | 4.15 (4.9 %) | +3.6 % | within | -2.5 % | +0.0 % |
| 80 | fkl_fused | host | 4.391 (0.6 %) | 4.392 (5.3 %) | +0.0 % | within | -0.2 % | -0.8 % |
| 80 | fkl_fused_half_pixel | host | 4.437 (5.4 %) | 4.425 (3.2 %) | -0.2 % | within | +1.1 % | -0.2 % |
| 90 | fkl_fused | host | 4.561 (0.6 %) | 4.893 (4.8 %) | +7.3 % | exceeds FLAG | -1.3 % | -0.9 % |
| 90 | fkl_fused_half_pixel | host | 4.599 (2.5 %) | 4.852 (0.5 %) | +5.5 % | exceeds FLAG | -0.7 % | -1.9 % |
| 100 | fkl_fused | host | 4.908 (3.6 %) | 5.057 (0.2 %) | +3.0 % | within | -1.0 % | -3.6 % |
| 100 | fkl_fused_half_pixel | host | 4.853 (0.3 %) | 5.04 (0.8 %) | +3.9 % | exceeds | -2.5 % | -3.4 % |
| 110 | fkl_fused | host | 5.401 (3.0 %) | 5.552 (0.4 %) | +2.8 % | within | -1.7 % | -2.4 % |
| 110 | fkl_fused_half_pixel | host | 5.368 (2.9 %) | 5.609 (2.1 %) | +4.5 % | exceeds | -1.4 % | -0.9 % |
| 120 | fkl_fused | host | 5.577 (3.4 %) | 5.808 (0.4 %) | +4.1 % | exceeds | -1.2 % | -2.1 % |
| 120 | fkl_fused_half_pixel | host | 5.563 (2.7 %) | 5.787 (2.5 %) | +4.0 % | exceeds | -1.6 % | -3.7 % |
| 130 | fkl_fused | host | 6.171 (1.1 %) | 6.415 (0.5 %) | +4.0 % | exceeds | -0.0 % | -2.1 % |
| 130 | fkl_fused_half_pixel | host | 6.131 (0.4 %) | 6.436 (1.4 %) | +5.0 % | exceeds | -1.0 % | -1.4 % |
| 140 | fkl_fused | host | 6.23 (3.1 %) | 6.594 (10.7 %) | +5.8 % | within FLAG | -0.9 % | -0.2 % |
| 140 | fkl_fused_half_pixel | host | 6.225 (1.7 %) | 6.572 (10.5 %) | +5.6 % | within FLAG | -0.5 % | -1.0 % |
| 150 | fkl_fused | host | 6.725 (2.7 %) | 7.122 (10.3 %) | +5.9 % | within FLAG | -2.9 % | -0.5 % |
| 150 | fkl_fused_half_pixel | host | 6.726 (1.8 %) | 7.288 (10.3 %) | +8.4 % | within FLAG | -2.9 % | +1.9 % |
| 160 | fkl_fused | host | 6.924 (4.7 %) | 7.272 (1.6 %) | +5.0 % | exceeds FLAG | -0.9 % | +0.1 % |
| 160 | fkl_fused_half_pixel | host | 6.861 (0.9 %) | 7.28 (1.1 %) | +6.1 % | exceeds FLAG | -2.4 % | +1.2 % |
| 170 | fkl_fused | host | 7.598 (2.8 %) | 7.804 (1.0 %) | +2.7 % | within | +0.3 % | -1.8 % |
| 170 | fkl_fused_half_pixel | host | 7.592 (3.4 %) | 7.898 (1.3 %) | +4.0 % | exceeds | +1.0 % | +0.4 % |
| 180 | fkl_fused | host | 7.463 (0.4 %) | 7.769 (1.5 %) | +4.1 % | exceeds | -1.0 % | -2.5 % |
| 180 | fkl_fused_half_pixel | host | 7.46 (4.1 %) | 7.84 (2.8 %) | +5.1 % | exceeds FLAG | -1.5 % | -1.4 % |
| 190 | fkl_fused | host | 7.785 (1.1 %) | 8.136 (1.2 %) | +4.5 % | exceeds | -0.3 % | -2.1 % |
| 190 | fkl_fused_half_pixel | host | 7.794 (0.5 %) | 8.134 (2.1 %) | +4.4 % | exceeds | -0.6 % | -2.3 % |
| 200 | fkl_fused | host | 8.116 (1.3 %) | 8.571 (1.7 %) | +5.6 % | exceeds FLAG | -1.3 % | -1.5 % |
| 200 | fkl_fused_half_pixel | host | 8.153 (1.5 %) | 8.567 (0.3 %) | +5.1 % | exceeds FLAG | -1.2 % | -1.6 % |
| 210 | fkl_fused | host | 8.385 (1.3 %) | 8.898 (0.3 %) | +6.1 % | exceeds FLAG | -2.2 % | -2.0 % |
| 210 | fkl_fused_half_pixel | host | 8.409 (0.9 %) | 8.909 (1.1 %) | +5.9 % | exceeds FLAG | -1.6 % | -1.9 % |
| 220 | fkl_fused | host | 8.755 (0.5 %) | 9.398 (6.1 %) | +7.3 % | exceeds FLAG | -1.0 % | +0.3 % |
| 220 | fkl_fused_half_pixel | host | 8.814 (2.4 %) | 9.358 (1.1 %) | +6.2 % | exceeds FLAG | -0.7 % | -0.2 % |
| 230 | fkl_fused | host | 9.124 (2.0 %) | 9.694 (1.2 %) | +6.3 % | exceeds FLAG | -0.2 % | -0.2 % |
| 230 | fkl_fused_half_pixel | host | 9.087 (0.8 %) | 9.713 (2.0 %) | +6.9 % | exceeds FLAG | -0.7 % | -0.1 % |
| 240 | fkl_fused | host | 9.402 (2.1 %) | 10.32 (12.1 %) | +9.8 % | within FLAG | -2.3 % | +3.8 % |
| 240 | fkl_fused_half_pixel | host | 9.417 (2.6 %) | 10.07 (12.2 %) | +6.9 % | within FLAG | -2.1 % | +1.3 % |
| 250 | fkl_fused | host | 9.751 (2.8 %) | 10.4 (1.3 %) | +6.7 % | exceeds FLAG | -1.9 % | +0.5 % |
| 250 | fkl_fused_half_pixel | host | 9.722 (0.9 %) | 10.36 (1.3 %) | +6.5 % | exceeds FLAG | -2.1 % | +0.7 % |
| 260 | fkl_fused | host | 10.14 (0.7 %) | 10.67 (0.6 %) | +5.2 % | exceeds FLAG | -2.1 % | -1.4 % |
| 260 | fkl_fused_half_pixel | host | 10.15 (0.8 %) | 10.66 (0.4 %) | +5.0 % | exceeds FLAG | -1.5 % | -1.9 % |
| 270 | fkl_fused | host | 10.45 (0.7 %) | 10.97 (1.3 %) | +5.0 % | exceeds | -0.7 % | -2.8 % |
| 270 | fkl_fused_half_pixel | host | 10.44 (0.6 %) | 11.07 (1.1 %) | +6.0 % | exceeds FLAG | -0.9 % | -1.1 % |
| 280 | fkl_fused | host | 10.83 (0.8 %) | 11.4 (1.3 %) | +5.3 % | exceeds FLAG | -1.9 % | -2.6 % |
| 280 | fkl_fused_half_pixel | host | 10.79 (1.4 %) | 11.46 (2.1 %) | +6.2 % | exceeds FLAG | -1.9 % | -2.3 % |
| 290 | fkl_fused | host | 11.04 (3.4 %) | 11.78 (2.4 %) | +6.7 % | exceeds FLAG | -2.4 % | -2.3 % |
| 290 | fkl_fused_half_pixel | host | 11.1 (3.7 %) | 11.8 (3.5 %) | +6.3 % | exceeds FLAG | -1.7 % | -1.9 % |
| 300 | fkl_fused | host | 11.83 (4.3 %) | 12.3 (1.7 %) | +3.9 % | within | -2.7 % | -0.3 % |
| 300 | fkl_fused_half_pixel | host | 12.14 (2.9 %) | 12.31 (3.5 %) | +1.4 % | within | -0.3 % | +0.1 % |

## gpu_mean_ms, every batch (standing form: new against old in this window; then the drift columns)

| batch | version | bound | old (spread) | new (spread) | change | | old against window B | new against e4-hand-cvgs |
|---|---|---|---|---|---|---|---|---|
| 10 | fkl_fused | GPU | 0.003869 (4.5 %) | 0.003993 (2.4 %) | +3.2 % | within FLAG | +2.3 % | +5.2 % |
| 10 | fkl_fused_half_pixel | GPU | 0.003938 (5.5 %) | 0.003945 (0.8 %) | +0.2 % | within | +3.9 % | +2.8 % |
| 20 | fkl_fused | GPU | 0.005067 (2.7 %) | 0.0051 (2.7 %) | +0.7 % | within | +2.0 % | +4.6 % |
| 20 | fkl_fused_half_pixel | GPU | 0.005088 (5.2 %) | 0.005155 (1.3 %) | +1.3 % | within | +3.1 % | +4.2 % |
| 30 | fkl_fused | GPU | 0.005707 (5.4 %) | 0.005894 (2.4 %) | +3.3 % | within FLAG | +1.7 % | +4.0 % |
| 30 | fkl_fused_half_pixel | GPU | 0.005767 (6.7 %) | 0.005961 (1.8 %) | +3.4 % | within FLAG | +2.6 % | +4.4 % |
| 40 | fkl_fused | GPU | 0.006849 (4.1 %) | 0.00689 (0.5 %) | +0.6 % | within | +4.9 % | +1.5 % |
| 40 | fkl_fused_half_pixel | GPU | 0.006902 (2.9 %) | 0.006969 (0.9 %) | +1.0 % | within | +3.5 % | +3.3 % |
| 50 | fkl_fused | GPU | 0.007664 (3.9 %) | 0.007812 (1.4 %) | +1.9 % | within | +1.8 % | +3.1 % |
| 50 | fkl_fused_half_pixel | GPU | 0.007632 (4.2 %) | 0.007861 (1.0 %) | +3.0 % | within FLAG | +1.5 % | +3.1 % |
| 60 | fkl_fused | GPU | 0.008491 (4.1 %) | 0.00865 (0.1 %) | +1.9 % | within | +2.9 % | +3.2 % |
| 60 | fkl_fused_half_pixel | GPU | 0.008523 (1.2 %) | 0.008624 (1.2 %) | +1.2 % | within | +2.8 % | +2.5 % |
| 70 | fkl_fused | GPU | 0.009366 (5.2 %) | 0.009551 (14.9 %) | +2.0 % | within | +1.7 % | +3.5 % |
| 70 | fkl_fused_half_pixel | GPU | 0.009388 (4.1 %) | 0.009524 (16.1 %) | +1.4 % | within | +1.1 % | +2.5 % |
| 80 | fkl_fused | GPU | 0.01026 (16.3 %) | 0.01033 (0.9 %) | +0.6 % | within | +2.9 % | +3.1 % |
| 80 | fkl_fused_half_pixel | GPU | 0.01031 (27.2 %) | 0.01035 (0.5 %) | +0.4 % | within | +3.2 % | +2.5 % |
| 90 | fkl_fused | GPU | 0.01108 (21.7 %) | 0.01131 (0.4 %) | +2.0 % | within | +4.0 % | +3.0 % |
| 90 | fkl_fused_half_pixel | GPU | 0.01108 (4.7 %) | 0.01131 (1.1 %) | +2.1 % | within | +3.2 % | +2.2 % |
| 100 | fkl_fused | GPU | 0.01178 (4.5 %) | 0.01199 (1.5 %) | +1.7 % | within | +3.4 % | +3.3 % |
| 100 | fkl_fused_half_pixel | GPU | 0.0118 (4.4 %) | 0.01205 (0.8 %) | +2.1 % | within | +3.0 % | +3.5 % |
| 110 | fkl_fused | GPU | 0.01258 (2.5 %) | 0.01294 (0.7 %) | +2.8 % | exceeds | +1.8 % | +2.7 % |
| 110 | fkl_fused_half_pixel | GPU | 0.01284 (10.9 %) | 0.01291 (2.1 %) | +0.6 % | within | +3.0 % | +2.5 % |
| 120 | fkl_fused | GPU | 0.01337 (2.8 %) | 0.01382 (1.9 %) | +3.4 % | exceeds FLAG | +2.4 % | +3.8 % |
| 120 | fkl_fused_half_pixel | GPU | 0.01334 (3.3 %) | 0.01377 (1.0 %) | +3.3 % | within FLAG | +2.0 % | +3.5 % |
| 130 | fkl_fused | GPU | 0.01445 (4.3 %) | 0.01458 (1.3 %) | +1.0 % | within | +4.0 % | +2.8 % |
| 130 | fkl_fused_half_pixel | GPU | 0.01429 (2.9 %) | 0.01459 (1.1 %) | +2.1 % | within | +2.7 % | +2.6 % |
| 140 | fkl_fused | GPU | 0.01488 (3.1 %) | 0.01526 (0.7 %) | +2.6 % | within | +2.9 % | +3.0 % |
| 140 | fkl_fused_half_pixel | GPU | 0.01492 (4.1 %) | 0.0152 (0.3 %) | +1.9 % | within | +2.6 % | +1.9 % |
| 150 | fkl_fused | GPU | 0.01595 (6.6 %) | 0.01615 (0.9 %) | +1.3 % | within | +3.1 % | +2.9 % |
| 150 | fkl_fused_half_pixel | GPU | 0.01598 (3.9 %) | 0.01619 (1.6 %) | +1.4 % | within | +2.8 % | +2.4 % |
| 160 | fkl_fused | GPU | 0.01661 (0.6 %) | 0.01694 (0.7 %) | +2.0 % | exceeds | +2.8 % | +2.5 % |
| 160 | fkl_fused_half_pixel | GPU | 0.01663 (0.5 %) | 0.01694 (0.6 %) | +1.9 % | exceeds | +1.6 % | +2.6 % |
| 170 | fkl_fused | GPU | 0.01777 (4.1 %) | 0.01797 (0.4 %) | +1.1 % | within | +3.8 % | +2.5 % |
| 170 | fkl_fused_half_pixel | GPU | 0.01773 (1.0 %) | 0.01819 (2.3 %) | +2.6 % | exceeds | +3.3 % | +3.9 % |
| 180 | fkl_fused | GPU | 0.01811 (0.2 %) | 0.01862 (0.7 %) | +2.8 % | exceeds | +2.8 % | +3.7 % |
| 180 | fkl_fused_half_pixel | GPU | 0.01826 (1.0 %) | 0.01868 (1.1 %) | +2.3 % | exceeds | +3.6 % | +3.6 % |
| 190 | fkl_fused | GPU | 0.01914 (1.3 %) | 0.01944 (0.7 %) | +1.6 % | exceeds | +3.8 % | +3.8 % |
| 190 | fkl_fused_half_pixel | GPU | 0.01937 (1.0 %) | 0.01947 (2.8 %) | +0.5 % | within | +4.8 % | +3.5 % |
| 200 | fkl_fused | GPU | 0.01978 (0.9 %) | 0.02013 (0.7 %) | +1.8 % | exceeds | +3.0 % | +3.2 % |
| 200 | fkl_fused_half_pixel | GPU | 0.01985 (1.3 %) | 0.02023 (1.1 %) | +1.9 % | exceeds | +3.3 % | +4.1 % |
| 210 | fkl_fused | GPU | 0.0206 (10.5 %) | 0.02107 (0.9 %) | +2.3 % | within | +3.1 % | +3.0 % |
| 210 | fkl_fused_half_pixel | GPU | 0.02061 (1.1 %) | 0.02102 (0.7 %) | +2.0 % | exceeds | +2.8 % | +3.4 % |
| 220 | fkl_fused | GPU | 0.02139 (0.5 %) | 0.02182 (9.7 %) | +2.0 % | within | +3.2 % | +3.5 % |
| 220 | fkl_fused_half_pixel | GPU | 0.02147 (0.3 %) | 0.02173 (1.0 %) | +1.2 % | exceeds | +3.1 % | +2.8 % |
| 230 | fkl_fused | GPU | 0.02208 (0.1 %) | 0.02255 (1.2 %) | +2.1 % | exceeds | +2.9 % | +2.8 % |
| 230 | fkl_fused_half_pixel | GPU | 0.02209 (0.8 %) | 0.02254 (0.8 %) | +2.0 % | exceeds | +2.7 % | +3.1 % |
| 240 | fkl_fused | GPU | 0.02287 (0.3 %) | 0.02348 (0.8 %) | +2.7 % | exceeds | +2.7 % | +2.9 % |
| 240 | fkl_fused_half_pixel | GPU | 0.02306 (1.1 %) | 0.02337 (1.4 %) | +1.4 % | within | +2.9 % | +2.5 % |
| 250 | fkl_fused | GPU | 0.02372 (0.7 %) | 0.02431 (1.0 %) | +2.5 % | exceeds | +2.4 % | +3.2 % |
| 250 | fkl_fused_half_pixel | GPU | 0.02371 (0.8 %) | 0.0242 (0.8 %) | +2.1 % | exceeds | +2.2 % | +2.6 % |
| 260 | fkl_fused | GPU | 0.02476 (1.2 %) | 0.02529 (1.9 %) | +2.2 % | exceeds | +3.9 % | +3.8 % |
| 260 | fkl_fused_half_pixel | GPU | 0.02459 (0.2 %) | 0.0267 (13.8 %) | +8.6 % | within FLAG | +3.0 % | +9.9 % |
| 270 | fkl_fused | GPU | 0.02546 (1.0 %) | 0.026 (0.8 %) | +2.1 % | exceeds | +3.9 % | +3.4 % |
| 270 | fkl_fused_half_pixel | GPU | 0.02538 (14.8 %) | 0.02607 (1.0 %) | +2.7 % | within | +3.5 % | +3.3 % |
| 280 | fkl_fused | GPU | 0.02615 (0.4 %) | 0.0269 (0.5 %) | +2.9 % | exceeds | +2.9 % | +3.8 % |
| 280 | fkl_fused_half_pixel | GPU | 0.02621 (0.2 %) | 0.0269 (0.5 %) | +2.6 % | exceeds | +2.8 % | +2.6 % |
| 290 | fkl_fused | GPU | 0.02703 (0.5 %) | 0.02766 (0.5 %) | +2.3 % | exceeds | +3.0 % | +3.4 % |
| 290 | fkl_fused_half_pixel | GPU | 0.02696 (7.1 %) | 0.0276 (0.7 %) | +2.4 % | within | +2.7 % | +2.7 % |
| 300 | fkl_fused | GPU | 0.02837 (7.9 %) | 0.02854 (1.0 %) | +0.6 % | within | +3.0 % | +2.8 % |
| 300 | fkl_fused_half_pixel | GPU | 0.02815 (1.2 %) | 0.0283 (0.6 %) | +0.5 % | within | +2.1 % | +2.7 % |

Flagged (new against old, above the flag and both spreads): 44

- host_enqueue_us E4a 90 fkl_fused (host): +7.1 %
- host_enqueue_us E4a 90 fkl_fused_half_pixel (host): +6.0 %
- host_enqueue_us E4a 160 fkl_fused (host): +6.3 %
- host_enqueue_us E4a 160 fkl_fused_half_pixel (host): +5.1 %
- host_enqueue_us E4a 180 fkl_fused_half_pixel (host): +5.1 %
- host_enqueue_us E4a 200 fkl_fused (host): +5.5 %
- host_enqueue_us E4a 210 fkl_fused (host): +5.5 %
- host_enqueue_us E4a 210 fkl_fused_half_pixel (host): +5.2 %
- host_enqueue_us E4a 220 fkl_fused (host): +8.0 %
- host_enqueue_us E4a 230 fkl_fused (host): +5.5 %
- host_enqueue_us E4a 230 fkl_fused_half_pixel (host): +6.6 %
- host_enqueue_us E4a 250 fkl_fused (host): +6.2 %
- host_enqueue_us E4a 250 fkl_fused_half_pixel (host): +7.0 %
- host_enqueue_us E4a 260 fkl_fused (host): +5.0 %
- host_enqueue_us E4a 260 fkl_fused_half_pixel (host): +5.4 %
- host_enqueue_us E4a 270 fkl_fused_half_pixel (host): +5.2 %
- host_enqueue_us E4a 280 fkl_fused (host): +5.6 %
- host_enqueue_us E4a 280 fkl_fused_half_pixel (host): +5.1 %
- host_enqueue_us E4a 290 fkl_fused (host): +6.5 %
- host_enqueue_us E4a 290 fkl_fused_half_pixel (host): +5.6 %
- host_launch_us E4a 30 fkl_fused (host): +6.0 %
- host_launch_us E4a 90 fkl_fused (host): +7.3 %
- host_launch_us E4a 90 fkl_fused_half_pixel (host): +5.5 %
- host_launch_us E4a 160 fkl_fused (host): +5.0 %
- host_launch_us E4a 160 fkl_fused_half_pixel (host): +6.1 %
- host_launch_us E4a 180 fkl_fused_half_pixel (host): +5.1 %
- host_launch_us E4a 200 fkl_fused (host): +5.6 %
- host_launch_us E4a 200 fkl_fused_half_pixel (host): +5.1 %
- host_launch_us E4a 210 fkl_fused (host): +6.1 %
- host_launch_us E4a 210 fkl_fused_half_pixel (host): +5.9 %
- host_launch_us E4a 220 fkl_fused (host): +7.3 %
- host_launch_us E4a 220 fkl_fused_half_pixel (host): +6.2 %
- host_launch_us E4a 230 fkl_fused (host): +6.3 %
- host_launch_us E4a 230 fkl_fused_half_pixel (host): +6.9 %
- host_launch_us E4a 250 fkl_fused (host): +6.7 %
- host_launch_us E4a 250 fkl_fused_half_pixel (host): +6.5 %
- host_launch_us E4a 260 fkl_fused (host): +5.2 %
- host_launch_us E4a 260 fkl_fused_half_pixel (host): +5.0 %
- host_launch_us E4a 270 fkl_fused_half_pixel (host): +6.0 %
- host_launch_us E4a 280 fkl_fused (host): +5.3 %
- host_launch_us E4a 280 fkl_fused_half_pixel (host): +6.2 %
- host_launch_us E4a 290 fkl_fused (host): +6.7 %
- host_launch_us E4a 290 fkl_fused_half_pixel (host): +6.3 %
- gpu_mean_ms E4a 120 fkl_fused (GPU): +3.4 %

## The runs and the SM clock

Sampler: 295 samples of nvidia-smi every 100 ms (gpu_clocks.csv).

| run | start | end | SM clock median MHz | min MHz | samples |
|---|---|---|---|---|---|
| warmup | 12:24:55 | 12:24:59 | 2,700 | 210 | 37 |
| binary-836a98e/run1 | 12:24:59 | 12:25:02 | 2,700 | 2,550 | 30 |
| binary-2dac1c8/run1 | 12:25:02 | 12:25:07 | 2,550 | 2,550 | 49 |
| binary-836a98e/run2 | 12:25:12 | 12:25:16 | 2,550 | 2,550 | 40 |
| binary-2dac1c8/run2 | 12:25:07 | 12:25:12 | 2,550 | 2,550 | 50 |
| binary-836a98e/run3 | 12:25:16 | 12:25:19 | 2,550 | 2,550 | 30 |
| binary-2dac1c8/run3 | 12:25:20 | 12:25:24 | 2,550 | 2,550 | 40 |

