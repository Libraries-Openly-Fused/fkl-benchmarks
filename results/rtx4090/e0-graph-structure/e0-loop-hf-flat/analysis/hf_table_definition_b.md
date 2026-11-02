Definition B: speedup = time of the baseline in its own loop / fused-kernel time of the plain-launch loop of the same run. Median of 3 runs (smallest to largest run). Runs: results/rtx4090/e0-graph-structure/e0-loop-hf-flat/run1, results/rtx4090/e0-graph-structure/e0-loop-hf-flat/run2, results/rtx4090/e0-graph-structure/e0-loop-hf-flat/run3

| crops | fused kernel, plain loop (us) | plain launches | graph of child nodes | graph of one capture (fastest) | C: plain | C: child | C: one capture |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 11 | 3.18 | 6.24 (6.19 to 6.26) | 4.34 (4.27 to 4.40) | 3.61 (3.58 to 3.64) |  |  |  |
| 101 | 5.41 | 31.27 (31.09 to 31.39) | 19.51 (19.38 to 19.56) | 16.61 (16.57 to 16.73) | 31.18 | 19.44 | 16.60 |
| 301 | 9.99 | 50.05 (49.63 to 50.93) | 31.11 (31.01 to 31.30) | 25.22 (25.11 to 25.32) | 50.04 | 31.10 | 25.17 |
| 601 | 17.33 | 57.75 (57.34 to 57.84) | 35.71 (35.65 to 35.74) | 28.93 (28.78 to 28.94) |  |  |  |
| 661 | 18.78 | 58.39 (57.68 to 59.07) | 36.22 (35.96 to 36.45) | 29.34 (29.24 to 29.48) | 59.50 | 36.90 | 29.84 |
| 831 | 23.19 | 59.40 (59.39 to 59.76) | 36.97 (36.94 to 37.08) | 29.85 (29.80 to 29.94) | 60.51 | 37.64 | 30.36 |
| 1001 | 30.66 | 54.27 (53.45 to 54.45) | 33.67 (33.39 to 33.68) | 27.12 (27.10 to 27.20) | 61.91 | 38.42 | 30.97 |
| 1101 | 35.21 | 51.83 (51.17 to 52.32) | 32.10 (31.79 to 32.36) | 26.00 (25.73 to 26.06) | 62.40 | 38.65 | 31.25 |
| 1191 | 39.00 | 50.86 (50.11 to 51.09) | 31.61 (31.35 to 32.48) | 25.33 (25.27 to 25.50) | 63.18 | 39.08 | 31.48 |

C = the same baselines (median over the E0 runs) over the fused kernel timed alone after a host synchronisation (mode alone_synced of hf_fused_time_probe, median of 3 runs); measured at 101, 301, 661, 831, 1001, 1101, 1191 crops only; it is the sensitivity of B to the state of the host at the launch of the fused kernel, not the definition of the paper.

Over the 120 batch sizes (definition B, median of the runs per batch size):

| series | maximum | at crops (runs at that batch) | median over the batch sizes |
|---|--:|--:|--:|
| plain launches | 59.67 | 821 (58.77 to 59.96) | 53.75 |
| graph of child nodes | 36.98 | 821 (36.71 to 37.03) | 33.27 |
| graph of one capture | 29.85 | 831 (29.80 to 29.94) | 26.93 |
| fastest structure (= graph of one capture in 358 of 360 run x batch pairs) | 29.85 | 831 (29.80 to 29.94) | 26.93 |

fastest structure over the runs x batch sizes: child 2, flat 358

Under C, only at the crop counts where it exists (101, 301, 661, 831, 1001, 1101, 1191):

| series | maximum over those crop counts | at crops |
|---|--:|--:|
| plain launches | 63.18 | 1191 |
| graph of child nodes | 39.08 | 1191 |
| graph of one capture | 31.48 | 1191 |
| fastest structure | 31.48 | 1191 |

For comparison, B at the same crop counts: plain launches maximum 59.40 at 831; graph of child nodes maximum 36.97 at 831; graph of one capture maximum 29.85 at 831.
