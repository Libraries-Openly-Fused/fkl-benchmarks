# Cleanup window D, 2026-11-08: E16, E2 at k = 5,002 and the E1 tables, published, reference now and cleaned

## Findings

1. **E16: no number above the spread and the threshold.**
   - 152 FKL numbers and 168 Python numbers are compared. The 12 math rows at S = 8,192 are skipped by the script on every side.
   - The paper's fractions FKL / best baseline move by at most 0.019 (D=64 causal S=1024: 0.978 published, 0.983 reference now, 0.997 cleaned). None moves by more than 0.02.
   - e16_summary.py's text lines move with drift, not with the cleanup. FKL is above the best baseline in 7 of 20 points in both the reference now and the cleaned runs (published: 6).
   - Per-row SM clocks (the join of the 100 ms sampler) have a median of 2,625 / 2,655 MHz (FKL / Python) in the reference and the cleaned runs, and 2,610 / 2,640 in the published runs.
   - The variant order of every round equals the published round of the same index, and is the same in the reference and the cleaned run.
   - FKL error columns are identical in all 152 rows across published, reference and cleaned: the host generator gives the published bytes.
   - The cleaned preparation (prep_us_per_call) is 2-14 ns per call for the tensor-core variants and 12-35 ns for simt_raw.
2. **E2 at k = 5,002: the two programs now time their passes in different orders.**
   - The FKL program runs warm, then kernel-only, then flushed. The Python driver runs warm, then flushed, then kernel-only.
   - The FKL program in this window repeats window C: the cleaned flushed number is 3.5 % lower (C: 3.0 %), since its flushed pass now follows its kernel-only pass. The warm number is within 1.4 %.
   - The flushed comparison of FKL against Inductor at k = 5,002 moves, because FKL's flushed number moved and Inductor's did not:
     - published: FKL 0.5445 ms, against inductor_default 0.5193 and inductor_wide 0.5187 (+4.9 %, +5.0 %);
     - cleaned: 0.5264, against 0.5297 and 0.5199 (-0.6 %, +1.2 %).
   - Kernel-only, where both follow their own spin: FKL 0.5242 ms, Inductor 0.5154-0.5165 (FKL +1.5 to +1.7 %), Triton 0.7493 (FKL -30 %).
   - The clock log does not resolve the order effect: every E2 FKL run of both builds samples 2,670-2,685 MHz.
   - The Python driver against the published large runs: within the spreads, except eager (-3.1 %) and inductor_wide host_launch_us (+5.3 %). Both are drift, as in window C.
3. **The clock of every run.**
   - The SM clock medians of the runs are 2,625-2,685 MHz.
   - The minima of 210 MHz in the cleaned E16 FKL runs and in the Python runs fall in idle moments between configurations or processes (P8), outside the timed blocks.
   - The rows' own clocks are 2,280-2,685 MHz on every side, as in the published runs (2,280-2,670).
4. **The E1 tables of the cleaned build** (window D's mangled ncu pass, window C's cleaned binaries and runs):
   - all static columns are identical to the published tables in all 53 kernels; achieved occupancy is within 0.6 / 5.3 / 3.7 points;
   - e1_compact.csv's registers, SASS and occupancy columns are equal to the published ones; its timing columns are as in the window C report;
   - E1a vs_ref at 5,002 and 10,002 operations is 0.994 and 0.989, and kernel-only 0.998 and 0.998 (`cleaned/e1_kernel_only_ratios.csv`).
5. **Agreement.**
   - The E2 FKL outputs are identical in all six runs, and equal to window C's and the published ones.
   - The Python accuracy, kernel and SASS columns at k = 5,002 equal the published large runs'.
   - The reference E16 Python script's error columns equal the published ones in 177 of 180 rows. In the other 3, Triton causal rows, the mean error takes one of two values in both sets: the autotuner's choice per process. The maximum is equal.

## Changes above the spread and above the 3 % (GPU) or 5 % (host) threshold

4 numbers of 358 compared:

- **E2 FKL program, cleanup**, 1 number: 5002 L2-flushed (GPU) -3.5 % (reference now 0.5452, cleaned 0.5264; spreads 3.3 %, 0.1 %). Cause: order of the passes, as in window C (-3.0 % there): the cleaned flushed pass follows the kernel-only pass of the same k (100 spins of 5 ms), the reference's follows the warm pass; the clock log does not resolve it: every E2 FKL run of both builds samples 2,670-2,685 MHz, and the k = 5,002 flushed pass lasts about 80 ms, about one 100 ms sample; the power medians differ (reference 126-196 W, cleaned 112-113 W: most of a cleaned run is the one-thread spin); not isolated
- **E2 at k = 5,002, the Python driver, change**, 1 number: 5002 inductor_wide host_launch_us (host) +5.3 % (published 24.81, cleaned 26.12; spreads 3.7 %, 5.0 %). Cause: drift, as in window C (+5 to +10 % on six host numbers there): the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them); this window pinned the driver to cores 0-3 and 16-19, the published runs' pinning is not recorded; not isolated
- **E2 at k = 5,002, the Python driver, change**, 1 number: 5002 eager warm (GPU) -3.1 % (published 141.4, cleaned 137.1; spreads 0.5 %, 0.0 %). Cause: drift, as in window C (-3.0 to -3.7 % at k = 100 and 500): unchanged code; eager enqueues 5,004 PyTorch kernels per call; the SM clock of the three runs is 2,685 MHz (median of each run; the published runs have no clock log); not isolated
- **E2 at k = 5,002, the Python driver, change**, 1 number: 5002 eager L2-flushed (GPU) -3.1 % (published 141.5, cleaned 137.2; spreads 0.2 %, 0.0 %). Cause: as the warm number of eager

## E16: the paper's fraction FKL / best baseline, configurations where it moves by more than 0.02 (published against cleaned)

- none

### E16, the FKL program (e16_attention), time per call (ms; median of the blocks; median of the three rounds)

| configuration, variant | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | SM MHz pub / ref / new | cause |
|---|---|---|---|---|---|---|---|---|---|---|
| D=64 S=512 full mma_epilogue_fused | GPU | 0.2541 (1.7 %) | 0.2537 (2.2 %) | 0.2538 (0.2 %) | -0.2 % | within | +0.1 % | within | 2640 / 2625 / 2625 |  |
| D=64 S=512 full mma_epilogue_separate | GPU | 0.2928 (1.3 %) | 0.2926 (1.7 %) | 0.292 (1.1 %) | -0.1 % | within | -0.2 % | within | 2610 / 2610 / 2610 |  |
| D=64 S=512 full mma_fp8kv_fused | GPU | 0.3156 (0.1 %) | 0.3144 (0.4 %) | 0.3146 (0.0 %) | -0.4 % | within | +0.1 % | within | 2655 / 2655 / 2670 |  |
| D=64 S=512 full mma_fp8kv_unfused | GPU | 0.3746 (0.1 %) | 0.3726 (0.3 %) | 0.3722 (0.0 %) | -0.6 % | exceeds | -0.1 % | within | 2655 / 2655 / 2670 |  |
| D=64 S=512 full mma_int8kv_fused | GPU | 0.3167 (0.2 %) | 0.3169 (0.5 %) | 0.3168 (0.2 %) | +0.1 % | within | -0.0 % | within | 2655 / 2670 / 2655 |  |
| D=64 S=512 full mma_int8kv_unfused | GPU | 0.3734 (0.1 %) | 0.3723 (0.3 %) | 0.3724 (0.3 %) | -0.3 % | within | +0.0 % | within | 2670 / 2670 / 2655 |  |
| D=64 S=512 full mma_raw | GPU | 0.2566 (3.8 %) | 0.2552 (4.0 %) | 0.254 (3.2 %) | -0.6 % | within | -0.5 % | within | 2565 / 2565 / 2580 |  |
| D=64 S=512 full simt_raw | GPU | 15.57 (1.2 %) | 15.57 (0.6 %) | 15.57 (0.2 %) | +0.0 % | within | -0.0 % | within | 2655 / 2655 / 2670 |  |
| D=64 S=1024 full mma_epilogue_fused | GPU | 0.4666 (1.5 %) | 0.4652 (2.5 %) | 0.4679 (1.4 %) | -0.3 % | within | +0.6 % | within | 2580 / 2655 / 2655 |  |
| D=64 S=1024 full mma_epilogue_separate | GPU | 0.5002 (0.8 %) | 0.5002 (0.7 %) | 0.5016 (0.3 %) | +0.0 % | within | +0.3 % | within | 2610 / 2640 / 2610 |  |
| D=64 S=1024 full mma_fp8kv_fused | GPU | 0.5803 (0.3 %) | 0.58 (0.5 %) | 0.5805 (0.1 %) | -0.1 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=1024 full mma_fp8kv_unfused | GPU | 0.5829 (0.3 %) | 0.5811 (0.4 %) | 0.5811 (0.1 %) | -0.3 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=1024 full mma_int8kv_fused | GPU | 0.5858 (0.2 %) | 0.5853 (0.2 %) | 0.5853 (0.0 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=1024 full mma_int8kv_unfused | GPU | 0.5823 (0.1 %) | 0.5813 (0.1 %) | 0.5812 (0.1 %) | -0.2 % | exceeds | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=1024 full mma_raw | GPU | 0.4663 (0.6 %) | 0.4628 (0.6 %) | 0.4596 (1.6 %) | -0.7 % | exceeds | -0.7 % | within | 2640 / 2655 / 2625 |  |
| D=64 S=1024 full simt_raw | GPU | 27.27 (0.0 %) | 27.22 (0.4 %) | 27.21 (0.3 %) | -0.2 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=2048 full mma_epilogue_fused | GPU | 0.8887 (1.4 %) | 0.8815 (1.7 %) | 0.8839 (0.9 %) | -0.8 % | within | +0.3 % | within | 2655 / 2640 / 2655 |  |
| D=64 S=2048 full mma_epilogue_separate | GPU | 0.9153 (0.4 %) | 0.9167 (0.6 %) | 0.9183 (0.5 %) | +0.2 % | within | +0.2 % | within | 2625 / 2655 / 2640 |  |
| D=64 S=2048 full mma_fp8kv_fused | GPU | 1.125 (0.1 %) | 1.122 (0.3 %) | 1.126 (0.2 %) | -0.3 % | within | +0.3 % | exceeds | 2655 / 2655 / 2655 |  |
| D=64 S=2048 full mma_fp8kv_unfused | GPU | 1 (0.2 %) | 0.9976 (0.2 %) | 0.9985 (0.1 %) | -0.3 % | exceeds | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=2048 full mma_int8kv_fused | GPU | 1.139 (0.2 %) | 1.137 (0.3 %) | 1.138 (0.1 %) | -0.2 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=2048 full mma_int8kv_unfused | GPU | 1 (0.2 %) | 0.9976 (0.2 %) | 0.9987 (0.1 %) | -0.2 % | exceeds | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=2048 full mma_raw | GPU | 0.8842 (1.3 %) | 0.8781 (1.1 %) | 0.8787 (1.0 %) | -0.7 % | within | +0.1 % | within | 2610 / 2655 / 2655 |  |
| D=64 S=2048 full simt_raw | GPU | 50.7 (0.2 %) | 50.59 (0.7 %) | 50.74 (0.1 %) | -0.2 % | within | +0.3 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 full mma_epilogue_fused | GPU | 1.712 (0.2 %) | 1.709 (0.3 %) | 1.711 (0.1 %) | -0.2 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 full mma_epilogue_separate | GPU | 1.748 (0.5 %) | 1.748 (0.2 %) | 1.75 (0.1 %) | +0.0 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 full mma_fp8kv_fused | GPU | 2.224 (0.2 %) | 2.213 (0.3 %) | 2.216 (0.1 %) | -0.5 % | exceeds | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 full mma_fp8kv_unfused | GPU | 1.833 (0.2 %) | 1.83 (0.2 %) | 1.831 (0.1 %) | -0.2 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 full mma_int8kv_fused | GPU | 2.25 (0.2 %) | 2.246 (0.3 %) | 2.248 (0.2 %) | -0.2 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 full mma_int8kv_unfused | GPU | 1.833 (0.2 %) | 1.83 (0.2 %) | 1.831 (0.1 %) | -0.2 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 full mma_raw | GPU | 1.709 (0.6 %) | 1.706 (0.3 %) | 1.708 (0.1 %) | -0.2 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 full mma_epilogue_fused | GPU | 3.384 (0.3 %) | 3.378 (0.3 %) | 3.379 (0.1 %) | -0.2 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 full mma_epilogue_separate | GPU | 3.419 (0.3 %) | 3.416 (0.2 %) | 3.417 (0.1 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 full mma_fp8kv_fused | GPU | 4.418 (0.2 %) | 4.411 (0.1 %) | 4.416 (0.2 %) | -0.1 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 full mma_fp8kv_unfused | GPU | 3.503 (0.3 %) | 3.497 (0.3 %) | 3.498 (0.1 %) | -0.2 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 full mma_int8kv_fused | GPU | 4.465 (0.2 %) | 4.462 (0.2 %) | 4.46 (0.2 %) | -0.1 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 full mma_int8kv_unfused | GPU | 3.504 (0.2 %) | 3.498 (0.3 %) | 3.499 (0.1 %) | -0.2 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 full mma_raw | GPU | 3.379 (0.3 %) | 3.374 (0.3 %) | 3.374 (0.1 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=512 causal mma_epilogue_fused | GPU | 0.1642 (2.4 %) | 0.1648 (3.0 %) | 0.1641 (2.4 %) | +0.3 % | within | -0.4 % | within | 2340 / 2385 / 2355 |  |
| D=64 S=512 causal mma_epilogue_separate | GPU | 0.2063 (0.8 %) | 0.2068 (1.0 %) | 0.2075 (0.6 %) | +0.2 % | within | +0.4 % | within | 2565 / 2580 / 2595 |  |
| D=64 S=512 causal mma_fp8kv_fused | GPU | 0.2208 (0.2 %) | 0.2209 (0.3 %) | 0.2206 (0.1 %) | +0.0 % | within | -0.1 % | within | 2655 / 2655 / 2670 |  |
| D=64 S=512 causal mma_fp8kv_unfused | GPU | 0.2832 (0.3 %) | 0.2819 (0.3 %) | 0.2818 (0.1 %) | -0.4 % | exceeds | -0.1 % | within | 2655 / 2655 / 2670 |  |
| D=64 S=512 causal mma_int8kv_fused | GPU | 0.2226 (0.2 %) | 0.2224 (0.3 %) | 0.2222 (0.0 %) | -0.1 % | within | -0.1 % | within | 2655 / 2655 / 2670 |  |
| D=64 S=512 causal mma_int8kv_unfused | GPU | 0.2826 (0.3 %) | 0.2826 (0.3 %) | 0.2822 (0.4 %) | +0.0 % | within | -0.1 % | within | 2655 / 2655 / 2670 |  |
| D=64 S=512 causal mma_raw | GPU | 0.1635 (1.0 %) | 0.1637 (0.9 %) | 0.1624 (1.8 %) | +0.1 % | within | -0.7 % | within | 2400 / 2400 / 2400 |  |
| D=64 S=512 causal simt_raw | GPU | 7.056 (0.3 %) | 7.029 (0.3 %) | 7.067 (0.0 %) | -0.4 % | exceeds | +0.5 % | exceeds | 2655 / 2655 / 2655 |  |
| D=64 S=1024 causal mma_epilogue_fused | GPU | 0.2913 (0.5 %) | 0.2929 (1.3 %) | 0.2922 (2.3 %) | +0.5 % | within | -0.2 % | within | 2430 / 2430 / 2415 |  |
| D=64 S=1024 causal mma_epilogue_separate | GPU | 0.3304 (2.3 %) | 0.3278 (2.7 %) | 0.3303 (2.2 %) | -0.8 % | within | +0.8 % | within | 2520 / 2550 / 2565 |  |
| D=64 S=1024 causal mma_fp8kv_fused | GPU | 0.364 (0.1 %) | 0.364 (0.2 %) | 0.3639 (0.1 %) | -0.0 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=1024 causal mma_fp8kv_unfused | GPU | 0.4002 (0.5 %) | 0.3991 (0.6 %) | 0.3988 (0.1 %) | -0.3 % | within | -0.1 % | within | 2610 / 2610 / 2610 |  |
| D=64 S=1024 causal mma_int8kv_fused | GPU | 0.3683 (0.1 %) | 0.3679 (0.3 %) | 0.3681 (0.0 %) | -0.1 % | within | +0.1 % | within | 2610 / 2655 / 2655 |  |
| D=64 S=1024 causal mma_int8kv_unfused | GPU | 0.3999 (0.7 %) | 0.3996 (0.2 %) | 0.3998 (0.9 %) | -0.1 % | within | +0.0 % | within | 2595 / 2610 / 2610 |  |
| D=64 S=1024 causal mma_raw | GPU | 0.2939 (1.1 %) | 0.2917 (1.2 %) | 0.2878 (3.1 %) | -0.8 % | within | -1.3 % | within | 2490 / 2490 / 2475 |  |
| D=64 S=1024 causal simt_raw | GPU | 12.83 (0.8 %) | 12.8 (0.4 %) | 12.82 (0.1 %) | -0.2 % | within | +0.2 % | within | 2610 / 2625 / 2655 |  |
| D=64 S=2048 causal mma_epilogue_fused | GPU | 0.5249 (0.6 %) | 0.5217 (0.1 %) | 0.5247 (0.3 %) | -0.6 % | exceeds | +0.6 % | exceeds | 2460 / 2475 / 2460 |  |
| D=64 S=2048 causal mma_epilogue_separate | GPU | 0.5609 (3.7 %) | 0.561 (3.1 %) | 0.5513 (2.3 %) | +0.0 % | within | -1.7 % | within | 2550 / 2550 / 2565 |  |
| D=64 S=2048 causal mma_fp8kv_fused | GPU | 0.6584 (0.2 %) | 0.658 (0.3 %) | 0.6575 (0.2 %) | -0.1 % | within | -0.1 % | within | 2655 / 2640 / 2655 |  |
| D=64 S=2048 causal mma_fp8kv_unfused | GPU | 0.6228 (0.3 %) | 0.6212 (0.7 %) | 0.6196 (0.2 %) | -0.3 % | within | -0.3 % | within | 2610 / 2625 / 2610 |  |
| D=64 S=2048 causal mma_int8kv_fused | GPU | 0.6688 (0.2 %) | 0.6675 (0.3 %) | 0.6683 (0.2 %) | -0.2 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=2048 causal mma_int8kv_unfused | GPU | 0.6305 (1.7 %) | 0.6274 (1.0 %) | 0.6204 (1.7 %) | -0.5 % | within | -1.1 % | within | 2595 / 2580 / 2610 |  |
| D=64 S=2048 causal mma_raw | GPU | 0.5221 (1.5 %) | 0.52 (1.6 %) | 0.5176 (1.7 %) | -0.4 % | within | -0.5 % | within | 2505 / 2520 / 2520 |  |
| D=64 S=2048 causal simt_raw | GPU | 24.83 (1.8 %) | 24.66 (1.8 %) | 24.8 (1.7 %) | -0.7 % | within | +0.6 % | within | 2610 / 2610 / 2550 |  |
| D=64 S=4096 causal mma_epilogue_fused | GPU | 0.9809 (0.2 %) | 0.9809 (0.3 %) | 0.9814 (0.3 %) | -0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 causal mma_epilogue_separate | GPU | 1.023 (0.2 %) | 1.022 (0.5 %) | 1.023 (0.1 %) | -0.1 % | within | +0.1 % | within | 2655 / 2670 / 2655 |  |
| D=64 S=4096 causal mma_fp8kv_fused | GPU | 1.255 (0.2 %) | 1.254 (0.2 %) | 1.253 (0.3 %) | -0.1 % | within | -0.1 % | within | 2655 / 2670 / 2655 |  |
| D=64 S=4096 causal mma_fp8kv_unfused | GPU | 1.102 (0.3 %) | 1.1 (0.2 %) | 1.103 (0.2 %) | -0.2 % | within | +0.2 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 causal mma_int8kv_fused | GPU | 1.274 (0.2 %) | 1.273 (0.2 %) | 1.273 (0.1 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 causal mma_int8kv_unfused | GPU | 1.103 (0.1 %) | 1.102 (0.3 %) | 1.101 (0.2 %) | -0.0 % | within | -0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 causal mma_raw | GPU | 0.9812 (0.3 %) | 0.9793 (0.3 %) | 0.9792 (0.2 %) | -0.2 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal mma_epilogue_fused | GPU | 1.905 (0.2 %) | 1.902 (0.2 %) | 1.904 (0.2 %) | -0.2 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal mma_epilogue_separate | GPU | 1.945 (0.1 %) | 1.945 (0.2 %) | 1.942 (0.2 %) | -0.0 % | within | -0.2 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal mma_fp8kv_fused | GPU | 2.434 (0.3 %) | 2.435 (0.1 %) | 2.436 (0.1 %) | +0.0 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal mma_fp8kv_unfused | GPU | 2.025 (0.3 %) | 2.023 (0.1 %) | 2.025 (0.1 %) | -0.1 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal mma_int8kv_fused | GPU | 2.465 (0.2 %) | 2.467 (0.1 %) | 2.469 (0.1 %) | +0.1 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal mma_int8kv_unfused | GPU | 2.029 (0.2 %) | 2.026 (0.3 %) | 2.026 (0.2 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal mma_raw | GPU | 1.903 (0.3 %) | 1.902 (0.3 %) | 1.903 (0.1 %) | -0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=512 full mma_epilogue_fused | GPU | 0.5099 (1.1 %) | 0.5068 (1.7 %) | 0.5035 (2.5 %) | -0.6 % | within | -0.7 % | within | 2445 / 2415 / 2550 |  |
| D=128 S=512 full mma_epilogue_separate | GPU | 0.6121 (1.6 %) | 0.6126 (2.1 %) | 0.6189 (1.9 %) | +0.1 % | within | +1.0 % | within | 2550 / 2550 / 2505 |  |
| D=128 S=512 full mma_fp8kv_fused | GPU | 0.6605 (0.7 %) | 0.6601 (1.8 %) | 0.6618 (2.1 %) | -0.1 % | within | +0.3 % | within | 2520 / 2535 / 2550 |  |
| D=128 S=512 full mma_fp8kv_unfused | GPU | 0.7402 (1.1 %) | 0.74 (0.9 %) | 0.7375 (1.3 %) | -0.0 % | within | -0.3 % | within | 2550 / 2580 / 2565 |  |
| D=128 S=512 full mma_int8kv_fused | GPU | 0.6724 (0.9 %) | 0.6703 (1.3 %) | 0.6671 (1.7 %) | -0.3 % | within | -0.5 % | within | 2535 / 2535 / 2595 |  |
| D=128 S=512 full mma_int8kv_unfused | GPU | 0.7353 (0.9 %) | 0.7359 (0.3 %) | 0.7409 (1.4 %) | +0.1 % | within | +0.7 % | within | 2565 / 2595 / 2535 |  |
| D=128 S=512 full mma_raw | GPU | 0.5086 (3.1 %) | 0.5066 (3.5 %) | 0.497 (3.0 %) | -0.4 % | within | -1.9 % | within | 2460 / 2475 / 2505 |  |
| D=128 S=512 full simt_raw | GPU | 27.86 (0.1 %) | 27.76 (1.8 %) | 27.52 (1.6 %) | -0.3 % | within | -0.9 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=1024 full mma_epilogue_fused | GPU | 0.9449 (0.6 %) | 0.9421 (1.5 %) | 0.9336 (1.0 %) | -0.3 % | within | -0.9 % | within | 2505 / 2520 / 2535 |  |
| D=128 S=1024 full mma_epilogue_separate | GPU | 1.051 (1.6 %) | 1.056 (1.6 %) | 1.058 (1.0 %) | +0.5 % | within | +0.2 % | within | 2505 / 2535 / 2520 |  |
| D=128 S=1024 full mma_fp8kv_fused | GPU | 1.219 (0.4 %) | 1.215 (1.0 %) | 1.223 (0.9 %) | -0.4 % | within | +0.7 % | within | 2580 / 2565 / 2565 |  |
| D=128 S=1024 full mma_fp8kv_unfused | GPU | 1.185 (0.5 %) | 1.179 (0.5 %) | 1.179 (1.9 %) | -0.5 % | within | +0.0 % | within | 2572 / 2565 / 2610 |  |
| D=128 S=1024 full mma_int8kv_fused | GPU | 1.243 (1.0 %) | 1.238 (1.3 %) | 1.245 (1.5 %) | -0.4 % | within | +0.6 % | within | 2565 / 2610 / 2565 |  |
| D=128 S=1024 full mma_int8kv_unfused | GPU | 1.179 (2.3 %) | 1.18 (1.8 %) | 1.168 (2.3 %) | +0.0 % | within | -1.0 % | within | 2565 / 2565 / 2595 |  |
| D=128 S=1024 full mma_raw | GPU | 0.9383 (3.5 %) | 0.938 (4.5 %) | 0.9438 (3.4 %) | -0.0 % | within | +0.6 % | within | 2490 / 2565 / 2520 |  |
| D=128 S=1024 full simt_raw | GPU | 47.39 (0.6 %) | 47.37 (0.1 %) | 47.39 (0.1 %) | -0.0 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=2048 full mma_epilogue_fused | GPU | 1.83 (1.7 %) | 1.819 (1.2 %) | 1.811 (1.2 %) | -0.6 % | within | -0.4 % | within | 2520 / 2520 / 2550 |  |
| D=128 S=2048 full mma_epilogue_separate | GPU | 1.921 (1.1 %) | 1.907 (0.6 %) | 1.915 (1.6 %) | -0.7 % | within | +0.4 % | within | 2572 / 2565 / 2565 |  |
| D=128 S=2048 full mma_fp8kv_fused | GPU | 2.382 (1.3 %) | 2.372 (1.0 %) | 2.366 (1.2 %) | -0.4 % | within | -0.2 % | within | 2610 / 2610 / 2595 |  |
| D=128 S=2048 full mma_fp8kv_unfused | GPU | 2.044 (1.1 %) | 2.033 (1.1 %) | 2.053 (1.5 %) | -0.6 % | within | +1.0 % | within | 2550 / 2550 / 2550 |  |
| D=128 S=2048 full mma_int8kv_fused | GPU | 2.364 (1.2 %) | 2.362 (0.5 %) | 2.379 (1.6 %) | -0.1 % | within | +0.7 % | within | 2610 / 2610 / 2595 |  |
| D=128 S=2048 full mma_int8kv_unfused | GPU | 2.048 (1.1 %) | 2.034 (2.0 %) | 2.033 (1.6 %) | -0.7 % | within | -0.1 % | within | 2550 / 2565 / 2550 |  |
| D=128 S=2048 full mma_raw | GPU | 1.823 (1.3 %) | 1.807 (1.8 %) | 1.799 (2.9 %) | -0.9 % | within | -0.5 % | within | 2565 / 2595 / 2565 |  |
| D=128 S=2048 full simt_raw | GPU | 86.84 (0.3 %) | 86.54 (0.3 %) | 86.71 (0.3 %) | -0.4 % | exceeds | +0.2 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=4096 full mma_epilogue_fused | GPU | 3.551 (0.9 %) | 3.55 (0.0 %) | 3.551 (0.6 %) | -0.0 % | within | +0.0 % | within | 2558 / 2550 / 2550 |  |
| D=128 S=4096 full mma_epilogue_separate | GPU | 3.658 (1.3 %) | 3.658 (0.9 %) | 3.637 (1.5 %) | +0.0 % | within | -0.6 % | within | 2565 / 2580 / 2565 |  |
| D=128 S=4096 full mma_fp8kv_fused | GPU | 4.691 (1.1 %) | 4.682 (0.7 %) | 4.663 (0.5 %) | -0.2 % | within | -0.4 % | within | 2640 / 2640 / 2655 |  |
| D=128 S=4096 full mma_fp8kv_unfused | GPU | 3.763 (1.5 %) | 3.746 (1.1 %) | 3.764 (1.1 %) | -0.4 % | within | +0.5 % | within | 2595 / 2595 / 2565 |  |
| D=128 S=4096 full mma_int8kv_fused | GPU | 4.606 (0.2 %) | 4.607 (0.3 %) | 4.624 (0.3 %) | +0.0 % | within | +0.4 % | exceeds | 2625 / 2610 / 2595 |  |
| D=128 S=4096 full mma_int8kv_unfused | GPU | 3.787 (0.6 %) | 3.787 (1.5 %) | 3.726 (2.7 %) | +0.0 % | within | -1.6 % | within | 2565 / 2610 / 2610 |  |
| D=128 S=4096 full mma_raw | GPU | 3.55 (0.6 %) | 3.518 (1.5 %) | 3.529 (1.9 %) | -0.9 % | within | +0.3 % | within | 2550 / 2565 / 2565 |  |
| D=128 S=8192 full mma_epilogue_fused | GPU | 6.97 (0.4 %) | 6.938 (0.6 %) | 6.969 (0.4 %) | -0.5 % | within | +0.4 % | within | 2580 / 2580 / 2580 |  |
| D=128 S=8192 full mma_epilogue_separate | GPU | 7.098 (0.5 %) | 7.057 (0.6 %) | 7.057 (1.1 %) | -0.6 % | within | +0.0 % | within | 2580 / 2610 / 2610 |  |
| D=128 S=8192 full mma_fp8kv_fused | GPU | 9.229 (0.8 %) | 9.206 (0.1 %) | 9.201 (0.1 %) | -0.2 % | within | -0.1 % | within | 2640 / 2655 / 2655 |  |
| D=128 S=8192 full mma_fp8kv_unfused | GPU | 7.225 (1.2 %) | 7.188 (0.7 %) | 7.187 (1.2 %) | -0.5 % | within | -0.0 % | within | 2595 / 2610 / 2625 |  |
| D=128 S=8192 full mma_int8kv_fused | GPU | 9.055 (0.1 %) | 9.067 (0.2 %) | 9.058 (0.1 %) | +0.1 % | within | -0.1 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 full mma_int8kv_unfused | GPU | 7.269 (1.1 %) | 7.233 (0.2 %) | 7.243 (0.6 %) | -0.5 % | within | +0.1 % | within | 2580 / 2602 / 2595 |  |
| D=128 S=8192 full mma_raw | GPU | 6.971 (0.0 %) | 6.931 (0.5 %) | 6.895 (1.6 %) | -0.6 % | exceeds | -0.5 % | within | 2580 / 2595 / 2610 |  |
| D=128 S=512 causal mma_epilogue_fused | GPU | 0.3249 (0.7 %) | 0.3245 (0.4 %) | 0.3245 (0.8 %) | -0.1 % | within | -0.0 % | within | 2355 / 2370 / 2445 |  |
| D=128 S=512 causal mma_epilogue_separate | GPU | 0.4508 (0.2 %) | 0.4508 (0.3 %) | 0.4504 (0.2 %) | -0.0 % | within | -0.1 % | within | 2580 / 2595 / 2565 |  |
| D=128 S=512 causal mma_fp8kv_fused | GPU | 0.4152 (0.1 %) | 0.4132 (0.4 %) | 0.4152 (0.0 %) | -0.5 % | exceeds | +0.5 % | exceeds | 2505 / 2550 / 2505 |  |
| D=128 S=512 causal mma_fp8kv_unfused | GPU | 0.5739 (0.0 %) | 0.5729 (0.0 %) | 0.5733 (0.0 %) | -0.2 % | exceeds | +0.1 % | exceeds | 2625 / 2625 / 2595 |  |
| D=128 S=512 causal mma_int8kv_fused | GPU | 0.4192 (0.6 %) | 0.4186 (0.9 %) | 0.4192 (0.5 %) | -0.1 % | within | +0.1 % | within | 2610 / 2520 / 2625 |  |
| D=128 S=512 causal mma_int8kv_unfused | GPU | 0.5748 (0.0 %) | 0.5748 (0.1 %) | 0.5741 (0.0 %) | -0.0 % | within | -0.1 % | exceeds | 2655 / 2655 / 2640 |  |
| D=128 S=512 causal mma_raw | GPU | 0.3182 (2.1 %) | 0.3169 (1.9 %) | 0.3163 (2.0 %) | -0.4 % | within | -0.2 % | within | 2415 / 2460 / 2460 |  |
| D=128 S=512 causal simt_raw | GPU | 11.31 (0.1 %) | 11.36 (0.1 %) | 11.33 (0.1 %) | +0.4 % | exceeds | -0.3 % | exceeds | 2655 / 2655 / 2655 |  |
| D=128 S=1024 causal mma_epilogue_fused | GPU | 0.5493 (1.4 %) | 0.5515 (1.5 %) | 0.5462 (1.9 %) | +0.4 % | within | -1.0 % | within | 2490 / 2475 / 2490 |  |
| D=128 S=1024 causal mma_epilogue_separate | GPU | 0.6584 (1.2 %) | 0.6567 (1.1 %) | 0.6582 (1.6 %) | -0.3 % | within | +0.2 % | within | 2535 / 2535 / 2550 |  |
| D=128 S=1024 causal mma_fp8kv_fused | GPU | 0.7106 (2.3 %) | 0.7073 (2.0 %) | 0.7077 (2.5 %) | -0.5 % | within | +0.0 % | within | 2625 / 2595 / 2580 |  |
| D=128 S=1024 causal mma_fp8kv_unfused | GPU | 0.7771 (1.6 %) | 0.7757 (1.5 %) | 0.7765 (1.5 %) | -0.2 % | within | +0.1 % | within | 2580 / 2580 / 2565 |  |
| D=128 S=1024 causal mma_int8kv_fused | GPU | 0.7165 (1.3 %) | 0.7133 (0.9 %) | 0.7141 (1.4 %) | -0.4 % | within | +0.1 % | within | 2595 / 2565 / 2610 |  |
| D=128 S=1024 causal mma_int8kv_unfused | GPU | 0.7796 (1.1 %) | 0.7787 (0.9 %) | 0.7823 (1.5 %) | -0.1 % | within | +0.5 % | within | 2550 / 2535 / 2565 |  |
| D=128 S=1024 causal mma_raw | GPU | 0.5489 (1.7 %) | 0.5455 (1.9 %) | 0.5457 (2.4 %) | -0.6 % | within | +0.0 % | within | 2475 / 2475 / 2460 |  |
| D=128 S=1024 causal simt_raw | GPU | 20.13 (0.1 %) | 20.13 (0.1 %) | 20.15 (0.1 %) | +0.0 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=2048 causal mma_epilogue_fused | GPU | 1.01 (1.3 %) | 1.003 (1.8 %) | 1.003 (1.7 %) | -0.7 % | within | +0.0 % | within | 2595 / 2580 / 2565 |  |
| D=128 S=2048 causal mma_epilogue_separate | GPU | 1.114 (2.4 %) | 1.12 (2.1 %) | 1.117 (2.4 %) | +0.5 % | within | -0.3 % | within | 2565 / 2595 / 2580 |  |
| D=128 S=2048 causal mma_fp8kv_fused | GPU | 1.294 (0.3 %) | 1.294 (0.4 %) | 1.294 (0.5 %) | +0.0 % | within | -0.0 % | within | 2595 / 2580 / 2565 |  |
| D=128 S=2048 causal mma_fp8kv_unfused | GPU | 1.244 (1.3 %) | 1.231 (0.9 %) | 1.228 (1.4 %) | -1.1 % | within | -0.3 % | within | 2610 / 2610 / 2595 |  |
| D=128 S=2048 causal mma_int8kv_fused | GPU | 1.298 (0.6 %) | 1.297 (1.0 %) | 1.304 (0.6 %) | -0.1 % | within | +0.5 % | within | 2565 / 2565 / 2640 |  |
| D=128 S=2048 causal mma_int8kv_unfused | GPU | 1.228 (1.1 %) | 1.229 (1.2 %) | 1.228 (0.7 %) | +0.1 % | within | -0.0 % | within | 2580 / 2610 / 2610 |  |
| D=128 S=2048 causal mma_raw | GPU | 1.002 (2.5 %) | 1.003 (2.4 %) | 0.9964 (2.2 %) | +0.0 % | within | -0.6 % | within | 2550 / 2550 / 2565 |  |
| D=128 S=2048 causal simt_raw | GPU | 39.98 (0.2 %) | 39.92 (0.2 %) | 40.02 (0.2 %) | -0.1 % | within | +0.2 % | exceeds | 2655 / 2655 / 2655 |  |
| D=128 S=4096 causal mma_epilogue_fused | GPU | 1.912 (0.6 %) | 1.91 (1.0 %) | 1.913 (1.2 %) | -0.1 % | within | +0.2 % | within | 2580 / 2565 / 2565 |  |
| D=128 S=4096 causal mma_epilogue_separate | GPU | 2.018 (0.5 %) | 2.016 (0.4 %) | 2.007 (2.7 %) | -0.1 % | within | -0.4 % | within | 2565 / 2595 / 2580 |  |
| D=128 S=4096 causal mma_fp8kv_fused | GPU | 2.476 (0.5 %) | 2.465 (0.5 %) | 2.439 (1.0 %) | -0.4 % | within | -1.1 % | exceeds | 2610 / 2625 / 2655 |  |
| D=128 S=4096 causal mma_fp8kv_unfused | GPU | 2.144 (0.4 %) | 2.141 (0.3 %) | 2.143 (1.0 %) | -0.1 % | within | +0.1 % | within | 2595 / 2610 / 2625 |  |
| D=128 S=4096 causal mma_int8kv_fused | GPU | 2.426 (0.5 %) | 2.423 (0.9 %) | 2.437 (0.7 %) | -0.1 % | within | +0.6 % | within | 2610 / 2610 / 2655 |  |
| D=128 S=4096 causal mma_int8kv_unfused | GPU | 2.143 (0.0 %) | 2.134 (0.6 %) | 2.127 (1.3 %) | -0.4 % | within | -0.3 % | within | 2595 / 2640 / 2580 |  |
| D=128 S=4096 causal mma_raw | GPU | 1.908 (0.4 %) | 1.902 (0.8 %) | 1.892 (1.3 %) | -0.3 % | within | -0.5 % | within | 2558 / 2565 / 2580 |  |
| D=128 S=8192 causal mma_epilogue_fused | GPU | 3.676 (0.4 %) | 3.67 (0.5 %) | 3.687 (1.1 %) | -0.2 % | within | +0.5 % | within | 2565 / 2595 / 2580 |  |
| D=128 S=8192 causal mma_epilogue_separate | GPU | 3.799 (0.7 %) | 3.798 (1.1 %) | 3.79 (1.3 %) | -0.0 % | within | -0.2 % | within | 2632 / 2595 / 2595 |  |
| D=128 S=8192 causal mma_fp8kv_fused | GPU | 4.792 (0.5 %) | 4.766 (0.1 %) | 4.768 (0.3 %) | -0.5 % | exceeds | +0.0 % | within | 2610 / 2625 / 2640 |  |
| D=128 S=8192 causal mma_fp8kv_unfused | GPU | 3.915 (1.1 %) | 3.903 (0.5 %) | 3.915 (1.2 %) | -0.3 % | within | +0.3 % | within | 2595 / 2610 / 2595 |  |
| D=128 S=8192 causal mma_int8kv_fused | GPU | 4.651 (0.9 %) | 4.632 (0.4 %) | 4.654 (0.4 %) | -0.4 % | within | +0.5 % | exceeds | 2625 / 2610 / 2580 |  |
| D=128 S=8192 causal mma_int8kv_unfused | GPU | 3.927 (1.0 %) | 3.92 (0.4 %) | 3.903 (1.8 %) | -0.2 % | within | -0.4 % | within | 2610 / 2595 / 2640 |  |
| D=128 S=8192 causal mma_raw | GPU | 3.692 (1.5 %) | 3.682 (0.7 %) | 3.685 (2.1 %) | -0.3 % | within | +0.1 % | within | 2580 / 2595 / 2595 |  |

### E16, the Python baselines (e16_baselines.py), time per call (ms; median of the blocks; median of the three rounds; the reference script makes its own torch.randn inputs, the cleaned one reads the FKL input files)

| configuration, variant | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | SM MHz pub / ref / new | cause |
|---|---|---|---|---|---|---|---|---|---|---|
| D=64 S=512 full cudnn | GPU | 0.2567 (0.2 %) | 0.2564 (0.1 %) | 0.2562 (0.1 %) | -0.1 % | within | -0.1 % | within | 2670 / 2670 / 2670 |  |
| D=64 S=512 full dequant_compiled_flash | GPU | 0.3242 (0.4 %) | 0.3261 (1.1 %) | 0.3257 (1.6 %) | +0.6 % | within | -0.1 % | within | 2655 / 2640 / 2655 |  |
| D=64 S=512 full dequant_flash | GPU | 0.3802 (1.3 %) | 0.3817 (1.7 %) | 0.383 (1.8 %) | +0.4 % | within | +0.3 % | within | 2625 / 2580 / 2565 |  |
| D=64 S=512 full efficient | GPU | 0.3259 (3.7 %) | 0.3221 (3.4 %) | 0.3204 (3.0 %) | -1.2 % | within | -0.5 % | within | 2355 / 2355 / 2355 |  |
| D=64 S=512 full epilogue_flash | GPU | 0.2948 (2.1 %) | 0.2967 (0.4 %) | 0.2948 (0.7 %) | +0.6 % | within | -0.6 % | within | 2580 / 2595 / 2610 |  |
| D=64 S=512 full flash | GPU | 0.248 (3.5 %) | 0.2465 (3.6 %) | 0.2469 (3.6 %) | -0.6 % | within | +0.2 % | within | 2565 / 2535 / 2535 |  |
| D=64 S=512 full flash_attn | GPU | 0.247 (0.5 %) | 0.2461 (0.4 %) | 0.2463 (0.4 %) | -0.3 % | within | +0.1 % | within | 2610 / 2625 / 2610 |  |
| D=64 S=512 full math | GPU | 6.357 (0.1 %) | 6.363 (0.1 %) | 6.361 (0.2 %) | +0.1 % | within | -0.0 % | within | 2655 / 2670 / 2670 |  |
| D=64 S=512 full triton | GPU | 0.2386 (0.0 %) | 0.2388 (0.3 %) | 0.2388 (0.3 %) | +0.1 % | within | -0.0 % | within | 2618 / 2625 / 2610 |  |
| D=64 S=1024 full cudnn | GPU | 0.4753 (0.1 %) | 0.4754 (0.2 %) | 0.4753 (0.0 %) | +0.0 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=1024 full dequant_compiled_flash | GPU | 0.536 (1.3 %) | 0.5387 (1.3 %) | 0.5396 (1.5 %) | +0.5 % | within | +0.2 % | within | 2625 / 2648 / 2602 |  |
| D=64 S=1024 full dequant_flash | GPU | 0.5914 (0.8 %) | 0.5902 (1.0 %) | 0.5917 (1.3 %) | -0.2 % | within | +0.3 % | within | 2550 / 2550 / 2625 |  |
| D=64 S=1024 full efficient | GPU | 0.6355 (2.3 %) | 0.63 (2.2 %) | 0.6298 (2.2 %) | -0.9 % | within | -0.0 % | within | 2355 / 2340 / 2370 |  |
| D=64 S=1024 full epilogue_flash | GPU | 0.5156 (1.3 %) | 0.5126 (1.6 %) | 0.5126 (1.3 %) | -0.6 % | within | +0.0 % | within | 2565 / 2580 / 2595 |  |
| D=64 S=1024 full flash | GPU | 0.4627 (0.7 %) | 0.4622 (0.8 %) | 0.4628 (0.1 %) | -0.1 % | within | +0.1 % | within | 2520 / 2520 / 2520 |  |
| D=64 S=1024 full flash_attn | GPU | 0.4604 (2.2 %) | 0.4596 (1.9 %) | 0.4597 (1.3 %) | -0.2 % | within | +0.0 % | within | 2595 / 2670 / 2670 |  |
| D=64 S=1024 full math | GPU | 11.8 (0.0 %) | 11.8 (0.1 %) | 11.8 (0.0 %) | +0.0 % | within | -0.1 % | within | 2662 / 2662 / 2670 |  |
| D=64 S=1024 full triton | GPU | 0.4469 (0.2 %) | 0.4461 (0.1 %) | 0.4465 (0.2 %) | -0.2 % | within | +0.1 % | within | 2640 / 2655 / 2655 |  |
| D=64 S=2048 full cudnn | GPU | 0.9142 (0.0 %) | 0.9145 (0.1 %) | 0.9152 (0.0 %) | +0.0 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=2048 full dequant_compiled_flash | GPU | 0.9639 (0.8 %) | 0.9711 (0.9 %) | 0.9662 (0.9 %) | +0.7 % | within | -0.5 % | within | 2640 / 2610 / 2655 |  |
| D=64 S=2048 full dequant_flash | GPU | 1.029 (1.5 %) | 1.023 (1.5 %) | 1.025 (1.4 %) | -0.5 % | within | +0.1 % | within | 2580 / 2595 / 2580 |  |
| D=64 S=2048 full efficient | GPU | 1.229 (1.5 %) | 1.21 (0.8 %) | 1.206 (1.3 %) | -1.6 % | exceeds | -0.3 % | within | 2370 / 2385 / 2385 |  |
| D=64 S=2048 full epilogue_flash | GPU | 0.9458 (1.5 %) | 0.9528 (2.1 %) | 0.9527 (2.1 %) | +0.7 % | within | -0.0 % | within | 2595 / 2610 / 2595 |  |
| D=64 S=2048 full flash | GPU | 0.8958 (1.5 %) | 0.8916 (1.6 %) | 0.8909 (1.1 %) | -0.5 % | within | -0.1 % | within | 2550 / 2565 / 2565 |  |
| D=64 S=2048 full flash_attn | GPU | 0.8999 (1.7 %) | 0.8999 (1.7 %) | 0.8999 (0.6 %) | +0.0 % | within | -0.0 % | within | 2610 / 2625 / 2625 |  |
| D=64 S=2048 full math | GPU | 22.64 (0.1 %) | 22.65 (0.1 %) | 22.65 (0.0 %) | +0.0 % | within | +0.0 % | within | 2670 / 2670 / 2670 |  |
| D=64 S=2048 full triton | GPU | 0.8769 (0.2 %) | 0.8683 (0.6 %) | 0.8649 (0.1 %) | -1.0 % | exceeds | -0.4 % | within | 2640 / 2640 / 2655 |  |
| D=64 S=4096 full cudnn | GPU | 1.796 (0.1 %) | 1.795 (0.0 %) | 1.794 (0.1 %) | -0.1 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 full dequant_compiled_flash | GPU | 1.83 (1.0 %) | 1.824 (0.1 %) | 1.822 (0.0 %) | -0.3 % | within | -0.1 % | within | 2640 / 2655 / 2655 |  |
| D=64 S=4096 full dequant_flash | GPU | 1.9 (1.3 %) | 1.897 (0.7 %) | 1.89 (0.3 %) | -0.2 % | within | -0.3 % | within | 2610 / 2610 / 2610 |  |
| D=64 S=4096 full efficient | GPU | 2.404 (1.1 %) | 2.383 (1.7 %) | 2.325 (0.6 %) | -0.9 % | within | -2.5 % | exceeds | 2385 / 2400 / 2400 |  |
| D=64 S=4096 full epilogue_flash | GPU | 1.815 (1.1 %) | 1.816 (0.9 %) | 1.826 (1.5 %) | +0.1 % | within | +0.5 % | within | 2610 / 2595 / 2640 |  |
| D=64 S=4096 full flash | GPU | 1.76 (0.6 %) | 1.75 (0.5 %) | 1.76 (1.6 %) | -0.6 % | within | +0.5 % | within | 2580 / 2595 / 2625 |  |
| D=64 S=4096 full flash_attn | GPU | 1.77 (0.0 %) | 1.77 (0.0 %) | 1.768 (0.0 %) | -0.0 % | within | -0.1 % | exceeds | 2640 / 2670 / 2655 |  |
| D=64 S=4096 full math | GPU | 44.08 (0.0 %) | 44.08 (0.0 %) | 44.07 (0.0 %) | +0.0 % | within | -0.0 % | within | 2670 / 2632 / 2670 |  |
| D=64 S=4096 full triton | GPU | 1.729 (0.8 %) | 1.715 (0.1 %) | 1.712 (0.0 %) | -0.8 % | exceeds | -0.2 % | exceeds | 2655 / 2655 / 2655 |  |
| D=64 S=8192 full cudnn | GPU | 3.557 (0.1 %) | 3.557 (0.2 %) | 3.558 (0.2 %) | +0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 full dequant_compiled_flash | GPU | 3.545 (0.6 %) | 3.545 (0.1 %) | 3.546 (0.1 %) | -0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 full dequant_flash | GPU | 3.642 (0.7 %) | 3.637 (0.4 %) | 3.626 (0.7 %) | -0.1 % | within | -0.3 % | within | 2610 / 2610 / 2640 |  |
| D=64 S=8192 full efficient | GPU | 4.792 (1.6 %) | 4.657 (2.2 %) | 4.684 (1.6 %) | -2.8 % | exceeds | +0.6 % | within | 2400 / 2400 / 2400 |  |
| D=64 S=8192 full epilogue_flash | GPU | 3.584 (0.9 %) | 3.563 (0.6 %) | 3.562 (0.4 %) | -0.6 % | within | -0.0 % | within | 2580 / 2640 / 2610 |  |
| D=64 S=8192 full flash | GPU | 3.49 (0.5 %) | 3.488 (1.1 %) | 3.471 (1.5 %) | -0.0 % | within | -0.5 % | within | 2610 / 2595 / 2610 |  |
| D=64 S=8192 full flash_attn | GPU | 3.54 (0.6 %) | 3.53 (0.4 %) | 3.527 (0.2 %) | -0.3 % | within | -0.1 % | within | 2580 / 2565 / 2610 |  |
| D=64 S=8192 full triton | GPU | 3.419 (0.2 %) | 3.415 (0.1 %) | 3.418 (0.1 %) | -0.1 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=512 causal cudnn | GPU | 0.2175 (0.1 %) | 0.2175 (0.1 %) | 0.2176 (0.1 %) | -0.0 % | within | +0.0 % | within | 2670 / 2670 / 2670 |  |
| D=64 S=512 causal dequant_compiled_flash | GPU | 0.2707 (0.0 %) | 0.2698 (0.1 %) | 0.2702 (0.2 %) | -0.4 % | exceeds | +0.1 % | within | 2670 / 2655 / 2655 |  |
| D=64 S=512 causal dequant_flash | GPU | 0.308 (2.7 %) | 0.3077 (2.7 %) | 0.308 (3.0 %) | -0.1 % | within | +0.1 % | within | 2670 / 2655 / 2670 |  |
| D=64 S=512 causal efficient | GPU | 0.2075 (0.7 %) | 0.2076 (0.9 %) | 0.2076 (0.2 %) | +0.0 % | within | +0.0 % | within | 2378 / 2400 / 2400 |  |
| D=64 S=512 causal epilogue_flash | GPU | 0.2356 (0.7 %) | 0.236 (0.5 %) | 0.2361 (0.7 %) | +0.2 % | within | +0.0 % | within | 2640 / 2640 / 2640 |  |
| D=64 S=512 causal flash | GPU | 0.1787 (1.3 %) | 0.1787 (1.8 %) | 0.1787 (1.1 %) | -0.0 % | within | +0.0 % | within | 2640 / 2655 / 2655 |  |
| D=64 S=512 causal flash_attn | GPU | 0.1787 (0.3 %) | 0.1789 (0.4 %) | 0.1789 (0.3 %) | +0.2 % | within | -0.1 % | within | 2655 / 2670 / 2648 |  |
| D=64 S=512 causal math | GPU | 7.516 (0.0 %) | 7.522 (0.0 %) | 7.521 (0.1 %) | +0.1 % | exceeds | -0.0 % | within | 2670 / 2670 / 2670 |  |
| D=64 S=512 causal triton | GPU | 0.163 (3.1 %) | 0.1633 (1.4 %) | 0.1632 (0.9 %) | +0.2 % | within | -0.1 % | within | 2475 / 2528 / 2535 |  |
| D=64 S=1024 causal cudnn | GPU | 0.3468 (0.2 %) | 0.3467 (0.1 %) | 0.3467 (0.1 %) | -0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=1024 causal dequant_compiled_flash | GPU | 0.3732 (0.0 %) | 0.3752 (0.4 %) | 0.3748 (0.4 %) | +0.5 % | exceeds | -0.1 % | within | 2655 / 2610 / 2655 |  |
| D=64 S=1024 causal dequant_flash | GPU | 0.4225 (1.5 %) | 0.4213 (1.4 %) | 0.4213 (1.9 %) | -0.3 % | within | -0.0 % | within | 2655 / 2655 / 2610 |  |
| D=64 S=1024 causal efficient | GPU | 0.3654 (0.2 %) | 0.3654 (0.2 %) | 0.364 (0.1 %) | +0.0 % | within | -0.4 % | exceeds | 2400 / 2415 / 2415 |  |
| D=64 S=1024 causal epilogue_flash | GPU | 0.3451 (0.6 %) | 0.3436 (0.4 %) | 0.3459 (0.8 %) | -0.4 % | within | +0.7 % | within | 2610 / 2655 / 2610 |  |
| D=64 S=1024 causal flash | GPU | 0.2948 (0.2 %) | 0.2948 (0.5 %) | 0.2949 (0.2 %) | +0.0 % | within | +0.0 % | within | 2640 / 2595 / 2610 |  |
| D=64 S=1024 causal flash_attn | GPU | 0.2934 (0.8 %) | 0.2922 (0.7 %) | 0.2922 (0.5 %) | -0.4 % | within | +0.0 % | within | 2655 / 2655 / 2662 |  |
| D=64 S=1024 causal math | GPU | 14.11 (0.0 %) | 14.1 (0.0 %) | 14.1 (0.0 %) | -0.0 % | exceeds | -0.0 % | within | 2670 / 2670 / 2670 |  |
| D=64 S=1024 causal triton | GPU | 0.2873 (0.4 %) | 0.2867 (0.7 %) | 0.2865 (0.2 %) | -0.2 % | within | -0.1 % | within | 2475 / 2490 / 2490 |  |
| D=64 S=2048 causal cudnn | GPU | 0.6138 (0.1 %) | 0.614 (0.1 %) | 0.6134 (0.0 %) | +0.0 % | within | -0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=2048 causal dequant_compiled_flash | GPU | 0.6139 (0.2 %) | 0.6146 (0.3 %) | 0.6153 (0.1 %) | +0.1 % | within | +0.1 % | within | 2625 / 2640 / 2640 |  |
| D=64 S=2048 causal dequant_flash | GPU | 0.6565 (1.4 %) | 0.6564 (1.0 %) | 0.6576 (1.2 %) | -0.0 % | within | +0.2 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=2048 causal efficient | GPU | 0.6714 (0.7 %) | 0.6679 (0.3 %) | 0.6675 (0.1 %) | -0.5 % | within | -0.1 % | within | 2430 / 2445 / 2445 |  |
| D=64 S=2048 causal epilogue_flash | GPU | 0.5853 (0.5 %) | 0.5845 (0.2 %) | 0.5848 (0.2 %) | -0.1 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=2048 causal flash | GPU | 0.5313 (1.6 %) | 0.53 (1.1 %) | 0.53 (1.2 %) | -0.3 % | within | +0.0 % | within | 2625 / 2655 / 2655 |  |
| D=64 S=2048 causal flash_attn | GPU | 0.5309 (0.1 %) | 0.5307 (0.3 %) | 0.5313 (0.1 %) | -0.0 % | within | +0.1 % | within | 2670 / 2670 / 2670 |  |
| D=64 S=2048 causal math | GPU | 27.34 (0.4 %) | 27.42 (0.4 %) | 27.32 (0.0 %) | +0.3 % | within | -0.4 % | exceeds | 2670 / 2670 / 2670 |  |
| D=64 S=2048 causal triton | GPU | 0.5349 (0.0 %) | 0.5344 (0.2 %) | 0.5341 (0.1 %) | -0.1 % | within | -0.1 % | within | 2655 / 2505 / 2505 |  |
| D=64 S=4096 causal cudnn | GPU | 1.146 (0.2 %) | 1.146 (0.0 %) | 1.146 (0.0 %) | +0.0 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 causal dequant_compiled_flash | GPU | 1.095 (0.1 %) | 1.098 (0.1 %) | 1.097 (0.1 %) | +0.2 % | exceeds | -0.1 % | within | 2640 / 2640 / 2655 |  |
| D=64 S=4096 causal dequant_flash | GPU | 1.14 (0.6 %) | 1.14 (0.5 %) | 1.14 (0.7 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 causal efficient | GPU | 1.281 (3.9 %) | 1.28 (4.0 %) | 1.279 (3.3 %) | -0.0 % | within | -0.1 % | within | 2445 / 2445 / 2445 |  |
| D=64 S=4096 causal epilogue_flash | GPU | 1.068 (0.4 %) | 1.068 (0.2 %) | 1.068 (0.1 %) | +0.0 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 causal flash | GPU | 1.015 (0.2 %) | 1.014 (0.2 %) | 1.013 (0.1 %) | -0.2 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=4096 causal flash_attn | GPU | 1.01 (0.1 %) | 1.011 (0.1 %) | 1.01 (0.3 %) | +0.1 % | within | -0.1 % | within | 2655 / 2670 / 2670 |  |
| D=64 S=4096 causal math | GPU | 57.79 (0.0 %) | 57.78 (0.0 %) | 57.77 (0.1 %) | -0.0 % | exceeds | -0.0 % | within | 2655 / 2670 / 2670 |  |
| D=64 S=4096 causal triton | GPU | 1.031 (0.1 %) | 1.03 (0.3 %) | 1.027 (0.2 %) | -0.1 % | within | -0.3 % | exceeds | 2520 / 2535 / 2535 |  |
| D=64 S=8192 causal cudnn | GPU | 2.211 (0.2 %) | 2.212 (0.2 %) | 2.211 (0.2 %) | +0.0 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal dequant_compiled_flash | GPU | 2.063 (0.2 %) | 2.066 (0.4 %) | 2.068 (0.2 %) | +0.1 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal dequant_flash | GPU | 2.112 (0.2 %) | 2.112 (0.2 %) | 2.114 (0.5 %) | +0.0 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal efficient | GPU | 2.512 (0.3 %) | 2.503 (0.5 %) | 2.501 (0.1 %) | -0.3 % | within | -0.1 % | within | 2445 / 2460 / 2655 |  |
| D=64 S=8192 causal epilogue_flash | GPU | 2.036 (0.2 %) | 2.037 (0.1 %) | 2.037 (0.5 %) | +0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal flash | GPU | 1.981 (0.5 %) | 1.982 (0.5 %) | 1.981 (0.1 %) | +0.1 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=64 S=8192 causal flash_attn | GPU | 1.973 (0.0 %) | 1.973 (0.2 %) | 1.975 (0.4 %) | +0.0 % | within | +0.1 % | within | 2640 / 2595 / 2640 |  |
| D=64 S=8192 causal triton | GPU | 2.022 (0.5 %) | 2.025 (0.4 %) | 2.017 (0.1 %) | +0.1 % | within | -0.4 % | exceeds | 2535 / 2550 / 2550 |  |
| D=128 S=512 full cudnn | GPU | 0.4967 (0.2 %) | 0.495 (0.6 %) | 0.4955 (0.1 %) | -0.3 % | within | +0.1 % | within | 2655 / 2670 / 2670 |  |
| D=128 S=512 full dequant_compiled_flash | GPU | 0.6641 (0.5 %) | 0.6633 (0.1 %) | 0.6625 (0.3 %) | -0.1 % | within | -0.1 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=512 full dequant_flash | GPU | 0.7511 (1.6 %) | 0.7449 (3.3 %) | 0.7451 (0.4 %) | -0.8 % | within | +0.0 % | within | 2580 / 2610 / 2595 |  |
| D=128 S=512 full efficient | GPU | 0.7483 (0.1 %) | 0.7441 (0.4 %) | 0.7447 (0.5 %) | -0.6 % | exceeds | +0.1 % | within | 2655 / 2580 / 2580 |  |
| D=128 S=512 full epilogue_flash | GPU | 0.6114 (0.9 %) | 0.6139 (0.9 %) | 0.608 (1.6 %) | +0.4 % | within | -1.0 % | within | 2565 / 2580 / 2610 |  |
| D=128 S=512 full flash | GPU | 0.4858 (2.5 %) | 0.4802 (2.9 %) | 0.4857 (2.8 %) | -1.1 % | within | +1.1 % | within | 2535 / 2595 / 2535 |  |
| D=128 S=512 full flash_attn | GPU | 0.466 (0.1 %) | 0.4646 (0.5 %) | 0.4655 (0.5 %) | -0.3 % | within | +0.2 % | within | 2625 / 2625 / 2625 |  |
| D=128 S=512 full math | GPU | 7.361 (0.1 %) | 7.363 (0.1 %) | 7.365 (0.0 %) | +0.0 % | within | +0.0 % | within | 2655 / 2670 / 2670 |  |
| D=128 S=512 full triton | GPU | 0.4771 (0.1 %) | 0.4764 (0.6 %) | 0.4738 (0.5 %) | -0.1 % | within | -0.6 % | within | 2610 / 2610 / 2610 |  |
| D=128 S=1024 full cudnn | GPU | 0.9265 (0.2 %) | 0.9261 (0.2 %) | 0.9265 (0.1 %) | -0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=1024 full dequant_compiled_flash | GPU | 1.086 (0.1 %) | 1.086 (0.1 %) | 1.085 (0.1 %) | +0.0 % | within | -0.1 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=1024 full dequant_flash | GPU | 1.166 (1.0 %) | 1.167 (0.1 %) | 1.166 (0.6 %) | +0.0 % | within | -0.1 % | within | 2580 / 2610 / 2640 |  |
| D=128 S=1024 full efficient | GPU | 1.41 (1.3 %) | 1.397 (0.3 %) | 1.398 (0.3 %) | -1.0 % | within | +0.1 % | within | 2655 / 2670 / 2625 |  |
| D=128 S=1024 full epilogue_flash | GPU | 1.033 (1.3 %) | 1.029 (0.5 %) | 1.033 (1.7 %) | -0.5 % | within | +0.4 % | within | 2610 / 2655 / 2640 |  |
| D=128 S=1024 full flash | GPU | 0.908 (1.2 %) | 0.9083 (1.0 %) | 0.9082 (2.5 %) | +0.0 % | within | -0.0 % | within | 2565 / 2565 / 2595 |  |
| D=128 S=1024 full flash_attn | GPU | 0.8815 (0.3 %) | 0.8813 (0.0 %) | 0.882 (0.1 %) | -0.0 % | within | +0.1 % | within | 2655 / 2670 / 2670 |  |
| D=128 S=1024 full math | GPU | 13.04 (0.1 %) | 13.05 (0.1 %) | 13.06 (0.0 %) | +0.1 % | exceeds | +0.0 % | within | 2670 / 2670 / 2670 |  |
| D=128 S=1024 full triton | GPU | 0.8967 (0.5 %) | 0.889 (0.0 %) | 0.8927 (0.5 %) | -0.9 % | exceeds | +0.4 % | within | 2625 / 2640 / 2640 |  |
| D=128 S=2048 full cudnn | GPU | 1.796 (0.1 %) | 1.795 (0.2 %) | 1.796 (0.2 %) | -0.0 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=2048 full dequant_compiled_flash | GPU | 1.944 (0.0 %) | 1.946 (0.1 %) | 1.946 (0.1 %) | +0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=2048 full dequant_flash | GPU | 2.024 (0.0 %) | 2.016 (0.8 %) | 2.019 (0.8 %) | -0.4 % | within | +0.2 % | within | 2640 / 2655 / 2655 |  |
| D=128 S=2048 full efficient | GPU | 2.739 (0.2 %) | 2.739 (0.0 %) | 2.739 (0.1 %) | +0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=2048 full epilogue_flash | GPU | 1.893 (0.4 %) | 1.886 (0.2 %) | 1.883 (0.0 %) | -0.4 % | within | -0.1 % | within | 2640 / 2655 / 2655 |  |
| D=128 S=2048 full flash | GPU | 1.752 (0.9 %) | 1.752 (0.1 %) | 1.753 (0.3 %) | -0.0 % | within | +0.1 % | within | 2595 / 2625 / 2640 |  |
| D=128 S=2048 full flash_attn | GPU | 1.737 (0.1 %) | 1.737 (0.2 %) | 1.737 (0.1 %) | -0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=2048 full math | GPU | 24.44 (0.1 %) | 24.48 (0.1 %) | 24.48 (0.3 %) | +0.2 % | exceeds | -0.0 % | within | 2670 / 2670 / 2670 |  |
| D=128 S=2048 full triton | GPU | 1.733 (0.6 %) | 1.725 (0.1 %) | 1.725 (0.1 %) | -0.5 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=4096 full cudnn | GPU | 3.537 (0.3 %) | 3.537 (0.1 %) | 3.532 (0.2 %) | +0.0 % | within | -0.2 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=4096 full dequant_compiled_flash | GPU | 3.678 (0.1 %) | 3.675 (0.1 %) | 3.675 (0.2 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=4096 full dequant_flash | GPU | 3.742 (0.3 %) | 3.742 (0.4 %) | 3.743 (0.2 %) | -0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=4096 full efficient | GPU | 5.423 (0.1 %) | 5.418 (0.1 %) | 5.423 (0.1 %) | -0.1 % | within | +0.1 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=4096 full epilogue_flash | GPU | 3.616 (0.1 %) | 3.612 (0.1 %) | 3.609 (0.2 %) | -0.1 % | within | -0.1 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=4096 full flash | GPU | 3.483 (0.9 %) | 3.48 (0.1 %) | 3.474 (0.1 %) | -0.1 % | within | -0.1 % | exceeds | 2648 / 2655 / 2655 |  |
| D=128 S=4096 full flash_attn | GPU | 3.458 (0.2 %) | 3.455 (0.1 %) | 3.455 (0.1 %) | -0.1 % | within | +0.0 % | within | 2670 / 2655 / 2655 |  |
| D=128 S=4096 full math | GPU | 47.04 (0.1 %) | 47.09 (0.2 %) | 47.1 (0.2 %) | +0.1 % | within | +0.0 % | within | 2670 / 2670 / 2670 |  |
| D=128 S=4096 full triton | GPU | 3.416 (0.1 %) | 3.413 (0.1 %) | 3.414 (0.2 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 full cudnn | GPU | 7.029 (0.3 %) | 7.028 (0.0 %) | 7.015 (0.1 %) | -0.0 % | within | -0.2 % | exceeds | 2655 / 2655 / 2655 |  |
| D=128 S=8192 full dequant_compiled_flash | GPU | 7.138 (0.1 %) | 7.133 (0.0 %) | 7.135 (0.2 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 full dequant_flash | GPU | 7.201 (0.2 %) | 7.195 (0.1 %) | 7.196 (0.4 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 full efficient | GPU | 10.75 (0.2 %) | 10.74 (0.3 %) | 10.74 (0.1 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 full epilogue_flash | GPU | 7.072 (0.1 %) | 7.067 (0.0 %) | 7.064 (0.2 %) | -0.1 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 full flash | GPU | 6.938 (0.1 %) | 6.933 (0.0 %) | 6.932 (0.2 %) | -0.1 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 full flash_attn | GPU | 6.901 (0.1 %) | 6.895 (0.1 %) | 6.893 (0.1 %) | -0.1 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 full triton | GPU | 6.801 (0.1 %) | 6.797 (0.1 %) | 6.797 (0.1 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=512 causal cudnn | GPU | 0.3431 (0.1 %) | 0.3431 (0.1 %) | 0.3432 (0.1 %) | +0.0 % | within | +0.0 % | within | 2670 / 2670 / 2670 |  |
| D=128 S=512 causal dequant_compiled_flash | GPU | 0.5321 (0.2 %) | 0.5324 (0.2 %) | 0.5324 (0.1 %) | +0.0 % | within | +0.0 % | within | 2655 / 2670 / 2610 |  |
| D=128 S=512 causal dequant_flash | GPU | 0.5858 (2.0 %) | 0.5865 (1.9 %) | 0.5866 (2.2 %) | +0.1 % | within | +0.0 % | within | 2625 / 2625 / 2565 |  |
| D=128 S=512 causal efficient | GPU | 0.485 (0.5 %) | 0.4833 (0.4 %) | 0.4849 (0.5 %) | -0.4 % | within | +0.3 % | within | 2655 / 2580 / 2670 |  |
| D=128 S=512 causal epilogue_flash | GPU | 0.4574 (0.3 %) | 0.4564 (0.2 %) | 0.457 (0.2 %) | -0.2 % | within | +0.1 % | within | 2550 / 2655 / 2640 |  |
| D=128 S=512 causal flash | GPU | 0.3181 (0.1 %) | 0.3175 (0.0 %) | 0.3175 (0.1 %) | -0.2 % | exceeds | +0.0 % | within | 2490 / 2520 / 2505 |  |
| D=128 S=512 causal flash_attn | GPU | 0.3144 (0.0 %) | 0.3141 (0.1 %) | 0.3142 (0.1 %) | -0.1 % | exceeds | +0.0 % | within | 2580 / 2610 / 2670 |  |
| D=128 S=512 causal math | GPU | 8.531 (0.0 %) | 8.534 (0.0 %) | 8.536 (0.1 %) | +0.0 % | exceeds | +0.0 % | within | 2655 / 2670 / 2670 |  |
| D=128 S=512 causal triton | GPU | 0.3468 (0.2 %) | 0.3468 (0.1 %) | 0.3469 (0.1 %) | +0.0 % | within | +0.0 % | within | 2625 / 2640 / 2655 |  |
| D=128 S=1024 causal cudnn | GPU | 0.5742 (0.3 %) | 0.5744 (0.5 %) | 0.5745 (0.2 %) | +0.0 % | within | +0.0 % | within | 2595 / 2655 / 2655 |  |
| D=128 S=1024 causal dequant_compiled_flash | GPU | 0.713 (0.4 %) | 0.7126 (0.1 %) | 0.7098 (0.7 %) | -0.1 % | within | -0.4 % | within | 2595 / 2550 / 2565 |  |
| D=128 S=1024 causal dequant_flash | GPU | 0.7942 (1.1 %) | 0.7905 (1.3 %) | 0.7924 (0.3 %) | -0.5 % | within | +0.2 % | within | 2520 / 2565 / 2550 |  |
| D=128 S=1024 causal efficient | GPU | 0.8252 (1.0 %) | 0.8205 (1.4 %) | 0.8209 (1.5 %) | -0.6 % | within | +0.1 % | within | 2610 / 2640 / 2655 |  |
| D=128 S=1024 causal epilogue_flash | GPU | 0.6535 (1.0 %) | 0.6513 (1.3 %) | 0.6512 (1.1 %) | -0.3 % | within | -0.0 % | within | 2535 / 2490 / 2550 |  |
| D=128 S=1024 causal flash | GPU | 0.541 (0.6 %) | 0.5378 (0.2 %) | 0.5378 (0.2 %) | -0.6 % | within | -0.0 % | within | 2475 / 2490 / 2475 |  |
| D=128 S=1024 causal flash_attn | GPU | 0.5159 (0.6 %) | 0.5144 (0.7 %) | 0.5119 (0.7 %) | -0.3 % | within | -0.5 % | within | 2580 / 2565 / 2670 |  |
| D=128 S=1024 causal math | GPU | 15.34 (0.1 %) | 15.35 (0.1 %) | 15.36 (0.1 %) | +0.1 % | within | +0.0 % | within | 2670 / 2670 / 2670 |  |
| D=128 S=1024 causal triton | GPU | 0.5788 (0.1 %) | 0.5779 (0.2 %) | 0.577 (0.2 %) | -0.2 % | within | -0.2 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=2048 causal cudnn | GPU | 1.045 (0.1 %) | 1.044 (0.1 %) | 1.044 (0.0 %) | -0.0 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=2048 causal dequant_compiled_flash | GPU | 1.173 (0.2 %) | 1.172 (0.1 %) | 1.171 (0.4 %) | -0.1 % | within | -0.1 % | within | 2565 / 2625 / 2610 |  |
| D=128 S=2048 causal dequant_flash | GPU | 1.254 (0.9 %) | 1.243 (2.7 %) | 1.245 (1.1 %) | -0.9 % | within | +0.2 % | within | 2535 / 2565 / 2550 |  |
| D=128 S=2048 causal efficient | GPU | 1.494 (0.1 %) | 1.495 (0.0 %) | 1.495 (0.0 %) | +0.0 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=2048 causal epilogue_flash | GPU | 1.121 (0.5 %) | 1.117 (1.0 %) | 1.116 (0.6 %) | -0.3 % | within | -0.1 % | within | 2535 / 2550 / 2565 |  |
| D=128 S=2048 causal flash | GPU | 1.002 (0.4 %) | 0.9927 (2.4 %) | 0.9959 (0.9 %) | -0.9 % | within | +0.3 % | within | 2520 / 2580 / 2550 |  |
| D=128 S=2048 causal flash_attn | GPU | 0.9607 (2.8 %) | 0.9603 (2.5 %) | 0.9604 (2.3 %) | -0.0 % | within | +0.0 % | within | 2595 / 2670 / 2610 |  |
| D=128 S=2048 causal math | GPU | 29.1 (0.1 %) | 29.13 (0.0 %) | 29.16 (0.1 %) | +0.1 % | exceeds | +0.1 % | within | 2670 / 2670 / 2670 |  |
| D=128 S=2048 causal triton | GPU | 1.063 (0.0 %) | 1.062 (1.2 %) | 1.063 (0.1 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=4096 causal cudnn | GPU | 1.985 (0.1 %) | 1.983 (0.0 %) | 1.983 (0.0 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=4096 causal dequant_compiled_flash | GPU | 2.101 (0.3 %) | 2.086 (0.8 %) | 2.077 (0.8 %) | -0.7 % | within | -0.5 % | within | 2610 / 2610 / 2640 |  |
| D=128 S=4096 causal dequant_flash | GPU | 2.151 (2.3 %) | 2.145 (2.0 %) | 2.145 (2.2 %) | -0.3 % | within | -0.0 % | within | 2565 / 2550 / 2580 |  |
| D=128 S=4096 causal efficient | GPU | 2.862 (0.1 %) | 2.859 (0.0 %) | 2.859 (0.0 %) | -0.1 % | within | +0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=4096 causal epilogue_flash | GPU | 2.051 (1.8 %) | 2.051 (1.9 %) | 2.044 (2.0 %) | +0.0 % | within | -0.4 % | within | 2535 / 2595 / 2565 |  |
| D=128 S=4096 causal flash | GPU | 1.929 (0.3 %) | 1.918 (0.1 %) | 1.908 (0.2 %) | -0.5 % | exceeds | -0.5 % | exceeds | 2565 / 2580 / 2580 |  |
| D=128 S=4096 causal flash_attn | GPU | 1.861 (0.1 %) | 1.86 (0.5 %) | 1.855 (0.3 %) | -0.1 % | within | -0.2 % | within | 2610 / 2640 / 2625 |  |
| D=128 S=4096 causal math | GPU | 60.71 (0.0 %) | 60.75 (0.1 %) | 60.81 (0.1 %) | +0.1 % | within | +0.1 % | exceeds | 2670 / 2670 / 2670 |  |
| D=128 S=4096 causal triton | GPU | 2.048 (0.1 %) | 2.046 (0.2 %) | 2.045 (0.3 %) | -0.1 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 causal cudnn | GPU | 3.857 (0.2 %) | 3.85 (0.2 %) | 3.85 (0.1 %) | -0.2 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 causal dequant_compiled_flash | GPU | 3.959 (1.4 %) | 3.954 (1.2 %) | 3.936 (1.7 %) | -0.1 % | within | -0.5 % | within | 2610 / 2610 / 2625 |  |
| D=128 S=8192 causal dequant_flash | GPU | 4.01 (1.0 %) | 4.009 (1.1 %) | 4.002 (1.2 %) | -0.0 % | within | -0.2 % | within | 2565 / 2595 / 2565 |  |
| D=128 S=8192 causal efficient | GPU | 5.582 (0.1 %) | 5.577 (0.1 %) | 5.574 (0.0 %) | -0.1 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |
| D=128 S=8192 causal epilogue_flash | GPU | 3.851 (0.2 %) | 3.837 (0.7 %) | 3.829 (0.9 %) | -0.4 % | within | -0.2 % | within | 2565 / 2565 / 2565 |  |
| D=128 S=8192 causal flash | GPU | 3.765 (0.9 %) | 3.738 (0.9 %) | 3.734 (1.8 %) | -0.7 % | within | -0.1 % | within | 2610 / 2610 / 2618 |  |
| D=128 S=8192 causal flash_attn | GPU | 3.72 (0.6 %) | 3.709 (0.6 %) | 3.686 (0.7 %) | -0.3 % | within | -0.6 % | within | 2625 / 2625 / 2640 |  |
| D=128 S=8192 causal triton | GPU | 3.99 (0.2 %) | 3.986 (0.1 %) | 3.985 (0.0 %) | -0.1 % | within | -0.0 % | within | 2655 / 2655 / 2655 |  |

Not timed on any side (the script skips them): D=64 S=8192 full math (skipped: S x S scores of 4.0 GiB); D=64 S=8192 causal math (skipped: S x S scores of 4.0 GiB); D=128 S=8192 full math (skipped: S x S scores of 4.0 GiB); D=128 S=8192 causal math (skipped: S x S scores of 4.0 GiB)

### E16, the paper's summary: FKL mma_raw against the best baseline (scripts/e16_summary.py on each side: TFLOP/s; fraction FKL / best per run, median [smallest-largest])

| D | mask | S | mma_raw TFLOP/s pub / ref / new | best baseline pub / ref / new | fraction pub | fraction ref | fraction new | clocks pub / ref / new |
|---|---|---|---|---|---|---|---|---|
| 64 | full | 512 | 133.9 / 134.7 / 135.3 | triton / triton / triton | 0.930 [0.924-0.960] | 0.933 [0.931-0.969] | 0.940 [0.930-0.963] | yes / yes / yes |
| 64 | full | 1024 | 147.4 / 148.5 / 149.5 | triton / triton / triton | 0.958 [0.958-0.963] | 0.964 [0.964-0.969] | 0.972 [0.959-0.976] | - / - / - |
| 64 | full | 2048 | 155.4 / 156.5 / 156.4 | triton / triton / triton | 0.990 [0.987-1.001] | 0.986 [0.986-0.995] | 0.984 [0.979-0.990] | - / - / - |
| 64 | full | 4096 | 160.9 / 161.1 / 161.0 | triton / triton / triton | 1.013 [1.012-1.016] | 1.006 [1.004-1.006] | 1.003 [1.003-1.004] | - / - / - |
| 64 | full | 8192 | 162.7 / 163.0 / 162.9 | triton / triton / triton | 1.012 [1.012-1.016] | 1.013 [1.011-1.014] | 1.013 [1.012-1.013] | - / - / - |
| 64 | causal | 512 | 105.1 / 105.0 / 105.8 | triton / triton / triton | 0.997 [0.996-1.017] | 1.003 [0.997-1.007] | 1.003 [0.994-1.019] | yes / yes / yes |
| 64 | causal | 1024 | 116.9 / 117.8 / 119.4 | triton / triton / triton | 0.978 [0.977-0.985] | 0.983 [0.978-0.996] | 0.997 [0.978-1.008] | - / yes / - |
| 64 | causal | 2048 | 131.6 / 132.1 / 132.8 | flash_attn / flash_attn / flash | 1.015 [1.015-1.031] | 1.018 [1.014-1.033] | 1.023 [1.014-1.034] | yes / yes / yes |
| 64 | causal | 4096 | 140.1 / 140.3 / 140.4 | flash_attn / flash_attn / flash_attn | 1.030 [1.029-1.032] | 1.032 [1.029-1.033] | 1.032 [1.029-1.033] | - / - / - |
| 64 | causal | 8192 | 144.4 / 144.5 / 144.4 | flash_attn / flash_attn / flash_attn | 1.036 [1.036-1.038] | 1.038 [1.035-1.039] | 1.038 [1.036-1.040] | - / yes / yes |
| 128 | full | 512 | 135.1 / 135.6 / 138.3 | flash_attn / flash_attn / flash_attn | 0.916 [0.913-0.943] | 0.920 [0.910-0.945] | 0.933 [0.920-0.947] | yes / yes / yes |
| 128 | full | 1024 | 146.5 / 146.5 / 145.6 | flash_attn / flash_attn / flash_attn | 0.940 [0.925-0.955] | 0.939 [0.923-0.966] | 0.934 [0.928-0.962] | yes / yes / yes |
| 128 | full | 2048 | 150.8 / 152.1 / 152.8 | triton / triton / triton | 0.953 [0.946-0.959] | 0.954 [0.943-0.960] | 0.958 [0.949-0.977] | yes / yes / yes |
| 128 | full | 4096 | 154.9 / 156.3 / 155.8 | triton / triton / triton | 0.962 [0.962-0.967] | 0.970 [0.962-0.977] | 0.967 [0.966-0.984] | yes / yes / yes |
| 128 | full | 8192 | 157.7 / 158.6 / 159.5 | triton / triton / triton | 0.976 [0.975-0.976] | 0.981 [0.976-0.981] | 0.986 [0.970-0.987] | yes / yes / yes |
| 128 | causal | 512 | 108.0 / 108.4 / 108.6 | flash_attn / flash_attn / flash_attn | 0.988 [0.977-0.997] | 0.991 [0.981-1.000] | 0.994 [0.981-1.001] | yes / yes / yes |
| 128 | causal | 1024 | 125.2 / 126.0 / 125.9 | flash_attn / flash_attn / flash_attn | 0.938 [0.932-0.945] | 0.943 [0.933-0.944] | 0.938 [0.933-0.949] | yes / yes / yes |
| 128 | causal | 2048 | 137.1 / 137.1 / 137.9 | flash_attn / flash_attn / flash_attn | 0.958 [0.951-1.001] | 0.958 [0.957-1.005] | 0.964 [0.959-1.003] | yes / yes / yes |
| 128 | causal | 4096 | 144.0 / 144.5 / 145.3 | flash_attn / flash_attn / flash_attn | 0.975 [0.975-0.979] | 0.975 [0.973-0.978] | 0.981 [0.973-0.982] | yes / yes / yes |
| 128 | causal | 8192 | 148.9 / 149.3 / 149.2 | flash_attn / flash_attn / flash_attn | 1.005 [1.001-1.014] | 1.008 [1.002-1.014] | 1.005 [0.994-1.013] | - / - / - |

### E16: the order of the variants in every round (the position column; round r ran with E16_ROTATE = r - 1 and --rotate r - 1)

Program order in each round: round 1 FKL reference, FKL cleaned, Python reference, Python cleaned; round 2 Python cleaned, Python reference, FKL cleaned, FKL reference; round 3 as round 1.

FKL variant order per configuration (the same in the reference and the cleaned runs of a round: True; the same as the published round of the same index: True):

| configuration | round 1 | round 2 | round 3 |
|---|---|---|---|
| D=64 S=512 full | raw i8F i8U f8F f8U epF epS simt | i8U i8F f8U f8F epS epF raw simt | f8F f8U epF epS raw i8F i8U simt |
| D=64 S=1024 full | f8F f8U epF epS raw i8F i8U simt | epS epF raw i8U i8F f8U f8F simt | raw i8F i8U f8F f8U epF epS simt |
| D=64 S=2048 full | raw i8F i8U f8F f8U epF epS simt | i8U i8F f8U f8F epS epF raw simt | f8F f8U epF epS raw i8F i8U simt |
| D=64 S=4096 full | f8F f8U epF epS raw i8F i8U | epS epF raw i8U i8F f8U f8F | raw i8F i8U f8F f8U epF epS |
| D=64 S=8192 full | raw i8F i8U f8F f8U epF epS | i8U i8F f8U f8F epS epF raw | f8F f8U epF epS raw i8F i8U |
| D=64 S=512 causal | i8U i8F f8U f8F epS epF raw simt | f8F f8U epF epS raw i8F i8U simt | epS epF raw i8U i8F f8U f8F simt |
| D=64 S=1024 causal | epS epF raw i8U i8F f8U f8F simt | raw i8F i8U f8F f8U epF epS simt | i8U i8F f8U f8F epS epF raw simt |
| D=64 S=2048 causal | i8U i8F f8U f8F epS epF raw simt | f8F f8U epF epS raw i8F i8U simt | epS epF raw i8U i8F f8U f8F simt |
| D=64 S=4096 causal | epS epF raw i8U i8F f8U f8F | raw i8F i8U f8F f8U epF epS | i8U i8F f8U f8F epS epF raw |
| D=64 S=8192 causal | i8U i8F f8U f8F epS epF raw | f8F f8U epF epS raw i8F i8U | epS epF raw i8U i8F f8U f8F |
| D=128 S=512 full | f8F f8U epF epS raw i8F i8U simt | epS epF raw i8U i8F f8U f8F simt | raw i8F i8U f8F f8U epF epS simt |
| D=128 S=1024 full | raw i8F i8U f8F f8U epF epS simt | i8U i8F f8U f8F epS epF raw simt | f8F f8U epF epS raw i8F i8U simt |
| D=128 S=2048 full | f8F f8U epF epS raw i8F i8U simt | epS epF raw i8U i8F f8U f8F simt | raw i8F i8U f8F f8U epF epS simt |
| D=128 S=4096 full | raw i8F i8U f8F f8U epF epS | i8U i8F f8U f8F epS epF raw | f8F f8U epF epS raw i8F i8U |
| D=128 S=8192 full | f8F f8U epF epS raw i8F i8U | epS epF raw i8U i8F f8U f8F | raw i8F i8U f8F f8U epF epS |
| D=128 S=512 causal | epS epF raw i8U i8F f8U f8F simt | raw i8F i8U f8F f8U epF epS simt | i8U i8F f8U f8F epS epF raw simt |
| D=128 S=1024 causal | i8U i8F f8U f8F epS epF raw simt | f8F f8U epF epS raw i8F i8U simt | epS epF raw i8U i8F f8U f8F simt |
| D=128 S=2048 causal | epS epF raw i8U i8F f8U f8F simt | raw i8F i8U f8F f8U epF epS simt | i8U i8F f8U f8F epS epF raw simt |
| D=128 S=4096 causal | i8U i8F f8U f8F epS epF raw | f8F f8U epF epS raw i8F i8U | epS epF raw i8U i8F f8U f8F |
| D=128 S=8192 causal | epS epF raw i8U i8F f8U f8F | raw i8F i8U f8F f8U epF epS | i8U i8F f8U f8F epS epF raw |

Python variant order per configuration (the same in the reference and the cleaned runs of a round: True; the same as the published round of the same index: True):

| configuration | round 1 | round 2 | round 3 |
|---|---|---|---|
| D=64 S=512 full | flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math flash_attn triton | cudnn efficient math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash | efficient math flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn |
| D=64 S=1024 full | efficient math flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn | math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient | flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math |
| D=64 S=2048 full | flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math | triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient math flash_attn | flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math flash_attn triton |
| D=64 S=4096 full | flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math flash_attn triton | cudnn efficient math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash | efficient math flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn |
| D=64 S=8192 full | math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient | math triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient flash_attn | math flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient flash_attn triton |
| D=64 S=512 causal | cudnn efficient math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash | efficient math flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn | math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient |
| D=64 S=1024 causal | math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient | flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math | triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient math flash_attn |
| D=64 S=2048 causal | triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient math flash_attn | flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math flash_attn triton | cudnn efficient math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash |
| D=64 S=4096 causal | cudnn efficient math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash | efficient math flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn | math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient |
| D=64 S=8192 causal | math triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient flash_attn | math flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient flash_attn triton | math cudnn efficient flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash |
| D=128 S=512 full | flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math | triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient math flash_attn | flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math flash_attn triton |
| D=128 S=1024 full | flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math flash_attn triton | cudnn efficient math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash | efficient math flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn |
| D=128 S=2048 full | efficient math flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn | math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient | flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math |
| D=128 S=4096 full | flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math | triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient math flash_attn | flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math flash_attn triton |
| D=128 S=8192 full | math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient | math triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient flash_attn | math flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient flash_attn triton |
| D=128 S=512 causal | triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient math flash_attn | flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math flash_attn triton | cudnn efficient math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash |
| D=128 S=1024 causal | cudnn efficient math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash | efficient math flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn | math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient |
| D=128 S=2048 causal | math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient | flash_attn triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math | triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient math flash_attn |
| D=128 S=4096 causal | triton dequant_compiled_flash dequant_flash epilogue_flash flash cudnn efficient math flash_attn | flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient math flash_attn triton | cudnn efficient math flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash |
| D=128 S=8192 causal | math triton flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient flash_attn | math flash epilogue_flash dequant_flash dequant_compiled_flash cudnn efficient flash_attn triton | math cudnn efficient flash_attn triton dequant_compiled_flash dequant_flash epilogue_flash flash |

### E16, the FKL program: preparation of one call (cleaned build; prep_us_per_call, host time of building one argument set and releasing the old one, us; median of the three rounds)

| variant | smallest | median | largest (over all configurations) |
|---|---|---|---|
| mma_epilogue_fused | 0.002959 | 0.004744 | 0.0125 |
| mma_epilogue_separate | 0.002476 | 0.004494 | 0.01217 |
| mma_fp8kv_fused | 0.002471 | 0.004724 | 0.01367 |
| mma_fp8kv_unfused | 0.002546 | 0.004089 | 0.01267 |
| mma_int8kv_fused | 0.002464 | 0.004859 | 0.01333 |
| mma_int8kv_unfused | 0.002457 | 0.004267 | 0.0115 |
| mma_raw | 0.002245 | 0.003999 | 0.01233 |
| simt_raw | 0.01167 | 0.025 | 0.035 |

### E2, the FKL program (e2_fkl_chain), GPU time per launch (ms; mean of 100; published: runs 2, 3, 4; window D)

| k, pass | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| 2 warm | GPU | 0.02115 (5.6 %) | 0.02038 (0.8 %) | 0.02029 (0.3 %) | -3.6 % | within FLAG | -0.5 % | within |  |
| 2 L2-flushed | GPU | 0.02656 (5.9 %) | 0.02586 (0.4 %) | 0.02592 (0.1 %) | -2.6 % | within | +0.2 % | within |  |
| 10 warm | GPU | 0.02123 (3.8 %) | 0.02044 (3.4 %) | 0.02043 (0.6 %) | -3.7 % | within FLAG | -0.1 % | within |  |
| 10 L2-flushed | GPU | 0.02691 (4.3 %) | 0.02625 (0.5 %) | 0.02627 (0.1 %) | -2.5 % | within | +0.1 % | within |  |
| 50 warm | GPU | 0.02285 (4.2 %) | 0.02279 (3.6 %) | 0.02274 (0.3 %) | -0.3 % | within | -0.2 % | within |  |
| 50 L2-flushed | GPU | 0.02798 (3.6 %) | 0.02781 (0.2 %) | 0.02782 (0.2 %) | -0.6 % | within | +0.0 % | within |  |
| 100 warm | GPU | 0.02545 (4.9 %) | 0.02559 (3.5 %) | 0.02555 (0.3 %) | +0.5 % | within | -0.1 % | within |  |
| 100 L2-flushed | GPU | 0.03032 (1.7 %) | 0.03006 (0.4 %) | 0.03 (0.3 %) | -0.9 % | within | -0.2 % | within |  |
| 500 warm | GPU | 0.06429 (0.3 %) | 0.06455 (0.2 %) | 0.06464 (0.2 %) | +0.4 % | exceeds | +0.1 % | within |  |
| 500 L2-flushed | GPU | 0.06903 (0.6 %) | 0.06927 (0.1 %) | 0.06924 (0.1 %) | +0.3 % | within | -0.0 % | within |  |
| 5002 warm | GPU | 0.5265 (0.7 %) | 0.5243 (2.5 %) | 0.5316 (2.0 %) | -0.4 % | within | +1.4 % | within |  |
| 5002 L2-flushed | GPU | 0.5445 (2.5 %) | 0.5452 (3.3 %) | 0.5264 (0.1 %) | +0.1 % | within | -3.5 % | exceeds FLAG | order of the passes, as in window C (-3.0 % there): the cleaned flushed pass follows the kernel-only pass of the same k (100 spins of 5 ms), the reference's follows the warm pass; the clock log does not resolve it: every E2 FKL run of both builds samples 2,670-2,685 MHz, and the k = 5,002 flushed pass lasts about 80 ms, about one 100 ms sample; the power medians differ (reference 126-196 W, cleaned 112-113 W: most of a cleaned run is the one-thread spin); not isolated |

### E2, the FKL program in windows C and D (the same binaries; medians of three runs each, ms)

| k | warm ref C / D | warm cleaned C / D | flushed ref C / D | flushed cleaned C / D | kernel-only cleaned C / D |
|---|---|---|---|---|---|
| 2 | 0.02069 / 0.02038 | 0.02028 / 0.02029 | 0.02592 / 0.02586 | 0.02586 / 0.02592 | 0.02048 / 0.02048 |
| 10 | 0.02042 / 0.02044 | 0.02037 / 0.02043 | 0.02626 / 0.02625 | 0.02618 / 0.02627 | 0.02048 / 0.02048 |
| 50 | 0.02276 / 0.02279 | 0.02271 / 0.02274 | 0.02776 / 0.02781 | 0.02774 / 0.02782 | 0.02286 / 0.02283 |
| 100 | 0.02544 / 0.02559 | 0.02548 / 0.02555 | 0.02983 / 0.03006 | 0.02992 / 0.03 | 0.02571 / 0.02572 |
| 500 | 0.06425 / 0.06455 | 0.06423 / 0.06464 | 0.06897 / 0.06927 | 0.06903 / 0.06924 | 0.06525 / 0.06554 |
| 5002 | 0.5289 / 0.5243 | 0.5289 / 0.5316 | 0.5398 / 0.5452 | 0.5234 / 0.5264 | 0.5217 / 0.5242 |

### E2 at k = 5,002, the Python driver (cleaned script only; published: the large runs 2, 3, 4, e2_chain_large.csv)

| k, version, column | bound | published (spread) | cleaned (spread) | change cleaned vs pub | | cause |
|---|---|---|---|---|---|---|
| 5002 triton_hand warm | GPU | 0.7954 (0.8 %) | 0.7956 (4.4 %) | +0.0 % | within |  |
| 5002 triton_hand L2-flushed | GPU | 0.7505 (3.9 %) | 0.7341 (0.2 %) | -2.2 % | within |  |
| 5002 triton_hand host_launch_us | host | 9.259 (22.3 %) | 9.265 (4.1 %) | +0.1 % | within |  |
| 5002 inductor_default warm | GPU | 0.5658 (2.5 %) | 0.5685 (0.6 %) | +0.5 % | within |  |
| 5002 inductor_default L2-flushed | GPU | 0.5193 (1.2 %) | 0.5297 (2.3 %) | +2.0 % | within |  |
| 5002 inductor_default host_launch_us | host | 25.07 (1.0 %) | 25.42 (1.6 %) | +1.4 % | within |  |
| 5002 inductor_wide warm | GPU | 0.569 (2.8 %) | 0.5455 (4.7 %) | -4.1 % | within FLAG |  |
| 5002 inductor_wide L2-flushed | GPU | 0.5187 (1.2 %) | 0.5199 (1.6 %) | +0.2 % | within |  |
| 5002 inductor_wide host_launch_us | host | 24.81 (3.7 %) | 26.12 (5.0 %) | +5.3 % | exceeds FLAG | drift, as in window C (+5 to +10 % on six host numbers there): the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them); this window pinned the driver to cores 0-3 and 16-19, the published runs' pinning is not recorded; not isolated |
| 5002 inductor_graphs warm | GPU | 0.579 (2.5 %) | 0.5789 (0.6 %) | -0.0 % | within |  |
| 5002 inductor_graphs L2-flushed | GPU | 0.528 (2.3 %) | 0.5295 (2.7 %) | +0.3 % | within |  |
| 5002 inductor_graphs host_launch_us | host | 39.83 (5.6 %) | 39.87 (3.1 %) | +0.1 % | within |  |
| 5002 eager warm | GPU | 141.4 (0.5 %) | 137.1 (0.0 %) | -3.1 % | exceeds FLAG | drift, as in window C (-3.0 to -3.7 % at k = 100 and 500): unchanged code; eager enqueues 5,004 PyTorch kernels per call; the SM clock of the three runs is 2,685 MHz (median of each run; the published runs have no clock log); not isolated |
| 5002 eager L2-flushed | GPU | 141.5 (0.2 %) | 137.2 (0.0 %) | -3.1 % | exceeds FLAG | as the warm number of eager |
| 5002 eager host_launch_us | host | 141,206 (0.2 %) | 136,765 (0.0 %) | -3.1 % | exceeds |  |
| 5002 eager_graphs warm | GPU | 130.8 (0.1 %) | 128.9 (0.0 %) | -1.5 % | exceeds |  |
| 5002 eager_graphs L2-flushed | GPU | 130.8 (0.2 %) | 128.9 (0.1 %) | -1.5 % | exceeds |  |
| 5002 eager_graphs host_launch_us | host | 5.691 (6.3 %) | 5.615 (15.9 %) | -1.3 % | within |  |

### E2 at k = 5,002: warm, kernel-only and flushed of every version (cleaned; medians of the three runs, ms; FKL from its program, run n with Python run n)

| version | warm (spread) | kernel-only (spread) | kernel-only / warm | L2-flushed (spread) |
|---|---|---|---|---|
| triton_hand | 0.7956 (4.4 %) | 0.7493 (0.1 %) | 0.942 | 0.7341 (0.2 %) |
| inductor_default | 0.5685 (0.6 %) | 0.5165 (0.2 %) | 0.908 | 0.5297 (2.3 %) |
| inductor_wide | 0.5455 (4.7 %) | 0.516 (0.1 %) | 0.946 | 0.5199 (1.6 %) |
| inductor_graphs | 0.5789 (0.6 %) | 0.5154 (0.3 %) | 0.89 | 0.5295 (2.7 %) |
| eager | 137.1 (0.0 %) | 137.2 (0.1 %) | 1 | 137.2 (0.0 %) |
| eager_graphs | 128.9 (0.0 %) | 128.9 (0.1 %) | 1 | 128.9 (0.1 %) |
| fkl | 0.5316 (2.0 %) | 0.5242 (0.3 %) | 0.986 | 0.5264 (0.1 %) |

## The SM clock of every run (the 100 ms sampler, between the run's start and end)

| run | start | s | SM MHz median | SM MHz min | SM MHz max | power W median | power W max | temperature max | pstates |
|---|---|---|---|---|---|---|---|---|---|
| reference/e16-attention/run1 | 01:25:15 | 106 | 2655 | 210 | 2670 | 441 | 456 | 74 | P2 P8 |
| cleaned/e16-attention/run1 | 01:27:01 | 115 | 2640 | 210 | 2670 | 436 | 460 | 74 | P0 P2 P3 P5 P8 |
| reference/e16-baselines/run1 | 01:28:56 | 207 | 2655 | 2280 | 2685 | 422 | 453 | 75 | P0 P2 |
| cleaned/e16-baselines/run1 | 01:32:23 | 208 | 2655 | 360 | 2685 | 418 | 455 | 74 | P0 P2 P3 P5 |
| cleaned/e16-baselines/run2 | 01:35:51 | 205 | 2655 | 510 | 2685 | 424 | 458 | 75 | P0 P2 P3 P5 |
| reference/e16-baselines/run2 | 01:39:16 | 204 | 2655 | 2280 | 2670 | 425 | 453 | 75 | P0 P2 P3 |
| cleaned/e16-attention/run2 | 01:42:40 | 115 | 2625 | 210 | 2685 | 438 | 461 | 75 | P0 P2 P3 P5 P8 |
| reference/e16-attention/run2 | 01:44:35 | 106 | 2625 | 2325 | 2670 | 444 | 455 | 75 | P2 |
| reference/e16-attention/run3 | 01:46:21 | 106 | 2625 | 2280 | 2655 | 443 | 455 | 75 | P0 P2 |
| cleaned/e16-attention/run3 | 01:48:07 | 116 | 2625 | 210 | 2670 | 437 | 466 | 75 | P0 P2 P3 P5 P8 |
| reference/e16-baselines/run3 | 01:50:03 | 204 | 2655 | 540 | 2685 | 426 | 458 | 75 | P0 P2 P3 |
| cleaned/e16-baselines/run3 | 01:53:27 | 206 | 2655 | 585 | 2685 | 421 | 456 | 75 | P0 P2 P3 |
| reference/e2-fkl/run1 | 01:56:53 | 1 | 2670 | 2670 | 2670 | 196 | 365 | 57 | P2 |
| cleaned/e2-fkl/run1 | 01:56:54 | 4 | 2670 | 2670 | 2670 | 113 | 148 | 55 | P0 P2 |
| cleaned/e2-fkl/run2 | 01:56:58 | 4 | 2670 | 2670 | 2670 | 112 | 147 | 54 | P0 P2 |
| reference/e2-fkl/run2 | 01:57:02 | 1 | 2670 | 2670 | 2670 | 126 | 132 | 57 | P2 |
| reference/e2-fkl/run3 | 01:57:03 | 1 | 2685 | 2670 | 2685 | 132 | 137 | 57 | P2 |
| cleaned/e2-fkl/run3 | 01:57:05 | 4 | 2685 | 2685 | 2685 | 112 | 146 | 52 | P0 P2 |
| cleaned/e2-python-k5002/run1 | 01:57:09 | 228 | 2685 | 210 | 2685 | 153 | 285 | 59 | P0 P2 P3 P5 P8 |
| cleaned/e2-python-k5002/run2 | 02:00:57 | 228 | 2685 | 210 | 2700 | 160 | 284 | 58 | P0 P2 P3 P5 P8 |
| cleaned/e2-python-k5002/run3 | 02:04:45 | 227 | 2685 | 210 | 2685 | 167 | 284 | 57 | P0 P2 P3 P5 P8 |

## E1: the tables of the cleaned build (the mangled ncu pass of this window, window C's cleaned binaries and run 1's timings)

- e1-vf-chain: 18 kernels (published 18, the same points and versions: True); static columns (SASS counts, registers, stack, local, cmem0, code bytes, block, grid, theoretical occupancy) different from the published table: 0; achieved occupancy within 0.60 points
- e1-hf-batch: 15 kernels (published 15, the same points and versions: True); static columns (SASS counts, registers, stack, local, cmem0, code bytes, block, grid, theoretical occupancy) different from the published table: 0; achieved occupancy within 5.25 points
- e1-preproc: 20 kernels (published 20, the same points and versions: True); static columns (SASS counts, registers, stack, local, cmem0, code bytes, block, grid, theoretical occupancy) different from the published table: 0; achieved occupancy within 3.73 points
- e1_compact.csv of the cleaned build (`cleaned/e1_compact.csv`): registers, SASS and occupancy columns equal to the published `../e1_compact.csv`: True; the timing columns:

| exp, point | ms_fkl pub / new | vs_ref pub / new | vs_best pub / new | vs_ref lo-hi pub / new |
|---|---|---|---|---|
| E1a 2 | 0.0202 / 0.0203 | 0.985 / 0.988 | 0.989 / 0.990 | 0.984-0.986 / 0.985-0.988 |
| E1a 102 | 0.0257 / 0.0256 | 0.961 / 0.959 | 0.968 / 0.966 | 0.960-0.962 / 0.957-0.961 |
| E1a 1002 | 0.1174 / 0.1150 | 0.996 / 0.987 | 0.997 / 0.988 | 0.987-1.004 / 0.986-0.988 |
| E1a 5002 | 0.5511 / 0.5275 | 1.026 / 0.994 | 0.993 / 1.006 | 1.012-1.039 / 0.992-0.998 |
| E1a 10002 | 1.0934 / 1.0784 | 1.018 / 0.989 | 0.997 / 0.994 | 1.008-1.019 / 0.963-1.007 |
| E1a 19902 | 2.2253 / 2.1944 | 1.012 / 0.995 | 1.000 / 0.996 | 1.006-1.012 / 0.993-1.005 |
| E1b 1 | 0.0029 / 0.0029 | 0.982 / 1.000 | 0.962 / 0.999 | 0.973-1.032 / 0.973-1.026 |
| E1b 51 | 0.0042 / 0.0043 | 1.002 / 1.006 | 1.002 / 1.014 | 0.979-1.008 / 1.001-1.029 |
| E1b 201 | 0.0078 / 0.0078 | 0.990 / 0.974 | 1.075 / 1.059 | 0.988-0.995 / 0.963-0.986 |
| E1b 601 | 0.0174 / 0.0174 | 0.980 / 0.979 | 1.109 / 1.113 | 0.975-0.983 / 0.977-0.983 |
| E1b 1191 | 0.0320 / 0.0321 | 0.970 / 0.980 | 1.135 / 1.141 | 0.967-0.973 / 0.977-0.984 |
| E1c 10 | 0.0036 / 0.0037 | 0.932 / 1.040 | 1.015 / 1.062 | 0.889-0.958 / 1.028-1.051 |
| E1c 50 | 0.0061 / 0.0058 | 1.005 / 1.031 | 1.035 / 1.043 | 1.005-1.008 / 1.027-1.099 |
| E1c 100 | 0.0087 / 0.0083 | 1.020 / 0.997 | 1.081 / 1.053 | 1.019-1.026 / 0.983-1.000 |
| E1c 200 | 0.0140 / 0.0133 | 1.010 / 1.010 | 1.076 / 1.089 | 1.007-1.012 / 1.010-1.013 |
| E1c 300 | 0.0198 / 0.0185 | 1.036 / 1.023 | 1.124 / 1.112 | 1.035-1.046 / 1.020-1.024 |

## Agreement

- E16 FKL error columns (max_abs_err, mean_abs_err): equal between the reference and the cleaned run of every round in 152 of 152 rows; equal to the published rows in 152 of 152
- E16 Python error columns: the reference script's equal to the published rows in 177 of 180 (the same script and seeds); the cleaned script reads the FKL inputs and checks the heads FKL checks
- E2 FKL outputs of window D: the six runs give the same six hashes: True; equal to window C's (and so to the published outputs): True
- E2 k = 5,002 Python accuracy, kernel and SASS columns (kernels, max_ulp_vs_exact, bitexact_fraction_vs_triton_hand, bitexact_fraction_vs_fkl, max_ulp_vs_fkl, sass_ffma, sass_fmul, sass_fadd, status) against the published large runs: all equal

