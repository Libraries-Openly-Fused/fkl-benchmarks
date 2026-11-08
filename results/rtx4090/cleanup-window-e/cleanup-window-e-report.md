# Cleanup window E, 2026-11-08: the E2 FKL program in the order of the Python driver

## Findings

1. **k = 5,002: the order effect is confirmed.**
   - With its flushed pass after its warm pass, the cleaned program's flushed time is 0.5502 ms (runs 0.5502, 0.5638, 0.5437).
   - The reference program, whose flushed pass has always followed its warm pass, gave 0.5398 (window C) and 0.5452 (window D); the published value is 0.5445.
   - In the old order (flushed after the kernel-only pass) it was 0.5234 (C) and 0.5264 (D).
   - Against the Python driver's flushed times of window D, FKL / Inductor at k = 5,002 is again 1.039 (default), 1.058 (wide) and 1.039 (graphs). The published values are 1.049, 1.050 and 1.031; the old order gave 0.994, 1.012 and 0.994.
   - Kernel-only, which does not depend on the order, stays at 0.5218 ms: FKL / Inductor 1.010-1.012, FKL / Triton 0.696.
2. **k = 10 to 500:** the new order moves the flushed and kernel-only times by 0.5 % at most (k = 500 kernel-only: -0.5 % against C, -0.9 % against D).
3. **k = 2 is not comparable, and the cause is in this window, not in the order.**
   - Every run of this window started from an idle GPU: 120 s of idle before the window, then 5 s before each run (a pause in the window script that windows C and D did not have; pstate P8 at each start). k = 2 is the first chain length of every run.
   - In windows C and D the FKL runs followed other GPU work.
   - k = 2 in this window: flushed +3.1 % / +2.9 % against C / D (about 0.0267 in all three runs), and kernel-only +1.7 %. The warm time of run 1 is high at k <= 100 (k = 2: 0.0222 against 0.0210 in runs 2 and 3), which gives the warm spreads of 2.5-5.9 %.
   - The reference's first run of window C, which also followed a different phase, gave the same k = 2 flushed time (0.02677 ms, against 0.0259 in its later runs).
   - Probable cause: the GPU's state after idle, at the first chain length; not isolated. The clock log shows 2,685-2,700 MHz in every run, with the 210 MHz samples at each start.
   - A repeat with the three runs back to back, after an untimed warm-up run of the same program, would give k = 2 under the conditions of windows C and D (about 1 minute).
4. **Agreement.** The outputs are the six published hashes in every run, and the CSV note of every run gives the new order.
5. **e2_per_k_summary.csv** is regenerated here, and supersedes window D's.
   - The fkl rows come from this window: the new order. The Python rows come from windows C and D, whose order is the same.
   - The fkl row at k = 2 carries the cold start of item 3.

The cleaned FKL program of windows C and D timed every k warm, kernel-only, L2-flushed (old order). Since PR #28 it times warm, L2-flushed, kernel-only, as triton/e2_chain.py (new order). The reference program has no kernel-only pass: its flushed pass follows its warm pass, as in the new order. Times in ms, the median of three runs, with the spread ((largest - smallest) / median) in brackets; change = new order against the old order of the window named.

## L2-flushed

| k | old order, window C | old order, window D | new order, window E | new against C | new against D | reference C / D | published |
|---|---|---|---|---|---|---|---|
| 2 | 0.02586 (0.3 %) | 0.02592 (0.1 %) | 0.02667 (0.5 %) | +3.1 % | +2.9 % | 0.02592 / 0.02586 | 0.02656 (5.9 %) |
| 10 | 0.02618 (0.4 %) | 0.02627 (0.1 %) | 0.02619 (0.3 %) | +0.0 % | -0.3 % | 0.02626 / 0.02625 | 0.02691 (4.3 %) |
| 50 | 0.02774 (0.2 %) | 0.02782 (0.2 %) | 0.02773 (0.2 %) | -0.0 % | -0.3 % | 0.02776 / 0.02781 | 0.02798 (3.6 %) |
| 100 | 0.02992 (0.2 %) | 0.03 (0.3 %) | 0.0299 (0.1 %) | -0.1 % | -0.3 % | 0.02983 / 0.03006 | 0.03032 (1.7 %) |
| 500 | 0.06903 (0.2 %) | 0.06924 (0.1 %) | 0.06897 (0.2 %) | -0.1 % | -0.4 % | 0.06897 / 0.06927 | 0.06903 (0.6 %) |
| 5002 | 0.5234 (0.0 %) | 0.5264 (0.1 %) | 0.5502 (3.7 %) | +5.1 % | +4.5 % | 0.5398 / 0.5452 | 0.5445 (2.5 %) |

## kernel-only

| k | old order, window C | old order, window D | new order, window E | new against C | new against D |
|---|---|---|---|---|---|
| 2 | 0.02048 (0.0 %) | 0.02048 (0.0 %) | 0.02082 (0.4 %) | +1.7 % | +1.7 % |
| 10 | 0.02048 (0.0 %) | 0.02048 (0.1 %) | 0.02057 (0.1 %) | +0.4 % | +0.4 % |
| 50 | 0.02286 (0.2 %) | 0.02283 (0.1 %) | 0.02289 (0.6 %) | +0.1 % | +0.3 % |
| 100 | 0.02571 (0.1 %) | 0.02572 (0.2 %) | 0.02575 (0.3 %) | +0.2 % | +0.1 % |
| 500 | 0.06525 (0.2 %) | 0.06554 (0.0 %) | 0.06493 (0.8 %) | -0.5 % | -0.9 % |
| 5002 | 0.5217 (0.1 %) | 0.5242 (0.3 %) | 0.5218 (0.0 %) | +0.0 % | -0.5 % |

## warm

| k | old order, window C | old order, window D | new order, window E | new against C | new against D | reference C / D | published |
|---|---|---|---|---|---|---|---|
| 2 | 0.02028 (0.4 %) | 0.02029 (0.3 %) | 0.021 (5.9 %) | +3.5 % | +3.5 % | 0.02069 / 0.02038 | 0.02115 (5.6 %) |
| 10 | 0.02037 (2.0 %) | 0.02043 (0.6 %) | 0.02032 (5.1 %) | -0.2 % | -0.5 % | 0.02042 / 0.02044 | 0.02123 (3.8 %) |
| 50 | 0.02271 (0.3 %) | 0.02274 (0.3 %) | 0.02272 (4.2 %) | +0.0 % | -0.1 % | 0.02276 / 0.02279 | 0.02285 (4.2 %) |
| 100 | 0.02548 (2.0 %) | 0.02555 (0.3 %) | 0.02547 (5.5 %) | -0.1 % | -0.3 % | 0.02544 / 0.02559 | 0.02545 (4.9 %) |
| 500 | 0.06423 (0.1 %) | 0.06464 (0.2 %) | 0.0643 (2.5 %) | +0.1 % | -0.5 % | 0.06425 / 0.06455 | 0.06429 (0.3 %) |
| 5002 | 0.5289 (0.9 %) | 0.5316 (2.0 %) | 0.5305 (1.8 %) | +0.3 % | -0.2 % | 0.5289 / 0.5243 | 0.5265 (0.7 %) |

## FKL against the Python driver at k = 5,002 (FKL's time over the version's: > 1 FKL is slower)

The Python rows are window D's (the driver's order is unchanged); published: the large runs 2, 3, 4.

| version | flushed: published | flushed: old order (D) | flushed: new order (E) | kernel-only: old order (D) | kernel-only: new order (E) |
|---|---|---|---|---|---|
| triton_hand | 0.726 | 0.717 | 0.749 | 0.700 | 0.696 |
| inductor_default | 1.049 | 0.994 | 1.039 | 1.015 | 1.010 |
| inductor_wide | 1.050 | 1.012 | 1.058 | 1.016 | 1.011 |
| inductor_graphs | 1.031 | 0.994 | 1.039 | 1.017 | 1.012 |
| eager | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 |
| eager_graphs | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 |

## The SM clock of every run (the 100 ms sampler, between the run's start and end)

| run | start | s | SM MHz median | SM MHz min | SM MHz max | power W median | power W max | temperature max | pstates |
|---|---|---|---|---|---|---|---|---|---|
| cleaned/e2-fkl/run1 | 02:27:07 | 4 | 2700 | 210 | 2700 | 99 | 111 | 44 | P2 P8 |
| cleaned/e2-fkl/run2 | 02:27:16 | 4 | 2685 | 210 | 2685 | 83 | 109 | 44 | P2 P8 |
| cleaned/e2-fkl/run3 | 02:27:25 | 4 | 2685 | 210 | 2685 | 81 | 110 | 44 | P2 P8 |

## Agreement

- The outputs of the three runs: the same six hashes in every run: True; equal to windows C and D and to the published outputs: True
- The CSV note of every run gives the new order: True

## e2_per_k_summary.csv

`cleaned/e2_per_k_summary.csv` (35 rows) supersedes `../cleanup-window-d/cleaned/e2_per_k_summary.csv` for the paper. Every row names its source:
the fkl rows come from this window's three runs; the Python rows come from window C (k = 2, 10, 100, 500) and window D (k = 5,002). Columns: warm with the launch, warm kernel-only and L2-flushed, each the median of three runs with the spread.

