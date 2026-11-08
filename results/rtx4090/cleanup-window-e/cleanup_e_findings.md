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
