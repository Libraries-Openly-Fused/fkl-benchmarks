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
