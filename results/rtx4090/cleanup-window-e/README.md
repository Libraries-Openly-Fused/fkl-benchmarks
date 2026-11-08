# Cleanup window E, 2026-11-08, 02:27:07-02:27:41: the E2 FKL program in the order of the Python driver

The cleaned E2 FKL program of windows C and D timed every k warm, kernel-only, L2-flushed.
- PR #28 (merged as 9a6a88b) makes it time warm, L2-flushed, kernel-only, as `triton/e2_chain.py` does.
- This window measures the program in the new order, so that the order effect is documented and the paper's per-k file has the FKL rows of the new order.
- The reference is not re-run. Its program has no kernel-only pass, so its flushed pass already follows the warm one (windows C and D).

- The sequence (`sequence-script.sh`, `sequence-console.txt`): step 0, 02:24:52-02:25:05, compile only; 120 s of idle; this window.
- RTX 4090 = nvidia-smi index 1, CUDA 13.4.3, driver 610.43.02. The desktop session kept running.
- A clock sampler ran through the whole window: `gpu_clocks.csv`, nvidia-smi every 100 ms, 317 samples; `gpu_clocks.err` is empty.

**The build** (nvcc 13.4.92 for sm_89, GNU 15.2.0, Release, against FKL 64ad9195, cvGPUSpeedup 6d26c65 and FastNPP a1b5e245):
- **cleaned:** main 9a6a88b. Step 0 built `e2_fkl_chain_cu` in a clean detached worktree (`worktrees/fkl-benchmarks-window-e`), with the submodules at the reference checkout's commits and preset `local-rtx4090`.
  - It refused to build unless the source had the new pass order.
  - The build info shows 9a6a88b without -dirty, and its other fields equal those of the reference build (`build/local-rtx4090`, a5dd9fa).
  - The cache options and the compile flags are equal.
  - SASS against the reference: 7 of 7 kernels identical, plus `e1::gpuSpin`, which only the cleaned build has.
  - Its record is in `step0/`. `binary-sha256.txt`, `main-commit.txt` and `build-info-*.txt` in this folder are copies of it.
  - The binary's sha256: `e2_fkl_chain_cu` in `binary-sha256.txt` (with the reference's for the record).

**What ran:** three runs of the cleaned program in `cleaned/e2-fkl/run1..3`, with all six chain lengths, pinned to cores 0-3,16-19, on the published `e2-work/e2_input.u8`.
- Each run was preceded by 5 s of idle (see the report: k = 2 is the first chain length of a run).
- The outputs went to `/home/oscar/tfkl-bench/e2-work-cleanup-e/cleaned-run<n>/`, with their sha256 in each run folder's `outputs-sha256.txt`: the six published hashes in every run.
- All three runs exited 0, with no INCOMPLETE line.

**Checks before the start:**
- the frozen-program guard;
- step 0's record and the sha256 of the binary;
- the commit string inside it;
- the sha256 of the E2 input;
- `busy_processes`, and no compute context on the RTX 4090 from a program outside /usr;
- new output folders.

**Report:** `cleanup-window-e-report.md`, with its numbers in `cleanup-window-e-report.csv`. `cleanup_e_report.py` wrote both and only reads the result folders; its findings section is `cleanup_e_findings.md`.
- FKL's L2-flushed, kernel-only and warm times in the old order (windows C and D) and the new order (this window), side by side for every k.
- FKL against the Python driver at k = 5,002.
- The SM clock of every run, and agreement.

`cleaned/e2_per_k_summary.csv` (35 rows) also comes from `cleanup_e_report.py`. It supersedes `../cleanup-window-d/cleaned/e2_per_k_summary.csv` for the paper.
- Rows: k = 2, 10, 100, 500, 5,002 × the seven versions.
- Columns: warm with the launch, warm kernel-only and L2-flushed, each the median of three runs with the spread.
- A `source` column names the runs of every row: the fkl rows come from this window; the Python rows from window C (k <= 500) and window D (k = 5,002).

In short:
- **k = 5,002:** the order effect is confirmed. The flushed time in the new order is 0.5502 ms, against 0.5234 and 0.5264 in the old order and 0.5398 and 0.5452 for the reference. FKL / Inductor flushed is again 1.039-1.058 (published 1.031-1.050).
- **k = 10 to 500:** unchanged within 0.5 %.
- **k = 2:** about 3 % higher. Probably this window's start from an idle GPU before every run, not the order (see the report).

**Files:**
- `step0/`: step 0's record:
  - its console;
  - `sass-e2.txt`, and the empty diffs `build-info.diff`, `cache.diff` and `flags-e2_fkl_chain_cu.diff`;
  - the configure and build logs;
  - its script.
  - The copies of the binaries it compared are not committed.
- `sequence-script.sh`, `sequence-console.txt`, `window-script.sh`, `window-console.txt`.
- `environment-before.txt`, `environment-after.txt`, `gpu-info.csv`, `gpu_clocks.csv`, `gpu_clocks.err`, `load.log`.
- Every run folder has `e2_fkl_chain.csv`, `stdout.txt`, `stderr.txt`, `exit_code.txt`, `load.txt` (start and end), `gpu-at-end.csv` and `outputs-sha256.txt`.
