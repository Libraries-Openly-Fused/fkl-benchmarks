# Cleanup window D, 2026-11-08, 01:25:14-02:08:43: E16, E2 at k = 5,002 and E1's ncu pass, reference against cleaned

Part of the re-measurement after the baseline cleanup of the CUDA programs (Oscar's decisions of 2026-11-07, coordinator session). This is the last window of the series.
- The sequence (`sequence-script.sh`, `sequence-console.txt`):
  - step 0, 01:18:10-01:18:30, compile only;
  - 30 s of idle;
  - the untimed pre-window steps, 01:19:00-01:23:11;
  - 120 s of idle;
  - this window.
- RTX 4090 = nvidia-smi index 1, CUDA 13.4.3, driver 610.43.02. The desktop session kept running.
- **A clock sampler ran through the whole window** (`gpu_clocks.csv`, nvidia-smi every 100 ms: timestamp, SM clock, memory clock, pstate, utilisation, power, temperature; `gpu_clocks.err` is empty).
  - It started before the first run and stopped after the last: 26,044 samples.
  - Every run folder's `load.txt` has the run's start and end, for the clock of each run.

**The builds** (nvcc 13.4.92 for sm_89, GNU 15.2.0, Release, against FKL 64ad9195, cvGPUSpeedup 6d26c65 and FastNPP a1b5e245):
- **reference:** `build/local-rtx4090` of the reference checkout, built from fkl-benchmarks a5dd9fa (the sources before the cleanup) on 2026-11-07 at 00:44-00:46 and not rebuilt.
  - It gives `e16_attention_cu` and `e2_fkl_chain_cu`.
  - The reference E16 Python script is `pytorch-attention/e16_baselines.py` of the reference checkout, equal to a5dd9fa's (checked before the start).
- **cleaned:**
  - E16: main f27c605, whose sources equal 0d2fd00's (the cleanup PRs #18 to #27). Step 0 built it in a clean detached worktree (`worktrees/fkl-benchmarks-window-d`) with the submodules at the reference checkout's commits and preset `local-rtx4090`.
    - The build info shows f27c605 without -dirty, and its other fields equal the reference's.
    - The cache options and the compile flags of `e16_attention_cu` are equal.
    - Kernel SASS against the reference, matched by demangled name: 41 of 41 identical. The 5 device input generators that PR #26 replaced with the bit-identical host generator are only in the reference.
    - Its record is in `step0/`. The cleaned E16 Python script is that worktree's `pytorch-attention/e16_baselines.py`.
  - E1 and E2: window C's cleaned builds, main 1a2535c, whose E1 and E2 sources are unchanged since (`../cleanup-window-c/`; their sha256 were checked against window C's record). Also window C's `triton/e2_chain.py`.

**Binaries (sha256):**

| program | reference (a5dd9fa) | cleaned |
|---|---|---|
| `e16_attention_cu` | `dc67710a080805366b3c9f14db4b9dea8e6ae442961bb2355c0c05fd7e76ad8c` | `268c78adf599b94b519c88cb49f91031ca7de486a21358d59600e6062ab18ccc` (f27c605) |
| `e2_fkl_chain_cu` | `95154fe4c5a040ad7a7047fcd8c715883fd64c4cef69f5bf9ddf9390980f1bc2` | `a11e531470f08ddb5e4027b2dd7bb073a6812e2383a2f21c33f069543436a42c` (1a2535c, window C's) |
| `e1_vf_chain_cu`, `e1_hf_batch_cu`, `e1_preproc_cu` | (not run) | window C's, see `binary-sha256-e1-e2-window-c.txt` |

**Before the window** (untimed; `pre-window/`):
- **The E16 input files.** The cleaned `e16_attention_cu` with E16_PROFILE=1 (one call per variant, no timing) and E16_INPUT_DIR=`/home/oscar/tfkl-bench/e16-work/inputs` wrote the files of the published configurations.
  - Files: `e16_input_d<D>_s<S>_bh<BH>.bin` for D = 64, 128, S = 512 to 8,192 and BH = 262,144 / S. 10 files, 2,558,525,440 bytes, outside the repository.
  - Their sizes (`inputs-sizes.txt`, 10 n + 16 BH S bytes each) and SHA-256 (`inputs-sha256.txt`) equal the ones in the notes of the program's CSV. The window checked the SHA-256 again before the start.
- **The error check.** The reference `e16_attention_cu` (its device generator) ran the same profile pass. The FKL error columns of the two are equal in 152 of 152 rows (`pre-window/e16-profile-{cleaned,reference}/`).
- **The k = 5,002 warm-up.** One untimed run of the cleaned Python driver at k = 5,002 (226 s, 7 ok rows). It warms the file cache and the Python imports; every (version, k) gets fresh Triton and Inductor caches anyway.
  - Its folder is not committed: it is not a measurement. Its console line is in `pre-window/console.txt`.

**What ran:** 24 runs, all pinned to cores 0-3,16-19, all exiting 0 with no INCOMPLETE line.
1. **E16, three rounds.** Round r ran with E16_ROTATE = r - 1 for the FKL program and --rotate r - 1 for the Python baselines. The variant order of each round is in the report.
   - Round 1: FKL reference, FKL cleaned, Python reference, Python cleaned. Round 2: Python cleaned, Python reference, FKL cleaned, FKL reference. Round 3: as round 1.
   - Folders: `{reference,cleaned}/e16-attention/run1..3` (`e16_attention.csv`) and `{reference,cleaned}/e16-baselines/run1..3` (`e16_baselines.csv`).
   - The cleaned program and the cleaned baselines read the input files; the reference ones make their own inputs (the device generator: the same bytes; seeded torch.randn).
   - The query tile is not re-run.
2. **E2 at k = 5,002.**
   - The FKL program, three runs per build, the build order flipping every run, on the published `e2-work/e2_input.u8`: `{reference,cleaned}/e2-fkl/run1..3`. The program always runs its six chain lengths.
     - Outputs: `/home/oscar/tfkl-bench/e2-work-cleanup-d/<build>-run<n>/`, sha256 in each run folder's `outputs-sha256.txt`. All six runs give the same six hashes as window C, and so as the published outputs.
   - Then the cleaned Python driver at k = 5,002 (`--ks 5002`), three runs: `cleaned/e2-python-k5002/run1..3`.
     - Run n has the outputs and the CSV of the cleaned FKL run n. Passes: warm, L2-flushed and kernel-only.
3. **E1's Nsight Compute pass of window C's cleaned binaries,** again, with --print-kernel-base mangled: `cleaned/e1-ncu/e1-{vf-chain,hf-batch,preproc}/` (18, 15 and 20 kernels, all mangled names). It replaces window C's pass, whose demangled names `scripts/e1_table.py` cannot join.

**Checks before the start** (the window refuses to start if one fails):
- the frozen-program guard on every binary;
- step 0's and the pre-window steps' records;
- the sha256 of the E16 binaries (step 0) and of the E1 and E2 binaries (window C's record);
- the commit strings inside the binaries;
- the sha256 of the E2 input and of the 10 E16 input files;
- the Python scripts equal to their commits;
- `busy_processes`, and no compute context on the RTX 4090 from a program outside /usr;
- new output folders.

**Summary files made after the window** (CPU only, from the recorded runs; no plots):
- **E1, the tables of the cleaned build** (Table III): `cleaned/e1-ncu/<workload>/table.csv`, in the published format.
  - Made by `scripts/e1_table.py` from this window's mangled ncu pass, window C's cleaned binaries, and the timings of window C's `cleaned/<workload>/run1`.
  - The occupancy probe's files sit next to them: `probe_input.tsv`, `occupancy.tsv` and the logs.
  - The extracted `.cubin` is not committed (`results/**/*.cubin` is in `.gitignore`, as for the published tables). `e1_table.py cubin` extracts it again from the binary.
  - `cleaned/e1_compact.csv`: `scripts/e1_compact.py` on those tables and window C's cleaned runs 1 to 3.
  - `cleaned/e1_kernel_only_ratios.csv`: FKL over each hand-written kernel per point, kernel-only (`*_kernel_mean_ms`) and normal. Per run of window C's cleaned runs 1 to 3, then the median, the smallest, the largest and the spread.
- **E2, per k:** `cleaned/e2_per_k_summary.csv`. For each k and version: warm with the launch, warm kernel-only and L2-flushed, each the median of three runs with the spread.
  - k = 2, 10, 100 and 500 come from window C's `cleaned/e2-python/run1..3`; k = 5,002 from this window's `cleaned/e2-python-k5002/run1..3`.
  - The fkl rows are the FKL program's times, merged by the driver.
- **E16, for the paper's text numbers:** `{reference,cleaned}/e16-summary/`, in the layout of the published folder.
  - `run1..3/` holds copies of the recorded `e16_attention.csv` and `e16_baselines.csv`, byte-identical to the run folders' (checked), and their `_clock.csv` versions from `scripts/e16_join_clock.py` with this window's sampler.
  - Beside them, what `scripts/e16_summary.py` (main f27c605) writes: `e16_summary_core.csv`, `e16_summary_fusion.csv`, `e16_summary_kernels.csv`, `e16_q2_compact.csv`, `e16_text_numbers.txt`, and the logs `join.log` and `summary.log`.
  - The join ran in a scratch layout of symlinks (`scratch/e16-d-summary/<side>/`, with this window's `gpu_clocks.csv` linked in), so `join.log` names those paths. To repeat it here, link `../../gpu_clocks.csv` into the folder.

**Compared with** the published folders:
- E16: `../e16-attention/run1..3` (2026-11-01; its `_clock.csv` files and summaries);
- E2: `../e2-chain/`, `../e2-chain-run3/` and `../e2-chain-run4/` (runs 2, 3, 4; `e2_fkl_chain.csv`, and `e2_chain_large.csv` for k = 5,002);
- E1: `../e1-vf-chain-v2/`, `../e1-hf-batch-v2/`, `../e1-preproc-v3/` and `../e1_compact.csv`.

**Report:** `cleanup-window-d-report.md`, with every number in `cleanup-window-d-report.csv`. `cleanup_d_report.py` wrote both (its findings section is `cleanup_d_findings.md`) and only reads the result folders. `cleanup_d_summaries.py` wrote the E1 ratio and E2 per-k files.
- As for windows A to C: published, reference now and cleaned, medians over the runs with their spreads; drift, cleanup effect, and the 3 % / 5 % thresholds.
- Also in the report: the SM clock of every E16 row and of every run, the paper's E16 summary on each side, the variant order of every round, the preparation column, the E1 tables, and agreement.

In short:
- **E16:** no number above the spread and the threshold (320 compared). The paper's fractions FKL / best baseline move by 0.019 at most. The FKL error columns are identical in every row on every side.
- **E2 at k = 5,002:** the cleaned FKL program's flushed time is 3.5 % lower, as in window C, because its flushed pass now follows its kernel-only pass.
  - The Python driver times its flushed pass before its kernel-only pass. The flushed comparison against Inductor therefore moves, from +4.9 % / +5.0 % (published) to -0.6 % / +1.2 % (cleaned).
  - Kernel-only, FKL is 1.5-1.7 % slower than Inductor and 30 % faster than the hand-written Triton kernel.
  - The clock log shows 2,670-2,685 MHz in every E2 FKL run of both builds.
- **E1:** the tables of the cleaned build have the published static columns in all 53 kernels.

**Files:**
- `binary-sha256-e16.txt` (step 0), `binary-sha256-e1-e2-window-c.txt` (window C's record), `build-info-{reference,cleaned}.txt` and `main-commit.txt` (step 0).
- `inputs-sha256.txt` and `inputs-sizes.txt`: the E16 input files.
- `step0/`: step 0's record:
  - its console;
  - its checks: `sass-e16.txt`, and the empty diffs `build-info.diff`, `cache.diff` and `flags-e16_attention_cu.diff`;
  - the configure and build logs;
  - its script.
  - The copies of the binaries it compared are not committed.
- `pre-window/`: the pre-window steps' console and script, the two profile passes, and the input records.
- `sequence-script.sh`, `sequence-console.txt`; `window-script.sh`, `window-console.txt`.
- `environment-before.txt` and `environment-after.txt`: load, top processes, and the git state of the three checkouts.
- `gpu-info.csv`, `gpu_clocks.csv`, `gpu_clocks.err`, and `load.log` (load every 5 s, logger on core 12 at nice 19; the sampler on core 13 at nice 19).
- Every run folder also has `stdout.txt`, `stderr.txt`, `exit_code.txt`, `load.txt` (start and end) and `gpu-at-end.csv`.
  - The ncu folders have the profile-mode CSV, `ncu.csv`, `ncu.err` and `ncu.exit`, plus the table files listed above (without the `.cubin`).
