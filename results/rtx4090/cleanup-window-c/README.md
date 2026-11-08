# Cleanup window C, 2026-11-08, 00:28:34-00:43:27: E1 and E2, reference against cleaned

Part of the re-measurement after the baseline cleanup of the CUDA programs (Oscar's decisions of 2026-11-07, coordinator session).
- The sequence: step 0 (00:25:15-00:26:34, compile only), 120 s of idle, then this window. Files: `sequence-script.sh`, `sequence-console.txt`.
- A first attempt of step 0 (00:22:49-00:24:05) stopped before anything was measured.
  - Its compile-flag comparison reported a difference that was only the names of the two build folders (`build/ref-rtx4090-e1` against `build/rtx4090-e1`). It had replaced the checkout paths before the build folders.
  - The comparison now replaces both build folders first. The attempt's worktree was removed and the whole sequence ran again.
  - Its record is in `step0-attempt1/`: its console, its sequence console, the configure and build logs, and the diff that stopped it.
- RTX 4090 = nvidia-smi index 1, CUDA 13.4.3, driver 610.43.02. The desktop session kept running.

**The builds** (nvcc 13.4.92 for sm_89, GNU 15.2.0, Release, against FKL 64ad9195, cvGPUSpeedup 6d26c65 and FastNPP a1b5e245):
- **reference:** two builds of the reference checkout, made from fkl-benchmarks a5dd9fa (the sources before the cleanup). Neither was rebuilt.
  - E1: `build/ref-rtx4090-e1`, built on 2026-11-07 at 21:21. It uses preset `rtx4090-e1` with OpenCV and NPP off: the flags of the published E1 builds, -lineinfo off.
  - E2: `build/local-rtx4090`, built on 2026-11-07 at 00:46, preset `local-rtx4090`.
- **cleaned:** main 1a2535c, the cleanup PRs #18 to #25.
  - Step 0 built it in a clean detached worktree (`worktrees/fkl-benchmarks-window-c`), with the submodules at the reference checkout's commits and the same two presets.
  - The build info shows 1a2535c without -dirty, and its other fields equal the reference's.
  - The cache options and the compile flags of the four targets are equal.
  - Kernel SASS against the reference, matched by demangled name:
    - E1: 54 of 54 identical, plus `e1::gpuSpin` in each of the three binaries (cleaned build only).
    - E2: 7 of 7, plus `e1::gpuSpin`.
  - Its record is in `step0/`. `build-info-*.txt`, `binary-sha256.txt` and `main-commit.txt` in this folder are copies of it.

**Binaries (sha256):**

| program | reference (a5dd9fa) | cleaned (main 1a2535c) |
|---|---|---|
| `e1_vf_chain_cu` | `018c34a3577c7c20cb51bcf3a48228b3c114f39b17ec701941a313f071f0d01a` | `23e09697400335cda7a71e0456adf35f8292c8cd456e621ada8c0292d67932b4` |
| `e1_hf_batch_cu` | `f111b4d2bace807286ed3af022aeda18e1982f7bab664dc886382aa25384fdfd` | `19e78db076f4a2bbaf420c409695e377471556a1c38c9434e5bfc2218982e0dc` |
| `e1_preproc_cu` | `b02bbdf8e5b4c9191c09e5bf11350595c079a053da6d3d449decc91ff0e91a12` | `85c031d961c7bb9878b8a21c3ac98cd9a4bdceaef8e9df82ba401d19f577fc68` |
| `e2_fkl_chain_cu` | `95154fe4c5a040ad7a7047fcd8c715883fd64c4cef69f5bf9ddf9390980f1bc2` | `a11e531470f08ddb5e4027b2dd7bb073a6812e2383a2f21c33f069543436a42c` |

The reference E1 binaries are in `build/ref-rtx4090-e1/bin/` and the reference E2 binary in `build/local-rtx4090/bin/`; the cleaned ones are in the worktree's `build/rtx4090-e1/bin/` and `build/local-rtx4090/bin/`.

**What ran:** 35 runs, all pinned to cores 0-3,16-19.
1. **E1**, three runs of `e1_vf_chain_cu`, `e1_hf_batch_cu` and `e1_preproc_cu` on each build: `{reference,cleaned}/e1-{vf-chain,hf-batch,preproc}/run1..3`.
   - The build order flipped every run: run 1 reference then cleaned, run 2 cleaned then reference, run 3 reference then cleaned. Each program's pair of runs was back to back.
   - The cleaned build also writes the new columns:
     - the preparation of every launch (`*_prep_*`, `*_total_mean_ms`);
     - the kernel-only times (`*_kernel_*`): the same launches, each queued behind a one-thread GPU spin of E1_SPIN_US = 5000 us.
   - Each variant's normal pass is followed by its kernel-only pass, before the next variant.
2. **The spin check:** each cleaned E1 program once more with E1_SPIN_US = 10000, in `cleaned/e1-*/spin2x`.
3. **E1's Nsight Compute pass** of the cleaned build: `cleaned/e1-*/profile` (E1_PROFILE=1, one launch of every kernel), and `cleaned/ncu-version.txt`.
   - **It ran with ncu's default kernel names, which are demangled.** The published pass and `scripts/e1_table.py`, which joins the ncu rows with the SASS of the binary, use mangled names (`--print-kernel-base mangled`).
   - The files are kept as recorded. The pass is repeated with mangled names, on the same cleaned binaries, in window D (`../cleanup-window-d/`).
   - The E1 tables of the cleaned build (`e1_table.py`, `e1_compact.py`) come from that pass. The report matches this pass with the published one by launch order.
4. **E2, the FKL program:** three runs per build, in `{reference,cleaned}/e2-fkl/run1..3`, the order flipping as in 1.
   - Input: the published `e2-work/e2_input.u8`; its sha256, `726b9ef38a8277d6021009d344446bf5915fe75f746d07950e608f63614a8995`, was checked before the start.
   - Passes: warm, L2-flushed and, in the cleaned build, kernel-only. The cleaned program times them in the order warm, kernel-only, flushed for each k.
   - Then the cleaned program once more with E1_SPIN_US = 10000, in `cleaned/e2-fkl/spin2x`.
5. **E2, the Python driver:** `triton/e2_chain.py` of the cleaned checkout, with venv-torch and `scripts/triton_env.sh`. Three runs, in `cleaned/e2-python/run1..3`.
   - Run n uses the outputs and the CSV of the cleaned FKL run n. The default chain lengths (2, 10, 50, 100, 500) and versions; no compile-time measurement, no large k.
   - Then one run with E1_SPIN_US = 10000 (`cleaned/e2-python/spin2x`), with the outputs and the CSV of the cleaned FKL spin2x run.
   - By plan there is no reference run of the driver: its changes are drift and cleanup together.
   - Its warm, flushed and host times run the same code as in the published runs. The cleaned script only adds the kernel-only pass, after them in the same process.

All 35 runs exited 0, with no INCOMPLETE line and no failed check.

**E2's FKL outputs:** each run's `fkl_out_k{2,10,50,100,500,5002}.f32`, 35,389,440 bytes each, are in `/home/oscar/tfkl-bench/e2-work-cleanup/<build>-run<n>/`, outside the repository.
- Their sha256 are in each run folder's `outputs-sha256.txt`.
- All six runs, both builds, give the same six hashes.
- These equal the hashes of the published outputs in `e2-work/`, written on 2026-10-31 at 20:49, in the window of the published runs 3 and 4.
- The spin2x run's outputs (`e2-work-cleanup/cleaned-spin2x/`) were hashed after the window and give the same six hashes.
- k = 100, 500 and 5,002 give the same bytes: the chain has reached its fixed point.

| file | sha256 |
|---|---|
| `fkl_out_k2.f32` | `c12f370b9bbd8a78294702a71a5b7ef6a2c599fba7bfec61dd05e319d5e753dd` |
| `fkl_out_k10.f32` | `e339d7e451c6ea8527715d0194644b4d8c663005c43d88ea3ef23a8a9c78bd07` |
| `fkl_out_k50.f32` | `8f7136a8f7d53f6d133014665b7c696de5c45fa4e7b6f9d3c07a29d5409b8987` |
| `fkl_out_k100.f32`, `fkl_out_k500.f32`, `fkl_out_k5002.f32` | `04844006cea64cd1d82a973a6e3946c13e79e04a75b53b6889354833e230bcc5` |

**Checks before the start** (the window refuses to start if one fails):
- the frozen-program guard (`refuse_blocked_binaries`) on every binary of both builds;
- step 0's record and the sha256 of every binary;
- the expected commit strings inside the binaries: a5dd9fa in the reference, 1a2535c without -dirty in the cleaned;
- the sha256 of the E2 input;
- `busy_processes` (scripts/quiet_check.sh): no compiler, profiler, benchmark binary, window or runner script;
- no compute context on the RTX 4090 from a program outside /usr (the desktop's /usr/bin/ptyxis holds one);
- new output folders, this one and `e2-work-cleanup/`.

**Compared with** the published folders:
- E1: the sets of `../e1_compact.csv`, three runs each, all with FKL 38c4f32:
  - `../e1-vf-chain-v2/` and `../e1-hf-batch-v2/`, 2026-10-30 at 10:44;
  - `../e1-preproc-v3/`, 2026-10-30 at 12:53-12:54.
- E2: the runs of `../e2-chain/e2_compact.csv`:
  - `../e2-chain/`, run 2, 2026-10-31, FKL 38c4f32;
  - `../e2-chain-run3/` and `../e2-chain-run4/`, 2026-10-31, FKL c8fcbe8.

**Report:** `cleanup-window-c-report.md`, with every number in `cleanup-window-c-report.csv`. `cleanup_c_report.py` wrote both and only reads the result folders.
- As for windows A and B: published, reference now and cleaned, each a median over the runs with its run-to-run spread.
- drift = reference now against published; cleanup effect = cleaned against reference now. Each change is judged against the spreads and the 3 % (GPU-bound) or 5 % (host-bound) threshold.
- Bound: host when the cleaned build's kernel-only time is below 80 % of its normal time; the L2-flushed passes count as GPU.
- Also in the report: the kernel-only and preparation columns of the cleaned build, the 10 ms spin against the 5 ms spin, the ratios of the paper's compact E1 table (vs_ref, vs_best), agreement, and the ncu pass.

In short (33 of 312 compared numbers above the spread and the threshold, each with a cause, none isolated):
- **Order of the passes.** In the cleaned programs every variant's normal pass comes after the previous variant's kernel-only pass: 100 one-thread spins of 5 ms, about 0.5 s of a nearly idle GPU. In the reference it came after the previous variant's normal pass.
  - E1a at 5,002 and 10,002 operations: in the reference and the published runs, the variants timed second and third (hand_imm, fkl) are 3-5 % slower than the first (hand_params). In the cleaned build the three agree within 0.8 %.
  - The paper's E1a vs_ref at 5,002 and 10,002 operations moves from 1.026 and 1.018 (published) to 0.994 and 0.989 (cleaned). Kernel-only it is 0.998 at both.
  - E2's flushed pass at k = 5,002 is 3.0 % faster: it now follows the kernel-only pass.
  - The probable cause is the GPU's clock or power state after sustained load. It is not isolated: there was no clock log in this window. Window D logs the clock.
- **E1c drift.** The published set `e1-preproc-v3` is 3-6 % slower than reference now on every version. Reference now matches `e1-preproc-v2` of the same morning within 2.3 %.
- **Kernel-only.**
  - At 10,002 and 19,902 operations E1a's kernel-only time is 4 % below normal: 40-90 us, far more than a dispatch, and the 10 ms spin lowers it by only 0.3-0.5 %.
  - E2's FKL warm interval holds no visible dispatch.
  - The Python driver's warm intervals of Triton and Inductor are 37-71 % dispatch at k <= 100.
- **E2 without the dispatch.** FKL's kernel is 10-38 % slower than Triton's and Inductor's at k <= 500. That is the order of the published L2-flushed times.
- **Spin.** 10 ms against 5 ms is within 1.9 % for the 72 kernel-only numbers above 10 us.
- **Agreement.** Every bit-exact flag is 1. The E2 FKL outputs are identical in every run, and equal to the published ones. The Python accuracy, kernel and SASS columns equal the published ones.

**Files:**
- `binary-sha256.txt`, `build-info-{reference,cleaned}-{e1,e2}.txt` and `main-commit.txt`: from step 0.
- `step0/`: step 0's record:
  - its console;
  - its checks: `sass-e1.txt`, `sass-e2.txt`, and the empty diffs `build-info-*.diff`, `cache-*.diff` and `flags-*.diff`;
  - the configure and build logs;
  - its script.
  - The copies of the binaries it compared are not committed.
- `step0-attempt1/`: the first attempt, which stopped.
- `sequence-script.sh` and `sequence-console.txt`; `window-script.sh` and `window-console.txt`: the scripts and their consoles.
- `environment-before.txt` and `environment-after.txt`: load and top processes, plus the git state of both checkouts.
- `gpu-info.csv`, and `load.log` (load every 5 s, logger on core 12 at nice 19).
- Every timed run folder also has `stdout.txt`, `stderr.txt`, `exit_code.txt`, `load.txt` (E1 and the E2 FKL program) and `gpu-at-end.csv`.
- The profile folders have the profile-mode CSV, `ncu.csv` (the program's output, then ncu's CSV), `ncu.err` and `ncu.exit`.
