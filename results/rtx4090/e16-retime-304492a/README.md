# E16 timed again at the new pin, FKL 304492a, against FKL b10fc66 (RTX 4090, RT-2)

**What.** E16, `fkl-self/attention/e16_attention.cu` and the PyTorch baselines `pytorch-attention/e16_baselines.py`, timed again after the
pin move of fkl-benchmarks to FKL #435:
- #435 was merged as `304492af4d777165bd556fdd6113439635724d02`;
- the pin move is #66, merged as af96aef.

**"old"** is the paper's E16 data, `data/e16-attention-rtx4090-cleanup` = `../cleanup-window-d/cleaned/e16-summary/run1..3` (c3c86a7):
window D's cleaned build, fkl-benchmarks f27c605 with FKL 64ad9195. The protocol is window D's E16 step. The window runs two builds of
the FKL program (`manifest.sha256`):
- **A:** fkl-benchmarks 976e42b with FKL b10fc66, the pin before the move. It has old's timed code, which the code check below shows.
- **B:** fkl-benchmarks main with FKL 304492a, the pin after it.

Decision: the coordinator's task RT-2 of 2026-11-27, Oscar's order of 2026-11-26 for the pin move. Oscar's rule of 2026-11-26 applies:
"Coordinate with fkl-reduce, fkl-reduce has priority". So the window starts only in a quiet slot agreed with that session, and any run it
disturbs is redone.

**State.** The window ran on the coordinator's GRANTED (fkl-reduce QUIET, HOLD COMPILES and WINDOW STARTS sent) from 2026-11-27 04:41:27
to 05:22:40 (+01:00; 03:41:27Z to 04:22:40Z, `window/window-console.txt`). The analysis below was done afterwards, on the CPU only.

## Result

**The window** (`window/`):
- **Runs:** 12 of 12 exited 0, in window D's order.
  - The FKL runs took 115 s each, with 152 data rows a run, no INCOMPLETE and an empty stderr.
  - The Python runs took 205-210 s each, with 180 rows a run and no INCOMPLETE. Each Python stderr holds 32,208 B of the torch profiler's
    "profiler_start/stop" lines, the size of window D's cleaned runs.
- **Guards:** the after-run check never fired, so no run is affected and none is redone. The idle wait never timed out: it waited 0 to 81 s
  before a run, and the 1-minute load at the start of a run was 0.17 to 0.49.
- **Clocks** (`gpu_clocks.csv`, `gpu_clocks.err` empty):
  - 18,257 of the 24,631 readings have a utilisation of 50 % or more, all at P2. Their SM clock was 2,655 MHz in 7,857 of them and 2,670 MHz
    in 3,985, the rest 2,355 to 2,640 MHz.
  - The temperature was 41 to 75 °C.
  - The median of the rows' SM clocks (`sm_mhz_median`) is 2,640 MHz for A's and B's FKL rows, and 2,625 MHz for old's.
- **The environment:** Python 3.12.13, torch 2.14.1+cu130 (CUDA 13.0, cuDNN 92400), triton 3.8.0 (`environment-before.txt`).

**The numbers** (`analysis/retime-report.md`, every number in `retime-numbers.csv`):
- **What they are:** every row is GPU-bound (window D's rule), so the flag is a change above 3 %.
- **How to read them:** "exceeds" means larger than the larger spread of the two sides, and "apart" means the two min-max ranges do not
  overlap.
- **Cause:** not shown for any change.

| | B against A | A against old | B against old |
|---|---|---|---|
| FKL program, 152 rows | -0.58 % to +0.74 %; 5 exceed, 7 apart; **no flag** | -2.69 % to +1.11 %; 44 exceed, 54 apart; no flag | -2.79 % to +1.06 %; 45 exceed, 58 apart; no flag |
| Python baselines, 176 timed rows | -0.43 % to +2.30 %; 4 exceed, 7 apart; no flag | -3.34 % to +1.03 %; 60 exceed, 69 apart; **1 flag** | -2.56 % to +2.13 %; 57 exceed, 62 apart; no flag |

- **The Python flag:** `efficient` (PyTorch's memory-efficient attention) at D=64 S=1024 BH=256, non-causal, A against old: 0.6298 → 0.6088 ms
  (-3.34 %).
  - The spreads are 2.16 % (old) and 1.71 % (A), and the ranges are apart.
  - The row's SM clock was 2,370 MHz in old and 2,400 in A.
  - It is a baseline of the same unchanged script.
- **A is faster than old in most rows:** 140 of 152 FKL rows and 147 of 176 Python rows, by up to 2.7 % and 3.3 %. Not shown to be the
  cause.
- **The 4 rows the scripts skip** (`math` at S = 8,192, scores of 4.0 GiB) are untimed on every side, as in window D.

**The paper's numbers** (each side's `e16-summary/`, by `scripts/e16_summary.py`):
- **The fraction FKL / best baseline** (`e16_summary_core.csv`, per configuration in `analysis/retime-report.md`). It moves by at most 0.0114
  between B and A (D=64 causal S=1024, whose runs span 0.98-1.02), 0.0211 between A and old, and 0.0265 between B and old (both D=64
  non-causal S=512).
- **The text numbers** (`e16_text_numbers.txt`; the diffs are `analysis/text-numbers-*.diff`):

| | old (paper) | A | B |
|---|---|---|---|
| fraction over all 20 points | 0.933 to 1.038 | 0.940 to 1.038 | 0.939 to 1.037 |
| points above the best baseline (median of the runs) | 7 of 20 | 6 of 20 | 5 of 20 |
| D=64 non-causal | 0.94 to 1.01 | 0.96 to 1.01 | 0.97 to 1.01 |
| D=64 causal | 1.00 to 1.04 | 1.00 to 1.04 | 0.99 to 1.04 |
| D=128 non-causal | 0.93 to 0.99 | 0.94 to 0.98 | 0.94 to 0.99 |
| D=128 causal | 0.94 to 1.00 | 0.94 to 1.00 | 0.94 to 0.99 |

  - **Points that change side of 1.0:**
    - D=64 causal S=512: above in old (1.003), below in A and B (0.999 each);
    - D=64 causal S=1024: below in old (0.997), above in A (1.006), below in B (0.995);
    - D=128 causal S=8192: above in old (1.005), below in A (0.990) and B (0.993).
  - These points lie within about 1 % of 1.0, inside their run-to-run ranges.
- **The ratio separate / fused** (`e16_q2_compact.csv`, median [min-max]):

| work, D=128, non-causal | S | old | A | B |
|---|---|---|---|---|
| int8 K,V prologue | 2,048 | 0.8602 [0.8507-0.8648] | 0.8601 [0.8581-0.8673] | 0.8578 [0.8559-0.8678] |
| int8 K,V prologue | 8,192 | 0.8006 [0.7976-0.8027] | 0.7909 [0.7894-0.7985] | 0.7931 [0.7917-0.7943] |
| fp8 e4m3 K,V prologue | 2,048 | 0.8678 [0.8528-0.8768] | 0.8648 [0.8648-0.8721] | 0.8676 [0.8599-0.8739] |
| fp8 e4m3 K,V prologue | 8,192 | 0.7813 [0.7811-0.7895] | 0.7818 [0.7803-0.7826] | 0.7792 [0.7771-0.7802] |
| epilogue o*a+b | 2,048 | 1.0586 [1.0297-1.0588] | 1.0576 [1.0314-1.0594] | 1.0530 [1.0307-1.0599] |
| epilogue o*a+b | 8,192 | 1.0126 [1.0065-1.0132] | 1.0185 [1.0080-1.0186] | 1.0180 [1.0133-1.0185] |

**Checks:**
- **The error columns:** the FKL error columns (max_abs_err, mean_abs_err) are equal in 456 of 456 rows (three rounds), A against old and B
  against A.
- **The order:** the position of every variant in every round is old's on both sides.
- **The rows:** every run has 152 FKL and 180 Python rows, and old's keys.
- **Not made here:** `e16_sm_clock_summary.csv`, the third derived file of the paper's package. The coordinator made the paper's from
  window D's cleaned `run{1,2,3}/e16_{attention,baselines}_clock.csv` (2026-11-08) and derives A's and B's the same way from
  `{A,B}/e16-summary/run1..3`.

## The files the window runs (`manifest.sha256`)

- **A** = `scratch/pin-435/base-b10fc66/bench-build/bin/e16_attention_cu`, sha256 `4374e41a...`.
  - This is #68's identity build (976e42b, FKL b10fc66, cvGPUSpeedup 6d26c65, FastNPP a1b5e24): RT-1's and RT-4's A.
- **B** = `scratch/pin-435/main-304492a/bench-build/bin/e16_attention_cu`, sha256 `531fa7ce...`.
  - This is RT-1's B build (main 73a9650, FKL 304492a).
  - Its E16 source is #66's port to FKL #435 (b51e53f): `executeOperations<DPP>`, and the automatic choice of `executeFlashAttentionMma`
    made in each launch, with explicit tiles.
- **The scripts:** `pytorch-attention/e16_baselines.py` and `scripts/triton_env.sh` of this branch.
  - They equal f27c605's, which window D's cleaned side ran.
  - `scripts/e16_summary.py` and `scripts/e16_join_clock.py`, which make the paper's summary, are unchanged since f27c605 too.
- **The Python environment:** `venv-torch`, torch 2.14.1+cu130, triton 3.8.0. Window D's environment file does not record the versions;
  the window records them in `environment-before.txt`.
- **The inputs:** the 10 files of `e16-work/inputs`, one per configuration, written in window D's pre-window step. Their sha256 equal
  window D's `inputs-sha256.txt` (checked on 2026-11-27, `code-check/inputs-check-2026-11-27.txt`), and the window checks them again
  before it starts.

## A against old: the code check (`code-check/`, CPU only)

`scripts-used/rt2_code_check.sh` compares old (window D's cleaned `e16_attention_cu`, sha256 268c78ad..., still in
`worktrees/fkl-benchmarks-window-d`) with A.

**Why a check is needed.** A's `e16_attention.cu` equals f27c605's. The two builds differ in:
- FKL, 64ad9195 → b10fc66;
- `common/cuda_timing.h` and `common/status_check.h`, in comments only;
- `cmake/fklb_write_build_info.cmake`;
- the build path.

**What it found:**
1. **The device code** (`sass-old-vs-A.txt`; #60's `sass_identity.py`, `scripts-used/sass_identity_60.py`, which pairs the kernels with
   FKL #432's defaulted `fk::BinaryType` removed from the names): **41 of 41 kernels identical**.
2. **The host code** (`host-old-vs-A.txt`; `host_compare435.py --no-rename --drop-binarytype-default --path-delta 5 --intervals`):
   - 332 of 336 functions are identical, and **every timed interval (74) is identical**, with its callees.
   - **3 functions differ only by the build path's length.** These are three `timeCall` instantiations of `Runner<64>`, whose error
     messages hold the `__FILE__` of FKL's `flash_attention.h` and `flash_attention_mma.h`, 5 characters longer in A's link farm.
   - **1 differs otherwise:** `fklb::writeSystemInfoHeader`, the CSV header's build information (the commit string: f27c605's 40
     characters, "unknown" in the link farm). No timed call reaches it.
   - A first run of this step used `--path-delta 9`, a miscount (the delta is 5). It reported the same 4 functions unclassified. It was rerun;
     the rerun is this record, and the first run's output stays in scratch.

**So "A against old" is the same timed code:** the device code and every timed interval. The CSV header's build information differs.

## B against A: the kernels (`resources/`)

`pr435_sass_identity.py` with FKL #435's rename rule, and `cuobjdump -res-usage` (13.4.3, sm_89): A has 41 kernels and B has 43
(`sass-A-vs-B.txt`, `.csv`).
- **39 identical,** the attention kernels (MMA, SIMT, split-KV) among them.
- **2 different:** the split-KV combines, A's `flashFwdSplitCombineKernel<bf16, 64|128, true>` and B's
  `launchDPP_Kernel<FlashAttentionSplitCombineDPP<GPU, bf16, 64|128, true, AttentionIdentityEpilogue>>`.
  - Registers 36 and 40 in both, and no stack.
  - Constant bank 0 is 376 B in A and 384 B in B.
  - Instructions: 416 in both for D=64; 448 in A and 440 in B for D=128.
- **2 in B only:** the same combines with a fused epilogue (`BinaryInstantiableOperation<...>` as the epilogue). Since #435, the split
  path applies the epilogue in the combine.
- B's E16 host code is #66's port. The C-PM-CUDA and C-NV records analysed it (`../pin-move-check-b10fc66-48fbf3f/`,
  `../fkl-pr435-check/d1cff06/`).

No cause of a time difference is claimed from this.

## The window (`window-script.sh`)

(Run on the coordinator's GRANTED, 2026-11-27 04:41:27 to 05:22:40 +01:00; the result is above.)

**The protocol is window D's E16 step.** Three rounds r = 0, 1, 2 run with `E16_ROTATE=r` (the FKL program) and `--rotate r` (the Python
baselines):
- round 0: FKL A, FKL B, Python A, Python B;
- round 1, the reverse: Python B, Python A, FKL B, FKL A;
- round 2: as round 0.

FKL A and B take window D's reference and cleaned places. The Python baselines run once per side and round, at the places of window D's
two scripts; both sides run the same script.
- **The folders:** `window/{A,B}/e16-attention/run1..3` and `window/{A,B}/e16-baselines/run1..3`.
- **The inputs:** every run reads the same input files, the FKL program through `E16_INPUT_DIR` and the Python script through
  `--input-dir`, as window D's cleaned side did.
- **Not run:** the query tile (below).

**Around every run:**
- **before it**, the script waits for an idle machine: no build, compile job or measurement, a 1-minute load below 0.5, and no other
  program on the RTX 4090 (up to 300 s);
- **after it**, it checks again for a build, a compile job or another program on the RTX 4090. If one appeared during the run, the window
  stops there, and that run is the affected one, to be redone.

**The runs:**
- each is pinned to cores 0-3,16-19, with window D's timeouts (1800 s for the FKL program, 3600 s for Python);
- the Python runs source `scripts/triton_env.sh`, as in window D;
- each must exit 0 and write its CSV with data rows and no `# INCOMPLETE`, and the window stops at the first failure;
- nvidia-smi reads the GPU every 100 ms (`gpu_clocks.csv`, window D's fields, for `scripts/e16_join_clock.py`).

**One difference from window D:** TMPDIR is on disk (`scratch/rt2/tmp`) for every run, the Python runs included (their torch caches). This is
the standing rule against the RAM disk /tmp.

**The guards before the start** (`GUARD_ONLY=1` runs only these):
- no blocked program;
- the committed manifest, all four files matching;
- A carries FKL b10fc66 and B 304492a, with no `-dirty` commit;
- the Python script and `triton_env.sh` are f27c605's;
- the 10 input files match window D's record, and the folder holds nothing else;
- HEAD differs from main 3f641d9 only under `results/`;
- the script is the committed one, run from its place;
- a new output folder;
- no build, compile job, measurement or other program on the RTX 4090.

## After the window (`scripts-used/after_window.sh`, CPU only)

1. **`rt2_summaries.sh`** writes each side's `{A,B}/e16-summary/` in the layout of the paper's folder:
   - `run1..3/` with copies of the run CSVs (checked byte for byte) and their `_clock.csv` from `scripts/e16_join_clock.py`, with this window's
     sampler;
   - beside them, `scripts/e16_summary.py`'s `e16_summary_core.csv`, `e16_summary_fusion.csv`, `e16_summary_kernels.csv`, `e16_q2_compact.csv`
     and `e16_text_numbers.txt`, and `join.log` and `summary.log`;
   - the scripts run in a scratch layout (`scratch/rt2/summary-work/<side>/`, with `gpu_clocks.csv` linked), as window D's did.
2. **`rt2_report.py`** writes `analysis/` (old = `../cleanup-window-d/cleaned/e16-summary`):
   - **Every timed row** of the FKL program (152 a run) and of the Python baselines (180), keyed as window D's report keyed them: old, A and B,
     each the median of the three rounds with the min-max and the spread.
   - **The changes:** B against A, A against old and B against old, with whether each exceeds the larger spread and whether the ranges
     overlap, and the SM clock of each row.
   - **Bound and flag:** every E16 row is GPU-bound, as in window D's report, so the flag is above 3 %.
   - **The paper's summary, old / A / B:** the fraction FKL / best baseline per configuration, and the ratio separate / fused of
     `e16_q2_compact.csv`.
   - **Also:** the FKL error columns (A against old, B against A), and the order of the variants in every round.
   - **The text numbers:** the diff of `e16_text_numbers.txt` (old against A, old against B, A against B).

**Tested on window D** (in scratch): its reference side as "A" and its cleaned side as "B", with its sampler.
- B's five summary files and its `_clock.csv` files came out byte-identical to the paper's folder.
- "B against old" is 0.00 % in every row.

## The query tile

`e16_query_tile_cu` is not run. The coordinator's brief: leave it out unless a package the paper uses holds its data.
- `../../paper-facing-provenance.md` (917ea96, 2026-11-06) lists `rtx4090/e16-query-tile` among the paper-facing E16 results.
- Window D did not re-run it.
- Whether a current package holds it is the coordinator's call.

## Files

| path | content |
|---|---|
| `manifest.sha256` | A, B and the two scripts the window runs |
| `window-script.sh` | the window's script |
| `code-check/` | A against old: the console, the device-code and host-code comparisons (with the timed intervals and the diffs), `cuobjdump -res-usage` of both, the input check |
| `resources/` | B against A: the SASS comparison and its CSV, `cuobjdump -res-usage` of both |
| `window/` | the window: console, environment before and after, `gpu-info.csv`, `gpu_clocks.csv` (and `.err`), `load.log`, the copy of the script it ran, and `{A,B}/e16-attention/run1..3`, `{A,B}/e16-baselines/run1..3` (each with its CSV, stdout, stderr, exit code, load and the GPU state at its end) |
| `A/e16-summary/`, `B/e16-summary/` | each side in the paper's layout: `run1..3` (the CSVs and their `_clock.csv`), `e16_summary_core.csv`, `e16_summary_fusion.csv`, `e16_summary_kernels.csv`, `e16_q2_compact.csv`, `e16_text_numbers.txt`, `join.log`, `summary.log` |
| `analysis/` | `retime-report.md`, `retime-numbers.csv` (`rt2_report.py`), and the diffs of the text numbers |
| `scripts-used/` | `rt2_code_check.sh`, `sass_identity_60.py` (#60's `sass_identity.py`), `host_compare435.py`, `pr435_sass_identity.py`, `rt2_summaries.sh`, `rt2_report.py`, `after_window.sh` |
