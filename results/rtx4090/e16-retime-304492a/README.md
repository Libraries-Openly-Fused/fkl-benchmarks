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

**State.** The code check and the kernel resources are done (CPU only; nothing was built for RT-2). The window waits for the coordinator's
GRANTED.

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

(To be run on the coordinator's GRANTED; about 35-45 minutes with the idle waits. Window D's E16 step took 32 min: the FKL program about
115 s a run, the Python baselines about 205 s.)

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
| `scripts-used/` | `rt2_code_check.sh`, `sass_identity_60.py` (#60's `sass_identity.py`), `host_compare435.py`, `pr435_sass_identity.py`, `rt2_summaries.sh`, `rt2_report.py`, `after_window.sh` |
