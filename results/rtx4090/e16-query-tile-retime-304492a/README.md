# The E16 query tile timed again at the new pin, FKL 304492a, against FKL b10fc66 (RTX 4090, RT-2b)

**What.** `fkl-self/attention/e16_query_tile.cu`, the E16 query-tile experiment (D = 128, non-causal, S = 512 to 8,192), timed again
after the pin move of fkl-benchmarks to FKL #435 (merged as `304492af4d777165bd556fdd6113439635724d02`; the pin move is #66).

**What the paper prints from it.** One sentence rests on the record `../e16-query-tile/` (2026-11-04):
- the extra time of the fused int8 kernel over the bf16 kernel, query tile 128 against 64, at S = 2,048, 4,096 and 8,192 (the ratio
  excess(128) / excess(64));
- the share of that extra time that the timing-only variant leaves out. That variant is the `nodq` copy, without the conversion arithmetic
  and the scale wait.

**The two builds.** The window runs two builds of the program alternately (`manifest.sha256`):
- **A:** fkl-benchmarks 976e42b with FKL b10fc66, the pin before the move;
- **B:** fkl-benchmarks main with FKL 304492a, the pin after it.

**"old"** is the record: its binary was built 2026-11-04 from 35ebfd8-dirty with FKL c8fcbe8, sha256 763e7e3c... A has old's kernels but
not old's timed host code, as the code check below shows.

Decision: the coordinator's task RT-2b of 2026-11-27. The paper agent confirmed that the sentence rests on this record. The coordinator:
the kernels are identical at 304492a, but the timed host code changed with #435 (QA3). Oscar's rule of 2026-11-26 applies: "Coordinate with
fkl-reduce, fkl-reduce has priority".

**State.** The code check is done (CPU only; nothing was built for RT-2b). The window waits for the coordinator's GRANTED.

## The binaries (`manifest.sha256`)

- **A** = `scratch/pin-435/base-b10fc66/bench-build/bin/e16_query_tile_cu`, sha256 `733b07a4...`. This is #68's identity build: RT-1's,
  RT-2's and RT-4's A.
- **B** = `scratch/pin-435/main-304492a/bench-build/bin/e16_query_tile_cu`, sha256 `05b78494...`.
  - This is RT-1's B build.
  - Its source is #66's port of the program to FKL #435 (b51e53f): `executeOperations<DPP>` with explicit tiles.
- **What the binaries carry.** Both name the repository commit "unknown" (a link farm) and carry their FKL commit, with no `-dirty` commit.

## A against old: the code check (`code-check/`, CPU only)

`scripts-used/rt2b_code_check.sh` compares the record's binary (old, `fkl-benchmarks/build/rtx4090/bin/e16_query_tile_cu`, only read) with A.

**Why they differ.** A's source is not the record's. The cleanup of the CUDA programs (#26: 33609e9 and 0e5af89) rewrote
`e16_query_tile.cu` (523 lines). After it:
- the program reads the input files of `e16_attention.cu`, where it had its own device generator;
- each call has its own prepared argument set;
- the timing is `common/cuda_timing.h`'s.

The library also moved, c8fcbe8 → b10fc66.

1. **The device code** (`sass-old-vs-A.txt`; #60's `sass_identity.py`, `scripts-used/sass_identity_60.py`, FKL #432's defaulted
   `fk::BinaryType` removed from the names):
   - **51 of 51 shared kernels identical.**
   - **7 kernels are in old only:** the device input generators (`fillNormal`, `affineBf16`, four `quantizeTokens`,
     `gatherHeadsQuant<true>`), which the cleanup removed.
2. **The host code** (`host-old-vs-A.txt`; `host_compare435.py --no-rename --drop-binarytype-default --path-delta 24 --intervals`;
   A's source root is 24 characters longer):
   - 86 of the 176 paired functions are identical.
   - **The timed functions are not paired at all:** 47 timed roots in old, 133 in A, and every timed interval is unpaired. The cleanup
     replaced the program's timing loop with `common/cuda_timing.h`.
   - The diffs are in `host-old-vs-A-diffs.txt.gz`.

**So "A against old" is the same kernels with different timed host code:** the cleanup's rewrite and the library move between c8fcbe8 and
b10fc66. The query tile was not timed in the cleanup windows, so A is the first timing of the cleaned program. A change of A against old is
not shown to come from either.

## B against A: the code (`resources/`)

- **The device code** (`sass-A-vs-B.txt`; `pr435_sass_identity.py` with FKL #435's rename rule; `cuobjdump -res-usage` of all three):
  - **31 identical:** the 28 non-split MMA kernels, `referenceAttention` and the two gathers.
  - **2 different:** the split-KV combines, as in E16: constant bank 0 376 → 384 B; instructions 416 → 416 (D=64) and 448 → 440 (D=128).
  - **18 in A only:** the split-KV instantiations (`SPLIT_KV = true`) of the experiment's tiles, for D = 64 and 128. A's
    `executeFlashAttentionMma` instantiates them. B's explicit `executeOperations` calls do not.
  - **What ran:** every row of the record ran with `split_kv = false` and one kernel per call, and the report checks the same in A's and B's
    rows.
- **The host code** (`host-A-vs-B.txt`; `host_compare435.py --intervals`, #435's rename rule): 74 timed intervals differ (40 event and
  34 clock intervals), from #66's port. The diffs are in `host-A-vs-B-diffs.txt.gz`.

No cause of a time difference is claimed from this.

## The window (`window-script.sh`)

(To be run on the coordinator's GRANTED; about 5-8 minutes with the idle waits. Each of the record's runs took 38-39 s.)

**The record's protocol** (`../e16-query-tile/window-script.sh`): `E16_DIMS=128 E16_CAUSAL=0 E16_ROTATE=r`, a 1500 s timeout, cores
0-3,16-19, and TMPDIR on disk. A and B alternate as in RT-2:
- round 0: A, B;
- round 1: B, A;
- round 2: A, B.

The folders are `window/{A,B}/run0..2`, the record's `run<r>`. There is no ncu pass: the printed Nsight Compute numbers are instruction
counts of kernels that are identical at both pins.

**The inputs:** both sides read the same bytes, the D = 128 files of `e16-work/inputs` (`E16_INPUT_DIR`).
- The 10 files are checked against window D's record before the start.
- Without `E16_INPUT_DIR` the program would look for the files, and write them, in the run folder.
- The record's program made its own inputs, with its device generator.

**Around every run:**
- **before it**, the script waits for an idle machine: no build, compile job or measurement, a 1-minute load below 0.5, and no other
  program on the RTX 4090;
- **after it**, it checks again. If a build, a compile job or another GPU program appeared during the run, the window stops there, and that
  run is the affected one, to be redone.

**The runs:** each must exit 0 and write its CSV with data rows and no `# INCOMPLETE`, and the window stops at the first failure. nvidia-smi
reads the GPU every 100 ms.

**The guards before the start** (`GUARD_ONLY=1` runs only these):
- no blocked program;
- the committed manifest, both binaries matching;
- A carries FKL b10fc66 and B 304492a, with no `-dirty` commit;
- the inputs match window D's record;
- HEAD differs from main d66e854 only under `results/`;
- the script is the committed one, run from its place;
- a new output folder;
- no build, compile job, measurement or other program on the RTX 4090.

## After the window (`scripts-used/after_window.sh`, CPU only)

1. **`rt2b_after.sh`** writes each side in the record's layout: `{A,B}/run0..2/e16_query_tile.csv` (copies, checked byte for byte), and
   `scripts/e16_query_tile_analyse.py` (unchanged) on that folder: `e16_query_tile_derived.csv` and `.md`, the record's files.
2. **`rt2b_report.py`** writes `analysis/` (old = `../e16-query-tile/`):
   - **The sentence's quantities** at S = 2,048, 4,096 and 8,192, with `e16_query_tile_analyse.py`'s definitions:
     - the excess at BQ 64 and 128;
     - the ratio excess(128) / excess(64);
     - the share left out by the timing-only variant at BQ 64 and 128;
   - **for each quantity:** per run, then the median of the three runs with the min-max and the spread, for old, A and B;
   - **the changes:** B against A, A against old and B against old, in % and as differences, with whether each exceeds the larger spread
     and whether the ranges overlap; the flag is above 3 % (GPU times and their ratios);
   - **also:** the same at S = 512 and 1,024, and every timed row;
   - **checks:** the rows, `split_kv` and the kernels per call of every row, the error columns of the rows that are not timing only, and the
     positions.

**Tested on the record** (in scratch, with both sides = the record): the derived files came out byte-identical to the record's, and every
change was 0.

No cause is named unless shown.

## Files

| path | content |
|---|---|
| `manifest.sha256` | A and B, the two binaries the window runs |
| `window-script.sh` | the window's script |
| `code-check/` | A against old: the console, the device-code and host-code comparisons (the diffs gzipped: `gzip -n -9`, over 1 MiB), `cuobjdump -res-usage` of both |
| `resources/` | B against A: the SASS comparison and its CSV, the host-code comparison with the timed intervals (its CSV and diffs gzipped, over 1 MiB), `cuobjdump -res-usage` of both |
| `scripts-used/` | `rt2b_code_check.sh`, `sass_identity_60.py` (#60's `sass_identity.py`), `host_compare435.py`, `pr435_sass_identity.py`, `rt2b_after.sh`, `rt2b_report.py`, `after_window.sh` |
