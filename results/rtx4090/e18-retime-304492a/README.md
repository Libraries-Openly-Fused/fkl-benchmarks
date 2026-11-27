# E18 timed again at the new pin, FKL 304492a, against FKL b10fc66 (RTX 4090, RT-4)

**What.** The three E18 programs, `e18_box_cu`, `e18_box_split_cu` and `e18_stencil_cu`, timed again after the pin move of fkl-benchmarks
to FKL #435:
- #435 was merged as `304492af4d777165bd556fdd6113439635724d02`;
- the pin move is #66, merged as af96aef.

The design and protocol are #63's (`../e18-retime-b10fc66/`). The window runs two builds of each program alternately (`binaries.sha256`):
- **A:** fkl-benchmarks 976e42b with FKL b10fc66, the pin before the move;
- **B:** fkl-benchmarks main with FKL 304492a, the pin after it.

**"old"** is #63's B (fkl-benchmarks 2160c70 with FKL b10fc66), whose timed code A repeats. The code check below shows it.

Decision: the coordinator's task RT-4 of 2026-11-27, Oscar's order of 2026-11-26 for the pin move. Oscar's rule of 2026-11-26 applies:
"Coordinate with fkl-reduce, fkl-reduce has priority". So every build here ran at nice 19 with at most 8 jobs, the window starts only in
a quiet slot agreed with that session, and any run it disturbs is redone.

**State.** The binaries, the code check and the kernel resources are done (compile only and CPU only). The window waits for the
coordinator's GRANTED.

## The binaries

No new build for the window. A and B are RT-1's builds (`../e19-crop-sweep-retime-304492a/`), both of `scripts/check_fkl_tree.sh` (a link
farm of the checkout) with the FKLB_* options, OpenCV and nvcc of the reference build `build/local-rtx4090`:
- **A** = `scratch/pin-435/base-b10fc66/bench-build/bin/<program>`.
  - This is #68's identity build: 976e42b with FKL b10fc66, cvGPUSpeedup 6d26c65 and FastNPP a1b5e24.
  - Its three sha256 are that record's (`../pin-move-check-b10fc66-48fbf3f/sass-identity/binary-sha256-base-b10fc66.txt`).
- **B** = `scratch/pin-435/main-304492a/bench-build/bin/<program>`.
  - This is RT-1's B build: main 73a9650 (658a896 and #67, which changes only `results/rx9060/`) with FKL 304492a, cvGPUSpeedup 6d26c65 and
    FastNPP a1b5e24.
  - Its three sha256 are those of RT-1's `compile-B/`.
- **What the binaries carry.** The link farm has no `.git`, so both builds name the repository commit "unknown". They carry their FKL
  commit, and no `-dirty` commit.

## A against old: the code check (`code-check/`, compile only and CPU only)

**#63's B binaries no longer exist.** #63's B was #60's head build, in `scratch/pin-433/head-b10fc66`, which was removed after #63. So the
check rests on four things.

1. **The same source and settings.**
   - **Source:** 2160c70 (#63's B) and 976e42b (A) differ only under `results/` (`git diff --name-only`). They have the same submodules:
     FKL b10fc66, cvGPUSpeedup 6d26c65 and FastNPP a1b5e24.
   - **The rebuild:** old was rebuilt in its own folder, `scratch/pin-433/head-b10fc66`, from A's checkout (`scripts-used/rebuild_old.sh`,
     `code-check/rebuild-old/`, 2026-11-27 04:09:59 to 04:10:26).
     - The checkout is the clean worktree `fkl-benchmarks-pin-435-base` at 976e42b.
     - The script is `check_fkl_tree_rt4.sh`: RT-1's `check_fkl_tree_rt.sh` with the targets cut to the three programs. It ran at nice 19
       with 8 jobs.
   - **Same configuration:** the rebuild's configure log is identical to #63's B's (#60's record `identity-builds/head-b10fc66/`), apart
     from CMake's two timing lines (`configure-old-vs-rebuild.diff`, empty).
2. **The device code, against what #60's record keeps of #63's B** (`linecounts-vs-60-A.txt`, `-rebuild.txt`).
   - **What the record keeps:** for each of its 17 E18 kernels that differed from 24478f0, the normalised SASS line count.
   - **The method:** `sass_linecounts_vs_60.py` runs #60's own `sass_identity.py` functions with #60's cuobjdump (CUDA 13.3). The record cuts
     every name at 240 characters, and several of these kernels share those first 240 characters. So the counts are compared per group of
     kernels with one 240-character prefix.
   - **The result:** in all 5 groups (17 kernels), A's line counts are #63's B's; the rebuild's too.
3. **The device code of A against the rebuild,** every kernel (`sass-A-vs-rebuild.txt`; `pr435_sass_identity.py` without the rename rule,
   both FKL b10fc66): 23 of 23 kernels identical.
4. **The host code of A against the rebuild,** every function, with the timed intervals (`host-A-vs-rebuild-<program>.txt`;
   `host_compare435.py --no-rename --intervals`): every function identical, 300, 252 and 234 per program.

**Why not byte for byte.** The rebuild does not reproduce #63's B's sha256 (`rebuild-old/rebuilt-old-sha256.txt` against
`old-manifest-63B.txt`):
- nvcc's anonymous-namespace module ids differ (`module-ids.txt`);
- the rebuild and A have the same ids although their folders differ;
- #63's B had others.

These ids appear in the kernel and symbol names, and the binaries differ with them. The comparisons above mask them.

**So "A against old" is the same timed code:** the same device code (checks 2 and 3) and host code (check 4) as a build of old's source in
old's folder with old's configuration (check 1).

## B against A: the kernels (`resources/`)

`pr435_sass_identity.py` with FKL #435's rename rule, and `cuobjdump -res-usage` (13.4.3, sm_89): 23 kernels in each build, all paired
(`sass-A-vs-B.txt`, `.csv`, `res-usage-{A,B}.txt`).

| program | kernels | identical | different (all LinearFilterDPP) | registers A / B | stack, LDL, STL | constant bank 0 | instructions B against A |
|---|--:|--:|--:|---|---|---|---|
| `e18_box_cu` | 9 | 1 (gpuSpin) | 8 | equal (39 or 40) | 0 in both | equal | equal in 5, +8 in 3 |
| `e18_box_split_cu` | 7 | 2 (gpuSpin, TransformDPP) | 5 | equal (39 or 40) | 0 in both | equal | equal in 4, +8 in 1 |
| `e18_stencil_cu` | 7 | 3 (gpuSpin, TransformDPP 2) | 4 | equal (40) | 0 in both | equal | equal in 4 |

- **The differences come with #435.** That PR passes the filter DPPs' details as a `__grid_constant__` kernel parameter and launches every
  DPP through `launchDPP_Kernel` (`../fkl-pr435-check/`).
- No cause of a time difference is claimed from this.

## The window (`window-script.sh`)

(To be run on the coordinator's GRANTED; about 10 minutes with the idle waits. #63's took 5 min.)

**The protocol is #63's.** The programs run in the order `e18_box`, `e18_box_split`, `e18_stencil`. For each program, one process per run:
- `warmup-A`, then `warmup-B` (not used);
- then `quiet-run1-A`, `quiet-run1-B`, `quiet-run2-A`, `quiet-run2-B`, `quiet-run3-A`, `quiet-run3-B`.

That is 24 runs. Each program keeps its own protocol: the checks, then `whole_call` and `kernel_only` of every row, in its own order. No
environment variable is set, and there is no nsys or ncu pass.

**Before every run**, the script waits for an idle machine: no build, compile job or measurement, a 1-minute load below 0.5, and no other
program on the RTX 4090 (up to 300 s).

**After every run** (RT-1's check, which #63 did not have), it checks again for a build, a compile job or another program on the RTX 4090. If
one appeared during the run, the window stops, and that run is the affected one, to be redone.

**The runs:** each is pinned to cores 0-3,16-19, with a 180 s timeout. Each must exit 0, print its checks line and write every CSV complete
(no `# INCOMPLETE`, no `-nan`); the window stops at the first failure. nvidia-smi reads the GPU every 100 ms.

**The guards before the start** (`GUARD_ONLY=1` runs only these):
- no blocked program;
- the committed manifest, naming the six binaries, all matching;
- A carries FKL b10fc66 and B 304492a, with no `-dirty` commit;
- HEAD differs from main 8c419c7 only under `results/`;
- the script is the committed one, run from its place;
- a new output folder;
- no build, compile job, measurement or other program on the RTX 4090.

## After the window (`scripts-used/after_window.sh`, CPU only)

1. **`rt4_report.py`** is #63's `retime_report.py`, with old = #63's B (`../e18-retime-b10fc66/window/<program>/quiet-run1..3-B/`). It writes
   `analysis/`:
   - **The numbers:** the median of the three quiet runs' means, with the min-max and the spread, of old, A and B.
   - **The changes:** B against A, B against old and A against old, each with whether it exceeds the larger spread, whether the ranges
     overlap, the bound and the flag. The bound and flag rules are #63's: GPU-bound when kernel_only is within 10 % of whole_call and its host
     preparation below 1 % of its GPU time; the flag is above 3 % (GPU-bound) or 5 % (host-bound).
   - **The cross-check:** old is checked against #63's `analysis/retime-summary.csv`, its B columns.
   - **Also:** the clocks per run, and every run's check CSVs against #63's B quiet-run1-B.
2. **`first_row_clock.py`** (#63's) gives the SM clock at the start of each run's timing.
3. **`rt4_formats.sh`** writes `formats/A/` and `formats/B/`: each side's outputs in the three E18 records' own formats, by the records'
   own scripts, unchanged:
   - `../e18-box/scripts-used/e18_box_report.py`;
   - `../e18-box-split/scripts-used/e18_box_split_report.py`, with `--old` = the same side's `e18_box_summary.csv`;
   - `../e18-stencil/scripts-used/e18_report.py`.

**Tested on #63's window** (in scratch, with `window/` a link to `../e18-retime-b10fc66/window`):
- old equals #63's B summary;
- the formats of #63's B give #63's B medians in all 46 rows.

No cause is named unless shown.

## Files

| path | content |
|---|---|
| `binaries.sha256` | A and B, the six binaries the window runs |
| `window-script.sh` | the window's script |
| `code-check/` | A against old: the console, the rebuild of old (`rebuild-old/`: console, sha256, #63's manifest lines, the report, configure and build logs, target lists, time), the configure-log diff, the line counts against #60's record, the SASS and host code of A against the rebuild, the module ids |
| `resources/` | B against A: the SASS comparison and its CSV, `cuobjdump -res-usage` of both |
| `scripts-used/` | `rebuild_old.sh`, `check_fkl_tree_rt4.sh`, `rt4_code_check.sh`, `sass_linecounts_vs_60.py`, `pr435_sass_identity.py`, `sass_identity_norename.py`, `host_compare435.py`, `rt4_report.py`, `first_row_clock.py`, `rt4_formats.sh`, `after_window.sh` |
