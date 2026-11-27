# E3 controlled compile repeat at the pin FKL 304492a, against the repeat at the head of FKL PR #423 (9e20bd4e) (RT-3)

**What.** The E3 compile times, the paper's section VI-L, measured again at the pin of fkl-benchmarks after the move to FKL #435:
- #435 was merged as `304492af4d777165bd556fdd6113439635724d02`;
- the pin move is #66.

The paper's compile-time table comes from `../repeat-pr423-9e20bd4e/` (2026-11-06). This set uses that repeat's protocol exactly, with
`scripts/e3_sweep.py` unchanged: the distinct-operations chain (variant A) at N = 1, 100, 500, 1,000, 1,500, 2,000, 3,000 and 4,000, three
compiles each, the median, and the same columns. The comparison is old (9e20bd4e) against new (304492a); there is no A side.

**Decision.** Task RT-3 of the coordinator, 2026-11-27. Oscar's own answer of 2026-11-26 (re-time question 4 of 5, "compile times, VI-L":
FKL's templates changed and #435 adds one, so the identity argument does not apply to compile times) was "Yes, re-measure them". The option
was "One quiet compile-timing run of the E3 chains at the new pin, with the original repeat's protocol and repetitions, reported as
old/new/spread".
- **An earlier decision:** "Do not repeat the compilation times due to issue 19" (2026-11-06). The coordinator reports it to Oscar.
- **The paper** keeps the pr423 table unless Oscar decides otherwise; this set is data.

Oscar's rule of 2026-11-26 applies too: "Coordinate with fkl-reduce, fkl-reduce has priority". The window runs only in a quiet slot agreed
with that session.

**State.** The window script is prepared and was tested on a tiny input. The window waits for the coordinator's GRANTED.

## The protocol (`window-script.sh`)

**The repeat's** (`../repeat-pr423-9e20bd4e/window-script.sh`):
- **The order:** one warm-up compile at N = 1 (`e3_repeat_warmup_A.csv`, kept, not used), then N = 1, 100, 500, 1,000, 1,500, 2,000, 3,000
  and 4,000, three compiles each in a row, ascending (`e3_repeat_A.csv`).
- **Each compile:** ONE core (core 8, the second CCD), `nvcc --threads 1 -O3 -DNDEBUG -std=c++17
  --generate-code=arch=compute_89,code=[compute_89,sm_89] -ftemplate-depth=1000 -lineinfo --resource-usage`, the flags of
  `scripts/e3_sweep.py`.
  - GNU time measures the wall, user and system time and the peak resident memory.
  - `nvcc --time` gives the phases (`nvcc-time/`), and `cuobjdump` the size of the cubin.
- **Everything else of mine** runs on cores 4-7,20-23 (the first CCD): the python parent of the sweep, cuobjdump, the watcher. The load
  logger runs on core 5, every 20 s, with the second CCD's load.
- **The tools:** CUDA 13.4.3 (nvcc 13.4.92), g++ 15.2.0 and `venv-torch`'s python.

**What differs from the repeat:**
1. **The library:** this branch's `external/fkl`, the pin 304492a, a clean submodule checkout. It is named in the CSV headers through
   `FKLB_FKL_TREE`, as the repeat named its clone of the PR #423 head.
2. **TMPDIR is on disk** (`scratch/rt3/tmp`, ext4), for nvcc and the sweep.
   - The repeat set none, so its temporaries were in /tmp, the RAM disk; its README notes it.
   - Oscar's standing rule (2026-11-03): no RAM disk, TMPDIR on disk.
   - The coordinator: if a length is flagged, a control at that length (the 9e20bd4e tree with TMPDIR on disk) can follow in its own window.
3. **The generated translation units** go to `scratch/rt3/work`. The repeat's went to the reference checkout's `build/e3-repeat-pr423`.
4. **The checks around the compiles** (Oscar's rule of 2026-11-26):
   - `wait_idle` before the warm-up and before the sweep: no measurement, build or compile job, no other program on the RTX 4090, and a
     1-minute load below 0.5, up to 300 s;
   - `still_quiet` after each;
   - a **watcher** during both, every 5 s. A build or compile job that is not one of the sweep's own processes, or a program on the RTX
     4090, stops the sweep at once. The window ends there, and `watcher.log` says what appeared and which compile was running; that compile
     and the ones after it are to be redone.

**The scripts the window runs** are in `scripts-manifest.sha256`, checked by the window:
- `scripts/e3_sweep.py`, `e1_table.py`, `e1_sass.py`, `sass_tools.py`, `common/fklb_csv.py`, `fkl-self/compiletime/gen_chain.py`;
- `load_logger.py`, `quiet_check.sh`, `os_kernel_lines.sh`;
- `e3_repeat_table.py`, `e3_repeat_pair.py`.

Since the repeat's commit b3c2153 only `scripts/e1_sass.py` changed, by one line: its `classify_kernels` accepts #435's kernel names for
E1. `e3_sweep.py` uses only `e1_table.text_section_sizes`, which does not call it. `e3_sweep.py` finds the kernel without names: the
largest SASS function, the first ptxas block, and the largest `.text` section.

**The guards before the start** (`GUARD_ONLY=1` runs only these):
- the script is the committed one, run from its place;
- HEAD differs from main d66e854 only under `results/`;
- the scripts match their manifest;
- `external/fkl` is at 304492a and clean;
- nvcc is 13.4.92;
- new output and work folders;
- no build, compile job, measurement or other program on the RTX 4090.

The window runs no GPU program and executes nothing: the sweep only compiles.

**The tiny test** (2026-11-27 05:44:28 to 05:44:50, scratch only; announced to the coordinator): the whole committed script ran with
`E3_NS=1 E3_REPS=1`.
- **What it showed:**
  - the guards, the warm-up, the sweep, the watcher, the CSVs, the phase files, the environment files and the unit hashes all work;
  - the TMPDIR folder was on ext4 and empty afterwards;
  - the N = 1 unit is byte-identical to the repeat's (sha256 7108e409...);
  - its registers (10), constant bank 0 (416 B), SASS instructions (32) and `.text` bytes (512) equal the repeat's; its cubin is 896 B larger
    (18,976 against 18,080 B).
- **Not a result:** its times are one compile outside a quiet window.
- `e3_repeat_table.py` and `e3_repeat_pair.py` were run on its output.

## After the window (`scripts-used/after_window.sh`, CPU only)

The repeat's own scripts, unchanged:
1. `scripts/e3_repeat_table.py window/` → `window/e3_repeat_summary.csv` / `.md` and `window/e3_table_chains.csv`, the repeat's tables;
2. `scripts/e3_repeat_pair.py ../repeat-pr423-9e20bd4e window/` → `window/e3_repeat_vs_9e20bd4e.csv` / `.md`, in the standing form:
   - old, new and the values in the order they ran;
   - the spread, the change, and whether it exceeds the spread;
   - the 5 % flag of a host-bound compile;
   - per length and quantity: the wall time, the peak memory in MiB, the phases, and the code quantities;
3. the sha256 of the eight translation units against the repeat's (`window/tu-sha256-vs-9e20bd4e.txt`).

The memory goes in MiB as recorded, and in GiB (MiB / 1024) where printed, never GB. No cause is named unless shown.

## Files

| path | content |
|---|---|
| `window-script.sh` | the window's script |
| `scripts-manifest.sha256` | the scripts the window runs |
| `scripts-used/after_window.sh` | the tables after the window |
