# E3 controlled compile repeat at the pin FKL 304492a, against the repeat at the head of FKL PR #423 (9e20bd4e) (RT-3)

**Dated note, 2026-11-27.** The paper prints no numbers from this record. Its compile times come from `../repeat-pr423-9e20bd4e/` (Oscar's answers of 2026-11-27).

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

**State.** The window ran on the coordinator's GRANTED (HOLD COMPILES and WINDOW STARTS sent) from 2026-11-27 05:50:06 to 06:42:49
(+01:00; 04:50:06Z to 05:42:49Z, `window/window-console.txt`), 52 min 43 s. The repeat took 52 min. The tables below were made afterwards,
on the CPU only.

## Result

**The window** (`window/`):
- **Compiles:** all 25 ok: the warm-up (05:50:06-05:50:11), then the sweep, 24 compiles (05:50:26-06:42:44).
- **The guards:**
  - the watcher never fired: there is no `watcher.log` and no `DISTURBED`;
  - `still_quiet` never fired;
  - `wait_idle` passed before both steps, at once before the warm-up and about 15 s before the sweep, from the timestamps. The console's
    start lines are cut at 300 characters, which cuts the idle note off.
- **The machine** (`load.log`, every 20 s, 159 samples):
  - the other cores of the compile CCD were at most 1 % busy (the repeat: at most 1 %);
  - on the other CCD, the desktop's file indexer `localsearch-3` showed 20 % or more in 9 samples. It is not in the repeat's log;
  - the governor was `performance`, with boost on.
- **The translation units:** all eight are byte-identical to the repeat's (`window/tu-sha256-vs-9e20bd4e.txt`).

**Old (9e20bd4e) against new (304492a)** (`window/e3_repeat_vs_9e20bd4e.md` and `.csv`, by `scripts/e3_repeat_pair.py`):
- **The form:** the median of three compiles each.
- **spread:** the larger of the two folders' (max - min) / median.
- **exceeds:** |change| > spread.
- **Flag:** 5 %, because a compile is host-bound.

| N | wall 9e20bd4e (s) | wall 304492a (s) | the new compiles | spread % | change % | exceeds? | > 5 %? | peak memory 9e20bd4e -> 304492a (MiB) | change % | exceeds? |
|---:|---:|---:|---|---:|---:|---|---|---|---:|---|
| 1 | 2.596 | 2.709 | 2.693 / 2.714 / 2.709 | 1.39 | +4.35 | yes | no | 291 -> 298 | +2.41 | yes |
| 100 | 3.078 | 3.200 | 3.200 / 3.202 / 3.194 | 0.25 | +3.96 | yes | no | 345 -> 352 | +2.03 | yes |
| 500 | 7.683 | 7.923 | 7.837 / 7.923 / 8.118 | 3.55 | +3.12 | no | no | 691 -> 699 | +1.16 | yes |
| 1,000 | 22.364 | 22.804 | 22.727 / 22.895 / 22.804 | 0.74 | +1.97 | yes | no | 1,470 -> 1,478 | +0.54 | yes |
| 1,500 | 50.903 | 51.266 | 51.412 / 51.219 / 51.266 | 0.38 | +0.71 | yes | no | 2,598 -> 2,608 | +0.38 | yes |
| 2,000 | 96.686 | 97.228 | 97.189 / 97.721 / 97.228 | 0.55 | +0.56 | yes | no | 4,073 -> 4,082 | +0.22 | yes |
| 3,000 | 266.403 | 266.460 | 266.460 / 266.761 / 266.346 | 0.94 | +0.02 | no | no | 8,070 -> 8,080 | +0.12 | yes |
| 4,000 | 579.593 | 587.751 | 586.417 / 587.751 / 588.363 | 0.33 | +1.41 | yes | no | 13,465 -> 13,475 | +0.07 | yes |

- **Wall time:** no length is above the 5 % flag.
  - The change is +4.35 % at N = 1, falling to +1.97 % at 1,000, then +0.71, +0.56, +0.02 and +1.41 % at 1,500 to 4,000.
  - It exceeds the run-to-run spread at 6 of the 8 lengths, the exceptions being N = 500 and 3,000.
  - The printed values at N = 2,000 / 3,000 / 4,000 become 97.2 / 266 / 588 s, against 96.7 / 266 / 580 s.
- **Peak memory:** +0.07 % to +2.41 %, below 5 % and more than the spread at every length. At N = 4,000 it is 13,465 MiB at 9e20bd4e
  and 13,475 MiB here.
- **The phases** (medians): `cudafe++` is +3.7 % to +5.0 % at every length (0.832 → 0.874 s at N = 1, 0.040 s at the most).
  - Three phase values are above 5 %: `cudafe++` at N = 1 (+5.05 %), and `ptxas` at N = 1 (+8.33 %, 0.012 → 0.013 s) and 1,500 (-5.26 %).
  - `cicc` is +0.0 % to +4.5 %.
  - The host compiler is +0.6 % to +4.9 %, with +4.9 % at N = 4,000.
- **The generated code is the same:** at every length, the registers, spill stores and loads, stack, constant bank 0, SASS instructions
  and `.text` bytes are equal.
- **The cubin is larger at every length:** +896 B at N = 1 (+4.96 %), down to +194,048 B at N = 4,000 (+0.10 %). The `.text` sections are
  the same size, so the bytes are in other sections; which ones is not examined here.

**What differs between the two sets.** None of these is separated by this set, so **no cause is shown**:
1. **The library:** 9e20bd4e → 304492a, the FKL #431 to #435 changes, with #435's `launchDPP_Kernel` names.
2. **TMPDIR:** on disk (ext4) here, where the repeat's temporaries were in /tmp, the RAM disk. This is Oscar's standing rule.
3. **The state of the machine:** 21 days apart, kernel 7.0.0-43 against 7.0.0-38, and the desktop's file indexer on the other CCD.

The coordinator: a control at a flagged length (the 9e20bd4e tree with TMPDIR on disk) can follow in its own window. No length is flagged
for the wall time.

**The repeat's tables of this set** (`window/e3_repeat_summary.csv` / `.md`, `window/e3_table_chains.csv`, by `scripts/e3_repeat_table.py`)
are in the repeat's format, for the paper's package.

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

The memory is in MiB throughout (GNU time reports KiB; the sweep divides by 1024), never MB or GB. No cause is named unless shown.

## Files

| path | content |
|---|---|
| `window-script.sh` | the window's script |
| `scripts-manifest.sha256` | the scripts the window runs |
| `scripts-used/after_window.sh` | the tables after the window |
| `window/` | the window: console, environment before and after, `load.log`, the copy of the script it ran, `e3_repeat_warmup_A.csv`, `e3_repeat_A.csv`, `nvcc-time/` (the `nvcc --time` file of every compile; the warm-up's has the name of the first N = 1 compile and was overwritten by it, as in the repeat), `tu-sha256.txt`; and after the window `e3_repeat_summary.csv` / `.md`, `e3_table_chains.csv`, `e3_repeat_vs_9e20bd4e.csv` / `.md`, `tu-sha256-vs-9e20bd4e.txt` and the two scripts' stdout |
