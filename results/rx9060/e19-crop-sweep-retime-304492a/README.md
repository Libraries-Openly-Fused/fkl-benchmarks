# E19 crop sweep re-time at FKL 304492a (the merge of FKL #435): RT-A1, 2026-11-26, 18:15:47-18:50:38Z, RX 9060

> **Dated note, 2026-11-27.** Side B of this record (FKL 304492a, `paper-data/B/`) backs the paper's RPP comparison sentence and the chart's RX 9060 line, in place of `../e19-crop-sweep/` (Oscar's answers of 2026-11-27). The text below, which says that the paper's RX 9060 stencil factors come from `../e19-crop-sweep/`, was written before this note.

`rpp/e19_crop_sweep.h` is E19 at N = 1, 10, 20, ..., 150 crops on the RX 9060. Its record is `../e19-crop-sweep/` (2026-11-24, fkl-benchmarks
c8eb575, FKL 24478f0); the paper's RX 9060 stencil factors come from it. The pin move to FKL #435 (#66, merged as af96aef with FKL 304492a)
left the device code identical but changed the timed host code of the fkl_batch row: the row's lambda, FKL's entry function and the new
`requireArguments` (`../pin-move-devcode-b10fc66-48fbf3f/`, #67). Oscar ordered this re-time on 2026-11-26, through the coordinator session.

The coordinator checked the scripts on #71 (cffcdcd, by sha256) and granted the window. `launch.sh` ran it as one detached call from
18:15:47 to 18:52:43Z; RT-A1 took 18:15:47-18:50:38Z, then the E18 re-time RT-A2 (`../e18-box-retime-304492a/`) ran.

Decision: Oscar's re-time order of 2026-11-26 (RT-A1 and RT-A2), relayed by the coordinator session that day.

## The two binaries

| | sources | e19_crop_sweep_hip sha256 | build strings |
|---|---|---|---|
| **A** | fkl-benchmarks 976e42b, FKL b10fc66: #67's base build (`tmp/vbuild/pin435-base-hip`, `pin_build.sh`, a git archive) | `9e7a4e0f758ea70365ed6382c1b16f0e9fcf08475972a6d9e7eba86cc82ddfa7`, as in #67's `binaries-sha256.txt` | FKL b10fc66; fkl-benchmarks `unknown` |
| **B** | fkl-benchmarks main 73a9650, FKL 304492a: a clean worktree, `step0/rt_build.sh` | `842ca53052ef19e0ae99eec01f405f26ddb590a226cdb4e384c4b8932ed2c5e9` (`step0/binary-sha256.txt`) | 73a9650 and 304492a, no `-dirty` |

- **B was built from a clean git worktree** of 73a9650 (`git status` empty), with `external/fkl` at 304492a and `external/FastRPP` at efbf009,
  both clean (`step0/source-state.txt`). The binary therefore records its commits, as the record's binary recorded c8eb575.
- **The build options are the record's.** B's CMake cache and its four build commands of `bin/e19_crop_sweep_hip` equal A's and those of the
  record's build (`tmp/vbuild/e19sweep-clean`, from the clean worktree of c8eb575), with the folders replaced: Release, gfx1200,
  amdclang++ 24 (ROCm 10.1.0), `-O3 -DNDEBUG -std=c++17`, `-march=native` for the host file, RPP 3.2.0. 0 errors, 0 warnings.

## The code (compile only; `code-checks/`, `step0/code_checks.sh`)

- **A against the record's binary** (f9fcb2a4..., the binary that `../e19-crop-sweep/` timed):
  - device code: 19 of 20 functions identical. The one that differs is fkl_loop's LinearFilterDPP kernel, whose details lost the image size in
    FKL #433. All 16 fkl_batch kernels are identical.
  - host code: no function reached from the fkl_batch or rpp timed calls, or from the timing harness, differs. What differs is fkl_loop's
    lambdas and stub, and `fklb::writeSystemInfoHeader` (the build strings).
  - So A repeats the record for fkl_batch and rpp.
- **B against #67's head build** (a4b493c with FKL 48fbf3f): device code 26 of 26 functions identical, also with labels as symbols; host code
  621 of 622 functions identical, the exception being `fklb::writeSystemInfoHeader` (the build strings). #67's results therefore hold for B.
- **A against B:** device code identical (20 of 20, paired by #67's task C rename). The host code differs as #67 section 2 describes: the
  fkl_batch row's lambda, `executeMorphology` against `executeOperations<MorphologyDPP<GPU_AMD, ...>>`, and `requireArguments`.

## The resources of fkl_batch's kernel (`resources/`, `step0/e19_resources.py`)

From each binary's gfx1200 code object, at every N, for the record's binary, A and B:

| | every N, in all three binaries |
|---|---|
| instructions | 308 |
| VGPRs (allocated) | 10 (16) |
| SGPRs | 38 |
| LDS | 1,232 bytes |
| scratch | 0 bytes, no dynamic stack |
| spills | 0 VGPR, 0 SGPR |
| kernel arguments | 352 + 24 N bytes (376 at N = 1, 3,952 at N = 150) |

At each N the kernel's listing has the same sha256 in the record's binary, in A and in B: 16 of 16 identical in each comparison. The
listings of different N differ, in the offsets into the kernel arguments. `resources/e19-fkl-batch-resources.md` has every row.

## The window (`launch.sh`, `window.sh`)

`launch.sh` runs RT-A1 (`window.sh` of this folder) and then, only if it ended with exit 0, RT-A2 (`../e18-box-retime-304492a/window.sh`).
It is run as one detached call after the coordinator's GRANTED. No other tool call is made until it exits.

- **The protocol is the record's own** (`../e19-crop-sweep/window.sh`), unchanged.
  - A sweep is 16 processes, one per N in ascending order, each a complete run: the checks, then the 2 s clock warm-up spin, then fkl_loop
    (for the order only), fkl_batch and rpp, each whole_call then kernel_only, with sclk before and after every interval.
  - The same rocprofv3 command, and the same hwmon sampler every 100 ms.
- **A and B alternate, sweep by sweep:**
  1. `unused-A`, `unused-B`;
  2. `quiet-run1-A`, `-B`, `quiet-run2-A`, `-B`, `quiet-run3-A`, `-B`;
  3. `rocprofv3-A`, `rocprofv3-B`, whose times are not used, as in the record. They give the kernels per call and the spin coverage.

  That is 160 runs. The record's 80 took 14 minutes, so this window should take about 30, and E18 about 2 more.
- **Guards before the start:**
  - the manifest (both binaries and `window.sh`);
  - the binaries' names and sha256;
  - A carries FKL b10fc66; B carries FKL 304492a and fkl-benchmarks 73a9650; neither carries a `-dirty` commit;
  - the output folder is new.
- **Guards before every run:**
  - no `Runner.Worker`;
  - a 1-minute load below 0.8;
  - no process holding the GPU: the amdkfd process list `/sys/class/kfd/kfd/proc` is empty, so no other ROCm program, FastRPP's
    included, is running.

  If the machine is not idle within 600 s, the window stops.
- **The first failure ends the window:** a run that does not exit 0 with "E19 checks passed", or whose CSVs hold `# INCOMPLETE` or `-nan`.
- **`GUARD_ONLY=1 launch.sh`** runs the guards of both scripts only. Run on 2026-11-26 at 19:08:11+01:00, it passed both: no
  `Runner.Worker`, load 0.14, no GPU processes.
- **`step0/mock_window_test.sh`** runs both scripts against stand-in binaries, with rocminfo and rocprofv3 replaced by stubs, so nothing
  opens the GPU. Its output is `step0/mock-window-test.txt`. It shows:
  - both windows run their runs in the protocol's order and number (E19: 10 sweeps of 16, each run with only `E19_CROPS` set; E18: 10 runs);
  - a failing run stops the window;
  - the guards refuse an existing folder, a wrong sha256 and a wrong commit, and create nothing;
  - a listed GPU process blocks the first run until IDLE_MAX.

## Result

- **All 160 runs exited 0 and printed "E19 checks passed".** No CSV has `# INCOMPLETE` or `-nan`.
  - 69 runs started after an idle wait: 56 of 15 s, 12 of 30 s and 1 of 45 s. The back-to-back runs raised the 1-minute load past 0.8, as in the
    record. No wait reached IDLE_MAX.
- **The check rows are the record's.** Every run's checks CSV has the data rows of the record's quiet-run1 at the same N: 80 of 80 runs of A
  and 80 of 80 of B (`summary/checks-identity.txt`).
- **The kernels per call and the spin coverage are the record's** at every N, in A and B (`summary/kernels-retime.txt`, `analysis/kernels-and-spin-{A,B}.tsv`):
  - fkl_batch: 1 kernel per call; rpp: N + 3;
  - the spin covers 100 of 100 kernel_only iterations at every N and row. The smallest margin is 2.46 ms (A) and 2.37 ms (B), both rpp.
- **From N = 30 to 150, B is within -1.5 % to +0.8 % of A in every row and interval.** No change there is flagged. Two exceed the larger spread:
  - fkl_batch whole_call at N = 50: +0.66 %, against spreads of 0.58 / 0.65 %;
  - fkl_batch kernel_only at N = 60: -0.90 %.
- **A repeats the record within -1.8 % to +0.9 % from N = 30 on,** with no flag. Two changes exceed the larger spread: fkl_batch whole_call at
  N = 50 (-0.74 %) and rpp whole_call at N = 150 (+0.09 %).
- **The flags are all at N = 1 to 20,** where the three runs' means spread most. B against A has 2 flags, A against old has 4:

| N | row, interval | old (ms, spread) | A (ms, spread) | B (ms, spread) | B vs A | exceeds? | A vs old | exceeds? |
|--:|---|--:|--:|--:|--:|---|--:|---|
| 1 | rpp whole_call | 0.0231 (23.5 %) | 0.0258 (19.6 %) | 0.0251 (9.0 %) | **-3.0 %** | no | **+11.6 %** | no |
| 10 | fkl_batch whole_call | 0.0521 (3.5 %) | 0.0492 (9.0 %) | 0.0526 (11.3 %) | **+6.7 %** | no | **-5.5 %** | no |
| 10 | fkl_batch kernel_only | 0.0533 (4.4 %) | 0.0510 (6.0 %) | 0.0521 (8.7 %) | +2.2 % | no | **-4.3 %** | no |
| 20 | fkl_batch whole_call | 0.0808 (2.2 %) | 0.0770 (3.2 %) | 0.0786 (4.1 %) | +2.1 % | no | **-4.7 %** | **yes** |

  These are calls of 14 to 80 µs. Only A against old at N = 20 exceeds the larger spread. No cause is claimed.
- **rpp / fkl_batch, whole_call** (the median of the three per-run ratios):
  - from N = 30 to 150, B's ratio is within -0.9 % to +1.6 % of A's, and A's within -0.3 % to +2.7 % of the record's;
  - at N = 1, 10 and 20 it moves by -2.4 %, -8.0 % and -1.9 % (B against A).

  `summary/ratios-retime.csv` has every N for old, A and B, both intervals.
- **The clocks** (`summary/{A,B}/clock.csv`): in the quiet runs' timing, 9 of 691 samples (A) and 11 of 690 (B) are below 3.1 GHz, against 10
  of 692 in the record. The lowest are 3,007 MHz (A) and 2,974 MHz (B).
- **The unused runs:**
  - the warm-up sweeps: 3 of 64 values of each binary are more than 3 % from its quiet median;
  - the rocprofv3 sweeps run under tracing: 56 (A) and 54 (B) of 64 are more than 3 % above, as in the record (49 of 64). Their times are not used.

**How the summary was made,** without the GPU: `analysis/post_window.sh`. It runs the record's scripts on each rocprofv3 run (`calls.txt`,
`rpp-spin-with-copies.txt`, with the traces gzipped as in the record), builds `views/A` and `views/B` (each binary's runs under the record's
folder names), runs the record's report on each, then `analysis/retime_report.py`.

`summary/retime-report.md` has the four full tables (fkl_batch and rpp, whole_call and kernel_only, all 16 N, old / A / B) and the ratios.
`summary/A/` and `summary/B/` are the record's own report (`../e19-crop-sweep/analysis/e19_crop_sweep_report.py`) on each binary's runs.

## For the paper (`paper-data/`)

`paper-data/A/` and `paper-data/B/` (B is the new pin) hold the coordinator's three package files, in its headers, with gpu `rx9060`:
- `e19_crop_sweep_times.csv`: per N, row (fkl_batch, rpp) and interval, the median, min and max of the three quiet runs' means (µs) and their
  spread; for rpp, its ratio to fkl_batch (the median, min and max over the three runs);
- `e19_crop_sweep_kernels.csv`: the kernels of one call, from the rocprofv3 sweep: 1 for fkl_batch, N + 3 for rpp;
- `e19_crop_sweep_differences.csv`: fkl_batch's tensor against rpp's at every N, from the dump check (`dump-check/`, 19:04:48-19:06:03Z).
  The check runs' dumps of A and B are the record's byte for byte (80 of 80 files each), so these hold the record's values:
  0.00774652 output units, 0.444 grey levels, at every N.
