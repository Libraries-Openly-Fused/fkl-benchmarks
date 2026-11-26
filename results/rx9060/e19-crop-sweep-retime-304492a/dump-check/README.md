# RT-A1's dump check: the output tensors of A and B, for e19_crop_sweep_differences.csv (2026-11-26, 19:04:48-19:06:03Z)

The coordinator chose a measured dump check (option (b), 2026-11-26) for `../paper-data/{A,B}/e19_crop_sweep_differences.csv`, which compares
fkl_batch's tensor with rpp's. The timing window wrote no tensors (its protocol unsets `E19_DUMP`).

- **The protocol** is the checks step of `../../e19-crop-sweep-phase1/window-checks.sh`, unchanged. At each of the 16 N, in ascending order,
  there is one run with the checks only (`E19_NO_TIMING=1`) and `E19_DUMP`, which writes the run's five tensors. A's 16 runs come first,
  then B's. Nothing is timed.
- **The binaries** are those of the timing window (`../launch.sh`): A 9e7a4e0f…, B 842ca530….
- **The guards** are those of `../window.sh`, plus a new dump folder and the five tensors written by every run:
  - before every run: no `Runner.Worker`, a load below 0.8, an empty amdkfd process list;
  - the first failure ends the window.
- **The dumps** (about 1.1 GB per binary) go to `/home/cudeiro/rocm-fkl-benchmarks/tmp/rt/dumps-304492a/{A,B}/nNNN`, outside the repository.
  Their sha256 and sizes will be in `dump-sha256-{A,B}.txt`.
- **`launch-dumps.sh`** runs it as one detached call after the coordinator's GRANTED. `GUARD_ONLY=1 launch-dumps.sh` passed on 2026-11-26 at
  20:02:17+01:00.
- **`mock_dumps_test.sh`** (output `mock-dumps-test.txt`) runs the script against stand-in binaries, so nothing opens the GPU. It shows:
  - 32 runs, A's then B's, each with only `E19_CROPS`, `E19_NO_TIMING=1` and `E19_DUMP` set;
  - a run that writes four tensors stops the window;
  - an existing dump folder is refused;
  - a listed GPU process blocks the first run.
- **After the window,** `../analysis/after_dumps.sh` writes:
  - the dumps' sha256, and how many equal the record's (`../../e19-crop-sweep-phase1/analysis/dump-sha256.txt`);
  - phase 1's prefix check and `between.py` for each binary;
  - the check rows against phase 1's check runs;
  - the two `e19_crop_sweep_differences.csv` files.

  Its listing reproduces the record's `dump-sha256.txt` line for line, 80 of 80, on the record's own dumps (still in `tmp/e19sweep/dumps`).

## Result

The coordinator checked the scripts on #71 (09fa17b, by sha256) and granted the window. `launch-dumps.sh` ran it as one detached call from
19:04:48 to 19:06:03Z, exit 0 (`window/window-console.txt`).
- **All 32 runs exited 0,** printed "E19 checks passed" and wrote their five tensors. 2 runs started after an idle wait (15 s and 30 s).
- **The dumps are the record's, byte for byte.** All 80 files of A and all 80 of B have the sha256 and size of the record's binary's dumps
  (`dump-sha256-{A,B}.txt` against `../../e19-crop-sweep-phase1/analysis/dump-sha256.txt`: 80 of 80 lines equal).
- **Every N's dump is the first N planes of N = 150's,** in A and B (`dump-prefix-{A,B}.txt`), as in phase 1.
- **`between.py` gives phase 1's `between.csv`** for A and for B (`between-{A,B}.csv`):
  - fkl_batch against rpp: at most 0.00774652 output units (0.444 grey levels) at every N; all values differ except 0 to 7 per N;
  - fkl_loop against fkl_batch: as in phase 1.
- **The check rows** of the 16 check runs of A and of B equal those of phase 1's check runs (`checks-identity.txt`).
- **`../paper-data/{A,B}/e19_crop_sweep_differences.csv`** are written from `between-{A,B}.csv`, in the coordinator's layout. Since the
  tensors are the record's, the two files hold the record's values.

The dumps (1.1 GB per binary) stay in `/home/cudeiro/rocm-fkl-benchmarks/tmp/rt/dumps-304492a`, outside the repository.
