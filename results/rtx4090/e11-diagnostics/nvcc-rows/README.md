# E11: the nvcc rows made on the RTX 4090 machine

The nvcc column of `../../rx9060/e11-diagnostics/` was compiled on the AMD machine; this folder repeats it here, as that README asks, with the corrected `scripts/e11_diagnostics.py` of main (3809465): the same three wrong chains
(`fkl-self/diagnostics/e11_*.cu`), the same three FKL trees (c8fcbe8 as it is, with the first commit of the chain check `chain-check-types-only.patch`, with the prototype `chain-check-prototype.patch`; the patches are in the AMD folder and were applied to a copy of `external/fkl` at c8fcbe8),
nvcc 13.4.92 (the CUDA 13.4.3 toolkit), `-arch sm_89`, **compile only** (no GPU job; the programs are not run). 9 rows: `e11_diagnostics.csv` (the columns of the AMD file; the label is the tree, as there), `logs/` the full output of the wrong chain of every cell.

The host compiler here is g++ 15.2.0 (`--ccbin /usr/bin/g++`); the AMD rows used g++ 13.4.0, which is not installed on this machine. Machine: Ryzen 9 5950X; wall times are the median of 5 and are in the CSV only.

**Result: every analysis column of the 9 rows (`outcome`, `exit`, `lines`, `errors`, `notes`, `static_assert`, `first_error`, the three `names_*` columns, `points_to_user_line`, `control_ok`) is identical to the nvcc row of the AMD machine for the same tree and chain,** although the host compiler and the machine differ. The wall times differ (1.8 to 2.5 s here).

| tree | type mismatch | read in the middle | batch of 50 read, 60 written |
|---|---|---|---|
| c8fcbe8 as it is | error, 24 lines, 1 error, 2 notes | error, 27 lines, 1 error, 3 notes | compiles, with output (154 lines, 25 notes) |
| + chain check, types only | `static_assert`, 15 lines, 1 error | error, 27 lines, 1 error, 3 notes | compiles, with output (154 lines, 25 notes) |
| + prototype (all) | `static_assert`, 16 lines, 1 error | `static_assert`, 14 lines, 1 error | `static_assert`, 14 lines, 1 error |

Reproduce (from a checkout of main; the trees as above): `python scripts/e11_diagnostics.py --compilers nvcc --ccbin /usr/bin/g++ --label <unchanged-c8fcbe8|chain-check-types-2d10e5c|chain-check-all-d8d00bf> --include <tree>/include --out <folder> --note "<where and how>"`.
The logs hold the paths of the trees that were used.
