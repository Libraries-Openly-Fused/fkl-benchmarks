# CPU back-end results: Ryzen 7 5800X3D

Results of the CPU back-end of FKL (`ParArch::CPU` and `CPU_MULTITHREAD`) against OpenCV on the Ryzen 7 5800X3D (8 cores, 16 threads, 60 GiB), g++ 13.4.0 and clang++ 23.1.5, OpenCV 4.14.0 with IPP:

- `e5-preprocessing-a/`: pipeline (a) with FKL `fb9ac3a` (before #414 and #416); `e5-preprocessing-a-c8fcbe8/`: the same with FKL `c8fcbe8`, the pin of this repository until 2026-11-07 (the README of that folder says which rows to cite); `e5-preprocessing-c/`: the resolution sweep (c).
  The comparison names of the agreement CSVs of pipelines (a) and (c) that have a comma are in double quotes: the files written before the benchmarks quoted them were repaired afterwards, and nothing else in them changed (`agreement-csv-quoting.txt` lists the files with the sha256 of each before and after; read them with `skipinitialspace=True`, the fields are separated by a comma and a space).
- Removed: the NV12 video pipeline (pipeline (b): the folders `e4b-nv12-portable/`, `e5-preprocessing-b/`, `e5-preprocessing-b-c8fcbe8/` and `e5-preprocessing-b-c8fcbe8-repeats/`) is not in the paper.
  Its code and results are in the git history up to `341ecd1`, for example `git show 341ecd1:results/cpu-5800x3d/e5-preprocessing-b-c8fcbe8/README.md`.

Pins: each folder names the FKL commit it was measured at. The pin of this repository was c8fcbe8 until 2026-11-07, then 64ad9195 (c8fcbe8 plus #422, #426, #423 and #427) until 2026-11-09; since then it is 1da77899, 64ad9195 plus #431, the image-processing DPPs (`../rtx4090/final-checks/README.md`). The device-code comparisons of c8fcbe8, 9e20bd4e, 64ad9195 and 1da77899 (`../rtx4090/fkl-pr423-check/`, `../rtx4090/fkl-pr427-check/`, `../rtx4090/fkl-pr431-check/`) cover the CUDA programs on the RTX 4090 only, not the CPU builds of this folder.

## Machine and software (read on 2026-11-01)

The results of this folder were committed from 2026-10-30 to 2026-11-01. The machine has been in one boot since 2026-10-26 18:39 CET, and its software state read on 2026-11-01 is described in `../rx9060/README.md` (section "Machine and software"): Ubuntu 26.04.1 LTS,
kernel 7.0.0-38-generic, `g++-13` 13.4.0 (Ubuntu 13.4.0-10ubuntu1), `clang++-23` 23.1.5 (Ubuntu build `++20261023092600+395ba2c4d274`), governor `powersave` (amd-pstate-epp, EPP balance_performance, boost on). The packages of the kernel, of `g++-13`, of `clang++-23` (upgraded to this build at 18:37 on 2026-10-26), of the
amdgpu driver and of ROCm have not changed since 2026-10-26 at 18:37; on 2026-10-29 at 20:52 `clang-21` (21.1.8) was installed next to `clang-23` and `lldb-23` was removed, and the default `g++` of the machine is 15.2.0 (the results use `g++-13` and `clang++-23`; the line `host compiler` of each CSV has the version).
The CSVs of this folder have no `# os:` or `# kernel:` line in their headers.
