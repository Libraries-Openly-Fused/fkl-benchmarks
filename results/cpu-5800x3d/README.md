# CPU back-end results: Ryzen 7 5800X3D

Results of the CPU back-end of FKL (`ParArch::CPU` and `CPU_MULTITHREAD`) against OpenCV on the Ryzen 7 5800X3D (8 cores, 16 threads, 60 GiB), g++ 13.4.0 and clang++ 23.1.5, OpenCV 4.14.0 with IPP:

- `e5-preprocessing-a/`: pipeline (a) with FKL `fb9ac3a` (before #414 and #416); `e5-preprocessing-a-c8fcbe8/`: the same with FKL `c8fcbe8`, the final pin of LTS-C++17 (the README of that folder says which rows to cite); `e5-preprocessing-c/`: the resolution sweep (c).
  The other folders of this directory have their own README.

## Machine and software (read on 2026-11-01)

The results of this folder were committed from 2026-10-30 to 2026-11-01. The machine has been in one boot since 2026-10-26 18:39 CET, and its software state read on 2026-11-01 is described in `../rx9060/README.md` (section "Machine and software"): Ubuntu 26.04.1 LTS,
kernel 7.0.0-38-generic, `g++-13` 13.4.0 (Ubuntu 13.4.0-10ubuntu1), `clang++-23` 23.1.5 (Ubuntu build `++20261023092600+395ba2c4d274`), governor `powersave` (amd-pstate-epp, EPP balance_performance, boost on). The packages of the kernel, of `g++-13`, of `clang++-23` (upgraded to this build at 18:37 on 2026-10-26), of the
amdgpu driver and of ROCm have not changed since 2026-10-26 at 18:37; on 2026-10-29 at 20:52 `clang-21` (21.1.8) was installed next to `clang-23` and `lldb-23` was removed, and the default `g++` of the machine is 15.2.0 (the results use `g++-13` and `clang++-23`; the line `host compiler` of each CSV has the version).
The CSVs of this folder have no `# os:` or `# kernel:` line in their headers.
