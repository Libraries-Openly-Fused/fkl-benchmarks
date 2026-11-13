# RX 9060 (gfx1200), ROCm 10.1: how the GPU results are measured

Machine: Ryzen 7 5800X3D, 60 GiB, AMD Radeon RX 9060 (gfx1200, 28 CUs, wave32, 8144 MiB), Ubuntu 26.04.1 LTS, Linux 7.0.0-38-generic, ROCm 10.1.0 (amdclang++ 24.0.0), amdgpu 7.1.9;
CPU governor `powersave` (amd-pstate-epp, EPP balance_performance, boost on); the machine is also the self-hosted ROCm CI runner, so every run waits for the
runner to be idle (no `Runner.Worker`, load average below 0.8).

Pins: each folder names the FKL commit it was measured at. The pin of this repository was c8fcbe8 until 2026-11-07, then 64ad9195 (c8fcbe8 plus #422, #426, #423 and #427) until 2026-11-09, then 1da77899 (64ad9195 plus #431, the image-processing DPPs; the merge of FKL #431 into LTS-C++17) until 2026-11-13; since then it is 24478f0, 1da77899 plus #432, the HIP versions of those DPPs (the merge of #432 into LTS-C++17, whose tree is that of the PR head 2c42a80; `../rtx4090/final-checks/README.md`). #431 ports main's image-processing DPPs for CUDA and the CPU. It also changes three headers that the HIP programs include, among them `arithmetic.h`, which gives `fk::Sub`, `fk::Mul` and `fk::Div` a 4th template parameter with a default. At each move the device code (gfx1200) of the HIP programs of that time is the same. Built from 0fb1f9f97, the six HIP programs of then are the same at c8fcbe8 and at 64ad9195, 625 of 625 functions (`pin-move-devcode-c8fcbe8-64ad9195/`); only the host code differs (the chain checks of #423, the crop checks of #427). Built from 8f4a95b, the device code of all 7 HIP programs is the same at 64ad9195 and at 1da77899, 771 of 771 functions. They are paired by a name without the new default argument, and 264 of them have a new name (`pin-move-devcode-64ad9195-1da77899/`). Built from 8c1fa98, the device code of all 7 HIP programs is the same at 1da77899 and at 2c42a80 (the tree of 24478f0), 801 of 801 functions, none renamed (`pin-move-devcode-1da77899-2c42a80/`). The CUDA programs were compared on the RTX 4090 (`../rtx4090/fkl-pr423-check/`, `../rtx4090/fkl-pr427-check/`, for 1da77899 by the NVIDIA session, `../rtx4090/fkl-pr431-check/`, and for 24478f0, `../rtx4090/fkl-pr432-check/`); the CPU builds were not compared, as the CPU results are kept in the repository only.

## Machine and software (read on 2026-11-01)

The results of this folder were committed from 2026-10-30 to 2026-11-01. The machine has been in one boot since 2026-10-26 18:39 CET (the boot before ran the same kernel), and the software state read on 2026-11-01 is:

- OS: Ubuntu 26.04.1 LTS (Resolute Raccoon), x86_64; kernel 7.0.0-38-generic (`#38-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 09:10:14 UTC 2026`, package 7.0.0-38.38 installed on 2026-10-12).
- ROCm 10.1.0 (`amdrocm` 10.1.0-3, installed on 2026-10-12), HIP driver and runtime 7.16.26385, amdgpu 7.1.9.31600000 (DKMS, installed on 2026-10-12, built for 7.0.0-38), amd-smi 27.1.0.
  No update of the kernel, of the amdgpu driver or of ROCm happened between 2026-10-12 and 2026-11-01.
- Compilers: g++ 13.4.0 (Ubuntu 13.4.0-10ubuntu1), clang++ 23.1.5 (Ubuntu build `++20261023092600+395ba2c4d274`, upgraded to this build on 2026-10-26 at 18:37; clang-21 21.1.8 was installed next to it on 2026-10-29 at 20:52), amdclang++ 24.0.0git (ROCm/llvm-project `d6f6cb691863`); the builds use the binaries `g++-13` and `clang++-23` (the default `g++` of the machine is 15.2.0).
- Board and CPU: ASUS PRIME B550M-K (BIOS 4304 of 2026-08-19), Ryzen 7 5800X3D, 60 GiB, governor `powersave`.
- GPU: AMD Radeon RX 9060 (Navi 44, gfx1200, PCI 1002:7590 revision C7, 28 CUs, 8144 MiB GDDR6 on 128 bits). The PCI-ID database (and so `lspci` and the board field of `amd-smi`) names the device ID 7590 "RX 9060 XT", without looking at the revision;
  `rocminfo`, `amd-smi` and the HIP runtime say "AMD Radeon RX 9060", which is what libdrm's `amdgpu.ids` has for the revision C7 (the XT is the revision C0).
  PCIe: 4.0 x16 (16 GT/s) on a root port of the CPU (0000:00:03.1); the card itself supports PCIe 5.0, the platform 4.0. The link was the same idle and under load in 114 samples read on 2026-11-01 (noted by hand, not kept in a file).

The CSVs of this folder have no `# os:` or `# kernel:` line in their headers; this section is the record of the software state. The report of `scripts/gpu_clock_trace.py` prints the OS and the kernel of the traces that it took with its current version (`meta.txt`).

## The graphics clock of the GPU decides the result of a short run

A GPU that was at rest does not run at its working clock when work arrives. On this RX 9060 (sysfs `hwmon/freq1_input`, sampled at about 180 Hz,
`scripts/gpu_clock_trace.py`) the clock goes from 31 MHz at rest to about **2.0 GHz, stays there for the first 0.7 s of continuous work, reaches 3.0 GHz
after about 1 s**, overshoots to 3.3 to 3.5 GHz for a moment, and settles at 3.0 to 3.2 GHz (the value depends on the kernel that is running). The warm-up of 0.5 s
that the first results of this folder used ended on the ramp. What it did to the numbers, measured:

- The fused chain of pipeline (a) with the default (ASYMMETRIC) resize, batch 300, is **22 percent slower at 2.0 GHz** than at 3.0 GHz: total 0.1550 ms (device
  0.1328 ms) after 3 s without GPU work and a warm-up of 0.5 s; 0.1265 ms (device 0.1050 ms) with a warm-up of 2 s or 5 s. The same kernel with the HALF_PIXEL
  sampling had been timed second, after the ramp, and looked unchanged. In the first RPP run after the 0.5 s warm-up the default chain read 0.1146 / 0.1577 ms
  at batch 200 / 300 with FKL c8fcbe8, against 0.0924 / 0.1241 ms in the first run with FKL 38c4f32 (same benchmark source and identical device code, `clock-ramp/`): the ramp had been faster in that run by chance.
- In the other direction, a warm-up that ends right after the ramp can be measured in the overshoot: the NV12 to RGB kernel of the E4b pipeline (removed later, see "Folders") at one crop read 0.2602 ms
  with a warm-up of 0.5 s (FKL fb9ac3a) and 0.2808 ms at the steady clock (FKL c8fcbe8, median of three repetitions; 7 percent too fast), and it varied with the
  number of crops (0.2602 to 0.2827 ms at fb9ac3a) although the kernel does not depend on it; with the steady warm-up it is 0.2808 to 0.2840 ms for 1 to 64 crops (c8fcbe8).
  The data are in the git history: `git show 341ecd1:results/rx9060/e4b-nv12/` and `git show 341ecd1:results/rx9060/e4b-nv12-steady-clock/`.
- The latency hiding series of `fkl-self` (the buffers are allocated and filled between two points, which leaves the GPU idle) was up to 52 percent slower
  without keeping the GPU busy before the timed runs of each point (301 operations: 6.10 ms without and 4.01 ms with the spin of 2 s, both runs with FKL c8fcbe8 and the same device code; the run with the spin
  was made with `FKLB_GPU_SPIN_SECONDS=2`, implemented in fkl-benchmarks #7; the first series, with FKL 38c4f32 and no spin, gave 4.14 ms).
- Nothing in FKL is involved: the benchmark binaries have identical device code at the pins 38c4f32, #411, #414, #416 and #417 of LTS-C++17 (c8fcbe8), and the
  times of the fused chain of pipeline (a) differ by 0.2 to 2.5 percent between the five pins over the 30 batches (0.16, 1.74 and 0.74 percent at the batches 100, 200 and 300), with no order by pin (`clock-ramp/`, section 1, `five-pins-spread.txt`;
  an earlier version of this README said "the same at the five pins within 1 percent", which holds only for part of the cells: the per-op batch differs by up to 8.3 percent).

## The rule

For the pipeline of `rpp/` (`preproc::measure` in `fkl-self/pipelines/preproc_common.h` for the results of this folder; the same rule is `fklb::measureWall` in `common/hip_wallclock.h` since the RPP rewrite, 282f734): a warm-up of at least 2 s and until the mean time of two consecutive
windows of 0.25 s differs by at most 2 percent (cap 10 s), then 30 timed runs; the variants of a point are timed in a rotated order (the first one is variant
(index of the point + E5_ROTATE_OFFSET) mod the number of variants) and the whole run is repeated three times with the offsets 0, 1 and 2; the result is the median of
the three medians and the CSV columns `warmup_s`, `warmup_steady`, `sclk_start_mhz`, `sclk_timed_start_mhz` and `sclk_end_mhz` give the warm-up that was run
and the graphics clock at the start of the warm-up, at the start of the timed runs and after them. For the benchmarks of `fkl-self` whose GPU work is not
continuous from one point to the next (latency hiding), the environment variable `FKLB_GPU_SPIN_SECONDS` (2 for the results of `fkl-self/`) makes the benchmark run the work of the point for that many seconds
before its timed runs. The variable is implemented in fkl-benchmarks #7 (`fkl-self/fkBenchmarksCommon.h` there), which also adds the `fkl-self/` results; it is not part of this pull request.

## Folders

- `rpp-preprocessing/`: FKL against RPP 3.2.0, pipeline (a), at c8fcbe8 (the pin until 2026-11-07) with the rule above (replaces the first run, which used a warm-up of 0.5 s
  at the pin 38c4f32 and is in the history of the branch).
- Removed: the NV12 video pipeline (pipeline (b), experiment E4b: the folders `e4b-nv12/` and `e4b-nv12-steady-clock/` and the pipeline (b) files of `rpp-preprocessing/` and `clock-ramp/`) is not in the paper.
  Its code and results are in the git history up to `341ecd1`, for example `git show 341ecd1:results/rx9060/e4b-nv12-steady-clock/README.md`.
  The `# notes:` header line of the pipeline (a) CSVs written before the removal (in `rpp-preprocessing/` and `clock-ramp/`) still names `rppt_yuv_to_rgb`, the RPP conversion that only pipeline (b) called; the benchmark does not write it any more.
- Removed: the exhaustive `fk::SaturateCast<float, T>` sweep over all 2^32 float bit patterns on the RX 9060 (experiment E6: the folder `e6-saturate-exhaustive/` and its program `fkl-self/numerics/e6_saturate_exhaustive_amd.h`) is not used by the paper.
  Its code and results are in the git history up to `5911330`, for example `git show 5911330:results/rx9060/e6-saturate-exhaustive/README.md`.
- `fkl-self/` (added by fkl-benchmarks #7): horizontal fusion, instructions per operation and latency hiding (FKL only).
- `clock-ramp/`: the evidence of this section.
