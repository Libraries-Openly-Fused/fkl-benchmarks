# E19 crop sweep, phase 1: the build and the checks window's script (RTX 4090)

This folder holds the build record of `e19_crop_sweep_cu` and the script of the checks window. The window's own record will be `results/rtx4090/e19-crop-sweep-checks/`. The script sits outside that folder so that the window's new-folder guard keeps working. Nothing in this folder ran on the GPU.

The program is `opencv-cuda/stencil/e19_crop_sweep.cu`. It is E19 (`e19_batched_box.cu` at main f5f79f4, unchanged) with N crops per process, N = 1, 10, 20, ..., 150. The PR description of #56 holds the full diff against E19 and the list of what differs.

## The build (compile only, 2026-11-24 08:00)

- Commit 246b1c8, not -dirty. The binary carries 246b1c8 and the pins: fkl 24478f0, cvGPUSpeedup 6d26c65, FastNPP a1b5e24.
- `cmake --preset local-rtx4090`, then `cmake --build build/local-rtx4090 --target e19_crop_sweep_cu`. CUDA 13.4 (`toolchains/env-cuda13.4.sh`), with TMPDIR on disk. The build took 23 s.
- `build/local-rtx4090/bin/e19_crop_sweep_cu`: sha256 `2b800f1d5626a6a32523cbfe50b960668cc3f15386f03b7be442df1cdbb26c8c` (`binary-sha256.txt`).
- 25 warnings, all from FKL's headers: vector_utils.h 20, thread_fusion.h 5. This is the same set as E19's builds.
- The binary has one cubin, sm_89, with 18 kernels:
  - 16 instances of `fk::morphologyDPPKernel`, fkl_batch's kernel, one per N: `BatchRead<PROCESS_ALL, N, ...>`;
  - 1 `fk::linearFilterDPPKernel`, fkl_loop's kernel, the same at every N;
  - `gpuSpin`.

## The SASS against E19's binary (`sass-compare.txt`, `res-usage.txt`)

The comparison is with `e19_batched_box_cu` at 8ff14e5, the binary of results/rtx4090/e19-batched-box-fkl-batch.

- At N = 30, fkl_batch's kernel (544 instructions), fkl_loop's kernel (688) and gpuSpin (24) are identical to E19's. gpuSpin's anonymous-namespace prefix holds a hash of the file name and is left out of the name.
- At N = 10 to 150, fkl_batch's kernel has 544 instructions. They equal N = 30's except for the constant-bank offsets of the parameters that follow the batch. Each instance uses 40 registers, 1,232 B of shared memory and no stack.
- **Observation at N = 1. No cause is claimed, and nothing was changed.** fkl_batch's kernel compiles differently at N = 1:
  - 568 instructions, 35 registers and a 40-byte stack frame.
  - It copies its 40-byte BatchRead parameter to the stack (5 `STL.64`) and reads the crop's pointer back from there (`LDL`, `LDL.64`).
  - It loads the 15 pixel bytes with generic loads (`LD.E.U8`).
  - At N ≥ 10, the kernel has no stack. It reads the crop's pointer from the parameter bank with `LDC` at a register index and loads the pixels with `LDG.E.U8` (15).

## fkl_batch's kernel parameters (`kernel-param-sizes.txt`)

The sizes come from `cuobjdump -elf` (EIATTR_CBANK_PARAM_SIZE and EIATTR_KPARAM_INFO).

| N | 1 | 10 | 30 | 60 | 100 | 150 |
|---|---|---|---|---|---|---|
| bytes of parameters | 120 | 336 | 816 | 1,536 | 2,496 | 3,696 |

- The size is 96 + 24 N bytes at every N. The BatchRead parameter is 16 + 24 N; the details take 24 B, the write 48 B and the passes 8 B.
- Every N is below 4,096 B.
- fkl_loop's kernel takes 112 B, and gpuSpin 8 B.

## The crop table, checked on the CPU (`crop-table.txt`, `scripts-used/crop_table.py`)

Crop i has c = i mod 6, r = (i div 6) mod 5 and k = i div 30. It starts at x = 64 + 600c + SX[k], y = 64 + 400r + SY[k], with SX = {0, 320, 0, 320, 192} and SY = {0, 0, 200, 200, 100}.

- Crops 0-29 are E19's, all 150 lie inside the 3840 x 2160 frame, and the 150 positions are distinct. The program checks the same three things before any GPU call.
- The largest right edge is 3,640 and the largest bottom edge 2,120.
- Overlap:
  - The crops are disjoint up to N = 60. From N = 70 on, some overlap, and from N = 120 on every crop overlaps another.
  - At most 2 crops lie on one pixel up to N = 120, and at most 3 from N = 130 on (322,560 pixels at N = 150).
- Alignment:
  - Each SX is a multiple of 64 pixels, and the frame is uchar3, so each shift is a multiple of 192 bytes (960 B for SX = 320, 576 B for SX = 192).
  - Each crop therefore starts at the same address as the E19 crop it repeats modulo 32 and 64 bytes, but not modulo 128 bytes. For k = 1, 3 and 4 the start moves by 64 B modulo 128. This holds for any frame pitch that is a multiple of 128 B.

## The checks window's script (`window-checks.sh`)

The guards run before anything starts:
- no blocked program;
- the binary's sha256 is the one above, and it carries 246b1c8 and no -dirty commit;
- HEAD is 246b1c8 or a descendant that differs from it only under results/, and the worktree has no local change outside results/;
- the script is the version committed here and runs from here;
- the output folder is `results/rtx4090/e19-crop-sweep-checks` and is new;
- no measurement, build or compile job runs, and no other program uses the RTX 4090.

The steps stop at the first failed run:
1. E19_NO_TIMING=1 at each N, in ascending order.
2. E19_PROFILE=1 under nsys at each N, in ascending order.
3. One complete run at N = 150 under nsys, after an idle wait. It is used for the spin's coverage only, not for times.

The nvidia-smi sampler reads every 100 ms, and the runs are pinned to cores 0-3,16-19.

## Files

| File | Contents |
|---|---|
| `binary-sha256.txt` | the sha256 of the binary |
| `configure.log`, `build.log` | the configure output and the build's standard output |
| `build-stderr.log` | the build's standard error: the 25 warnings, then the output of `/usr/bin/time -v` |
| `sass-compare.txt` | the SASS comparison |
| `res-usage.txt` | `cuobjdump -res-usage` per kernel |
| `kernel-param-sizes.txt` | the kernel parameter sizes |
| `crop-table.txt` | the crop-table check |
| `scripts-used/` | the scripts that wrote the three reports above |
| `window-checks.sh` | the checks window's script |

Left out: the full SASS (`cuobjdump -sass`) and ELF (`cuobjdump -elf`) dumps of the binary, about 2.1 MiB each, derived from the binary. They are in the scratch folder.
