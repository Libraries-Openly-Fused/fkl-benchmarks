# E18 phase 1 (e18-stencil-phase1), 2026-11-13: the stencil after a resize on the RX 9060, checks only, NOT timed

This folder is phase 1 of E18 on AMD. Phase 1 is the code, the build and the checks. **Nothing here is timed.** The timing (phase 2) runs in a GPU window scheduled with the coordinator after review.
- **The program:** `rpp/e18_stencil.h`, target `e18_stencil_hip`, at 4c167f4 (branch `e18-rx9060`). It is the FKL side of `opencv-cuda/stencil/e18_stencil.cu` (the RTX 4090 program, bd5900d), with the same chain, variants, checks and timing code.
- **FKL:** 2c42a80, the head of FKL #432 (the HIP port of the seven DPPs; LinearFilterDPP gets its `ParArch::GPU_AMD` specialization there). It is pinned on this branch only. The branch is rebased onto the pin move after Oscar merges #432.
- **Decision:** E18 RX 9060 phase 1, Oscar's decision in the coordinator session, 2026-11-13: start now, pinned to #432's head on a branch.

**The chain** is the RTX 4090's:
- one 3840 x 2160 8UC3 frame, E4's seeded source (LCG, seed 12345);
- crop 2560 x 1440 at (640, 360), through `Ptr2D::crop2D`;
- conversion to float with no rounding, bilinear resize to 1920 x 1080;
- sharpen 0 -1 0 / -1 5 -1 / 0 -1 0, border REPLICATE;
- saturating cast to uchar3, round to nearest with ties to even.

**The variants:** `fkl_stencil_asym`, `fkl_stencil_halfpixel` and their `_once` forms.
- Each is one `fk::executeLinearFilter` with `LinearFilterDPPDetails<float3>` (the default 16 x 8 tile and 7 x 7 largest kernel), so one kernel per call.
- The per-call forms build the crop view, every IOp and the details in each call. The `_once` forms build them once.

**No RPP baseline.** RPP 3.2.0 has no 2D filter with user weights. Its filters are box, median, Gaussian, Sobel and emboss, with border REPLICATE only. It also has no F32 to U8 conversion in the HIP back-end. So the chain cannot be written with one documented RPP call per step, and the baseline was stopped as the brief says. The inventory went to the coordinator on 2026-11-13. Without a baseline there is no "FKL halfpixel vs RPP" agreement row.

## Differences from the FKL code of the CUDA program
The FKL calls are the CUDA program's: the same IOps, template arguments, details and order. The only differences are those of the back-end:
1. **The stream:** a `hipStream_t` in `fk::Stream_<fk::ParArch::GPU_AMD>` (CUDA: `cudaStream_t`, `GPU_NVIDIA`).
2. **The resize-alone check** uses `fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>` (CUDA: `GPU_NVIDIA`).
3. **Vector types:** `fk::float3` and `fk::uchar3`, through `using` declarations in `launch()`. Under HIP, `::float3` and `::uchar3` are HIP's own types; FKL LTS uses its own vector types on every back-end. The weights are built as `float3{v, v, v}` (CUDA: `make_float3(v, v, v)`).
4. **Copies:** `hipMemcpy2D` and `FKLB_HIP_CHECK` (CUDA: `cudaMemcpy2D`, `FKLB_CUDA_CHECK`).
5. **Timing:** `common/hip_timing.h` (new). It is `common/cuda_timing.h` with HIP events, with the same `timeIt` and `timePreparedLaunch`, the same intervals and 100 iterations after one warm-up.
6. **The spin:** `gpuSpin` waits on `wall_clock64()` (s_memrealtime, the constant-rate clock), converted from µs with `hipDeviceAttributeWallClockRate`. CUDA waits on `%globaltimer`. Both are one thread and touch no memory. The default is still 5,000 µs (`E18_SPIN_US`).
7. **Not ported:** NVTX ranges and the `E18_PROFILE` mode. They are CUDA profiler hooks and have no counterpart in this program.
8. **The baselines and their rows are gone:** OpenCV-CUDA and NPP and their pairs, as is the check of a baseline's resized image against FKL's. The order of the variants is the CUDA order without the baselines: asym, halfpixel, asym_once, halfpixel_once.
9. **The output files** end in `_GPU_AMD` (`e18_stencil_agreement_GPU_AMD.csv`, `e18_stencil_checks_GPU_AMD.csv`, and in phase 2 `e18_stencil_GPU_AMD.csv`). The helpers are in namespace `e18amd`, because the program is a launcher header.

## The build (`build/`)
- amdclang++ 24.0.0 (ROCm 10.1.0), gfx1200, Release. The configuration is the HIP part of the benchmark builds (`-DFKLB_BUILD_RPP=ON`, CUDA off).
- Built from the clean worktree at 4c167f4 with the submodules checked out: FKL 2c42a80, FastRPP efbf009 (not used by this target).
- 0 errors, 0 warnings. sha256 in `build/binary-sha256.txt`.
- No CI job (Runner.Worker) was running on this machine during the run.

## The checks (`console.txt`, `e18_stencil_checks_GPU_AMD.csv`, `e18_stencil_agreement_GPU_AMD.csv`)
One run with `E18_NO_TIMING=1`: 1 s, exit 0, empty stderr, "E18 checks passed".

**Checks of FKL's path:**

| check | ASYMMETRIC | HALF_PIXEL |
|---|--:|--:|
| LinearFilterDPP with the identity kernel vs the resize alone, bitwise | **identical** | **identical** |
| resize alone (float3) vs the double-precision resize, max abs difference | 0.0278 | 0.0428 |

**Agreement** of the 8-bit outputs: 6,220,800 values each (1920 x 1080 x 3).

| variant | against | values differing | pixels differing | max (levels) | max distance of the reference to a half level |
|---|---|--:|--:|--:|--:|
| fkl_stencil_asym | double reference, ASYMMETRIC | 7,728 | 7,688 | 1 | 0.0556 |
| fkl_stencil_halfpixel | double reference, HALF_PIXEL | 139,111 | 135,473 | 1 | 0.1667 |
| fkl_stencil_asym_once | double reference, ASYMMETRIC | 7,728 | 7,688 | 1 | 0.0556 |
| fkl_stencil_halfpixel_once | double reference, HALF_PIXEL | 139,111 | 135,473 | 1 | 0.1667 |
| fkl_stencil_asym_once | fkl_stencil_asym | **0** | 0 | 0 | - |
| fkl_stencil_halfpixel_once | fkl_stencil_halfpixel | **0** | 0 | 0 | - |

- Every difference from the reference is one level, where the unrounded reference is near a rounding boundary: at most 0.167 from a half level.
- The `_once` forms are bitwise equal to the per-call forms.

**Against the RTX 4090** (`../../rtx4090/e18-stencil-phase1/`, FKL 1da77899, nvcc 13.4 with -fmad):

| | RTX 4090 | RX 9060 |
|---|--:|--:|
| asym vs reference: values differing | 24,274 | 7,728 |
| halfpixel vs reference: values differing | 139,104 | 139,111 |
| asym, max distance to a half level | 0.1667 | 0.0556 |
| resize alone vs double, ASYMMETRIC | 0.0415 | 0.0278 |
| resize alone vs double, HALF_PIXEL | 0.0428 | 0.0428 |
| identity kernel bitwise equal to the resize alone | yes, both | yes, both |

- The HALF_PIXEL rows are close: 7 values apart, with the same largest resize error.
- The ASYMMETRIC rows differ: the RX 9060 has a third as many differing values and a smaller resize error. A likely reason is that the two compilers evaluate the float arithmetic of the resize differently (for example, which multiply-adds they contract into FMAs), but this was not checked.
- **The cause was not determined.** No difference is larger than one level on either GPU.

## The fused kernel on gfx1200 (`isa/`)
- `isa/disassembly.s`: `llvm-objdump -d --symbolize-operands` of the gfx1200 code object of the binary.
- `isa/notes.txt`: its metadata (`llvm-readelf --notes`).
- `isa/resource-usage-remarks.txt`: `-Rpass-analysis=kernel-resource-usage` from a recompile of the same source with the same command.
- `isa/kernel-census.tsv`: these three combined by `isa/isa_census.py`.

| | fused ASYMMETRIC | fused HALF_PIXEL |
|---|--:|--:|
| instructions | 520 | 528 |
| VGPRs | 23 | 23 |
| SGPRs | 37 | 37 |
| scratch (private segment), spills | 0 B, 0 / 0 | 0 B, 0 / 0 |
| dynamic stack | no | no |
| LDS | 4,284 B per block | 4,284 B |
| kernarg segment | 384 B, of which 122 B are the five explicit arguments | the same |
| wave size | 32 | 32 |
| occupancy (compiler, waves per SIMD) | 16 (the maximum) | 16 |
| block, grid | 16 x 8 (128 threads = 4 waves), 120 x 135 | the same |

- **Explicit arguments:** 122 B, the same as on the RTX 4090 (`sizeof`: details 24, the two Reads 72 (image 48, weights 24), multiply 1, accumulate 1, the epilogue 24). The remaining 262 B are HIP's hidden arguments.
- **No scratch:** the kernel reads its parameters with `s_load` from the kernarg segment, and everything is inlined.
- **LDS:** 4,284 B, as the RTX 4090's shared memory. It is the halo of the default tile (22 x 14 float3) plus the 7 x 7 coefficients.
- **Occupancy:** the value is the compiler's static one. It is limited neither by the registers nor by the LDS. The achieved occupancy needs a profiler run and is left for phase 2, if wanted.
- The other kernels of the binary (`gpuSpin` and the identity and resize-alone kernels of the checks) are in the same table. The identity-check kernels have 341 and 348 instructions and the same 23 VGPRs, 37 SGPRs and LDS.
- The RTX 4090 kernel has 840 and 848 SASS instructions and 40 registers. The counts of the two ISAs are not comparable one to one.

## Phase 2 (not run)
- `whole_call` is E4's interval and `kernel_only` E1's, behind the 5 ms spin, with the host preparation timed separately. The order is fixed: asym, halfpixel, asym_once, halfpixel_once.
- It runs in a window scheduled with the coordinator, never while a CI job runs on this runner.
