# E18 stencil, phase 1: correctness, kernels per call and kernel metrics (RTX 4090, CUDA 13.4.3; nothing is timed)

This is the record of the first phase of E18 (`opencv-cuda/stencil/e18_stencil.cu`). It covers:
- the agreement of the outputs;
- the checks of FKL's path;
- the number of kernels of one call of every implementation;
- the metrics of the fused kernel.

Nothing is timed in this phase. The timing comes later, with Figure 15's protocol.

**The chain.**
- **Input:** one 3840 x 2160 8UC3 frame, E4's seeded source.
- **Steps:**
  1. Crop 2560 x 1440 at (640, 360).
  2. Convert to float before the resize.
  3. Resize bilinearly to 1920 x 1080, a uniform scale of 0.75, each implementation with its own sampling convention.
  4. Sharpen with the 3 x 3 kernel 0 -1 0 / -1 5 -1 / 0 -1 0, with the border REPLICATE.
  5. Convert to 8UC3 with saturation. Values at or above 255 give 255 and values at or below 0 give 0. The others round to the nearest integer, ties to even: FKL's `SaturateCast`, OpenCV's `saturate_cast` in `convertTo`, NPP's `NPP_RND_NEAR`.

**The implementations.** They are described in the header comment of the program. In short:

| version | convention | what one call is |
|---|---|---|
| `fkl_stencil_asym` | ASYMMETRIC | one `fk::executeLinearFilter`, with the crop view, every IOp and the details built in the call |
| `fkl_stencil_halfpixel` | HALF_PIXEL | the same, with `fk::ResizeSampling::HALF_PIXEL` |
| `fkl_stencil_asym_once`, `fkl_stencil_halfpixel_once` | as above | everything built once; the call is `fk::executeLinearFilter` alone |
| `opencv_cuda` | ASYMMETRIC (`cv::cuda::resize`) | GpuMat ROI, `convertTo`, `cv::cuda::resize`, `cvtColor` BGR to BGRA, the linear filter (`createLinearFilter`, CV_32FC4, `BORDER_REPLICATE`, created once), `convertTo`, `cvtColor` BGRA to BGR |
| `npp` | HALF_PIXEL (`nppiResize`) | `nppiConvert_8u32f_C3R_Ctx`, `nppiResize_32f_C3R_Ctx`, `nppiFilterBorder_32f_C3R_Ctx` (`NPP_BORDER_REPLICATE`), `nppiConvert_32f8u_C3R_Ctx` (`NPP_RND_NEAR`) |

**Setup.**
- Machine: RTX 4090, nvidia-smi index 1, with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. Driver 610.43.02.
- Code: fkl-benchmarks 39e64d4, the commit of the program on the branch `stencil-experiment`, based on main 5caa9d1. FKL 1da77899, cvGPUSpeedup 6d26c65, FastNPP a1b5e245.
- Build: preset `local-rtx4090`, nvcc 13.4.92 for sm_89, GNU 15.2.0, Release, OpenCV 4.14.
  - Only the target `e18_stencil_cu` was built: `cmake --build build/local-rtx4090 --target e18_stencil_cu`.
  - Its only warnings are FKL's 25 `long4`-type deprecation lines, which every CUDA program of this repository has.
- The CSV headers show 39e64d4 without -dirty.

## Agreement (`e18_stencil_agreement.csv`, `console.txt`)
The program was run once, with no environment variable set. It exited 0, its stderr was empty, and its last line is "E18 checks passed".

Each output is 1920 x 1080 x 3 = 6,220,800 values.

| version | against | values differing | pixels differing | largest difference | largest distance of the reference to a half level, at the differing values |
|---|---|--:|--:|--:|--:|
| fkl_stencil_asym | the double reference, ASYMMETRIC | 24,274 | 23,942 | 1 | 0.167 |
| opencv_cuda | the double reference, ASYMMETRIC | 24,271 | 23,939 | 1 | 0.167 |
| fkl_stencil_halfpixel | the double reference, HALF_PIXEL | 139,104 | 135,465 | 1 | 0.167 |
| npp | the double reference, HALF_PIXEL | 139,121 | 135,484 | 1 | 0.167 |
| fkl_stencil_asym | opencv_cuda | 3 | 3 | 1 | 0.056 |
| fkl_stencil_halfpixel | npp | 85 | 85 | 1 | 0.111 |
| fkl_stencil_asym_once | fkl_stencil_asym | 0 | 0 | 0 | |
| fkl_stencil_halfpixel_once | fkl_stencil_halfpixel | 0 | 0 | 0 | |

The `_once` variants against the baselines give the same rows as their per-call variants.

**Why the double references differ.** All four float implementations differ from the double references by at most one level, and only at values whose exact result is within 1/6 of a level of a rounding boundary.
- The cause is the resize: 2560 / 1920 is not exact in float.
- The three libraries' resized float images differ from the double resize by up to 0.042 (ASYMMETRIC) and 0.043 (HALF_PIXEL) levels, and from each other by less than 5e-5 levels (`e18_stencil_checks.csv`).
- The kernel's weights add up to 9 in absolute value, so the resize error can move a sharpened value by up to about 0.4 levels.
- FKL against its baseline differs in 3 values (OpenCV) and in 85 values (NPP), each by one level. In every one of them the exact value is within 0.056 (OpenCV) or 0.111 (NPP) of a rounding boundary.

## Checks of FKL's path (`e18_stencil_checks.csv`)
These check the use of a Read IOp with a resize as the image of LinearFilterDPP, from uchar3 to float3, which FKL's tests do not cover.
- **The resize alone:** the same image Read IOp written as float3 by `fk::executeOperations`. Against the double resize the largest difference is 0.0415 levels (ASYMMETRIC) and 0.0428 levels (HALF_PIXEL).
- **LinearFilterDPP with the identity kernel** (1 at the anchor, a float3 write) against the resize alone: **bitwise identical** for both conventions. The tile staged in shared memory therefore holds the resized pixels at the right places, the replicated border included.
- **The baselines' own resized images**, their intermediates read after the call:
  - OpenCV's CV_32FC3 against FKL's ASYMMETRIC resize: largest difference 3.05e-5;
  - NPP's 32f C3 against FKL's HALF_PIXEL resize: largest difference 4.58e-5.

## Kernels of one call (`profile/`)
- **How they were counted:** with `E18_PROFILE=1`, every implementation is called once inside an NVTX range named after it, after its setup has finished. Nothing is compared or timed.
- **The command:**
  ```
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none ../../bin/e18_stencil_cu
  ```
  The nsys was the one of CUDA 13.4.3 (2026.3.2), and its temporary files went to `TMPDIR` on disk. Then `nsys stats --report nvtx_kern_sum`.
- **The counts:**
  - **fkl_stencil_asym, fkl_stencil_halfpixel and both `_once` variants: 1 kernel**, `fk::linearFilterDPPKernel`.
  - **opencv_cuda: 6 kernels**: `transformSmart` (8U to 32F), `resize_linear<float3>`, `transformSimple` (BGR to BGRA), `filter2D` (texture reader), `transformSmart` (32F to 8U), `transformSimple` (BGRA to BGR).
  - **npp: 4 kernels**: `ForEachPixelNaiveInLargeImage` (8u to 32f), `resizeNaive`, `ForEachPixelNaiveInLargeImage` (`Filter32fReplicateBorderSharedFunctor`), `ForEachPixelNaiveInLargeImage` (32f to 8u).

## The fused kernel (`kernel-census.tsv`, from `kernel_census.py`)
The data come from cuobjdump -sass and -res-usage of the binary. sm_89; the instructions are counted as in `scripts/sass_tools.py`.

| kernel | SASS instructions | registers | stack | local | LDL / STL | shared | CONSTANT[0] | arguments |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `linearFilterDPPKernel`, the stencil, ASYMMETRIC | 840 | 40 | 0 | 0 | 0 / 0 | 4,284 B | 480 B | 122 B |
| `linearFilterDPPKernel`, the stencil, HALF_PIXEL | 848 | 40 | 0 | 0 | 0 / 0 | 4,284 B | 480 B | 122 B |

- **Arguments:** 122 B is the sum of `sizeof` of the five: the details 24, the two Reads 72 (image 48, weights 24), multiply 1, accumulate 1, the epilogue write 24 (`console.txt`).
- **Shared memory:** the halo of the default tile, 22 x 14 float3, plus the 7 x 7 coefficients, for the default 16 x 8 tile and 7 x 7 largest kernel of `LinearFilterDPPDetails<float3>`.
- **How the parameters are read:** 76 instructions read them as constant-bank operands (c[0x0][...]), with no indexed load and no local memory.
- **No `__grid_constant__`:** this kernel does not declare its parameters `__grid_constant__` (FKL 1da77899, `linear_filter.h:285`). Everything is inlined, so it makes no difference here.
- **Other kernels:** the census also lists the two identity-check filter kernels (752 and 760 instructions) and the two resize-alone TransformDPP kernels of the checks. They are not part of any timed call.

## Files
- `console.txt`: the output of the correctness run.
- `e18_stencil_agreement.csv` and `e18_stencil_checks.csv`: the program's output.
- `profile/`:
  - `kernels-per-call-nvtx_kern_sum.csv`;
  - `profile-console.txt`, the nsys output, with its progress lines;
  - `e18-profile.nsys-rep`.
- `kernel-census.tsv`: the census of every kernel of the binary.
- `kernel_census.py`: the script that wrote it.
- **Left out:**
  - the empty stderr of the run;
  - the SQLite export of the profile (651 KB, derived from the `.nsys-rep`);
  - the SASS dump (`e18_stencil_cu.sass`);
  - the build log.

  They stay in `scratch/` on this machine. Every file here is the complete output of the program named; nothing was filtered.
