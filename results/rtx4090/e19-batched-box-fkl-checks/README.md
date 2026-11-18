# E19 batched box, phase 2: the checks window e19-fkl-checks (RTX 4090, 2026-11-18 11:07:24-11:07:25 +01:00)

`e19_batched_box_cu` at f487b3f, which adds the row `fkl_loop` (FKL at the pin 24478f0, one `fk::executeLinearFilter` per crop). It is the build of 11:03. `build.log` has 25 warnings, all of them CUDA 13 deprecations inside FKL's headers, and the build info is f487b3f without -dirty. sha256 `2b59cd1c…` (`binary-sha256.txt`). The SASS census of its kernels is in `kernel-census.tsv`.

The window had two steps (`window-script.sh`, which stops at a failed step):
1. **checks/**: one checks run, `E19_NO_TIMING=1`, with `E19_DUMP`. It exited 1: fkl_loop's check failed (below).
2. **nsys**: did not run. The script stopped after step 1, so this record has no kernel count and no host enqueue for fkl_loop.

Nothing was timed, and no ncu ran. The dump (30.0 MiB) is left out of this record. It is on the measuring machine, in `scratch/e19-batched-box/dump-p2/`.

**In short:**
- **fkl_loop failed against its reference in double.** 1,956,214 of 1,966,080 values are beyond 1e-5, and the largest difference is 0.0076106.
  - The reference used FKL's grey as written for a float output: 0.299f R + 0.587f G + 0.114f B, not rounded.
- **FKL rounds that value at 24478f0.** In `fk::RGB2Gray::exec` (`color_conversion.h:55-69`) the branch `std::is_signed_v<OutputType>` comes before `std::is_floating_point_v<OutputType>`, and `std::is_signed_v<float>` is true. With a float output the grey is therefore `(float)__float2int_rn(luminance)`, rounded to the nearest integer with ties to even.
  - The kernel's SASS has F2I.NTZ after every grey FFMA chain.
- **Every other check passed,** with the same values as the re-check window.

## The checks (checks/e19_batched_box_checks.csv)
- **opencv_cuda and npp:** all 22 rows are identical, value for value, to those of the re-check window (`../e19-batched-box-recheck/`).
  - Their grey, box and tensor outputs are byte-identical to phase 1's dump (`dump-comparison.txt`).
- **fkl_loop:**

| compared with | values beyond 1e-5 | largest difference | not finite | ok |
|---|--:|--:|--:|---|
| fklReference (grey unrounded, box and conversion in double) | 1,956,214 | 0.0076106 | 0 | **no** |
| opencv_cuda's tensor (reported) | | 0.0193662 | 0 | |
| npp's tensor (reported) | | 0.0154930 | 0 | |

## The grey of fkl_loop, from the dump (`fkl-grey-check.txt`, `scripts-used/fkl_grey_check.py`; CPU only)
Each reference keeps the program's box and conversion in double and changes only the grey value:

| grey value | largest difference | beyond 1e-5 |
|---|--:|--:|
| exact float-weight sum (the program's reference) | 0.0076106 | 1,956,214 |
| FMA chain G * 0.587f, fma R * 0.299f, fma B * 0.114f, in float | 0.0076106 | 1,956,213 |
| that FMA chain rounded to the nearest integer, ties to even | 6.37e-7 | 0 |
| the exact sum rounded to the nearest integer, ties to even | 0.0038734 | 8,281 |

- **FKL's grey** is therefore the compiled float chain rounded to an integer.
- **The rounded exact sum** differs from the rounded chain at 926 grey values, where the value lies near a half.
- **Where the grey is rounded this way,** the kernel matches the reference within 6.37e-7.

## The SM clock (`gpu_clocks.csv`; `clock-readings.txt`)
- **11:07:24.399:** 210 MHz, P8.
- **11:07:24.605:** 210 MHz, P0.
- **11:07:25.106-11:07:25.206:** 2,550 MHz, P2 (2 readings).
- **The end of the run** (`checks/gpu-at-end.csv`): 2,550 MHz.
