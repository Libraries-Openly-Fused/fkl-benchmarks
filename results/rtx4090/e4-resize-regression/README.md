# Does the batched pre-processing kernel change between the library trees 38c4f32, d776713 and c8fcbe8? (RTX 4090, sm_89, CUDA 13.4.3)

Question (from the AMD session's result on gfx1200): the fused pre-processing chain with the default resize (ASYMMETRIC) is about 25 % slower at c8fcbe8 than at 38c4f32 for batches of 200 and 300 crops; the same chain
with `ResizeSampling::HALF_PIXEL` did not change. The 4090 answer: **no, the kernels are identical and the times are the same.**

## What differs between the trees
`git diff` of `include/`: 38c4f32 -> d776713 (#411, #414): `color_conversion.h` (a compile-time error instead of a zero matrix for combinations without a matrix) and `constexpr_cmath.h` (host-run-time `nearbyint`, under
`FK_HOST_RUNTIME_MATH`, which is not defined in the device passes of nvcc and HIP). d776713 -> c8fcbe8 (#416, #417): `constexpr_saturate.h` (host run-time saturate, same guard) and `parameter_pack_utils.h` (`get_arg`:
`pp_detail::Nth<I>::pick(Skip<S>..., T&&, Rest&&...)` instead of `TupleUtil::get<I>(forward_as_tuple(...))`, same `T&&` return). The only change that reaches device code is `get_arg`.

## Programs and builds
`e4_preprocessing_cu` (E4: `fkl_fused` = ASYMMETRIC and `fkl_fused_half_pixel`, 60 x 120 crops of a 3840 x 2160 8UC3 image to 64 x 128, batch 1 ... 300) and `e1_preproc_cu` (E1c: the same chain, batches 10, 50, 100, 200, 300,
FKL against hand-written CUDA), both built from fkl-benchmarks e9e599d (`handwritten-cuda/e1_preproc.cu` is identical to the one at 74f47c6 that produced the E1c numbers) with `external/fkl` replaced by each tree
(`scripts/regression_build.sh`): nvcc 13.4.92, g++ 15.2, sm_89, `-O3`, `-lineinfo`, thread fusion off.

## 1. Kernels (`kernels_e4_preprocessing.csv`, `kernels_e1_preproc.csv`, `summary_*.txt`; `scripts/kernel_resources.py`)
All kernels of the two programs that run the pipeline, matched by mangled name across the three binaries: E4 62 kernels (batches 1, 10, 20, ..., 300 times the two sampling policies), E1c 20 kernels (5 FKL, 15 hand-written).
**Normalised SASS (cuobjdump, registers not renamed): identical in all 82 kernels in the three trees; registers, stack, shared memory and constant bank 0 (the kernel parameter block): identical in all 82.**
FKL kernel, any batch: 34 registers, no stack, no shared memory, 272 SASS instructions (ASYMMETRIC) and 280 (HALF_PIXEL). The parameter block grows with the batch (constant bank 0: 496 B at batch 1, 928 B at 10,
11,680 B at 100, 16,480 B at 200, 21,280 B at 300) and is the same in the three trees at every batch.

## 2. Times (`timings/`, `timing_summary.csv`; `scripts/regression_timing_summary.py`)
The three builds were run alternately, 5 rounds, the order of the trees rotating every round (round 1: 38c4f32, d776713, c8fcbe8; round 2: d776713, c8fcbe8, 38c4f32; ...), on the RTX 4090 with the process on cores 0-3,16-19.
Quiet machine: the load average at the start was 0.46 (1 min) and 0.79 (5 min); the compile jobs started for this check (`pin_transfer.sh`, `regression_build.sh`) had finished (their logs end 20:22:51 and 20:21:31, the window started 20:24:40, and the window script refuses to start while they or an E0/E4 measurement run); 30 runs, all exit code 0.
Each run: 100 launches after a warm-up, CUDA events around one launch (`gpu_mean_ms`). Median over the 5 rounds of the ratio to 38c4f32 measured in the same round (range of the 5 ratios in brackets):

| program | batch | version | 38c4f32 (ms, mean of 5) | d776713 | c8fcbe8 |
|---|---|---|---|---|---|
| E4 | 100 | fkl_fused (ASYMMETRIC) | 0.00872 | 0.996 (0.985-1.007) | 1.004 (0.985-1.007) |
| E4 | 200 | fkl_fused (ASYMMETRIC) | 0.01429 | 0.999 (0.992-1.007) | 1.002 (0.994-1.006) |
| E4 | 300 | fkl_fused (ASYMMETRIC) | 0.01941 | 0.999 (0.998-1.004) | 0.999 (0.996-1.001) |
| E4 | 100 | fkl_fused_half_pixel | 0.00876 | 0.995 (0.991-1.010) | 0.998 (0.996-1.005) |
| E4 | 200 | fkl_fused_half_pixel | 0.01420 | 1.003 (0.987-1.009) | 1.003 (0.999-1.006) |
| E4 | 300 | fkl_fused_half_pixel | 0.01971 | 1.004 (0.996-1.010) | 1.002 (0.996-1.004) |
| E1c | 100 | fkl | 0.00878 | 1.003 (0.941-1.014) | 0.998 (0.940-1.012) |
| E1c | 200 | fkl | 0.01402 | 1.002 (0.990-1.004) | 0.998 (0.996-1.000) |
| E1c | 300 | fkl | 0.01964 | 1.000 (0.992-1.011) | 1.001 (0.988-1.005) |

E1c batches 10 and 50 and the hand-written kernels (control: they do not depend on the library) are in `timing_summary.csv`: the fkl ratios at batches 10 and 50 are 0.991 to 1.000; the hand-written kernels vary between 0.992 and
1.011 (median of the per-round ratios), the FKL kernels between 0.991 and 1.004.

## 3. The E4 kernel at c8fcbe8 against the earlier E1c measurement at 38c4f32
`../e1-preproc-v3/` (library 38c4f32, fkl-benchmarks 74f47c6, median of its 3 runs of `fkl_mean_ms`) against `fkl_fused` of the E4 program at c8fcbe8 in this window (median of 5): batch 100: 0.00873 against 0.00869 ms (ratio 0.995);
batch 200: 0.01404 against 0.01430 (1.019); batch 300: 0.01980 against 0.01942 (0.981). The E1c program itself, rebuilt at c8fcbe8 and run in this window, against the earlier 38c4f32 numbers: 0.979, 0.988, 0.996, 0.996, 0.991 for
batches 10, 50, 100, 200, 300. Nothing moved by more than 2.1 % (E1c batch 10), which is within the spread of the three earlier runs at 38c4f32 among themselves (up to 2.8 % at batch 10, 0.2 to 0.9 % at the other batches).

## Conclusion
On the 4090 the kernels of the batched pre-processing chain are identical at 38c4f32, d776713 and c8fcbe8 (SASS, registers, parameter size), and so are their times (within 1 % in the median; the batches of 200 and 300 are not slower
at c8fcbe8). The 25 % of gfx1200 is not present in the CUDA back-end of the library.
