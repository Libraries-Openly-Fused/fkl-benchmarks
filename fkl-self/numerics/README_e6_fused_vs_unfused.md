# E6 (reduced set): fused against kernel-per-op, -fmad, special values, aliasing

This file is written and committed BEFORE the program `e6_fused_vs_unfused.cu` exists and before any result: it fixes what is compared and what counts as the specified result. Results are judged against it and it is not edited afterwards except to add a dated note at the end.

## What is compared

1. **Fused against kernel-per-op.** Each pipeline runs (a) fused, in one `executeOperations`, and (b) with one `executeOperations` per operation, the intermediates in `Ptr2D` device buffers of the type the operation returns (float stays float, so (b) rounds to fp32 after every operation; (a) keeps the values in registers). P2's fused path is the E4a program's own kernel; its per-op path is FKL, one launch per operation. The README of the results lists the type and buffer of every step.
2. **-fmad.** The same source is built with `rtx4090` (-fmad on, the nvcc default) and `rtx4090-fmad-off` (-fmad=false). Outputs of the same case are compared across the builds, fused and per-op, and each against an fp64 host reference. `-fmad` concerns device code only; `-ftz` and `-prec-div`/`-prec-sqrt` are not changed by either preset (defaults: denormals kept, IEEE division).
3. **Special values** per operation type (below). 4. **Aliasing** (below).

## Measures

- Bit identity: `memcmp` of the output bit patterns. `value ==` is never used for 'identical'. +0 and -0 are different bit patterns and counted separately. A NaN is identical only when the bit patterns equal; when the claim is 'is a NaN' it is stated so.
- ULP distance (fp32 outputs): the bit patterns mapped to a monotonic integer, the difference of the integers. NaN pairs are excluded from the ULP statistics and counted (NaN count of each side and of the pair mismatches). Integer outputs: the absolute integer difference.
- Against fp64: the host computes the same operation sequence in double (compiled with `-ffp-contract=off`, so GCC does not contract the host reference), rounds the final result once to float (nearest even) and the ULP distance of the device result to it is reported; additionally the error of the device result against the unrounded double result in ULP of the result (a real number; at most 0.5 means correctly rounded).
- Every comparison reports the NaN count and the Inf count of both sides.
- 5 seeds per pipeline, recorded; the CSV has the value of each seed, the worst case and the spread; numbers in the text are the worst case.

## Pre-registered expected results (special values)

The input set is: quiet NaN, signalling NaN (payload 0x200000 and sign variants), +Inf, -Inf, +0, -0, the smallest and the largest positive denormal, FLT_MIN, FLT_MAX, +-0.5, +-1.5, +-2.5, 254.5, 255.5, 127.5, the values just inside and outside each integer saturation limit (255/256, 127/128, -128/-129, 32767/32768, -32768/-32769, 65535/65536, 2^31 -1/2^31, -2^31), +-2^32.

| operation | input | expected (specified) result |
|---|---|---|
| Add, Sub, Mul, Div, Mul-add (float) | any operand NaN | a NaN (any payload and sign; a signalling NaN is quieted by the hardware; the test is 'is a NaN' and the payloads are listed separately) |
| | Inf op finite | IEEE: Inf for Add/Sub/Mul by a nonzero finite (sign by the rules), Inf - Inf = NaN, Inf x 0 = NaN, Inf / Inf = NaN, finite / Inf = +-0, x / +-0 = +-Inf (0 / 0 = NaN) |
| | zeros | IEEE signs: (+0) + (-0) = +0, (-0) + (-0) = -0, (-0) x positive = -0, x - x = +0 |
| | denormals | kept (no -ftz): the IEEE result, e.g. denormal + denormal exact, FLT_MIN x 0.5 = the denormal with exponent 0 mantissa 0x400000 |
| | overflow | FLT_MAX x 2 = +Inf, FLT_MAX + FLT_MAX = +Inf |
| Cast float -> uchar/schar/short/ushort/int (non-saturating `Cast`) | NaN, +-Inf, out of range | undefined behaviour in C++: NOT scored pass/fail; the observed result per back-end (the device here) is reported as a table |
| | in range, ties | round toward zero as the C++ conversion (the exact rule of `Cast` is read from the source and written here after reading it, before running; see the dated note) |
| SaturateCast float -> T | NaN | 0 (all T) |
| | +-Inf and out of range | the limit of T (saturation) |
| | ties | round to nearest even (0.5 -> 0, 1.5 -> 2, 2.5 -> 2, 254.5 -> 254, 255.5 -> 255 (saturates), -0.5 -> 0 / -0.0 -> 0, -127.5 -> -128 by evenness and range) |
| | in range | the nearest integer |
| Resize (INTER_LINEAR, ASYMMETRIC and HALF_PIXEL) on an image with the special values planted at known pixels | output pixel whose bilinear neighbourhood contains a NaN | NaN (a NaN weight x any = NaN, so NaN propagates through the neighbours with a nonzero or zero weight alike: 0 x NaN = NaN) |
| | neighbourhood contains +Inf (no NaN) | +Inf, or NaN where the weights make Inf - Inf or 0 x Inf (weight 0 times Inf = NaN) |
| | otherwise | the documented formula (bilinear with the sampling convention) evaluated in double on the planted values, rounded to float: within 1 ULP of it is a pass |
| ColorConversion (RGB<->BGR, RGB to gray with the coefficients of the source) | NaN or Inf in a channel | gray: NaN or Inf by the same arithmetic rules as above for the formula written in the source; channel reorder: the bits move unchanged |
| Normalize (Mul by scale, Sub mean, Div std) | as the arithmetic rows | the arithmetic rows applied in sequence |

The pass criterion for a value is 'the result has the specified class or value'; for finite results of Resize, ColorConversion and Normalize 'within 1 ULP of the double evaluation of the documented formula'. The paper's sentence will read 'N of M give the specified result; the exceptions are: ...' with this table as the specification.

## Aliasing (pre-registered)

- Element-wise per-thread chain (1,000 mul/add on 2,073,600 floats), input pointer = output pointer: expected bit-identical to the out-of-place result (each thread reads and writes only its own element).
- ReadBack gather (a 3-tap neighbour read, i.e. a thread reads elements written by other threads), in place: expected NOT identical (a race: undefined), run three times, the number of differing elements of each run is reported.

## Notes (dated, appended after the fact only)

- **2026-11-04, before the program and before any run: the rules read from the FKL source (external/fkl at c8fcbe8).** `fk::Cast<I, O>::exec` is `static_cast<O>(input)` (`cxp::cast`, core/constexpr_libs/constexpr_cmath.h:387): float to integer rounds toward zero, so in-range ties give 0.5 -> 0, 1.5 -> 1, 2.5 -> 2, 254.5 -> 254, -0.5 -> 0, -1.5 -> -1, -127.5 -> -127; this is the specified result of the 'in range, ties' row of Cast (not round to nearest even, which is SaturateCast). `RGB2Gray` (CCIR 601, float output) is `(x * 0.299f) + (y * 0.587f) + (z * 0.114f)` evaluated in this order in float (a left-to-right sum of three float products); the double evaluation of the same formula with the float constants converted to double is the reference of 'within 1 ULP'; with -fmad on the device may contract the products and sums into fma, which is a result of the -fmad comparison, not an exception of this table. ColorConversion RGB<->BGR moves the channels unchanged. Resize INTER_LINEAR sampling: ASYMMETRIC source coordinate = x * (srcSize / dstSize), HALF_PIXEL = (x + 0.5) * (srcSize / dstSize) - 0.5; neighbours beyond the ROI are clamped to the ROI (E4a convention); the output pixels that read a clamped neighbour are listed by the program from this formula before the kernel runs. The pass criteria of the table are not changed by this note.
