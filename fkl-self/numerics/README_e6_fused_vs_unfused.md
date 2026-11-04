# E6 (reduced set): fused against kernel-per-op, -fmad on and off

This file is written and committed BEFORE the program `e6_fused_vs_unfused.cu` exists and before any result: it fixes what is compared and what counts as the specified result. Results are judged against it and it is not edited afterwards except to add a dated note at the end.

## What is compared

1. **Fused against kernel-per-op.** Each pipeline runs (a) fused, in one `executeOperations`, and (b) with one `executeOperations` per operation, the intermediates in `Ptr2D` device buffers of the type the operation returns (float stays float, so (b) rounds to fp32 after every operation; (a) keeps the values in registers). P2's fused path is the E4a program's own kernel; its per-op path is FKL, one launch per operation. The README of the results lists the type and buffer of every step.
2. **-fmad.** The same source is built with `rtx4090` (-fmad on, the nvcc default) and `rtx4090-fmad-off` (-fmad=false). Outputs of the same case are compared across the builds, fused and per-op, and each against an fp64 host reference. `-fmad` concerns device code only; `-ftz` and `-prec-div`/`-prec-sqrt` are not changed by either preset (defaults: denormals kept, IEEE division).
(The special-value study and the aliasing test that this file originally also specified were removed on 2026-11-04 with their results; the full pre-registered text is at c04c2db and the dated notes at 91dfcdd.)

## Measures

- Bit identity: `memcmp` of the output bit patterns. `value ==` is never used for 'identical'. +0 and -0 are different bit patterns and counted separately. A NaN is identical only when the bit patterns equal; when the claim is 'is a NaN' it is stated so.
- ULP distance (fp32 outputs): the bit patterns mapped to a monotonic integer, the difference of the integers. NaN pairs are excluded from the ULP statistics and counted (NaN count of each side and of the pair mismatches). Integer outputs: the absolute integer difference.
- Against fp64: the host computes the same operation sequence in double (compiled with `-ffp-contract=off`, so GCC does not contract the host reference), rounds the final result once to float (nearest even) and the ULP distance of the device result to it is reported; additionally the error of the device result against the unrounded double result in ULP of the result (a real number; at most 0.5 means correctly rounded).
- Every comparison reports the NaN count and the Inf count of both sides.
- 5 seeds per pipeline, recorded; the CSV has the value of each seed, the worst case and the spread; numbers in the text are the worst case.

## Notes (dated, appended after the fact only)

- **2026-11-04, before the program and before any run:** the rules read from the FKL source (Cast, the gray formula, the sampling conventions) were written here; the part that still matters for the kept comparison is the sampling convention of Resize: ASYMMETRIC source coordinate = x * (srcSize / dstSize), HALF_PIXEL = (x + 0.5) * (srcSize / dstSize) - 0.5, neighbours beyond the ROI clamped to the ROI (E4a convention); the full note is at 91dfcdd.
- **2026-11-04, after run 1 and run 2 (no row above is changed):** the special-value findings (RGB2Gray with float output, one Normalize contraction) are in the history (d6e0421); no row of the kept specification was changed.
