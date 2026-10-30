# E6: fk::SaturateCast<float, T> over all 2^32 float bit patterns (RTX 4090, CUDA 13.4.3, OpenCV 4.14.0, g++ 15.2)

`fkl-self/numerics/saturate_exhaustive.cu`: every one of the 4,294,967,296 binary32 bit patterns converted to uchar, schar, short, ushort and int by (device) FKL's `SaturateCast`,
`cv::cuda::device::saturate_cast` and the PTX `cvt.rni.sat` (round to nearest even, saturating: what the hardware does), and (host) FKL's `SaturateCast` (cxp::saturate_cast) and `cv::saturate_cast`
(`cvRound`). Six comparisons per type, each over all patterns; the disagreeing set per class of float (NaN, +Inf, -Inf, denormal, zero, normal) with examples is in `e6_saturate_exhaustive.csv`.
The library tree is the pin of fkl-benchmarks (38c4f32); `constexpr_saturate.h` is identical in the LTS-C++17 tip d776713.

| type | FKL dev = OpenCV dev = hardware | FKL host = FKL dev | OpenCV host = OpenCV dev |
|---|---|---|---|
| uchar | all 2^32 patterns | all | 822,083,584 disagree |
| schar | all | all | 1,442,840,576 disagree |
| short | all | all | 1,442,841,218 disagree |
| ushort | all | all | 822,084,353 disagree |
| int | all | **16,777,215 disagree** | 1,619,001,346 disagree |

- **Device:** FKL, OpenCV's device `saturate_cast` and the hardware conversion agree on every pattern of every type, NaN included (a NaN converts to 0 everywhere on the device).
- **FKL host against FKL device:** identical for uchar, schar, short and ushort (a NaN gives 0 on the host as well). For **int** they differ in 16,777,215 patterns: all 16,777,214 NaN patterns (the host gives -2,147,483,648, the device 0) and the single pattern 0x4f000000 (exactly 2^31: host -2,147,483,648, device 2,147,483,647). What the code does (`include/fused_kernel/core/constexpr_libs/constexpr_saturate.h`): it compares the value with the limits of the output type, and inside the range converts `static_cast<T>(nearbyint(s))`. A NaN fails every comparison, `nearbyint` returns a NaN unchanged, and the cast of a NaN is undefined behaviour in C++: on this compiler and CPU it gives -2,147,483,648 for int (and 0 for the narrower types, which is why those agree with the device here, not because the standard says so). For the pattern 2^31 the value passes the "greater than the maximum" test, so the range check does not catch it and the same overflowing cast runs; the likely reason (hypothesis, the generic comparison `cmp_greater` was not traced) is that `INT_MAX` is converted to float, 2,147,483,647 becoming 2,147,483,648.0f, so 2^31 is not greater than it. A fix would use a check that excludes the boundary for int (`s >= 2147483648.0f` gives the maximum) and a NaN rule (0, as the device); it is a library change and has not been made.
- **OpenCV host function against its device version:** they differ for the values from 2^31 on (+Inf included): 822,083,584 patterns for uchar, 1,442,840,576 for schar, 1,442,841,218 for short, 822,084,353 for ushort and 1,619,001,346 for int (examples in `e6_saturate_exhaustive.csv`: `cv::saturate_cast<uchar>` of 0x4f000000 is 0 on the host and 255 on the device). FKL's host function gives 255 there, so FKL host and OpenCV host differ in the same sets (sixth comparison).
