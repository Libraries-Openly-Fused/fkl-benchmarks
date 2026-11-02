# E6 (AMD): fk::SaturateCast<float, T> over all 2^32 float bit patterns on the RX 9060 (gfx1200) against the host

`e6_saturate_exhaustive_amd_hip` (`fkl-self/numerics/e6_saturate_exhaustive_amd.h`), the AMD counterpart of `saturate_exhaustive.cu` of the CUDA side (which compares with `cv::cuda::device::saturate_cast` and the PTX
conversion, neither of which exists for AMD). For T in `uchar`, `schar`, `short`, `ushort`, `int` and all 4,294,967,296 float bit patterns, in chunks of 2^26:

- `fkl_dev`: `fk::SaturateCast<float, T>::exec` on the device (gfx1200);
- `fkl_host`: the same function on the host (`cxp::saturate_cast`, the branch free host path of FKL #416);
- `hw`: the conversion of HIP on the device, `__float2int_rn` (round to nearest even; HIP's header defines it as `(int)__builtin_elementwise_rint(x)`, so the result for an out-of-range value or a NaN is not specified by the language: on this toolchain it saturates to the range of `int` and a NaN gives 0, which is what `hw` records) clamped to the range of T;
- `ref`: an independent reference on the host, a NaN gives 0, otherwise `std::nearbyint` of the value as a double clamped to the range of T.

Four comparisons: `fkl_dev_vs_hw`, `fkl_dev_vs_ref`, `fkl_host_vs_fkl_dev`, `fkl_host_vs_ref`. The disagreeing set is reported per class of float (NaN: 16,777,214 patterns, +Inf: 1, -Inf: 1, denormal: 16,777,214, zero: 2, normal: 4,261,412,864)
with the first examples, in the columns of `e6_saturate_exhaustive.csv` of the CUDA program (`type, comparison, class, patterns_in_class, disagreeing, examples`).

## Result: FKL c8fcbe8 (the final pin), fkl-benchmarks `1bb2465`, ROCm 10.1, amdclang++ 24.0.0, gfx1200 (`c8fcbe8/`)

**0 disagreeing patterns in all 20 combinations** (5 types x 4 comparisons), every class. The device and the host of FKL give the same value for every one of the 2^32 float patterns for all five integer types, and both give the value of the HIP
conversion and of the independent reference: a NaN of any sign and payload gives 0 for all five types (on the device and on the host), 2^31 as a float gives the maximum of `int` (2147483647), the limits of the narrow types saturate,
and the ties go to even. Run time about 17 s (the device half takes seconds; the host half runs on 16 threads); idle machine. The program prints neither the time nor the thread count: these three facts were noted by hand when it ran and are not in a file.

## Negative control: the same program against the headers of FKL d776713, before #416 (`negative-control-headers-of-d776713/`)

The program is compiled with the include directory of the FKL tree at d776713 (the `fkl:` line of the CSV header names the pin of the submodule, c8fcbe8, because the include path was replaced after configuring). Result:

| type | fkl_host_vs_fkl_dev and fkl_host_vs_ref | class | examples |
|---|--:|---|---|
| uchar, schar, short, ushort | 0 | | |
| int | 16,777,214 | NaN | `0x7f800001: -2147483648 vs 0` (the host gave the smallest `int`, the device and the reference 0) |
| int | 1 | normal | `0x4f000000` (2^31 as a float): `-2147483648 vs 2147483647` |

16,777,215 patterns in all: exactly the two cases that #416 fixes (NaN to `int`, 2^31 as a float to `int`); `fkl_dev_vs_hw` and `fkl_dev_vs_ref` are 0 also before #416, because the device code is the same. So the host-side counting of the program does detect
differences, and the zeros of `fkl_host_vs_fkl_dev` and `fkl_host_vs_ref` at c8fcbe8 are real. (The host half is the same function in both builds; only the `FK_HOST_RUNTIME_MATH` branch of `constexpr_saturate.h` differs.)
What the control does not show: the device-side counter of `fkl_dev_vs_hw` is 0 in both runs, so no run shows that this path can report a disagreement; and `hw` is `(int)__builtin_elementwise_rint(x)`, a conversion that C++ leaves undefined for
out-of-range values and that saturates with a NaN giving 0 on this toolchain, so "equal to the HIP conversion" says what this toolchain does, not a property of `__float2int_rn`. The zeros against `ref`, which is defined for every pattern, do not depend on that.

## Reproduce

```
cmake -S . -B build/rocm -DFKLB_ENABLE_HIP=ON -DFKLB_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DCMAKE_CXX_COMPILER=<rocm>/bin/amdclang++ -DCMAKE_HIP_COMPILER=<rocm>/bin/amdclang++
ninja -C build/rocm e6_saturate_exhaustive_amd_hip && (cd build/rocm/run/e6_saturate_exhaustive_amd_hip && ../../bin/e6_saturate_exhaustive_amd_hip)    # E6_HOST=0 skips the host half
```
