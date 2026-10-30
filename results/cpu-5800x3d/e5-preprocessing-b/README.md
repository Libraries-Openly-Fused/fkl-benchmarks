# E5, pipeline (b): NV12 video pre-processing, FKL CPU back-end against OpenCV, Ryzen 7 5800X3D

Produced by `opencv-cpu/preprocessing/bench_preproc_b.cpp` (`e5_preprocessing_b_cpp`); the FKL code is
`fkl-self/pipelines/nv12_pipeline.h`, shared with the CUDA and HIP targets (`benchmark_nv12_pipeline_{cu,hip,cpp}`).

A 3840x2160 NV12 frame (seeded noise, luma 16..235, chroma 16..240), N = 1, 8, 16, 32, 64 detection crops of 48..176 x 96..352 pixels.
`convert`: the NV12 frame to a full RGB frame (FKL: ReadYUV + ConvertYUVToRGB, BT.601 studio swing to 8 bit full range RGB;
OpenCV: `cv::cvtColor(COLOR_YUV2RGB_NV12)`). `chain`: the crops, resized to 64x128, normalised, split to an NCHW float tensor.
`pipeline`: both (the FKL path launches two kernels). Variants as in pipeline (a): `fkl_*_fused`, `fkl_*_perop_batch` (one
execution per operation for all the crops), `fkl_*_perop_crop`, `_mt` = ParArch::CPU_MULTITHREAD; `opencv_*_percrop_default`,
`_1thread`, `_parallel` (crops in parallel with cv::parallel_for_).

- `g++-13/`, `clang++-23/`: timing (3 warm-up, 30 timed runs, wall clock, median/min/max/mean) and agreement.
- `natural-image-g++-13/`: only the agreement, with the Kodak image kodim23 tiled over the frame and converted to NV12.

**The conversion dominates the FKL CPU pipeline.** `fkl_convert_cpu` takes 226 ms single threaded (15.5 ms with 16 threads), against
3.4 ms (1 thread) and 0.38 ms (16 threads) for `cv::cvtColor`. The cause is the 8 bit output: the same conversion with a float3 output
takes 11.8 ms (1.4 ns per pixel), the 8 bit one 27 ns per pixel. `ConvertYUVToRGB` saturates and rounds with `cxp::saturate_cast`,
whose `cxp::nearbyint` is a branchy, constexpr friendly round-half-even (data dependent branches on the fractional part) instead of
one `cvtss2si`. This is a property of the FKL CPU code at fb9ac3a, not tuned here.

Numerical agreement as in pipeline (a): FKL against a scalar reference of its sampling (ASYMMETRIC) from its own RGB frame (1e-4),
OpenCV against one with HALF_PIXEL sampling from its own RGB frame (1e-4), single against multithreaded byte for byte. FKL against OpenCV
is informational: the RGB frames differ by at most 1 level (mean 0.035 on the noise, 0.046 on the photograph; 10% and 13% of the
pixels) and the sampling conventions differ (`common/resize_reference.h`). Built with `-DE5_RESIZE_HALF_PIXEL` against an FKL with
ResizeSampling (FKL PR #409), the full chain agrees with OpenCV to 0.0175 (one level of the RGB frame, normalised), mean 6e-4 to 7e-4.

Conditions: idle machine, governor `powersave` (amd-pstate-epp, balance_performance), boost on, OpenCV 4.14.0 (`scripts/build-opencv-cpu.sh`);
memory speed not recorded (needs root).
