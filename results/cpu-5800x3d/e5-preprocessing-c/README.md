# E5, resolution sweep (c): the chain of pipeline (a) on whole frames, FKL CPU back-end against OpenCV, Ryzen 7 5800X3D

`opencv-cpu/preprocessing/bench_preproc_c.cpp` (`e5_preprocessing_c_cpp`): one image per run (batch 1), 720x480, 1920x1080 and 3840x2160
8UC3 (seeded noise), resized to 64/60 x 128/120 of its size (768x512, 2048x1152, 4096x2304; the factor 15/16 is exact in float), multiply
by 0.3, RGB to BGR, subtract {1, 4, 3.2}, divide by {3.2, 0.6, 11.8}, split to an NCHW float tensor. A 4K frame is 25 MB in and 113 MB
out: the regime that the 60x120 crops of pipeline (a) never reach (a working set far above the caches).
`g++-13/` and `clang++-23/`: `e5_preprocessing_c_CPU.csv` (3 warm-up and 30 timed runs, wall clock, median/min/max/mean) and the agreement.
Variants: `opencv_default` and `opencv_1thread`, `fkl_fused_cpu[_mt]` (one executeOperations), `fkl_perop_cpu[_mt]` (one per operation, through
two host buffers). Checks as in pipeline (a). Conditions: idle machine, governor powersave (amd-pstate-epp, balance_performance), boost on.

Numerics, the float source positions: FKL's `Resize` computes the source position `x * (src / dst)` in `float`. When the factor is not exact in
`float` the positions carry a rounding error that grows with the coordinate: with the 480p frame at 640x480 (output 683x512, factor 640/683) the fused
FKL result differs from the double precision scalar reference of `common/resize_reference.h` by up to 0.0049 (the tolerance of the check is 1e-4;
g++ 13, the same benchmark with only the 480p entry changed), against 7.6e-6 at 720x480 (output 768x512, factor 15/16, exact). The 480p entry is
therefore 720x480. It is a property of the float source positions of the library, not of this benchmark; the library is not changed for it.
