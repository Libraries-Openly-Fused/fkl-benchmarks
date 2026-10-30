# E4b, NV12 pipeline: the portable benchmark built for the CPU (`benchmark_nv12_pipeline_cpp`), Ryzen 7 5800X3D

The same header as the HIP and CUDA targets, built with g++ 13 for `ParArch::CPU` and `ParArch::CPU_MULTITHREAD`. Rows for both back-ends
in one CSV (`back_end` column). See `results/rx9060/e4b-nv12/README.md` for the variants; the CPU back-ends use 3 warm-up runs. The CPU
rows with OpenCV are in `results/cpu-5800x3d/e5-preprocessing-b/`.
