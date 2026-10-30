# E4b, NV12 pipeline on the HIP back-end: RX 9060 (gfx1200), ROCm 10.1

`benchmark_nv12_pipeline_hip` (fkl-self/pipelines/benchmark_nv12_pipeline.h), the same header as the CUDA and CPU targets. The NV12 frame is
already in device memory (the upload is not timed); wall clock of the run including the stream synchronisation, a warm-up of 3 runs and at
least 0.5 s (a GPU at rest runs at a low clock), 30 timed runs, median/min/max/mean. The `clocks` line of the header is read before the warm-up.
`convert` = kernel 1 (NV12 to RGB), `chain_*` = kernel 2 (crops, resize, normalise, split) fused or one execution per operation, `pipeline_*` =
both. Agreement with a scalar double precision reference of kernel 2 in `benchmark_nv12_pipeline_agreement_GPU_AMD.csv` (1e-4).
