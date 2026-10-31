# E4b, NV12 pipeline on the HIP back-end: RX 9060 (gfx1200), ROCm 10.1, FKL c8fcbe8, steady-state warm-up

`benchmark_nv12_pipeline_hip` (fkl-self/pipelines/benchmark_nv12_pipeline.h) at fkl-benchmarks `534fca7` and FKL `c8fcbe8` (the final pin of LTS-C++17). The NV12 frame is already
in device memory (the upload is not timed); wall clock of the run including the stream synchronisation; 3 warm-up runs, then a warm-up of at least 2 s and until the mean time of
two consecutive windows of 0.25 s differs by at most 2 percent (the GPU clock of the RX 9060 needs about 1 s of continuous work to reach its working value: see `../README.md`),
30 timed runs, median/min/max/mean. **Three repetitions of the whole run with the variants in a rotated order** (`E5_ROTATE_OFFSET` = 0, 1, 2 in `run0`, `run1`, `run2`; the
first variant of a point is variant ((log2 of the crops) + offset) mod 5); the table gives the median of the three medians. Columns `warmup_s`, `warmup_steady`, `sclk_start_mhz`,
`sclk_timed_start_mhz` and `sclk_end_mhz`: the warm-up that was run and the graphics clock at the start of the warm-up, at the start of the timed runs and after them.
`convert` = kernel 1 (NV12 to RGB), `chain_*` = kernel 2 (crops, resize, normalise, split) fused or one execution per operation, `pipeline_*` = both. Agreement with a scalar
double precision reference of kernel 2 in `benchmark_nv12_pipeline_agreement_GPU_AMD.csv` (1e-4). Idle machine (no CI job on the runner), governor `powersave`. Seeded noise frame.

These numbers replace `../e4b-nv12/` (warm-up of 0.5 s, pin 38c4f32), whose first variant was timed on the clock ramp or in its overshoot: convert at one crop read 0.2602 ms there
and 0.2808 ms here, and no longer varies with the number of crops (the kernel does not depend on it). Median ms of the three repetitions:

| crops | convert | chain_fused | chain_perop_batch | pipeline_fused | pipeline_perop_batch | largest spread |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.2808 | 0.0254 | 0.0381 | 0.2843 | 0.2964 | 2.5 % |
| 8 | 0.2811 | 0.0297 | 0.0489 | 0.2893 | 0.3070 | 4.8 % |
| 16 | 0.2840 | 0.0338 | 0.0559 | 0.2928 | 0.3184 | 1.2 % |
| 32 | 0.2827 | 0.0380 | 0.0795 | 0.3014 | 0.3440 | 1.8 % |
| 64 | 0.2836 | 0.0482 | 0.1239 | 0.3138 | 0.3913 | 1.7 % |
