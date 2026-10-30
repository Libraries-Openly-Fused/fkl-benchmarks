# E5, pipeline (a): batched DNN pre-processing, FKL CPU back-end against OpenCV, Ryzen 7 5800X3D

Produced by `opencv-cpu/preprocessing/bench_preproc_a.cpp` (`e5_preprocessing_a_cpp`), commit in the `# repo:` line of each CSV.

- `g++-13/`, `clang++-23/`: the timing sweep (batch 10 to 300 step 10) and the numerical agreement of every point.
  `e5_preprocessing_a_CPU.csv`: median, min, max and mean wall clock ms of one full pipeline run (all crops of the batch),
  3 warm-up runs and 30 timed runs per point and variant. `e5_preprocessing_a_agreement_CPU.csv`: max and mean absolute
  difference between outputs, the fraction of values over 1e-4, and whether the check passed (`required` 0 = informational).
- `natural-image-g++-13/`: only the agreement (no timing), with a natural image (Kodak kodim23) instead of the seeded noise.

Variants: `opencv_default` and `opencv_1thread` (the "OCVBatch" loop of cvGPUSpeedup's test_batchresize_x_split3D on the CPU,
with cv::getNumThreads() threads and with 1), `opencv_percrop_*` (every operation on each crop), `opencv_percrop_parallel` (the
crops processed in parallel with cv::parallel_for_), `fkl_fused_cpu[_mt]` (one executeOperations for the batch),
`fkl_perop_batch_cpu[_mt]` (one executeOperations per operation for the batch), `fkl_perop_crop_cpu[_mt]` (one per operation and
per crop). `_mt` is ParArch::CPU_MULTITHREAD.

Conditions: the machine was idle (nothing else running), governor `powersave` with amd-pstate-epp / balance_performance, boost
on (recorded in the header), OpenCV 4.14.0 built by `scripts/build-opencv-cpu.sh`, FKL pinned in `external/fkl`. FKL and OpenCV do not
sample at the same source positions (`common/resize_reference.h`), so FKL against OpenCV is informational; each is checked against
the scalar reference of its own convention. The g++ 13 sweep was run twice (a scratch build and this repository's build): over all points and variants the medians differ by
0.9% (median absolute difference), the worst point by 17.5% (`fkl_perop_batch_cpu_mt`, batch 100); the multithreaded points at small
batches are the noisiest ones (see the min and max columns).
