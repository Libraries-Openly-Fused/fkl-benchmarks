# Thread-fusion check of cvgs_fused, 2026-11-08, 11:41-11:43 (compile only, nothing was run)

**The question.** In `opencv-cuda/batched-preprocessing/e4_preprocessing.cu`, `cvgs_fused` calls `cvGS::executeOperations` with its default arguments, which request thread fusion. Does that change the kernel?

**How it was checked.** `tfcheck.cu` holds the `cvgs_fused` call, at batch 10, 150 and 300:
- an array of GpuMat ROIs -> `cvGS::resize<CV_8UC3, INTER_LINEAR, BATCH>` -> `cvtColor<RGB2BGR>` -> `multiply` -> `subtract` -> `divide` -> `split`;
- once with the default `cvGS::executeOperations`, and once, with `-DTF_CHECK_EXECUTE_FALSE=1`, with `cvGS::executeOperations<false>`.

Both objects were compiled with the nvcc command line of `e4_preprocessing_cu` at the head e7a85d4 of #34: its defines, its include response file (`includes_CUDA.rsp`) and its flags, sm_89. The SASS of their kernels was then compared with `scripts/sass_compare_builds.py`, which normalises it per kernel.

**Result: identical.** At batch 10, 150 and 300:
- `sass-default-vs-tf-false.txt`: 3 of 3 kernels identical.
- The normalised listings `sass-default-batch<N>.txt` and `sass-tf-false-batch<N>.txt` are byte-identical: 280 instructions each, with their sha256 in `kernels.tsv`.
- The demangled names differ only in two arguments (`name-*-batch10.txt`): `(fk::TF)1` against `(fk::TF)0`, and the thread-fusion flag of `fk::TransformDPPDetails_<void, true/false, ...>`.

**Why it cannot take effect.** FKL fuses threads only when the read and the write of the chain support it, and here neither does (external/fkl at 64ad9195):
- cvGS's batched resize is FKL's conditional batch read, `BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT>`, whose `THREAD_FUSION` is `false` (`core/execution_model/operation_model/batch_operations.h`).
- `TensorSplit` is `TF::DISABLED` (`algorithms/basic_ops/memory_operations.h`, l.184).

**These are the kernels that were measured.**
- `sass-e4-vs-default.txt`: the default build's three kernels are, by name and SASS, the `cvgs_fused` kernels of `e4_preprocessing_cu` at e7a85d4 (3 identical).
- `sass-merged-main-vs-default.txt`: they are also the kernels of the binary this window measured (main 2dac1c8, sha256 8b56ffc2...; 3 identical). This comparison was a direct `scripts/sass_compare_builds.py` call after the scripts below.

**The scripts, in the order they ran.** They are kept as they ran, with their scratch paths. `console.txt` is their combined console.
1. `tf-check.sh`: the two builds (`build-default.log`, `build-tf-false.log`), the per-kernel listings and `kernels.tsv`.
   - Its two `sass_compare_builds.py` calls compared nothing, because that script takes only executable files and an `nvcc -c` object is not one.
2. `tf-check-compare.sh`: the two objects marked executable, and the comparisons again.
   - The comparison with E4 gave 3 identical.
   - The default against `<false>` paired nothing, because the names also differ in the `TransformDPPDetails_` flag.
3. `tf-check-compare2.sh`: that flag ignored as `(bool)0/1`, which paired nothing. `c++filt`, which the script uses, writes the flag as `true`/`false`.
4. `tf-check-compare3.sh`: the ignore expressions of the E9 comparison, `'\(fk::TF\)[01],'` and `'(?<=TransformDPPDetails_<void,)(?:true|false),'`. This gave 3 of 3 identical.

The two objects are not committed. Their sha256:
- default `6c256ac911729a4e22276327296de2df8635601e9275acb2973b085ffebcd9a7`;
- `<false>` `793d8947754bd6d12fadf6f346388c7e200914677298e2fd6b274961e46570d3`.
