# Do the numbers of E0 to E2 transfer to the final library pin c8fcbe8? (RTX 4090, sm_89, CUDA 13.4.3)

The numbers of the paper were measured with older trees of the library: E0 (and the E6 NaN program) with fb9ac3a (LTS-C++17 after #406), E1, E2 and the data-size mul4 sweep with 38c4f32 (after #409). The pin of the
revision is c8fcbe8 (merge of #417). **Every kernel of every program that produced a number was rebuilt against c8fcbe8 and compared, one by one, with the kernel of the binary that produced the number: 1,117 kernels in 21 binaries,
1,117 identical, 0 different, 0 present on one side only.**

## What changed in the library between the trees
`git diff --stat fb9ac3a c8fcbe8 -- include` (5 files):
| file | change | reaches device code of these programs |
|---|---|---|
| `algorithms/image_processing/resize.h` (#409, in 38c4f32) | `ResizeSampling` policy (`ASYMMETRIC` default, `HALF_PIXEL`), a defaulted template argument `SAMPLING` of the resize operations | the name of resize kernels (mangled name only); the code of the default is the same |
| `algorithms/image_processing/color_conversion.h` (#411) | a combination without a conversion matrix is a compile-time error instead of a zero matrix | no (compile-time check) |
| `core/constexpr_libs/constexpr_cmath.h` (#414) | `nearbyint` at host run time uses the CPU instruction, under `FK_HOST_RUNTIME_MATH` | no: the macro is not defined in the device passes of nvcc and HIP |
| `core/constexpr_libs/constexpr_saturate.h` (#416) | host run-time saturate_cast without branches, same guard | no |
| `core/utils/parameter_pack_utils.h` (#417) | `get_arg<I>` through `pp_detail::Nth<I>::pick` instead of a tuple of references | yes, it is used by the fused kernels; it removes the 1,024-operation cliff |

## Method (`scripts/pin_transfer.sh`, `scripts/sass_compare_builds.py`, `scripts/pin_transfer_table.py`)
For every result the program is taken from the commit of fkl-benchmarks that produced it (the `# repo:` line of its CSV; e1_vf_chain and e1_hf_batch were measured at 5201059, whose sources of these two programs are identical to 74f47c6; for the programs added later, e9e599d) in a separate worktree with `external/fkl` replaced by c8fcbe8,
configured with the same preset (`local-rtx4090`, or `rtx4090-e1` without OpenCV and NPP for E1), built with the same nvcc (13.4.92) and flags, and every kernel of the sm_89 SASS (`cuobjdump -sass`) is compared with the same
kernel of the frozen binary (`build/e0-frozen-<commit>/bin` for E0, `build/rtx4090-e1/bin` for E1, `build/e4/bin` for E2 and mul4, `build/local-rtx4090/bin` for nan_saturate). The SASS is normalised (addresses and encodings removed) and compared
instruction by instruction, **register numbers included** (no renaming). Compile only: nothing is executed in this folder.
Kernels are matched by mangled name. Where some names exist on one side only (`sass_<group>.txt`), the comparison is repeated matching by demangled name, whitespace removed, with `, (fk::ResizeSampling)0` removed (`sass_<group>_demangled.txt`); that
is the only normalisation of names. Two causes of different mangled names exist, both unrelated to the code: (1) the defaulted `ResizeSampling` template argument that PR #409 added to the resize operations (the frozen E0 binaries were built
with fb9ac3a, before it), which appears in the names of the resize kernels; (2) the per-compile hash of the anonymous namespace (`_GLOBAL__N__<hash>_`) in the names of hand-written kernels of the E1 programs and of the `l2Flush` kernel of E2.
Source checks: rebuilding the E2 and mul4 programs at the OLD library (38c4f32) gives kernels identical to the frozen binaries (E2 7 of 7, mul4 2 of 2), so the frozen binaries are what the sources produce and the rebuild at c8fcbe8 is a like for like comparison.

## Every binary compared (`transfer_table.md`, `transfer_table.csv`)
| group | benchmarks commit | baseline (library) | binary | results it produced | kernels in baseline | in c8fcbe8 build | identical | different | only on one side | matched by |
|---|---|---|---|---|--:|--:|--:|--:|--:|---|
| a334424 | a334424 | frozen binary of the measurement (fb9ac3a) | `benchmark_latency_hiding_cu` | e0-latency-hiding | 61 | 61 | 61 | 0 | 0 | mangled name |
| a334424 | a334424 | frozen binary of the measurement (fb9ac3a) | `benchmark_numinstructions_x_operation_cu` | e0-numinstructions | 185 | 185 | 185 | 0 | 0 | mangled name |
| 269d229 | 269d229 | frozen binary of the measurement (fb9ac3a) | `benchmark_CPUandGPU_cvGS_vs_fk_cu` | e0-wrapper-overhead | 2 | 2 | 2 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 269d229 | 269d229 | frozen binary of the measurement (fb9ac3a) | `benchmark_CPU_OpenCV_vs_cvGS_cu` | e0-cpu-time-opencv | 16 | 16 | 16 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 269d229 | 269d229 | frozen binary of the measurement (fb9ac3a) | `benchmark_data_type_cu` | e0-data-type | 26 | 26 | 26 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 5aac244 | 5aac244 | frozen binary of the measurement (fb9ac3a) | `benchmark_CPU_NPP_vs_FastNPP_cu` | e0-cpu-time-npp | 16 | 16 | 16 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 5aac244 | 5aac244 | frozen binary of the measurement (fb9ac3a) | `benchmark_npp_batchresize_x_split3D_cu` | e0-npp | 15 | 15 | 15 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 1a70cab | 1a70cab | frozen binary of the measurement (fb9ac3a) | `benchmark_image_resolution_MAD_loop_cu` | e0-data-size | 6 | 6 | 6 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 1a70cab | 1a70cab | frozen binary of the measurement (fb9ac3a) | `benchmark_horizontal_fusion_cu` | e0-hf | 122 | 122 | 122 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 1a70cab | 1a70cab | frozen binary of the measurement (fb9ac3a) | `benchmark_CPUandGPU_cvGS_vs_fk_cu` | e0-wrapper-overhead-order | 2 | 2 | 2 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 1a70cab | 1a70cab | frozen binary of the measurement (fb9ac3a) | `resize_convention_cu` | e6-resize-convention | 2 | 2 | 2 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| a266bc9 | a266bc9 | frozen binary of the measurement (fb9ac3a) | `graph_instantiate_time_cu` | e0-graph-instantiate | 0 | 0 | 0 | 0 | 0 | mangled name |
| a266bc9 | a266bc9 | frozen binary of the measurement (fb9ac3a) | `benchmark_vertical_fusionCompareMulAdd1_cu` | e0-vf-mul-add1, e0-vf-mul-add1-run1 | 200 | 200 | 200 | 0 | 0 | mangled name |
| a266bc9 | a266bc9 | frozen binary of the measurement (fb9ac3a) | `benchmark_vertical_fusionCompareMul1_cu` | e0-vf-mul1 | 200 | 200 | 200 | 0 | 0 | mangled name |
| a266bc9 | a266bc9 | frozen binary of the measurement (fb9ac3a) | `benchmark_vertical_fusionCompareMulAdd_cu` | e0-vf-mul-add | 200 | 200 | 200 | 0 | 0 | mangled name |
| e1 | 74f47c6 | frozen binary of the measurement (38c4f32) | `e1_vf_chain_cu` | e1-vf-chain-v2 | 18 | 18 | 18 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| e1 | 74f47c6 | frozen binary of the measurement (38c4f32) | `e1_hf_batch_cu` | e1-hf-batch-v2 | 16 | 16 | 16 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| e1 | 74f47c6 | frozen binary of the measurement (38c4f32) | `e1_preproc_cu` | e1-preproc-v3 | 20 | 20 | 20 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| e2-old | e9e599d | rebuild at 38c4f32 against the frozen binary (source check) (38c4f32) | `e2_fkl_chain_cu` | e2-chain (same binary rebuilt at the old library) | 7 | 7 | 7 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| e2-new | e9e599d | frozen binary of the measurement (38c4f32) | `e2_fkl_chain_cu` | e2-chain | 7 | 7 | 7 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| mul4-old | e9e599d | rebuild at 38c4f32 against the frozen binary (source check) (38c4f32) | `benchmark_image_resolution_MAD_loop_mul4_cu` | e0-data-size-mul4 (same binary rebuilt at the old library) | 2 | 2 | 2 | 0 | 0 | mangled name |
| mul4-new | e9e599d | frozen binary of the measurement (38c4f32) | `benchmark_image_resolution_MAD_loop_mul4_cu` | e0-data-size-mul4, e0-data-size-parity | 2 | 2 | 2 | 0 | 0 | mangled name |
| nan_saturate | f3afadc | frozen binary of the measurement (fb9ac3a) | `nan_saturate_cu` | e6-nan-saturate | 1 | 1 | 1 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| **total (new tree against the baseline)** | | | 21 binaries | | **1117** | **1117** | **1117** | **0** | **0** | |

Kernel counts are the kernels in the sm_89 SASS of the binary (the binaries also carry compute_89 PTX, which is not compared). `graph_instantiate_time_cu` has no kernel of its own: it measures capture and instantiation of the OpenCV-CUDA
baseline with CUDA Graphs, and its kernels are in the OpenCV libraries, which this proof does not change.

## What is not covered by this proof, and why
- **Results that are not from a program of this list.** E3 (compile time): `../e3-compile/` (the tree check at c8fcbe8). E4 (`../e4-preprocessing/`) and `../e4-resize-regression/` were built against c8fcbe8 or against all three trees directly. E6 exhaustive saturate_cast:
  `../e6-saturate-exhaustive-c8fcbe8/` (run against c8fcbe8; the saturate and cmath headers are also byte-identical to PR #416's head 231dec8, which `../e6-saturate-exhaustive-pr416/` tested).
  The NPP series (`e0-npp`, `e0-cpu-time-npp`, E4a NPP) are listed above for completeness; they will be re-run with the ctxfix configuration after FastNPP #45 is merged.
- **Superseded:** `e0-graph-instantiate-preliminary` (first half of the graph-instantiate experiment, replaced by `e0-graph-instantiate`, same program).
- **Host code.** The kernels are identical, so the device side of every time is the same. The host-side code of the library differs in `get_arg` (#417) and in the host run-time paths of `nearbyint` and `saturate_cast` (#414, #416), which no experiment executes on the host.
  Whether the host-side time of the programs that report a host time (`e0-cpu-time-opencv`, `e0-cpu-time-npp`, `e0-wrapper-overhead`, `e0-data-type`) moved is measured in `host-check/` (next section); the NPP program is not part of it (it waits for FastNPP #45).

## Host-side time: the frozen binary against the rebuild (`host-check/`)
The kernels are identical, so the question left is whether the host time of the calls (the time the CPU spends enqueuing the work) moved with the library's host-side changes (`get_arg` #417, the host run-time paths of `nearbyint` and `saturate_cast`).
The window of 2026-11-01 17:38 to 17:43 (second part of its script), 17:41 to 17:43, pinned to cores 0-3,16-19, no other job running (`processes-before.txt`), load every 20 s and SM clock every 100 ms:
the three programs whose number is a host time, each run from the **frozen binary** (`build/e0-frozen-269d229/bin`: fkl-benchmarks 269d229 against FKL fb9ac3a, the binary that made the E0 numbers) and from the **rebuild** (fkl-benchmarks 269d229 against FKL c8fcbe8,
`pin-transfer-work/src-269d229`), 10 rounds, alternating (odd rounds frozen first, even rounds rebuild first), 60 runs, all exit 0. Binaries and trees: `host-check/binary-sha256.txt`; one folder per run, `<program>_<frozen|rebuild>_r<round>`.
`python scripts/host_check_summary.py results/rtx4090/e0-pin-transfer/host-check` (writes `host_check_summary.csv`): per round, the ratio rebuilt / frozen of the mean host time of the cvGS column row by row, then the median over the rows; the table gives the median, smallest and largest over the 10 rounds.
The second column of each file is the baseline of that file: in `e0-cpu-time-opencv` and `e0-data-type` it is OpenCV-CUDA, whose calls do not depend on the library, so its ratio is the spread of the method (the control); in `e0-wrapper-overhead` it is the raw `fk::` API,
which is FKL code too, so there it is a second measurement of the library, not a control.

| program (experiment) | rows | cvGS column, rebuilt / frozen | second column (OpenCV-CUDA, or `fk::` API in the wrapper file), rebuilt / frozen |
|---|---:|---|---|
| `benchmark_CPU_OpenCV_vs_cvGS` (e0-cpu-time-opencv) | 16 batch sizes | 0.999 [0.987-1.009] | 0.999 [0.980-1.012] |
| `benchmark_CPUandGPU_cvGS_vs_fk`, GPU file (e0-wrapper-overhead) | 1 | 0.997 [0.969-1.014] | 1.003 [0.974-1.025] |
| the same, CPU-enqueue file | 1 | 1.005 [0.960-1.036] | 1.007 [0.968-1.025] |
| `benchmark_data_type` (e0-data-type) | 26 types | 1.001 [0.994-1.005] | 1.001 [0.996-1.008] |

All four medians are within 0.5 % of 1, and the spread over the 10 rounds (1 % to 4 %) is of the size of the spread of the second column, which in the two OpenCV files is the control: the host-side numbers of `e0-cpu-time-opencv`, `e0-wrapper-overhead` and `e0-data-type` made with the frozen binary hold at c8fcbe8 within the run-to-run spread, and they are kept as they are.
The files in `../e0-cpu-time-opencv/` against the median of the 10 frozen runs, speedup per batch size: median 1.003, smallest 0.975 (batch 112), largest 1.060 (batch 122: 400.3x in the E0 file, 377.6x median of the 10 runs, range 368.7x to 387.7x), i.e. the E0 file is one sample of the same distribution.

PCIe link (relevant for every host-bound number above, since each launch is submitted over the link): not logged during these runs; the link check under load made afterwards, `../pcie-check/` (generation 4, width 16 whenever the GPU leaves the idle state; pinned copies 25.6 and 24.8 GB/s), is the evidence that the link of the RTX 4090 runs at PCIe 4.0 x16 when it works.

## Reproduce
`source scripts/env-cuda13.4.sh; scripts/pin_transfer.sh external/fkl <work dir> results/rtx4090/e0-pin-transfer <worktree of 38c4f32>` (needs the frozen builds in `build/`), then `python scripts/pin_transfer_table.py results/rtx4090/e0-pin-transfer`.
`ONLY="e1 e2-new"` runs just some groups. The raw comparison of every group is in `sass_<group>.txt` (per binary: kernels on each side, identical, different).
