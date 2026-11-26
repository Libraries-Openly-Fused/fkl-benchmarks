# The pin move to FKL #435 (48fbf3f), CUDA side: the ports, device code, host code and paper facts (RTX 4090 machine, CUDA 13.4.3; compile only, nothing ran)

FKL #435 (branch `LTS-C++17-executeoperations-dpps`, head 48fbf3f08e7551c3eb70ac2302d8ebc15ecad645, on LTS-C++17 b10fc66) makes every DPP run
through `fk::executeOperations<DPP>(stream, ...)`, with one kernel template, `launchDPP_Kernel<DPP, Details, IOps...>`. The attention DPPs
take a leading `ParArch`, and invalid arguments throw `std::invalid_argument` instead of returning `false`. Its CUDA check is
`../fkl-pr435-check/` (#65) and its HIP check is `../../rx9060/fkl-pr435-check/` (#64). On 2026-11-26 it was open, not merged.

fkl-benchmarks #66 (branch `pin-fkl-435`) moves `external/fkl` to it. This record is the CUDA side of that branch at **2518c75**, on main 976e42b:
- AMD's two commits: ede9e5e (`external/fkl` from b10fc66 to 48fbf3f) and a4b493c (the HIP programs, `../../rx9060/pin-move-devcode-b10fc66-48fbf3f/`, #67);
- the four CUDA commits of this record (section 1): f3d66da, b51e53f, 8165270 and 2518c75.

**The checks.** Every CUDA program of the repository was built twice, with the same tools and flags:
- base: fkl-benchmarks 976e42b (the code of main bd019f6) with FKL b10fc66, its gitlink;
- head: `pin-fkl-435` at 2518c75 with FKL 48fbf3f.

Then:
- **(a)** their sm_89 device code, kernel by kernel (section 2);
- **(b)** the x86-64 host code of the programs behind the RTX 4090 paper rows: base against head (what the pin move changes), and the binary
  that made each record against the base build (what main changes since the measured commit) (section 3);
- **(c)** compile-only facts of the paper at the head: VI-P (E11), VI-L (E3) and the SHORT LISTING blocks of `npp/e9_cross_library.cu` (section 4).

**Nothing was run.** No GPU was visible to any step (`CUDA_VISIBLE_DEVICES` empty). The frozen 2025 programs, among them
`benchmark_image_resolution_MAD_loop_cu` and `opencv_parity_probe_cu` that 8165270 edits, were compiled and their code compared, nothing more.
This was done on 2026-11-26 from 15:35 to 17:46 on the RTX 4090 machine (Ryzen 9 5950X, g++ 15.2.0, nvcc 13.4.92).

Decision: the coordinator's assignment C-PM-CUDA of 2026-11-26, the CUDA side of the pin move to FKL #435. It follows Oscar's task C
decisions in the coordinator session: QA3 for E16 (the automatic choice made in place in each launch, no helper) and Q2 for the edit of the
two frozen programs (compiled only, never run).

## Result

- **Device code:** 1,776 of the 1,816 paired kernels are identical (39 programs).
  - The 40 that differ are the E18/E19 filter kernels (19 LinearFilterDPP, 17 MorphologyDPP) and 4 split-KV combine kernels of E16.
  - Every other kernel behind a paper row is identical.
  - The E19 `fkl_batch` kernel (MorphologyDPP) is among the 17. At N = 1 crop its 40 B of stack go (LDL 2 and STL 5 go to 0); the other 15 keep
    their instruction counts.
  - 18 kernels are in the base only and 2 in the head only, all split-KV kernels that no measured configuration launches (section 2).
- **Host code, base against head (the pin move):**
  - **unchanged, except the moved `__LINE__` values**, for every function of the E1, E2, E4, NPP and CPU-time programs (`e1_*`, `e2_fkl_chain`,
    `e4_preprocessing`, `benchmark_npp_batchresize_x_split3D`, `benchmark_CPU_NPP_vs_FastNPP`, `benchmark_CPU_OpenCV_vs_cvGS`). Their
    sources do not change; FKL #435 moves the lines of `executors.h`.
  - **E19:** the `opencv_cuda` row is identical. The `npp` row differs in one function, `std::operator+`, which it reaches only through the
    error message of a failed status check. `fkl_batch` differs in its ported call: the row's lambda, and the new `fk::executor_detail::requireArguments`,
    which runs on every call.
  - **E16:** every row differs, because the port moves FKL's automatic choice into the program (QA3). Its kernels are the measured ones (the choice
    check below). One difference is inside the timed calls. For the 6 measured MMA kernels with more than 48,000 B of dynamic shared memory, the
    head calls `cudaFuncSetAttribute` **twice per call** (in `fk::flashAttentionMmaSplits` and in the executor), where the base called it once
    (section 3.5). Its effect on a time is not measured here.
- **Host code, the measured binary against main** (sha256 of each binary checked against its record):
  - **E1, E2, E16, NPP, CPU-time, E4 build-once (f534a38):** every function the timed calls reach is identical up to `__LINE__` and the
    build path. What differs is outside them: the build-information header (`writeSystemInfoHeader`), and for E2 one CSV note string in `launch()`.
  - **E4 at 836a98e (VI-K) and at 469aa6d (Figure 15, whole call):** the sources of the rows' launch lambdas and of `fklb::timeIt` are the same
    at main. `timeIt`'s own interval (the instructions between its two events) is identical at 123 of the 124 rows-and-N of 836a98e, and at
    224 of the 244 of 469aa6d (19 of the other 20 differ only in register names). But in the whole-call rows `fkl_fused`,
    `fkl_fused_half_pixel` and `cvgs_fused`, the code GCC made of the unchanged lambdas differs. At main the host stubs of the 180 FKL kernel
    instantiations are separate functions; in the measured binaries they were inlined into the lambdas. FKL's crop checks (`crop2D`,
    `checkAxis`) differ too. The same out-of-line stubs are in the f534a38 build, which has main's E4 source with the old FKL 64ad9195. So
    they come with the E4 source of #37 (the build-once rows), not with FKL (section 3.4). No cause beyond that is claimed.
  - **E19:** #60's record (`../fkl-pr433-pin-check/e19-host-code/`) compared exactly the measured source and FKL (a8f9daa, 24478f0) with main's
    (2160c70, b10fc66). There the `fkl_batch`, `opencv_cuda` and `npp` timed calls were identical.
- **E16 choice:** at all 420 MMA rows of the record, the ported automatic choice names the measured kernel and gives 1 split. That makes
  12 MMA kernels and 2 SIMT kernels. The 150 rows of `e16_query_tile` give 1 split by the base's formula.
- **VI-P (E11), nvcc:** the same three diagnostics at the head as at the base: one `static_assert` each with the rule's message, 16, 14
  and 14 lines (24, 27 and 154 lines without the checks, at c8fcbe8). The logs differ only in `executors.h` line numbers.
- **VI-L (E3), nvcc:** variant A keeps at most 17 registers, with no spills and no stack, up to N = 4,000. Variant B compiles at N = 3,847, and at
  3,848 ptxas stops with "uses too much parameter space (0x8000 bytes, 0x7ffc max)", as in `../e3-compile/`.
- **`npp/e9_cross_library.cu`:** unchanged by the pin move; it compiles at the head with its two SHORT LISTING blocks, and its kernels are identical.
- **Not checked here:** GCC 13 and Clang 23 for VI-P, and the Clang limits of VI-L (stack, `-fbracket-depth`). This machine has g++ 15 and
  clang 21 only, and no amdclang++ (section 5).

## 1. The ports (pin-fkl-435, f3d66da to 2518c75)

| commit | what | files |
|---|---|---|
| f3d66da | **E18 and E19:** 37 filter calls in 5 programs. `fk::executeLinearFilter(...)` becomes `fk::executeOperations<fk::LinearFilterDPP<fk::ParArch::GPU_NVIDIA, D>>(stream, details, ...)`, and `fk::executeBoxBlur<1>(...)` becomes `fk::executeOperations<fk::MorphologyDPP<fk::ParArch::GPU_NVIDIA, D>>(stream, details, input, output, fk::boxBlurReducers<float, 1>(details))`. The `bool` checks around the calls go, and each call stays in its row's timed lambda. The comments that named the removed functions are updated. | `opencv-cuda/stencil/e18_box.cu`, `e18_box_split.cu`, `e18_stencil.cu`, `e19_batched_box.cu`, `e19_crop_sweep.cu` (+118 / −131) |
| b51e53f | **E16 (QA3):** each automatic variant asks `fk::flashAttentionMmaTiles<D, K, V>(bh, seq, causal)`, names the `FlashAttentionMmaDPP` of those tiles in place, asks `fk::flashAttentionMmaSplits<MmaDPP>(...)`, and launches the MMA DPP (1 split) or `FlashAttentionMmaSplitKVDPP` and its `CombineDPP`, with no helper. Tiles that the choice cannot return throw `std::logic_error`. Explicit tiles stay explicit (`executeOperations<FlashAttentionMmaDPP<..., BQ, BKV, NW>>`). The attention DPPs get their leading `ParArch`. `parseAttentionKernel` reads the new names and positions, and the CSV notes name the new calls. The nodq copy of `flash_attention_mma.h` is re-made from 48fbf3f with the same patch. | `fkl-self/attention/e16_attention.cu` (+766), `e16_query_tile.cu` (+175), `nodq/` (+1,238 / −248 in all) |
| 8165270 | **One-line renames** of the `launchTransformDPP_Kernel<...><<<...>>>` launch to `fk::launchDPP_Kernel<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED, D, true>><<<...>>>`. Two of the programs are frozen (Q2: edited, compiled and compared, never run). | `fkl-self/numerics/e6_fused_vs_unfused.cu:111`, `opencv-cuda/benchmark_image_resolution_MAD_loop.cu:67`, `opencv-cuda/parity/opencv_parity_probe.cu:97` |
| 2518c75 | **Scripts** that match kernel names accept the new names too (mangled `16launchDPP_KernelINS_12TransformDPPI...` or demangled `launchDPP_Kernel<fk::TransformDPP<`); `e16_sass_counts.py` reads the new attention DPP names and skips the combine kernel. | `scripts/e1_sass.py`, `e3_clang_limits.py`, `e9_analyse.py`, `e9_sass_hash.py`, `isa_pipeline_kernels.py`, `e16_sass_counts.py`, `triton/e2_kernel_facts.py` |

- **A GCC finding (f3d66da).** `e18_box.cu` takes the address of the filter kernel for `cudaFuncGetAttributes`. With g++ 15.2.0 as the host compiler,
  `auto k = fk::launchDPP_Kernel<DPP, D, IOps...>` (a non-empty pack) does not compile: "no matches converting function". The form
  `reinterpret_cast<const void*>(&fk::launchDPP_Kernel<...>)` compiles, and so does every form with a clang host. `e18_box.cu` uses the `void*` form.
- **Left as they are:**
  - `fkl-self/compiletime/README.md:26` names `launchTransformDPP_Kernel` in prose about the history of E3;
  - the comments of `e19_batched_box.cu` (lines 79 and 422) and `e19_crop_sweep.cu` (95 and 471) cite `morphology.h:179-208` and `189-208`,
    which are b10fc66's lines. At 48fbf3f `boxBlurReducers` is at lines 182 to 216.
- **The scripts used to make the ports** are in `scripts-used/` (`port_e18_e19.py`, `port_e18_e19_comments.py`, `port_e16.py`). The
  comments were then wrapped by hand.

## 2. Device code (`sass-identity/`)

`scripts-used/pr435_sass_identity.py` is the script of `../fkl-pr435-check/` (sha256 c9cedd90..., checked by the build script). It reads every
`*_cu` executable of both builds with cuobjdump 13.4.3 and pairs the kernels by their demangled names, A's rewritten by FKL #435's rename rule:
- `launchTransformDPP_Kernel<PA, TF, TD, D, IOps...>` becomes `launchDPP_Kernel<TransformDPP<PA, TF, D, TD, void>, D, IOps...>`;
- the linear, median and morphology kernels become `launchDPP_Kernel<LinearFilterDPP<PA, D>, D, ...>` (and so on), the Quad kernels `<DPP, XQuadDetails, ...>`;
- the attention kernels get `(fk::ParArch)1` as their first argument; the Mma kernel with SPLIT = true becomes `FlashAttentionMmaSplitKVDPP`,
  and `flashFwdSplitCombineKernel` becomes `FlashAttentionSplitCombineDPP<..., AttentionIdentityEpilogue>`.

It compares the SASS text (with addresses and the anonymous-namespace module ids masked), registers, stack, shared and local memory, LDL / STL and
the size of constant bank 0.

| | base (b10fc66) | head (48fbf3f) |
|---|---|---|
| programs (`*_cu`) | 39 | 39 |
| kernels | 1,834 | 1,818 |
| paired | 1,816 (1,665 through the rename rule, 147 whose names differ in the module id only, 4 with the same name) | |
| identical | 1,776 | |
| different | 40 | |
| in one build only | 18 | 2 |

The 40 that differ (`differences-b10fc66-vs-48fbf3f.txt` lists them by DPP):
- **19 LinearFilterDPP** (e18_box 8, e18_box_split 5, e18_stencil 4, e19_batched_box 1, e19_crop_sweep 1: the `fkl_loop` row);
- **17 MorphologyDPP** (e19_crop_sweep 16: the `fkl_batch` row at each N; e19_batched_box 1).
  - The 16 of the crop sweep keep 544 instructions and 40 registers, except N = 1: 568 → 544 instructions, 35 → 40 registers, stack 40 → 0 B, LDL 2 → 0, STL 5 → 0.
  - FKL #435 passes the details of these DPPs as a `__grid_constant__` parameter; `../fkl-pr435-check/` shows the same change in its harness.
- **4 FlashAttentionSplitCombineDPP** (e16_attention 2, e16_query_tile 2): constant bank 0 grows from 376 to 384 B, and 2 of them lose 8 instructions.
  These are the combine kernels of the split-KV path; FKL #435 changes them (`../fkl-pr435-check/`). No measured configuration splits.

In one build only:
- **18 in the base only:** `e16_query_tile`'s split-KV Mma kernels (SPLIT = true) of its explicit tiles. At b10fc66 `executeFlashAttentionMma<D, BQ, BKV, NW>`
  could split explicit tiles. At 48fbf3f the executor of an explicit `FlashAttentionMmaDPP` never splits, so they are not instantiated.
- **2 in the head only:** `e16_attention`'s combine kernels with the epilogue (`FlashAttentionSplitCombineDPP<..., epilogue>`), instantiated by
  the split branch of its epilogue variant.

None of the 22 runs at a measured configuration, since every measured row has 1 split (`e16-choice-check.txt`).

The binaries (`binary-sha256-*.txt`, 44 per build): one pair is byte-identical, `batch_build_compilation_time_cpp`. The others differ at
least in the FKL commit string they write and in their kernel names.

The builds (`identity-builds/`) used `scripts/check_fkl_tree.sh --only bench` of each checkout. They had the FKLB options, OpenCV and nvcc of the reference build `build/local-rtx4090`,
which was only read, and ran at the same time on separate cores. Each built 44 of 44 targets: the base in 3,442 s and the head in 3,473 s
(`report.txt`, `bench-report.csv`; `console.txt` is the script's output).

## 3. Host code of the timed calls (`host-code/`)

### 3.1 Method

`scripts-used/host_compare435.py` compares two binaries function by function, with `objdump -d -C`, and reads them only. It
starts from the normalisation of `../fkl-pr433-pin-check/scripts-used/host_compare.py`:
- addresses dropped and padding removed;
- jump targets inside a function as instruction indices, and calls to another function as that function's key;
- RIP displacements and nvcc's module ids masked.

On top of that it applies AMD's two normalisations (#67), so that both GPUs' records compare alike:
- the **kernel host stubs**, and the static locals and guard variables of their `__device_stub__` (the kernel handles), are keyed by the
  kernel instantiation, A's through the rename rule;
- the **`__LINE__` argument** of `fklb::statusMessage` (`%r8d`) is masked, and on this side also that of `fk::gpuAssert` (`%edx`, FKL's `gpuErrchk`).

More, given on the command line (`host_runs.sh`):
- `--line-pairs`: the lines that moved (`moved-lines/`, from `moved_lines.py`, which pairs the lines of the error checks of two versions of a
  file). An immediate that is a moved line, also as the digits that an inlined `std::to_string` writes, counts as `__LINE__` only:
  - base against head: `executors.h` for every program, and the program's source and its DPP headers for E16 and E19;
  - the measured binary against main: FKL 64ad9195 → b10fc66, and `common/cuda_timing.h` for 836a98e.
- measured against main only:
  - no rename rule (the same FKL kernel names);
  - `--drop-binarytype-default`: FKL #432's defaulted 4th argument `fk::BinaryType` of `fk::Sub`, `fk::Mul` and `fk::Div` is dropped from the names;
  - `--path-delta 5` (or 7): the source path of the base build is 5 (or 7) characters longer than the measured build's, and an immediate larger
    by exactly that much is the length of a `__FILE__` string.
- `--intervals`: the **timed intervals**. These are the instructions between consecutive calls of `cudaEventRecord` (*event*: the GPU times of the
  paper) or of `steady_clock::now` (*clock*: host times, not in the paper) in one function, and the functions those instructions call.
  - Where GCC inlined the row's lambda into the timing function, its body is inside the interval. Otherwise the lambda is a called function.
  - Each is identical, `__LINE__` / path only, registers (the same instructions under one consistent renaming of the registers), or different.
  - Indirect calls (a `std::function`) are not followed: E19's rows are its call lambdas, and `--row` covers them.
- `--row NAME=REGEX` (E16, E19): a timed lambda belongs to the row whose expression matches what it reaches. The report lists, per row, the
  functions its lambdas reach that differ, and those the harness reaches.

**The binaries:**
- **main against head:** the two identity builds (section 2).
- **measured against main:** the binary that made each record against the base build. Each sha256 is checked against the one the record names.
  - `worktrees/fkl-benchmarks-window` at 836a98e: E4 (VI-K), NPP and the CPU-time programs;
  - `fkl-benchmarks-window-c`, `build/rtx4090-e1` at 1a2535c: E1;
  - `fkl-benchmarks-window-d` at f27c605: E16;
  - `fkl-benchmarks-window-e` at 9a6a88b: E2;
  - `fkl-benchmarks-window-g` at 469aa6d: E4 whole call;
  - `fkl-benchmarks-window-h` at f534a38: E4 build once.
- **E19's measured binary (246b1c8) no longer exists.** #60's comparison stands in for it (section 3.6).
- **The host compiler:** every one of these binaries says it was built by GCC (Ubuntu 15.2.0-16ubuntu1) 15.2.0 (`.comment`).

`summary.txt` has the totals of all 21 comparisons. Each comparison has these files:
- `<program>.txt`: the report, with the totals, every function that differs (with its first differing instruction and the timed calls that reach it), the rows and the intervals;
- `<program>.csv`: every function, with its result;
- `<program>-intervals.csv`: every timed interval, with its callees that are not identical and the external calls;
- `<program>-diffs.txt[.gz]`: the unified diff of every function that differs.

`console.txt` is the output of `host_runs.sh` (2026-11-26 17:20:46 to 17:21:39).

### 3.2 main against the head: what the pin move changes

| program | functions base / head | identical | only `__LINE__` | other | in one only | event intervals |
|---|---|---|---|---|---|---|
| `benchmark_CPU_OpenCV_vs_cvGS_cu` | 129 / 129 | 129 | 0 | 0 | 0 / 0 | (clock only: 16 identical) |
| `benchmark_CPU_NPP_vs_FastNPP_cu` | 202 / 202 | 186 | 16 | 0 | 0 / 0 | (clock only: 16 identical) |
| `benchmark_npp_batchresize_x_split3D_cu` | 326 / 326 | 296 | 30 | 0 | 0 / 0 | 30 identical |
| `e1_vf_chain_cu` | 203 / 203 | 185 | 18 | 0 | 0 / 0 | 36 identical |
| `e1_hf_batch_cu` | 220 / 220 | 194 | 26 | 0 | 0 / 0 | 38 identical, 2 `__LINE__` |
| `e1_preproc_cu` | 229 / 229 | 214 | 15 | 0 | 0 / 0 | 40 identical |
| `e2_fkl_chain_cu` | 108 / 108 | 84 | 24 | 0 | 0 / 0 | 18 identical |
| `e4_preprocessing_cu` | 1,834 / 1,834 | 1,598 | 236 | 0 | 0 / 0 | 364 identical (116 call functions that differ in `__LINE__` only) |
| `e16_attention_cu` | 336 / 346 | 202 | 0 | 87 | 47 / 57 | 38, all different (the port) |
| `e19_crop_sweep_cu` | 698 / 697 | 637 | 0 | 59 | 2 / 1 | 17 identical (the rows are behind `std::function`: below) |

- **E1, E2, E4, NPP and the CPU-time programs:** everything that differs is the line of an error check in FKL's `executors.h` (469 → 486 and
  others, `moved-lines/lp-unchanged.txt`). So the pin move leaves their host code as it was.
- **E19** (`e19_crop_sweep_cu.txt`, "rows"):
  - `opencv_cuda`: identical, everything its lambdas reach;
  - `npp`: one function differs, `std::operator+` (a libstdc++ template, 138 → 111 instructions). The npp lambdas reach it only through
    `fklb::statusMessage`, the message of a failed status check; so does the harness;
  - `fkl_batch` (and `fkl_loop`, which is not in the paper): the ported call lambdas (`{lambda()#12}` of `launchWithCrops<N>` for `fkl_batch`
    and `{lambda()#6}` for `fkl_loop`, 16 each), `fk::gpuAssert` (an error path) and `std::operator+`, and one function new in the head,
    `fk::executor_detail::requireArguments(bool, const char*, const char*)`.
    - At b10fc66 `executeBoxBlur` checked the details inline (`if (!DPPDetails::valid(details)) return false;`).
    - At 48fbf3f the executor calls `requireArguments(DPPDetails::valid(details), ...)` (`morphology.h:855`), out of line, on every call. It throws
      when the details are invalid.
    - These are the changes AMD's record finds on the HIP side.
  - outside the timed calls: `envIs1` (the environment switches), `CsvWriter::row` and 5 `std::function` handlers of the download lambdas
    (`void(grey, box, tensor)`), which copy the outputs back for the checks after the timing.
- **E16:** section 3.5.

### 3.3 The measured binary against main: what changed since the measurement

| record (commit) | program | functions measured / main | identical | only `__LINE__` / path | other | event intervals (interval / its callees) |
|---|---|---|---|---|---|---|
| cleanup-window-a (836a98e) | `benchmark_CPU_OpenCV_vs_cvGS_cu` | 129 / 129 | 128 | 0 | 1 | (clock only: 16 identical) |
| cleanup-window-a (836a98e) | `benchmark_CPU_NPP_vs_FastNPP_cu` | 202 / 202 | 171 | 30 | 1 | (clock only: 3 identical, 13 identical / `__LINE__`) |
| cleanup-window-a (836a98e) | `benchmark_npp_batchresize_x_split3D_cu` | 326 / 326 | 279 | 46 | 1 | 15 identical, 15 identical / `__LINE__` |
| cleanup-window-c e1 (1a2535c) | `e1_vf_chain_cu` | 203 / 203 | 184 | 18 | 1 | 36 identical |
| cleanup-window-c e1 (1a2535c) | `e1_hf_batch_cu` | 220 / 220 | 198 | 21 | 1 | 38 identical, 2 `__LINE__` |
| cleanup-window-c e1 (1a2535c) | `e1_preproc_cu` | 229 / 229 | 207 | 21 | 1 | 40 identical |
| cleanup-window-e (9a6a88b) | `e2_fkl_chain_cu` | 108 / 108 | 87 | 19 | 2 | 18 identical |
| cleanup-window-d (f27c605) | `e16_attention_cu` | 336 / 336 | 332 | 3 | 1 | 37 identical |
| e4-build-once (f534a38) | `e4_preprocessing_cu` | 1,834 / 1,834 | 1,597 | 236 | 1 | 248 identical, 116 identical / `__LINE__` |
| cleanup-window-b (836a98e) | `e4_preprocessing_cu` | 571 / 1,834 | 338 | 0 | 109 | section 3.4 |
| e4-hand-roi (469aa6d) | `e4_preprocessing_cu` | 1,258 / 1,834 | 841 | 1 | 416 | section 3.4 |

- **The 1 "other" of each program is `fklb::writeSystemInfoHeader`.** It prints the generated build information
  (`build/.../generated/fklb_build_info.h`). The base build was made in `check_fkl_tree.sh`'s link farm, so it writes
  `FKLB_COMMIT_REPO "unknown"` where the measured builds write their commit. The E1 build `rtx4090-e1` was configured with
  `FKLB_LINEINFO off` and without OpenCV ("none"); the base build has `on` and 4.14.0. It runs once, before any timing.
- **E2's second is `launch()`,** the program's driver, where one string is one character longer: the CSV note says "256 MiB" where it said
  "256 MB" (7cef26c). It is outside the timed calls.
- **So for E1, E2, E16, NPP, the CPU-time programs and E4's build-once record, the code the timed calls run at main is the measured code,**
  up to `__LINE__` values and the length of the source path. FKL moved from 64ad9195 to b10fc66 at main and changed none of it.

### 3.4 E4 by row (`e4-rows.txt`)

E4 times each row with `fklb::timeIt<L>` (`common/cuda_timing.h`), L being the row's launch lambda in `runCase<N>`. GCC numbers the lambdas of
`runCase` in source order, and the numbers agree at all three commits:
- **#1 to #4** at 836a98e: `fkl_fused` (also `_late` and `_after`), `fkl_fused_half_pixel`, `opencv_percrop`, `npp`;
- **#5 to #8**, added between 836a98e and 469aa6d: `hand_batch`, `hand_padded`, `hand_roi`, `cvgs_fused`;
- **#9 to #12**, added at main by #37: the build-once rows.

The lambdas' source is the same at main (`source-history.txt`; the E4 changes since 469aa6d add the build-once rows, and `measure()` returns the
checked output).

| row (lambda) | 836a98e against main | 469aa6d against main |
|---|---|---|
| `fkl_fused` (#1), `fkl_fused_half_pixel` (#2) | interval identical at all 31 N; the lambda differs | the same |
| `opencv_percrop` (#3) | identical at 30 N; at N = 1 the interval differs (inlined at main, called in the measured binary) | identical at 30 N; registers only at N = 1 |
| `npp` (#4) | interval identical; the callees that differ are on error paths (`statusMessage`, `to_string`) | the same |
| `hand_batch` (#5) | (not yet in the program) | identical, with its callees, at all 30 N |
| `hand_padded` (#6) | | identical; at 14 N the error-path `statusMessage` differs |
| `hand_roi` (#7) | | interval registers only at 18 N and different at N = 10; at 10 N its lambda and the kernel's host stub differ |
| `cvgs_fused` (#8) | | interval identical; the lambda differs |

**What differs in the lambdas of `fkl_fused`, `fkl_fused_half_pixel` and `cvgs_fused`:**
- **The host stubs of the FKL kernels.** At main they are separate functions (`_ZL<len>__device_stub__...launchTransformDPP_Kernel...`, 180 of them,
  one per kernel instantiation). In both measured binaries there are none: GCC inlined them into the lambdas.
  - The f534a38 build has the same 180 out-of-line stubs. f534a38 has main's E4 source with FKL 64ad9195.
  - So the stubs come with the E4 source of #37, which added the build-once rows, and not with the library.
- **FKL's crop checks**, `fk::Ptr2D::crop2D` and `fk::ptr_crop::checkAxis`, which these rows run for every crop. They are compiled
  differently, though their source did not change.
- **The error paths**, `fk::gpuAssert` and `std::to_string`.

**The measured whole-call rows of `cleanup-window-b` (VI-K) and `e4-hand-roi` (Figure 15) therefore ran host code that main no longer
produces.** The source is the same, but the code GCC makes of it differs, in the ways above. The rows timed again in `e4-build-once`
(f534a38) ran main's code. No cause of the compiler's choices is claimed, and their effect on a time is not measured here.

The *clock* intervals of `measure()` (`host_enqueue_us`, a host time not in the paper) do not pair at 836a98e, where `measure()` returned
nothing, and they differ at 469aa6d.

### 3.5 E16: the port

**Base against head.** Every timed call of E16 changes, by design. At b10fc66 a row called `executeFlashAttentionMma<D>(...)`, which chose
the tiles, decided the split and launched. At 48fbf3f the program makes those calls itself (section 1). The comparison
(`e16_attention_cu.txt`, "rows") finds the ported lambdas and FKL's executors different in every row.

**The kernels are the measured ones.** `e16_choice_check.py` reads the record's CSVs. At each of the 420 MMA rows
(`cleanup-window-d/cleaned/e16-attention`):
- `flashAttentionMmaTiles` returns the measured tiles, with the same conditions as b10fc66's choice;
- the split formula, from the measured kernel's blocks per SM × 128 SMs, gives 1 split.

That makes 12 MMA kernels, plus the 2 SIMT kernels, which have one launch at both commits. The 150 rows of `e16_query_tile` give 1 split by the base's formula (`e16-choice-check.txt`).

**One host call more per call.** Six of the 12 measured MMA kernels need more than 48,000 B of dynamic shared memory: all four at D = 128
(raw 64×64×4, with and without the epilogue; int8 and e4m3 K/V 64×32×4), and the two at D = 64 with 64×128×4 tiles (raw, with and without
the epilogue). Each needs 49,152 B; the other six need 24,576 B. These sizes come from `SMEM_BYTES` in `flash_attention_mma.h` at 48fbf3f,
lines 361 to 413, with `FK_FA_DUAL_Q` and `FK_ENABLE_FP8_QK` at their default 0.

For these kernels:
- b10fc66's `executeFlashAttentionMma` calls `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` once per call (lines 1966-1969);
- at 48fbf3f, `attention_mma_detail::setDynamicSharedMemory<MmaDPP>()` (lines 1939-1947) is called twice per call: by
  `flashAttentionMmaSplits` (line 2040) and by the executor of `FlashAttentionMmaDPP` (line 2112).

That is one `cudaFuncSetAttribute` more in each launch of E16's timed blocks (event `a`, the block of launches, event `b`). The occupancy query
of the split decision is made once per kernel type at both commits (a static local).

**The measured binary against main** (f27c605): all 37 timed intervals and everything they reach are identical, up to `__LINE__` and the path.

### 3.6 E19 against its measured commit

E19's crop sweep was measured at 246b1c8, and that build no longer exists. Between 246b1c8 and main, E19's source and the headers it includes
changed in one commit only, 2160c70 (#60: the pin to b10fc66 and the one-line details port; `source-history.txt`).

#60 compared the host code of `e19_crop_sweep_cu` built at a8f9daa with FKL 24478f0 against 2160c70 with FKL b10fc66. a8f9daa has the source and FKL of 246b1c8, and 2160c70 has main's.
That record (`../fkl-pr433-pin-check/e19-host-code/`) found no function inside the timed calls of `fkl_batch`, `opencv_cuda` and `npp` different.

## 4. Paper facts at the head (`paper-facts/`)

### 4.1 VI-P: the three mistakes (E11), nvcc 13.4.92

`scripts/e11_diagnostics.py --compilers nvcc --ccbin /usr/bin/g++ --label <tree> --include <tree>/include` was run with the head checkout's
script, 5 repetitions, on both trees (`e11/e11_diagnostics.csv`, `e11/logs/`). Compile only: the three programs were not run.

| mistake (`fkl-self/diagnostics/`) | base b10fc66 | head 48fbf3f | without the checks (c8fcbe8, `../e11-diagnostics/nvcc-rows/`) |
|---|---|---|---|
| type mismatch | `static_assert`, 16 lines, 1 error | the same | error, 24 lines |
| complete read in the middle | `static_assert`, 14 lines, 1 error | the same | error, 27 lines |
| batch of 50 read, 60 written | `static_assert`, 14 lines, 1 error | the same | compiles, 154 lines of output |

- **The messages are the rule's:**
  - "FKL: the OutputType of the previous Op is not convertible to the InputType of the next Op";
  - "Complete Read Operations as continuations are not allowed. It has to be an IncompleteReadBackOperation.";
  - "Fusing two batch operations of different BATCH size is not allowed."

  They come from `chain_check.h` (lines 115, 143 and 195), and each control (the correct chain) compiles.
- **The logs of the two trees differ only in the line numbers of `executors.h`** in the context lines (477 → 494 and others).
- **The analysis columns are those of the prototype rows** of `../e11-diagnostics/nvcc-rows/`.
- **The host compiler is g++ 15.2.0;** GCC 13 and Clang 23 are not on this machine (section 5).

### 4.2 VI-L: the E3 limits (nvcc 13.4.92)

The head checkout's `scripts/e3_sweep.py` was run through `scripts-used/e3_sweep_nogpu.py` (`scripts-used/e3_runs.sh`, 2026-11-26 17:20:14 to 17:39:00):
- one compile at a time per variant, variant A on core 8 and B on core 9, one repetition;
- with the sweep's flags (`-O3 -DNDEBUG -std=c++17`, sm_89 with PTX, `-ftemplate-depth=1000`, `-lineinfo`, `--threads 1`).

The wrapper exists because the sweep's CSV header (`common/fklb_csv.py`, `system_info_lines`) imports torch and asks the GPU and nvidia-smi
for their properties. The wrapper writes the same header without them ("gpu: not queried"). The compiles and the parsing of their output are the
sweep's own. The compile times are not a result of this check, and the timings were not controlled.

| variant | N | result | registers | spills | stack | peak memory of the compile |
|---|---|---|---|---|---|---|
| A (float) | 1 | ok | 10 | 0 | 0 B | 297 MiB |
| A | 100, 500, 1,000, 1,500, 2,000, 3,000 | ok | 17 | 0 | 0 B | 350 MiB to 7.9 GiB |
| A | 4,000 | ok | 17 | 0 | 0 B | 13.2 GiB (13,474 MiB) |
| B (float3 / int3) | 3,847 | ok | 19 | 0 | 0 B | 12.4 GiB (12,671 MiB) |
| B | 3,848 | **ptxas error** (exit 255) | | | | 12.4 GiB (12,677 MiB) |

- **Variant A: at most 17 registers, with no spills and no stack (local memory),** at every N from 1 to 4,000. That is the paper's fact, at 48fbf3f.
- **Variant B: N = 3,847 compiles and N = 3,848 does not.** ptxas stops at the fused kernel's parameter block:

  ```
  ptxas .../tmpxft_..._current_B_3848.ptx, line 43891; error   : Entry function '_ZN2fk16launchDPP_KernelINS_12TransformDPPILNS_7ParArchE1ELNS_2TFE0ENS_20TransformDPPDetails_IvLb0EJNS_ ... _S14_EEEvT0_DpT1_' uses too much parameter space (0x8000 bytes, 0x7ffc max).
  ptxas fatal   : Ptx assembly aborted due to errors
  ```

  - 0x8000 is 32,768 B and the maximum 0x7ffc is 32,764 B. `../e3-compile/` found the same limit at its tree: N = 3,847 compiles and
    N = 3,848 fails in ptxas. Its layout model gives 32,768 B at N = 3,848.
  - The kernel is now named `launchDPP_Kernel<fk::TransformDPP<...>, ...>` (FKL #435). Its parameters are the same.
- **The full message.** The sweep's CSV keeps a shortened first line (`B/e3_compile.csv`, column `why`). So the N = 3,848 translation unit of
  the sweep was compiled once more with the same nvcc command, and its stderr was kept whole: `ptxas-B-3848/nvcc-stderr.txt`, 24,844 B,
  most of it the mangled kernel name. That ran from 17:39:32 to 17:46:28 (`ptxas-B-3848/run.sh`, `console.txt`).
  - The "nvcc exit 0" line of that `console.txt` is wrong: the script reads `$?` after a `$(date)` in the same `echo`, so it prints `date`'s
    status.
  - nvcc failed: the sweep recorded exit 255 for the same command, and no executable was written.
- The CSVs are `A/e3_compile.csv` and `B/e3_compile.csv`, with the sweep's own logs `sweep.log`. `console.txt` is the output of `e3_runs.sh`.

### 4.3 `npp/e9_cross_library.cu`

- **The pin move does not change the file** (`git diff 976e42b 2518c75 -- npp/e9_cross_library.cu` is empty).
- **Its two SHORT LISTING blocks (lines 65-75, the user operation; 302-315, the call) are compiled code**, under no `#if`.
- **`e9_cross_library_cu` compiles at the head** (`identity-builds/head-48fbf3f/bench-report.csv`: OK), and all its kernels are identical to the base's (section 2).

## 5. Not checked here

- **VI-P with GCC 13 and Clang 23:** only g++ 15.2.0 and clang 21 are installed. The nvcc rows above use g++ 15.2.0 as the host compiler.
- **VI-L's Clang facts** (the stack threshold of about 1,370 and `-fbracket-depth` from N = 2,047) are amdclang++'s and clang++-23's. Neither is
  installed here. AMD's record (#67) finds them at 48fbf3f: the 8 MiB stack first fails at N = 1,368, and the fold limit stops N = 2,047 until `-fbracket-depth=2049`.
- **No time** was measured. The effect of the host-code differences of sections 3.4 and 3.5 on a time is unknown.

## Files

- `README.md`: this file.
- `source-history.txt`: per measured program, its files at main (the source and the repository headers it includes), the gitlinks at the
  measured commit and at main, and `git diff --stat` and the commits between them (`scripts-used/source_history.py`).
- `e16-choice-check.txt`: section 3.5 (`scripts-used/e16_choice_check.py` on the CSVs of `../cleanup-window-d/cleaned/e16-attention/` and of
  `e16_query_tile`).
- `identity-builds/{base-b10fc66,head-48fbf3f}/`: the report, target lists, configure log and build time of `check_fkl_tree.sh --only bench`;
  `identity-builds/console.txt`: the output of `scripts-used/identity_builds.sh` (2026-11-26 15:35:00 to 16:41:23).
- `sass-identity/`:
  - `sass-identity-b10fc66-vs-48fbf3f.txt` and `.csv.gz` (every kernel of both builds, with its result);
  - `differences-b10fc66-vs-48fbf3f.txt`;
  - `binary-sha256-{base-b10fc66,head-48fbf3f}.txt`.
- `host-code/`:
  - `summary.txt`, `e4-rows.txt` and `console.txt`;
  - `moved-lines/`, the line pairs given to `--line-pairs`;
  - `main-vs-head/` and `measured-vs-main/`, the reports of section 3.1, one set per program or record.
- `paper-facts/e11/` (section 4.1) and `paper-facts/e3/` (section 4.2; `ptxas-B-3848/` has the full message).
- `scripts-used/`:
  - `identity_builds.sh`, `pr435_sass_identity.py` and `diff_summary.py`: section 2;
  - `host_runs.sh` (both host comparisons), `host_compare435.py`, `moved_lines.py`, `host_summary.py`, `e4_rows.py` and `callers.py` (who calls a function, used for sections 3.2 and 3.4): section 3;
  - `source_history.py` and `e16_choice_check.py`;
  - `e3_runs.sh` and `e3_sweep_nogpu.py`: section 4.2;
  - `port_e18_e19.py`, `port_e18_e19_comments.py` and `port_e16.py`: section 1.

Every file is complete. The files over 1 MiB are gzipped (`gzip -n -9`). Read them with `zcat`.
