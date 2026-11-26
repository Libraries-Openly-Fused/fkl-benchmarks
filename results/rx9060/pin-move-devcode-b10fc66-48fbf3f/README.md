# Device and host code of the HIP programs at FKL b10fc66 and at 48fbf3f (FKL #435, task C)

FKL #435 (branch `LTS-C++17-executeoperations-dpps`, head 48fbf3f08e7551c3eb70ac2302d8ebc15ecad645, on LTS-C++17 b10fc66) makes every
DPP run through `fk::executeOperations<DPP>`, with one kernel, `launchDPP_Kernel<DPP, Details, IOps...>`. Invalid arguments throw
`std::invalid_argument` instead of returning `false`. Its HIP and CUDA checks are `../fkl-pr435-check/` (#64) and
`../../rtx4090/fkl-pr435-check/` (#65). On 2026-11-26 it was open and ready for review, not merged.

fkl-benchmarks #66 (branch `pin-fkl-435`) moves `external/fkl` to it. This record covers its first two commits, on main 976e42b:
- ede9e5e: `external/fkl` from b10fc66 to 48fbf3f;
- a4b493c: the three HIP programs that call FKL's filter entry points, `rpp/e18_box.h`, `rpp/e19_batched_box.h` and `rpp/e19_crop_sweep.h`.
  - `fk::executeLinearFilter(details, ..., stream)` becomes `fk::executeOperations<fk::LinearFilterDPP<fk::ParArch::GPU_AMD, D>>(stream, details, ...)`.
  - `fk::executeBoxBlur<1>(details, input, output, stream)` becomes
    `fk::executeOperations<fk::MorphologyDPP<fk::ParArch::GPU_AMD, D>>(stream, details, input, output, fk::boxBlurReducers<float, 1>(details))`.
  - The `bool` checks around the calls go. Each call stays in place, in its row's timed lambda.

The NVIDIA session then added the CUDA ports to the same branch (f3d66da to 2518c75). They change `.cu` files, `scripts/` and one header of the E16 CUDA
variant `fkl-self/attention/nodq/`. B's ninja dependency log lists no file of `fkl-self/attention/`, so the sources of every HIP program are
the same at 2518c75 as at a4b493c. The CUDA side is not part of this record.

**The check.** Every HIP program of this repository was built twice, with the same tools and flags:
- A: 976e42b with FKL b10fc66, the pin on main;
- B: a4b493c with FKL 48fbf3f.

Three comparisons were made:
- **(a)** their gfx1200 device code, function by function;
- **(b)** the x86-64 host code of the four programs behind RX 9060 paper rows, with what each row's timed call reaches;
- **(c)** three compile-only facts of the paper at FKL 48fbf3f: E11 (VI-P), E3 (VI-L) and the number of HIP tests (VI-N).

Nothing was run on the GPU. The frozen program `benchmark_latency_hiding_hip` was compiled and compared only. This was done on 2026-11-26
on the RX 9060 machine (`../README.md`).

Decision: the coordinator's assignment C-PM-HIP of 2026-11-26, the HIP side of the pin move to FKL #435. It follows Oscar's task C decisions
in the coordinator session (Q1 to Q17 and QA1 to QA3, 2026-11-25 and 2026-11-26).

## Result

- **Device code: identical.** All 832 functions of the 10 HIP programs are identical, and 691 of them are paired through the task C rename.
  This includes the E18 and E19 filter kernels and every kernel behind a paper row.
- **Host code of the timed calls:**
  - **identical** for every row of `benchmark_rpp_preprocessing_hip` (VI-K), `benchmark_vfhf_max_speedup_hip` and `benchmark_latency_hiding_hip`.
    All their host functions are identical.
  - **identical** for the rpp row of `e19_crop_sweep_hip` (VI-M) and for its timing harness.
  - **different** for the fkl_batch row of `e19_crop_sweep_hip` (VI-M), and for its fkl_loop row, which is not in the paper. What differs is
    the ported call: the row's lambda, FKL's entry function, and `fk::executor_detail::requireArguments`, which is new.
- **E11 (VI-P):** the same three diagnostics at 48fbf3f as at b10fc66. The logs differ in one line number.
- **E3 (VI-L):** the same limits at 48fbf3f as in `../e3-compile-limits/` at c8fcbe8. The 8 MiB stack of the compiler first fails at N = 1,368,
  and with 16 MiB the fold nesting limit stops N = 2,047; `-fbracket-depth=2049` (N + 2) lifts it.
- **VI-N:** 52 HIP tests at 48fbf3f and at b10fc66.

## 1. Device code: 832 of 832 functions identical

| HIP program | functions in A / B | renamed | identical | different | in one build only |
|---|---|---|---|---|---|
| `benchmark_horizontal_fusion_hip` | 125 / 125 | 123 | 125 | 0 | 0 |
| `benchmark_thread_fusion_hip` | 153 / 153 | 151 | 153 | 0 | 0 |
| `benchmark_latency_hiding_hip` (frozen: compiled only) | 63 / 63 | 61 | 63 | 0 | 0 |
| `benchmark_numinstructions_x_operation_hip` | 187 / 187 | 185 | 187 | 0 | 0 |
| `benchmark_rpp_preprocessing_hip` | 247 / 247 | 125 | 247 | 0 | 0 |
| `benchmark_vfhf_max_speedup_hip` (19 code objects, paired by position) | 26 / 26 | 24 | 26 | 0 | 0 |
| `e18_box_hip` | 6 / 6 | 3 | 6 | 0 | 0 |
| `e19_batched_box_hip` | 5 / 5 | 2 | 5 | 0 | 0 |
| `e19_crop_sweep_hip` | 20 / 20 | 17 | 20 | 0 | 0 |
| `batch_build_compilation_time_hip` | no device code in either build | | | | |
| **10 programs** | **832 / 832** | **691** | **832** | **0** | **0** |

- **Renamed** means that the name in A is rewritten by the task C rule of `step0/devcode_compare_taskc.py` before the pairing. Every kernel of
  FKL's executors is now one template, for example:
  - `launchTransformDPP_Kernel<PA, TF, TD, Det, IOps...>` becomes `launchDPP_Kernel<fk::TransformDPP<PA, TF, Det, TD, void>, Det, IOps...>`;
  - `linearFilterDPPKernel<PA, D, ...>` becomes `launchDPP_Kernel<fk::LinearFilterDPP<PA, D>, D, ...>`;
  - the morphology, median and Quad kernels follow the same pattern.

  The kernels that keep their names are RPP's, FastRPP's and the hand-written ones. The script also keeps the rule of
  `scripts/devcode_compare_renamed.py` (#45). After the rename there are no key collisions, and no function is in one build only.
- **With the branch targets as separate symbols** (`--labels-as-symbols`), 2,610 of 2,610 symbols are identical.
- **The filter kernels of E18 and E19** (LinearFilterDPP and MorphologyDPP) are identical too. FKL #435 adds `__grid_constant__` to their
  parameter (Q3). With nvcc this changes their SASS (#65); with amdclang++ the code is the same.
- **The whole executables differ** (`binaries-sha256.txt`). The causes are the kernel names in the HIP registration code, the source
  paths, the `__hip_cuid_<hash>` names and the FKL commit string that the programs write (section 2).

## 2. Host code of the timed calls

`step0/host_compare_taskc.py` is #61's `host_compare.py` with two options, `--rename-taskc` and `--line-args`. It compares the x86-64 code of
one program, A against B, function by function, and follows the direct calls from each row's timed call. The default mode uses both options:
- **`--rename-taskc`** pairs every function named after a renamed kernel with its new name: the host stubs `fk::__device_stub__<kernel>`, and the
  data symbol that is the kernel's host-side handle.
- **`--line-args`** writes the source line that a call of `fklb::statusMessage` (FKLB_*_CHECK, line in `%r8d`) or `fk::gpuAssert` (FKL's
  `gpuErrchk`, line in `%edx`) passes as `$LINE`. The `gpuErrchk` lines of FKL's `executors.h` moved down by 18 (in rpp_preprocessing, 587
  became 605), and the port removed lines above the rpp row of `e19_crop_sweep.h`. So these constants changed without a change of code.

The strict mode adds `--rodata-bytes 16`: the first 16 bytes of every read-only datum an instruction refers to are compared too.

| program | paper | rows compared | host functions in A / B | identical | functions that differ |
|---|---|---|---|---|---|
| `benchmark_rpp_preprocessing_hip` | VI-K | all 13 | 1,980 / 1,980 | 1,980 | none |
| `benchmark_vfhf_max_speedup_hip` | VI-N, GPU-size figure | the whole program | 295 / 295 | 295 | none |
| `benchmark_latency_hiding_hip` (frozen) | GPU-size figure | the whole program | 293 / 293 | 293 | none |
| `e19_crop_sweep_hip` | VI-M | fkl_batch, rpp; fkl_loop | 621 / 622 | 572 | 32, plus 17 in A only and 18 in B only |

**Every function reached from a timed call:**

| program: row | functions reached (A / B) | that differ (A / B) |
|---|---|---|
| e19_crop_sweep: **rpp** | 38 / 38 | **0 / 0** |
| e19_crop_sweep: **fkl_batch** | 79 / 80 | **32 / 33** |
| e19_crop_sweep: fkl_loop (not in the paper) | 52 / 53 | 17 / 18 |
| e19_crop_sweep: the harness (`fklb::timeIt`, `fklb::timePreparedLaunch`) | 21 / 21 | 0 / 0 |
| rpp_preprocessing: fkl_fused, fkl_fused_halfpixel, fkl_perop_batch | 136, 136, 141 | 0 |
| rpp_preprocessing: rpp_crop_batch, rpp_crop_batch_unsplit | 66, 66 | 0 |
| rpp_preprocessing: hand_batch, hand_padded, hand_roi | 64, 64, 65 | 0 |
| rpp_preprocessing: fkl_fused_halfpixel_once, hand_roi_halfpixel_once | 62, 64 | 0 |
| rpp_preprocessing: fastrpp_shared, fastrpp_crop_batch, fastrpp_shared_once | 196, 196, 162 | 0 |
| rpp_preprocessing: the harness (`fklb::measureWall`, `rppbench::timeVariant`) | 20 | 0 |

The roots are each row's timed lambda for every N: 16 numbers of crops in E19 and 30 in rpp_preprocessing. Each lambda is found by its
position in the source, as the compiler numbers them. Calls into shared libraries (the HIP runtime, RPP) end at the PLT. The same libraries
were used for both builds.

**What differs in the fkl_batch row,** for each of the 16 numbers of crops:
- **The row's lambda** (`e19amd::runCrops<N>()::{lambda()#3}`) is 19 instructions shorter. The `if (!fk::executeMorphology(...)) throw`
  is gone, and it calls the new entry function.
- **FKL's entry function.** A has `bool fk::executeMorphology<MorphologyDPPDetails<float, 16, 8, 7, 7>, ...>` (104 instructions at N = 150);
  B has `void fk::executeOperations<fk::MorphologyDPP<GPU_AMD, ...>, ...>` (108).
  - Both make the same calls in the same order: `__hipPushCallConfiguration`, `memcpy`, `__hipPopCallConfiguration`, `hipLaunchKernel`,
    `hipGetLastError` and `fk::gpuAssert`.
  - Before them, B calls `fk::executor_detail::requireArguments(bool, const char*, const char*)` (100 instructions, new in B). It throws
    `std::invalid_argument` when the details are invalid.
- **The kernel's host stub is identical.** It is paired by the rename, and the launch refers to it by address.

The fkl_loop row differs in the same way, for LinearFilterDPP: lambda #1 (17 instructions shorter), `executeLinearFilter` against
`executeOperations<fk::LinearFilterDPP<GPU_AMD, ...>>`, and `requireArguments`.

**The four comparison levels.** These are the counts of functions that differ in the whole program:

| program | raw (#61's script) | `--rename-taskc` | default (`--rename-taskc --line-args`) | strict (default + `--rodata-bytes 16`) |
|---|---|---|---|---|
| `e19_crop_sweep_hip` | 230 | 195 | 67 | 69 |
| `benchmark_rpp_preprocessing_hip` | 376 | 125 | 0 | 2 |
| `benchmark_vfhf_max_speedup_hip` | 91 | 24 | 0 | 20 |
| `benchmark_latency_hiding_hip` | 184 | 61 | 0 | 2 |

- **Raw:** the renamed stubs and handles are in one build only, and so are the calls that refer to them.
- **`--rename-taskc`:** the functions that remain differ only in the line constant of `fk::gpuAssert` or `fklb::statusMessage`. In rpp_preprocessing
  these are 125 functions, each with one `mov $0x24b,%edx` that became `mov $0x25d,%edx`.
- **Strict:** besides the ported calls of E19, the functions that differ are:
  - the HIP registration functions (`__hip_module_ctor`, `__hip_register_globals`), because the kernel name strings changed;
  - `fklb::writeSystemInfoHeader`, which writes the FKL commit (b10fc66 or 48fbf3f) into the CSV header.

  No timed call reaches either of them. In strict mode the rows have the same counts as in the table above.

### The sources behind the paper rows

The pin move changes only the four files above. Between 976e42b and main bd019f6, only `results/` changed. So the programs of these rows
have the same sources at main as at 976e42b (A), and the same sources at the head as at A, apart from the three ported E18 and E19 files.

Each paper row was measured at an earlier commit. The commits come from the `# repo:` and `# fkl:` lines of the records' CSVs; for
`../rpp-preprocessing-fastrpp/`, whose CSVs say `unknown`, from its README:

| record | rows | measured at (fkl-benchmarks, FKL) | its row code at main 976e42b |
|---|---|---|---|
| `../rpp-preprocessing-fastrpp/` | all 13 | 84d0bfa, FKL 1da77899 | same: every row lambda, `timeVariant`, `timeRotated`, `fklb::measureWall` |
| `../rpp-preprocessing-hand-batch/` | the first 6 | **6dc0541** (not 151d517), FKL 64ad9195 | the row lambdas: same; `timeVariant`, `measureWall`: see below |
| `../rpp-preprocessing-hand-roi/` | the first 8 | 151d517, FKL 64ad9195 | the same as for 6dc0541 |
| `../rpp-preprocessing-built-once/` | the first 10 | bdc7d19, FKL 64ad9195 | same: every row lambda, `timeVariant`, `timeRotated`, `fklb::measureWall` |
| `../e19-crop-sweep/` | fkl_batch, rpp | c8eb575, FKL 24478f0 | see below |
| `../vfhf-max-speedup/` | all | 7b96c47, FKL 64ad9195 | `fkl-self/vfhf/` and the `common/` headers it includes: unchanged |


- **At 6dc0541 and 151d517**, `rppbench::timeVariant` and `fklb::measureWall` differ from main. The cause is b446695 ("the card-internal PCIe
  link and the link to the host, both per row"), which added sysfs link readings before the warm-up, before the timed runs and after them.
  The timed loop and the warm-up loop of `measureWall` are text-identical at all four commits (`vi-k-sources.txt`: the same sha256 after the
  whitespace is collapsed).
- **The E19 crop sweep** changed once between c8eb575 and 976e42b: the fkl_loop row's `LoopDetails` lost the crop size (#60, the details port
  of FKL #433). Its effect is in `../pin-move-devcode-24478f0-b10fc66/` (#61): the fkl_batch and rpp rows' host code was identical, and the
  device code differed only in the fkl_loop kernel.
- **The FKL pin of the vfhf and rpp_preprocessing rows** moved from 64ad9195 or 1da77899 to b10fc66. Each step has a device code record:
  `../pin-move-devcode-64ad9195-1da77899/`, `../pin-move-devcode-1da77899-2c42a80/` and `../pin-move-devcode-24478f0-b10fc66/`.

`vi-k-sources.txt` is the output of `step0/row_sources.py`. For each measured commit, it compares the source of every measured row's lambda,
whitespace collapsed, and the timing functions, with 976e42b.

## 3. Compile-only facts at FKL 48fbf3f

### VI-P: E11, the three mistakes (amdclang++ 24, `scripts/e11_diagnostics.py`, 5 repeats)

| FKL | `e11_type_mismatch` | `e11_read_in_middle` | `e11_batch_mismatch` |
|---|---|---|---|
| c8fcbe8 (without the chain checks) | error: 46 lines, 1 error (`no matching function for call to 'exec'`) | error: 46 lines, 1 error (`invalid operands to binary expression`) | **compiles, no diagnostic** |
| b10fc66 (the pin) | static_assert: 40 lines, 1 error | static_assert: 34 lines, 1 error | static_assert: 34 lines, 1 error |
| **48fbf3f (the head)** | **static_assert: 40 lines, 1 error** | **static_assert: 34 lines, 1 error** | **static_assert: 34 lines, 1 error** |

The rules in the static_assert messages, at b10fc66 and at 48fbf3f:
- `chain_check.h:115`: "FKL: the OutputType of the previous Op is not convertible to the InputType of the next Op ...";
- `chain_check.h:143`: "FKL: Complete Read Operations as continuations are not allowed. It has to be an IncompleteReadBackOperation.";
- `chain_check.h:195`: "FKL: Fusing two batch operations of different BATCH size is not allowed ...".

The logs of b10fc66 and 48fbf3f are identical except for one line number, `executors.h:595` at b10fc66 against `executors.h:613` at 48fbf3f.
The c8fcbe8 lines are those of `../e11-diagnostics/`.

### VI-L: E3, where a fused chain stops compiling (amdclang++ 24, HIP, gfx1200)

`step0/e3_series.sh` runs `scripts/e3_clang_limits.py` (fkl-benchmarks a4b493c) with `--fkl-include` set to the FKL worktree at 48fbf3f.
It uses only the `bisect` and `run` modes, which compile and never execute; the `check` mode, which runs the kernel on the GPU, is not used.
Each N is one translation unit with a chain of N operations. The comparison is `../e3-compile-limits/` (FKL c8fcbe8, 2026-11-07), on
the same machine and with the same compiler.

| | 48fbf3f (this record) | c8fcbe8 (`../e3-compile-limits/`) |
|---|---|---|
| bisection, stack limit 8 MiB, address randomisation off | largest N that compiles **1,367**, first that does not **1,368** (9 compiles between 1,300 and 1,450) | 1,367 / 1,368 (12 compiles between 1,100 and 2,046) |
| 5 repeats with randomisation on: N = 1,367 / 1,368 / 1,369 | compiles 4 of 5 / 0 of 5 / 0 of 5 | 1 of 5 / 0 of 5 / 0 of 5 (1,366: 5 of 5) |
| the failure at 8 MiB | crash (exit 254) after 15.0 s at 3.2 GiB, while generating code for `fk::TransformDPPBase<...>` (`data_parallel_patterns.h:78`) | crash, 3.2 to 3.9 GiB used: the stack, not memory |
| N = 2,046, stack 16 MiB | compiles: 82.6 s, 9.99 GiB | compiles: 81.6 s, 9.7 GiB |
| N = 2,047, stack 16 MiB | `iop_fuser.h:150:17: fatal error: instantiating fold expression with 2049 arguments exceeded expression nesting limit of 2048`, after 1.0 s | the same error at the same place, after 1.0 s |
| N = 2,047, stack 16 MiB, `-fbracket-depth=2049` (N + 2) | compiles: 82.6 s, 10.0 GiB | (N = 2,100 there: 2,100 and 2,101 fail, 2,102 = N + 2 compiles) |

- **The threshold is the same, about 1,370 operations.** With randomisation on, N = 1,367 compiled 4 times of 5 here and once of 5 at
  c8fcbe8. This is the run-to-run spread that `../e3-compile-limits/` describes ("the boundary moves by about one operation from run to
  run"). The repeats do not measure a change of the stack use per operation.
- **The register columns of `e3/e3_limits.csv` are empty.** At a4b493c, `e3_clang_limits.py` reads the resource usage remark of the kernel
  named `launchTransformDPP_Kernel`. After FKL #435 that kernel is `launchDPP_Kernel<fk::TransformDPP<...>, ...>`, so the script found no
  remark. The compiles themselves are unaffected. The NVIDIA session's 2518c75 on `pin-fkl-435` adds the new name.
- **The kernel's resources, with 2518c75's script** (`e3/registers/e3_limits.csv`; N = 1,100 and 2,046, stack 16 MiB): 54 SGPRs, 9 VGPRs,
  no scratch, no spills, occupancy 16 at both N. These are the values that `../e3-compile-limits/` gives at c8fcbe8 for N = 1,100 and 2,046.
  `fkl-self/compiletime/gen_chain.py` is unchanged from a4b493c to 2518c75. The two files were run from a copy of those paths.
- The first run of the series stopped before the 16 MiB compiles: `ulimit -s 8192` in the script had also lowered the hard limit. The script
  now sets the soft limit only, and `e3/` holds the complete second run. Its bisection and repeats gave the same outcomes as the first run's.
- `e3/`: `e3_limits.csv`, `bisect.txt`, `series.log` and the compiler logs of the compiles that failed.

### VI-N: the HIP tests of FKL

Counted in the ROCm CI configuration builds of #64 (`../fkl-pr435-check/`), at both FKL commits (`vi-n.txt`):

| FKL | ctest tests named `*_hip` | HIP compile-rejection tests (`*_hip_rejected`) | `_hip` build targets | all ctest tests |
|---|---|---|---|---|
| b10fc66 | 52 | 4 | 68 | 119 |
| 48fbf3f | 52 | 4 | 68 | 119 |

The 68 targets are the 52 test executables, the 4 `utest_chain_check_wrong<k>_hip` that the rejection tests compile, and 12
`utest_saturate_<type>_hip` shared libraries of `utest_saturate_hip`. #64's GPU window ran the 52 tests at 48fbf3f: 52 of 52 passed.

## Builds

| build | sources | executables | failed | warnings | time |
|---|---|---|---|---|---|
| A, HIP | fkl-benchmarks 976e42b, FKL b10fc66 | 15: 10 `_hip` + 5 `_cpp` | 0 | 0 | 2:54 |
| B, HIP | fkl-benchmarks a4b493c, FKL 48fbf3f | the same 15 names | 0 | 0 | 2:56 |

- **The builds use `step0/pin_build.sh`**, the script of #45 and #61, byte for byte.
  - fkl-benchmarks is a `git archive` of each commit.
  - `external/fkl` is a symbolic link to a clean FKL worktree at that commit.
  - `external/FastRPP` is a link to a clean checkout of efbf009, the pin at both commits.
- **Tools:** amdclang++ (AMD clang 24.0.0git, ROCm 10.1.0), RPP 3.2.0, gfx1200, Release, `ninja -j2`, `FKLB_BUILD_RPP=ON`.
- **The comparisons** use `ROCM_PATH=/opt/rocm/core-10.1` and GNU objdump 2.46.

`builds.txt` has the commits and summary lines of `pin_build.sh` and the tool versions. `step0/run_checks.sh` writes every other output of
this folder except `e3/`, which `step0/e3_series.sh` writes.

## Not covered here

- **The CUDA programs.** Their ports and checks are the NVIDIA session's, on the same branch.
- **Run times.** Nothing was run. Whether the E19 fkl_batch number is measured again, now that its timed host code differs, is Oscar's
  decision.
- **The CPU code** of the `_cpp` programs was not compared.

## Files

- `devcode-compare.txt`, `devcode-compare-labels-as-symbols.txt`, `functions/<program>.tsv`: section 1.
- `host-code/{default,strict}/<program>.summary.txt` and `host-code/{raw,rename}/<program>.summary.txt.gz`: section 2 at each level.
  - `default/` and `strict/` also hold the instruction diffs (`.diffs.txt`), the function tables (`.functions.tsv.gz`) and the functions
    reached from each row (`.reach-<row>-{A,B}.txt.gz`).
- `vi-k-sources.txt`: the row sources at the measured commits against main.
- `e11/`: E11 at c8fcbe8, b10fc66 and 48fbf3f (CSV, logs).
- `e3/`: E3 at 48fbf3f (`e3_limits.csv`, `bisect.txt`, `series.log`, the logs of the compiles that failed; `registers/e3_limits.csv`).
- `vi-n.txt`: the test counts.
- `binaries-sha256.txt`: the 30 executables of the two builds. `builds.txt`: the builds.
- `step0/`: `pin_build.sh`, `run_checks.sh`, `e3_series.sh`, `devcode_compare_taskc.py`, `host_compare_taskc.py` and `row_sources.py`.
