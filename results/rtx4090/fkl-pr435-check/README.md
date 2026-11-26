# FKL #435 on the RTX 4090: the CUDA tests, the device code of every CUDA test kernel, and the DPP outputs before and after (CUDA 13.4.3, sm_89)

**The library change.** FusedKernelLibrary #435 (draft, written by the AMD session, Oscar's task C decisions):
- **One entry point:** every DPP runs through `fk::executeOperations<DPP>(stream, details, iOps...)`.
- **One kernel template:** `launchDPP_Kernel<DPP, DPPDetails, IOps...>` takes every parameter as `const __grid_constant__` and only calls `DPP::exec`. It replaces 13 kernel templates:
  - `launchTransformDPP_Kernel`, `launchDivergentBatchTransformDPP_Kernel`;
  - `linearFilterDPPKernel`, `medianFilterDPPKernel`, `morphologyDPPKernel`, which had no `__grid_constant__`;
  - the four Quad kernels;
  - softmax, SIMT flash attention, flash attention MMA, flash decode, and `flashFwdSplitCombineKernel`.
- **Invalid details** throw `std::invalid_argument`. The per-DPP and convenience functions are removed, and `fk::boxBlurReducers` is public.
- **The attention DPPs** take a leading `ParArch`.
- **Flash attention MMA:**
  - the tiles come from the type, and `flashAttentionMmaTiles` is the automatic choice;
  - split-KV is two DPPs and two calls (`FlashAttentionMmaSplitKVDPP`, then its `CombineDPP`);
  - the combine now applies the epilogue. The base's split path dropped it.
- **Commits:**
  - base: `b10fc6609426bd18020b9213c7196b528fc677a8` (LTS-C++17, the merge of #433, the FKL pin of this repository);
  - head: `48fbf3f08e7551c3eb70ac2302d8ebc15ecad645` (branch LTS-C++17-executeoperations-dpps).
- **The head moved during this check.** Step 1 built e322cad, the head until 2026-11-26 14:50. 48fbf3f is e322cad plus one commit, which changes only `utests/algorithm/attention/utest_flash_attention_mma.h`. The commit loosens the split-KV test: at e322cad it compared split with unsplit within 1e-4, and #435's CUDA CI failed that. Step 1b rebuilt the head at 48fbf3f, and everything after step 1b uses 48fbf3f.

Nothing is timed. This repository's pin does not change.

**State.** Done:
- the builds (steps 1 and 1b);
- the device-code comparison;
- the harness build.

**Pending:** the GPU window (`window-script.sh`), after the coordinator's grant.

## Result (compile only)
**Builds** (`compile/`): FKL's test matrix built at both commits, with no failure.

| | targets | built | failed | expected rejections held | build |
|---|--:|--:|--:|--:|--:|
| b10fc66 (base) | 162 | 150 | 0 | 12 of 12 | 503 s |
| e322cad (head, step 1) | 162 | 150 | 0 | 12 of 12 | 522 s |
| 48fbf3f (head, step 1b, incremental) | 162 | 150 | 0 | 12 of 12 | 15 s |

- **Targets:** both commits have the same 162 target names.
- **CUDA code:** 59 `_cu` executables and 12 `libutest_saturate_*_cu.so` shared libraries, the same names at both commits.
- **Step 1b:** after the move to 48fbf3f and a configure with the same command, make rebuilt one object and one executable, `utest_flash_attention_mma_cu` (`compile/fkl-head48-recompiled.txt`). The configure message names `0.1.14-pr-435-48fbf3f`.
- **Version strings:** FKL's version (branch and hash) is only printed by CMake. No generated header or define carries it into the binaries.
- **Warnings:** 1,495 warning lines in each full build (`compile/fkl-*-build.log`).

**Device code, base against head** (`sass-identity/sass-identity-fkl-tests-b10fc66-vs-48fbf3f.txt`, `.csv`, over all 71 files):

| | kernels |
|---|--:|
| base | 1,009 |
| head | 1,011 |
| pairs (by #435's rename rule) | 1,003, of which 996 are renamed |
| identical SASS | 939 |
| different SASS | 64 |
| only at the base | 6 |
| only at the head | 8 |

- **The 64 different:** exactly the three filter DPPs that gain `__grid_constant__`, and the combine.

  | DPP | kernels | different |
  |---|--:|--:|
  | LinearFilterDPP | 28 | 28 |
  | MedianFilterDPP | 14 | 14 |
  | MorphologyDPP | 20 | 20 |
  | FlashAttentionSplitCombineDPP (was flashFwdSplitCombineKernel) | 2 | 2 |

- **The rest is identical:** 864 TransformDPP, 8 DivergentBatchTransformDPP, 45 Quad (11 BoxFilter, 11 Conv, 12 Median, 11 Morph), 2 Softmax, 9 SIMT FlashAttention and 4 FlashAttentionMma (unsplit and split, 64 x 64 x 4) kernels, and the 7 kernels that #435 does not rename.
- **The filter kernels** (`sass-identity/differences-fkl-tests-b10fc66-vs-48fbf3f.txt` groups them; the .txt lists every kernel). The parameters now stay in the parameter space, and some kernels no longer copy them to a stack frame:
  - **LinearFilterDPP:** in 4 of the 28 the stack goes to 0 (from 80-144 B), with LDL 2-5 → 0, STL 10-18 → 0, 24-96 fewer instructions and 0 or 1 register less. The other 24 change by -8 to +8 instructions and by 0 to -3 registers.
  - **MedianFilterDPP:** its window still needs a stack.
    - In 8 of the 14 the stack shrinks by 88-192 B, with LDL -3 to -6, STL -11 to -24, 56-104 fewer instructions and -6 to +1 registers.
    - In the other 6 the stack stays 400 B, the instructions change by -8 to +8 and the registers by -2; in one of them LDL drops by 11.
  - **MorphologyDPP:** stack 0 at both commits; 0 to +16 instructions and 0 to +3 registers (34 → 37 at most).
  - **Unchanged:** the shared memory and constant bank 0 of every filter kernel.
- **The combine** (`sass-identity/combine-b10fc66-vs-48fbf3f.txt`, `scripts-used/combine_compare.py`):
  - **What stays:** the registers (36 for D = 64, 40 for D = 128), the stack (0) and the parameter reads (`c[0x0][0x160..0x174]`, the same offsets).
  - **Constant bank 0:** grows from 376 to 384 B, from the epilogue member of `Params`.
  - **D = 64:**
    - the floating-point instructions are the same sequence with the same operands;
    - what differs is the address arithmetic: the HEAD_DIM offset of the row-maximum read moves from the LDG immediates (-0x100) into a `UIMAD` addend (+0x40), with one `UMOV` more at each of 3 places;
    - also the BSSY targets and 5 NOPs of padding.
  - **D = 128:**
    - the same number of every floating-point opcode and of LDG and STG;
    - IMAD -3, S2R -3, UMOV +3, SHF -1, NOP +4;
    - the schedule and the register assignment differ.
- **Unpaired:** all 14 come from #435's changes to the tests, not to the library.
  - `utest_flash_attention_mma_cu`, 6 only at the base: the 128 x 128 x 8, 128 x 64 x 4 and 64 x 128 x 4 Mma kernels, unsplit and split. The base test called `executeFlashAttentionMma<HEAD_DIM>`, whose run-time tile choice instantiates every tile. The head test names `FlashAttentionMmaDPP<..., 64, 64, 4>`.
  - `utest_flash_attention_mma_cu`, 6 only at the head: the new split-KV test's Mma and SplitKV DPPs with the epilogue o * 2 + 0.5 (D = 64 and 128), and their two combines with that epilogue. The base's combine has no epilogue, so it has one combine per D.
  - `test_linear_filter_dpp_cu` and `test_median_filter_dpp_cu`, 1 each only at the head: the new GPU rejection case with a two-plane write (`twoPlaneWrite`, a `PerThreadWrite<ND::_3D>`). It throws before the launch, so the kernel is compiled and never launched.
- **Demangling:** GNU c++filt does not demangle the one kernel of `utest_get_arg_large_pack_cu` (6,906 characters) at either commit. llvm-cxxfilt 21 demangles it, and the pair is identical.

**The harness's own kernels** (`sass-identity/sass-identity-harness-b10fc66-vs-48fbf3f.txt`): 61 at the base, 59 at the head.
- **Pairs:** 57, of which 37 are identical and 20 different. The 20 are the same kinds as above: LinearFilterDPP 7, MedianFilterDPP 4, MorphologyDPP 7 and the 2 identity combines.
- **Only at the base, 4:** the split variants of the 128 x 128 x 8, 128 x 64 x 4, 64 x 128 x 4 and the int8 128 x 32 x 4 Mma kernels. The base's automatic `executeFlashAttentionMma` instantiates them, and for these shapes it does not launch them.
- **Only at the head, 2:** the combines with the epilogue.

## The harness (`harness/`)
**What it is.** A before/after program of this task, in two programs: `pr435_image.cu` and `pr435_attention.cu`.
- Each is compiled against b10fc66 and against 48fbf3f (`-DPR435_HEAD`). Only the calls into FKL differ between the two builds.
- Each case runs once on fixed inputs, an integer hash of the coordinates. Its output buffer is first filled with 0xff bytes. The case writes the output's bytes to `<case>.bin`.

**The calls.**
- **Base:** the entry functions of b10fc66, including the convenience functions #435 removes (`executeBoxFilter`, `executeErode/Dilate/Open/Close`, `executeBoxBlur`).
- **Head:** `executeOperations<DPP>`, with the arguments the PR says callers build instead:
  - `ConstantFilterRead` of 1 / (kw * kh) with Mul and Add;
  - `make_tuple(Min / Max ...)`;
  - `boxBlurReducers<T, PASSES>(details)`.

**The cases.**
- **LinearFilterDPP:**
  - single images: float 5 x 3, fused float 7 x 7, the box filter float 5 x 5 and uchar3 3 x 3;
  - batches of 3 crops of different sizes: float 3 x 5 to a TensorWrite and to a BatchWrite, and uchar3 to float3.
- **MedianFilterDPP:**
  - single images: uchar 3 x 3, uchar3 5 x 5;
  - a batch of 3 float crops of different sizes, fused: the median 3 x 3 to a TensorWrite, the minimum 5 x 3 to a BatchWrite.
- **MorphologyDPP:**
  - erode, dilate, open and close;
  - the box blur: 1 pass on a batch of 3 crops, and 2 passes on float3;
  - 3 explicit passes on a uchar batch.
- **The 4 Quads:** batches of 3 crops of different sizes, and single vector images.
- **TransformDPP:** without thread fusion; with it, on 64 x 32 (thread divisible) and on 61 x 43 (not).
- **DivergentBatchTransformDPP.**
- **SoftmaxDPP:** two cases, one with a fused prologue.
- **SIMT FlashAttentionDPP:** D = 64 dense, D = 128 causal, int8 K/V, and the epilogue o * 2 + 0.5.
- **FlashAttentionMmaDPP:**
  - explicit 64 x 64 x 4 tiles on 3 shapes the base does not split;
  - the automatic choice on 5 shapes, chosen to reach the tiles 64 x 128 x 4, 128 x 64 x 4, 128 x 128 x 8 and 128 x 32 x 4 (int8 K/V), and one automatic split. At the head the harness does what the PR says the caller does: `flashAttentionMmaTiles`, then `flashAttentionMmaSplits`, then SplitKV and CombineDPP when it splits.
- **The split path:** on the three shapes of the PR's `testMmaSplit`, with the identity and with the epilogue o * 2 + 0.5.
- **FlashDecodeDPP:** bf16 with 16 splits, int8 K/V with 4 splits, one split.

**The comparison** (`harness/source/compare_harness.py`, after the window, CPU only):
- **Bit for bit:** every output of the base against the head's, except the three split outputs with the epilogue.
- **The split path,** with the checks of the PR's test at 48fbf3f: TOL = 3e-3, against a double oracle on the bf16-rounded inputs. The harness computes the oracle on the host, as the test's `cpuAttention` does.
  - **Head:**
    - split + combine with the identity within TOL of the oracle;
    - split + combine with the epilogue, and the unsplit DPP with it, within 2 * TOL of 2 * oracle + 0.5;
    - the largest |split - unsplit|, reported.
  - **Base:** whether its split with the epilogue is bit for bit its split with the identity (the epilogue dropped), reported.
- **The split counts:**
  - the head uses `flashAttentionMmaSplits`;
  - the harness also recomputes the base's formula from `executeFlashAttentionMma`'s code, with the occupancy of each commit's unsplit kernel, and prints it;
  - at the head the harness requires the two counts to be equal.

**Build** (`harness/build/build-console.txt`, `harness/source/build_harness.sh`), 2026-11-26 14:55:37-14:56:36:
- nvcc 13.4.92 with the CUDA flags of FKL's tests: `-O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_89,code=[compute_89,sm_89] -ftemplate-depth=1000 -lineinfo`;
- FKL's include folder, compile only. The sources' and binaries' sha256 are in the console.

## The GPU window (`window-script.sh`, pending)
The window starts only after the coordinator's grant. It stops at the first failure.
1. **FKL's CUDA tests at 48fbf3f:** the 59 ctest tests named `*_cu`, with `ctest -j1 --stop-on-failure --timeout 300`. They include `utest_flash_attention_mma_cu` with the new split-KV test.
2. **The same 59 at b10fc66,** the baseline.
3. **The harness, base build:** `pr435_image`, then `pr435_attention`.
4. **The harness, head build:** the same.

Then, on the CPU, `compare_harness.py`.

**Guards:**
- no blocked program;
- the sha256 of all 146 binaries against `window-binaries.sha256`: 59 tests and 12 libraries per build, and the 4 harness programs;
- the same 59 tests in both builds;
- clean FKL trees at their commits, and the builds configured from them;
- the record's branch differs from main only in this folder;
- the script and the comparison are the committed versions;
- new output folders;
- no busy process and no foreign GPU context.

## Method
- **Compile only.** No GPU was visible (`CUDA_VISIBLE_DEVICES` empty), and nothing ran except the compile-fail ctest tests (`*_rejected`), which only build.
- **Checkouts:** clean detached worktrees of the FusedKernelLibrary clone, with thread-pool 27e733f:
  - `worktrees/fkl-pr435-base-b10fc66`;
  - `worktrees/fkl-pr435-e322cad`, at e322cad in step 1 and at 48fbf3f since step 1b. Its path is kept, so the build tree stays valid.

  The scripts checked that each was at its commit with no change, before and after each build.
- **Step 1** (`scripts-used/compile.sh`, `compile/console.txt`), 2026-11-26 14:34:05-14:43:09 (+01:00):
  - FKL's CMake out of source, in Release, with `-DCUDA_ARCH=89`, the flags of the #433 check (`../task-b-sm89-check/`). FKL ignores `CMAKE_CUDA_ARCHITECTURES` and gives each target `CUDA_ARCH`.
  - The targets come from CMake's code model, and the report from `scripts/check_fkl_tree_report.py`.
  - The two builds ran at the same time on separate cores, 12 jobs each, at nice 10.
  - Toolchain: nvcc 13.4.92, cmake 4.2.3, g++ 15.2.0.
  - The exact commands are in `compile/fkl-*-commands.txt`.
- **Step 1b** (`scripts-used/compile-48fbf3f.sh`, `compile/fkl-head48-console.txt`), 14:52:30-14:53:05: the move to 48fbf3f, the configure, and the same make command and target list.
- **The device-code comparison** (`scripts-used/pr435_sass_identity.py`, CPU only):
  - **SASS:** `cuobjdump -arch sm_89 -sass` of every `*_cu` executable and `lib*_cu.so`, split per kernel and normalised by `scripts/sass_tools.py`: the addresses and encodings removed, the registers not renamed.
  - **Resources:** registers, stack, shared memory, local memory and constant bank 0 from `cuobjdump -res-usage`. LDL, STL and the opcode counts come from the SASS.
  - **Pairing:** by the demangled name, whitespace removed. The base's names are rewritten by #435's rename rule; the docstring lists every rule. For example:
    - `launchTransformDPP_Kernel<PA, TFEN, TD, D, IOps...>` becomes `launchDPP_Kernel<TransformDPP<PA, TFEN, D, TD, void>, D, IOps...>`;
    - `linearFilterDPPKernel<PA, D, ...>` becomes `launchDPP_Kernel<LinearFilterDPP<PA, D>, D, ...>`;
    - the attention DPPs gain `(fk::ParArch)1` as their first argument, also inside their `Params` types;
    - `flashFwdSplitCombineKernel<OT, D, LOG2D>` becomes the combine DPP with `AttentionIdentityEpilogue`.

    Nothing else in the names changes. A key found twice in one file would be an error; none was found.
  - **Summaries:** `scripts-used/diff_summary.py` groups the differences. `scripts-used/combine_compare.py` compares the combines with the registers renamed and the branch targets removed.

## Files
| File | Contents |
|---|---|
| `compile/console.txt`, `compile/fkl-{base,head}-*` | step 1: console, commands, configure and build logs, target lists, rejection tests, per-target report, build time (`head` = e322cad) |
| `compile/fkl-head48-*` | step 1b at 48fbf3f: the same, and `fkl-head48-recompiled.txt` |
| `compile/fkl-{base,head}-cu-executables.txt`, `compile/ctest-{base,head}-all.txt` | the CUDA executables, and every ctest test of each build |
| `sass-identity/sass-identity-fkl-tests-b10fc66-vs-48fbf3f.{txt,csv}` | the comparison of FKL's tests: report, and every kernel with its numbers |
| `sass-identity/differences-*.txt` | the differing kernels grouped by DPP |
| `sass-identity/sass-identity-harness-b10fc66-vs-48fbf3f.{txt,csv}` | the same for the harness's two programs |
| `sass-identity/combine-b10fc66-vs-48fbf3f.txt` | the combine kernels, line by line |
| `harness/source/` | the harness, its build script and its comparison |
| `harness/build/` | the harness build's console and the compiler output |
| `scripts-used/` | compile.sh, compile-48fbf3f.sh, pr435_sass_identity.py, diff_summary.py, combine_compare.py |
| `window-script.sh`, `window-binaries.sha256` | the window's script and the sha256 of the 146 binaries it runs |

**Left out:**
- the build trees and the binaries, in `scratch/pr435/` of the RTX 4090 machine;
- the harness outputs, which the window writes to `scratch/pr435/harness-out/`. Their sha256 will be in `window/`.
