# FKL #436 on the RTX 4090: task C on FKL main, the CUDA tests, the device code of every CUDA test kernel, and the DPP outputs before and after (CUDA 13.4.3, sm_89)

**The library change.** FusedKernelLibrary #436 (draft, branch `main-executeoperations-dpps`, head
`1df2738e7bd8eba1bfb1a0baf9ad215062d014ca`) ports task C to FKL main: 4 commits and 29 files on main
`ebb3891bea18a152d92dfd5092e117160490e34b`, which is C++20.
- **What it applies:** Oscar's task C decisions, as #435 applied them to LTS-C++17 (`../fkl-pr435-check/`): every DPP through
  `fk::executeOperations<DPP>`, one kernel `launchDPP_Kernel<DPP, Details, IOps...>`, invalid arguments throwing.
- **Plus Oscar's three main answers:**
  - M1: `SoftmaxDPP<ParArch PA, DPPDetails = SoftmaxDPPDetails<256>>`, which keeps main's operation-first form;
  - M2: `utest_flash_attention_mma.h` is added to main;
  - M3: the docs.
- **The HIP side:** AMD found the HIP device code identical (174/174) and the CPU tests passing 50/50 with g++-13 and clang++-23
  (`../../rx9060/fkl-pr436-check/`, #72).

Decision: the coordinator's assignment C-MAIN-NV of 2026-11-26, the CUDA check of #436. Another session (fkl-reduce) has priority on this machine
(Oscar, 2026-11-26). So every build here ran at nice 19 with at most 8 jobs, and the window starts only in a slot agreed with it.

**State.** The compile-only steps are done: the builds, the device code and the harness. The GPU window waits for the coordinator's GRANTED.

## Result (compile only)

- **Builds** (`compile/`, `scripts-used/compile.sh`, 2026-11-26 20:54:37 to 21:06:06): FKL's own test matrix at ebb3891 and at 1df2738, from clean
  detached worktrees (thread-pool 27e733f).
  - The settings are task C-NV's: Release, `-DCUDA_ARCH=89`, nvcc 13.4.3. The targets come from CMake's code model.
  - The jobs ran at nice 19, 4 per build, on separate cores.
  - Base: 142 of 142 targets compiled. Head: 143 of 143 (the new `utest_flash_attention_mma_cu`). Main has no expected-rejection tests.
  - The build times are not results.
- **CUDA tests:** 56 `*_cu` ctest tests at the base and 57 at the head. The head's list is the base's plus `utest_flash_attention_mma_cu`.
- **Device code** (`sass-identity/`), the 56 executables both builds have:
  - 145 kernels in each, **all paired**: 139 through a rename (137 by #435's rule, 2 by the main softmax rule below) and 6 with the same name.
  - **109 identical.**
  - **36 different, every one by design:** LinearFilterDPP 10, MedianFilterDPP 4 and MorphologyDPP 20 (the `__grid_constant__` details
    of Q3), and the 2 softmax kernels (their NullType argument goes, M1).
  - Every other kernel is identical, the TransformDPP, attention, flash-attention MMA and Quad kernels among them.
  - Main's tests instantiate no split-KV combine kernel at the base, so none is compared.
  - In the head only: `utest_flash_attention_mma_cu`, with 14 kernels.
- **The harness** (`harness/`): C-NV's before/after harness, adapted to FKL main. Both programs compile at both commits.
- **A finding outside #436:** **FKL main's FlashDecodeDPP does not compile when it is instantiated, at either commit.**
  - `flash_decode.h` (ebb3891:69, 1df2738:73) calls `attnToF32` unqualified.
  - Main defines it in `fk::low_precision` (`softmax.h:57` / `58`). LTS-C++17 defines it in `fk` (b10fc66 `softmax.h:81`), so it compiles there.
  - No test of main instantiates FlashDecodeDPP, so the harness leaves its decode cases out.
  - Nothing was changed in FKL; this is for Oscar.

## Device code (`sass-identity/`)

`scripts-used/sass_identity_main.py` runs `pr435_sass_identity.py`, task C-NV's script, unchanged, with one rule added for FKL main.
#435's rename rule applies as it is, since main's TransformDPP template is LTS's.
- **The added rule:** FKL main before #436 already launched its softmax through a kernel named `launchDPP_Kernel`, with five template
  parameters: `launchDPP_Kernel<fk::SoftmaxDPP, Details, InIOp, NullType, OutIOp>` (`softmax.h:160` at ebb3891).
- #436 makes SoftmaxDPP a template on ParArch and its details, and launches the common kernel without the NullType:
  `launchDPP_Kernel<fk::SoftmaxDPP<(fk::ParArch)1, Details>, Details, InIOp, OutIOp>`.
- The rule rewrites A's names of the first form into the second, and then #435's rule runs as before.

The comparison covers the SASS text, with addresses and module ids masked, and the registers, stack, LDL / STL and constant bank 0
(`sass-identity-ebb3891-vs-1df2738.txt`, `.csv.gz`; by DPP in `differences-ebb3891-vs-1df2738.txt`).

| DPP | kernels that differ | what changes |
|---|---|---|
| LinearFilterDPP | 10 | instructions equal or −8; registers equal or 1 to 3 fewer (39 → 36 at most); no stack |
| MedianFilterDPP | 4 | instructions equal or −8; registers 58 → 56 in 2; stack 400 B in both; LDL 85 → 74 in 1 |
| MorphologyDPP | 20 | instructions −8 to +16; registers equal or 1 to 3 more (34 → 37 at most); no stack |
| SoftmaxDPP | 2 | instructions and registers equal; constant bank 0 416 → 408 and 424 → 416 B (the NullType parameter goes) |

**The harness's own kernels**, base against head (`sass-harness-*.txt`):
- `pr436_image`: 22 pairs, 9 identical. The 13 that differ are LinearFilterDPP 3, MedianFilterDPP 3 and MorphologyDPP 7.
- `pr436_attention`: 22 pairs, 18 identical. The 4 that differ are 2 split-KV combines and the 2 softmax kernels.
  - In the base only: 4 split-KV MMA kernels. The base's `executeFlashAttentionMma` instantiates the split variant of each tile choice.
  - In the head only: 2 combines with the epilogue.

## The harness (`harness/`)

**Source** (`harness/source/`): `pr436_common.h`, `pr436_image.cu`, `pr436_attention.cu`, `build_harness.sh`, and task C-NV's
`compare_harness.py`, unchanged (sha256 54fe97ba...). Every change from C-NV's sources is in `changes-from-c-nv.diff`.
- **The calls.** Only the calls into FKL differ between the two builds (`#ifdef PR436_HEAD`):
  - the base calls main's old entry functions (`executeLinearFilter`, `executeBoxFilter`, `executeMedianFilter`,
    `executeErode/Dilate/Open/Close`, `executeBoxBlur`, `executeMorphology`, `execute*Quad<DPP>`, `executeSoftmax<256>`, `executeFlashAttention`,
    `executeFlashAttentionMma`);
  - the head calls `executeOperations<DPP>`, with the arguments the removed functions built.
- **The image cases, adapted to main's image DPPs.** These differ from LTS-C++17's, where C-NV's harness was written (#433 is not on main):
  - the details carry the image's width and height;
  - each call filters one image, with no batches;
  - the Quad DPPs take scalar pixels.

  So the image cases are 22 outputs of single images of float or uchar:
  - LinearFilterDPP 4, of which 2 box filters;
  - MedianFilterDPP 3;
  - MorphologyDPP 7 (erode, dilate, open, close, 2 box blurs, explicit reducers);
  - the four Quad DPPs, one each;
  - TransformDPP 3 and DivergentBatchTransformDPP 1, as in C-NV.

  The box filter's replacement at the head builds what main's `executeBoxFilter` built at ebb3891 (`linear_filter.h:333-339`): the
  coefficient `1 / (kw * kh)` in the image type, a `ConstantFilterRead` of it, and `Mul` and `Add` of that type.
- **The attention cases are C-NV's**, with two changes:
  - softmax uses main's operation-first form, as main's `utest_softmax.h` does at each commit;
  - the three FlashDecode cases are left out (the finding above).

  The split cases are those of the PR's `testMmaSplit`: the identity across the commits bit for bit, and at the head the double-oracle checks
  within TOL = 3e-3, or 2 * TOL with the epilogue. Main's base combine applies no epilogue, as LTS-C++17's did before #435. That check is
  only reported.
- **Two compile details of main**, both in the harness only:
  - **the vector types:** the image harness names `fk::float3`, `fk::float4` and `fk::uchar3`. On main, FKL has its own vector types in `fk`, and
    the bare names are ambiguous with CUDA's;
  - **the using-directive:** `using namespace fk` is in a named namespace, not an unnamed one. The host stub code that nvcc adds at global scope
    names CUDA's `uint3`, and a using-directive in an unnamed namespace would reach the global scope.

**Build** (`harness/compile/`, 2026-11-26 21:15:42 to 21:15:54): `build_harness.sh` with main's CUDA test flags (`-std=c++20`, sm_89, `-O3`,
`-lineinfo`), nice 19, the 4 compiles at once. All 4 compiled.

## The GPU window (`window-script.sh`)

(To be run on the coordinator's GRANTED, after RT-1's window.)

**The protocol is task C-NV's**, with RT-1's idle checks. In this order, stopping at the first failure:
1. FKL's 57 CUDA tests at 1df2738 (`ctest -R '_cu$' -j1 --stop-on-failure --timeout 300`);
2. the 56 at ebb3891;
3. the harness at the base, `pr436_image` then `pr436_attention`;
4. the harness at the head, the same.

**Around every step:** `wait_idle` waits for an idle machine before it, and `still_quiet` checks afterwards that no build, compile job or
other GPU program appeared during it; if one did, the window stops there and that step is the one affected.

**Afterwards, on the CPU only:** `compare_harness.py` compares base and head bit for bit, and runs the split checks.

The 117 binaries it runs are in `window-binaries.sha256`: 56 + 57 test executables and the 4 harness programs. Nothing is timed.

## Files

| path | content |
|---|---|
| `compile/` | the builds: console, commands, configure logs, target lists, build logs, reports, times |
| `sass-identity/` | the test kernels and the harness kernels, base against head, and the differences by DPP |
| `harness/source/` | the harness, its build script, the comparison, and `changes-from-c-nv.diff` |
| `harness/compile/` | the harness build's console and compiler logs |
| `scripts-used/` | `compile.sh`, `check_fkl_tree_report.py`, `pr435_sass_identity.py`, `sass_identity_main.py`, `diff_summary.py` |
| `window-script.sh`, `window-binaries.sha256` | the window's script and the sha256 of its 117 binaries |
