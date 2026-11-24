# The pin move to FKL b10fc66 (the merge of #433): device-code identity of the CUDA programs, and the host code of E19 (RTX 4090 machine, CUDA 13.4.3; compile only, nothing ran)

**The library change.** FusedKernelLibrary #433 batches the LinearFilter, MedianFilter and Quad DPPs, with sizes per image.
- **The details:** `LinearFilterDPPDetails`, `MedianFilterDPPDetails` and the Quad details lose their `width` and `height`. The filter kernels take each image's size from the image itself.
- **Also new:** `AccumulatorType = VectorType_t<float, cn<T>>` and `neighborhood_detail` (activeThreadsOf, widthOf, heightOf, canRun). The write operations get `num_elems_y/z`.
- **Unchanged:** `morphology.h` is not among its files, so `MorphologyDPPDetails`, which E19's `fkl_batch` uses, keeps its sizes.
- **The merge:** #433 (head 4b11bc6) was merged into LTS-C++17 on 2026-11-24 at 22:02:02Z as `b10fc6609426bd18020b9213c7196b528fc677a8`, with parents 24478f0 and 4b11bc6.
  - Its tree, `62024145`, is the tree of 4b11bc6. The sm_89 check of 4b11bc6 (`../task-b-sm89-check/`) therefore holds for b10fc66: FKL's tests and the bit-for-bit outputs of the five programs before and after.
- **The port:** the programs that fill the filter details change with the pin. The first commit of this pin move applies two diffs as they are on main:
  - `../task-b-sm89-check/details-port.diff`: 27 hunks in the 5 CUDA programs under `opencv-cuda/stencil/` (e18_box, e18_box_split, e18_stencil, e19_batched_box, e19_crop_sweep);
  - `../../rx9060/task-b-gpu-window/fklb-after-details-port.diff`: 7 hunks in the three HIP headers under `rpp/`, built by AMD.

## Result
**Builds:** both compiled all 44 targets of the bench group, 39 `_cu` and 5 `_cpp`, with no failure (`identity-builds/*/report.txt`).
- base: fkl-benchmarks main a8f9daa with FKL 24478f0, 2,854 s;
- head: 2160c70 (the pin and the port) with FKL b10fc66, 2,847 s.

**Device code, base against head** (`sass-identity-24478f0-vs-b10fc66.txt`):
- **Kernels:** 1,834 in each build. All pair up, with no key collision.
- **SASS:** 1,815 kernels identical, 19 different.
- **The 19:** exactly the `linearFilterDPPKernel` instantiations of the five programs that use the filter DPP:

  | program | kernels | different |
  |---|--:|--:|
  | `e18_box_cu` | 9 | 8 |
  | `e18_box_split_cu` | 7 | 5 |
  | `e18_stencil_cu` | 7 | 4 |
  | `e19_batched_box_cu` | 3 | 1 |
  | `e19_crop_sweep_cu` | 18 | 1 |

  - Each is 8 or 16 SASS lines longer. This is the change `../task-b-sm89-check/` described: constant bank 0 is 8 B smaller, with the same stack and shared memory.
  - Every other kernel of the 39 programs is identical, including the 17 `morphologyDPPKernel` instances of `fkl_batch`.
- **Renamed:** 151 kernels, in 15 programs. All 151 are `hash_only` in `rename-causes-24478f0-vs-b10fc66.txt`: their mangled names are equal once nvcc's module id of the anonymous namespace (`_GLOBAL__N__<8 hex>_<len>_<file>_<8 hex>`) is masked. No name differs otherwise.
  - **Which kernels:** exactly the kernels that carry that id. Each build has 151 such kernel entries, in the same 15 programs (`anon-kernels-vs-renames-*.txt`, from the cubins' symbol tables).
  - **What changes:** the first 8 hex digits. For example `e19_batched_box_cu` has `87af3afa` at the base and `d7dcc2a3` at the head; the last 8 (`34ed719f`) are the same everywhere.
  - **Why #47 had none:** #47 built both sides from one checkout of fkl-benchmarks, with only the FKL tree swapped. This check builds each side from its own checkout, because the port changes the source.
  - **Two checkouts are enough.** The five programs have byte-identical source and FKL 24478f0 both in task B's "before" build and in this base build, yet their ids differ: `e18_box_cu` has `6636be59` there and `5bba7898` here (`module-ids-e18-e19.txt`).
  - **What is shown, and what not:** an FKL change confined to `FK_HIP_BACKEND` blocks, in one checkout, left the ids unchanged (#47), and a second checkout changes them. Whether #433's FKL change alone would change them in one checkout was not tested. What nvcc derives the id from was not determined.

**The five programs at the head against the task B "after" binaries** (`sass-identity-taskb-after-vs-b10fc66.txt`):
- **The "after" binaries:** fkl-benchmarks b4fb594 with the port uncommitted (so their build information says b4fb594-dirty) and FKL 4b11bc6. They are the binaries whose outputs `../task-b-sm89-check/` compared bit for bit.
- **Integrity:** before the comparison, `identity_builds.sh` checked their sha256 against `../task-b-sm89-check/window-binaries.sha256`; all 5 matched.
- **Result:** 44 of 44 kernels have identical SASS; no kernel is unpaired and no name collides. Five are renamed, all `hash_only`.
- **What it ties:** the committed source of the pin move compiles to the device code whose outputs task B checked.

**The binaries' bytes** (`binary-sha256-*.txt`):
- 43 of the 44 differ between the builds. `batch_build_compilation_time_cpp` is byte-identical.
- In both builds the build information names the repository commit `unknown`. `check_fkl_tree.sh` builds from a link farm of the checkout that has no `.git`, as in `../fkl-pr432-check/`.
- The binaries carry the FKL commit: 24478f0 in the base build, b10fc66 in the head build.

## The host code of E19 (`e19-host-code/`)
**The question.** E19's numbers are whole-call times. Does the pin move change any code that runs inside the timed call of the `fkl_batch` row, beyond its kernel?
- **Source:** in `e19_batched_box.cu` and `e19_crop_sweep.cu` the port changes one line each, inside `callFkl` (the `fkl_loop` row). `callFklBatch` is unchanged.
- **Inlining:** the compiled host code can still differ, because inlining depends on the whole translation unit. So `e19_batched_box_cu` and `e19_crop_sweep_cu` of the two builds were compared function by function.

**Method** (`scripts-used/host_compare.py`, run 2026-11-25 at 00:22, CPU only):
- **Disassembly:** `objdump -d -C --no-show-raw-insn -w -j .text` of each binary (GNU objdump 2.46). A function is the code between two symbol headers; its `.cold` clone is a function of its own.
- **Normalisation:**
  - instruction addresses, RIP-relative displacements and symbol offsets are removed;
  - a branch target inside the same function, or the start of another function, becomes the index of its instruction;
  - nvcc's anonymous-namespace module ids are masked;
  - padding (nop forms, int3) is removed. The assembler aligns loop heads relative to the section, so the number of nops follows the function's address, not its code.
- **Pairing:** by demangled name.
- **Which function is which row.** Every timed call is a lambda kept in a `std::function` (`Row::call`). The rows are recognised by what each `_M_invoke` reaches through direct calls and jumps, and through functions whose address it takes:
  - `morphologyDPPKernel` → `fkl_batch`;
  - `linearFilterDPPKernel` → `fkl_loop`;
  - an `nppi*` call → `npp`;
  - a `cv::cuda::` call → `opencv_cuda`.

  The lambda numbers are not used for this. In the crop sweep's function template, GCC also emits mangling aliases, so `{lambda()#8}` and `{lambda()#12}` of one `launchWithCrops<N>` are the same function at one address.
- **The harness:** `fklb::timeIt<std::function<void ()> >` and each `fklb::timePreparedLaunch<CTX::...>` record the events of the interval. `launch()` and `launchWithCrops<N>()` only call them.
- **What a row's timed call covers:** everything reached from its lambda and from the harness.
- **Not followed:**
  - indirect calls. The only ones in the `fkl_batch` call are the 4 calls of the `std::function` in the harness, and their target, the row's `_M_invoke`, is a starting point.
  - the shared libraries (libcudart, NPP, OpenCV, libstdc++). They are the same files for both builds and were not compared.

**Result:**

| program | functions | identical | different | unpaired |
|---|--:|--:|--:|--:|
| `e19_batched_box_cu` | 210 | 204 | 6 | 0 |
| `e19_crop_sweep_cu` | 698 | 662 | 36 | 0 |

- **`fkl_batch`: no function inside its timed call differs,** in `e19_batched_box_cu` and in each of the crop sweep's 16 `launchWithCrops<N>`.
  - This covers its own lambda and the `_M_invoke` of its `std::function`, and everything they reach: `executeBoxBlur<1>`, the batch build with `crop2D`, the stream and the host stub of `morphologyDPPKernel`. The harness is covered too.
  - The row has a function of its own. It shares 13 callees with other rows (`fk::gpuAssert`, `fk::ptr_crop::checkAxis`, `fk::Ptr::freePtr` and string helpers), and all of them are identical.
  - The harness is shared by all four rows, and it is identical.
- **`opencv_cuda` and `npp`:** no function of their timed calls differs, in any context.
- **What differs:**
  - **`fkl_loop`:**
    - its lambda (509 → 541 instructions in `e19_batched_box_cu`) and its `.cold` clone;
    - the two host stubs of `linearFilterDPPKernel`, whose stack frame grows from 0x90 to 0xa0 bytes;
    - in the crop sweep, also `fk::PtrCrop<3, float>::check`, which only `fkl_loop` reaches (the crop of its tensor plane).
  - **Outside every timed call:**
    - `launch()` and its `.cold` clone, where stack slots move;
    - the 16 `launchWithCrops<N>()`;
    - one instance of `fklb::CsvWriter::row`.
- **Every function that differs** is listed with its first differing instruction and with the rows that reach it (`host-*-24478f0-vs-b10fc66.txt`). Every function, with its result, is in the CSVs.

## Method
- **Compile only.** No program was run, for any purpose. The frozen 2025 programs were compiled and their SASS dumped, nothing more. `CUDA_VISIBLE_DEVICES` was empty.
- **Builds** (`scripts-used/identity_builds.sh`, sha256 0416a423…; output in `identity-builds/console.txt`):
  - Run: 2026-11-24 23:14:11 to 2026-11-25 00:16:00 (+01:00). The two builds ran at the same time on separate cores.
  - Base: the clean detached worktree `worktrees/fkl-benchmarks-pin-433-base` at main a8f9daa, with its `external/fkl` at the gitlink 24478f0.
  - Head: the branch worktree `worktrees/fkl-benchmarks-pin-433` at 2160c70, with its `external/fkl` at the gitlink b10fc66.
  - Both have cvGPUSpeedup 6d26c65, FastNPP a1b5e245 and FKL's thread-pool at 27e733f. The script checked that every checkout was at its commit with no tracked change, before and after.
  - Toolchain: nvcc 13.4.92 for sm_89, g++ 15.2.0, Release, OpenCV 4.14 with CUDA.
- **The build line,** once per checkout, from that checkout:
  ```
  scripts/check_fkl_tree.sh <checkout>/external/fkl --out <dir> --only bench --ref-build build/local-rtx4090 --jobs 12 --cores 8-15,24-31
  ```
  - The head build used `--cores 0-7,16-23`.
  - `--ref-build` takes the `FKLB_*` options, `OPENCV_DIR` and nvcc from the CMakeCache of the reference build of preset `local-rtx4090`. That build was only read.
- **The comparisons**, run by `identity_builds.sh` after the builds:
  ```
  python3 -I scripts-used/sass_identity.py base-24478f0/bench-build/bin head-b10fc66/bench-build/bin --tools scripts --arch sm_89 --glob '*_cu' --out sass-identity-24478f0-vs-b10fc66.txt
  python3 -I scripts-used/rename_causes.py base-24478f0/bench-build/bin head-b10fc66/bench-build/bin --arch sm_89 --glob '*_cu' > rename-causes-24478f0-vs-b10fc66.txt
  python3 -I scripts-used/sass_identity.py <task B after>/build/local-rtx4090/bin <links to the 5 head binaries> --tools scripts --arch sm_89 --glob '*_cu' --out sass-identity-taskb-after-vs-b10fc66.txt
  python3 -I scripts-used/rename_causes.py <the same two folders> --arch sm_89 --glob '*_cu' > rename-causes-taskb-after-vs-b10fc66.txt
  ```
  - `sass_identity.py` and `rename_causes.py` are byte-identical to those of `../fkl-pr432-check/scripts-used/`. The script checked the sha256 of `sass_identity.py`, c8ab1c23….
  - The SASS is normalised by `scripts/sass_tools.py`: addresses and encodings are removed, and registers are not renamed.
  - Pairing is by demangled name. The scripts' `drop_default` of `,fk::BinaryType` is applied to both sides alike.
- **The host code:**
  ```
  python3 -I scripts-used/host_compare.py base-24478f0/bench-build/bin/<program> head-b10fc66/bench-build/bin/<program> --out host-<program>-24478f0-vs-b10fc66.txt --csv host-<program>-24478f0-vs-b10fc66.csv --reach reach-<program>-24478f0-vs-b10fc66.txt --diffs diffs-<program>-24478f0-vs-b10fc66.txt
  ```
  - These are the outputs of 2026-11-25 00:22.
  - A first run at 00:16 had no `--reach` or `--diffs`. Its report and CSV are byte-identical to these.

## What this says and what not
- **What it says:**
  - At the pin move, the device code of every CUDA program of this repository changes only in the 19 `linearFilterDPPKernel` instantiations. Their outputs are bitwise those of 24478f0 (`../task-b-sm89-check/`).
  - The host code of E19's `fkl_batch`, `opencv_cuda` and `npp` timed calls is unchanged, instruction for instruction after the normalisation above.
  - So the E19 results of `fkl_batch`, `opencv_cuda` and `npp` stand for b10fc66, as far as the code goes.
- **What it does not cover:**
  - The GPU times of the rows that launch `linearFilterDPPKernel` were not measured at b10fc66 here. These are E18's FKL rows and E19's `fkl_loop`.
  - The host code of programs other than the two E19 programs was not compared.
  - The HIP builds are AMD's to check.
  - The CPU builds only compiled.
  - Nothing here measures time.

## Files
- `sass-identity-24478f0-vs-b10fc66.txt`, `rename-causes-24478f0-vs-b10fc66.txt`: base against head, every `_cu` program.
- `sass-identity-taskb-after-vs-b10fc66.txt`, `rename-causes-taskb-after-vs-b10fc66.txt`: the five programs, task B's "after" binaries against the head.
- `binary-sha256-base-24478f0.txt`, `binary-sha256-head-b10fc66.txt`: the sha256 of the 44 binaries of each build.
- `anon-kernels-vs-renames-24478f0.txt`, `anon-kernels-vs-renames-b10fc66.txt`: per program, its kernel entries and how many carry the anonymous-namespace module id (`scripts-used/anon_kernels.sh <bin folder> rename-causes-24478f0-vs-b10fc66.txt`), with its pairs and `hash_only` renames.
- `module-ids-e18-e19.txt`: the module ids of the five programs in four builds (task B's "before" and "after" builds, and this check's base and head; `scripts-used/module_ids.sh`).
- `e19-host-code/`, per program (`e19_batched_box_cu`, `e19_crop_sweep_cu`), base against head:
  - `host-<program>-24478f0-vs-b10fc66.txt`: the report (totals, the rows of every context, each row's differing and shared functions, and every function that differs with its first differing instruction and the rows that reach it);
  - `host-<program>-24478f0-vs-b10fc66.csv`: every function of either build, with its instructions in each, its result and the rows that reach it;
  - `reach-<program>-24478f0-vs-b10fc66.txt`: per context and row, every function its timed call reaches, from its lambda or from the harness only, with its result and the other rows that reach it;
  - `diffs-<program>-24478f0-vs-b10fc66.txt`: per function that differs, the unified diff of its normalised instructions.
- `scripts-used/identity_builds.sh`, `scripts-used/sass_identity.py`, `scripts-used/rename_causes.py`, `scripts-used/host_compare.py`, `scripts-used/anon_kernels.sh`, `scripts-used/module_ids.sh`.
- `identity-builds/console.txt`: the console of `identity_builds.sh`.
- `identity-builds/base-24478f0/` and `identity-builds/head-b10fc66/`: per build, the files that `check_fkl_tree.sh` wrote:
  - `report.txt`;
  - `bench-report.csv`;
  - `bench-targets.txt`;
  - `bench-targets-built.txt`;
  - `bench-time.txt`;
  - `bench-configure.log`.
- **Left out:**
  - the build logs, `bench-build.log`, 17.20 MiB each;
  - the build directories;
  - the two empty `bench-rejected-targets.txt` (the bench group has no expected rejections);
  - the two console logs of `check_fkl_tree.sh`, which are byte-identical to `report.txt`;
  - the folder of links to the five head binaries;
  - the 5 lines of task B's manifest that the script checked. They are in `../task-b-sm89-check/window-binaries.sha256`.

  They stay in `scratch/pin-433/` on this machine. Every file here is the complete output of the program named; nothing was filtered.
