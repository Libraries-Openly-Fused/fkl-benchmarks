# Device code of the HIP programs at the two pins, 64ad9195 and 1da77899

The pin of `external/fkl` moves from 64ad9195 to 1da77899, the merge of FKL #431 into LTS-C++17 (2026-11-09). The move itself is in a separate pull request by the NVIDIA session. The RX 9060 results of `results/rx9060/` were measured at 64ad9195 or at earlier pins.

#431 is the NVIDIA session's port of main's image-processing DPPs to LTS-C++17, for CUDA and the CPU. It also changes three headers that the HIP programs include, as on main (see "FKL commits" below). One of them is `arithmetic.h`, in which `fk::Sub`, `fk::Mul` and `fk::Div` get a 4th template parameter that defaults to `fk::BinaryType`. So every kernel with one of them in its name has a new name at 1da77899.

The check: every HIP program of this repository was built twice, once against each of the two FKL commits, from the same sources with the same tool and flags. Their gfx1200 device code was then compared function by function, with the functions paired by a name that leaves out the new default argument. Nothing was run. The 64ad9195 build was made on 2026-11-08, and the 1da77899 build and the comparison on 2026-11-09, on the RX 9060 machine (`../README.md`). The build needs no GPU.

## Result: the device code is the same at both pins

| HIP program | functions at 64ad9195 / 1da77899 | renamed | identical | different | in one build only |
|---|---|---|---|---|---|
| `benchmark_horizontal_fusion_hip` | 125 / 125 | 121 | 125 | 0 | 0 |
| `benchmark_thread_fusion_hip` | 153 / 153 | 0 | 153 | 0 | 0 |
| `benchmark_latency_hiding_hip` | 63 / 63 | 60 | 63 | 0 | 0 |
| `benchmark_numinstructions_x_operation_hip` | 187 / 187 | 0 | 187 | 0 | 0 |
| `benchmark_rpp_preprocessing_hip` | 217 / 217 | 63 | 217 | 0 | 0 |
| `benchmark_vfhf_max_speedup_hip` (19 code objects, paired by position) | 26 / 26 | 20 | 26 | 0 | 0 |
| `batch_build_compilation_time_hip` | no device code in either build | | | | |
| **7 programs** | **771 / 771** | **264** | **771** | **0** | **0** |

Every function pairs with exactly one function of the other build: 771 pairs. No two names of one build give the same key, so there are 0 key collisions. The 771 functions are the 759 kernels, plus the two runtime stubs `__cxa_pure_virtual` and `__cxa_deleted_virtual` in each of the six programs with device code. In `benchmark_vfhf_max_speedup_hip` each function is compared code object by code object. Its 26 functions have 62 definitions over the 19 code objects, and all of them are the same.

The 264 renamed functions are exactly those with `fk::Sub`, `fk::Mul` or `fk::Div` in their name at 64ad9195. At 1da77899 their names carry 2,880 `,fk::BinaryType` arguments, each the 4th argument of one of these three:

| program | `,fk::BinaryType` arguments |
|---|---|
| horizontal fusion | 1,452 |
| latency hiding | 476 |
| RPP preprocessing | 732 |
| vfhf | 220 |

At 64ad9195 no name has `,fk::BinaryType` as an argument of these three. Its 880 occurrences there all belong to `fk::Add`, which had the four-parameter form at both pins, so its names did not change. No function of `benchmark_thread_fusion_hip` or `benchmark_numinstructions_x_operation_hip` has `fk::Sub`, `fk::Mul` or `fk::Div` in its name.

The whole executables differ (`binaries-sha256.txt`). Their host code was not compared instruction by instruction. Its known differences are names and strings:
- the kernel names;
- the FKL commit that six of the programs record (`FKLB_COMMIT_FKL`);
- the `__hip_cuid_<hash>` symbol of each HIP object, and the source paths. Both come from the two build folders.

## How the functions are paired

`scripts/devcode_compare.py` pairs the functions of two builds by their mangled names, so it would see a renamed kernel as one function only at 64ad9195 and another only at 1da77899. `scripts/devcode_compare_renamed.py` pairs them by a key instead:
1. Demangle the name and remove all whitespace.
2. Remove `,fk::BinaryType` where it is the 4th top-level template argument of `fk::Sub`, `fk::Mul` or `fk::Div`. This is `drop_default`, the NVIDIA session's rule for its CUDA check of #431, copied verbatim from its code.
3. Change nothing else in the name.

A mangled name inside the instruction text would be replaced by its key in the same way. No instruction line holds one, in either build. After each program's line the script prints a `names:` line: the pairs, the renamed functions, the arguments removed in each build and the key collisions. A key collision makes the last line `DEVICE CODE DIFFERENCES`.

**Validation of the script.** Before this comparison the script was run, in both modes, on three pairs of earlier builds and compared with `scripts/devcode_compare.py` as of 8f4a95b.
- **The two builds of the previous pin move** (`../pin-move-devcode-c8fcbe8-64ad9195/`), where no function was renamed. Without its `names:` and `NAMES TOTAL:` lines, its two outputs are byte for byte that folder's `devcode-compare.txt` and `devcode-compare-labels-as-symbols.txt`.
- **Two builds of `benchmark_vfhf_max_speedup_hip`** (19 code objects, paired by position). Its output is the same as that of `devcode_compare.py`.
- **Two builds that differ.** Both scripts report the same differences.

In all three, its tables give every function the same hash as the tables that `devcode_compare.py` as of 8f4a95b writes for the same builds. The tables committed in the previous folder were made with the script of 4acf63c and have other hash values (see its README).

The naming was also unit-tested, in 26 cases:
- 13 of the key itself. A 4th argument other than `fk::BinaryType`, such as `fk::UnaryType`, stays in the name.
- 13 of the pairing, the key collisions and the names in the instruction text.

The script was also run on a small HIP program compiled in four variants and never run:
- the renamed kernels paired as identical;
- a kernel with the `fk::UnaryType` form stayed unpaired;
- a changed body of `fk::Sub` was reported as different.

## What was compared

**Sources.** fkl-benchmarks 8f4a95be2d3d6db8646e061d228d4b694fb21e9d, main before the pin move. It was extracted with `git archive` into one folder per build, and in each, `external/fkl` was a symbolic link to that build's FKL checkout.

Since the sources of the previous pin move (0fb1f9f97), the device code of two programs changed:
- The RPP program was rewritten for RPP's documented batched form (#32). It also gained the hand-written variants (#33, #36) and the variants built once (#38). These are 8 commits to `rpp/`, from 282f734 to 32e2e20. It now has 217 functions, among them the 97 of then.
- `benchmark_vfhf_max_speedup_hip` was added (#39).

The other four programs with device code have the same functions as then, with the same hashes at 64ad9195. The host code of every program changed too: `common/main.cpp` and `common/csv_writer.h`, in 0957391, a6e3e69 and 4c8dd85.

**FKL commits.** Each was a clean detached checkout:
- 64ad9195cd785d0c93a70b35e02d5a6c7c4a5048, the pin until this move;
- 1da778992756e0fa08ec289225fe18dca134412c, the merge of #431 into LTS-C++17. Its parents are 64ad9195 and c6f7473ed8092adf708359c1e4b2e205ddadbc9d, and its tree is that of c6f7473.

Both use the `thread-pool` submodule at 27e733f50f7f40f171fd84655e05d5acf1fdfbee, with the same files on both sides. #431 is four commits on 64ad9195:
- 988dcd1, `arithmetic.h`: the unary forms of `Sub`, `Mul` and `Div` on a `Tuple<I1, I2>`, as on main (the arithmetic.h part of main's 85c3240), with the 4th template parameter;
- 0097b07, `fused_operation.h`: the `exec` of a fused compute-then-write IOp compiles, as on main;
- 733b5fa, `utils.h`: `FK_DEVICE_STATIC`, as on main;
- c6f7473: the image-processing DPPs of main (morphology, linear filter, median filter and the quad DPPs), for the CPU, CPU_MULTITHREAD and CUDA.

The HIP programs include the headers of the first three commits; `benchmark_thread_fusion_hip` does not include `arithmetic.h`. None of them includes a header of the new DPPs.

**Programs.** The seven targets with a `hip` back-end, declared with `BACKENDS ... hip`:
- `fkl-self/CMakeLists.txt`: 5 programs;
- `fkl-self/vfhf/CMakeLists.txt`: 1 program, built from 19 translation units;
- `rpp/CMakeLists.txt`: 1 program.

The other directories have no HIP target.

**Build.** amdclang++ (AMD clang 24.0.0git, ROCm 10.1.0) with RPP 3.2.0, the same for both builds and as at the previous pin move:

```
cmake -S <sources> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DFKLB_ENABLE_CUDA=OFF -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF \
  -DFKLB_BUILD_OPENCV_CPU=OFF -DFKLB_BUILD_HANDWRITTEN=OFF -DFKLB_BUILD_RPP=ON
nice -n 10 ninja -C <build> -j2 -k 0
```

The two CMake caches are equal apart from the paths. So are the 55 build steps: 54 compile and link commands, and the writing of `generated/fklb_build_info.h`. Each build produced 12 executables with no failures and no warnings:

| executable | gfx1200 device code |
|---|---|
| `benchmark_horizontal_fusion_hip` | 1 code object |
| `benchmark_thread_fusion_hip` | 1 code object |
| `benchmark_latency_hiding_hip` | 1 code object |
| `benchmark_numinstructions_x_operation_hip` | 1 code object |
| `benchmark_rpp_preprocessing_hip` | 1 code object |
| `benchmark_vfhf_max_speedup_hip` | 19 code objects, one per translation unit |
| `batch_build_compilation_time_hip` | none |
| `batch_build_compilation_time_cpp` | none |
| `benchmark_horizontal_fusion_cpp` | none |
| `benchmark_thread_fusion_cpp` | none |
| `benchmark_latency_hiding_cpp` | none |
| `benchmark_numinstructions_x_operation_cpp` | none |

The six executables without device code have no `.hip_fatbin` section, no offload bundle and no AMDGPU ELF anywhere in the file. The comparison of the seven `_hip` executables therefore covers all the device code of the build.

**Comparison.** `scripts/devcode_compare_renamed.py`, run with `ROCM_PATH=/opt/rocm/core-10.1`. For each executable it does what `scripts/devcode_compare.py` as of 8f4a95b does (see that script's header):
1. It reads every offload bundle of `.hip_fatbin` and copies out its gfx1200 code object.
2. It disassembles the code objects with `llvm-objdump -d --symbolize-operands`.
3. It compares the sha256 of the instruction text of each function, without the comments and with the `__hip_cuid_<hash>` names normalised.

Apart from the naming described above (the key, the `names:` lines and the key collisions), it is the same script.

```
python3 -I scripts/devcode_compare_renamed.py <build 64ad9195>/bin <build 1da77899>/bin --glob '*_hip' --table functions > devcode-compare.txt
python3 -I scripts/devcode_compare_renamed.py <build 64ad9195>/bin <build 1da77899>/bin --glob '*_hip' --labels-as-symbols > devcode-compare-labels-as-symbols.txt
```

## Controls

1. **The stricter form.** With the branch targets as separate symbols, numbers included (`--labels-as-symbols`), 1,795 of 1,795 symbols are identical.
2. **The two builds are different.** The whole executables differ in all seven programs (`binaries-sha256.txt`). The names show that each build compiled its own FKL: the 2,880 `,fk::BinaryType` arguments of `fk::Sub`, `fk::Mul` and `fk::Div` are in the names at 1da77899 only. Inside the code objects only the names and the layout differ (check 4 below).

## Four independent checks

The result was checked in four more ways on the same two builds, each with its own scripts, which are not kept in the repository.

1. **Provenance.**
   - Both FKL checkouts were clean at their commits.
   - The two source trees are the same, and equal to 8f4a95be2.
   - The CMake caches, `build.ninja` and the 55 build steps are the same apart from the paths.
   - The dependency lists of each build name the same 69 FKL files, all inside that build's own checkout. Three of them differ between the pins: the three headers above.
2. **Bytes.** This check used its own reader of the offload bundles and its own implementation of the key.
   - The `.text` section is byte-identical in all 24 pairs of code objects.
   - All 807 function definitions are byte-identical, each at the same offset in `.text`. These are the 745 functions of the five programs with one code object and the 62 definitions of the 26 functions of `benchmark_vfhf_max_speedup_hip`.
   - It found the same pairs, renamed functions and removed arguments as above.
3. **Counts.** Every count of the two outputs and the tables adds up. The names were also counted again from the executables:
   - the 264 renamed functions are exactly those with `fk::Sub`, `fk::Mul` or `fk::Div` in their name at 64ad9195;
   - `,fk::BinaryType` occurs 880 times in the names at 64ad9195, all of them from `fk::Add`, and 3,760 times at 1da77899, so 2,880 more;
   - each renamed name at 1da77899 becomes the name at 64ad9195 when these arguments are removed;
   - only the six executables above hold device code.
4. **What the instruction text leaves out.**
   - The kernel metadata is identical for all 759 kernels apart from the names: register and spill counts, LDS and scratch sizes, the layout of the kernel arguments, workgroup and wavefront sizes.
   - The kernel descriptors are identical apart from `kernel_code_entry_byte_offset` in 344 of them: all 123 of horizontal fusion, all 215 of RPP preprocessing and the 6 of the first vfhf code object. The longer names at 1da77899 move `.rodata` and `.text` by different amounts, and each entry still points to its kernel, at the same offset in `.text`.
   - `.rodata` holds only the descriptors, and there is no `.data`.
   - `.bss` is the one-byte `__hip_cuid_<hash>` object, whose name is derived from the build path.

## Not covered here

- **The CUDA programs.** The NVIDIA session compared their SASS on an RTX 4090 for #431 itself, with the kernels paired the same way. It built against 64ad9195 and against the branch head c6f7473, which has the tree of 1da77899. 1,790 of 1,790 kernels of 34 programs are identical (FKL #431's description, "SASS identity check").
- **The CPU builds.** Their results are kept in the repository only.
- **The host code** of the executables, apart from the names and strings listed above.
- **Run times.** Nothing was run.

## Files

- `devcode-compare.txt`: the output of the comparison. It has one line per program. The line of each program with device code is followed by a `names:` line with the pairs, the renamed functions, the removed arguments and the key collisions.
- `devcode-compare-labels-as-symbols.txt`: the stricter form (control 1).
- `functions/<program>.tsv`: one line per key. Each line gives the key, the function's name at 64ad9195 and at 1da77899, its sha256 at each pin, and whether they are the same. There is one file for each program with device code. For `benchmark_vfhf_max_speedup_hip` each sha256 is given per code object, as `<position>:<sha256>`.
- `binaries-sha256.txt`: the sha256 of the whole executables (control 2).
- The comparison script is `../../../scripts/devcode_compare_renamed.py`. It is `scripts/devcode_compare.py` with the functions paired by a key: the demangled name without whitespace and without the default 4th argument `fk::BinaryType` of `fk::Sub`, `fk::Mul` and `fk::Div` (the NVIDIA session's `drop_default`).
