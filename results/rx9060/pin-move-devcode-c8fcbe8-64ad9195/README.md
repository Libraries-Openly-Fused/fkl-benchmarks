# Device code of the HIP programs at the two pins, c8fcbe8 and 64ad9195

On 2026-11-07 the pin of `external/fkl` moved from c8fcbe8 to 64ad9195, the final library version (bdf53a8). The RX 9060 results of `results/rx9060/` were measured at c8fcbe8 or at earlier pins.

The check: every HIP program of this repository was built twice, once against each of the two FKL commits, from the same sources with the same tool and flags. Their gfx1200 device code was then compared function by function. Nothing was run. This was done on 2026-11-07 on the RX 9060 machine (`../README.md`), and the build needs no GPU.

## Result: the device code is the same at both pins

| HIP program | functions at c8fcbe8 / 64ad9195 | identical | different | in one build only |
|---|---|---|---|---|
| `benchmark_horizontal_fusion_hip` | 125 / 125 | 125 | 0 | 0 |
| `benchmark_thread_fusion_hip` | 153 / 153 | 153 | 0 | 0 |
| `benchmark_latency_hiding_hip` | 63 / 63 | 63 | 0 | 0 |
| `benchmark_numinstructions_x_operation_hip` | 187 / 187 | 187 | 0 | 0 |
| `benchmark_rpp_preprocessing_hip` | 97 / 97 | 97 | 0 | 0 |
| `batch_build_compilation_time_hip` | no device code in either build | | | |
| **6 programs** | **625 / 625** | **625** | **0** | **0** |

The host code of the programs differs between the two pins:
- #423 added the chain checks, which run at compile time.
- #427 added the crop checks, which run on the host.

## What was compared

**Sources.** fkl-benchmarks 0fb1f9f97d5c8c9ad0f9efcbb5a9c8ac2689f2c3. The program sources did not change from there to a5dd9fa, which includes the pin move of bdf53a8: `git diff 0fb1f9f97 a5dd9fa -- CMakeLists.txt CMakePresets.json cmake common fkl-self rpp` is empty. The cleanup commits after a5dd9fa changed `common/main.cpp` and `common/csv_writer.h` (host code of every program) and added files to `common/`; they are not part of this comparison, which isolates the change of the pin. The sources were extracted with `git archive`, and `external/fkl` was a symbolic link to each FKL checkout in turn.

**FKL commits.** Each was a clean detached checkout:
- c8fcbe816d629377886cbdd19c83527f37ccfc04, the pin until 2026-11-07;
- 64ad9195cd785d0c93a70b35e02d5a6c7c4a5048, the merge of #427 into LTS-C++17.

Both use the `thread-pool` submodule at 27e733f50f7f40f171fd84655e05d5acf1fdfbee, with the same files on both sides. 64ad9195 is c8fcbe8 plus four merges:
- #422: a test only;
- #426: a CI workflow;
- #423: the chain checks of `executeOperations`, a `BATCH` member of `BatchWrite`, and the declared output type of `IsEven` for vectors;
- #427: the strides and checks of `Ptr::crop`.

**Programs.** The six targets with a `hip` back-end: `BACKENDS ... hip` in `fkl-self/CMakeLists.txt` (5 programs) and in `rpp/CMakeLists.txt` (1). The other directories have no HIP target.

**Build.** amdclang++ (AMD clang 24.0.0git, ROCm 10.1) with RPP 3.2.0, the same for both builds:

```
cmake -S <sources> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DFKLB_ENABLE_CUDA=OFF -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF \
  -DFKLB_BUILD_OPENCV_CPU=OFF -DFKLB_BUILD_HANDWRITTEN=OFF -DFKLB_BUILD_RPP=ON
ninja -C <build> -j2
```

The two CMake caches are equal apart from the paths. So are the compile commands, for example the HIP object of `benchmark_horizontal_fusion_hip`: `amdclang++ ... -O3 -DNDEBUG -std=c++17 --offload-arch=gfx1200 -Werror=c++20-extensions -ftemplate-depth=1000 -x hip -c .../launcher.hip`. Each build produced 11 executables (6 `_hip` and 5 `_cpp`) with no failures.

**Comparison.** `scripts/devcode_compare.py`, run with `ROCM_PATH=/opt/rocm/core-10.1`. For each executable it does three things:
1. It extracts the gfx1200 code object from `.hip_fatbin` (`llvm-objcopy`, `clang-offload-bundler`).
2. It disassembles it with `llvm-objdump -d --symbolize-operands`.
3. It compares the sha256 of the instruction text of each function. Comments, which hold the addresses and the encodings, are removed, and the `__hip_cuid_<hash>` names are normalised.

`llvm-objdump` names the local branch targets `L<n>`, numbered per code object. By default each one belongs to the function that contains it, and its number is not compared. The `...` that `llvm-objdump` prints for the zero bytes after the last instruction of a function (the padding before the next one) is not compared either; a `...` inside a function would be. The tables in `functions/` were computed again with the script of the RPP rewrite (fkl-benchmarks #32), which leaves that padding out of the hashes: their hash values changed, the verdicts did not (625 of 625 the same, and 1,505 of 1,505 with `--labels-as-symbols`).

```
python3 -I scripts/devcode_compare.py <build c8fcbe8>/bin <build 64ad9195>/bin --glob '*_hip' --table functions > devcode-compare.txt
python3 -I scripts/devcode_compare.py <build c8fcbe8>/bin <build 64ad9195>/bin --glob '*_hip' --labels-as-symbols > devcode-compare-labels-as-symbols.txt
```

## Controls

1. **The stricter form.** With the branch targets as separate symbols, numbers included (`--labels-as-symbols`), 1,505 of 1,505 symbols are identical.
2. **The two builds are different.** The whole executables differ in all six programs (`binaries-sha256.txt`), so the two builds used different FKL headers. The differences are in the host code: the changes of #423 and #427 listed above, and the source paths, which differ between the two build folders.

## Not covered here

- **The CUDA programs.** They were compared on the RTX 4090: `../../rtx4090/fkl-pr423-check/` and `../../rtx4090/fkl-pr427-check/`.
- **The CPU builds.** Their results are kept in the repository only.
- **Run times.** Nothing was run.

## Files

- `devcode-compare.txt`: the output of the comparison, one line per program.
- `devcode-compare-labels-as-symbols.txt`: the stricter form (control 1).
- `functions/<program>.tsv`: one line per function, giving the function, its sha256 at c8fcbe8 and at 64ad9195, and whether they are the same. There is one file for each program with device code.
- `binaries-sha256.txt`: the sha256 of the whole executables (control 2).
- The comparison script is `../../../scripts/devcode_compare.py`.
