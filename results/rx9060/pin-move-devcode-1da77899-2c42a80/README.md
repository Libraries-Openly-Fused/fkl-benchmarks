# Device code of the HIP programs at FKL 1da77899 and at 2c42a80 (FKL #432's head)

FKL #432 adds the HIP (`ParArch::GPU_AMD`) versions of the 7 image-processing DPPs of #431. Its head, 2c42a80, is two commits on 1da77899, the
pin of this repository. This folder is the HIP evidence for a move of the pin to #432. It is a draft: the move waits for Oscar's merge of #432, and
the merge commit's tree is expected to be that of 2c42a80.

The check: every HIP program of this repository was built twice, once against each FKL commit, from the same sources with the same tool and
flags, and their gfx1200 device code was compared function by function. Nothing was run. Done on 2026-11-13 on the RX 9060 machine
(`../README.md`); the build needs no GPU.

## Result: the device code is the same

| HIP program | functions at 1da77899 / 2c42a80 | renamed | identical | different | in one build only |
|---|---|---|---|---|---|
| `benchmark_horizontal_fusion_hip` | 125 / 125 | 0 | 125 | 0 | 0 |
| `benchmark_thread_fusion_hip` | 153 / 153 | 0 | 153 | 0 | 0 |
| `benchmark_latency_hiding_hip` | 63 / 63 | 0 | 63 | 0 | 0 |
| `benchmark_numinstructions_x_operation_hip` | 187 / 187 | 0 | 187 | 0 | 0 |
| `benchmark_rpp_preprocessing_hip` | 247 / 247 | 0 | 247 | 0 | 0 |
| `benchmark_vfhf_max_speedup_hip` (19 code objects, paired by position) | 26 / 26 | 0 | 26 | 0 | 0 |
| `batch_build_compilation_time_hip` | no device code in either build | | | | |
| **7 programs** | **801 / 801** | **0** | **801** | **0** | **0** |

- With the branch targets as separate symbols (`--labels-as-symbols`), 2,245 of 2,245 symbols are identical.
- No function is renamed and there are no key collisions.
- The whole executables differ (`binaries-sha256.txt`). The known causes are the source paths, the `__hip_cuid_<hash>` names (both derived from
  the build folders) and the FKL commit string that the programs record.
- None of the programs includes a header that #432 changes: it adds `FK_HIP_BACKEND` blocks to the 7 DPP headers and `neighborhood.h`, and
  widens the guards of the 7 DPP tests.

**The same check at the previous sources.** `earlier-8f4a95b/` is the check made for #432 before it was opened. It used fkl-benchmarks 8f4a95b
(before the FastRPP variants of #44) and FKL c0154e6, the commit that became 2c42a80 with only its message changed (the same tree,
153ebf2b6d4d9866189c9416d7330a560ee58576). Its result: 771 of 771 functions identical, 1,795 of 1,795 with `--labels-as-symbols`, 0 renamed.

## What was compared

- **Sources:** fkl-benchmarks 8c1fa98 (main), extracted with `git archive` into one folder per build. In each folder, `external/fkl` is a
  symbolic link to that build's clean FKL checkout, and `external/FastRPP` is a symbolic link to a clean checkout of efbf009, the pin.
- **FKL commits:** 1da778992756e0fa08ec289225fe18dca134412c, the pin; and 2c42a80e12168f4183ba835cdde353cc1e33be07, #432's head (ed122be and
  2c42a80 on 1da77899). Both use the `thread-pool` submodule at 27e733f with the same files.
- **Build:** `step0/pin_build.sh`, the configuration of the earlier pin moves: amdclang++ (AMD clang 24.0.0git, ROCm 10.1.0), RPP 3.2.0, gfx1200,
  Release, `ninja -j2`. Each build produced 12 executables with no failures and no warnings.
- **Comparison:** `scripts/devcode_compare_renamed.py`, as in `../pin-move-devcode-64ad9195-1da77899/`, with `ROCM_PATH=/opt/rocm/core-10.1`.

## Not covered here

- **The CUDA programs.** The NVIDIA session checks them for the same move.
- **The CPU builds.** Their results are kept in the repository only.
- **Run times.** Nothing was run.

## Files

- `devcode-compare.txt` and `devcode-compare-labels-as-symbols.txt`: the two outputs.
- `functions/<program>.tsv`: one line per key, for each program with device code.
- `binaries-sha256.txt`: the sha256 of the whole executables.
- `earlier-8f4a95b/`: the outputs and tables of the earlier check.
- `step0/pin_build.sh`: the build script.
