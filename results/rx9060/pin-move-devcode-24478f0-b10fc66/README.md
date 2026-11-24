# Device code of the HIP programs at FKL 24478f0 and at b10fc66 (the merge of FKL #433)

FKL #433 batches LinearFilterDPP, MedianFilterDPP and the four Quad DPPs, and removes the image sizes from their details (task B). It was merged
into LTS-C++17 on 2026-11-24 at 22:02:02Z as b10fc6609426bd18020b9213c7196b528fc677a8. The merge's parents are 24478f0 and 4b11bc6, #433's
head, and its tree is 4b11bc6's (62024145c9f1a1a8aa55033d182e87bbdd448827).

fkl-benchmarks #60 (branch `pin-fkl-433`, head 2160c70d34f980b87601bebf7ad16658184a019f, one commit on a8f9daa) moves `external/fkl` from
24478f0 to b10fc66. It also ports the details of the programs that call LinearFilterDPP: in the HIP programs these are `rpp/e18_box.h`,
`rpp/e19_batched_box.h` and `rpp/e19_crop_sweep.h`, changed by `../task-b-gpu-window/fklb-after-details-port.diff` (sha256 ca8cc3d6...).
This folder is the HIP evidence for that move.

The check: every HIP program of this repository was built twice, once at a8f9daa with FKL 24478f0 (A) and once at 2160c70 with FKL b10fc66
(B), with the same tool and flags. Their gfx1200 device code was compared function by function. For the two E19 programs, the x86-64 host
code was also compared (`e19-host-code.md`). Nothing was run. This was done on 2026-11-24 on the RX 9060 machine (`../README.md`); the
builds need no GPU.

## Result: only the 5 linearFilterDPPKernel instantiations differ

| HIP program | functions in A / B | renamed | identical | different | in one build only |
|---|---|---|---|---|---|
| `benchmark_horizontal_fusion_hip` | 125 / 125 | 0 | 125 | 0 | 0 |
| `benchmark_thread_fusion_hip` | 153 / 153 | 0 | 153 | 0 | 0 |
| `benchmark_latency_hiding_hip` | 63 / 63 | 0 | 63 | 0 | 0 |
| `benchmark_numinstructions_x_operation_hip` | 187 / 187 | 0 | 187 | 0 | 0 |
| `benchmark_rpp_preprocessing_hip` | 247 / 247 | 0 | 247 | 0 | 0 |
| `benchmark_vfhf_max_speedup_hip` (19 code objects, paired by position) | 26 / 26 | 0 | 26 | 0 | 0 |
| `e18_box_hip` | 6 / 6 | 0 | 3 | **3** | 0 |
| `e19_batched_box_hip` | 5 / 5 | 0 | 4 | **1** | 0 |
| `e19_crop_sweep_hip` | 20 / 20 | 0 | 19 | **1** | 0 |
| `batch_build_compilation_time_hip` | no device code in either build | | | | |
| **10 programs** | **832 / 832** | **0** | **827** | **5** | **0** |

- **The 5 that differ** are the `fk::linearFilterDPPKernel` instantiations:
  - 3 in `e18_box_hip`, with `LinearFilterDPPDetails<Float3, 16, 8, 7, 7>`;
  - 1 in each E19 program, with `LinearFilterDPPDetails<float, 16, 8, 7, 7>` (the kernel of the fkl_loop row).

  They are the same 5 as in FKL #433's description, where 4b11bc6 was checked against 24478f0. The details lose `width` and `height`, and
  the kernels take each image's size from the read.
- **Everything else is identical.** This includes:
  - the batched `MorphologyDPP` kernel of the E19 fkl_batch row;
  - the kernels of E18 that are not LinearFilterDPP;
  - the seven programs of `../pin-move-devcode-1da77899-2c42a80/` (#45), which have the same 801 functions as there.
- **No renames and no key collisions.** No function is renamed, none is in only one build, and there are no key collisions.
- **With the branch targets as separate symbols** (`--labels-as-symbols`), 2,524 of 2,610 symbols are identical. Every one of the 86 that
  differ lies inside one of the 5 kernels, in A and in B (`labels-as-symbols-owners.txt`, from `step0/label_owners.py`).
- **The whole executables differ** (`binaries-sha256.txt`). The known causes are the source paths, the `__hip_cuid_<hash>` and
  `__hip_gpubin_handle_<hash>` names (both derived from the build folders) and the FKL commit string that the programs record.

## The three rpp programs against the binaries of the task B GPU window

This compares `e18_box_hip`, `e19_batched_box_hip` and `e19_crop_sweep_hip` with the binaries whose outputs `../task-b-gpu-window/` (#58)
checked, on the GPU, before and after #433:

| | against | functions | identical |
|---|---|---|---|
| B (2160c70, FKL b10fc66) | `fklb-after/`: b4fb594 with the details-port diff, FKL 4b11bc6 | 31 | 31 |
| A (a8f9daa, FKL 24478f0) | `fklb-before/`: b4fb594, FKL 24478f0 | 31 | 31 |

The device code built from the committed sources of #60 is therefore the code whose outputs #58 found equal before and after: the E18 box
digests, and the dumps of E19 and the E19 crop sweep. The sources also agree:
- The three rpp files at 2160c70 are byte for byte b4fb594's files with that diff applied.
- From b4fb594 to a8f9daa, no build source changed, only `results/`.

## The E19 host code

`e19-host-code.md` compares the x86-64 code of `e19_batched_box_hip` and `e19_crop_sweep_hip`, A against B, function by function. It also
follows what each row's timed call executes.

No function reached from the timed call of the **fkl_batch** row differs, in either program and in either comparison mode. The same holds
for the **rpp** row and for the timing harness (`fklb::timeIt`, `fklb::timePreparedLaunch`). The functions that differ are fkl_loop's
(`executeLinearFilter` and its lambda) and the host stub of fkl_loop's kernel. In the stricter mode, `fklb::writeSystemInfoHeader` also
differs, because it writes the FKL commit into the CSV header before the timing.

## Builds

| build | sources | executables | failed | warnings | time |
|---|---|---|---|---|---|
| A, HIP | fkl-benchmarks a8f9daa, FKL 24478f0 | 15: 10 `_hip` + 5 `_cpp` | 0 | 0 | 2:57 |
| B, HIP | fkl-benchmarks 2160c70, FKL b10fc66 | 15: the same 15 names | 0 | 0 | 2:56 |
| A, opencv-cpu | fkl-benchmarks a8f9daa, FKL 24478f0 | 2: `e5_preprocessing_a_cpp`, `e5_preprocessing_c_cpp` | 0 | 0 | 0:20 |
| B, opencv-cpu | fkl-benchmarks 2160c70, FKL b10fc66 | the same 2 | 0 | 0 | 0:20 |

**The HIP builds** use `step0/pin_build.sh`, the script of #45, byte for byte.
- **Sources:**
  - fkl-benchmarks: a `git archive` of each commit into its own folder.
  - FKL: `external/fkl` is a symbolic link to a clean FKL worktree at that build's commit.
  - FastRPP: `external/FastRPP` is a link to a clean checkout of efbf009, the pin at both commits.
- **Tools:** amdclang++ (AMD clang 24.0.0git, ROCm 10.1.0), RPP 3.2.0, gfx1200, Release, `ninja -j2`, with `FKLB_BUILD_RPP=ON` and the CUDA,
  NPP, OpenCV and hand-written suites off.
- **The 10 `_hip` executables** are every target with a hip back-end at a8f9daa: the 7 of #45, plus `e18_box`, `e19_batched_box` and
  `e19_crop_sweep`.
- **The 5 `_cpp` executables** are fkl-self's `batch_build_compilation_time`, `benchmark_horizontal_fusion`, `benchmark_latency_hiding`,
  `benchmark_numinstructions_x_operation` and `benchmark_thread_fusion`.
- **The thread-pool submodule** of FKL was not checked out during the HIP builds, as in #45. Only the CPU_MULTITHREAD benchmarks of
  opencv-cpu use it.

**The opencv-cpu builds** use `step0/ocv_build.sh`. They are a build check of `opencv-cpu/` (`FKLB_BUILD_OPENCV_CPU`), which is off by
default and was in no earlier pin-move build.
- They use the same archived sources, only the cpp back-end and only that suite.
- The FKL worktree has the thread-pool submodule at 27e733f, checked out after the HIP builds.
- OpenCV is the machine's 4.14.0 CPU build (core and imgproc; imgcodecs, which is optional, is not in it), with amdclang++ 24.
- At both pins its two targets build. Their code was not compared and they were not run.

`builds.txt` has the commits and the states of the checkouts of the four builds.

**The comparisons** use `scripts/devcode_compare_renamed.py`, as in #45, with `ROCM_PATH=/opt/rocm/core-10.1`, and `step0/host_compare.py`
with GNU objdump 2.46. `step0/run_checks.sh` writes every output of this folder from the four builds and the binaries of `../task-b-gpu-window/`.

## Not covered here

- **The CUDA programs.** They are the NVIDIA session's check of the same move (#60).
- **Run times.** Nothing was run.
- **The CPU code** of the `_cpp` and opencv-cpu programs was not compared.
- **The Pins note of `../README.md`** is not changed here. It changes after #60 is merged, as with #45.

## Files

- `devcode-compare.txt` and `devcode-compare-labels-as-symbols.txt`: the two outputs of the device code comparison, A against B.
- `functions/<program>.tsv`: one line per key, for each program with device code.
- `labels-as-symbols-owners.txt`: the function that holds each symbol that differs with `--labels-as-symbols`.
- `devcode-head-vs-task-b-window-after.txt` and `devcode-base-vs-task-b-window-before.txt`: the three rpp programs against the task B window's binaries.
- `e19-host-code.md` and `e19-host-code/`: the host code comparison of the two E19 programs.
- `binaries-sha256.txt`: the sha256 of the 34 executables of the four builds.
- `builds.txt`: the commits and the summary lines of the four builds.
- `step0/`: `pin_build.sh` (from #45), `ocv_build.sh`, `run_checks.sh`, `host_compare.py` and `label_owners.py`.
