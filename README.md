# fkl-benchmarks

Benchmarks and measurement tooling for the Fused Kernel Library (FKL) paper (IEEE TPDS, revision 2) and its two wrapper
libraries, cvGPUSpeedup-FastOpenCV and FastNPP. Apache-2.0 (see `LICENSE`); moved code keeps its original licence headers.
The libraries are pinned as submodules; the benchmarks were moved here from their `benchmarks/` directories (see
`PROVENANCE.md` for the source commit and checksum of every moved file, and `git log -p` for what was changed).

## Layout

| folder | content |
|---|---|
| `external/` | submodules, pinned: `fkl` (FusedKernelLibrary, LTS-C++17), `cvGPUSpeedup` (cvGPUSpeedup-FastOpenCV, LTS-C++17), `FastNPP` (LTS-C++17) |
| `cmake/` | the CUDA 13.4 pin, target helpers (every target declares its back-ends `cu` / `hip` / `cpp`), build info |
| `common/` | shared harness: `main`, one-block CSV writers (C++ and Python), `system_info.h` (the CSV header), device-attribute dump |
| `fkl-self/` | FKL against itself: horizontal fusion (+ CUDA Graphs), thread fusion, latency hiding, instructions per operation, compile-time seed |
| `opencv-cuda/` | cvGPUSpeedup against OpenCV-CUDA: vertical fusion chains (generated), data size, CPU time, wrapper overhead, batched DNN preprocessing |
| `opencv-cpu/` | FKL CPU back-end against OpenCV on the CPU (`-DFKLB_BUILD_OPENCV_CPU=ON`; needs only OpenCV core, imgproc, imgcodecs) |
| `npp/` | FastNPP against NPP |
| `handwritten-cuda/`, `triton/`, `torch-inductor/`, `pytorch-attention/` | baselines added in revision 2 (to come) |
| `scripts/` | SASS/PTX tools, system-info dump, one script per figure (to come) |
| `results/` | committed CSVs: `results/<system>/<experiment>/`, plus `results/<system>/system-info/` |

## Build

Everything is built with CUDA 13.4 and strict C++17. Source the environment file first, then use a preset:

```bash
export FKLB_CUDA_HOME=/usr/local/cuda-13.4 CUDA_VISIBLE_DEVICES=0   # the CUDA 13.4 toolkit root and the GPU (nvidia-smi index)
source scripts/env-cuda13.4.sh                                       # PATH, LD_LIBRARY_PATH, PCI_BUS_ID order
git submodule update --init                 # external/fkl, external/cvGPUSpeedup, external/FastNPP
cmake --preset rtx4090 -DOPENCV_DIR=<OpenCV install prefix>
cmake --build build/rtx4090 -j 16
```

- The presets (`rtx4090`: sm_89, `rtx4090-fmad-off`: the same with `-fmad=false` for the numerics experiment, `rtx5050`:
  sm_120, flash attention only) take the `nvcc` from `$FKLB_CUDA_HOME`, and the configure fails if the nvcc is not 13.4
  (`FKLB_REQUIRED_NVCC`), so a different CUDA on `PATH` or in `/usr/local/cuda` can never be picked up silently.
- A `CMakeUserPresets.json` (ignored by git) can inherit a preset and set `OPENCV_DIR` for the machine.
- Targets are named `<benchmark>_<backend>` (`benchmark_latency_hiding_cu`, `..._cpp`, `..._hip`). A benchmark builds only the
  back-ends it declares: the `opencv-cuda/` and `npp/` ones are `cu` only; the `fkl-self/` ones also build for `cpp` and `hip`.
  The HIP back-end needs `amdclang++` and an AMD GPU (it was not built on the NVIDIA machine).
- The vertical fusion chains generate one translation unit per experiment (`NUM_EXPERIMENTS`, 200 by default = the paper's
  19,902 operations; nvcc writes the phase times of each to `build/.../kernelN.csv`) and take hours to build. The three used
  by the paper (`mul_add1`, `mul1`, `mul_add`) are built by default; `-DFKLB_BUILD_VF_EXTRA=ON` adds the other four.

## Run

Run a benchmark from the directory where its CSV should go (`ctest` does it from `build/<preset>/run/<target>/`). Every CSV
starts with `# key: value` lines: the commits of this repository and of the three submodules (with `-dirty` when the working
tree is not clean), nvcc, flags, the GPU, driver, CUDA driver API and runtime versions and the clocks. Read them with
`pandas.read_csv(path, comment='#', skipinitialspace=True)`. A CSV has one block per type combination; the warm-up sweep
that used to write a first block of junk is no longer written.

Serialise the GPU jobs: one benchmark at a time on the GPU being measured. `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES` (set by the environment file) select it.

## Environment of the reported results

- GPU: RTX 4090 (sm_89, `results/rtx4090/system-info/`); the RTX 5050 (sm_120) is used for the flash-attention experiment only.
- Toolkit: CUDA 13.4.3 (nvcc 13.4.92); driver 610.43.02 (nvidia-smi reports "CUDA 13.3" for the driver; the 13.4 runtime runs
  on it through CUDA minor-version compatibility, and the CSV header records both the driver API and the runtime version).
- OpenCV 4.14.0 with the contrib CUDA modules, built against CUDA 13.4 for sm_89 only (`CUDA_ARCH_BIN=8.9`, no PTX), as a
  system build outside this repository. Consequence: the `opencv-cuda/` benchmarks run on the RTX 4090 only.
- Host: AMD Ryzen 9 5950X, g++ 15.2, Ubuntu 26.04.
- Python baselines: `venv-torch` (see `PROVENANCE.md`): torch 2.14.1+cu130, triton 3.8.0 using the 13.4 `ptxas`, flash-attn 2.8.3.

## Fixes made while moving the benchmarks

Latency hiding (`StaticLoop<X, 0>` runs X once, so the labels 1, 21, 41... ran 20, 20, 40... operations), the shadowed
bit-exact check and the mismatching `BatchRead` of the wrapper-overhead benchmark, the junk first CSV block, and the sweep
sizes of horizontal fusion (120 x 10), instructions per operation (100 points, 500 instructions) and latency hiding (60 points).
Details are in the commit messages.
