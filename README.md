# fkl-benchmarks

Benchmarks and measurement tooling for the Fused Kernel Library (FKL) paper (IEEE TPDS, revision 2) and its two wrapper libraries.
Licensed under Apache-2.0 (see `LICENSE`); moved code keeps its original licence headers.

Status: bootstrap. The first commit holds the benchmark sources exactly as they were in the library repositories (see
`PROVENANCE.md`); the following commits make them build from this repository and fix the known harness bugs.

## Layout

| folder | content |
|---|---|
| `external/` | git submodules, pinned: `fkl` (FusedKernelLibrary, LTS-C++17), `cvGPUSpeedup` (cvGPUSpeedup-FastOpenCV, LTS-C++17), `FastNPP` (LTS-C++17) |
| `cmake/` | target helpers: every benchmark target declares its back-ends (`cu`, `hip`, `cpp`) explicitly |
| `common/` | shared harness: `main`, CSV writer, system-info dump, SASS/PTX tooling |
| `fkl-self/` | FKL against itself: horizontal fusion (+ CUDA Graphs), latency hiding, instructions per op, thread fusion |
| `opencv-cuda/` | cvGPUSpeedup vs OpenCV-CUDA: vertical fusion, VF + HF, data size, CPU time, wrapper overhead |
| `opencv-cpu/` | FKL CPU back-end vs OpenCV CPU |
| `npp/` | FastNPP vs NPP |
| `handwritten-cuda/`, `triton/`, `torch-inductor/`, `pytorch-attention/` | baselines added in revision 2 |
| `scripts/` | one script per figure, reading `results/` |
| `results/` | committed CSVs, `results/<system>/<experiment>/`, each with a system-info header |

## Moved files

`PROVENANCE.md` lists every file taken from a library repository: source repository, pinned commit, last change and checksum of the verbatim copy.
