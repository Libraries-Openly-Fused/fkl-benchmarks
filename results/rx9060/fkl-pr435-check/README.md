# FKL #435 (task C) on the RX 9060: FKL's HIP tests and the outputs of every DPP, before and after

FKL #435 (draft, https://github.com/Libraries-Openly-Fused/FusedKernelLibrary/pull/435) makes every DPP run through
`fk::executeOperations<DPP>` with one kernel, `launchDPP_Kernel`. It implements Oscar's decisions in the coordinator session of 2026-11-26
(task C). This record is the AMD machine's part of its verification:
- before = FKL LTS-C++17 b10fc6609426bd18020b9213c7196b528fc677a8;
- after = the #435 head 48fbf3f08e7551c3eb70ac2302d8ebc15ecad645.
  - 48fbf3f is e322cad plus one commit, which changes only a CUDA utest (utest_flash_attention_mma.h): the bound of its new split-KV checks, which #435's CUDA CI jobs failed at e322cad.
  - The HIP code and the harness are the same at both.

The CUDA side (tests, SASS, outputs on the RTX 4090) is the NVIDIA session's check.

## What is here

- `harness/dpps_before_after.cpp`, `harness/build.sh` and `harness/compare.sh`: the output harness.
  - One source, built against before (the entry functions) and after (`executeOperations<DPP>`). The calls of the two forms are in its adaptors.
  - It runs 24 cases on ParArch::CPU and CPU_MULTITHREAD, and in a HIP build with `HARNESS_GPU=1` also those 24 plus 1 (DivergentBatchTransformDPP) on GPU_AMD.
  - **The cases:**
    - LinearFilterDPP: 4 cases, with the box filter's constant coefficient read, a BatchRead of crops of different sizes and a 3D read.
    - MedianFilterDPP: 3 cases, one with batched uchar3 crops.
    - MorphologyDPP: 5 cases, with E19's fkl_batch form (Crop, Resize, box blur) and a float3 box blur.
    - The four Quad DPPs: 2 each, a scalar 2D case and batched vector crops with run-time sizes.
    - TransformDPP: 4 cases, the three kernels of its run-time choice included.
  - Every output buffer is pre-filled with a sentinel and written whole to a file.
  - `compare.sh` compares the before and after runs byte for byte.
- `cpu-run/`: the harness's CPU-only builds (amdclang++ as C++, no HIP: nothing touched the GPU), run before the window.
  - Result: **48 of 48 outputs identical** (24 cases × CPU and CPU_MULTITHREAD), each written (`compare.txt`).
  - The batched cases leave exactly the 439 values outside their three images untouched (23 × 19 × 3 − (21 × 13 + 23 × 12 + 17 × 19)).
- `window.sh`: the GPU window, with its guards. It runs FKL's HIP tests at the head (`ctest -R "_hip$" -E "^benchmark_" --stop-on-failure --timeout 300`), then the harness's HIP builds A and B. `manifests/` holds the sha256 of every binary it runs.
- `window/`: the window's outputs (see below).

## Builds

- **Before:** the FKL worktree at b10fc66 (clean, thread-pool 27e733f).
- **After:** the FKL worktree at 48fbf3f (clean, thread-pool 27e733f).
- **The test build for the window:** `tmp/vbuild/pr435-head-hip`, configured from the clean tree.
  - It was first configured at e322cad, then re-configured and rebuilt at 48fbf3f, 0 failed.
  - Its configure line reads `0.1.14-LTS-C++17-executeoperations-dpps-48fbf3f`.
  - It uses the ROCm CI's flags: amdclang++ 24 (ROCm 10.1) for CXX and HIP, `-Werror=c++20-extensions -pedantic-errors -Wno-gnu-anonymous-struct`, Release, gfx1200, ENABLE_CPU and ENABLE_CPU_MULTITHREAD.
- **The harness:** `harness/build.sh <tree> <binary> cpu|hip`, which refuses a tree with changes and compiles its HEAD into the binary.

## The window

Guard output (GUARD_ONLY=1) is in `manifests/guard-only.txt`. The window runs with:

```
TEST_BUILD=/home/cudeiro/rocm-fkl-benchmarks/tmp/vbuild/pr435-head-hip \
TEST_MANIFEST=<this folder>/manifests/test-manifest.sha256 \
HARNESS_MANIFEST=/home/cudeiro/rocm-fkl-benchmarks/tmp/taskc/phase2/harness-bins/manifest.sha256 \
CONFIGURE_LOG=/home/cudeiro/rocm-fkl-benchmarks/tmp/vbuild/pr435-head-hip.configure-48fbf3f.log \
BEFORE_COMMIT=b10fc6609426bd18020b9213c7196b528fc677a8 AFTER_COMMIT=48fbf3f08e7551c3eb70ac2302d8ebc15ecad645 \
./window.sh <this folder>/window
```

- `manifests/harness-manifest.sha256` is a copy of the harness manifest.
- `cpu-run/binaries.sha256` holds the sha256 of the CPU-only harness binaries.

## The window's result (granted by the coordinator; ran 2026-11-26 14:51-14:53 +01:00, RX 9060 gfx1200, ROCm 10.1)

The console is `window/window-console.txt`.
- **Start:** 14:51:25. It waited 90 s for an idle runner (no Runner.Worker, load below 0.8), then ran without waiting.
- **FKL's HIP tests at 48fbf3f:** `ctest -R "_hip$" -E "^benchmark_" --stop-on-failure --timeout 300 -j1`, exit 0, **100% tests passed, 0 tests failed out of 52** (`window/tests/ctest.txt`, 9.0 s).
- **The harness on the GPU:** A (b10fc66) and B (48fbf3f), each exit 0 with 73 outputs written. That is 24 cases × CPU, CPU_MULTITHREAD and GPU_AMD, plus DivergentBatchTransformDPP on GPU_AMD.
- **compare.sh:** exit 0. **73 outputs, 73 identical, 0 different** (`window/harness/compare.txt`): every GPU_AMD output of the after build is byte for byte the before build's.
- **Not part of the check, both builds alike:**
  - CPU_MULTITHREAD gives the same bytes as CPU in all 24 cases.
  - GPU_AMD gives the same bytes as CPU in 22. The two exceptions are float cases, linear-uchar3-box3 and morph-batch-crop-resize-boxblur1, in A and B alike: the GPU and the CPU compilers round float arithmetic differently.
- **No timing:** nothing was timed, and nothing else used the GPU.
