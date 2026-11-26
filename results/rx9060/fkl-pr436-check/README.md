# FKL #436 (task C on main) on the RX 9060: FKL's HIP tests and the outputs of every DPP, before and after (prepared, window not run yet)

FKL #436 (draft, https://github.com/Libraries-Openly-Fused/FusedKernelLibrary/pull/436) ports task C to FKL main: every DPP runs through
`fk::executeOperations<DPP>`, with one kernel, `launchDPP_Kernel`. It implements Oscar's task C decisions (LTS-C++17 #435) and his order of
2026-11-26 to port them to main, with his answers M1-M3. This record is the AMD machine's part of its verification:
- before = FKL main ebb3891bea18a152d92dfd5092e117160490e34b;
- after = the #436 head 1df2738e7bd8eba1bfb1a0baf9ad215062d014ca.

The CUDA side is the NVIDIA session's check.

Decision: the coordinator's brief of 2026-11-26 for #436's HIP window, after Oscar's order to port task C to main.

## What is here

- **`harness/`: the output harness.** It is `../fkl-pr435-check/harness/` (#64), adapted to main.
  - `dpps_before_after.cpp`, built by `build.sh` against before (main's entry functions) and after (`executeOperations<DPP>`). The calls of
    the two forms are in its adaptors, and nothing else differs.
  - **Main's differences from the LTS version:**
    - main's LinearFilterDPP, MedianFilterDPP and Quad DPPs are unbatched, their details carry the image size, and the Quads take scalar
      types only. So the LTS batched, 3D and vector Quad cases are 2D cases here: one with static and one with run-time kernel sizes per Quad.
    - LinearFilterDPP accumulates in its value type on main, so the box case is float3 (LTS: uchar3 to float3).
    - main's Quad entry functions return `void`.
    - main has CPU and CPU_MULTITHREAD executors for DivergentBatchTransformDPP, so that case runs on every back-end.
    - The flags are C++20, those of main's tests.
  - **24 cases per back-end:**
    - LinearFilterDPP: 3;
    - MedianFilterDPP: 3;
    - MorphologyDPP: 5, with E19's fkl_batch form (Crop, Resize, box blur) and a float3 box blur;
    - the four Quad DPPs: 2 each;
    - TransformDPP: 4, the three kernels of its run-time choice included;
    - DivergentBatchTransformDPP: 1.

    The cases run on ParArch::CPU and CPU_MULTITHREAD; a HIP build with `HARNESS_GPU=1` also runs them on GPU_AMD.
  - Every output buffer is pre-filled with a sentinel and written whole to a file. `compare.sh` compares the before and after runs byte for
    byte.
- **`cpu-run/`: the harness's CPU-only builds,** run before the window. They are amdclang++ as C++ with no HIP, so nothing touched the GPU,
  and `ldd` shows no HIP library.
  - Result: **48 of 48 outputs identical** (24 cases on CPU and on CPU_MULTITHREAD), each written (`compare.txt`).
  - The sentinel values left are the uchar sentinel 7 where it is also a computed value: 5 in morph-open-uchar-3x5 and 44 in
    morphquad-uchar-3x3, the same in A and B.
- **`window.sh`: the GPU window, with its guards.** It is #64's, plus the GPU-process guard. It runs:
  1. FKL's HIP tests at the head (`ctest -R "_hip$" -E "^benchmark_" --stop-on-failure --timeout 300`, 48 tests);
  2. the harness's HIP builds A and B.

  `launch.sh` holds its arguments. `manifests/` holds the sha256 of every binary it runs.

## Builds

- **Before:** the FKL worktree at ebb3891 (clean, thread-pool 27e733f).
- **After:** the FKL worktree at 1df2738 (clean, thread-pool 27e733f).
- **The test build for the window:** `tmp/cmain/build-hip`.
  - It uses main's ROCm CI configuration: `ENABLE_HIP=ON ENABLE_CUDA=OFF ENABLE_CPU=OFF ENABLE_CPU_MULTITHREAD=ON`, amdclang++ 24 (ROCm
    10.1) for CXX and HIP, Release, gfx1200.
  - It was configured from the clean tree at 1df2738; its configure line reads `0.2.1-main-executeoperations-dpps-1df2738`.
  - 48 `_hip` executables, 0 failed.
- **The harness:** `harness/build.sh <tree> <binary> cpu|hip`, which refuses a tree with changes and compiles its HEAD into the binary.

## The window

- `launch.sh` runs `window.sh` with the test build, the manifests and the two commits, as one detached call after the coordinator's GRANTED.
- **Guards before anything runs:**
  - the test manifest (48 executables) and the harness manifest;
  - every selected test has its executable in the manifest;
  - the test build's tree is at 1df2738, clean, and so it was at configure time;
  - harness A carries ebb3891 and B carries 1df2738;
  - the output folder is new;
  - the amdkfd process list exists.
- **Before every step:** no `Runner.Worker`, a load below 0.8, and an empty amdkfd process list (no other ROCm program, FastRPP's included).
  The first failure ends the window. Nothing is timed.
- **After the window,** without the GPU: `harness/compare.sh` of the two harness runs.
- `GUARD_ONLY=1 launch.sh` passed on 2026-11-26 at 20:58:30+01:00 (`manifests/guard-only.txt`): no `Runner.Worker`, 0 GPU processes.
