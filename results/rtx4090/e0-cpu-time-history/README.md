# CPU-time speedups of 2026 against the original paper: what the histories show (CPU only, nothing was run; written 2026-11-06)

The original paper's CPU-time figure peaks at about 120x (cvGS against OpenCV-CUDA, flat from batch 62) and about 92x (FastNPP against NPP); the runs of 2026 give 419.5x (batch 142) and 288.2x. The smaller differences (NPP with the CPU part 61x then, 109.7x now at batch 150) are in the same direction. **This file lists facts; it does not name a cause.**

## (a) The 2025 data
Not found. No CSV was ever committed in the histories of FusedKernelLibrary, cvGPUSpeedup and FastNPP (`git log --all --diff-filter=A -- '*.csv'` finds nothing), and no file on this machine holds the benchmark headers (`Mean Speedup`, `OpenCV MeanTime`, `NPP MeanTime`, ...) with a date before 2026-10-29 (the earliest are the first runs of this project). The folder name of the 2025 runs, `CUDAGraphs-8-7-2025`, read as 8 July 2025 (day-month-year; it could also be 7 August) is the date on which FastNPP moved its FKL submodule to Beta-0.1.8 (`72ddf8a` 2025-07-08 "Upgrade fkl to Beta 0.1.8 to fix a bug in Executors"), which would make **FKL Beta-0.1.8 (1f33224, tag 2025-05-22) the library of that run** for FastNPP; the cvGPUSpeedup pointer was 1f33224 (Beta-0.1.8) from Alpha-0.0.20 (2025-05-23) to 2025-09-16 (`e52a1f8`, then ab236aa = Beta-0.1.11). This is an inference from dates; the measurement date and the exact commits have to come from the paper side.

## Reading of the 2025 figures by the coordinator (read off the PDFs of the reviewed manuscript, accuracy about +-2, NOT data; added 2026-11-06)
These are values read from the figures of the manuscript under review, not measurements and not files of the 2025 runs.
- 2025 NPP figure (`test_npp_batchresize_x_split3D_No_CPU_Overhead`, and the file with the CPU part), FastNPP against NPP: **without the CPU overhead about 36 at batch 10 and about 146 at batch 150; with the CPU overhead about 24 and about 61.5.**
- 2026 values at the same batch sizes (the old single run of 2026-10-30; the median of the 3 runs after #45 in brackets): without the CPU overhead 36.3 / 143.5 (36.2 / 141.5); with the CPU overhead 30.3 / 110.7 (31.0 / 109.7).
- 2025 CPU-time figure: cvGS against OpenCV-CUDA about 120 at batch 62 and about 110 at batch 152; FastNPP against NPP about 92 at batch 142 and about 90 at batch 152. 2026 (run 1 after #45 for FastNPP, the old single run for cvGS): cvGS 273.9 at batch 62 and 411.5 at 152; FastNPP 288.2 at batch 142 and 287.0 at 152.
What follows from the arithmetic, with no cause claimed: the speedup WITHOUT the host part agrees within about 2 percent at both ends (36 / 36.2, 146 / 141.5), which is consistent with unchanged GPU-side times of both programs (it is not shown by a ratio: equal changes of numerator and denominator would hide), and the difference between 2025 and 2026 is in the measurement that includes the host part, as the arithmetic above suggests. If the NPP time at batch 150 was the same in 2025 as now (1944 microseconds with the CPU part): the GPU time of the fused call would be 13.3 microseconds in 2025 (1944 / 146) and 13.7 now (1944 / 141.5); the call with the host part would take 31.6 microseconds in 2025 (1944 / 61.5) and 17.7 now (1944 / 109.7); the host-side part, the difference of the two, would be about 18 microseconds in 2025 and about 4 now. **No cause is claimed**: the numbers say where the difference is (the host part of the fused call, if the NPP side did not change), not why.

## The harness did not change
- cvGPUSpeedup `benchmarks/benchmark_CPU_OpenCV_vs_cvGS.cu`: created 2025-04-11 (`00bc940`), last change of timed code 2025-04-13 (`c74383b` "Set same initial value as in NPP"), moved to `benchmarks/` 2025-05-20 (`eccfa8d`), and one change in 2026 (`ece01c9` 2026-03-04: the CUDA 11.8 branch of `NUM_EXPERIMENTS` removed). The timing code (`initCPUBenchmark`, `processExecution` in `tests/testsCommon.cuh`) is the same from Alpha-0.0.21 (2025-06-17) to 6d26c65 except `getSourceDir` and the API renames (`fk::_2D` -> `fk::ND::_2D`, `fk::Point{}`).
- cvGS wrapper of the batched resize (`include/cvGPUSpeedup.cuh`, the `std::array<RawPtr>` path): the same code from Alpha-0.0.20 (2025-05-23) to 6d26c65 (enum renames only). The wrapper calls `PerThreadRead::build_batch(array of RawPtr)` and `Resize::build(...)` of FKL, so what changed is below them.
- FastNPP `benchmarks/CPU/benchmark_CPU_NPP_vs_FastNPP.cu` created 2025-04-12; after 2025-04-21 only: the submodule moves, `6ba1647` (2026-10-28, `NUM_EXPERIMENTS` for CUDA >= 12), `d1b4e25` (brace-initialised `fk::Point`) and `328573d` (2026-10-28: the NPP FP32 crop descriptors are sized to the crop, `crop_size`, instead of the resize target, `up_size`; this changes the descriptors NPP is given, not the number of calls). The wrapper `include/fast_npp.cuh` -> `fast_npp.h` changed API names only (`fk::ResizeRead` -> `fk::Resize`, enum class, a `float[3]` overload); it builds `std::array<fk::Ptr2D<uchar3>, BATCH>` per call in 2025 and today.
- FKL pointers of the wrappers (the pairs that were actually used): cvGPUSpeedup Alpha-0.0.21 (2025-06-17) -> 1f33224 (Beta-0.1.8); then 17 pointer moves up to 6d26c65 -> b984f10 (ab236aa 2025-09-16, afa7083, 0d1b066, 1c19b9f, 386803e, c657e82, 2627141, 8a73116, 94294dc, 9fc6cbf, 4642ba1 = Beta-0.1.12-LTS 2025-10-30, 66dd936, 9a3aa58 = Beta-0.1.14-LTS 2026-02-26, ccf3b8b, 3b408db, 1a9bfe8, b984f10). FastNPP: 0f2e5c2 (2025-04-21), 06fa452 / 0305abd / 461ef01 (2025-05-14/15), `72ddf8a` (2025-07-08, Beta-0.1.8), 5cfd38a / 3b1909d (2025-12-26, 0.1.13-LTS and the latest LTS), d384c41 / ac43fe9 / 9f13194 (2026-06-24), 089e947, 8a82180, 66499ff, 9dfc4f4 (2026-10-28/29), a1b5e245 (#45, 2026-11-06).

## What the 2026 columns say (arithmetic, not a cause)
Host time per call of the baseline in 2026 is linear in the batch: OpenCV-CUDA 12.5 to 13.2 microseconds per crop (7 calls per crop), NPP 12.8 to 13.3 microseconds per crop. The fused call is nearly flat: cvGS 1.8 microseconds at batch 2, 4.3 at batch 142, marginal cost 0.020 microseconds per crop between batch 62 and 152; FastNPP 1.9 to 6.3 microseconds, marginal 0.032 per crop.
If the 2025 baseline times per crop were the same as today's, the 2025 plateaus imply fused host times at batch 142 of 14.8 microseconds (cvGS, 120x; 2026: 4.3, so 10.5 more) and 19.7 microseconds (FastNPP, 92x; 2026: 6.3, so 13.4 more), i.e. about 0.10 and 0.14 microseconds per crop. The third number, NPP with the CPU part, at batch 150 (NPP 1,944 microseconds): 61x means 31.9 microseconds for FastNPP, today 17.8, i.e. 14.1 more. The three excesses (10.5, 13.4, 14.1 microseconds) are of the same size. This holds only if the OpenCV-CUDA and NPP sides were the same in 2025; the 2025 raw columns (numerators and denominators) would show it. Also useful from the 2025 data: the speedup WITHOUT the CPU part (2026: 141x at batch 150): if it was the same in 2025, the GPU time is not what changed.

## (b) FKL commits between Beta-0.1.8 and c8fcbe8 that touch the host path of a batched call
The files: `core/execution_model/executors.h`, `data_parallel_patterns.h`, `memory_operations.h`, `stream.h`, `operation_model/{batch_operations,instantiable_operations,fused_operation,iop_fuser,operation_tuple}.h`, `executor_details/executor_kernels.h`, `core/data/{array,ptr_nd,rawptr,tuple}.h`. 89 commits between Beta-0.1.8 (2025-05-22) and Beta-0.1.12-LTS (2025-10-30), 93 between Beta-0.1.12-LTS and c8fcbe8 (all tags are ancestors of c8fcbe8; Beta-0.2.0 of main is not). All of them are listed at the end. The ones that by subject and size change what a batched call does on the host (CANDIDATES by reading the log, none of them shown to matter):

Beta-0.1.8 -> Beta-0.1.12-LTS: `3b51609` 2025-05-26 (2,730 changed lines: .cuh renamed to .h, CPU compilers) and `2568b7f` / `3aff7b0` 05-26/27 (CPU executor); `6170bf2` 05-28 (stream.h, MemType::DeviceAndPinned); `2d33915` 06-01 (DPPType enum removed, TF flag), `3d0a3e4` 06-01; `2bcdc9d` 06-04 (reference counting in fk::Stream); `ba1f517`, `4976486` 06-12 (executeOperations for any Ptr ND, Array constructor); `2440bcb` 06-15 (GPU_NVIDIA_JIT executor), `0f83577` 06-17 (fk::String), `df5fa85` 06-30 (value serialisation of operations), `513f9f8` 07-01 (runtime serialisation removed), `2ff1b70` 07-01 (executors.h reorganised), `1058b02` 07-30 (JIT removal prepared); `10f704f` 08-04 (492 lines, NVRTC changes); `bf5f78b` 08-11 (Ptr pitch and reference counting); `98461e1` 08-30 and `26caaf6` 08-31 (fuseBack done in executeOperations_helper); `5319262` 09-01 (instantiable operations simplified); `64e345c` 09-04 (757 lines: BatchOperations, ReadBackOperations refactored); `c5e87fc` 09-21 and `1c19b9f` 09-22 (divergent horizontal fusion and circular tensor executors); `94294dc` 10-29 (Ptr copy and move constructors).
Beta-0.1.12-LTS -> c8fcbe8: `30e7827` 2026-01-21 (operator| for the execution API, & for the build API), `07b2cb3` / `a5c91f9` 01-23 (fold expression in TransformDPP, exec in instantiable operations), `bd96a25` 01-23 (fk::get return type, Tuple), `4af7167` / `5269448` 01-28 (BackFuser everywhere), `33b53be` 01-29 (operation_data with the new fk::Array; the union Array becomes ArrayVector), `33db1a7` 01-29 (BatchUtils refactored), `00793b6` .. `3751f25` 02-02..09 (new OperationTuple and FusedOperation replace the old ones), `3cb7c71` 02-11 (fk::Array an aggregate), `51d3ce3` 02-11 (Point an aggregate), `d25afe2` 02-17 (OperationTuple an aggregate), `e536e14` 02-18 (values in registers passed by value, not by const reference), `00bb4e0` 02-25 (InputFoldType an aggregate), `e0995ea` 03-10 (host apply and device apply_d), `1683a9a` 10-11 (reference counts of Ptr allocations shared by crops and planes), `6724ed1` 10-28 (the leak of every CUDA stream created by fk::Stream fixed), `6c56600`, `4a57ffb`, `28101e3`, `e72f9f3` 10-28/29 (host-side alignment checks and fall-backs per call), `2ce1f46` 10-29 (grid sizes by integer division), `1af0ffc` 10-29 (DivergentBatchTransformDPP through a lambda).

## (c) What a bisect would need and cost (estimate only; nothing was built or run)
- **The space is the pairs the wrappers pinned, not single FKL commits.** The 2025 wrappers do not compile against the 2026 FKL API (`.cuh` -> `.h`, enum class, aggregates, `Point{}`), and the 2026 wrappers do not compile against the 2025 FKL; each FKL commit therefore needs the wrapper (cvGPUSpeedup / FastNPP commit) of its time. The usable pairs are the ones above: 1 + 17 for cvGPUSpeedup (Beta-0.1.8 ... b984f10), plus c8fcbe8 with the 2026 wrappers; 14 for FastNPP. About 18 pairs for cvGS and 14 for FastNPP; a finer resolution inside a pair needs the wrapper to be adapted by hand.
- **Cost.** One binary per pair (the CPU benchmark of the pair's own `benchmarks/`): a few minutes of compile each on one core (not measured for these programs; the 40 targets of fkl-benchmarks take hours because of the 600 vertical-fusion kernels, the CPU benchmark is a single translation unit); all pairs can be compiled in parallel on the 16 compile cores in roughly 10 to 20 minutes. One CPU-time window: the programs take 3 s (cvGS) and 5 s (FastNPP) per run; 18 + 14 binaries, 3 alternating rounds: about 5 minutes with the GPU idle, or 1 to 2 minutes for a single round (the window of about 1 to 2 minutes that you mention).
- **What is not available.** (1) The tags and pairs precede the branch rule: they are ancestors of c8fcbe8 and C++17 (CUDA_STANDARD 17 in their CMake), but working on them is an exception to 'LTS-C++17 branches only' that needs Oscar's decision. (2) CUDA 12.8 and driver 570 of 2025 are not here (CUDA 13.4, driver 610), nor the OpenCV and NPP versions of then: the test can isolate the library (and the wrappers) from the harness, not the toolchain; a host-time difference of the OpenCV-CUDA side (numerator) between toolchains cannot be separated this way. (3) Old code may not compile with nvcc 13.4 and g++ 15.2 (not tested).

## Appendix: the host-path commits (git log of the files above)
### Beta-0.1.8 .. Beta-0.1.12-LTS (89)
- 0126f89 2025-10-30 Removing unsused parameter that whas making linux compilation to fail
- 94294dc 2025-10-29 Fix Ptr Copy and Move constructors on VS2017
- 5e48dc8 2025-10-29 Added some magic number explanation
- a92ada2 2025-10-29 Removing unnecessary spaces
- 1c19b9f 2025-09-22 Finished implementing the executor for circular tensor, adding ReadSet support for ND::T3D, fixing shiftFactor usage for device.
- 50ff5eb 2025-09-22 Removing dead code, trying to fix compilation error
- 0d1b066 2025-09-21 Solving CircularTensor issues, one at a time
- dfad921 2025-09-21 Merge remote-tracking branch 'origin/main' into 136-implement-executor-for-divergentbatchtransformdpp
- c5e87fc 2025-09-21 Cleaning some code and working on having the divergent horizontal fusion executor implemented
- 6aca1f9 2025-09-21 Adding static_get and vector_at
- 759c52d 2025-09-21 Removing empty file
- 7a011cc 2025-09-20 Continuing with the cxp refactor
- b231d6e 2025-09-19 Cleaning basic ops
- afa7083 2025-09-17 Starting to implement. Pending to decide how to handle the fuseBack part before calling the kernel.
- 7f14357 2025-09-13 removed CLANG_DEVICE macro
- 5f51084 2025-09-13 undef CLANG_HOST and use only CLANG_HOST_DEVICE and CLANG_DEVICE
- d9e6c41 2025-09-13 stream is device code
- 2b4a8cd 2025-09-13 removed extra parenthesis from macro
- 4f1e5f6 2025-09-13 remove HIP and NVRTC macros from preprocessor checks
- caffb1d 2025-09-13 new clang macros for detecting host and device code
- 160cc5c 2025-09-13 Merge branch 'main' into 128-support-llvm-as-a-cuda-kernel-compiler
- 2368e6a 2025-09-12 Add missing "public:" in BatchRead
- 160ab07 2025-09-08 workaround for clang  variable of non-literal type 'const cg::thread_block' cannot be defined in a constexpr function before C++23.
- cb43f05 2025-09-07 added missing __CUDA_ preprocessor definition in utils.h and dataparallelpatterns
- d94a67d 2025-09-07 added __CUDA__ preprocessor check to detect that we are building cuda kernels with clang .
- ef75c91 2025-09-05 Using a more descriptive macro to detect VS2017 compiler versions
- 6139634 2025-09-05 Removing unnecessary code
- 70972a8 2025-09-05 Moving NumElems to a more appropiate place
- ae7985b 2025-09-05 Still trying...
- 60a9c0e 2025-09-04 Merge remote-tracking branch 'origin/main' into 119-make-main-branch-to-compile-on-vs2017
- 8df0753 2025-09-04 Removing dead code and fixing template parameter shadowing on Linux
- 64e345c 2025-09-04 Refactored BatchOperations, ReadBackOperations, created a new IncompleteReadBackType, and it's ParentOperation, InstantiableOperati
- e36fda0 2025-09-01 Removing unnecessary parameter from BatchReadBase
- eac7647 2025-09-01 A few estetic changes plus removing some useless code
- 7f75336 2025-09-01 Making sure there are no issues by using an existing type name as a template type name
- 5319262 2025-09-01 Simplifying Instantiable Operations implementation code, and removing dead code.
- 26caaf6 2025-08-31 Added utest for executors and fixed idxFirstNonBack and fuseBack
- 98461e1 2025-08-30 Implemented fuseBack, and used it so that all executeOperations_helper calls, get the "back fusion" already done.
- 8ebc673 2025-08-21 Fixing several issues, related to class template parameters, more issues pending to be solved
- 1521c4e 2025-08-18 Making the evaluation of fk::BoolN values easier, refactoring vector operator tests for completeness, and adding an informative sta
- 03b56cf 2025-08-15 Complete BackFunction to BackIOp renaming across codebase
- bda3cd0 2025-08-12 Avoiding a bug in nvcc compiler
- bf5f78b 2025-08-11 Fixing pitch and reference counting in Ptr
- 1147419 2025-08-11 Adding some utilities and ReadYUV unit test
- ee4d98f 2025-08-06 Added TestCaseBuilder for ReadType Operations
- 48603da 2025-08-04 Fix CUDA compilation errors by adding missing enum class scope qualifiers
- 4ca7f32 2025-08-04 Complete enum to enum class conversion with most tests passing
- 9e24d2d 2025-08-04 Convert ND, MemType, ShiftDirection, CircularDirection, CircularTensorOrder, ColorPlanes enums to enum class
- 10f704f 2025-08-04 Saving files with UTF-8 without BOM, plus some other changes to help with NVRTC usage
- 414e21b 2025-07-31 Removing unnecessary undef
- e57c48a 2025-07-31 Removing uneeded headers
- 1058b02 2025-07-30 Preparing for JIT compilation removal
- 66520be 2025-07-03 Adding call to BaseStream constructor
- 216ba8a 2025-07-03 Fixing compilation issues in Linux
- f7e616e 2025-07-02 Removing dead code
- 2ff1b70 2025-07-01 Reorganizing code
- 513f9f8 2025-07-01 Removing incomplete runtime data serialization to focus on runtime compilation
- 0b9a4cb 2025-06-30 Merge branch 'main' into 57-using-the-toy-example-include-the-fkl-and-solve-the-missing-std-library-headers
- df5fa85 2025-06-30 Making value serialization of operations to work in both constexpr and runtime.
- f6baaa6 2025-06-18 Renaming the macro FK_STATIC_STRUCT_SELFTYPE to FK_STATIC_STRUCT
- 0f83577 2025-06-17 Adding fk::String to have a constexpr string representation that works with operator<<, operator+ and operator== in a constexpr con
- 2440bcb 2025-06-15 WIP: adding Executor<ParArch::GPU_NVIDIA_JIT>, still compilation problems to solve on both host and device sides
- ee95c48 2025-06-15 Work in progress
- 7c9b89d 2025-06-12 Fixed an issue with vector types of one channel and Thread Fusion
- ba1f517 2025-06-12 Fixed executeOperations functions to allow Ptr of any enum ND.
- 4976486 2025-06-12 Fixed Array constructor
- 4a79623 2025-06-11 oscar implemnetation for base case uint1,uint1
- cce732b 2025-06-09 Removing FK_STATIC_STRUCT and using FK_STATIC_STRUCT_SELFTYPE always, due to compilation issues on Linux with the first one.
- ebd7326 2025-06-09 Renamed the macro FK_STATIC_STRUCT_CHILD to FK_STATIC_STRUCT_SELFTYPE
- 8078073 2025-06-09 Enforce Operations can not be instantiated
- 2585247 2025-06-05 Removing the tuple constructors, because then VS2022 expects the constructor based on initializer list, and we don't need that cons
- ada69bb 2025-06-05 Fixed constructor bug in Array<T, 1>
- c103b59 2025-06-05 Removed unused code from array.h, added constexpr constructors to fk::Tuple
- b0deb66 2025-06-05 Fixed compilation, added operator= and constructor from CUDA vector types to Array union
- 9aa694b 2025-06-04 Fixed compilation issues on Linux
- 2bcdc9d 2025-06-04 Added reference counting to fk::Stream
- 2d33915 2025-06-01 Removing the DPPType enum. Using DPP's directly. Using enum TF as the flag to choose if we want to allow for TF or not
- 6bf1f64 2025-06-01 Removing duplicated DPPTransform specialization
- 3d0a3e4 2025-06-01 Improved modularity of DPPTransform code
- 84501e7 2025-05-31 Fixed compilation, but we will make all the logical operators that typicaly return bool, to return the corresponding boolN vector t
- 6620261 2025-05-29 Fixed remaining Benchmarks and few other issues
- 92bc9bb 2025-05-29 Fixing details in Ptr related to the new DeviceAndPinned MemType, and adapting benchmark_horizontal_fusion to the new fk::Stream. P
- 6170bf2 2025-05-28 Adding stream.h and adding MemType::DeviceAndPinned. WIP
- d0e2f9f 2025-05-28 Making default code to adapt to the compiler being used, to ease code portability to different architectures.
- fcac932 2025-05-27 Removed useless directive from return type of cr_point functions
- 2440037 2025-05-27 Fixed MSVC CPU compilation
- 3aff7b0 2025-05-27 Making FKL compilable with MSVC and GCC to execute on CPU
- 2568b7f 2025-05-26 Now, we can also use the CPU (single core) to execute the "kernels".
- 3b51609 2025-05-26 Removed .cuh file extension, so that we can compile the files with CPU compilers directly. Missing to add the TransforDPP's and pro

### Beta-0.1.12-LTS .. c8fcbe8 (93)
- 71b0076 2026-10-30 Add upload, download and setTo for ParArch::CPU_MULTITHREAD
- e5608ea 2026-10-30 Add TransformDPP and its executor for ParArch::CPU_MULTITHREAD
- 1069e37 2026-10-30 Add ParArch::CPU_MULTITHREAD, the thread pool wrapper and its stream
- 1af0ffc 2026-10-29 DivergentBatchTransformDPP: call launchTransformDPP through a lambda
- 25c8559 2026-10-29 Stream_: clear the error of a failed stream destroy; HIP destroy comment
- 2ce1f46 2026-10-29 Size grids and thread fusion thread spaces with integer divisions
- 7309118 2026-10-29 Select the HIP backend with FK_HIP_BACKEND, not __HIPCC__
- 1683a9a 2026-10-11 Fix the reference counts of Ptr allocations shared by crops and planes
- 2fa4401 2026-10-29 HIP executor: fall back to the non thread fused kernel for misaligned data
- e354f0e 2026-10-29 C++17 conformance: no C++20 extensions in array.h, ptr_nd.h
- 8a03505 2026-10-29 HIP backend: ParArch::GPU_AMD streams, memory, DPPs and executors
- 6724ed1 2026-10-28 Fix the leak of every CUDA stream created by fk::Stream
- e72f9f3 2026-10-29 Check circular batch params for thread fusion alignment
- 28101e3 2026-10-28 Fall back to the non thread fused kernel for misaligned GPU data
- 4a57ffb 2026-10-28 Fall back to the non thread fused path for misaligned CPU data
- 6c56600 2026-10-28 Align host allocations and pad host 2D pitches for thread fusion
- e068e95 2026-06-14 Fix divergent batch DPP z indexing
- ad186c7 2026-03-17 Merge pull request #235 from Libraries-Openly-Fused/copilot/sub-pr-223-another-one
- b376c05 2026-03-17 Remove outdated NVRTC comment above TransformDPP GPU_NVIDIA specialization
- 549f3e9 2026-03-17 Revert NVRTC guard changes, keep typo fix and clean up comment
- 85660e2 2026-03-17 Fix typo and expand NVRTC guard in data_parallel_patterns.h
- cac0650 2026-03-17 Fix #endif comment to match #if defined(__NVCC__) guard
- 862ca32 2026-03-17 Merge branch 'LTS-C++17' into 221-remove-vs2017-support
- 584c8c6 2026-03-12 Add missing Operation type aliases for fk::Instantiable
- 948edac 2026-03-10 Adding static assert
- e0995ea 2026-03-10 Refactored apply into host apply and device apply_d
- 531a02a 2026-03-06 removed vs2017 and clang with cuda support
- a0b42e2 2026-02-26 Update include/fused_kernel/core/execution_model/executors.h
- 00bb4e0 2026-02-25 Making InputFoldType an aggregate
- b6222ac 2026-02-25 Fixing ColorConversion FusedOperation based implementations, fixing fuseBackSequence, removing && and & from parameters that will b
- e536e14 2026-02-18 Made all values that reside in registers, to be passed as const value, instead of const reference, to be more consistent with the r
- 952eff1 2026-02-18 Simplified get wrapper and get_opt implementations
- fd8274d 2026-02-18 Making get_opt generic like get
- 6ac036a 2026-02-17 Using correct macro in OpenType Instantiable Operation
- 5040b13 2026-02-17 further simplifying operation_types file
- d9b56a3 2026-02-17 Making make and cat for OperationTuple host only functions. They should be used only on CPU.
- 42d2889 2026-02-17 Fixing comments and removing unnecesary decltype(auto)
- d25afe2 2026-02-17 Making OperationTuple an aggregate
- 847e1b8 2026-02-17 Removing unnecessary decltype(auto)
- 638ec8f 2026-02-17 Making some changes to avoid confusion with the two types of operator| in InstantiableOperations
- ee964cf 2026-02-17 Added operator[] implementations with index passed as size_t
- 6c2e5af 2026-02-17 Removing unnecesary decltyp(auto)
- 50e909c 2026-02-16 Cleaning unnecessary decltype(auto) usage
- 08c70ec 2026-02-16 Merge pull request #193 from Libraries-Openly-Fused/copilot/sub-pr-188-yet-again
- f1dd700 2026-02-16 Fix copyright year updates per review feedback - correct rules applied
- 0008978 2026-02-16 Fix WriteType specialization: exclude write op from fold and extract .input
- 4e08dfa 2026-02-15 Update copyright year to 2026 for Oscar Amoros Huguet in all modified files
- ce5196a 2026-02-15 Update include/fused_kernel/core/data/ptr_nd.h
- 21983d0 2026-02-15 Update include/fused_kernel/core/execution_model/operation_model/instantiable_operations.h
- 72007a7 2026-02-15 Update include/fused_kernel/core/execution_model/operation_model/instantiable_operations.h
- c99e7cb 2026-02-15 Fixing Unary FusedOperation input
- acfb977 2026-02-15 Using decltype(auto) where possible.
- 51d3ce3 2026-02-11 Making Point an aggregate, and catching writeOp as a const&
- 3cb7c71 2026-02-11 Making fk::Array an aggregate
- 4f8da1f 2026-02-11 Adding nice vector type aliases
- 2fa7c9c 2026-02-09 Removing unnecessary Parent Operation, and fixing some bugs
- 6c5623d 2026-02-09 Unifying template parameter pack across FusedOperation implementations
- 22e829f 2026-02-09 Removing unused variable
- b6a3a30 2026-02-09 Finished implementing the last FusedOperation specializations
- 7ca62a4 2026-02-09 Refactored TypeList, so that it is faster for the compiler to find an element. Created a variadic version of TypeListCat_t with fol
- b80b8b2 2026-02-09 Moving memory_operations.h to algorithms/basic_ops, and removing NVRTC condition
- 3751f25 2026-02-09 Removing old commencted out versions of OperationTuple and FusedOperation, and renaming the new ones to the final name
- fdf02aa 2026-02-09 Compiling version. Still pending to implement a few FusedOperation versions
- 1b46d34 2026-02-07 Adding OpenType to ParamsTypes in OperationData, temporarily renaming get<>(NewOperationTuple) to get_opt, to ease debugging, fixin
- d319009 2026-02-07 Adding a modification to the API, that will simplify the FusedOperation, and the DivergentTransformDPP
- 3abf46d 2026-02-06 Adapting tests, and working on final details. We need to avoid having nested FusedOperations.
- e94dd65 2026-02-05 Adding FusedType since we need a different exec specification we did not have before. Completed NewFusedOperation, missing tests. A
- 8bf4fc3 2026-02-04 Making sure we preserve the r/l-vaue and constness nature of what get<>(NewOperationTuple) returns.
- 36e79c2 2026-02-04 Finishing new OperationTuple implementation with basic set of tests.
- 00793b6 2026-02-02 Created and tested NewOperationTuple. Mising all unary version.
- ce5fe40 2026-01-30 Adding some fixes
- 7623a4f 2026-01-30 Fixing compilation issues with clang
- 923cea8 2026-01-29 removing std::make_index_sequence that are not necessary and optimize some more functions
- e0a748a 2026-01-29 Optimizing compilation times for iop_fuser
- 33b53be 2026-01-29 Refactoring operation_data, using new fk::Array. Changing previous union fk::Array to fk::ArrayVector. Optimizing compilation times
- d696582 2026-01-29 Fixed constexpr function evaluation in BatchUtils
- 33db1a7 2026-01-29 Refactored BatchUtils, wich made compilation of batch_build_compilation_time.cu more than 4x faster
- 8d1d29c 2026-01-29 Merge branch '178-create-benchmark-to-compare-fkl-with-bvf-and-fkl-without-bvf' into 179-refactor-recursive-template-implementation
- 5269448 2026-01-28 Removing Back and using BackFuser everywhere
- 4af7167 2026-01-28 Creating new BackFuser based on new Tuple
- 1aae01a 2026-01-28 Adding operator& to InstantiableReadOperation for the case of BatchOperation<Operation> where Operation is Incomplete
- b69e7bd 2026-01-28 Renaming cat to tuple_cat, to avoid mixing with cat for OperationTuple
- 2bb5273 2026-01-28 Removing unnecesary recursive expansion funtions in Fuser
- 1a8dd78 2026-01-28 Removed unnecessary exec function in Instantiable operations
- 758b7af 2026-01-25 Adding operator| to read operations, with Point as the first parameter
- 4fd648b 2026-01-24 Adding more implementations fo get, make_tuple, forward_as_tuple and tie
- 07b2cb3 2026-01-23 Adding fold expression expansion strategy to TransformDPP
- a5c91f9 2026-01-23 Adding exec function to instantiable operations, to allow propagation of the thread index when using fold expressions with operator
- bd96a25 2026-01-23 Fixing fk::get<Idx>(Tuple) return type, to avoid unnecessary local memory usage, refactored Tuple and Tuple utils with fold expresi
- 30e7827 2026-01-21 Adding friend binary operators to all IOps, | for the execution API, and & for the build API. This will allow to expand IOp paramet
- 6827d73 2026-01-17 Compiling, but there is a bug somewhere.
- 05b7b64 2025-12-20 Fixing macro code selection
- 0044220 2025-10-30 Fixing make_fusedArray
