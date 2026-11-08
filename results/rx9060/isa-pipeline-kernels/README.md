# The gfx1200 code of the pipeline (a) kernels on the RX 9060 (2026-11-08)

This is the counterpart on the RX 9060 of the SASS review of Table III. It describes the code of the kernels of Figure 15's pipeline, read from
the code object of the measured binary. It is compile-only: nothing was run on the GPU for it.

**Scratch.** None of these kernels uses scratch, the private segment that is AMD's counterpart of CUDA's local memory. The private segment is 0
bytes in the code-object metadata and in the kernel descriptor of every kernel. No listing holds a `scratch_`, `flat_`, `buffer_` or `ds_`
instruction: the only memory instructions are global loads and stores, and scalar loads of the kernel arguments. None of the kernels has a
dynamic stack, none spills a VGPR or an SGPR, and none uses LDS either.

## Source

- **The binary:** `benchmark_rpp_preprocessing_hip` of `results/rx9060/rpp-preprocessing-built-once/`, at fkl-benchmarks bdc7d19 with FKL
  64ad9195. sha256 ed953c9da3d39e7a7fbb33c3bf752feb632c76ea3e86b45559aaf69174dadf1e (`hashes.txt`).
- **Its code object:** the gfx1200 one (`hipv4-amdgcn-amd-amdhsa--gfx1200`), the only offload bundle of its `.hip_fatbin`: the program is one HIP
  translation unit.
- **The kernels:**
  - `fkl_fused` and `fkl_fused_halfpixel`, FKL's fused kernels. The second is also the kernel of `fkl_fused_halfpixel_once`.
  - `hand_batch`, `hand_padded` and `hand_roi`, the hand-written kernels of E1c and E4.
  - `hand_roi_halfpixel` (`e1c::handPreprocRoiHalfPixel`), the kernel of `hand_roi_halfpixel_once`.

  Each at 10, 150 and 300 crops.

## The table

| variant | N | instructions | VGPRs (allocated) | SGPRs | scratch, bytes | LDS, bytes | kernarg, bytes | waves per SIMD |
|---|---|---|---|---|---|---|---|---|
| fkl_fused | 10 | 202 | 21 (24) | 22 | 0 | 0 | 832 | 16 |
| fkl_fused | 150 | 202 | 21 (24) | 22 | 0 | 0 | 7552 | 16 |
| fkl_fused | 300 | 202 | 21 (24) | 22 | 0 | 0 | 14752 | 16 |
| fkl_fused_halfpixel | 10 | 212 | 21 (24) | 22 | 0 | 0 | 832 | 16 |
| fkl_fused_halfpixel | 150 | 212 | 21 (24) | 22 | 0 | 0 | 7552 | 16 |
| fkl_fused_halfpixel | 300 | 212 | 21 (24) | 22 | 0 | 0 | 14752 | 16 |
| hand_batch | 10 | 177 | 24 (24) | 16 | 0 | 0 | 376 | 16 |
| hand_batch | 150 | 177 | 24 (24) | 16 | 0 | 0 | 1496 | 16 |
| hand_batch | 300 | 177 | 24 (24) | 16 | 0 | 0 | 2696 | 16 |
| hand_padded | 10 | 180 | 24 (24) | 17 | 0 | 0 | 784 | 16 |
| hand_padded | 150 | 180 | 24 (24) | 17 | 0 | 0 | 7504 | 16 |
| hand_padded | 300 | 180 | 24 (24) | 17 | 0 | 0 | 14704 | 16 |
| hand_roi | 10 | 179 | 24 (24) | 13 | 0 | 0 | 600 | 16 |
| hand_roi | 150 | 179 | 24 (24) | 13 | 0 | 0 | 5080 | 16 |
| hand_roi | 300 | 179 | 24 (24) | 13 | 0 | 0 | 9880 | 16 |
| hand_roi_halfpixel | 10 | 192 | 24 (24) | 12 | 0 | 0 | 600 | 16 |
| hand_roi_halfpixel | 150 | 192 | 24 (24) | 12 | 0 | 0 | 5080 | 16 |
| hand_roi_halfpixel | 300 | 192 | 24 (24) | 12 | 0 | 0 | 9880 | 16 |

`isa-table.csv` has every column, the mangled kernel name and the hash of each listing. All kernels are wave32, none has a dynamic stack, and
none spills.

**Instructions by kind.** The counts are the same at 10, 150 and 300 crops (`instruction-kinds.csv`):

| variant | instructions | scheduling | global memory | scalar loads | vector ALU | other scalar |
|---|---|---|---|---|---|---|
| fkl_fused | 202 | 50 | 11 | 9 | 109 | 23 |
| fkl_fused_halfpixel | 212 | 53 | 11 | 9 | 116 | 23 |
| hand_batch | 177 | 41 | 11 | 4 | 106 | 15 |
| hand_padded | 180 | 45 | 11 | 4 | 102 | 18 |
| hand_roi | 179 | 43 | 11 | 6 | 102 | 17 |
| hand_roi_halfpixel | 192 | 50 | 11 | 6 | 108 | 17 |

The kinds are:
- **scheduling:** `s_wait_*`, `s_delay_alu`, `s_clause` and `s_nop`;
- **scalar loads:** the loads of the kernel arguments;
- **other scalar:** the rest of the `s_` instructions (scalar ALU, branches, `s_endpgm`).

No instruction falls outside these kinds.

## How each column is obtained

`scripts/isa_pipeline_kernels.py` stops if any of its cross-checks fails.

- **instructions:** the lines of the function's disassembly (`llvm-objdump -d --symbolize-operands`), grouped as `scripts/devcode_compare.py` groups
  them (the function with its branch-target blocks), without the `...` lines that stand for zero bytes. The count includes the scheduling
  instructions.
- **VGPRs:** `.vgpr_count` of the code-object metadata (`NT_AMDGPU_METADATA`). In parentheses is the allocation, `.amdhsa_next_free_vgpr` of the
  kernel descriptor, in whole granules of 24.
- **SGPRs:** `.sgpr_count` of the metadata.
- **scratch:** the private segment. `.private_segment_fixed_size` of the metadata must equal `.amdhsa_private_segment_fixed_size` of the kernel
  descriptor.
- **LDS:** the group segment. `.group_segment_fixed_size` must equal `.amdhsa_group_segment_fixed_size`.
- **kernarg:** `.kernarg_segment_size` must equal `.amdhsa_kernarg_size`. It includes the hidden kernel arguments of the runtime.
- **waves per SIMD:** the compiler's own theoretical occupancy, "Occupancy [waves/SIMD]" of `-Rpass-analysis=kernel-resource-usage`. It is taken
  from a build of the same sources with that flag, whose device code is the same as the measured binary's: 217 of 217 functions identical
  (`remark-build-devcode.txt`). The remark's VGPRs and SGPRs must equal the metadata's, and its occupancy must equal the formula below.
  `remarks-pipeline-kernels.txt` has the 18 remark blocks.

## The gfx1200 limits and the occupancy formula

```
waves per SIMD = min(16, floor(1536 / (24 * ceil(VGPRs / 24))))     (wave32, no LDS, no scratch)
```

- **wave32:** every kernel's metadata gives a wavefront size of 32.
- **At most 16 waves per SIMD.**
- **A file of 1,536 VGPRs per SIMD in wave32** (each VGPR is 32 lanes of 4 bytes: 192 KB per SIMD), allocated in **granules of 24**.
- **SGPRs** do not limit the occupancy on gfx10 and later in the compiler's model. **LDS** would, but these kernels use none.

**Evidence for these limits in the compiler's model of gfx1200.** `limits/pressure.hip` holds synthetic kernels that keep K floats live.
Compiled for gfx1200 with the same remark
(`limits/remarks.txt`), they give:

| VGPRs in use | 17 to 91 | 97, 101 | 121, 129 | 161 | 193 | 251 |
|---|---|---|---|---|---|---|
| waves per SIMD | 16 | 12 | 10 | 9 | 7 | 5 |

All of these are the formula's values. A file of 1,024 VGPRs would give 10 waves at 91 VGPRs, and a granule of 8 would give 14 at 97, so neither
fits.

**What the formula leaves out.** It is the compiler's theoretical occupancy from the kernel's own resources. It leaves out the hardware's dynamic
limits, such as the workgroup slots of a WGP. The kernels run in WGP mode (`.amdhsa_workgroup_processor_mode 1`), with workgroups of 32 x 8
threads, i.e. 8 waves. That is the hand-written kernels' block (`rpp/benchmark_rpp_preprocessing.h:497`), and line 132 asserts that it is
FKL's default block for this output size.

With 21 or 24 VGPRs (24 allocated), every kernel here reaches the maximum of 16. VGPRs would start to limit only above 96.

## What the table shows

- **Same occupancy, no local memory.** All six kernels have the same theoretical occupancy, the maximum of 16 waves per SIMD. None uses scratch,
  LDS or spills.
- **The code does not depend on the batch size.** Each kernel's listings at 10, 150 and 300 crops differ only in the immediate offsets of 2 to
  6 scalar loads. These are the loads of the kernel arguments that follow the per-crop array; their offsets are N times the per-crop size plus a
  constant.
  - FKL's fused kernel has 202 instructions (ASYMMETRIC) and 212 (HALF_PIXEL).
  - The hand-written kernels have 177 (`hand_batch`), 180 (`hand_padded`), 179 (`hand_roi`) and 192 (`hand_roi_halfpixel`).
  - HALF_PIXEL adds 10 instructions to FKL's kernel (7 vector, 3 scheduling) and 13 to the hand-written one (6 vector, 7 scheduling). The
    vector instructions it adds are:
    - a compare and a select for each coordinate (`v_cmp_le_f32`, `v_cndmask_b32`), the clamp at 0 of the source position;
    - additions and subtractions for the ±0.5 offsets, with the multiply-adds rearranged around them.
- **Registers.** FKL's fused kernels use fewer VGPRs (21) and more SGPRs (22) than the hand-written ones (24 VGPRs, 12 to 17 SGPRs).
- **Kernel arguments.** They grow with N by:
  - 48 bytes per crop for FKL;
  - 8 for `hand_batch`;
  - 48 for `hand_padded` (by construction);
  - 32 for `hand_roi` and `hand_roi_halfpixel`.

  At 300 crops that is 14,752 bytes for FKL against 9,880 for `hand_roi_halfpixel`.
- **Figure 15's difference is not occupancy or scratch.** Built once, the ratio `hand_roi_halfpixel_once` / `fkl_fused_halfpixel_once` (the
  median of the three runs) is:
  - 0.965 to 0.998 at 10 to 150 crops, i.e. 0.2 to 3.5 percent faster;
  - 0.940 to 0.998 at 10 to 300 crops.

  All three runs are below 1 at 25 of the 30 batches (`results/rx9060/rpp-preprocessing-built-once/`). The two kernels have the same
  theoretical occupancy and neither uses scratch, so neither explains the difference. What differs in their code:
  - **Instructions:** 212 against 192. The global memory instructions are the same (8 loads and 3 stores). FKL's kernel has:
    - 8 more vector instructions (116 against 108);
    - 6 more other scalar ones (23 against 17: three integer multiplications, an addition, a shift and an or);
    - 3 more scalar loads of kernel arguments (9 against 6);
    - 3 more scheduling instructions (53 against 50).
  - **SGPRs:** 22 against 12.
  - **Kernel arguments:** 48 against 32 bytes per crop, i.e. 7,552 against 5,080 bytes at 150 crops and 14,752 against 9,880 at 300.

  A kernel-only timing (the launch queued behind a GPU spin) would separate the kernel from its launch; it was not run.

## Files

- **`isa-table.csv`:** the table with every column, the mangled kernel names and the listing hashes.
- **`instruction-kinds.csv`:** the instructions of each kernel by kind, and how its listings differ between the batch sizes.
- **`listings/<variant>-N<N>.txt`:** the instruction lines of each kernel, with the branch labels folded and no newline after the last line. The
  sha256 of each file is `scripts/devcode_compare.py`'s hash of the function (`hashes.txt`). All 18 equal that script's hashes in the step-0
  comparison of the built-once run.
- **`remarks-pipeline-kernels.txt`:** the compiler's resource-usage remarks of the 18 kernels.
- **`remark-build-devcode.txt`:** the device-code comparison of the remark build with the measured binary.
- **`limits/pressure.hip`, `limits/remarks.txt`:** the synthetic kernels and their remarks.
- **`hashes.txt`:** the binary, its code object, the remark build and the listings.

## How to reproduce

1. **The remark build:** the configure line of `results/rx9060/rpp-preprocessing-built-once/step0/rpp_run_build.sh` plus
   `-DCMAKE_HIP_FLAGS=-Rpass-analysis=kernel-resource-usage`, then `ninja benchmark_rpp_preprocessing_hip > build.log 2>&1`.
2. **The table:**
   `python3 -I scripts/isa_pipeline_kernels.py <the measured binary> build.log results/rx9060/isa-pipeline-kernels [--llvm /opt/rocm/core-10.1/lib/llvm/bin]`.
3. **The limits:**
   `amdclang++ -x hip --offload-arch=gfx1200 -std=c++17 -O3 --cuda-device-only -c -Rpass-analysis=kernel-resource-usage limits/pressure.hip -o /dev/null`.
