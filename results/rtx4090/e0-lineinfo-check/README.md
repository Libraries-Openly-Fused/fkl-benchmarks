# Does -lineinfo change the SASS of the E0 binaries?

The E0 benchmarks are built with `-lineinfo` (the library default, `FKLB_LINEINFO=ON`); the SASS comparisons of E1 are built without it
(preset `rtx4090-e1`). This check builds six E0 targets twice from the same commit of fkl-benchmarks (see `commit.txt`), once with
`-lineinfo` and once with `-DFKLB_LINEINFO=OFF`, nothing else changed (CUDA 13.4.3, sm_89, same pinned fkl and cvGPUSpeedup), compiles only
(no program was executed, no GPU time was used), and compares the SASS of every kernel with `scripts/sass_compare_builds.py`
(`cuobjdump -sass`, addresses and encodings removed, compared kernel by kernel by mangled name).

Targets: benchmark_latency_hiding, benchmark_numinstructions_x_operation, benchmark_horizontal_fusion, benchmark_data_type,
benchmark_CPUandGPU_cvGS_vs_fk, benchmark_vertical_fusionCompareMulAdd1 (all `_cu`). With: 21 of 26 `flags.make` files contain `-lineinfo`
(the five that do not are the `_cpp` CPU back-end targets, which have CXX flags only; all six `_cu` targets of the list carry it); without: 0 of 26, and the cubin has no `nv_debug_line_sass` section.

Result (`sass_compare_lineinfo_on_vs_off.txt`): 596 kernels in both builds, 397 identical, 199 "different", all 199 in
`benchmark_vertical_fusionCompareMulAdd1_cu`. Every one of those 199 has the same number of lines in both builds and differs in ONE line only:
the header line of the dump, `ptxasOptions = -split-compile=0 --generate-line-info` against `ptxasOptions = -split-compile=0`. That is 199
differing lines of 996,390 in those kernels, and none of them is an instruction. So the instruction streams are identical: `-lineinfo` adds the
line table (debug section) and nothing to the code, and the E0 timings do not need to be repeated for this reason.

Build time: 521 s with, 500 s without (6 targets, 14 jobs, cores 4-7, 9-12, 20-23, 25-28). Build directories: `build/lineinfo-on`, `build/lineinfo-off`
(cleanup candidates).
