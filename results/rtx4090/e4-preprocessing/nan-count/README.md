# NaN and Inf count of the E4 outputs (and of the E16 outputs): a check run, not a measurement

Why: the comparison behind the agreement columns of E4 (`max_abs_err_own_reference`, `ok_own`, `max_abs_err_vs_asymmetric`, `max_abs_err_vs_half_pixel`) is `maxAbsDiff` of `opencv-cuda/batched-preprocessing/e4_preprocessing.cu` (lines 196-200), which takes the maximum with `std::max(m, |d|)`:
`std::max(m, NaN)` returns `m`, so a NaN in an output never reaches the maximum, and the committed CSVs hold only maxima. The committed runs therefore cannot show that no output contained a NaN; this run counts them.
(The same review found that the E2, E0 horizontal-fusion, E0 wrapper, E3 and E11 comparisons are not blind to NaN; see the reply to the coordinator of 2026-11-01.)

What ran (window 2026-11-01 20:05:00 to 20:05:07, `window-run.txt` is the output of the window script (the first line is the start line of the window, the last line the end marker), guard passed, nothing else on the machine, the RTX 4090 also drives the display):
- `e4/`: a local copy of `e4_preprocessing.cu` with ONE added block that counts the NaN, Inf and zero values of every output after the check launch and prints `NANCOUNT <experiment> batch <n> <version> (<convention>): ...` to stdout (the diff is `e4_nancount.diff`; the CSV code, the timing code and the flags of the committed `build/e4` are untouched, except `--threads 4` of nvcc),
  run with `E4_PROFILE=1`: every implementation launches once and nothing is timed, over the whole sweep (the 162 outputs of the committed agreement CSVs). The output buffer is cleared with `cudaMemsetAsync(0)` before each launch, so a value that a version does not write is a 0, which is counted in the zeros column, not a NaN.
  The timing columns of `e4/e4_preprocessing.csv` of this run are zeros: **this is a check run, no time of it is used anywhere.** The agreement CSV of this run is a repeat of the committed check.
- `e16/`: the same kind of copy of `fkl-self/attention/e16_attention.cu` (`e16_nancount.diff`): a kernel counts the NaN, Inf and zero values of ALL BH x S x D values of the output of each variant (the committed error columns compare two heads), one launch per variant (`E16_PROFILE=1`), D = 64 and 128, S = 512 and 8192, both masks.
- `binary-sha256.txt` has the hashes of the counting binaries and of the committed ones; the sources of the counting copies are the committed sources plus the diffs.

## E4: NaN and Inf values per version (every output of the sweep, E4_PROFILE=1)

| version | convention | outputs | values | NaN | Inf | zeros | most NaN in one output | most Inf in one output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fkl_fused | ASYMMETRIC | 33 | 150,847,488 | 0 | 0 | 0 | 0 | 0 |
| fkl_fused_half_pixel | HALF_PIXEL | 33 | 150,847,488 | 0 | 0 | 0 | 0 | 0 |
| opencv_percrop | ASYMMETRIC | 33 | 150,847,488 | 0 | 0 | 0 | 0 | 0 |
| opencv_batch | ASYMMETRIC | 30 | 114,278,400 | 0 | 0 | 0 | 0 | 0 |
| npp | HALF_PIXEL | 33 | 150,847,488 | 0 | 0 | 0 | 0 | 0 |
| all |  | 162 | 717,668,352 | 0 | 0 | 0 | 0 | 0 |

The agreement CSV of this run has 162 rows, ok_own {'1': 162}, max_abs_err_own_reference {'5.35117255e-06': 162} (the committed runs: 162 rows, ok_own = 1, 5.35117255e-06 in every row).

## E16: NaN and Inf values per variant (all BH x S x D output values; D = 64 and 128, S = 512 and 8192, both masks, E16_PROFILE=1)

| variant | outputs | values | NaN | Inf | zeros | most NaN in one output | most Inf in one output |
|---|---:|---:|---:|---:|---:|---:|---:|
| mma_raw | 8 | 201,326,592 | 0 | 0 | 5 | 0 | 0 |
| mma_int8kv_fused | 8 | 201,326,592 | 0 | 0 | 842 | 0 | 0 |
| mma_int8kv_unfused | 8 | 201,326,592 | 0 | 0 | 843 | 0 | 0 |
| mma_fp8kv_fused | 8 | 201,326,592 | 0 | 0 | 9 | 0 | 0 |
| mma_fp8kv_unfused | 8 | 201,326,592 | 0 | 0 | 9 | 0 | 0 |
| mma_epilogue_fused | 8 | 201,326,592 | 0 | 0 | 0 | 0 | 0 |
| mma_epilogue_separate | 8 | 201,326,592 | 0 | 0 | 0 | 0 | 0 |
| simt_raw | 4 | 100,663,296 | 0 | 0 | 1 | 0 | 0 |
| all | 60 | 1,509,949,440 | 0 | 0 | 1709 | 0 | 0 |

## Result

- **E4: 0 NaN and 0 Inf in all 162 outputs (717,668,352 values), and no zero** (every element of every output was written: the buffers are cleared to 0 before each launch). The agreement CSV of this run repeats the committed one: 162 rows, ok_own = 1, `max_abs_err_own_reference` 5.35117255e-06 in every row.
- **E16: 0 NaN and 0 Inf in all 60 outputs of the FKL program (1,509,949,440 values, every value of each output)** for D = 64 and 128, S = 512 and 8192, causal and not. The zeros (1,709 in all: 842 and 843 in the two int8 variants, 9 in each fp8 variant, 5 in `mma_raw`, 1 in `simt_raw`, none in the epilogue variants)
  are not NaN; two different kernels that read the same quantised K and V (the fused and the unfused int8 variants) give 842 and 843, the bf16 kernel on the unquantised data 5, so they follow the data and not a kernel; I did not look into them further, and no paper statement rests on the E16 outputs beyond the error columns.
- The counts are of the outputs of the check launch of each implementation (the one `ok_own` is computed from); the timed launches of the committed runs are the same launches of the same code.
- A check run, not a measurement: it took 7 seconds (E4 5 s, E16 2 s), nothing in it is timed, and no time of it is used anywhere.

Reproduce: `scripts/nancount_summary.py <this folder>` writes `nancount_e4.csv`, `nancount_e16.csv` and `nancount_summary.md` from `e4/stdout.txt` and `e16/stdout.txt`.
