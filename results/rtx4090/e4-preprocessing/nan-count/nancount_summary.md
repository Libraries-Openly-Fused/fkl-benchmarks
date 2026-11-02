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

