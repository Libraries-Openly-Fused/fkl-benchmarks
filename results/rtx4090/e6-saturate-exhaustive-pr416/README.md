# The same exhaustive test against the head of FKL PR #416 (231dec8)

`fkl-self/numerics/saturate_exhaustive.cu`, the same source and the same flags as `../e6-saturate-exhaustive`, built against the include tree of the head of PR #416 (`cxp::saturate_cast`: no branches from
floating point to integer at run time on the host; worktree `231dec821c13a43c5d60070826dda4bdeb49ebdc`) with `scripts/e6_build_against.sh`; only the FKL include path changes. Run on the RTX 4090 (CUDA 13.4.3), all 2^32 float bit patterns,
five target types. `e6_saturate_exhaustive.csv` has the disagreeing set per class of float.

| type | FKL dev = OpenCV dev = hardware | FKL host = FKL device | OpenCV host function = its device version |
|---|---|---|---|
| uchar | all 2^32 patterns | **all 2^32 patterns** | 822,083,584 patterns differ |
| schar | all | **all** | 1,442,840,576 differ |
| short | all | **all** | 1,442,841,218 differ |
| ushort | all | **all** | 822,084,353 differ |
| int | all | **all** (before: 16,777,215 differ, the NaN patterns and 2^31) | 1,619,001,346 differ |

With the change of #416, FKL's host function and FKL's device function give the same result for every float bit pattern for uchar, schar, short, ushort and int (NaN gives 0 on both, 2^31 gives INT_MAX on both).
The OpenCV host function (`cv::saturate_cast`) differs from its own device version (`cv::cuda::device::saturate_cast`) for the values from 2^31 on (+Inf included): 822,083,584 patterns for uchar, 1,442,840,576 for schar, 1,442,841,218 for short, 822,084,353 for ushort and 1,619,001,346 for int.
