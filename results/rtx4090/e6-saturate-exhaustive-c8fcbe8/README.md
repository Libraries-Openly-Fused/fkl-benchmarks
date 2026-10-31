# E6: fk::SaturateCast<float, T> over all 2^32 float bit patterns, against the final pin c8fcbe8 (RTX 4090, CUDA 13.4.3)

Program `fkl-self/numerics/saturate_exhaustive.cu` (see `../e6-saturate-exhaustive/README.md` for the six comparisons and the classes), built from fkl-benchmarks 51fa9e6 with the library at c8fcbe8 (header lines `# repo`, `# fkl`), run on the RTX 4090
(device half) and the CPU (host half), 109 s. Device and host halves for T in uchar, schar, short, ushort, int.

**Result: all 180 rows (5 types x 6 comparisons x 6 classes of float) are identical, disagreeing count and examples, to the run against PR #416's head 231dec8 (`../e6-saturate-exhaustive-pr416/`).**
- FKL device = OpenCV device = the hardware conversion (`cvt.rni.sat`) on all 2^32 patterns for the five types (columns `fkl_dev_vs_opencv_dev`, `fkl_dev_vs_hw`, `opencv_dev_vs_hw`: 0).
- FKL host = FKL device on all 2^32 patterns for the five types (`fkl_host_vs_fkl_dev`: 0).
- OpenCV's host `cv::saturate_cast` differs from its device version (and from FKL's host) in 822,083,584 patterns for uchar, 1,442,840,576 for schar, 1,442,841,218 for short, 822,084,353 for ushort and 1,619,001,346 for int (the values from 2^31 up, as in the earlier runs).
At the pin 38c4f32 the host function of FKL differed from the device one for int in 16,777,215 patterns (`../e6-saturate-exhaustive/`); the saturate and cmath headers of c8fcbe8 are byte-identical to those of 231dec8.
