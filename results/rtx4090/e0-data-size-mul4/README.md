# Data-size experiment with every size a multiple of 4 (the best case for OpenCV-CUDA)

`benchmark_image_resolution_MAD_loop_mul4_cu`: the source of `e0-data-size` with the increment 282,272 instead of 282,270 (sizes 100 + 282,272 k, k = 0 ... 59: every one a multiple of 4, 100 to 16,654,148 elements) and only
the two one-channel pairs (8UC1 -> 32FC1 and 32FC1 -> 32FC1); everything else (200 operations, the cvGS fused kernel against 201 OpenCV-CUDA kernels, timing, checks) is the same. It exists because OpenCV-CUDA's one-channel
kernels are vectorised (`cv::transformSmart<4>`) only when the element count is a multiple of 4: see `../e0-data-size-parity/README.md`. The original sweep is kept unchanged in `../e0-data-size`. (The CSV header says
`404da83...-dirty`: the binary was built from the tree of that commit plus the uncommitted source change that was committed afterwards as 7b51168; the run took 152 s.)

| pair | OpenCV-CUDA at 282,372 / 16,654,148 elements (ms) | speedup of cvGS, minimum ... maximum | speedup at the last size | largest step between neighbouring points |
|---|---|---|---|---|
| 8UC1 -> 32FC1 | 0.601 / 29.10 | 130x ... 305x (at 16,654,148) | 305x | 16 % |
| 32FC1 -> 32FC1 | 0.575 / 29.15 | 104x ... 203x (at 16,089,604) | 201x | 13 % |

In the original sweep the same pairs, at sizes that alternate between multiples of 4 and not, give OpenCV-CUDA 53.59 / 53.56 ms at the last size (16,654,030), speedups from 135x / 106x up to 724x / 662x (409x at the last size for the float pair),
and steps of up to 286 % / 281 % between neighbouring points. Restricted to the sizes of the original sweep that are multiples of 4, the original gives 135x to 301x and 106x to 201x: the same range as this sweep. So the curve of the paper's experiment
with OpenCV-CUDA at its best is smooth and rises to about 200x (float) and 300x (8UC1); the original curve mixes that with the scalar-kernel sizes, whose OpenCV-CUDA time is 1.8 times as high at the large sizes (53.6 ms at 16,654,030 elements against 29.2 ms at 16,654,148).
