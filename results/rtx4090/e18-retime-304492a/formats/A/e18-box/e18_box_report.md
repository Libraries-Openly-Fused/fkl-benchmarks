| version | interval | median of the 3 means (ms) | min-max of the means (ms) | spread | host preparation (ms, median) |
|---|---|--:|--:|--:|--:|
| fkl_box_asym | whole_call | 0.0424 | 0.0423-0.0439 | 3.7 % |  |
| fkl_box_asym | kernel_only | 0.0426 | 0.0419-0.0428 | 2.1 % | 0.0001 |
| fkl_box_halfpixel | whole_call | 0.0410 | 0.0410-0.0410 | 0.1 % |  |
| fkl_box_halfpixel | kernel_only | 0.0422 | 0.0421-0.0422 | 0.1 % | 0.0001 |
| fkl_trunc_asym | whole_call | 0.0395 | 0.0395-0.0395 | 0.2 % |  |
| fkl_trunc_asym | kernel_only | 0.0409 | 0.0409-0.0409 | 0.0 % | 0.0002 |
| fkl_trunc_halfpixel | whole_call | 0.0398 | 0.0398-0.0398 | 0.1 % |  |
| fkl_trunc_halfpixel | kernel_only | 0.0410 | 0.0410-0.0410 | 0.0 % | 0.0002 |
| opencv_cuda | whole_call | 0.0484 | 0.0484-0.0484 | 0.1 % |  |
| opencv_cuda | kernel_only | 0.0492 | 0.0492-0.0492 | 0.0 % | 0.0001 |
| npp | whole_call | 0.0335 | 0.0335-0.0335 | 0.1 % |  |
| npp | kernel_only | 0.0340 | 0.0340-0.0340 | 0.2 % | 0.0000 |
| fkl_asym_once | whole_call | 0.0406 | 0.0405-0.0407 | 0.4 % |  |
| fkl_asym_once | kernel_only | 0.0420 | 0.0420-0.0420 | 0.0 % | 0.0000 |
| fkl_half_once | whole_call | 0.0412 | 0.0411-0.0412 | 0.3 % |  |
| fkl_half_once | kernel_only | 0.0423 | 0.0422-0.0424 | 0.3 % | 0.0000 |
| fkl_trunc_asym_once | whole_call | 0.0395 | 0.0394-0.0395 | 0.1 % |  |
| fkl_trunc_asym_once | kernel_only | 0.0409 | 0.0408-0.0409 | 0.1 % | 0.0000 |
| fkl_trunc_halfpixel_once | whole_call | 0.0398 | 0.0397-0.0398 | 0.3 % |  |
| fkl_trunc_halfpixel_once | kernel_only | 0.0410 | 0.0410-0.0410 | 0.0 % | 0.0000 |

| baseline / FKL | interval | median speedup | min-max |
|---|---|--:|--:|
| opencv_cuda / fkl_trunc_asym | whole_call | 1.23 | 1.22-1.23 |
| opencv_cuda / fkl_trunc_asym | kernel_only | 1.20 | 1.20-1.20 |
| npp / fkl_trunc_halfpixel | whole_call | 0.84 | 0.84-0.84 |
| npp / fkl_trunc_halfpixel | kernel_only | 0.83 | 0.83-0.83 |
| opencv_cuda / fkl_box_asym | whole_call | 1.14 | 1.10-1.14 |
| opencv_cuda / fkl_box_asym | kernel_only | 1.15 | 1.15-1.17 |
| npp / fkl_box_halfpixel | whole_call | 0.82 | 0.82-0.82 |
| npp / fkl_box_halfpixel | kernel_only | 0.81 | 0.81-0.81 |
| opencv_cuda / fkl_trunc_asym_once | whole_call | 1.23 | 1.23-1.23 |
| opencv_cuda / fkl_trunc_asym_once | kernel_only | 1.20 | 1.20-1.20 |
| npp / fkl_trunc_halfpixel_once | whole_call | 0.84 | 0.84-0.85 |
| npp / fkl_trunc_halfpixel_once | kernel_only | 0.83 | 0.83-0.83 |
| opencv_cuda / fkl_asym_once | whole_call | 1.19 | 1.19-1.19 |
| opencv_cuda / fkl_asym_once | kernel_only | 1.17 | 1.17-1.17 |
| npp / fkl_half_once | whole_call | 0.81 | 0.81-0.82 |
| npp / fkl_half_once | kernel_only | 0.80 | 0.80-0.80 |

| version | kernel_only / whole_call (median of the 3 per-run ratios) | min-max |
|---|--:|--:|
| fkl_box_asym | 0.9907 | 0.9711-1.0101 |
| fkl_box_halfpixel | 1.0287 | 1.0266-1.0291 |
| fkl_trunc_asym | 1.0358 | 1.0339-1.0362 |
| fkl_trunc_halfpixel | 1.0295 | 1.0293-1.0301 |
| opencv_cuda | 1.0148 | 1.0147-1.0162 |
| npp | 1.0148 | 1.0145-1.0161 |
| fkl_asym_once | 1.0348 | 1.0318-1.0360 |
| fkl_half_once | 1.0287 | 1.0280-1.0292 |
| fkl_trunc_asym_once | 1.0363 | 1.0343-1.0367 |
| fkl_trunc_halfpixel_once | 1.0299 | 1.0297-1.0327 |

Largest spread: 3.7 % (fkl_box_asym, whole_call).
