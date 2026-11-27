| version | interval | median of the 3 means (ms) | min-max of the means (ms) | spread | host preparation (ms, median) |
|---|---|--:|--:|--:|--:|
| fkl_box_asym | whole_call | 0.0424 | 0.0422-0.0426 | 0.9 % |  |
| fkl_box_asym | kernel_only | 0.0427 | 0.0422-0.0429 | 1.5 % | 0.0001 |
| fkl_box_halfpixel | whole_call | 0.0408 | 0.0408-0.0408 | 0.1 % |  |
| fkl_box_halfpixel | kernel_only | 0.0420 | 0.0420-0.0421 | 0.2 % | 0.0001 |
| fkl_trunc_asym | whole_call | 0.0394 | 0.0394-0.0394 | 0.2 % |  |
| fkl_trunc_asym | kernel_only | 0.0406 | 0.0406-0.0406 | 0.1 % | 0.0002 |
| fkl_trunc_halfpixel | whole_call | 0.0396 | 0.0396-0.0396 | 0.1 % |  |
| fkl_trunc_halfpixel | kernel_only | 0.0410 | 0.0410-0.0410 | 0.0 % | 0.0001 |
| opencv_cuda | whole_call | 0.0483 | 0.0483-0.0483 | 0.1 % |  |
| opencv_cuda | kernel_only | 0.0492 | 0.0492-0.0492 | 0.0 % | 0.0001 |
| npp | whole_call | 0.0336 | 0.0335-0.0336 | 0.3 % |  |
| npp | kernel_only | 0.0340 | 0.0340-0.0340 | 0.1 % | 0.0000 |
| fkl_asym_once | whole_call | 0.0406 | 0.0405-0.0406 | 0.2 % |  |
| fkl_asym_once | kernel_only | 0.0420 | 0.0420-0.0420 | 0.0 % | 0.0000 |
| fkl_half_once | whole_call | 0.0411 | 0.0410-0.0411 | 0.4 % |  |
| fkl_half_once | kernel_only | 0.0421 | 0.0421-0.0421 | 0.1 % | 0.0000 |
| fkl_trunc_asym_once | whole_call | 0.0394 | 0.0393-0.0394 | 0.3 % |  |
| fkl_trunc_asym_once | kernel_only | 0.0405 | 0.0405-0.0406 | 0.1 % | 0.0000 |
| fkl_trunc_halfpixel_once | whole_call | 0.0396 | 0.0395-0.0396 | 0.1 % |  |
| fkl_trunc_halfpixel_once | kernel_only | 0.0410 | 0.0410-0.0410 | 0.0 % | 0.0000 |

| baseline / FKL | interval | median speedup | min-max |
|---|---|--:|--:|
| opencv_cuda / fkl_trunc_asym | whole_call | 1.23 | 1.22-1.23 |
| opencv_cuda / fkl_trunc_asym | kernel_only | 1.21 | 1.21-1.21 |
| npp / fkl_trunc_halfpixel | whole_call | 0.85 | 0.85-0.85 |
| npp / fkl_trunc_halfpixel | kernel_only | 0.83 | 0.83-0.83 |
| opencv_cuda / fkl_box_asym | whole_call | 1.14 | 1.13-1.15 |
| opencv_cuda / fkl_box_asym | kernel_only | 1.15 | 1.15-1.16 |
| npp / fkl_box_halfpixel | whole_call | 0.82 | 0.82-0.82 |
| npp / fkl_box_halfpixel | kernel_only | 0.81 | 0.81-0.81 |
| opencv_cuda / fkl_trunc_asym_once | whole_call | 1.23 | 1.22-1.23 |
| opencv_cuda / fkl_trunc_asym_once | kernel_only | 1.21 | 1.21-1.21 |
| npp / fkl_trunc_halfpixel_once | whole_call | 0.85 | 0.85-0.85 |
| npp / fkl_trunc_halfpixel_once | kernel_only | 0.83 | 0.83-0.83 |
| opencv_cuda / fkl_asym_once | whole_call | 1.19 | 1.19-1.19 |
| opencv_cuda / fkl_asym_once | kernel_only | 1.17 | 1.17-1.17 |
| npp / fkl_half_once | whole_call | 0.82 | 0.82-0.82 |
| npp / fkl_half_once | kernel_only | 0.81 | 0.81-0.81 |

| version | kernel_only / whole_call (median of the 3 per-run ratios) | min-max |
|---|--:|--:|
| fkl_box_asym | 1.0106 | 0.9913-1.0126 |
| fkl_box_halfpixel | 1.0302 | 1.0295-1.0320 |
| fkl_trunc_asym | 1.0295 | 1.0291-1.0304 |
| fkl_trunc_halfpixel | 1.0347 | 1.0338-1.0349 |
| opencv_cuda | 1.0169 | 1.0167-1.0183 |
| npp | 1.0117 | 1.0113-1.0141 |
| fkl_asym_once | 1.0340 | 1.0335-1.0359 |
| fkl_half_once | 1.0252 | 1.0238-1.0262 |
| fkl_trunc_asym_once | 1.0299 | 1.0282-1.0305 |
| fkl_trunc_halfpixel_once | 1.0354 | 1.0344-1.0358 |

Largest spread: 1.5 % (fkl_box_asym, kernel_only).
