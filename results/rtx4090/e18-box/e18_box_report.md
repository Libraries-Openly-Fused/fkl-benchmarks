| version | interval | median of the 3 means (ms) | min-max of the means (ms) | spread | host preparation (ms, median) |
|---|---|--:|--:|--:|--:|
| fkl_box_asym | whole_call | 0.0420 | 0.0420-0.0421 | 0.3 % |  |
| fkl_box_asym | kernel_only | 0.0416 | 0.0415-0.0421 | 1.3 % | 0.0001 |
| fkl_box_halfpixel | whole_call | 0.0406 | 0.0406-0.0406 | 0.1 % |  |
| fkl_box_halfpixel | kernel_only | 0.0420 | 0.0420-0.0420 | 0.0 % | 0.0001 |
| fkl_trunc_asym | whole_call | 0.0393 | 0.0393-0.0394 | 0.3 % |  |
| fkl_trunc_asym | kernel_only | 0.0403 | 0.0402-0.0403 | 0.3 % | 0.0001 |
| fkl_trunc_halfpixel | whole_call | 0.0396 | 0.0395-0.0396 | 0.2 % |  |
| fkl_trunc_halfpixel | kernel_only | 0.0409 | 0.0409-0.0410 | 0.1 % | 0.0001 |
| opencv_cuda | whole_call | 0.0484 | 0.0484-0.0485 | 0.2 % |  |
| opencv_cuda | kernel_only | 0.0492 | 0.0492-0.0492 | 0.0 % | 0.0001 |
| npp | whole_call | 0.0336 | 0.0335-0.0336 | 0.1 % |  |
| npp | kernel_only | 0.0341 | 0.0340-0.0341 | 0.2 % | 0.0000 |
| fkl_asym_once | whole_call | 0.0403 | 0.0402-0.0403 | 0.1 % |  |
| fkl_asym_once | kernel_only | 0.0416 | 0.0415-0.0416 | 0.2 % | 0.0000 |
| fkl_half_once | whole_call | 0.0409 | 0.0409-0.0409 | 0.0 % |  |
| fkl_half_once | kernel_only | 0.0420 | 0.0420-0.0420 | 0.1 % | 0.0000 |
| fkl_trunc_asym_once | whole_call | 0.0393 | 0.0393-0.0393 | 0.1 % |  |
| fkl_trunc_asym_once | kernel_only | 0.0403 | 0.0403-0.0403 | 0.1 % | 0.0000 |
| fkl_trunc_halfpixel_once | whole_call | 0.0395 | 0.0395-0.0396 | 0.1 % |  |
| fkl_trunc_halfpixel_once | kernel_only | 0.0409 | 0.0409-0.0410 | 0.0 % | 0.0000 |

| baseline / FKL | interval | median speedup | min-max |
|---|---|--:|--:|
| opencv_cuda / fkl_trunc_asym | whole_call | 1.23 | 1.23-1.23 |
| opencv_cuda / fkl_trunc_asym | kernel_only | 1.22 | 1.22-1.22 |
| npp / fkl_trunc_halfpixel | whole_call | 0.85 | 0.85-0.85 |
| npp / fkl_trunc_halfpixel | kernel_only | 0.83 | 0.83-0.83 |
| opencv_cuda / fkl_box_asym | whole_call | 1.15 | 1.15-1.16 |
| opencv_cuda / fkl_box_asym | kernel_only | 1.18 | 1.17-1.18 |
| npp / fkl_box_halfpixel | whole_call | 0.83 | 0.83-0.83 |
| npp / fkl_box_halfpixel | kernel_only | 0.81 | 0.81-0.81 |
| opencv_cuda / fkl_trunc_asym_once | whole_call | 1.23 | 1.23-1.23 |
| opencv_cuda / fkl_trunc_asym_once | kernel_only | 1.22 | 1.22-1.22 |
| npp / fkl_trunc_halfpixel_once | whole_call | 0.85 | 0.85-0.85 |
| npp / fkl_trunc_halfpixel_once | kernel_only | 0.83 | 0.83-0.83 |
| opencv_cuda / fkl_asym_once | whole_call | 1.20 | 1.20-1.20 |
| opencv_cuda / fkl_asym_once | kernel_only | 1.18 | 1.18-1.18 |
| npp / fkl_half_once | whole_call | 0.82 | 0.82-0.82 |
| npp / fkl_half_once | kernel_only | 0.81 | 0.81-0.81 |

| version | kernel_only / whole_call (median of the 3 per-run ratios) | min-max |
|---|--:|--:|
| fkl_box_asym | 0.9907 | 0.9890-0.9994 |
| fkl_box_halfpixel | 1.0341 | 1.0335-1.0345 |
| fkl_trunc_asym | 1.0237 | 1.0233-1.0269 |
| fkl_trunc_halfpixel | 1.0337 | 1.0336-1.0361 |
| opencv_cuda | 1.0146 | 1.0136-1.0151 |
| npp | 1.0150 | 1.0146-1.0154 |
| fkl_asym_once | 1.0318 | 1.0317-1.0325 |
| fkl_half_once | 1.0264 | 1.0257-1.0266 |
| fkl_trunc_asym_once | 1.0248 | 1.0239-1.0259 |
| fkl_trunc_halfpixel_once | 1.0360 | 1.0352-1.0364 |

Largest spread: 1.3 % (fkl_box_asym, kernel_only).
