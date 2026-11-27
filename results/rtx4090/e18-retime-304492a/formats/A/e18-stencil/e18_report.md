| version | interval | median of the 3 means (ms) | min-max of the means (ms) | spread | host preparation (ms, median) |
|---|---|--:|--:|--:|--:|
| fkl_stencil_asym | whole_call | 0.0404 | 0.0388-0.0409 | 5.4 % |  |
| fkl_stencil_asym | kernel_only | 0.0405 | 0.0399-0.0411 | 2.9 % | 0.0001 |
| fkl_stencil_halfpixel | whole_call | 0.0388 | 0.0387-0.0388 | 0.2 % |  |
| fkl_stencil_halfpixel | kernel_only | 0.0399 | 0.0399-0.0399 | 0.0 % | 0.0001 |
| opencv_cuda | whole_call | 0.1740 | 0.1739-0.1740 | 0.1 % |  |
| opencv_cuda | kernel_only | 0.1684 | 0.1684-0.1690 | 0.4 % | 0.0001 |
| npp | whole_call | 0.1212 | 0.1208-0.1218 | 0.8 % |  |
| npp | kernel_only | 0.1193 | 0.1190-0.1194 | 0.3 % | 0.0000 |
| fkl_stencil_asym_once | whole_call | 0.0389 | 0.0387-0.0390 | 0.7 % |  |
| fkl_stencil_asym_once | kernel_only | 0.0399 | 0.0399-0.0400 | 0.0 % | 0.0000 |
| fkl_stencil_halfpixel_once | whole_call | 0.0390 | 0.0387-0.0391 | 0.8 % |  |
| fkl_stencil_halfpixel_once | kernel_only | 0.0400 | 0.0399-0.0400 | 0.1 % | 0.0000 |

| baseline / FKL | interval | median speedup | min-max |
|---|---|--:|--:|
| opencv_cuda / fkl_stencil_asym | whole_call | 4.30 | 4.25-4.49 |
| opencv_cuda / fkl_stencil_asym | kernel_only | 4.18 | 4.10-4.22 |
| npp / fkl_stencil_halfpixel | whole_call | 3.12 | 3.12-3.14 |
| npp / fkl_stencil_halfpixel | kernel_only | 2.99 | 2.98-2.99 |
| opencv_cuda / fkl_stencil_asym_once | whole_call | 4.47 | 4.47-4.50 |
| opencv_cuda / fkl_stencil_asym_once | kernel_only | 4.22 | 4.22-4.23 |
| npp / fkl_stencil_halfpixel_once | whole_call | 3.12 | 3.09-3.13 |
| npp / fkl_stencil_halfpixel_once | kernel_only | 2.99 | 2.98-2.99 |
