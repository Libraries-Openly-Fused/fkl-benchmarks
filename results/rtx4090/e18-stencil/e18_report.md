| version | interval | median of the 3 means (ms) | min-max of the means (ms) | spread | host preparation (ms, median) |
|---|---|--:|--:|--:|--:|
| fkl_stencil_asym | whole_call | 0.0385 | 0.0382-0.0386 | 0.8 % |  |
| fkl_stencil_asym | kernel_only | 0.0393 | 0.0392-0.0393 | 0.3 % | 0.0001 |
| fkl_stencil_halfpixel | whole_call | 0.0385 | 0.0384-0.0386 | 0.5 % |  |
| fkl_stencil_halfpixel | kernel_only | 0.0398 | 0.0395-0.0398 | 0.7 % | 0.0001 |
| opencv_cuda | whole_call | 0.1737 | 0.1737-0.1739 | 0.1 % |  |
| opencv_cuda | kernel_only | 0.1687 | 0.1686-0.1691 | 0.3 % | 0.0001 |
| npp | whole_call | 0.1213 | 0.1206-0.1215 | 0.7 % |  |
| npp | kernel_only | 0.1199 | 0.1199-0.1212 | 1.1 % | 0.0000 |
| fkl_stencil_asym_once | whole_call | 0.0387 | 0.0386-0.0387 | 0.1 % |  |
| fkl_stencil_asym_once | kernel_only | 0.0399 | 0.0399-0.0399 | 0.0 % | 0.0000 |
| fkl_stencil_halfpixel_once | whole_call | 0.0388 | 0.0387-0.0388 | 0.3 % |  |
| fkl_stencil_halfpixel_once | kernel_only | 0.0399 | 0.0399-0.0399 | 0.0 % | 0.0000 |

| baseline / FKL | interval | median speedup | min-max |
|---|---|--:|--:|
| opencv_cuda / fkl_stencil_asym | whole_call | 4.52 | 4.51-4.54 |
| opencv_cuda / fkl_stencil_asym | kernel_only | 4.30 | 4.29-4.32 |
| npp / fkl_stencil_halfpixel | whole_call | 3.15 | 3.13-3.16 |
| npp / fkl_stencil_halfpixel | kernel_only | 3.03 | 3.01-3.05 |
| opencv_cuda / fkl_stencil_asym_once | whole_call | 4.49 | 4.49-4.50 |
| opencv_cuda / fkl_stencil_asym_once | kernel_only | 4.23 | 4.22-4.23 |
| npp / fkl_stencil_halfpixel_once | whole_call | 3.13 | 3.10-3.14 |
| npp / fkl_stencil_halfpixel_once | kernel_only | 3.00 | 3.00-3.04 |
