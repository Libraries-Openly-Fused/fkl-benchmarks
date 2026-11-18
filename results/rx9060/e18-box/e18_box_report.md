| version | whole_call (ms) | kernel_only (ms) |
|---|--:|--:|
| rpp | 0.1677 (0.1656-0.1684) | 0.1584 (0.1560-0.1598) |
| rpp_ltrb | 0.1592 (0.1566-0.1594) | 0.1555 (0.1517-0.1563) |
| fkl_box_halfpixel | 0.2300 (0.2289-0.2344) | 0.2083 (0.2083-0.2099) |
| fkl_half_once | 0.2318 (0.2318-0.2326) | 0.2079 (0.2075-0.2090) |

| baseline / FKL | whole_call | kernel_only |
|---|--:|--:|
| rpp / fkl_box_halfpixel | 0.72 (0.72-0.73) | 0.76 (0.74-0.77) |
| rpp / fkl_half_once | 0.72 (0.71-0.73) | 0.76 (0.75-0.77) |
| rpp_ltrb / fkl_box_halfpixel | 0.69 (0.67-0.70) | 0.74 (0.73-0.75) |
| rpp_ltrb / fkl_half_once | 0.69 (0.68-0.69) | 0.75 (0.73-0.75) |

Largest spread of the three runs' means: 3.0 %.
Lowest sclk at an interval boundary: 2904 MHz (BELOW 3100 MHz).

sclk in gpu_clocks.csv during the timing of each run (criterion: at least 3100 MHz):
- quiet-run1: 20 samples in 2.20 s, min 3274, median 3389, max 3451 MHz; below 3100: 0
- quiet-run2: 19 samples in 2.20 s, min 2969, median 3415, max 3463 MHz; below 3100: 2
- quiet-run3: 19 samples in 2.20 s, min 3191, median 3418, max 3465 MHz; below 3100: 0
- rocprofv3: 19 samples in 2.22 s, min 3256, median 3327, max 3396 MHz; below 3100: 0
