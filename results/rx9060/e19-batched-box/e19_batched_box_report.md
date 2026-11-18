| version | whole_call (ms) | kernel_only (ms) |
|---|--:|--:|
| fkl_loop | 0.2202 (0.2184-0.2210) | 0.2170 (0.2168-0.2171) |
| fkl_batch | 0.1136 (0.1132-0.1141) | 0.1029 (0.1027-0.1034) |
| rpp | 0.1887 (0.1887-0.1896) | 0.2010 (0.2006-0.2011) |

| baseline / FKL | whole_call | kernel_only |
|---|--:|--:|
| rpp / fkl_loop | 0.86 (0.85-0.86) | 0.93 (0.93-0.93) |
| rpp / fkl_batch | 1.66 (1.65-1.68) | 1.95 (1.95-1.95) |
| fkl_loop / fkl_batch | 1.94 (1.91-1.95) | 2.11 (2.10-2.11) |

Largest spread of the three runs' means: 1.2 %.
Lowest sclk at an interval boundary: 2994 MHz (BELOW 3100 MHz).

sclk in gpu_clocks.csv during the timing of each run (criterion: at least 3100 MHz):
- quiet-run1: 13 samples in 1.64 s, min 3351, median 3447, max 3481 MHz; below 3100: 0
- quiet-run2: 13 samples in 1.64 s, min 3229, median 3480, max 3481 MHz; below 3100: 0
- quiet-run3: 13 samples in 1.64 s, min 3255, median 3478, max 3481 MHz; below 3100: 0
- rocprofv3: 13 samples in 1.69 s, min 3322, median 3394, max 3470 MHz; below 3100: 0
