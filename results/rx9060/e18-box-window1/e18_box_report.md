| version | whole_call (ms) | kernel_only (ms) |
|---|--:|--:|
| rpp | 0.2450 (0.2443-0.2475) | 0.1931 (0.1864-0.1934) |
| rpp_ltrb | 0.1587 (0.1570-0.1601) | 0.1558 (0.1551-0.1595) |
| fkl_box_halfpixel | 0.2325 (0.2322-0.2334) | 0.2242 (0.2239-0.2242) |
| fkl_half_once | 0.2321 (0.2321-0.2322) | 0.2218 (0.2214-0.2221) |

| baseline / FKL | whole_call | kernel_only |
|---|--:|--:|
| rpp / fkl_box_halfpixel | 1.05 (1.05-1.07) | 0.86 (0.83-0.86) |
| rpp / fkl_half_once | 1.06 (1.05-1.07) | 0.87 (0.84-0.87) |
| rpp_ltrb / fkl_box_halfpixel | 0.68 (0.67-0.69) | 0.70 (0.69-0.71) |
| rpp_ltrb / fkl_half_once | 0.68 (0.68-0.69) | 0.70 (0.70-0.72) |

Largest spread of the three runs' means: 3.6 %.
