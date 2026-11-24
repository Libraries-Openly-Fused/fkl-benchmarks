# E19 RX 9060 repeat check: the summary (A = E19's binary, B = the crop sweep's at N = 30)

## The times: the median of the three quiet runs' means (min-max), ms

| program | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |
|---|--:|--:|--:|--:|
| A | 0.1098 (0.1096-0.1101) | 0.1018 (0.1017-0.1018) | 0.1854 (0.1850-0.1861) | 0.1860 (0.1842-0.1868) |
| B | 0.1093 (0.1092-0.1116) | 0.1018 (0.1017-0.1019) | 0.1849 (0.1824-0.1866) | 0.1859 (0.1798-0.1860) |

Spreads (%): A fkl_batch whole_call 0.5; A fkl_batch kernel_only 0.1; A rpp whole_call 0.6; A rpp kernel_only 1.4; B fkl_batch whole_call 2.2; B fkl_batch kernel_only 0.1; B rpp whole_call 2.3; B rpp kernel_only 3.4

## The comparisons (old / new / change / spread old / new / exceeds the larger spread / > 3 %)

| comparison | row, interval | old (ms) | new (ms) | change | spread old / new | exceeds? | > 3 %? |
|---|---|--:|--:|--:|--:|---|---|
| A against E19's window | fkl_batch whole_call | 0.1136 | 0.1098 | -3.4 % | 0.8 / 0.5 % | yes | FLAG |
| A against E19's window | fkl_batch kernel_only | 0.1029 | 0.1018 | -1.1 % | 0.7 / 0.1 % | yes | no |
| A against E19's window | rpp whole_call | 0.1887 | 0.1854 | -1.8 % | 0.4 / 0.6 % | yes | no |
| A against E19's window | rpp kernel_only | 0.2010 | 0.1860 | -7.5 % | 0.2 / 1.4 % | yes | FLAG |
| B against the crop sweep's N = 30 | fkl_batch whole_call | 0.1116 | 0.1093 | -2.1 % | 1.7 / 2.2 % | no | no |
| B against the crop sweep's N = 30 | fkl_batch kernel_only | 0.1018 | 0.1018 | -0.0 % | 0.2 / 0.1 % | no | no |
| B against the crop sweep's N = 30 | rpp whole_call | 0.1850 | 0.1849 | -0.0 % | 2.7 / 2.3 % | no | no |
| B against the crop sweep's N = 30 | rpp kernel_only | 0.1858 | 0.1859 | +0.1 % | 2.9 / 3.4 % | no | no |
| B against A, this window | fkl_batch whole_call | 0.1098 | 0.1093 | -0.5 % | 0.5 / 2.2 % | no | no |
| B against A, this window | fkl_batch kernel_only | 0.1018 | 0.1018 | +0.1 % | 0.1 / 0.1 % | no | no |
| B against A, this window | rpp whole_call | 0.1854 | 0.1849 | -0.2 % | 0.6 / 2.3 % | no | no |
| B against A, this window | rpp kernel_only | 0.1860 | 0.1859 | -0.1 % | 1.4 / 3.4 % | no | no |

## GPU- or host-bound (kernel_only against whole_call; GPU-bound below 3 %)

- A fkl_batch: whole_call 0.109815 ms, kernel_only 0.101755 ms (-7.34 %): host
- A rpp: whole_call 0.185352 ms, kernel_only 0.186008 ms (+0.35 %): GPU
- B fkl_batch: whole_call 0.109257 ms, kernel_only 0.101812 ms (-6.81 %): host
- B rpp: whole_call 0.184949 ms, kernel_only 0.185858 ms (+0.49 %): GPU

## The clock in each run's timing (hwmon every 100 ms)

| run | samples | sclk min / median / max (MHz) | below 3.1 GHz | median power (W) | max edge / junction / memory (°C) | sclk after the warm-up |
|---|--:|---|--:|--:|---|--:|
| unused-A | 13 | 3217 / 3351 / 3427 | 0 | 44 | 39 / 42 / 38 | 3284 |
| unused-B | 13 | 3377 / 3476 / 3481 | 0 | 46 | 40 / 46 / 38 | 3481 |
| quiet-run1-A | 13 | 3428 / 3480 / 3481 | 0 | 48 | 41 / 44 / 40 | 3482 |
| quiet-run1-B | 13 | 3388 / 3480 / 3481 | 0 | 48 | 42 / 46 / 40 | 3482 |
| quiet-run2-A | 13 | 3365 / 3479 / 3481 | 0 | 48 | 42 / 48 / 40 | 3481 |
| quiet-run2-B | 13 | 3355 / 3478 / 3480 | 0 | 49 | 43 / 48 / 42 | 3481 |
| quiet-run3-A | 13 | 3108 / 3477 / 3481 | 0 | 48 | 44 / 48 / 42 | 3482 |
| quiet-run3-B | 13 | 3217 / 3477 / 3481 | 0 | 48 | 44 / 48 / 42 | 3481 |
| rocprofv3-A | 13 | 3215 / 3462 / 3481 | 0 | 47 | 45 / 49 / 42 | 3408 |
| rocprofv3-B | 13 | 3336 / 3444 / 3481 | 0 | 47 | 44 / 48 / 42 | 3383 |

## The unused runs against their program's quiet median

- unused-A fkl_batch whole_call: 0.110703 ms (+0.81 %)
- unused-A fkl_batch kernel_only: 0.104798 ms (+2.99 %)
- unused-A rpp whole_call: 0.186400 ms (+0.57 %)
- unused-A rpp kernel_only: 0.187615 ms (+0.86 %)
- unused-B fkl_batch whole_call: 0.111295 ms (+1.86 %)
- unused-B fkl_batch kernel_only: 0.100814 ms (-0.98 %)
- unused-B rpp whole_call: 0.180127 ms (-2.61 %)
- unused-B rpp kernel_only: 0.185083 ms (-0.42 %)
- rocprofv3-A fkl_batch whole_call: 0.116854 ms (+6.41 %) FLAG
- rocprofv3-A fkl_batch kernel_only: 0.108189 ms (+6.32 %) FLAG
- rocprofv3-A rpp whole_call: 0.273637 ms (+47.63 %) FLAG
- rocprofv3-A rpp kernel_only: 0.275240 ms (+47.97 %) FLAG
- rocprofv3-B fkl_batch whole_call: 0.118092 ms (+8.09 %) FLAG
- rocprofv3-B fkl_batch kernel_only: 0.108505 ms (+6.57 %) FLAG
- rocprofv3-B rpp whole_call: 0.274947 ms (+48.66 %) FLAG
- rocprofv3-B rpp kernel_only: 0.265084 ms (+42.63 %) FLAG
