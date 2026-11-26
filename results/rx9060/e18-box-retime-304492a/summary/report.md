# E18 box re-time after the pin move to FKL 304492a (RT-A2): the summary

A = e18_box_hip at 976e42b with FKL b10fc66; B = at 73a9650 with FKL 304492a; old = B of `results/rx9060/e18-box-retime-b10fc66/` (#62, 2026-11-24).

## The times: the median of the three quiet runs' means (min-max), ms

| version | interval | old | A | B |
|---|---|--:|--:|--:|
| `rpp` | whole_call | 0.1672 (0.1651-0.1700), spread 2.9 % | 0.1672 (0.1658-0.1700), spread 2.5 % | 0.1700 (0.1669-0.1701), spread 1.9 % |
| `rpp` | kernel_only | 0.1542 (0.1540-0.1542), spread 0.1 % | 0.1543 (0.1528-0.1567), spread 2.5 % | 0.1547 (0.1545-0.1548), spread 0.2 % |
| `rpp_ltrb` | whole_call | 0.1537 (0.1534-0.1539), spread 0.3 % | 0.1544 (0.1536-0.1551), spread 0.9 % | 0.1544 (0.1530-0.1547), spread 1.1 % |
| `rpp_ltrb` | kernel_only | 0.1501 (0.1498-0.1501), spread 0.2 % | 0.1501 (0.1496-0.1515), spread 1.3 % | 0.1502 (0.1500-0.1502), spread 0.1 % |
| `fkl_box_halfpixel` | whole_call | 0.2281 (0.2276-0.2312), spread 1.6 % | 0.2309 (0.2290-0.2315), spread 1.1 % | 0.2300 (0.2298-0.2312), spread 0.6 % |
| `fkl_box_halfpixel` | kernel_only | 0.2045 (0.2043-0.2048), spread 0.3 % | 0.2050 (0.2046-0.2059), spread 0.6 % | 0.2051 (0.2048-0.2052), spread 0.2 % |
| `fkl_half_once` | whole_call | 0.2297 (0.2273-0.2308), spread 1.5 % | 0.2314 (0.2300-0.2326), spread 1.1 % | 0.2321 (0.2303-0.2326), spread 1.0 % |
| `fkl_half_once` | kernel_only | 0.2044 (0.2043-0.2048), spread 0.2 % | 0.2049 (0.2047-0.2050), spread 0.2 % | 0.2050 (0.2045-0.2051), spread 0.3 % |

## The comparisons

| comparison | version, interval | old (ms) | new (ms) | change | spread old / new | exceeds the larger spread? | bound (new) | flag |
|---|---|--:|--:|--:|--:|---|---|---|
| B against A, this window | `rpp` whole_call | 0.1672 | 0.1700 | +1.7 % | 2.5 / 1.9 % | no | host | no |
| B against A, this window | `rpp` kernel_only | 0.1543 | 0.1547 | +0.2 % | 2.5 / 0.2 % | no | host | no |
| B against A, this window | `rpp_ltrb` whole_call | 0.1544 | 0.1544 | -0.0 % | 0.9 / 1.1 % | no | GPU | no |
| B against A, this window | `rpp_ltrb` kernel_only | 0.1501 | 0.1502 | +0.0 % | 1.3 / 0.1 % | no | GPU | no |
| B against A, this window | `fkl_box_halfpixel` whole_call | 0.2309 | 0.2300 | -0.4 % | 1.1 / 0.6 % | no | host | no |
| B against A, this window | `fkl_box_halfpixel` kernel_only | 0.2050 | 0.2051 | +0.1 % | 0.6 / 0.2 % | no | host | no |
| B against A, this window | `fkl_half_once` whole_call | 0.2314 | 0.2321 | +0.3 % | 1.1 / 1.0 % | no | host | no |
| B against A, this window | `fkl_half_once` kernel_only | 0.2049 | 0.2050 | +0.0 % | 0.2 / 0.3 % | no | host | no |
| B against old | `rpp` whole_call | 0.1672 | 0.1700 | +1.7 % | 2.9 / 1.9 % | no | host | no |
| B against old | `rpp` kernel_only | 0.1542 | 0.1547 | +0.3 % | 0.1 / 0.2 % | yes | host | no |
| B against old | `rpp_ltrb` whole_call | 0.1537 | 0.1544 | +0.4 % | 0.3 / 1.1 % | no | GPU | no |
| B against old | `rpp_ltrb` kernel_only | 0.1501 | 0.1502 | +0.0 % | 0.2 / 0.1 % | no | GPU | no |
| B against old | `fkl_box_halfpixel` whole_call | 0.2281 | 0.2300 | +0.8 % | 1.6 / 0.6 % | no | host | no |
| B against old | `fkl_box_halfpixel` kernel_only | 0.2045 | 0.2051 | +0.3 % | 0.3 / 0.2 % | yes | host | no |
| B against old | `fkl_half_once` whole_call | 0.2297 | 0.2321 | +1.1 % | 1.5 / 1.0 % | no | host | no |
| B against old | `fkl_half_once` kernel_only | 0.2044 | 0.2050 | +0.3 % | 0.2 / 0.3 % | no | host | no |
| A against old | `rpp` whole_call | 0.1672 | 0.1672 | -0.0 % | 2.9 / 2.5 % | no | host | no |
| A against old | `rpp` kernel_only | 0.1542 | 0.1543 | +0.1 % | 0.1 / 2.5 % | no | host | no |
| A against old | `rpp_ltrb` whole_call | 0.1537 | 0.1544 | +0.4 % | 0.3 / 0.9 % | no | GPU | no |
| A against old | `rpp_ltrb` kernel_only | 0.1501 | 0.1501 | +0.0 % | 0.2 / 1.3 % | no | GPU | no |
| A against old | `fkl_box_halfpixel` whole_call | 0.2281 | 0.2309 | +1.2 % | 1.6 / 1.1 % | no | host | no |
| A against old | `fkl_box_halfpixel` kernel_only | 0.2045 | 0.2050 | +0.2 % | 0.3 / 0.6 % | no | host | no |
| A against old | `fkl_half_once` whole_call | 0.2297 | 0.2314 | +0.8 % | 1.5 / 1.1 % | no | host | no |
| A against old | `fkl_half_once` kernel_only | 0.2044 | 0.2049 | +0.3 % | 0.2 / 0.2 % | yes | host | no |

## GPU- or host-bound (kernel_only against whole_call; GPU-bound below 3 %)

| side | version | kernel_only vs whole_call | bound |
|---|---|--:|---|
| old | `rpp` | -7.8 % | host |
| old | `rpp_ltrb` | -2.4 % | GPU |
| old | `fkl_box_halfpixel` | -10.4 % | host |
| old | `fkl_half_once` | -11.0 % | host |
| A | `rpp` | -7.7 % | host |
| A | `rpp_ltrb` | -2.8 % | GPU |
| A | `fkl_box_halfpixel` | -11.2 % | host |
| A | `fkl_half_once` | -11.4 % | host |
| B | `rpp` | -9.0 % | host |
| B | `rpp_ltrb` | -2.7 % | GPU |
| B | `fkl_box_halfpixel` | -10.9 % | host |
| B | `fkl_half_once` | -11.7 % | host |

## The unused runs against their binary's quiet median

- unused-A `rpp` whole_call: +1.6 %
- unused-A `rpp` kernel_only: +1.6 %
- unused-A `rpp_ltrb` whole_call: -0.2 %
- unused-A `rpp_ltrb` kernel_only: +0.8 %
- unused-A `fkl_box_halfpixel` whole_call: -0.2 %
- unused-A `fkl_box_halfpixel` kernel_only: +0.4 %
- unused-A `fkl_half_once` whole_call: +0.4 %
- unused-A `fkl_half_once` kernel_only: -0.2 %
- unused-B `rpp` whole_call: -2.6 %
- unused-B `rpp` kernel_only: +0.1 %
- unused-B `rpp_ltrb` whole_call: -0.5 %
- unused-B `rpp_ltrb` kernel_only: -0.0 %
- unused-B `fkl_box_halfpixel` whole_call: +0.5 %
- unused-B `fkl_box_halfpixel` kernel_only: -0.1 %
- unused-B `fkl_half_once` whole_call: -1.0 %
- unused-B `fkl_half_once` kernel_only: +0.0 %

## The clocks (`clock.csv`; sclk at the interval boundaries in `summary.csv`)

- unused-A: sclk 99 MHz before the warm-up, 3354 after; timing 2.192 s, 17 samples, sclk min / median / max 3336 / 3410 / 3461 MHz, power median 52.0 W, junction max 58.0 °C
- unused-B: sclk 115 MHz before the warm-up, 3427 after; timing 2.193 s, 17 samples, sclk min / median / max 3074 / 3455 / 3465 MHz, power median 54.0 W, junction max 62.0 °C
- quiet-run1-A: sclk 63 MHz before the warm-up, 3431 after; timing 2.191 s, 18 samples, sclk min / median / max 3320 / 3458 / 3466 MHz, power median 55.0 W, junction max 61.0 °C
- quiet-run1-B: sclk 60 MHz before the warm-up, 3430 after; timing 2.192 s, 17 samples, sclk min / median / max 3182 / 3456 / 3464 MHz, power median 50.0 W, junction max 56.0 °C
- quiet-run2-A: sclk 94 MHz before the warm-up, 3445 after; timing 2.190 s, 17 samples, sclk min / median / max 3117 / 3459 / 3468 MHz, power median 54.0 W, junction max 62.0 °C
- quiet-run2-B: sclk 1439 MHz before the warm-up, 3439 after; timing 2.191 s, 17 samples, sclk min / median / max 3075 / 3457 / 3465 MHz, power median 51.0 W, junction max 57.0 °C
- quiet-run3-A: sclk 59 MHz before the warm-up, 3357 after; timing 2.190 s, 18 samples, sclk min / median / max 3212 / 3408 / 3463 MHz, power median 49.0 W, junction max 58.0 °C
- quiet-run3-B: sclk 1387 MHz before the warm-up, 3440 after; timing 2.191 s, 17 samples, sclk min / median / max 3092 / 3455 / 3465 MHz, power median 53.0 W, junction max 59.0 °C
- rocprofv3-A: sclk 31 MHz before the warm-up, 3282 after; timing 2.211 s, 18 samples, sclk min / median / max 3219 / 3366 / 3429 MHz, power median 47.0 W, junction max 57.0 °C
- rocprofv3-B: sclk 103 MHz before the warm-up, 3277 after; timing 2.210 s, 17 samples, sclk min / median / max 3281 / 3354 / 3408 MHz, power median 49.0 W, junction max 59.0 °C

## The rocprofv3 runs (apart: never in the times above)

Their means against their binary's quiet median (`rocprofv3-times.csv`):

- rocprofv3-A `rpp` whole_call: +6.2 %
- rocprofv3-A `rpp` kernel_only: +9.5 %
- rocprofv3-A `rpp_ltrb` whole_call: +5.5 %
- rocprofv3-A `rpp_ltrb` kernel_only: +8.1 %
- rocprofv3-A `fkl_box_halfpixel` whole_call: +2.0 %
- rocprofv3-A `fkl_box_halfpixel` kernel_only: +6.2 %
- rocprofv3-A `fkl_half_once` whole_call: +1.6 %
- rocprofv3-A `fkl_half_once` kernel_only: +4.4 %
- rocprofv3-B `rpp` whole_call: +5.2 %
- rocprofv3-B `rpp` kernel_only: +10.5 %
- rocprofv3-B `rpp_ltrb` whole_call: +5.6 %
- rocprofv3-B `rpp_ltrb` kernel_only: +8.1 %
- rocprofv3-B `fkl_box_halfpixel` whole_call: +0.8 %
- rocprofv3-B `fkl_box_halfpixel` kernel_only: +5.2 %
- rocprofv3-B `fkl_half_once` whole_call: +1.3 %
- rocprofv3-B `fkl_half_once` kernel_only: +4.7 %

The median GPU duration of each kernel per timed range, us (`kernel-durations.txt`; dispatches in brackets):

| range | kernel | old | A | B |
|---|---|--:|--:|--:|
| whole_call:rpp | `roi_conversion_xywh_to_ltrb` | 2.36 (101) | 2.36 (101) | 2.36 (101) |
| whole_call:rpp | `resize_bilinear_pkd_hip_tensor` | 40.68 (101) | 41.08 (101) | 41.12 (101) |
| whole_call:rpp | `box_filter_3x3_pkd_hip_tensor` | 100.24 (101) | 100.60 (101) | 100.36 (101) |
| kernel_only:rpp | `roi_conversion_xywh_to_ltrb` | 2.40 (101) | 2.40 (101) | 2.36 (101) |
| kernel_only:rpp | `resize_bilinear_pkd_hip_tensor` | 42.76 (101) | 44.32 (101) | 43.96 (101) |
| kernel_only:rpp | `box_filter_3x3_pkd_hip_tensor` | 99.24 (101) | 99.12 (101) | 99.20 (101) |
| kernel_only:rpp | `e18box::gpuSpin` | 5024.45 (100) | 5024.41 (100) | 5024.27 (100) |
| whole_call:rpp_ltrb | `resize_bilinear_pkd_hip_tensor` | 40.20 (101) | 41.36 (101) | 40.64 (101) |
| whole_call:rpp_ltrb | `box_filter_3x3_pkd_hip_tensor` | 100.48 (101) | 99.84 (101) | 100.72 (101) |
| kernel_only:rpp_ltrb | `resize_bilinear_pkd_hip_tensor` | 45.00 (101) | 45.24 (101) | 45.28 (101) |
| kernel_only:rpp_ltrb | `box_filter_3x3_pkd_hip_tensor` | 98.04 (101) | 97.92 (101) | 98.12 (101) |
| kernel_only:rpp_ltrb | `e18box::gpuSpin` | 5024.37 (100) | 5024.37 (100) | 5024.45 (100) |
| whole_call:fkl_box_halfpixel | `fk::linearFilterDPPKernel` | 216.32 (101) | 219.60 (101) |  |
| kernel_only:fkl_box_halfpixel | `fk::linearFilterDPPKernel` | 200.80 (101) | 200.88 (101) |  |
| kernel_only:fkl_box_halfpixel | `e18box::gpuSpin` | 5025.53 (100) | 5025.55 (100) | 5025.33 (100) |
| whole_call:fkl_half_once | `fk::linearFilterDPPKernel` | 218.92 (101) | 219.08 (101) |  |
| kernel_only:fkl_half_once | `fk::linearFilterDPPKernel` | 199.04 (101) | 199.48 (101) |  |
| kernel_only:fkl_half_once | `e18box::gpuSpin` | 5025.53 (100) | 5025.53 (100) | 5024.49 (100) |
| whole_call:fkl_box_halfpixel | `fk::launchDPP_Kernel` |  |  | 218.32 (101) |
| kernel_only:fkl_box_halfpixel | `fk::launchDPP_Kernel` |  |  | 201.24 (101) |
| whole_call:fkl_half_once | `fk::launchDPP_Kernel` |  |  | 219.12 (101) |
| kernel_only:fkl_half_once | `fk::launchDPP_Kernel` |  |  | 200.32 (101) |

Per call after the spin (kernel_only ranges), median GPU span / sum of kernel durations, us:

| range | old | A | B |
|---|--:|--:|--:|
| kernel_only:rpp | 152.66 / 144.28 | 153.92 / 145.72 | 153.42 / 145.54 |
| kernel_only:rpp_ltrb | 146.96 / 142.88 | 147.26 / 143.18 | 147.20 / 143.22 |
| kernel_only:fkl_box_halfpixel | 200.78 / 200.78 | 200.88 / 200.88 | 201.24 / 201.24 |
| kernel_only:fkl_half_once | 199.04 / 199.04 | 199.48 / 199.48 | 200.32 / 200.32 |

The kernels of every call and the spin coverage of A's and B's traces: `rocprofv3-A-calls.txt`, `rocprofv3-B-calls.txt` (the record's rocprof_calls.py --spin).

## The check rows

40 of 40 files have the record's data rows (`checks-identity.txt`).
