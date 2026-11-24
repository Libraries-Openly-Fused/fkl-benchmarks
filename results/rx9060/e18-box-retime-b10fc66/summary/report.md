# E18 box re-time after the pin move to FKL b10fc66: the summary

A = e18_box_hip at a8f9daa with FKL 24478f0; B = at 2160c70 with FKL b10fc66; old = `results/rx9060/e18-box/` (2026-11-17).

## The times: the median of the three quiet runs' means (min-max), ms

| version | interval | old | A | B |
|---|---|--:|--:|--:|
| `rpp` | whole_call | 0.1677 (0.1656-0.1684), spread 1.7 % | 0.1645 (0.1627-0.1680), spread 3.2 % | 0.1672 (0.1651-0.1700), spread 2.9 % |
| `rpp` | kernel_only | 0.1584 (0.1560-0.1598), spread 2.4 % | 0.1541 (0.1530-0.1543), spread 0.9 % | 0.1542 (0.1540-0.1542), spread 0.1 % |
| `rpp_ltrb` | whole_call | 0.1592 (0.1566-0.1594), spread 1.8 % | 0.1537 (0.1537-0.1549), spread 0.8 % | 0.1537 (0.1534-0.1539), spread 0.3 % |
| `rpp_ltrb` | kernel_only | 0.1555 (0.1517-0.1563), spread 3.0 % | 0.1501 (0.1501-0.1503), spread 0.1 % | 0.1501 (0.1498-0.1501), spread 0.2 % |
| `fkl_box_halfpixel` | whole_call | 0.2300 (0.2289-0.2344), spread 2.4 % | 0.2292 (0.2284-0.2306), spread 1.0 % | 0.2281 (0.2276-0.2312), spread 1.6 % |
| `fkl_box_halfpixel` | kernel_only | 0.2083 (0.2083-0.2099), spread 0.7 % | 0.2064 (0.2062-0.2069), spread 0.4 % | 0.2045 (0.2043-0.2048), spread 0.3 % |
| `fkl_half_once` | whole_call | 0.2318 (0.2318-0.2326), spread 0.4 % | 0.2321 (0.2317-0.2362), spread 1.9 % | 0.2297 (0.2273-0.2308), spread 1.5 % |
| `fkl_half_once` | kernel_only | 0.2079 (0.2075-0.2090), spread 0.7 % | 0.2065 (0.2062-0.2069), spread 0.3 % | 0.2044 (0.2043-0.2048), spread 0.2 % |

## The comparisons

| comparison | version, interval | old (ms) | new (ms) | change | spread old / new | exceeds the larger spread? | bound (new) | flag |
|---|---|--:|--:|--:|--:|---|---|---|
| B against A, this window | `rpp` whole_call | 0.1645 | 0.1672 | +1.7 % | 3.2 / 2.9 % | no | host | no |
| B against A, this window | `rpp` kernel_only | 0.1541 | 0.1542 | +0.0 % | 0.9 / 0.1 % | no | host | no |
| B against A, this window | `rpp_ltrb` whole_call | 0.1537 | 0.1537 | +0.0 % | 0.8 / 0.3 % | no | GPU | no |
| B against A, this window | `rpp_ltrb` kernel_only | 0.1501 | 0.1501 | -0.0 % | 0.1 / 0.2 % | no | GPU | no |
| B against A, this window | `fkl_box_halfpixel` whole_call | 0.2292 | 0.2281 | -0.5 % | 1.0 / 1.6 % | no | host | no |
| B against A, this window | `fkl_box_halfpixel` kernel_only | 0.2064 | 0.2045 | -0.9 % | 0.4 / 0.3 % | yes | host | no |
| B against A, this window | `fkl_half_once` whole_call | 0.2321 | 0.2297 | -1.0 % | 1.9 / 1.5 % | no | host | no |
| B against A, this window | `fkl_half_once` kernel_only | 0.2065 | 0.2044 | -1.0 % | 0.3 / 0.2 % | yes | host | no |
| B against old | `rpp` whole_call | 0.1677 | 0.1672 | -0.3 % | 1.7 / 2.9 % | no | host | no |
| B against old | `rpp` kernel_only | 0.1584 | 0.1542 | -2.6 % | 2.4 / 0.1 % | yes | host | no |
| B against old | `rpp_ltrb` whole_call | 0.1592 | 0.1537 | -3.5 % | 1.8 / 0.3 % | yes | GPU | FLAG (> 3 %) |
| B against old | `rpp_ltrb` kernel_only | 0.1555 | 0.1501 | -3.5 % | 3.0 / 0.2 % | yes | GPU | FLAG (> 3 %) |
| B against old | `fkl_box_halfpixel` whole_call | 0.2300 | 0.2281 | -0.8 % | 2.4 / 1.6 % | no | host | no |
| B against old | `fkl_box_halfpixel` kernel_only | 0.2083 | 0.2045 | -1.9 % | 0.7 / 0.3 % | yes | host | no |
| B against old | `fkl_half_once` whole_call | 0.2318 | 0.2297 | -0.9 % | 0.4 / 1.5 % | no | host | no |
| B against old | `fkl_half_once` kernel_only | 0.2079 | 0.2044 | -1.7 % | 0.7 / 0.2 % | yes | host | no |
| A against old | `rpp` whole_call | 0.1677 | 0.1645 | -2.0 % | 1.7 / 3.2 % | no | host | no |
| A against old | `rpp` kernel_only | 0.1584 | 0.1541 | -2.7 % | 2.4 / 0.9 % | yes | host | no |
| A against old | `rpp_ltrb` whole_call | 0.1592 | 0.1537 | -3.5 % | 1.8 / 0.8 % | yes | GPU | FLAG (> 3 %) |
| A against old | `rpp_ltrb` kernel_only | 0.1555 | 0.1501 | -3.5 % | 3.0 / 0.1 % | yes | GPU | FLAG (> 3 %) |
| A against old | `fkl_box_halfpixel` whole_call | 0.2300 | 0.2292 | -0.3 % | 2.4 / 1.0 % | no | host | no |
| A against old | `fkl_box_halfpixel` kernel_only | 0.2083 | 0.2064 | -0.9 % | 0.7 / 0.4 % | yes | host | no |
| A against old | `fkl_half_once` whole_call | 0.2318 | 0.2321 | +0.1 % | 0.4 / 1.9 % | no | host | no |
| A against old | `fkl_half_once` kernel_only | 0.2079 | 0.2065 | -0.7 % | 0.7 / 0.3 % | no | host | no |

## GPU- or host-bound (kernel_only against whole_call; GPU-bound below 3 %)

| side | version | kernel_only vs whole_call | bound |
|---|---|--:|---|
| old | `rpp` | -5.6 % | host |
| old | `rpp_ltrb` | -2.3 % | GPU |
| old | `fkl_box_halfpixel` | -9.4 % | host |
| old | `fkl_half_once` | -10.3 % | host |
| A | `rpp` | -6.3 % | host |
| A | `rpp_ltrb` | -2.4 % | GPU |
| A | `fkl_box_halfpixel` | -9.9 % | host |
| A | `fkl_half_once` | -11.0 % | host |
| B | `rpp` | -7.8 % | host |
| B | `rpp_ltrb` | -2.4 % | GPU |
| B | `fkl_box_halfpixel` | -10.4 % | host |
| B | `fkl_half_once` | -11.0 % | host |

## The unused runs against their binary's quiet median

- unused-A `rpp` whole_call: -3.4 %
- unused-A `rpp` kernel_only: +5.5 %
- unused-A `rpp_ltrb` whole_call: +3.2 %
- unused-A `rpp_ltrb` kernel_only: +2.2 %
- unused-A `fkl_box_halfpixel` whole_call: +0.9 %
- unused-A `fkl_box_halfpixel` kernel_only: +2.3 %
- unused-A `fkl_half_once` whole_call: -0.4 %
- unused-A `fkl_half_once` kernel_only: +1.4 %
- unused-B `rpp` whole_call: +0.9 %
- unused-B `rpp` kernel_only: +1.9 %
- unused-B `rpp_ltrb` whole_call: -0.8 %
- unused-B `rpp_ltrb` kernel_only: +1.3 %
- unused-B `fkl_box_halfpixel` whole_call: -0.8 %
- unused-B `fkl_box_halfpixel` kernel_only: +0.8 %
- unused-B `fkl_half_once` whole_call: -1.2 %
- unused-B `fkl_half_once` kernel_only: +0.3 %

## The clocks (`clock.csv`; sclk at the interval boundaries in `summary.csv`)

- unused-A: sclk 64 MHz before the warm-up, 3283 after; timing 2.194 s, 18 samples, sclk min / median / max 3180 / 3354 / 3425 MHz, power median 42.0 W, junction max 52.0 °C
- unused-B: sclk 67 MHz before the warm-up, 3345 after; timing 2.192 s, 18 samples, sclk min / median / max 3099 / 3406 / 3470 MHz, power median 46.5 W, junction max 47.0 °C
- quiet-run1-A: sclk 73 MHz before the warm-up, 3450 after; timing 2.192 s, 18 samples, sclk min / median / max 3167 / 3466 / 3473 MHz, power median 48.0 W, junction max 46.0 °C
- quiet-run1-B: sclk 109 MHz before the warm-up, 3450 after; timing 2.189 s, 17 samples, sclk min / median / max 3112 / 3466 / 3473 MHz, power median 50.0 W, junction max 54.0 °C
- quiet-run2-A: sclk 1728 MHz before the warm-up, 3450 after; timing 2.194 s, 17 samples, sclk min / median / max 3313 / 3464 / 3473 MHz, power median 49.0 W, junction max 56.0 °C
- quiet-run2-B: sclk 1665 MHz before the warm-up, 3450 after; timing 2.190 s, 17 samples, sclk min / median / max 3020 / 3467 / 3474 MHz, power median 52.0 W, junction max 54.0 °C
- quiet-run3-A: sclk 1572 MHz before the warm-up, 3445 after; timing 2.193 s, 17 samples, sclk min / median / max 3274 / 3463 / 3470 MHz, power median 50.0 W, junction max 53.0 °C
- quiet-run3-B: sclk 55 MHz before the warm-up, 3437 after; timing 2.190 s, 18 samples, sclk min / median / max 3307 / 3460 / 3470 MHz, power median 49.5 W, junction max 56.0 °C
- rocprofv3-A: sclk 31 MHz before the warm-up, 3295 after; timing 2.211 s, 17 samples, sclk min / median / max 3079 / 3373 / 3425 MHz, power median 46.0 W, junction max 56.0 °C
- rocprofv3-B: sclk 29 MHz before the warm-up, 3283 after; timing 2.209 s, 17 samples, sclk min / median / max 3177 / 3361 / 3424 MHz, power median 48.0 W, junction max 57.0 °C

## The rocprofv3 runs (apart: never in the times above)

Their means against their binary's quiet median (`rocprofv3-times.csv`):

- rocprofv3-A `rpp` whole_call: +10.3 %
- rocprofv3-A `rpp` kernel_only: +9.1 %
- rocprofv3-A `rpp_ltrb` whole_call: +5.8 %
- rocprofv3-A `rpp_ltrb` kernel_only: +8.0 %
- rocprofv3-A `fkl_box_halfpixel` whole_call: +2.1 %
- rocprofv3-A `fkl_box_halfpixel` kernel_only: +6.4 %
- rocprofv3-A `fkl_half_once` whole_call: +2.6 %
- rocprofv3-A `fkl_half_once` kernel_only: +4.3 %
- rocprofv3-B `rpp` whole_call: +7.9 %
- rocprofv3-B `rpp` kernel_only: +8.8 %
- rocprofv3-B `rpp_ltrb` whole_call: +5.5 %
- rocprofv3-B `rpp_ltrb` kernel_only: +7.9 %
- rocprofv3-B `fkl_box_halfpixel` whole_call: +2.3 %
- rocprofv3-B `fkl_box_halfpixel` kernel_only: +5.3 %
- rocprofv3-B `fkl_half_once` whole_call: +2.3 %
- rocprofv3-B `fkl_half_once` kernel_only: +5.5 %

The median GPU duration of each kernel per timed range, us (`kernel-durations.txt`; dispatches in brackets):

| range | kernel | old | A | B |
|---|---|--:|--:|--:|
| whole_call:rpp | `roi_conversion_xywh_to_ltrb` | 2.36 (101) | 2.36 (101) | 2.36 (101) |
| whole_call:rpp | `resize_bilinear_pkd_hip_tensor` | 40.96 (101) | 40.44 (101) | 40.68 (101) |
| whole_call:rpp | `box_filter_3x3_pkd_hip_tensor` | 101.44 (101) | 100.08 (101) | 100.24 (101) |
| kernel_only:rpp | `roi_conversion_xywh_to_ltrb` | 2.60 (101) | 2.36 (101) | 2.40 (101) |
| kernel_only:rpp | `resize_bilinear_pkd_hip_tensor` | 42.76 (101) | 43.28 (101) | 42.76 (101) |
| kernel_only:rpp | `box_filter_3x3_pkd_hip_tensor` | 100.52 (101) | 99.20 (101) | 99.24 (101) |
| kernel_only:rpp | `e18box::gpuSpin` | 5025.89 (100) | 5024.43 (100) | 5024.45 (100) |
| whole_call:rpp_ltrb | `resize_bilinear_pkd_hip_tensor` | 40.52 (101) | 41.12 (101) | 40.20 (101) |
| whole_call:rpp_ltrb | `box_filter_3x3_pkd_hip_tensor` | 100.76 (101) | 100.00 (101) | 100.48 (101) |
| kernel_only:rpp_ltrb | `resize_bilinear_pkd_hip_tensor` | 43.76 (101) | 45.16 (101) | 45.00 (101) |
| kernel_only:rpp_ltrb | `box_filter_3x3_pkd_hip_tensor` | 98.60 (101) | 97.92 (101) | 98.04 (101) |
| kernel_only:rpp_ltrb | `e18box::gpuSpin` | 5025.51 (100) | 5024.33 (100) | 5024.37 (100) |
| whole_call:fkl_box_halfpixel | `fk::linearFilterDPPKernel` | 221.48 (101) | 219.44 (101) | 216.32 (101) |
| kernel_only:fkl_box_halfpixel | `fk::linearFilterDPPKernel` | 204.80 (101) | 202.72 (101) | 200.80 (101) |
| kernel_only:fkl_box_halfpixel | `e18box::gpuSpin` | 5032.25 (100) | 5025.53 (100) | 5025.53 (100) |
| whole_call:fkl_half_once | `fk::linearFilterDPPKernel` | 219.36 (101) | 221.36 (101) | 218.92 (101) |
| kernel_only:fkl_half_once | `fk::linearFilterDPPKernel` | 202.88 (101) | 201.04 (101) | 199.04 (101) |
| kernel_only:fkl_half_once | `e18box::gpuSpin` | 5032.41 (100) | 5024.49 (100) | 5025.53 (100) |

Per call after the spin (kernel_only ranges), median GPU span / sum of kernel durations, us:

| range | old | A | B |
|---|--:|--:|--:|
| kernel_only:rpp | 154.14 / 145.78 | 153.04 / 144.84 | 152.66 / 144.28 |
| kernel_only:rpp_ltrb | 146.48 / 142.32 | 147.04 / 143.00 | 146.96 / 142.88 |
| kernel_only:fkl_box_halfpixel | 204.76 / 204.76 | 202.72 / 202.72 | 200.78 / 200.78 |
| kernel_only:fkl_half_once | 202.86 / 202.86 | 201.04 / 201.04 | 199.04 / 199.04 |

The kernels of every call and the spin coverage of A's and B's traces: `rocprofv3-A-calls.txt`, `rocprofv3-B-calls.txt` (the record's rocprof_calls.py --spin).

## The check rows

35 of 40 files have the record's data rows (`checks-identity.txt`).
