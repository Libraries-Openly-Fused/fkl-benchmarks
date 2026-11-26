# E19 crop sweep, RX 9060: the timing summary (fkl_loop is in the raw CSVs only)

## The times: the median of the three quiet runs' means (min-max), ms

| N | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |
|--:|--:|--:|--:|--:|
| 1 | 0.0138 (0.0136-0.0139) | 0.0163 (0.0163-0.0163) | 0.0258 (0.0233-0.0283) | 0.0279 (0.0279-0.0279) |
| 10 | 0.0492 (0.0474-0.0518) | 0.0510 (0.0486-0.0517) | 0.0838 (0.0800-0.0844) | 0.0793 (0.0793-0.0793) |
| 20 | 0.0770 (0.0767-0.0791) | 0.0715 (0.0708-0.0715) | 0.1331 (0.1296-0.1332) | 0.1351 (0.1347-0.1372) |
| 30 | 0.1095 (0.1073-0.1098) | 0.1019 (0.1016-0.1019) | 0.1849 (0.1847-0.1853) | 0.1858 (0.1858-0.1859) |
| 40 | 0.1460 (0.1450-0.1464) | 0.1316 (0.1314-0.1319) | 0.2626 (0.2573-0.2628) | 0.2576 (0.2509-0.2578) |
| 50 | 0.1773 (0.1772-0.1782) | 0.1616 (0.1611-0.1617) | 0.3590 (0.3580-0.3595) | 0.3533 (0.3531-0.3537) |
| 60 | 0.2125 (0.2118-0.2171) | 0.1955 (0.1949-0.1963) | 0.4273 (0.4265-0.4279) | 0.4213 (0.4209-0.4213) |
| 70 | 0.2547 (0.2536-0.2549) | 0.2339 (0.2321-0.2367) | 0.4992 (0.4983-0.5071) | 0.4955 (0.4930-0.4965) |
| 80 | 0.2923 (0.2914-0.2941) | 0.2698 (0.2697-0.2699) | 0.5788 (0.5770-0.5807) | 0.5744 (0.5729-0.5762) |
| 90 | 0.3239 (0.3239-0.3274) | 0.3018 (0.3018-0.3019) | 0.6737 (0.6726-0.6762) | 0.6675 (0.6665-0.6689) |
| 100 | 0.3583 (0.3568-0.3610) | 0.3328 (0.3321-0.3354) | 0.7598 (0.7593-0.7632) | 0.7509 (0.7507-0.7512) |
| 110 | 0.3930 (0.3917-0.3948) | 0.3626 (0.3625-0.3628) | 0.8473 (0.8461-0.8475) | 0.8387 (0.8387-0.8397) |
| 120 | 0.4256 (0.4252-0.4299) | 0.3938 (0.3933-0.3940) | 0.9358 (0.9356-0.9362) | 0.9280 (0.9276-0.9284) |
| 130 | 0.4589 (0.4573-0.4607) | 0.4221 (0.4219-0.4222) | 1.0116 (1.0109-1.0132) | 1.0054 (0.9991-1.0057) |
| 140 | 0.4887 (0.4878-0.4915) | 0.4507 (0.4507-0.4508) | 1.0863 (1.0862-1.0883) | 1.0808 (1.0805-1.0820) |
| 150 | 0.5212 (0.5207-0.5246) | 0.4800 (0.4799-0.4802) | 1.1594 (1.1593-1.1596) | 1.1536 (1.1535-1.1541) |

The largest spread of the three runs' means: 19.6 % (rpp, whole_call, N = 1).

## rpp / fkl_batch: the median of the three per-run ratios (min-max)

| N | whole_call | kernel_only |
|--:|--:|--:|
| 1 | 1.86 (1.71-2.05) | 1.71 (1.71-1.71) |
| 10 | 1.69 (1.63-1.70) | 1.55 (1.53-1.63) |
| 20 | 1.68 (1.68-1.74) | 1.90 (1.89-1.92) |
| 30 | 1.69 (1.68-1.72) | 1.82 (1.82-1.83) |
| 40 | 1.80 (1.77-1.80) | 1.95 (1.91-1.96) |
| 50 | 2.02 (2.01-2.03) | 2.19 (2.18-2.20) |
| 60 | 2.01 (1.96-2.02) | 2.16 (2.15-2.16) |
| 70 | 1.97 (1.96-1.99) | 2.11 (2.09-2.14) |
| 80 | 1.98 (1.96-1.99) | 2.13 (2.12-2.14) |
| 90 | 2.08 (2.07-2.08) | 2.21 (2.21-2.22) |
| 100 | 2.12 (2.10-2.14) | 2.26 (2.24-2.26) |
| 110 | 2.16 (2.15-2.16) | 2.31 (2.31-2.32) |
| 120 | 2.20 (2.18-2.20) | 2.36 (2.35-2.36) |
| 130 | 2.20 (2.19-2.22) | 2.38 (2.37-2.38) |
| 140 | 2.22 (2.21-2.23) | 2.40 (2.40-2.40) |
| 150 | 2.23 (2.21-2.23) | 2.40 (2.40-2.40) |

## GPU- or host-bound: kernel_only against whole_call of the medians (GPU-bound below 3 %)

| N | fkl_batch | rpp |
|--:|--:|--:|
| 1 | +17.72 % (host) | +7.89 % (host) |
| 10 | +3.59 % (host) | -5.41 % (host) |
| 20 | -7.12 % (host) | +1.47 % (GPU) |
| 30 | -6.97 % (host) | +0.49 % (GPU) |
| 40 | -9.88 % (host) | -1.91 % (GPU) |
| 50 | -8.85 % (host) | -1.60 % (GPU) |
| 60 | -7.98 % (host) | -1.40 % (GPU) |
| 70 | -8.17 % (host) | -0.75 % (GPU) |
| 80 | -7.71 % (host) | -0.76 % (GPU) |
| 90 | -6.81 % (host) | -0.92 % (GPU) |
| 100 | -7.12 % (host) | -1.17 % (GPU) |
| 110 | -7.73 % (host) | -1.01 % (GPU) |
| 120 | -7.47 % (host) | -0.83 % (GPU) |
| 130 | -8.01 % (host) | -0.61 % (GPU) |
| 140 | -7.78 % (host) | -0.51 % (GPU) |
| 150 | -7.90 % (host) | -0.50 % (GPU) |

## The clock, power and temperature in the quiet runs' timing (hwmon every 100 ms)

| N | samples | sclk min / median / max (MHz) | below 3.1 GHz | median power (W) | max edge / junction / memory (°C) | sclk after the 2 s warm-up (MHz) |
|--:|--:|---|--:|--:|---|---|
| 1 | 36 | 3471 / 3476/3476/3476 / 3478 | 0 | 38/38/38 | 50 / 53 / 46 | 3477/3476/3476 |
| 10 | 39 | 3382 / 3473/3475/3474 / 3477 | 0 | 45/40/44 | 50 / 54 / 48 | 3477/3476/3477 |
| 20 | 39 | 3103 / 3472/3475/3473 / 3476 | 0 | 47/48/48 | 51 / 57 / 48 | 3473/3471/3477 |
| 30 | 39 | 3249 / 3469/3469/3471 / 3476 | 0 | 50/50/49 | 52 / 59 / 50 | 3476/3476/3477 |
| 40 | 39 | 3358 / 3468/3462/3469 / 3476 | 0 | 52/53/52 | 53 / 57 / 50 | 3476/3468/3476 |
| 50 | 42 | 3075 / 3468/3464/3450 / 3475 | 1 | 52/54/54 | 52 / 60 / 50 | 3476/3477/3470 |
| 60 | 42 | 3143 / 3448/3452/3460 / 3470 | 0 | 55/58/56 | 52 / 63 / 52 | 3471/3476/3476 |
| 70 | 43 | 3042 / 3456/3458/3444 / 3469 | 2 | 58/58/54 | 52 / 63 / 52 | 3476/3466/3471 |
| 80 | 44 | 3007 / 3453/3438/3459 / 3474 | 2 | 60/60/58 | 52 / 59 / 52 | 3471/3471/3476 |
| 90 | 44 | 3058 / 3435/3450/3450 / 3472 | 1 | 63/61/61 | 51 / 60 / 52 | 3439/3477/3477 |
| 100 | 45 | 3242 / 3453/3448/3446 / 3469 | 0 | 53/61/62 | 52 / 58 / 52 | 3476/3477/3477 |
| 110 | 46 | 3213 / 3436/3447/3439 / 3469 | 0 | 64/54/58 | 52 / 61 / 52 | 3477/3476/3472 |
| 120 | 46 | 3088 / 3426/3440/3440 / 3472 | 1 | 65/64/66 | 52 / 59 / 52 | 3476/3472/3478 |
| 130 | 48 | 3065 / 3424/3423/3438 / 3469 | 1 | 64/62/64 | 52 / 64 / 52 | 3476/3477/3477 |
| 140 | 49 | 3024 / 3437/3431/3429 / 3475 | 1 | 63/66/65 | 52 / 60 / 54 | 3476/3476/3477 |
| 150 | 50 | 3100 / 3428/3428/3430 / 3468 | 0 | 67/66/67 | 51 / 62 / 52 | 3475/3477/3479 |

## sclk at the interval boundaries (before-after, min-max over the three quiet runs, MHz)

| N | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |
|--:|---|---|---|---|
| 1 | 3476-3477 / 3431-3448 | 3431-3448 / 3472-3478 | 3472-3478 / 3471-3475 | 3471-3475 / 3476-3477 |
| 10 | 3471-3473 / 3333-3395 | 3333-3395 / 3473-3477 | 3473-3477 / 3395-3427 | 3395-3427 / 3470-3475 |
| 20 | 3453-3465 / 3110-3277 | 3110-3277 / 3475-3476 | 3475-3476 / 3381-3409 | 3381-3409 / 3474-3476 |
| 30 | 3452-3476 / 2981-3240 | 2981-3240 / 3470-3477 | 3470-3477 / 3399-3412 | 3399-3412 / 3467-3476 |
| 40 | 3450-3472 / 3093-3134 | 3093-3134 / 3470-3473 | 3470-3473 / 3409-3429 | 3409-3429 / 3471-3475 |
| 50 | 3438-3443 / 2917-3077 | 2917-3077 / 3467-3473 | 3467-3473 / 3413-3423 | 3413-3423 / 3470-3475 |
| 60 | 3437-3445 / 2969-3079 | 2969-3079 / 3463-3465 | 3463-3465 / 3401-3419 | 3401-3419 / 3466-3470 |
| 70 | 3428-3440 / 2923-3022 | 2923-3022 / 3462-3464 | 3462-3464 / 3402-3413 | 3402-3413 / 3469-3473 |
| 80 | 3427-3434 / 3007-3039 | 3007-3039 / 3463-3470 | 3463-3470 / 3418-3429 | 3418-3429 / 3462-3470 |
| 90 | 3424-3470 / 2993-3058 | 2993-3058 / 3457-3461 | 3457-3461 / 3421-3429 | 3421-3429 / 3468-3469 |
| 100 | 3411-3424 / 3022-3044 | 3022-3044 / 3434-3466 | 3434-3466 / 3411-3422 | 3411-3422 / 3465-3470 |
| 110 | 3408-3417 / 2940-3037 | 2940-3037 / 3443-3464 | 3443-3464 / 3411-3426 | 3411-3426 / 3466-3470 |
| 120 | 3407-3446 / 2986-3031 | 2986-3031 / 3446-3451 | 3446-3451 / 3350-3436 | 3350-3436 / 3465-3472 |
| 130 | 3403-3410 / 2961-3013 | 2961-3013 / 3426-3436 | 3426-3436 / 3426-3435 | 3426-3435 / 3467-3469 |
| 140 | 3374-3416 / 2999-3026 | 2999-3026 / 3399-3437 | 3399-3437 / 3419-3432 | 3419-3432 / 3465-3466 |
| 150 | 3378-3391 / 2993-3023 | 2993-3023 / 3437-3458 | 3437-3458 / 3429-3437 | 3429-3437 / 3464-3476 |

## The N = 30 repeat against results/rx9060/e19-batched-box/

| row, interval | old (ms) | new (ms) | change | spread old / new | exceeds the larger spread? | > 3 %? |
|---|--:|--:|--:|--:|---|---|
| fkl_batch whole_call | 0.1136 | 0.1095 | -3.6 % | 0.8 / 2.2 % | yes | FLAG |
| fkl_batch kernel_only | 0.1029 | 0.1019 | -1.0 % | 0.7 / 0.3 % | yes | no |
| rpp whole_call | 0.1887 | 0.1849 | -2.0 % | 0.4 / 0.3 % | yes | no |
| rpp kernel_only | 0.2010 | 0.1858 | -7.6 % | 0.2 / 0.1 % | yes | FLAG |

## The unused runs (warm-up, rocprofv3) against the quiet medians: 59 of 128 values above 3 %

- warmup: change -3.3 to +4.7 %; above 3 %: 3 of 64
- rocprofv3: change +1.3 to +63.1 %; above 3 %: 56 of 64
  - FLAG warmup N = 10 fkl_batch whole_call: 0.051550 ms against 0.049248 (+4.68 %)
  - FLAG warmup N = 20 rpp whole_call: 0.128718 ms against 0.133107 (-3.30 %)
  - FLAG warmup N = 20 rpp kernel_only: 0.130754 ms against 0.135068 (-3.19 %)
  - FLAG rocprofv3 N = 1 fkl_batch whole_call: 0.020050 ms against 0.013850 (+44.77 %)
  - FLAG rocprofv3 N = 1 fkl_batch kernel_only: 0.024193 ms against 0.016304 (+48.39 %)
  - FLAG rocprofv3 N = 1 rpp whole_call: 0.041232 ms against 0.025832 (+59.62 %)
  - FLAG rocprofv3 N = 1 rpp kernel_only: 0.045462 ms against 0.027869 (+63.13 %)
  - FLAG rocprofv3 N = 10 fkl_batch whole_call: 0.058261 ms against 0.049248 (+18.30 %)
  - FLAG rocprofv3 N = 10 fkl_batch kernel_only: 0.059797 ms against 0.051013 (+17.22 %)
  - FLAG rocprofv3 N = 10 rpp whole_call: 0.121984 ms against 0.083823 (+45.53 %)
  - FLAG rocprofv3 N = 10 rpp kernel_only: 0.122292 ms against 0.079290 (+54.23 %)
  - FLAG rocprofv3 N = 20 fkl_batch whole_call: 0.087491 ms against 0.077002 (+13.62 %)
  - FLAG rocprofv3 N = 20 fkl_batch kernel_only: 0.078399 ms against 0.071521 (+9.62 %)
  - FLAG rocprofv3 N = 20 rpp whole_call: 0.190267 ms against 0.133107 (+42.94 %)
  - FLAG rocprofv3 N = 20 rpp kernel_only: 0.186972 ms against 0.135068 (+38.43 %)
  - FLAG rocprofv3 N = 30 fkl_batch whole_call: 0.117253 ms against 0.109540 (+7.04 %)
  - FLAG rocprofv3 N = 30 fkl_batch kernel_only: 0.108949 ms against 0.101901 (+6.92 %)
  - FLAG rocprofv3 N = 30 rpp whole_call: 0.262087 ms against 0.184903 (+41.74 %)
  - FLAG rocprofv3 N = 30 rpp kernel_only: 0.258800 ms against 0.185806 (+39.29 %)
  - FLAG rocprofv3 N = 40 fkl_batch whole_call: 0.151630 ms against 0.146019 (+3.84 %)
  - FLAG rocprofv3 N = 40 fkl_batch kernel_only: 0.139025 ms against 0.131596 (+5.65 %)
  - FLAG rocprofv3 N = 40 rpp whole_call: 0.380225 ms against 0.262625 (+44.78 %)
  - FLAG rocprofv3 N = 40 rpp kernel_only: 0.369819 ms against 0.257616 (+43.55 %)
  - FLAG rocprofv3 N = 50 fkl_batch whole_call: 0.182858 ms against 0.177307 (+3.13 %)
  - FLAG rocprofv3 N = 50 fkl_batch kernel_only: 0.169495 ms against 0.161623 (+4.87 %)
  - FLAG rocprofv3 N = 50 rpp whole_call: 0.503019 ms against 0.359040 (+40.10 %)
  - FLAG rocprofv3 N = 50 rpp kernel_only: 0.493929 ms against 0.353295 (+39.81 %)
  - FLAG rocprofv3 N = 60 fkl_batch whole_call: 0.220429 ms against 0.212455 (+3.75 %)
  - FLAG rocprofv3 N = 60 fkl_batch kernel_only: 0.202300 ms against 0.195494 (+3.48 %)
  - FLAG rocprofv3 N = 60 rpp whole_call: 0.594598 ms against 0.427259 (+39.17 %)
  - FLAG rocprofv3 N = 60 rpp kernel_only: 0.591422 ms against 0.421290 (+40.38 %)
  - FLAG rocprofv3 N = 70 fkl_batch kernel_only: 0.241304 ms against 0.233882 (+3.17 %)
  - FLAG rocprofv3 N = 70 rpp whole_call: 0.693422 ms against 0.499203 (+38.91 %)
  - FLAG rocprofv3 N = 70 rpp kernel_only: 0.686758 ms against 0.495462 (+38.61 %)
  - FLAG rocprofv3 N = 80 fkl_batch kernel_only: 0.278440 ms against 0.269764 (+3.22 %)
  - FLAG rocprofv3 N = 80 rpp whole_call: 0.800376 ms against 0.578815 (+38.28 %)
  - FLAG rocprofv3 N = 80 rpp kernel_only: 0.789896 ms against 0.574437 (+37.51 %)
  - FLAG rocprofv3 N = 90 fkl_batch whole_call: 0.333601 ms against 0.323877 (+3.00 %)
  - FLAG rocprofv3 N = 90 fkl_batch kernel_only: 0.311517 ms against 0.301820 (+3.21 %)
  - FLAG rocprofv3 N = 90 rpp whole_call: 0.927757 ms against 0.673724 (+37.71 %)
  - FLAG rocprofv3 N = 90 rpp kernel_only: 0.913205 ms against 0.667493 (+36.81 %)
  - FLAG rocprofv3 N = 100 fkl_batch kernel_only: 0.344309 ms against 0.332765 (+3.47 %)
  - FLAG rocprofv3 N = 100 rpp whole_call: 1.034231 ms against 0.759767 (+36.12 %)
  - FLAG rocprofv3 N = 100 rpp kernel_only: 1.022659 ms against 0.750886 (+36.19 %)
  - FLAG rocprofv3 N = 110 fkl_batch kernel_only: 0.376226 ms against 0.362577 (+3.76 %)
  - FLAG rocprofv3 N = 110 rpp whole_call: 1.146992 ms against 0.847333 (+35.36 %)
  - FLAG rocprofv3 N = 110 rpp kernel_only: 1.135756 ms against 0.838748 (+35.41 %)
  - FLAG rocprofv3 N = 120 fkl_batch kernel_only: 0.407650 ms against 0.393824 (+3.51 %)
  - FLAG rocprofv3 N = 120 rpp whole_call: 1.262738 ms against 0.935771 (+34.94 %)
  - FLAG rocprofv3 N = 120 rpp kernel_only: 1.251575 ms against 0.927989 (+34.87 %)
  - FLAG rocprofv3 N = 130 fkl_batch kernel_only: 0.436884 ms against 0.422138 (+3.49 %)
  - FLAG rocprofv3 N = 130 rpp whole_call: 1.390644 ms against 1.011618 (+37.47 %)
  - FLAG rocprofv3 N = 130 rpp kernel_only: 1.354715 ms against 1.005426 (+34.74 %)
  - FLAG rocprofv3 N = 140 fkl_batch kernel_only: 0.466216 ms against 0.450672 (+3.45 %)
  - FLAG rocprofv3 N = 140 rpp whole_call: 1.507123 ms against 1.086277 (+38.74 %)
  - FLAG rocprofv3 N = 140 rpp kernel_only: 1.452281 ms against 1.080786 (+34.37 %)
  - FLAG rocprofv3 N = 150 fkl_batch kernel_only: 0.503273 ms against 0.479979 (+4.85 %)
  - FLAG rocprofv3 N = 150 rpp whole_call: 1.625672 ms against 1.159363 (+40.22 %)
  - FLAG rocprofv3 N = 150 rpp kernel_only: 1.561787 ms against 1.153598 (+35.38 %)
- warmup: samples below 3.1 GHz in the timing: 4 (N = 100: 1, N = 120: 1, N = 140: 1, N = 150: 1)
- rocprofv3: samples below 3.1 GHz in the timing: 11 (N = 30: 1, N = 50: 1, N = 80: 2, N = 100: 1, N = 110: 1, N = 120: 1, N = 130: 2, N = 140: 1, N = 150: 1)

## The idle waits: 33 of 80 runs started after a wait

- warmup/n110: 15 s
- warmup/n140: 15 s
- quiet-run1/n020: 15 s
- quiet-run1/n040: 15 s
- quiet-run1/n050: 15 s
- quiet-run1/n090: 15 s
- quiet-run1/n100: 15 s
- quiet-run1/n120: 15 s
- quiet-run1/n140: 15 s
- quiet-run1/n150: 15 s
- quiet-run2/n050: 15 s
- quiet-run2/n080: 15 s
- quiet-run2/n090: 15 s
- quiet-run2/n110: 15 s
- quiet-run2/n120: 15 s
- quiet-run2/n150: 15 s
- quiet-run3/n001: 15 s
- quiet-run3/n010: 15 s
- quiet-run3/n020: 15 s
- quiet-run3/n050: 15 s
- quiet-run3/n080: 15 s
- quiet-run3/n090: 15 s
- quiet-run3/n130: 15 s
- quiet-run3/n150: 30 s
- rocprofv3/n020: 15 s
- rocprofv3/n040: 15 s
- rocprofv3/n050: 15 s
- rocprofv3/n060: 15 s
- rocprofv3/n080: 15 s
- rocprofv3/n090: 30 s
- rocprofv3/n110: 15 s
- rocprofv3/n120: 15 s
- rocprofv3/n140: 30 s
