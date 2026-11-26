# E19 crop sweep, RX 9060: the timing summary (fkl_loop is in the raw CSVs only)

## The times: the median of the three quiet runs' means (min-max), ms

| N | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |
|--:|--:|--:|--:|--:|
| 1 | 0.0138 (0.0137-0.0138) | 0.0163 (0.0163-0.0164) | 0.0251 (0.0232-0.0254) | 0.0279 (0.0278-0.0280) |
| 10 | 0.0526 (0.0479-0.0538) | 0.0521 (0.0478-0.0523) | 0.0816 (0.0800-0.0838) | 0.0774 (0.0755-0.0791) |
| 20 | 0.0786 (0.0780-0.0813) | 0.0715 (0.0709-0.0717) | 0.1324 (0.1289-0.1330) | 0.1349 (0.1287-0.1372) |
| 30 | 0.1095 (0.1092-0.1122) | 0.1019 (0.1019-0.1024) | 0.1864 (0.1853-0.1904) | 0.1858 (0.1858-0.1957) |
| 40 | 0.1439 (0.1430-0.1459) | 0.1317 (0.1317-0.1318) | 0.2631 (0.2626-0.2637) | 0.2577 (0.2576-0.2577) |
| 50 | 0.1785 (0.1776-0.1788) | 0.1615 (0.1605-0.1615) | 0.3558 (0.3479-0.3584) | 0.3502 (0.3426-0.3537) |
| 60 | 0.2118 (0.2099-0.2136) | 0.1937 (0.1936-0.1941) | 0.4269 (0.4261-0.4270) | 0.4211 (0.4209-0.4213) |
| 70 | 0.2535 (0.2524-0.2549) | 0.2333 (0.2321-0.2338) | 0.4982 (0.4970-0.4994) | 0.4946 (0.4936-0.4964) |
| 80 | 0.2906 (0.2896-0.2942) | 0.2699 (0.2696-0.2723) | 0.5774 (0.5769-0.5777) | 0.5732 (0.5728-0.5734) |
| 90 | 0.3233 (0.3232-0.3261) | 0.3015 (0.3007-0.3018) | 0.6739 (0.6716-0.6739) | 0.6672 (0.6658-0.6674) |
| 100 | 0.3595 (0.3593-0.3601) | 0.3322 (0.3319-0.3328) | 0.7596 (0.7589-0.7604) | 0.7510 (0.7506-0.7514) |
| 110 | 0.3932 (0.3900-0.3946) | 0.3626 (0.3621-0.3629) | 0.8466 (0.8459-0.8477) | 0.8390 (0.8387-0.8396) |
| 120 | 0.4277 (0.4244-0.4277) | 0.3936 (0.3931-0.3937) | 0.9359 (0.9359-0.9363) | 0.9287 (0.9283-0.9294) |
| 130 | 0.4606 (0.4591-0.4608) | 0.4222 (0.4216-0.4223) | 1.0119 (1.0112-1.0120) | 1.0058 (1.0055-1.0059) |
| 140 | 0.4903 (0.4877-0.4968) | 0.4501 (0.4501-0.4560) | 1.0873 (1.0872-1.0956) | 1.0812 (1.0808-1.1191) |
| 150 | 0.5241 (0.5215-0.5251) | 0.4810 (0.4792-0.4940) | 1.1593 (1.1394-1.1739) | 1.1539 (1.1331-1.1629) |

The largest spread of the three runs' means: 11.3 % (fkl_batch, whole_call, N = 10).

## rpp / fkl_batch: the median of the three per-run ratios (min-max)

| N | whole_call | kernel_only |
|--:|--:|--:|
| 1 | 1.81 (1.69-1.85) | 1.71 (1.70-1.72) |
| 10 | 1.55 (1.49-1.75) | 1.52 (1.44-1.62) |
| 20 | 1.65 (1.63-1.69) | 1.89 (1.81-1.91) |
| 30 | 1.70 (1.69-1.71) | 1.82 (1.82-1.91) |
| 40 | 1.83 (1.80-1.84) | 1.96 (1.95-1.96) |
| 50 | 2.00 (1.95-2.01) | 2.17 (2.14-2.19) |
| 60 | 2.02 (2.00-2.03) | 2.17 (2.17-2.18) |
| 70 | 1.96 (1.95-1.98) | 2.12 (2.12-2.14) |
| 80 | 1.99 (1.96-1.99) | 2.12 (2.11-2.12) |
| 90 | 2.08 (2.07-2.08) | 2.21 (2.21-2.22) |
| 100 | 2.11 (2.11-2.11) | 2.26 (2.26-2.26) |
| 110 | 2.15 (2.15-2.17) | 2.31 (2.31-2.32) |
| 120 | 2.19 (2.19-2.21) | 2.36 (2.36-2.36) |
| 130 | 2.20 (2.20-2.20) | 2.38 (2.38-2.39) |
| 140 | 2.22 (2.21-2.23) | 2.40 (2.40-2.45) |
| 150 | 2.21 (2.18-2.24) | 2.36 (2.35-2.40) |

## GPU- or host-bound: kernel_only against whole_call of the medians (GPU-bound below 3 %)

| N | fkl_batch | rpp |
|--:|--:|--:|
| 1 | +18.39 % (host) | +11.33 % (host) |
| 10 | -0.81 % (GPU) | -5.15 % (host) |
| 20 | -9.07 % (host) | +1.85 % (GPU) |
| 30 | -6.90 % (host) | -0.29 % (GPU) |
| 40 | -8.43 % (host) | -2.05 % (GPU) |
| 50 | -9.52 % (host) | -1.56 % (GPU) |
| 60 | -8.52 % (host) | -1.36 % (GPU) |
| 70 | -7.96 % (host) | -0.71 % (GPU) |
| 80 | -7.14 % (host) | -0.73 % (GPU) |
| 90 | -6.74 % (host) | -0.99 % (GPU) |
| 100 | -7.59 % (host) | -1.12 % (GPU) |
| 110 | -7.79 % (host) | -0.90 % (GPU) |
| 120 | -7.97 % (host) | -0.77 % (GPU) |
| 130 | -8.35 % (host) | -0.61 % (GPU) |
| 140 | -8.19 % (host) | -0.56 % (GPU) |
| 150 | -8.21 % (host) | -0.47 % (GPU) |

## The clock, power and temperature in the quiet runs' timing (hwmon every 100 ms)

| N | samples | sclk min / median / max (MHz) | below 3.1 GHz | median power (W) | max edge / junction / memory (°C) | sclk after the 2 s warm-up (MHz) |
|--:|--:|---|--:|--:|---|---|
| 1 | 36 | 3468 / 3477/3476/3477 / 3478 | 0 | 38/38/38 | 50 / 53 / 46 | 3471/3476/3477 |
| 10 | 39 | 3384 / 3466/3473/3476 / 3480 | 0 | 46/40/39 | 51 / 56 / 48 | 3471/3477/3477 |
| 20 | 39 | 3151 / 3472/3471/3472 / 3477 | 0 | 48/48/48 | 51 / 58 / 50 | 3476/3476/3477 |
| 30 | 39 | 3094 / 3473/3473/3466 / 3477 | 1 | 51/49/50 | 52 / 58 / 50 | 3476/3477/3472 |
| 40 | 39 | 3092 / 3468/3464/3471 / 3476 | 1 | 54/53/52 | 52 / 61 / 50 | 3481/3477/3477 |
| 50 | 40 | 3029 / 3468/3464/3468 / 3475 | 2 | 56/52/56 | 52 / 56 / 50 | 3476/3477/3477 |
| 60 | 42 | 3146 / 3462/3464/3462 / 3470 | 0 | 54/54/52 | 52 / 62 / 50 | 3470/3476/3472 |
| 70 | 42 | 3390 / 3444/3457/3445 / 3474 | 0 | 58/58/61 | 52 / 59 / 52 | 3451/3477/3472 |
| 80 | 44 | 3011 / 3440/3461/3459 / 3474 | 1 | 58/56/61 | 52 / 59 / 52 | 3476/3477/3477 |
| 90 | 45 | 3102 / 3429/3452/3428 / 3469 | 0 | 62/55/62 | 52 / 64 / 52 | 3476/3471/3472 |
| 100 | 45 | 3205 / 3447/3452/3450 / 3469 | 0 | 58/60/62 | 52 / 58 / 54 | 3471/3476/3471 |
| 110 | 45 | 3026 / 3424/3435/3448 / 3472 | 3 | 60/64/68 | 52 / 65 / 52 | 3476/3476/3476 |
| 120 | 48 | 3031 / 3440/3432/3438 / 3473 | 1 | 63/54/66 | 52 / 63 / 52 | 3477/3471/3472 |
| 130 | 48 | 2974 / 3437/3420/3420 / 3466 | 1 | 68/68/68 | 52 / 60 / 52 | 3476/3478/3476 |
| 140 | 50 | 3176 / 3423/3429/3432 / 3471 | 0 | 64/65/64 | 52 / 60 / 52 | 3477/3478/3477 |
| 150 | 49 | 2993 / 3337/3423/3427 / 3470 | 1 | 67/69/64 | 52 / 64 / 52 | 3282/3476/3476 |

## sclk at the interval boundaries (before-after, min-max over the three quiet runs, MHz)

| N | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |
|--:|---|---|---|---|
| 1 | 3475-3477 / 3397-3434 | 3397-3434 / 3476-3478 | 3476-3478 / 3474-3476 | 3474-3476 / 3477-3477 |
| 10 | 3466-3477 / 3318-3365 | 3318-3365 / 3472-3478 | 3472-3478 / 3420-3426 | 3420-3426 / 3475-3476 |
| 20 | 3462-3464 / 3057-3269 | 3057-3269 / 3473-3476 | 3473-3476 / 3414-3419 | 3414-3419 / 3475-3476 |
| 30 | 3457-3461 / 3149-3175 | 3149-3175 / 3473-3476 | 3473-3476 / 3381-3404 | 3381-3404 / 3476-3477 |
| 40 | 3442-3451 / 2874-3131 | 2874-3131 / 3468-3482 | 3468-3482 / 3381-3417 | 3381-3417 / 3470-3476 |
| 50 | 3442-3445 / 3025-3083 | 3025-3083 / 3459-3468 | 3459-3468 / 3414-3416 | 3414-3416 / 3470-3475 |
| 60 | 3419-3440 / 2961-3064 | 2961-3064 / 3466-3468 | 3466-3468 / 3408-3417 | 3408-3417 / 3466-3470 |
| 70 | 3430-3472 / 2923-3046 | 2923-3046 / 3460-3463 | 3460-3463 / 3402-3427 | 3402-3427 / 3469-3474 |
| 80 | 3430-3471 / 3021-3051 | 3021-3051 / 3456-3468 | 3456-3468 / 3420-3428 | 3420-3428 / 3470-3482 |
| 90 | 3418-3423 / 3012-3029 | 3012-3029 / 3459-3471 | 3459-3471 / 3421-3434 | 3421-3434 / 3465-3469 |
| 100 | 3414-3419 / 3027-3036 | 3027-3036 / 3438-3470 | 3438-3470 / 3418-3425 | 3418-3425 / 3463-3469 |
| 110 | 3407-3415 / 2965-3024 | 2965-3024 / 3446-3454 | 3446-3454 / 3416-3434 | 3416-3434 / 3462-3472 |
| 120 | 3404-3414 / 2999-3033 | 2999-3033 / 3446-3446 | 3446-3446 / 3422-3427 | 3422-3427 / 3466-3472 |
| 130 | 3390-3460 / 2994-2997 | 2994-2997 / 3434-3449 | 3434-3449 / 3386-3434 | 3386-3434 / 3463-3466 |
| 140 | 3380-3389 / 3005-3035 | 3005-3035 / 3423-3434 | 3423-3434 / 3411-3426 | 3411-3426 / 3461-3474 |
| 150 | 3294-3386 / 2980-3012 | 2980-3012 / 3362-3434 | 3362-3434 / 3352-3428 | 3352-3428 / 3444-3470 |

## The N = 30 repeat against results/rx9060/e19-batched-box/

| row, interval | old (ms) | new (ms) | change | spread old / new | exceeds the larger spread? | > 3 %? |
|---|--:|--:|--:|--:|---|---|
| fkl_batch whole_call | 0.1136 | 0.1095 | -3.6 % | 0.8 / 2.7 % | yes | FLAG |
| fkl_batch kernel_only | 0.1029 | 0.1019 | -0.9 % | 0.7 / 0.5 % | yes | no |
| rpp whole_call | 0.1887 | 0.1864 | -1.3 % | 0.4 / 2.7 % | no | no |
| rpp kernel_only | 0.2010 | 0.1858 | -7.6 % | 0.2 / 5.4 % | yes | FLAG |

## The unused runs (warm-up, rocprofv3) against the quiet medians: 57 of 128 values above 3 %

- warmup: change -5.5 to +11.9 %; above 3 %: 3 of 64
- rocprofv3: change +0.2 to +66.0 %; above 3 %: 54 of 64
  - FLAG warmup N = 1 rpp whole_call: 0.028036 ms against 0.025052 (+11.91 %)
  - FLAG warmup N = 10 fkl_batch whole_call: 0.049663 ms against 0.052553 (-5.50 %)
  - FLAG warmup N = 10 rpp whole_call: 0.084083 ms against 0.081605 (+3.04 %)
  - FLAG rocprofv3 N = 1 fkl_batch whole_call: 0.020289 ms against 0.013767 (+47.37 %)
  - FLAG rocprofv3 N = 1 fkl_batch kernel_only: 0.024311 ms against 0.016298 (+49.16 %)
  - FLAG rocprofv3 N = 1 rpp whole_call: 0.041590 ms against 0.025052 (+66.02 %)
  - FLAG rocprofv3 N = 1 rpp kernel_only: 0.045476 ms against 0.027890 (+63.05 %)
  - FLAG rocprofv3 N = 10 fkl_batch whole_call: 0.057851 ms against 0.052553 (+10.08 %)
  - FLAG rocprofv3 N = 10 fkl_batch kernel_only: 0.059947 ms against 0.052127 (+15.00 %)
  - FLAG rocprofv3 N = 10 rpp whole_call: 0.121919 ms against 0.081605 (+49.40 %)
  - FLAG rocprofv3 N = 10 rpp kernel_only: 0.115379 ms against 0.077404 (+49.06 %)
  - FLAG rocprofv3 N = 20 fkl_batch whole_call: 0.087478 ms against 0.078591 (+11.31 %)
  - FLAG rocprofv3 N = 20 fkl_batch kernel_only: 0.078314 ms against 0.071463 (+9.59 %)
  - FLAG rocprofv3 N = 20 rpp whole_call: 0.197411 ms against 0.132406 (+49.10 %)
  - FLAG rocprofv3 N = 20 rpp kernel_only: 0.196491 ms against 0.134859 (+45.70 %)
  - FLAG rocprofv3 N = 30 fkl_batch whole_call: 0.115548 ms against 0.109497 (+5.53 %)
  - FLAG rocprofv3 N = 30 fkl_batch kernel_only: 0.107319 ms against 0.101943 (+5.27 %)
  - FLAG rocprofv3 N = 30 rpp whole_call: 0.256021 ms against 0.186373 (+37.37 %)
  - FLAG rocprofv3 N = 30 rpp kernel_only: 0.258645 ms against 0.185836 (+39.18 %)
  - FLAG rocprofv3 N = 40 fkl_batch whole_call: 0.150569 ms against 0.143869 (+4.66 %)
  - FLAG rocprofv3 N = 40 fkl_batch kernel_only: 0.137460 ms against 0.131739 (+4.34 %)
  - FLAG rocprofv3 N = 40 rpp whole_call: 0.354977 ms against 0.263078 (+34.93 %)
  - FLAG rocprofv3 N = 40 rpp kernel_only: 0.351188 ms against 0.257695 (+36.28 %)
  - FLAG rocprofv3 N = 50 fkl_batch kernel_only: 0.169673 ms against 0.161492 (+5.07 %)
  - FLAG rocprofv3 N = 50 rpp whole_call: 0.503393 ms against 0.355753 (+41.50 %)
  - FLAG rocprofv3 N = 50 rpp kernel_only: 0.494452 ms against 0.350199 (+41.19 %)
  - FLAG rocprofv3 N = 60 fkl_batch whole_call: 0.220120 ms against 0.211770 (+3.94 %)
  - FLAG rocprofv3 N = 60 fkl_batch kernel_only: 0.204558 ms against 0.193737 (+5.59 %)
  - FLAG rocprofv3 N = 60 rpp whole_call: 0.597569 ms against 0.426950 (+39.96 %)
  - FLAG rocprofv3 N = 60 rpp kernel_only: 0.590707 ms against 0.421144 (+40.26 %)
  - FLAG rocprofv3 N = 70 fkl_batch kernel_only: 0.242184 ms against 0.233290 (+3.81 %)
  - FLAG rocprofv3 N = 70 rpp whole_call: 0.693224 ms against 0.498178 (+39.15 %)
  - FLAG rocprofv3 N = 70 rpp kernel_only: 0.686841 ms against 0.494650 (+38.85 %)
  - FLAG rocprofv3 N = 80 rpp whole_call: 0.756451 ms against 0.577395 (+31.01 %)
  - FLAG rocprofv3 N = 80 rpp kernel_only: 0.759837 ms against 0.573207 (+32.56 %)
  - FLAG rocprofv3 N = 90 fkl_batch whole_call: 0.334384 ms against 0.323292 (+3.43 %)
  - FLAG rocprofv3 N = 90 fkl_batch kernel_only: 0.312626 ms against 0.301497 (+3.69 %)
  - FLAG rocprofv3 N = 90 rpp whole_call: 0.928876 ms against 0.673902 (+37.84 %)
  - FLAG rocprofv3 N = 90 rpp kernel_only: 0.914124 ms against 0.667210 (+37.01 %)
  - FLAG rocprofv3 N = 100 fkl_batch kernel_only: 0.343686 ms against 0.332227 (+3.45 %)
  - FLAG rocprofv3 N = 100 rpp whole_call: 1.034220 ms against 0.759554 (+36.16 %)
  - FLAG rocprofv3 N = 100 rpp kernel_only: 1.022623 ms against 0.751035 (+36.16 %)
  - FLAG rocprofv3 N = 110 fkl_batch kernel_only: 0.375769 ms against 0.362588 (+3.64 %)
  - FLAG rocprofv3 N = 110 rpp whole_call: 1.147916 ms against 0.846591 (+35.59 %)
  - FLAG rocprofv3 N = 110 rpp kernel_only: 1.135585 ms against 0.839012 (+35.35 %)
  - FLAG rocprofv3 N = 120 fkl_batch kernel_only: 0.406555 ms against 0.393582 (+3.30 %)
  - FLAG rocprofv3 N = 120 rpp whole_call: 1.261598 ms against 0.935942 (+34.79 %)
  - FLAG rocprofv3 N = 120 rpp kernel_only: 1.250435 ms against 0.928719 (+34.64 %)
  - FLAG rocprofv3 N = 130 fkl_batch kernel_only: 0.436679 ms against 0.422160 (+3.44 %)
  - FLAG rocprofv3 N = 130 rpp whole_call: 1.382975 ms against 1.011893 (+36.67 %)
  - FLAG rocprofv3 N = 130 rpp kernel_only: 1.322806 ms against 1.005751 (+31.52 %)
  - FLAG rocprofv3 N = 140 fkl_batch kernel_only: 0.467952 ms against 0.450138 (+3.96 %)
  - FLAG rocprofv3 N = 140 rpp whole_call: 1.506904 ms against 1.087325 (+38.59 %)
  - FLAG rocprofv3 N = 140 rpp kernel_only: 1.454340 ms against 1.081204 (+34.51 %)
  - FLAG rocprofv3 N = 150 fkl_batch kernel_only: 0.498700 ms against 0.481030 (+3.67 %)
  - FLAG rocprofv3 N = 150 rpp whole_call: 1.615881 ms against 1.159284 (+39.39 %)
  - FLAG rocprofv3 N = 150 rpp kernel_only: 1.527302 ms against 1.153861 (+32.36 %)
- warmup: samples below 3.1 GHz in the timing: 3 (N = 140: 2, N = 150: 1)
- rocprofv3: samples below 3.1 GHz in the timing: 10 (N = 60: 1, N = 70: 1, N = 90: 1, N = 100: 1, N = 110: 1, N = 120: 2, N = 140: 1, N = 150: 2)

## The idle waits: 36 of 80 runs started after a wait

- warmup/n040: 15 s
- warmup/n060: 15 s
- warmup/n070: 15 s
- warmup/n090: 15 s
- warmup/n120: 30 s
- warmup/n130: 15 s
- warmup/n150: 15 s
- quiet-run1/n020: 15 s
- quiet-run1/n050: 15 s
- quiet-run1/n060: 30 s
- quiet-run1/n070: 15 s
- quiet-run1/n120: 45 s
- quiet-run1/n150: 30 s
- quiet-run2/n010: 15 s
- quiet-run2/n040: 30 s
- quiet-run2/n060: 30 s
- quiet-run2/n080: 15 s
- quiet-run2/n100: 15 s
- quiet-run2/n110: 15 s
- quiet-run2/n130: 15 s
- quiet-run3/n001: 15 s
- quiet-run3/n020: 30 s
- quiet-run3/n050: 15 s
- quiet-run3/n080: 15 s
- quiet-run3/n090: 30 s
- quiet-run3/n120: 15 s
- quiet-run3/n140: 15 s
- rocprofv3/n001: 15 s
- rocprofv3/n020: 15 s
- rocprofv3/n040: 30 s
- rocprofv3/n080: 15 s
- rocprofv3/n090: 15 s
- rocprofv3/n100: 30 s
- rocprofv3/n110: 15 s
- rocprofv3/n130: 15 s
- rocprofv3/n150: 15 s
