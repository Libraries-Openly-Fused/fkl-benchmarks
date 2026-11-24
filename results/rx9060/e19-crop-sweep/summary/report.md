# E19 crop sweep, RX 9060: the timing summary (fkl_loop is in the raw CSVs only)

## The times: the median of the three quiet runs' means (min-max), ms

| N | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |
|--:|--:|--:|--:|--:|
| 1 | 0.0137 (0.0136-0.0137) | 0.0163 (0.0162-0.0163) | 0.0231 (0.0231-0.0285) | 0.0278 (0.0278-0.0278) |
| 10 | 0.0521 (0.0514-0.0532) | 0.0533 (0.0523-0.0546) | 0.0838 (0.0800-0.0841) | 0.0783 (0.0777-0.0844) |
| 20 | 0.0808 (0.0795-0.0813) | 0.0715 (0.0715-0.0716) | 0.1323 (0.1322-0.1330) | 0.1351 (0.1350-0.1351) |
| 30 | 0.1116 (0.1104-0.1123) | 0.1018 (0.1017-0.1019) | 0.1849 (0.1799-0.1850) | 0.1858 (0.1810-0.1863) |
| 40 | 0.1448 (0.1429-0.1469) | 0.1314 (0.1307-0.1317) | 0.2639 (0.2581-0.2643) | 0.2580 (0.2522-0.2586) |
| 50 | 0.1786 (0.1783-0.1789) | 0.1615 (0.1615-0.1617) | 0.3565 (0.3552-0.3579) | 0.3531 (0.3498-0.3531) |
| 60 | 0.2123 (0.2118-0.2150) | 0.1952 (0.1951-0.1956) | 0.4264 (0.4246-0.4272) | 0.4209 (0.4208-0.4210) |
| 70 | 0.2541 (0.2523-0.2543) | 0.2330 (0.2324-0.2334) | 0.4994 (0.4977-0.5000) | 0.4952 (0.4944-0.4963) |
| 80 | 0.2924 (0.2920-0.2940) | 0.2697 (0.2684-0.2702) | 0.5770 (0.5663-0.5770) | 0.5731 (0.5683-0.5735) |
| 90 | 0.3256 (0.3242-0.3261) | 0.3019 (0.3010-0.3019) | 0.6731 (0.6721-0.6732) | 0.6674 (0.6666-0.6675) |
| 100 | 0.3615 (0.3593-0.3615) | 0.3319 (0.3303-0.3322) | 0.7587 (0.7466-0.7593) | 0.7502 (0.7464-0.7509) |
| 110 | 0.3916 (0.3904-0.3929) | 0.3628 (0.3625-0.3629) | 0.8462 (0.8460-0.8465) | 0.8384 (0.8381-0.8394) |
| 120 | 0.4288 (0.4275-0.4291) | 0.3936 (0.3931-0.3937) | 0.9357 (0.9350-0.9364) | 0.9283 (0.9282-0.9284) |
| 130 | 0.4597 (0.4578-0.4598) | 0.4221 (0.4217-0.4226) | 1.0123 (1.0114-1.0125) | 1.0055 (1.0055-1.0059) |
| 140 | 0.4909 (0.4888-0.4931) | 0.4506 (0.4504-0.4508) | 1.0879 (1.0871-1.0886) | 1.0816 (1.0807-1.0820) |
| 150 | 0.5211 (0.5209-0.5242) | 0.4798 (0.4789-0.4803) | 1.1583 (1.1581-1.1588) | 1.1530 (1.1525-1.1532) |

The largest spread of the three runs' means: 23.5 % (rpp, whole_call, N = 1).

## rpp / fkl_batch: the median of the three per-run ratios (min-max)

| N | whole_call | kernel_only |
|--:|--:|--:|
| 1 | 1.70 (1.68-2.09) | 1.71 (1.71-1.72) |
| 10 | 1.61 (1.50-1.64) | 1.49 (1.43-1.58) |
| 20 | 1.64 (1.64-1.66) | 1.89 (1.89-1.89) |
| 30 | 1.65 (1.63-1.66) | 1.82 (1.78-1.83) |
| 40 | 1.80 (1.78-1.85) | 1.96 (1.93-1.96) |
| 50 | 2.00 (1.99-2.01) | 2.18 (2.17-2.19) |
| 60 | 2.00 (1.99-2.01) | 2.16 (2.15-2.16) |
| 70 | 1.97 (1.96-1.98) | 2.13 (2.12-2.13) |
| 80 | 1.96 (1.94-1.98) | 2.12 (2.12-2.13) |
| 90 | 2.07 (2.06-2.07) | 2.21 (2.21-2.22) |
| 100 | 2.10 (2.08-2.10) | 2.26 (2.26-2.26) |
| 110 | 2.16 (2.15-2.17) | 2.31 (2.31-2.32) |
| 120 | 2.18 (2.18-2.19) | 2.36 (2.36-2.36) |
| 130 | 2.20 (2.20-2.21) | 2.38 (2.38-2.38) |
| 140 | 2.21 (2.21-2.23) | 2.40 (2.40-2.40) |
| 150 | 2.22 (2.21-2.22) | 2.40 (2.40-2.41) |

## GPU- or host-bound: kernel_only against whole_call of the medians (GPU-bound below 3 %)

| N | fkl_batch | rpp |
|--:|--:|--:|
| 1 | +19.05 % (host) | +20.20 % (host) |
| 10 | +2.27 % (GPU) | -6.54 % (host) |
| 20 | -11.51 % (host) | +2.09 % (GPU) |
| 30 | -8.72 % (host) | +0.44 % (GPU) |
| 40 | -9.23 % (host) | -2.25 % (GPU) |
| 50 | -9.57 % (host) | -0.98 % (GPU) |
| 60 | -8.02 % (host) | -1.30 % (GPU) |
| 70 | -8.30 % (host) | -0.84 % (GPU) |
| 80 | -7.73 % (host) | -0.67 % (GPU) |
| 90 | -7.27 % (host) | -0.86 % (GPU) |
| 100 | -8.19 % (host) | -1.11 % (GPU) |
| 110 | -7.35 % (host) | -0.92 % (GPU) |
| 120 | -8.20 % (host) | -0.79 % (GPU) |
| 130 | -8.18 % (host) | -0.67 % (GPU) |
| 140 | -8.21 % (host) | -0.58 % (GPU) |
| 150 | -7.94 % (host) | -0.46 % (GPU) |

## The clock, power and temperature in the quiet runs' timing (hwmon every 100 ms)

| N | samples | sclk min / median / max (MHz) | below 3.1 GHz | median power (W) | max edge / junction / memory (°C) | sclk after the 2 s warm-up (MHz) |
|--:|--:|---|--:|--:|---|---|
| 1 | 37 | 3436 / 3477/3476/3477 / 3482 | 0 | 38/38/38 | 50 / 53 / 46 | 3477/3477/3482 |
| 10 | 37 | 3382 / 3473/3473/3474 / 3478 | 0 | 42/44/46 | 50 / 54 / 46 | 3477/3478/3477 |
| 20 | 39 | 3093 / 3475/3469/3471 / 3476 | 1 | 48/48/47 | 51 / 56 / 48 | 3478/3472/3482 |
| 30 | 39 | 3012 / 3471/3464/3473 / 3476 | 1 | 49/50/50 | 51 / 57 / 50 | 3477/3472/3477 |
| 40 | 42 | 3130 / 3454/3466/3464 / 3476 | 0 | 52/52/52 | 51 / 58 / 50 | 3476/3477/3477 |
| 50 | 40 | 3219 / 3461/3465/3463 / 3471 | 0 | 54/55/52 | 52 / 57 / 50 | 3477/3476/3477 |
| 60 | 41 | 3045 / 3450/3464/3460 / 3471 | 1 | 55/55/56 | 52 / 59 / 50 | 3471/3471/3472 |
| 70 | 43 | 3131 / 3460/3458/3446 / 3471 | 0 | 54/57/56 | 52 / 63 / 52 | 3476/3468/3476 |
| 80 | 44 | 3054 / 3452/3446/3439 / 3475 | 1 | 59/61/60 | 52 / 61 / 52 | 3476/3477/3472 |
| 90 | 45 | 3011 / 3452/3456/3429 / 3475 | 1 | 55/62/61 | 52 / 63 / 52 | 3471/3476/3472 |
| 100 | 46 | 3293 / 3449/3432/3449 / 3472 | 0 | 54/63/63 | 52 / 58 / 52 | 3472/3471/3472 |
| 110 | 45 | 3126 / 3447/3439/3447 / 3468 | 0 | 64/62/65 | 52 / 58 / 54 | 3474/3470/3472 |
| 120 | 46 | 3010 / 3439/3439/3441 / 3469 | 1 | 54/59/63 | 52 / 63 / 52 | 3476/3476/3475 |
| 130 | 48 | 3007 / 3430/3430/3440 / 3467 | 1 | 54/69/66 | 52 / 64 / 52 | 3482/3475/3481 |
| 140 | 49 | 3109 / 3418/3419/3425 / 3466 | 0 | 66/67/69 | 52 / 62 / 52 | 3477/3476/3478 |
| 150 | 51 | 2997 / 3429/3425/3430 / 3468 | 3 | 70/67/66 | 51 / 64 / 52 | 3478/3481/3479 |

## sclk at the interval boundaries (before-after, min-max over the three quiet runs, MHz)

| N | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |
|--:|---|---|---|---|
| 1 | 3472-3477 / 3446-3450 | 3446-3450 / 3476-3477 | 3476-3477 / 3475-3476 | 3475-3476 / 3476-3483 |
| 10 | 3470-3473 / 3343-3387 | 3343-3387 / 3476-3478 | 3476-3478 / 3411-3438 | 3411-3438 / 3471-3477 |
| 20 | 3463-3478 / 3082-3270 | 3082-3270 / 3471-3476 | 3471-3476 / 3365-3422 | 3365-3422 / 3471-3477 |
| 30 | 3456-3473 / 3050-3209 | 3050-3209 / 3469-3476 | 3469-3476 / 3368-3405 | 3368-3405 / 3471-3476 |
| 40 | 3444-3451 / 3109-3128 | 3109-3128 / 3470-3473 | 3470-3473 / 3387-3401 | 3387-3401 / 3470-3474 |
| 50 | 3437-3443 / 3043-3060 | 3043-3060 / 3464-3470 | 3464-3470 / 3410-3420 | 3410-3420 / 3470-3472 |
| 60 | 3435-3455 / 2933-3071 | 2933-3071 / 3466-3469 | 3466-3469 / 3404-3420 | 3404-3420 / 3469-3471 |
| 70 | 3426-3434 / 2975-3021 | 2975-3021 / 3463-3467 | 3463-3467 / 3371-3411 | 3371-3411 / 3465-3477 |
| 80 | 3423-3450 / 2863-2968 | 2863-2968 / 3455-3462 | 3455-3462 / 3416-3426 | 3416-3426 / 3465-3468 |
| 90 | 3423-3426 / 3015-3034 | 3015-3034 / 3462-3466 | 3462-3466 / 3428-3430 | 3428-3430 / 3468-3475 |
| 100 | 3416-3470 / 3004-3035 | 3004-3035 / 3431-3455 | 3431-3455 / 3419-3422 | 3419-3422 / 3466-3473 |
| 110 | 3405-3417 / 3025-3054 | 3025-3054 / 3435-3454 | 3435-3454 / 3413-3414 | 3413-3414 / 3462-3467 |
| 120 | 3397-3409 / 2906-3012 | 2906-3012 / 3441-3451 | 3441-3451 / 3424-3436 | 3424-3436 / 3462-3479 |
| 130 | 3372-3403 / 2973-3047 | 2973-3047 / 3435-3440 | 3435-3440 / 3420-3431 | 3420-3431 / 3462-3466 |
| 140 | 3392-3460 / 2966-3021 | 2966-3021 / 3421-3429 | 3421-3429 / 3399-3421 | 3399-3421 / 3459-3466 |
| 150 | 3381-3398 / 2996-3040 | 2996-3040 / 3424-3450 | 3424-3450 / 3429-3439 | 3429-3439 / 3465-3469 |

## The N = 30 repeat against results/rx9060/e19-batched-box/

| row, interval | old (ms) | new (ms) | change | spread old / new | exceeds the larger spread? | > 3 %? |
|---|--:|--:|--:|--:|---|---|
| fkl_batch whole_call | 0.1136 | 0.1116 | -1.8 % | 0.8 / 1.7 % | yes | no |
| fkl_batch kernel_only | 0.1029 | 0.1018 | -1.0 % | 0.7 / 0.2 % | yes | no |
| rpp whole_call | 0.1887 | 0.1849 | -2.0 % | 0.4 / 2.7 % | no | no |
| rpp kernel_only | 0.2010 | 0.1858 | -7.6 % | 0.2 / 2.9 % | yes | FLAG |

## The unused runs (warm-up, rocprofv3) against the quiet medians: 50 of 128 values above 3 %

- warmup: change -3.0 to +2.9 %; above 3 %: 1 of 64
- rocprofv3: change +1.1 to +80.8 %; above 3 %: 49 of 64
  - FLAG warmup N = 20 fkl_batch whole_call: 0.078388 ms against 0.080814 (-3.00 %)
  - FLAG rocprofv3 N = 1 fkl_batch whole_call: 0.019238 ms against 0.013656 (+40.87 %)
  - FLAG rocprofv3 N = 1 fkl_batch kernel_only: 0.024192 ms against 0.016258 (+48.80 %)
  - FLAG rocprofv3 N = 1 rpp whole_call: 0.041841 ms against 0.023140 (+80.81 %)
  - FLAG rocprofv3 N = 1 rpp kernel_only: 0.045560 ms against 0.027814 (+63.80 %)
  - FLAG rocprofv3 N = 10 fkl_batch whole_call: 0.058189 ms against 0.052134 (+11.62 %)
  - FLAG rocprofv3 N = 10 fkl_batch kernel_only: 0.059682 ms against 0.053315 (+11.94 %)
  - FLAG rocprofv3 N = 10 rpp whole_call: 0.120982 ms against 0.083803 (+44.37 %)
  - FLAG rocprofv3 N = 10 rpp kernel_only: 0.122518 ms against 0.078318 (+56.44 %)
  - FLAG rocprofv3 N = 20 fkl_batch whole_call: 0.084644 ms against 0.080814 (+4.74 %)
  - FLAG rocprofv3 N = 20 fkl_batch kernel_only: 0.078084 ms against 0.071512 (+9.19 %)
  - FLAG rocprofv3 N = 20 rpp whole_call: 0.196926 ms against 0.132308 (+48.84 %)
  - FLAG rocprofv3 N = 20 rpp kernel_only: 0.198590 ms against 0.135079 (+47.02 %)
  - FLAG rocprofv3 N = 30 fkl_batch whole_call: 0.118054 ms against 0.111556 (+5.82 %)
  - FLAG rocprofv3 N = 30 fkl_batch kernel_only: 0.108398 ms against 0.101830 (+6.45 %)
  - FLAG rocprofv3 N = 30 rpp whole_call: 0.271546 ms against 0.184950 (+46.82 %)
  - FLAG rocprofv3 N = 30 rpp kernel_only: 0.278046 ms against 0.185760 (+49.68 %)
  - FLAG rocprofv3 N = 40 fkl_batch whole_call: 0.151022 ms against 0.144791 (+4.30 %)
  - FLAG rocprofv3 N = 40 fkl_batch kernel_only: 0.138713 ms against 0.131420 (+5.55 %)
  - FLAG rocprofv3 N = 40 rpp whole_call: 0.379955 ms against 0.263922 (+43.96 %)
  - FLAG rocprofv3 N = 40 rpp kernel_only: 0.369703 ms against 0.257974 (+43.31 %)
  - FLAG rocprofv3 N = 50 fkl_batch kernel_only: 0.171414 ms against 0.161536 (+6.12 %)
  - FLAG rocprofv3 N = 50 rpp whole_call: 0.504809 ms against 0.356548 (+41.58 %)
  - FLAG rocprofv3 N = 50 rpp kernel_only: 0.495484 ms against 0.353053 (+40.34 %)
  - FLAG rocprofv3 N = 60 fkl_batch whole_call: 0.220868 ms against 0.212277 (+4.05 %)
  - FLAG rocprofv3 N = 60 fkl_batch kernel_only: 0.203568 ms against 0.195246 (+4.26 %)
  - FLAG rocprofv3 N = 60 rpp whole_call: 0.596867 ms against 0.426429 (+39.97 %)
  - FLAG rocprofv3 N = 60 rpp kernel_only: 0.590324 ms against 0.420900 (+40.25 %)
  - FLAG rocprofv3 N = 70 fkl_batch kernel_only: 0.242047 ms against 0.232997 (+3.88 %)
  - FLAG rocprofv3 N = 70 rpp whole_call: 0.692655 ms against 0.499372 (+38.71 %)
  - FLAG rocprofv3 N = 70 rpp kernel_only: 0.684893 ms against 0.495176 (+38.31 %)
  - FLAG rocprofv3 N = 80 fkl_batch kernel_only: 0.278073 ms against 0.269749 (+3.09 %)
  - FLAG rocprofv3 N = 80 rpp whole_call: 0.801730 ms against 0.576974 (+38.95 %)
  - FLAG rocprofv3 N = 80 rpp kernel_only: 0.790286 ms against 0.573095 (+37.90 %)
  - FLAG rocprofv3 N = 90 rpp whole_call: 0.926614 ms against 0.673140 (+37.66 %)
  - FLAG rocprofv3 N = 90 rpp kernel_only: 0.914334 ms against 0.667377 (+37.00 %)
  - FLAG rocprofv3 N = 100 rpp whole_call: 0.976519 ms against 0.758664 (+28.72 %)
  - FLAG rocprofv3 N = 100 rpp kernel_only: 0.966152 ms against 0.750245 (+28.78 %)
  - FLAG rocprofv3 N = 110 rpp whole_call: 1.144986 ms against 0.846238 (+35.30 %)
  - FLAG rocprofv3 N = 110 rpp kernel_only: 1.088727 ms against 0.838441 (+29.85 %)
  - FLAG rocprofv3 N = 120 rpp whole_call: 1.260949 ms against 0.935655 (+34.77 %)
  - FLAG rocprofv3 N = 120 rpp kernel_only: 1.250172 ms against 0.928261 (+34.68 %)
  - FLAG rocprofv3 N = 130 rpp whole_call: 1.386362 ms against 1.012283 (+36.95 %)
  - FLAG rocprofv3 N = 130 rpp kernel_only: 1.352946 ms against 1.005541 (+34.55 %)
  - FLAG rocprofv3 N = 140 fkl_batch kernel_only: 0.464211 ms against 0.450568 (+3.03 %)
  - FLAG rocprofv3 N = 140 rpp whole_call: 1.503062 ms against 1.087890 (+38.16 %)
  - FLAG rocprofv3 N = 140 rpp kernel_only: 1.452890 ms against 1.081596 (+34.33 %)
  - FLAG rocprofv3 N = 150 fkl_batch kernel_only: 0.495474 ms against 0.479761 (+3.28 %)
  - FLAG rocprofv3 N = 150 rpp whole_call: 1.609588 ms against 1.158280 (+38.96 %)
  - FLAG rocprofv3 N = 150 rpp kernel_only: 1.553635 ms against 1.152966 (+34.75 %)
- warmup: samples below 3.1 GHz in the timing: 2 (N = 120: 1, N = 140: 1)
- rocprofv3: samples below 3.1 GHz in the timing: 9 (N = 90: 2, N = 100: 1, N = 110: 1, N = 120: 1, N = 130: 1, N = 140: 2, N = 150: 1)

## The idle waits: 21 of 80 runs started after a wait

- warmup/n140: 15 s
- quiet-run1/n030: 15 s
- quiet-run1/n050: 15 s
- quiet-run1/n080: 15 s
- quiet-run1/n120: 30 s
- quiet-run1/n150: 30 s
- quiet-run2/n040: 15 s
- quiet-run2/n060: 30 s
- quiet-run2/n090: 15 s
- quiet-run2/n120: 15 s
- quiet-run2/n150: 30 s
- quiet-run3/n001: 45 s
- quiet-run3/n020: 15 s
- quiet-run3/n070: 15 s
- quiet-run3/n130: 15 s
- quiet-run3/n150: 15 s
- rocprofv3/n020: 30 s
- rocprofv3/n070: 15 s
- rocprofv3/n080: 15 s
- rocprofv3/n110: 15 s
- rocprofv3/n150: 15 s
