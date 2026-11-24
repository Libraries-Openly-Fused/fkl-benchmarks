# Task B on the RX 9060: before (A, FKL 24478f0) and after (B, the task B head)

## The FKL filter tests

| built at | test | exit | PASS lines | FAIL lines |
|---|---|--:|--:|--:|
| the PR head | test_box_filter_quad_dpp_hip | 0 | 12 | 0 |
| the PR head | test_convolution_fast_dpp_hip | 0 | 10 | 0 |
| the PR head | test_linear_filter_batch_hip | 0 | 24 | 0 |
| the PR head | test_linear_filter_dpp_hip | 0 | 1 | 0 |
| the PR head | test_median_fast_dpp_hip | 0 | 11 | 0 |
| the PR head | test_median_filter_batch_hip | 0 | 22 | 0 |
| the PR head | test_median_filter_dpp_hip | 0 | 1 | 0 |
| the PR head | test_morphology_fast_dpp_hip | 0 | 11 | 0 |
| the PR head | test_quad_dpps_batch_hip | 0 | 55 | 0 |
| 24478f0 | test_box_filter_quad_dpp_hip | 0 | 12 | 0 |
| 24478f0 | test_convolution_fast_dpp_hip | 0 | 10 | 0 |
| 24478f0 | test_linear_filter_dpp_hip | 0 | 1 | 0 |
| 24478f0 | test_median_fast_dpp_hip | 0 | 11 | 0 |
| 24478f0 | test_median_filter_dpp_hip | 0 | 1 | 0 |
| 24478f0 | test_morphology_fast_dpp_hip | 0 | 11 | 0 |

## fkl-benchmarks E18 and E19: the outputs of the checks, before and after

| program | output | equal before and after |
|---|---|---|
| e18_box | digest rpp output | yes |
| e18_box | digest rpp_ltrb output | yes |
| e18_box | digest fkl_box_halfpixel output | yes |
| e18_box | digest fkl_half_once output | yes |
| e18_box | digest rpp intermediate | yes |
| e18_box | digest rpp_ltrb intermediate | yes |
| e19_batched_box | dump fkl_batch_tensor.f32 | yes |
| e19_batched_box | dump fkl_loop_tensor.f32 | yes |
| e19_batched_box | dump rpp_box.u8 | yes |
| e19_batched_box | dump rpp_grey.u8 | yes |
| e19_batched_box | dump rpp_tensor.f32 | yes |
| e19_crop_sweep_n030 | dump fkl_batch_tensor.f32 | yes |
| e19_crop_sweep_n030 | dump fkl_loop_tensor.f32 | yes |
| e19_crop_sweep_n030 | dump rpp_box.u8 | yes |
| e19_crop_sweep_n030 | dump rpp_grey.u8 | yes |
| e19_crop_sweep_n030 | dump rpp_tensor.f32 | yes |

## The times: the median of the three quiet runs' means (min-max), ms

| case | interval | A (before) | B (after) |
|---|---|--:|--:|
| median3 | whole_call | 0.0140 (0.0138-0.0144) | 0.0140 (0.0138-0.0141) |
| median3 | kernel_only | 0.0166 (0.0166-0.0167) | 0.0165 (0.0164-0.0165) |
| median5 | whole_call | 0.0390 (0.0388-0.0392) | 0.0382 (0.0378-0.0383) |
| median5 | kernel_only | 0.0419 (0.0419-0.0420) | 0.0418 (0.0418-0.0419) |
| median7x3-fused | whole_call | 0.0307 (0.0307-0.0308) | 0.0301 (0.0300-0.0301) |
| median7x3-fused | kernel_only | 0.0341 (0.0340-0.0341) | 0.0334 (0.0333-0.0336) |
| minimum-selection | whole_call | 0.0119 (0.0117-0.0142) | 0.0118 (0.0111-0.0137) |
| minimum-selection | kernel_only | 0.0147 (0.0146-0.0147) | 0.0142 (0.0138-0.0147) |

## B against A (old / new / change / spread old / new / exceeds the larger spread / > 3 %)

| case, interval | old (ms) | new (ms) | change | spread old / new | exceeds? | > 3 %? |
|---|--:|--:|--:|--:|---|---|
| median3 whole_call | 0.0140 | 0.0140 | -0.3 % | 4.5 / 2.0 % | no | no |
| median3 kernel_only | 0.0166 | 0.0165 | -1.1 % | 0.2 / 0.2 % | yes | no |
| median5 whole_call | 0.0390 | 0.0382 | -2.0 % | 1.0 / 1.4 % | yes | no |
| median5 kernel_only | 0.0419 | 0.0418 | -0.2 % | 0.1 / 0.1 % | yes | no |
| median7x3-fused whole_call | 0.0307 | 0.0301 | -2.1 % | 0.3 / 0.5 % | yes | no |
| median7x3-fused kernel_only | 0.0341 | 0.0334 | -1.8 % | 0.2 / 0.8 % | yes | no |
| minimum-selection whole_call | 0.0119 | 0.0118 | -0.7 % | 21.2 / 22.0 % | no | no |
| minimum-selection kernel_only | 0.0147 | 0.0142 | -3.4 % | 0.5 / 5.9 % | no | FLAG |

## The outputs of the float filter instantiations (filters_outputs.bin)

- unused-A: 4632a677a9dac493... (the same as unused-A)
- unused-B: 4632a677a9dac493... (the same as unused-A)
- quiet-run1-A: 4632a677a9dac493... (the same as unused-A)
- quiet-run1-B: 4632a677a9dac493... (the same as unused-A)
- quiet-run2-A: 4632a677a9dac493... (the same as unused-A)
- quiet-run2-B: 4632a677a9dac493... (the same as unused-A)
- quiet-run3-A: 4632a677a9dac493... (the same as unused-A)
- quiet-run3-B: 4632a677a9dac493... (the same as unused-A)

## The clock in each run's timing (hwmon every 100 ms)

| run | samples | sclk min / median / max (MHz) | below 3.1 GHz | median power (W) | max edge / junction / memory (°C) | sclk after the warm-up |
|---|--:|---|--:|--:|---|--:|
| unused-A | 17 | 3293 / 3391 / 3479 | 0 | 32 | 39 / 42 / 36 | 3284 |
| unused-B | 16 | 3481 / 3481 / 3490 | 0 | 35 | 40 / 43 / 36 | 3487 |
| quiet-run1-A | 17 | 3480 / 3482 / 3483 | 0 | 35 | 40 / 43 / 36 | 3482 |
| quiet-run1-B | 16 | 3481 / 3482 / 3483 | 0 | 36 | 41 / 44 / 36 | 3482 |
| quiet-run2-A | 17 | 3481 / 3481 / 3482 | 0 | 36 | 42 / 44 / 38 | 3480 |
| quiet-run2-B | 17 | 3479 / 3481 / 3482 | 0 | 36 | 42 / 45 / 38 | 3481 |
| quiet-run3-A | 17 | 3481 / 3482 / 3485 | 0 | 36 | 43 / 45 / 38 | 3481 |
| quiet-run3-B | 16 | 3481 / 3482 / 3483 | 0 | 36 | 43 / 46 / 40 | 3481 |

## The unused runs against their program's quiet median

- unused-A median3 whole_call: 0.016971 ms (+20.82 %) FLAG
- unused-A median3 kernel_only: 0.016842 ms (+1.17 %)
- unused-A median5 whole_call: 0.040144 ms (+2.97 %)
- unused-A median5 kernel_only: 0.042999 ms (+2.56 %)
- unused-A median7x3-fused whole_call: 0.031202 ms (+1.55 %)
- unused-A median7x3-fused kernel_only: 0.034524 ms (+1.35 %)
- unused-A minimum-selection whole_call: 0.011501 ms (-3.33 %) FLAG
- unused-A minimum-selection kernel_only: 0.014087 ms (-4.03 %) FLAG
- unused-B median3 whole_call: 0.014017 ms (+0.08 %)
- unused-B median3 kernel_only: 0.016475 ms (+0.08 %)
- unused-B median5 whole_call: 0.038038 ms (-0.39 %)
- unused-B median5 kernel_only: 0.041669 ms (-0.40 %)
- unused-B median7x3-fused whole_call: 0.029884 ms (-0.68 %)
- unused-B median7x3-fused kernel_only: 0.033394 ms (-0.12 %)
- unused-B minimum-selection whole_call: 0.011803 ms (-0.09 %)
- unused-B minimum-selection kernel_only: 0.014760 ms (+4.13 %) FLAG
