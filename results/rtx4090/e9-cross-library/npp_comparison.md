# NPP re-runs after FastNPP #45 against the single runs of 2026-10-30

old = ONE run each (2026-10-30, FastNPP 9dfc4f4, the old FKL pin); new = median of 3 runs (FastNPP a1b5e245, FKL c8fcbe8). Times in ms. exceeds: |change| > 3 % (GPU-bound) or 5 % (host-bound).

## e0-npp, with CPU overhead

15 batch sizes in common (old 15, new [15, 15, 15]).

### NPP, GPU-bound, flag 3 %, must NOT change (same NPP calls)

| batch | old | new median | spread % | change % | exceeds | beyond spread |
|---:|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.133734 | 0.133062 | 2.6 | -0.5 | no | no |
| 20 | 0.261618 | 0.261639 | 1.8 | +0.0 | no | no |
| 30 | 0.395033 | 0.389323 | 0.9 | -1.4 | no | yes |
| 40 | 0.526247 | 0.51874 | 1.1 | -1.4 | no | yes |
| 50 | 0.655237 | 0.647122 | 0.4 | -1.2 | no | yes |
| 60 | 0.78353 | 0.776977 | 0.9 | -0.8 | no | no |
| 70 | 0.90694 | 0.910718 | 1.3 | +0.4 | no | no |
| 80 | 1.0523 | 1.0385 | 2.1 | -1.3 | no | no |
| 90 | 1.1812 | 1.16399 | 1.2 | -1.5 | no | yes |
| 100 | 1.31761 | 1.29842 | 1.4 | -1.5 | no | yes |
| 110 | 1.44884 | 1.42535 | 1.6 | -1.6 | no | yes |
| 120 | 1.56328 | 1.55601 | 1.6 | -0.5 | no | no |
| 130 | 1.69673 | 1.67623 | 1.8 | -1.2 | no | no |
| 140 | 1.84038 | 1.81204 | 1.7 | -1.5 | no | no |
| 150 | 1.9521 | 1.94405 | 1.8 | -0.4 | no | no |

e0-npp, with CPU overhead / NPP: 15 points; change min -1.6 %, median -1.2 %, max +0.4 %; exceeds the 3 % flag in 0, beyond the spread of the 3 runs in 6; largest spread of the 3 runs 2.6 %; GPU-bound; must NOT change (same NPP calls)

### FastNPP (the 'FK' column), host-bound, flag 5 %, may change (#45: pixel-centre resize)

| batch | old | new median | spread % | change % | exceeds | beyond spread |
|---:|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.00451232 | 0.00434752 | 2.0 | -3.7 | no | yes |
| 20 | 0.005368 | 0.0053648 | 1.5 | -0.1 | no | no |
| 30 | 0.00629471 | 0.00632288 | 0.2 | +0.4 | no | yes |
| 40 | 0.00733376 | 0.00746048 | 1.6 | +1.7 | no | yes |
| 50 | 0.00831488 | 0.00815136 | 3.1 | -2.0 | no | no |
| 60 | 0.0091424 | 0.00904672 | 1.4 | -1.0 | no | no |
| 70 | 0.0102694 | 0.010151 | 0.2 | -1.2 | no | yes |
| 80 | 0.0113094 | 0.0111875 | 0.6 | -1.1 | no | yes |
| 90 | 0.0119888 | 0.0118691 | 0.1 | -1.0 | no | yes |
| 100 | 0.0129568 | 0.0128659 | 0.7 | -0.7 | no | yes |
| 110 | 0.0139098 | 0.0136819 | 1.2 | -1.6 | no | yes |
| 120 | 0.0146621 | 0.0146848 | 1.1 | +0.2 | no | no |
| 130 | 0.0156787 | 0.015624 | 0.4 | -0.3 | no | no |
| 140 | 0.0168394 | 0.0167466 | 0.7 | -0.6 | no | no |
| 150 | 0.0176576 | 0.017679 | 1.1 | +0.1 | no | no |

e0-npp, with CPU overhead / FastNPP (the 'FK' column): 15 points; change min -3.7 %, median -0.7 %, max +1.7 %; exceeds the 5 % flag in 0, beyond the spread of the 3 runs in 8; largest spread of the 3 runs 3.1 %; host-bound; may change (#45: pixel-centre resize)

## e0-npp, GPU only (No_CPU_Overhead)

15 batch sizes in common (old 15, new [15, 15, 15]).

### NPP, GPU-bound, flag 3 %, must NOT change (same NPP calls)

| batch | old | new median | spread % | change % | exceeds | beyond spread |
|---:|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.133392 | 0.132254 | 1.5 | -0.9 | no | no |
| 20 | 0.261274 | 0.257841 | 1.4 | -1.3 | no | no |
| 30 | 0.395414 | 0.389065 | 1.1 | -1.6 | no | yes |
| 40 | 0.523153 | 0.518972 | 1.8 | -0.8 | no | no |
| 50 | 0.653662 | 0.65036 | 3.5 | -0.5 | no | no |
| 60 | 0.780292 | 0.773897 | 2.5 | -0.8 | no | no |
| 70 | 0.918949 | 0.910042 | 2.2 | -1.0 | no | no |
| 80 | 1.04692 | 1.03748 | 2.0 | -0.9 | no | no |
| 90 | 1.18684 | 1.16202 | 3.5 | -2.1 | no | no |
| 100 | 1.30498 | 1.30203 | 1.1 | -0.2 | no | no |
| 110 | 1.46217 | 1.42379 | 1.8 | -2.6 | no | yes |
| 120 | 1.57483 | 1.55798 | 1.7 | -1.1 | no | no |
| 130 | 1.69362 | 1.68241 | 0.9 | -0.7 | no | no |
| 140 | 1.81358 | 1.81337 | 1.6 | -0.0 | no | no |
| 150 | 1.95414 | 1.9514 | 1.0 | -0.1 | no | no |

e0-npp, GPU only (No_CPU_Overhead) / NPP: 15 points; change min -2.6 %, median -0.9 %, max -0.0 %; exceeds the 3 % flag in 0, beyond the spread of the 3 runs in 2; largest spread of the 3 runs 3.5 %; GPU-bound; must NOT change (same NPP calls)

### FastNPP (the 'FK' column), GPU-bound, flag 3 %, may change (#45: pixel-centre resize)

| batch | old | new median | spread % | change % | exceeds | beyond spread |
|---:|---:|---:|---:|---:|:---:|:---:|
| 10 | 0.00374016 | 0.00373728 | 4.1 | -0.1 | no | no |
| 20 | 0.00446112 | 0.00463104 | 4.9 | +3.8 | YES | no |
| 30 | 0.00522688 | 0.00519968 | 4.0 | -0.5 | no | no |
| 40 | 0.00584352 | 0.00592608 | 4.9 | +1.4 | no | no |
| 50 | 0.00646912 | 0.00650944 | 1.4 | +0.6 | no | no |
| 60 | 0.00718368 | 0.00721408 | 1.9 | +0.4 | no | no |
| 70 | 0.00810944 | 0.0080672 | 4.0 | -0.5 | no | no |
| 80 | 0.00883296 | 0.0087824 | 3.4 | -0.6 | no | no |
| 90 | 0.00922912 | 0.009168 | 1.2 | -0.7 | no | no |
| 100 | 0.00994208 | 0.0099312 | 1.1 | -0.1 | no | no |
| 110 | 0.0109946 | 0.0105709 | 0.6 | -3.9 | YES | yes |
| 120 | 0.0114758 | 0.0112947 | 0.3 | -1.6 | no | yes |
| 130 | 0.0122819 | 0.0119834 | 0.7 | -2.4 | no | yes |
| 140 | 0.0131392 | 0.0128186 | 1.5 | -2.4 | no | yes |
| 150 | 0.0136416 | 0.0136912 | 1.9 | +0.4 | no | no |

e0-npp, GPU only (No_CPU_Overhead) / FastNPP (the 'FK' column): 15 points; change min -3.9 %, median -0.5 %, max +3.8 %; exceeds the 3 % flag in 2, beyond the spread of the 3 runs in 4; largest spread of the 3 runs 4.9 %; GPU-bound; may change (#45: pixel-centre resize)

## e0-cpu-time-npp, CPU time

16 batch sizes in common (old 16, new [16, 16, 16]).

### NPP, host-bound, flag 5 %, must NOT change (same NPP calls)

| batch | old | new median | spread % | change % | exceeds | beyond spread |
|---:|---:|---:|---:|---:|:---:|:---:|
| 2 | 0.0284535 | 0.0275322 | 1.9 | -3.2 | no | yes |
| 12 | 0.159104 | 0.158108 | 2.1 | -0.6 | no | no |
| 22 | 0.28538 | 0.285227 | 3.6 | -0.1 | no | no |
| 32 | 0.41358 | 0.413149 | 1.4 | -0.1 | no | no |
| 42 | 0.545103 | 0.54139 | 0.7 | -0.7 | no | yes |
| 52 | 0.676243 | 0.671087 | 1.0 | -0.8 | no | no |
| 62 | 0.80636 | 0.799904 | 0.6 | -0.8 | no | yes |
| 72 | 0.935662 | 0.932492 | 1.0 | -0.3 | no | no |
| 82 | 1.0562 | 1.0584 | 2.7 | +0.2 | no | no |
| 92 | 1.19093 | 1.18999 | 0.1 | -0.1 | no | no |
| 102 | 1.31355 | 1.31082 | 1.8 | -0.2 | no | no |
| 112 | 1.44746 | 1.43842 | 3.3 | -0.6 | no | no |
| 122 | 1.57135 | 1.56135 | 1.7 | -0.6 | no | no |
| 132 | 1.69484 | 1.69411 | 0.5 | -0.0 | no | no |
| 142 | 1.82313 | 1.82021 | 1.2 | -0.2 | no | no |
| 152 | 1.95796 | 1.95757 | 2.3 | -0.0 | no | no |

e0-cpu-time-npp, CPU time / NPP: 16 points; change min -3.2 %, median -0.3 %, max +0.2 %; exceeds the 5 % flag in 0, beyond the spread of the 3 runs in 3; largest spread of the 3 runs 3.6 %; host-bound; must NOT change (same NPP calls)

### FastNPP (the 'FK' column), host-bound, flag 5 %, may change (#45: pixel-centre resize)

| batch | old | new median | spread % | change % | exceeds | beyond spread |
|---:|---:|---:|---:|---:|:---:|:---:|
| 2 | 0.00200249 | 0.00191038 | 2.0 | -4.6 | no | yes |
| 12 | 0.00273409 | 0.00272855 | 1.5 | -0.2 | no | no |
| 22 | 0.00268615 | 0.00272735 | 2.4 | +1.5 | no | no |
| 32 | 0.00304477 | 0.00299488 | 1.8 | -1.6 | no | no |
| 42 | 0.00359243 | 0.00341575 | 6.1 | -4.9 | no | no |
| 52 | 0.0037116 | 0.00374793 | 6.9 | +1.0 | no | no |
| 62 | 0.00400542 | 0.00393217 | 1.8 | -1.8 | no | yes |
| 72 | 0.00421742 | 0.00427544 | 2.4 | +1.4 | no | no |
| 82 | 0.00454987 | 0.00455103 | 1.2 | +0.0 | no | no |
| 92 | 0.00492055 | 0.00489211 | 1.6 | -0.6 | no | no |
| 102 | 0.00505873 | 0.00510232 | 0.3 | +0.9 | no | yes |
| 112 | 0.00553707 | 0.00548393 | 8.6 | -1.0 | no | no |
| 122 | 0.00581373 | 0.0057219 | 0.9 | -1.6 | no | yes |
| 132 | 0.00632041 | 0.00611493 | 1.0 | -3.3 | no | yes |
| 142 | 0.00642532 | 0.0064063 | 1.4 | -0.3 | no | no |
| 152 | 0.00687894 | 0.00682712 | 2.1 | -0.8 | no | no |

e0-cpu-time-npp, CPU time / FastNPP (the 'FK' column): 16 points; change min -4.9 %, median -0.7 %, max +1.5 %; exceeds the 5 % flag in 0, beyond the spread of the 3 runs in 5; largest spread of the 3 runs 8.6 %; host-bound; may change (#45: pixel-centre resize)

## Agreement (informational maximum difference FastNPP against NPP, and which convention each follows)

- e0-npp, with CPU overhead: FastNPP_follows old ['FKL'] (30 rows) -> new ['pixel-centre'] ([30, 30, 30] rows); NPP_follows old ['pixel-centre'] -> new ['pixel-centre']; FastNPP_vs_NPP_max old 4.13643 to 4.26534 -> new 3.8147e-06 to 3.8147e-06; FastNPP_vs_ref_pixel_centre_max new max 3.52063e-06; NPP_vs_ref_pixel_centre_max new max 3.56033e-06 (tolerance of the check 1e-3)
- e0-cpu-time-npp, CPU time: FastNPP_follows old ['FKL'] (32 rows) -> new ['pixel-centre'] ([32, 32, 32] rows); NPP_follows old ['pixel-centre'] -> new ['pixel-centre']; FastNPP_vs_NPP_max old 3.96621 to 4.26534 -> new 3.8147e-06 to 3.8147e-06; FastNPP_vs_ref_pixel_centre_max new max 3.52063e-06; NPP_vs_ref_pixel_centre_max new max 3.56033e-06 (tolerance of the check 1e-3)
