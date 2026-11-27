| row | interval | median of the 3 means (us) | min-max of the means (us) | spread | host preparation (us, median) |
|---|---|--:|--:|--:|--:|
| fkl_box_asym | whole_call | 42.258 | 42.236-42.278 | 0.1 % |  |
| fkl_box_asym | kernel_only | 42.254 | 42.134-42.464 | 0.8 % | 0.140 |
| fkl_trunc_halfpixel | whole_call | 39.413 | 39.394-39.455 | 0.2 % |  |
| fkl_trunc_halfpixel | kernel_only | 40.339 | 40.326-40.417 | 0.2 % | 0.152 |
| npp | whole_call | 33.233 | 33.214-33.339 | 0.4 % |  |
| npp | kernel_only | 33.966 | 33.909-33.979 | 0.2 % | 0.042 |
| npp_resize | whole_call | 12.278 | 12.209-12.332 | 1.0 % |  |
| npp_resize | kernel_only | 12.291 | 12.289-12.291 | 0.0 % | 0.041 |
| fkl_resize | whole_call | 14.496 | 14.475-14.581 | 0.7 % |  |
| fkl_resize | kernel_only | 14.598 | 14.554-14.628 | 0.5 % | 0.101 |
| npp_box | whole_call | 23.013 | 22.963-23.067 | 0.4 % |  |
| npp_box | kernel_only | 23.552 | 23.552-23.553 | 0.0 % | 0.038 |
| fkl_box | whole_call | 31.178 | 31.155-31.190 | 0.1 % |  |
| fkl_box | kernel_only | 31.759 | 31.751-31.777 | 0.1 % | 0.057 |

| ratio | interval | median of the 3 per-run ratios | min-max |
|---|---|--:|--:|
| npp_box / fkl_box | whole_call | 0.738 | 0.737-0.740 |
| npp_resize / fkl_resize | whole_call | 0.846 | 0.842-0.848 |
| npp / fkl_trunc_halfpixel | whole_call | 0.844 | 0.843-0.845 |
| (fkl_resize + fkl_box) / fkl_trunc_halfpixel | whole_call | 1.158 | 1.158-1.161 |
| (npp_resize + npp_box) / npp | whole_call | 1.063 | 1.056-1.064 |
| npp_box / fkl_box | kernel_only | 0.742 | 0.741-0.742 |
| npp_resize / fkl_resize | kernel_only | 0.842 | 0.840-0.845 |
| npp / fkl_trunc_halfpixel | kernel_only | 0.841 | 0.840-0.843 |
| (fkl_resize + fkl_box) / fkl_trunc_halfpixel | kernel_only | 1.150 | 1.146-1.150 |
| (npp_resize + npp_box) / npp | kernel_only | 1.055 | 1.055-1.057 |

| row | kernel_only / whole_call (median of the 3 per-run ratios) | min-max |
|---|--:|--:|
| fkl_box_asym | 1.0004 | 0.9971-1.0044 |
| fkl_trunc_halfpixel | 1.0232 | 1.0224-1.0260 |
| npp | 1.0221 | 1.0171-1.0230 |
| npp_resize | 1.0011 | 0.9966-1.0066 |
| fkl_resize | 1.0054 | 1.0012-1.0091 |
| npp_box | 1.0234 | 1.0210-1.0257 |
| fkl_box | 1.0186 | 1.0180-1.0200 |

| row | interval | old (window e18-box) median, min-max (us) | new median, min-max (us) | old / new spread (us) | change | exceeds the spread | above 3 % |
|---|---|--:|--:|--:|--:|---|---|
| fkl_box_asym | whole_call | 42.402, 42.187-42.562 | 42.258, 42.236-42.278 | 0.375 / 0.042 | -0.144 us (-0.3 %) | no | no |
| fkl_box_asym | kernel_only | 42.720, 42.192-42.851 | 42.254, 42.134-42.464 | 0.659 / 0.330 | -0.466 us (-1.1 %) | no | no |
| fkl_trunc_halfpixel | whole_call | 39.585, 39.579-39.621 | 39.413, 39.394-39.455 | 0.042 / 0.061 | -0.172 us (-0.4 %) | yes | no |
| fkl_trunc_halfpixel | kernel_only | 40.960, 40.960-40.960 | 40.339, 40.326-40.417 | 0.000 / 0.092 | -0.621 us (-1.5 %) | yes | no |
| npp | whole_call | 33.570, 33.517-33.604 | 33.233, 33.214-33.339 | 0.086 / 0.124 | -0.337 us (-1.0 %) | yes | no |
| npp | kernel_only | 33.984, 33.961-33.991 | 33.966, 33.909-33.979 | 0.030 / 0.070 | -0.017 us (-0.1 %) | no | no |

Largest spread: 1.0 % (npp_resize, whole_call).
