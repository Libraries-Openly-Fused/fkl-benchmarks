| row | interval | median of the 3 means (us) | min-max of the means (us) | spread | host preparation (us, median) |
|---|---|--:|--:|--:|--:|
| fkl_box_asym | whole_call | 43.892 | 42.272-44.000 | 3.9 % |  |
| fkl_box_asym | kernel_only | 42.032 | 41.951-42.247 | 0.7 % | 0.148 |
| fkl_trunc_halfpixel | whole_call | 39.533 | 39.511-39.578 | 0.2 % |  |
| fkl_trunc_halfpixel | kernel_only | 40.864 | 40.790-40.868 | 0.2 % | 0.146 |
| npp | whole_call | 33.261 | 33.247-33.304 | 0.2 % |  |
| npp | kernel_only | 33.953 | 33.934-33.964 | 0.1 % | 0.040 |
| npp_resize | whole_call | 12.255 | 12.242-12.320 | 0.6 % |  |
| npp_resize | kernel_only | 12.295 | 12.292-12.303 | 0.1 % | 0.040 |
| fkl_resize | whole_call | 14.506 | 14.464-14.544 | 0.6 % |  |
| fkl_resize | kernel_only | 14.563 | 14.543-14.595 | 0.4 % | 0.105 |
| npp_box | whole_call | 22.965 | 22.959-23.079 | 0.5 % |  |
| npp_box | kernel_only | 23.552 | 23.552-23.552 | 0.0 % | 0.038 |
| fkl_box | whole_call | 31.239 | 31.124-31.258 | 0.4 % |  |
| fkl_box | kernel_only | 31.751 | 31.747-31.758 | 0.0 % | 0.056 |

| ratio | interval | median of the 3 per-run ratios | min-max |
|---|---|--:|--:|
| npp_box / fkl_box | whole_call | 0.738 | 0.735-0.738 |
| npp_resize / fkl_resize | whole_call | 0.846 | 0.845-0.847 |
| npp / fkl_trunc_halfpixel | whole_call | 0.841 | 0.840-0.842 |
| (fkl_resize + fkl_box) / fkl_trunc_halfpixel | whole_call | 1.156 | 1.154-1.158 |
| (npp_resize + npp_box) / npp | whole_call | 1.061 | 1.057-1.063 |
| npp_box / fkl_box | kernel_only | 0.742 | 0.742-0.742 |
| npp_resize / fkl_resize | kernel_only | 0.844 | 0.843-0.845 |
| npp / fkl_trunc_halfpixel | kernel_only | 0.831 | 0.831-0.832 |
| (fkl_resize + fkl_box) / fkl_trunc_halfpixel | kernel_only | 1.133 | 1.133-1.136 |
| (npp_resize + npp_box) / npp | kernel_only | 1.056 | 1.055-1.057 |

| row | kernel_only / whole_call (median of the 3 per-run ratios) | min-max |
|---|--:|--:|
| fkl_box_asym | 0.9576 | 0.9534-0.9994 |
| fkl_trunc_halfpixel | 1.0325 | 1.0318-1.0344 |
| npp | 1.0211 | 1.0189-1.0212 |
| npp_resize | 1.0031 | 0.9980-1.0050 |
| fkl_resize | 1.0039 | 0.9999-1.0090 |
| npp_box | 1.0256 | 1.0205-1.0258 |
| fkl_box | 1.0164 | 1.0157-1.0204 |

| row | interval | old (window e18-box) median, min-max (us) | new median, min-max (us) | old / new spread (us) | change | exceeds the spread | above 3 % |
|---|---|--:|--:|--:|--:|---|---|
| fkl_box_asym | whole_call | 42.369, 42.308-43.855 | 43.892, 42.272-44.000 | 1.547 / 1.728 | +1.523 us (+3.6 %) | no | yes |
| fkl_box_asym | kernel_only | 42.589, 41.913-42.797 | 42.032, 41.951-42.247 | 0.884 / 0.296 | -0.557 us (-1.3 %) | no | no |
| fkl_trunc_halfpixel | whole_call | 39.790, 39.763-39.795 | 39.533, 39.511-39.578 | 0.032 / 0.067 | -0.256 us (-0.6 %) | yes | no |
| fkl_trunc_halfpixel | kernel_only | 40.962, 40.960-40.962 | 40.864, 40.790-40.868 | 0.002 / 0.078 | -0.098 us (-0.2 %) | yes | no |
| npp | whole_call | 33.500, 33.490-33.519 | 33.261, 33.247-33.304 | 0.028 / 0.057 | -0.238 us (-0.7 %) | yes | no |
| npp | kernel_only | 34.014, 33.977-34.041 | 33.953, 33.934-33.964 | 0.064 / 0.030 | -0.061 us (-0.2 %) | no | no |

Largest spread: 3.9 % (fkl_box_asym, whole_call).
