| row | interval | median of the 3 means (us) | min-max of the means (us) | spread | host preparation (us, median) |
|---|---|--:|--:|--:|--:|
| fkl_box_asym | whole_call | 40.031 | 39.951-40.052 | 0.3 % |  |
| fkl_box_asym | kernel_only | 41.019 | 40.991-41.029 | 0.1 % | 0.143 |
| fkl_trunc_halfpixel | whole_call | 39.195 | 39.159-39.234 | 0.2 % |  |
| fkl_trunc_halfpixel | kernel_only | 40.134 | 40.129-40.191 | 0.2 % | 0.146 |
| npp | whole_call | 33.190 | 33.172-33.196 | 0.1 % |  |
| npp | kernel_only | 33.929 | 33.864-33.970 | 0.3 % | 0.039 |
| npp_resize | whole_call | 12.214 | 12.206-12.278 | 0.6 % |  |
| npp_resize | kernel_only | 12.291 | 12.288-12.300 | 0.1 % | 0.039 |
| fkl_resize | whole_call | 14.561 | 14.529-14.644 | 0.8 % |  |
| fkl_resize | kernel_only | 14.414 | 14.394-14.493 | 0.7 % | 0.105 |
| npp_box | whole_call | 22.996 | 22.966-23.007 | 0.2 % |  |
| npp_box | kernel_only | 23.552 | 23.552-23.552 | 0.0 % | 0.038 |
| fkl_box | whole_call | 31.141 | 31.060-31.142 | 0.3 % |  |
| fkl_box | kernel_only | 31.756 | 31.751-31.772 | 0.1 % | 0.056 |

| ratio | interval | median of the 3 per-run ratios | min-max |
|---|---|--:|--:|
| npp_box / fkl_box | whole_call | 0.739 | 0.738-0.739 |
| npp_resize / fkl_resize | whole_call | 0.839 | 0.834-0.845 |
| npp / fkl_trunc_halfpixel | whole_call | 0.846 | 0.846-0.848 |
| (fkl_resize + fkl_box) / fkl_trunc_halfpixel | whole_call | 1.166 | 1.165-1.166 |
| (npp_resize + npp_box) / npp | whole_call | 1.061 | 1.060-1.063 |
| npp_box / fkl_box | kernel_only | 0.742 | 0.741-0.742 |
| npp_resize / fkl_resize | kernel_only | 0.853 | 0.848-0.854 |
| npp / fkl_trunc_halfpixel | kernel_only | 0.844 | 0.844-0.847 |
| (fkl_resize + fkl_box) / fkl_trunc_halfpixel | kernel_only | 1.150 | 1.149-1.153 |
| (npp_resize + npp_box) / npp | kernel_only | 1.057 | 1.055-1.058 |

| row | kernel_only / whole_call (median of the 3 per-run ratios) | min-max |
|---|--:|--:|
| fkl_box_asym | 1.0244 | 1.0240-1.0267 |
| fkl_trunc_halfpixel | 1.0244 | 1.0240-1.0248 |
| npp | 1.0223 | 1.0209-1.0233 |
| npp_resize | 1.0062 | 1.0008-1.0077 |
| fkl_resize | 0.9907 | 0.9843-0.9953 |
| npp_box | 1.0242 | 1.0237-1.0255 |
| fkl_box | 1.0202 | 1.0196-1.0224 |

| row | interval | old (window e18-box) median, min-max (us) | new median, min-max (us) | old / new spread (us) | change | exceeds the spread | above 3 % |
|---|---|--:|--:|--:|--:|---|---|
| fkl_box_asym | whole_call | 42.006, 41.970-42.089 | 40.031, 39.951-40.052 | 0.119 / 0.101 | -1.975 us (-4.7 %) | yes | yes |
| fkl_box_asym | kernel_only | 41.580, 41.543-42.065 | 41.019, 40.991-41.029 | 0.522 / 0.038 | -0.561 us (-1.3 %) | yes | no |
| fkl_trunc_halfpixel | whole_call | 39.595, 39.535-39.618 | 39.195, 39.159-39.234 | 0.084 / 0.075 | -0.400 us (-1.0 %) | yes | no |
| fkl_trunc_halfpixel | kernel_only | 40.950, 40.929-40.960 | 40.134, 40.129-40.191 | 0.031 / 0.061 | -0.816 us (-2.0 %) | yes | no |
| npp | whole_call | 33.567, 33.535-33.575 | 33.190, 33.172-33.196 | 0.040 / 0.024 | -0.377 us (-1.1 %) | yes | no |
| npp | kernel_only | 34.080, 34.026-34.083 | 33.929, 33.864-33.970 | 0.058 / 0.106 | -0.151 us (-0.4 %) | yes | no |

Largest spread: 0.8 % (fkl_resize, whole_call).
