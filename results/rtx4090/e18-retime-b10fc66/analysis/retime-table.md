| program | version | interval | old (us) | A (us) | B (us) | B / A | B / old | A / old |
|---|---|---|--:|--:|--:|--:|--:|--:|
| `e18_box` | `fkl_box_asym` | whole_call | 42.01 (41.97-42.09) | 42.35 (41.97-42.95) | 42.39 (42.14-42.53) | +0.10 % | +0.92 % **exceeds** (apart) | +0.82 % |
| `e18_box` | `fkl_box_asym` | kernel_only | 41.58 (41.54-42.06) | 42.21 (41.52-42.40) | 42.08 (41.93-42.45) | -0.33 % | +1.19 % | +1.53 % |
| `e18_box` | `fkl_box_halfpixel` | whole_call | 40.60 (40.60-40.63) | 40.57 (40.48-40.72) | 40.97 (40.96-41.10) | +0.99 % **exceeds** (apart) | +0.90 % **exceeds** (apart) | -0.09 % |
| `e18_box` | `fkl_box_halfpixel` | kernel_only | 41.99 (41.99-42.00) | 41.99 (41.98-41.99) | 42.16 (42.14-42.23) | +0.42 % **exceeds** (apart) | +0.41 % **exceeds** (apart) | -0.01 % (apart) |
| `e18_box` | `fkl_trunc_asym` | whole_call | 39.29 (39.27-39.38) | 39.29 (39.11-39.30) | 39.49 (39.48-39.49) | +0.50 % **exceeds** (apart) | +0.49 % **exceeds** (apart) | -0.01 % |
| `e18_box` | `fkl_trunc_asym` | kernel_only | 40.31 (40.21-40.33) | 40.19 (40.11-40.25) | 40.90 (40.90-40.92) | +1.78 % **exceeds** (apart) | +1.47 % **exceeds** (apart) | -0.30 % |
| `e18_box` | `fkl_trunc_halfpixel` | whole_call | 39.59 (39.53-39.62) | 39.52 (39.34-39.53) | 39.85 (39.83-39.90) | +0.82 % **exceeds** (apart) | +0.63 % **exceeds** (apart) | -0.18 % (apart) |
| `e18_box` | `fkl_trunc_halfpixel` | kernel_only | 40.95 (40.93-40.96) | 40.96 (40.86-40.96) | 40.96 (40.96-40.98) | +0.00 % (apart) | +0.03 % (apart) | +0.03 % |
| `e18_box` | `opencv_cuda` | whole_call | 48.45 (48.44-48.52) | 48.30 (48.17-48.39) | 48.42 (48.32-48.53) | +0.26 % | -0.06 % | -0.32 % (apart) |
| `e18_box` | `opencv_cuda` | kernel_only | 49.17 (49.16-49.18) | 49.15 (49.15-49.15) | 49.15 (49.14-49.15) | +0.00 % | -0.03 % (apart) | -0.03 % (apart) |
| `e18_box` | `npp` | whole_call | 33.57 (33.53-33.57) | 33.46 (33.42-33.51) | 33.48 (33.24-33.49) | +0.06 % | -0.25 % (apart) | -0.31 % **exceeds** (apart) |
| `e18_box` | `npp` | kernel_only | 34.08 (34.03-34.08) | 33.97 (33.89-34.00) | 33.93 (33.80-33.99) | -0.12 % | -0.43 % (apart) | -0.31 % (apart) |
| `e18_box` | `fkl_asym_once` | whole_call | 40.27 (40.24-40.29) | 40.17 (40.04-40.23) | 40.50 (40.28-40.59) | +0.84 % **exceeds** (apart) | +0.58 % | -0.26 % (apart) |
| `e18_box` | `fkl_asym_once` | kernel_only | 41.57 (41.51-41.58) | 41.47 (41.24-41.47) | 41.98 (41.47-41.98) | +1.25 % **exceeds** (apart) | +1.00 % | -0.25 % (apart) |
| `e18_box` | `fkl_half_once` | whole_call | 40.94 (40.93-40.95) | 40.90 (40.66-40.93) | 41.15 (40.99-41.18) | +0.59 % (apart) | +0.50 % **exceeds** (apart) | -0.09 % |
| `e18_box` | `fkl_half_once` | kernel_only | 42.01 (42.00-42.04) | 42.00 (41.99-42.01) | 42.28 (42.12-42.30) | +0.67 % **exceeds** (apart) | +0.64 % **exceeds** (apart) | -0.03 % |
| `e18_box` | `fkl_trunc_asym_once` | whole_call | 39.31 (39.30-39.33) | 39.23 (39.18-39.36) | 39.53 (39.42-39.57) | +0.77 % **exceeds** (apart) | +0.56 % **exceeds** (apart) | -0.21 % |
| `e18_box` | `fkl_trunc_asym_once` | kernel_only | 40.28 (40.27-40.32) | 40.24 (40.19-40.24) | 40.91 (40.90-40.92) | +1.67 % **exceeds** (apart) | +1.56 % **exceeds** (apart) | -0.10 % (apart) |
| `e18_box` | `fkl_trunc_halfpixel_once` | whole_call | 39.54 (39.51-39.56) | 39.48 (39.46-39.48) | 39.76 (39.73-39.77) | +0.71 % **exceeds** (apart) | +0.55 % **exceeds** (apart) | -0.16 % **exceeds** (apart) |
| `e18_box` | `fkl_trunc_halfpixel_once` | kernel_only | 40.95 (40.95-40.96) | 40.95 (40.95-40.96) | 40.96 (40.96-40.96) | +0.02 % (apart) | +0.03 % **exceeds** (apart) | +0.01 % |
| `e18_box_split` | `fkl_box_asym` | whole_call | 40.03 (39.95-40.05) | 43.29 (41.97-43.49) | 43.41 (42.33-43.68) | +0.28 % | +8.44 % **exceeds** (apart) **FLAG** | +8.13 % **exceeds** (apart) **FLAG** |
| `e18_box_split` | `fkl_box_asym` | kernel_only | 41.02 (40.99-41.03) | 41.89 (41.61-42.29) | 42.44 (42.37-42.58) | +1.30 % (apart) | +3.46 % **exceeds** (apart) **FLAG** | +2.13 % **exceeds** (apart) |
| `e18_box_split` | `fkl_trunc_halfpixel` | whole_call | 39.19 (39.16-39.23) | 39.29 (39.27-39.31) | 39.53 (39.50-39.57) | +0.61 % **exceeds** (apart) | +0.85 % **exceeds** (apart) | +0.24 % **exceeds** (apart) |
| `e18_box_split` | `fkl_trunc_halfpixel` | kernel_only | 40.13 (40.13-40.19) | 40.22 (40.18-40.27) | 40.84 (40.84-40.88) | +1.55 % **exceeds** (apart) | +1.77 % **exceeds** (apart) | +0.21 % **exceeds** |
| `e18_box_split` | `npp` | whole_call | 33.19 (33.17-33.20) | 33.23 (33.22-33.28) | 33.27 (33.25-33.29) | +0.13 % | +0.25 % **exceeds** (apart) | +0.12 % (apart) |
| `e18_box_split` | `npp` | kernel_only | 33.93 (33.86-33.97) | 33.96 (33.83-33.99) | 33.93 (33.92-33.97) | -0.11 % | -0.01 % | +0.10 % |
| `e18_box_split` | `npp_resize` | whole_call | 12.21 (12.21-12.28) | 12.24 (12.23-12.25) | 12.32 (12.20-12.37) | +0.69 % | +0.90 % | +0.21 % |
| `e18_box_split` | `npp_resize` | kernel_only | 12.29 (12.29-12.30) | 12.29 (12.29-12.29) | 12.29 (12.29-12.29) | -0.00 % | -0.00 % | +0.00 % |
| `e18_box_split` | `fkl_resize` | whole_call | 14.56 (14.53-14.64) | 14.55 (14.51-14.57) | 14.53 (14.49-14.57) | -0.15 % | -0.22 % | -0.07 % |
| `e18_box_split` | `fkl_resize` | kernel_only | 14.41 (14.39-14.49) | 14.59 (14.55-14.60) | 14.59 (14.53-14.66) | +0.01 % | +1.25 % **exceeds** (apart) | +1.25 % **exceeds** (apart) |
| `e18_box_split` | `npp_box` | whole_call | 23.00 (22.97-23.01) | 23.02 (22.96-23.05) | 22.99 (22.90-23.03) | -0.14 % | -0.02 % | +0.12 % |
| `e18_box_split` | `npp_box` | kernel_only | 23.55 (23.55-23.55) | 23.55 (23.55-23.55) | 23.55 (23.55-23.55) | +0.00 % | +0.00 % | +0.00 % |
| `e18_box_split` | `fkl_box` | whole_call | 31.14 (31.06-31.14) | 31.18 (31.12-31.18) | 31.17 (31.16-31.19) | -0.03 % | +0.09 % (apart) | +0.13 % |
| `e18_box_split` | `fkl_box` | kernel_only | 31.76 (31.75-31.77) | 31.75 (31.74-31.76) | 31.75 (31.75-31.77) | +0.01 % | -0.01 % | -0.02 % |
| `e18_stencil` | `fkl_stencil_asym` | whole_call | 38.45 (38.25-38.56) | 40.14 (38.49-40.18) | 41.05 (40.82-41.73) | +2.25 % (apart) | +6.75 % **exceeds** (apart) **FLAG** | +4.40 % **exceeds** **FLAG** |
| `e18_stencil` | `fkl_stencil_asym` | kernel_only | 39.26 (39.16-39.26) | 40.58 (39.45-40.77) | 40.51 (40.39-41.03) | -0.16 % | +3.19 % **exceeds** (apart) **FLAG** | +3.36 % **exceeds** (apart) **FLAG** |
| `e18_stencil` | `fkl_stencil_halfpixel` | whole_call | 38.52 (38.39-38.58) | 38.57 (38.55-38.57) | 38.77 (38.72-38.85) | +0.52 % **exceeds** (apart) | +0.64 % **exceeds** (apart) | +0.12 % |
| `e18_stencil` | `fkl_stencil_halfpixel` | kernel_only | 39.79 (39.52-39.79) | 39.84 (39.81-39.86) | 39.94 (39.94-39.94) | +0.23 % **exceeds** (apart) | +0.38 % (apart) | +0.15 % (apart) |
| `e18_stencil` | `opencv_cuda` | whole_call | 173.72 (173.69-173.86) | 173.87 (173.77-173.93) | 173.88 (173.64-174.71) | +0.00 % | +0.09 % | +0.09 % |
| `e18_stencil` | `opencv_cuda` | kernel_only | 168.75 (168.56-169.10) | 168.73 (168.64-169.01) | 169.23 (168.69-169.27) | +0.30 % | +0.28 % | -0.01 % |
| `e18_stencil` | `npp` | whole_call | 121.31 (120.58-121.47) | 120.68 (120.57-120.87) | 121.24 (120.71-121.45) | +0.46 % | -0.06 % | -0.53 % |
| `e18_stencil` | `npp` | kernel_only | 119.87 (119.87-121.23) | 119.26 (119.25-120.52) | 119.63 (119.30-120.67) | +0.32 % | -0.20 % | -0.51 % |
| `e18_stencil` | `fkl_stencil_asym_once` | whole_call | 38.66 (38.65-38.67) | 38.66 (38.61-38.71) | 38.90 (38.90-38.98) | +0.63 % **exceeds** (apart) | +0.61 % **exceeds** (apart) | -0.02 % |
| `e18_stencil` | `fkl_stencil_asym_once` | kernel_only | 39.94 (39.94-39.94) | 39.94 (39.94-39.94) | 39.94 (39.94-39.94) | +0.00 % (apart) | +0.00 % (apart) | +0.00 % |
| `e18_stencil` | `fkl_stencil_halfpixel_once` | whole_call | 38.80 (38.73-38.84) | 38.82 (38.78-38.83) | 39.02 (39.01-39.08) | +0.52 % **exceeds** (apart) | +0.57 % **exceeds** (apart) | +0.05 % |
| `e18_stencil` | `fkl_stencil_halfpixel_once` | kernel_only | 39.94 (39.94-39.94) | 39.94 (39.94-39.94) | 39.97 (39.96-39.99) | +0.08 % **exceeds** (apart) | +0.08 % **exceeds** (apart) | +0.00 % |
