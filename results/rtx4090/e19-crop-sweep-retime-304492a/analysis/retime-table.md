| N | row, interval | old, us (spread) | A, us (spread) | B, us (spread) | A vs old | exceeds? | B vs A | exceeds? | bound (B) | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | fkl_batch whole_call | 4.71 (1.61 %) | 4.80 (0.23 %) | 3.93 (0.67 %) | +1.85 % | yes (no overlap) | -18.04 % | yes (no overlap) | host-bound | B vs A |
| 1 | fkl_batch kernel_only | 4.13 (0.94 %) | 4.27 (0.74 %) | 3.46 (1.57 %) | +3.46 % | yes (no overlap) | -18.87 % | yes (no overlap) | host-bound | B vs A |
| 1 | opencv_cuda whole_call | 9.74 (8.23 %) | 9.69 (0.71 %) | 9.83 (1.35 %) | -0.51 % | no | +1.41 % | yes (no overlap) | host-bound | - |
| 1 | opencv_cuda kernel_only | 8.65 (3.47 %) | 9.18 (0.22 %) | 9.14 (0.78 %) | +6.07 % | yes (no overlap) | -0.33 % | no | host-bound | A vs old |
| 1 | npp whole_call | 8.48 (1.09 %) | 8.41 (0.85 %) | 8.43 (0.74 %) | -0.78 % | no (no overlap) | +0.26 % | no | GPU-bound | - |
| 1 | npp kernel_only | 8.21 (0.27 %) | 8.19 (0.21 %) | 8.19 (0.00 %) | -0.26 % | no | +0.00 % | no | GPU-bound | - |
| 10 | fkl_batch whole_call | 9.59 (1.00 %) | 9.60 (0.32 %) | 9.75 (0.50 %) | +0.14 % | no | +1.57 % | yes (no overlap) | host-bound | - |
| 10 | fkl_batch kernel_only | 9.22 (0.09 %) | 9.22 (0.19 %) | 9.22 (0.09 %) | -0.00 % | no | +0.02 % | no | host-bound | - |
| 10 | opencv_cuda whole_call | 80.75 (0.09 %) | 81.37 (0.35 %) | 81.40 (0.23 %) | +0.76 % | yes (no overlap) | +0.03 % | no | GPU-bound | - |
| 10 | opencv_cuda kernel_only | 78.90 (0.01 %) | 80.43 (0.05 %) | 80.41 (0.01 %) | +1.95 % | yes (no overlap) | -0.03 % | no | GPU-bound | - |
| 10 | npp whole_call | 72.49 (0.06 %) | 73.00 (1.77 %) | 72.82 (0.23 %) | +0.70 % | no (no overlap) | -0.25 % | no | GPU-bound | - |
| 10 | npp kernel_only | 72.60 (0.19 %) | 72.70 (0.00 %) | 72.70 (0.00 %) | +0.14 % | no (no overlap) | +0.00 % | no | GPU-bound | - |
| 20 | fkl_batch whole_call | 15.81 (0.68 %) | 15.87 (0.66 %) | 16.07 (0.23 %) | +0.36 % | no | +1.28 % | yes (no overlap) | host-bound | - |
| 20 | fkl_batch kernel_only | 15.38 (0.15 %) | 15.43 (0.16 %) | 15.56 (0.38 %) | +0.29 % | yes (no overlap) | +0.84 % | yes (no overlap) | host-bound | - |
| 20 | opencv_cuda whole_call | 159.35 (0.54 %) | 161.03 (0.21 %) | 160.94 (0.43 %) | +1.05 % | yes (no overlap) | -0.05 % | no | GPU-bound | - |
| 20 | opencv_cuda kernel_only | 156.91 (0.05 %) | 160.28 (0.02 %) | 160.29 (0.01 %) | +2.15 % | yes (no overlap) | +0.00 % | no | GPU-bound | - |
| 20 | npp whole_call | 143.46 (0.16 %) | 144.50 (0.13 %) | 144.44 (0.10 %) | +0.72 % | yes (no overlap) | -0.04 % | no | GPU-bound | - |
| 20 | npp kernel_only | 143.54 (0.02 %) | 144.36 (0.02 %) | 144.38 (0.01 %) | +0.57 % | yes (no overlap) | +0.01 % | no | GPU-bound | - |
| 30 | fkl_batch whole_call | 22.04 (0.55 %) | 22.18 (0.45 %) | 22.33 (0.45 %) | +0.62 % | yes (no overlap) | +0.67 % | yes (no overlap) | GPU-bound | - |
| 30 | fkl_batch kernel_only | 21.70 (0.13 %) | 21.69 (0.31 %) | 22.28 (0.32 %) | -0.09 % | no | +2.75 % | yes (no overlap) | GPU-bound | - |
| 30 | opencv_cuda whole_call | 238.84 (0.18 %) | 241.60 (0.25 %) | 240.93 (0.08 %) | +1.16 % | yes (no overlap) | -0.28 % | yes | GPU-bound | - |
| 30 | opencv_cuda kernel_only | 236.35 (0.06 %) | 241.67 (0.01 %) | 241.67 (0.00 %) | +2.25 % | yes (no overlap) | +0.00 % | no | GPU-bound | - |
| 30 | npp whole_call | 214.64 (0.08 %) | 216.46 (0.38 %) | 216.08 (0.05 %) | +0.85 % | yes (no overlap) | -0.18 % | no | GPU-bound | - |
| 30 | npp kernel_only | 215.05 (0.02 %) | 215.45 (0.09 %) | 215.42 (0.07 %) | +0.18 % | yes (no overlap) | -0.01 % | no | GPU-bound | - |
| 40 | fkl_batch whole_call | 28.22 (0.99 %) | 28.30 (0.17 %) | 28.66 (0.19 %) | +0.30 % | no | +1.27 % | yes (no overlap) | GPU-bound | - |
| 40 | fkl_batch kernel_only | 28.01 (0.26 %) | 28.07 (0.70 %) | 28.67 (0.00 %) | +0.23 % | no | +2.13 % | yes (no overlap) | GPU-bound | - |
| 40 | opencv_cuda whole_call | 317.22 (0.14 %) | 321.17 (0.13 %) | 321.05 (0.13 %) | +1.25 % | yes (no overlap) | -0.04 % | no | GPU-bound | - |
| 40 | opencv_cuda kernel_only | 315.07 (0.04 %) | 321.65 (0.01 %) | 321.66 (0.03 %) | +2.09 % | yes (no overlap) | +0.00 % | no | GPU-bound | - |
| 40 | npp whole_call | 285.63 (0.07 %) | 287.65 (0.06 %) | 287.69 (0.01 %) | +0.71 % | yes (no overlap) | +0.01 % | no | GPU-bound | - |
| 40 | npp kernel_only | 286.38 (0.05 %) | 286.83 (0.02 %) | 286.84 (0.01 %) | +0.16 % | yes (no overlap) | +0.00 % | no | GPU-bound | - |
| 50 | fkl_batch whole_call | 34.33 (0.21 %) | 34.53 (0.15 %) | 34.75 (0.17 %) | +0.58 % | yes (no overlap) | +0.62 % | yes (no overlap) | GPU-bound | - |
| 50 | fkl_batch kernel_only | 34.22 (0.50 %) | 34.69 (0.46 %) | 34.82 (0.00 %) | +1.40 % | yes (no overlap) | +0.35 % | no (no overlap) | GPU-bound | - |
| 50 | opencv_cuda whole_call | 396.26 (0.15 %) | 400.39 (0.04 %) | 400.24 (0.09 %) | +1.04 % | yes (no overlap) | -0.04 % | no | GPU-bound | - |
| 50 | opencv_cuda kernel_only | 392.85 (0.02 %) | 401.87 (0.03 %) | 401.82 (0.02 %) | +2.30 % | yes (no overlap) | -0.01 % | no | GPU-bound | - |
| 50 | npp whole_call | 356.90 (0.13 %) | 359.23 (0.04 %) | 359.28 (0.04 %) | +0.65 % | yes (no overlap) | +0.02 % | no | GPU-bound | - |
| 50 | npp kernel_only | 357.64 (0.03 %) | 358.45 (0.02 %) | 358.45 (0.01 %) | +0.23 % | yes (no overlap) | +0.00 % | no | GPU-bound | - |
| 60 | fkl_batch whole_call | 40.61 (0.15 %) | 40.63 (0.08 %) | 41.04 (0.19 %) | +0.04 % | no | +1.02 % | yes (no overlap) | GPU-bound | - |
| 60 | fkl_batch kernel_only | 40.46 (0.27 %) | 40.88 (0.18 %) | 40.96 (0.02 %) | +1.03 % | yes (no overlap) | +0.21 % | yes (no overlap) | GPU-bound | - |
| 60 | opencv_cuda whole_call | 473.28 (0.23 %) | 477.98 (0.13 %) | 478.60 (0.23 %) | +0.99 % | yes (no overlap) | +0.13 % | no | GPU-bound | - |
| 60 | opencv_cuda kernel_only | 466.99 (0.01 %) | 476.46 (0.01 %) | 476.52 (0.02 %) | +2.03 % | yes (no overlap) | +0.01 % | no | GPU-bound | - |
| 60 | npp whole_call | 427.89 (0.11 %) | 431.04 (0.03 %) | 430.94 (0.03 %) | +0.74 % | yes (no overlap) | -0.02 % | no | GPU-bound | - |
| 60 | npp kernel_only | 428.97 (0.02 %) | 430.11 (0.00 %) | 430.09 (0.00 %) | +0.27 % | yes (no overlap) | -0.00 % | no | GPU-bound | - |
| 70 | fkl_batch whole_call | 46.77 (0.17 %) | 46.86 (0.17 %) | 47.28 (0.03 %) | +0.18 % | yes (no overlap) | +0.91 % | yes (no overlap) | GPU-bound | - |
| 70 | fkl_batch kernel_only | 46.79 (0.28 %) | 46.88 (0.11 %) | 47.11 (0.01 %) | +0.19 % | no | +0.51 % | yes (no overlap) | GPU-bound | - |
| 70 | opencv_cuda whole_call | 553.34 (0.21 %) | 559.58 (0.06 %) | 559.33 (0.13 %) | +1.13 % | yes (no overlap) | -0.05 % | no (no overlap) | GPU-bound | - |
| 70 | opencv_cuda kernel_only | 548.40 (0.02 %) | 560.01 (0.01 %) | 560.00 (0.02 %) | +2.12 % | yes (no overlap) | -0.00 % | no | GPU-bound | - |
| 70 | npp whole_call | 499.02 (0.03 %) | 502.63 (0.05 %) | 502.53 (0.05 %) | +0.72 % | yes (no overlap) | -0.02 % | no | GPU-bound | - |
| 70 | npp kernel_only | 500.46 (0.01 %) | 501.77 (0.00 %) | 501.76 (0.00 %) | +0.26 % | yes (no overlap) | -0.00 % | no | GPU-bound | - |
| 80 | fkl_batch whole_call | 52.58 (0.26 %) | 52.95 (0.42 %) | 53.65 (1.12 %) | +0.70 % | yes (no overlap) | +1.32 % | yes (no overlap) | GPU-bound | - |
| 80 | fkl_batch kernel_only | 52.71 (0.28 %) | 52.91 (0.14 %) | 53.26 (0.01 %) | +0.39 % | yes (no overlap) | +0.66 % | yes (no overlap) | GPU-bound | - |
| 80 | opencv_cuda whole_call | 632.73 (0.17 %) | 639.80 (0.11 %) | 639.96 (0.17 %) | +1.12 % | yes (no overlap) | +0.03 % | no | GPU-bound | - |
| 80 | opencv_cuda kernel_only | 628.17 (0.03 %) | 643.03 (0.03 %) | 643.05 (0.03 %) | +2.37 % | yes (no overlap) | +0.00 % | no | GPU-bound | - |
| 80 | npp whole_call | 569.89 (0.11 %) | 574.45 (0.03 %) | 574.17 (0.04 %) | +0.80 % | yes (no overlap) | -0.05 % | yes (no overlap) | GPU-bound | - |
| 80 | npp kernel_only | 571.56 (0.05 %) | 573.44 (0.00 %) | 573.44 (0.00 %) | +0.33 % | yes (no overlap) | +0.00 % | no | GPU-bound | - |
| 90 | fkl_batch whole_call | 59.02 (0.68 %) | 59.27 (0.45 %) | 60.22 (0.09 %) | +0.41 % | no | +1.60 % | yes (no overlap) | GPU-bound | - |
| 90 | fkl_batch kernel_only | 58.75 (0.06 %) | 59.22 (0.17 %) | 59.43 (0.03 %) | +0.80 % | yes (no overlap) | +0.34 % | yes (no overlap) | GPU-bound | - |
| 90 | opencv_cuda whole_call | 711.16 (0.12 %) | 718.81 (0.15 %) | 718.91 (0.14 %) | +1.08 % | yes (no overlap) | +0.01 % | no | GPU-bound | - |
| 90 | opencv_cuda kernel_only | 704.94 (0.04 %) | 720.16 (0.11 %) | 720.03 (0.07 %) | +2.16 % | yes (no overlap) | -0.02 % | no | GPU-bound | - |
| 90 | npp whole_call | 641.02 (0.05 %) | 646.13 (0.13 %) | 646.07 (0.09 %) | +0.80 % | yes (no overlap) | -0.01 % | no | GPU-bound | - |
| 90 | npp kernel_only | 643.00 (0.05 %) | 645.12 (0.00 %) | 645.12 (0.00 %) | +0.33 % | yes (no overlap) | +0.00 % | no | GPU-bound | - |
| 100 | fkl_batch whole_call | 65.35 (0.20 %) | 65.55 (0.10 %) | 66.27 (0.06 %) | +0.31 % | yes (no overlap) | +1.10 % | yes (no overlap) | GPU-bound | - |
| 100 | fkl_batch kernel_only | 64.98 (0.34 %) | 65.21 (0.19 %) | 65.62 (0.03 %) | +0.35 % | yes | +0.63 % | yes (no overlap) | GPU-bound | - |
| 100 | opencv_cuda whole_call | 803.48 (0.45 %) | 811.54 (0.05 %) | 811.14 (0.10 %) | +1.00 % | yes (no overlap) | -0.05 % | no | GPU-bound | - |
| 100 | opencv_cuda kernel_only | 797.20 (0.39 %) | 807.82 (0.02 %) | 807.80 (0.02 %) | +1.33 % | yes (no overlap) | -0.00 % | no | GPU-bound | - |
| 100 | npp whole_call | 713.00 (0.43 %) | 718.95 (0.42 %) | 718.93 (0.22 %) | +0.83 % | yes (no overlap) | -0.00 % | no | GPU-bound | - |
| 100 | npp kernel_only | 714.76 (0.55 %) | 718.06 (0.01 %) | 718.05 (0.04 %) | +0.46 % | no | -0.00 % | no | GPU-bound | - |
| 110 | fkl_batch whole_call | 71.52 (0.20 %) | 71.74 (0.16 %) | 72.47 (0.04 %) | +0.31 % | yes (no overlap) | +1.01 % | yes (no overlap) | GPU-bound | - |
| 110 | fkl_batch kernel_only | 71.00 (0.10 %) | 71.12 (0.22 %) | 71.80 (0.12 %) | +0.17 % | no | +0.96 % | yes (no overlap) | GPU-bound | - |
| 110 | opencv_cuda whole_call | 890.58 (0.22 %) | 899.10 (0.25 %) | 899.85 (0.17 %) | +0.96 % | yes (no overlap) | +0.08 % | no | GPU-bound | - |
| 110 | opencv_cuda kernel_only | 882.89 (0.33 %) | 893.87 (0.01 %) | 893.84 (0.01 %) | +1.24 % | yes (no overlap) | -0.00 % | no | GPU-bound | - |
| 110 | npp whole_call | 792.16 (0.16 %) | 800.59 (0.06 %) | 800.81 (0.03 %) | +1.06 % | yes (no overlap) | +0.03 % | no | GPU-bound | - |
| 110 | npp kernel_only | 791.87 (0.28 %) | 801.84 (0.04 %) | 801.77 (0.01 %) | +1.26 % | yes (no overlap) | -0.01 % | no | GPU-bound | - |
| 120 | fkl_batch whole_call | 77.67 (0.05 %) | 77.92 (0.28 %) | 78.67 (0.09 %) | +0.32 % | yes (no overlap) | +0.96 % | yes (no overlap) | GPU-bound | - |
| 120 | fkl_batch kernel_only | 77.06 (0.14 %) | 77.35 (0.09 %) | 77.95 (0.07 %) | +0.38 % | yes (no overlap) | +0.77 % | yes (no overlap) | GPU-bound | - |
| 120 | opencv_cuda whole_call | 985.64 (0.32 %) | 995.99 (0.15 %) | 995.24 (0.05 %) | +1.05 % | yes (no overlap) | -0.08 % | no | GPU-bound | - |
| 120 | opencv_cuda kernel_only | 978.02 (0.11 %) | 990.81 (0.04 %) | 991.00 (0.01 %) | +1.31 % | yes (no overlap) | +0.02 % | no | GPU-bound | - |
| 120 | npp whole_call | 874.80 (0.23 %) | 880.80 (0.05 %) | 880.67 (0.12 %) | +0.69 % | yes (no overlap) | -0.02 % | no | GPU-bound | - |
| 120 | npp kernel_only | 875.81 (0.28 %) | 881.11 (0.04 %) | 880.99 (0.05 %) | +0.60 % | yes (no overlap) | -0.01 % | no | GPU-bound | - |
| 130 | fkl_batch whole_call | 83.84 (0.08 %) | 84.01 (0.03 %) | 85.04 (0.18 %) | +0.21 % | yes (no overlap) | +1.22 % | yes (no overlap) | GPU-bound | - |
| 130 | fkl_batch kernel_only | 83.27 (0.20 %) | 83.48 (0.09 %) | 84.14 (0.01 %) | +0.25 % | yes (no overlap) | +0.80 % | yes (no overlap) | GPU-bound | - |
| 130 | opencv_cuda whole_call | 1080.27 (0.12 %) | 1090.84 (0.05 %) | 1090.96 (0.17 %) | +0.98 % | yes (no overlap) | +0.01 % | no | GPU-bound | - |
| 130 | opencv_cuda kernel_only | 1075.15 (0.14 %) | 1091.78 (0.02 %) | 1091.85 (0.01 %) | +1.55 % | yes (no overlap) | +0.01 % | no | GPU-bound | - |
| 130 | npp whole_call | 948.94 (0.26 %) | 957.13 (0.05 %) | 956.90 (0.02 %) | +0.86 % | yes (no overlap) | -0.03 % | no | GPU-bound | - |
| 130 | npp kernel_only | 949.32 (0.28 %) | 957.36 (0.02 %) | 957.21 (0.01 %) | +0.85 % | yes (no overlap) | -0.01 % | no (no overlap) | GPU-bound | - |
| 140 | fkl_batch whole_call | 89.92 (0.13 %) | 90.29 (0.04 %) | 91.16 (0.03 %) | +0.40 % | yes (no overlap) | +0.96 % | yes (no overlap) | GPU-bound | - |
| 140 | fkl_batch kernel_only | 89.33 (0.20 %) | 89.65 (0.18 %) | 90.43 (0.14 %) | +0.36 % | yes (no overlap) | +0.87 % | yes (no overlap) | GPU-bound | - |
| 140 | opencv_cuda whole_call | 1168.11 (0.28 %) | 1183.45 (0.03 %) | 1183.26 (0.05 %) | +1.31 % | yes (no overlap) | -0.02 % | no | GPU-bound | - |
| 140 | opencv_cuda kernel_only | 1166.46 (0.06 %) | 1182.90 (0.01 %) | 1182.89 (0.01 %) | +1.41 % | yes (no overlap) | -0.00 % | no | GPU-bound | - |
| 140 | npp whole_call | 1025.35 (0.19 %) | 1037.63 (0.08 %) | 1037.64 (0.06 %) | +1.20 % | yes (no overlap) | +0.00 % | no | GPU-bound | - |
| 140 | npp kernel_only | 1025.32 (0.27 %) | 1038.33 (0.03 %) | 1038.38 (0.01 %) | +1.27 % | yes (no overlap) | +0.01 % | no | GPU-bound | - |
| 150 | fkl_batch whole_call | 96.08 (0.54 %) | 96.38 (0.18 %) | 97.34 (0.03 %) | +0.31 % | no | +0.99 % | yes (no overlap) | GPU-bound | - |
| 150 | fkl_batch kernel_only | 95.51 (0.22 %) | 95.79 (0.06 %) | 96.81 (0.13 %) | +0.29 % | yes (no overlap) | +1.07 % | yes (no overlap) | GPU-bound | - |
| 150 | opencv_cuda whole_call | 1254.84 (0.11 %) | 1269.98 (0.02 %) | 1270.19 (0.05 %) | +1.21 % | yes (no overlap) | +0.02 % | no | GPU-bound | - |
| 150 | opencv_cuda kernel_only | 1249.23 (0.05 %) | 1271.16 (0.03 %) | 1271.05 (0.02 %) | +1.76 % | yes (no overlap) | -0.01 % | no | GPU-bound | - |
| 150 | npp whole_call | 1101.08 (0.33 %) | 1112.93 (0.34 %) | 1112.84 (0.06 %) | +1.08 % | yes (no overlap) | -0.01 % | no | GPU-bound | - |
| 150 | npp kernel_only | 1100.73 (0.26 %) | 1113.92 (0.02 %) | 1113.64 (0.01 %) | +1.20 % | yes (no overlap) | -0.02 % | yes (no overlap) | GPU-bound | - |
