| id | kernel | duration_us | sm_frequency | elapsed_cycles | compute_sm_throughput_pct | memory_throughput_pct | dram_throughput_pct | l1_throughput_pct | l2_throughput_pct | issue_slots_busy_pct | executed_ipc_active | issued_ipc_active | sm_busy_pct |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | mma_raw attention (bf16 path) | 1803.49 | 2,539,941,844.01 | 4,588,863 | 45.78 | 49.88 | 14.23 | 25.52 | 49.88 | 15.24 | 0.61 | 0.61 | 45.78 |
| 1 | mma_int8kv_fused attention | 2413.25 | 2,541,755,042.47 | 6,143,024 | 34.19 | 30.58 | 7.66 | 30.76 | 18.95 | 24.34 | 0.98 | 0.98 | 34.19 |
| 2 | dequantKV (int8 unfused) | 247.94 | 2,537,048,417.19 | 629,791 | 28.67 | 70.68 | 70.68 | 21.04 | 26.82 | 28.67 | 1.16 | 1.16 | 28.67 |
| 3 | mma_int8kv_unfused attention (bf16 path) | 1799.58 | 2,544,560,500.45 | 4,583,387 | 45.80 | 50.26 | 14.25 | 25.51 | 50.26 | 15.25 | 0.61 | 0.61 | 45.80 |
| 4 | mma_fp8kv_fused attention | 2414.37 | 2,540,959,222.61 | 6,145,413 | 34.18 | 30.59 | 7.70 | 30.74 | 19.73 | 28.11 | 1.13 | 1.13 | 34.18 |
| 5 | mma_epilogue_fused attention | 1807.62 | 2,539,384,068.70 | 4,599,599 | 45.69 | 50.10 | 14.19 | 25.51 | 50.10 | 15.28 | 0.62 | 0.62 | 45.69 |
| 6 | mma_epilogue_separate attention (bf16 path) | 1806.94 | 2,539,590,199.10 | 4,597,937 | 45.70 | 50.13 | 14.29 | 25.51 | 50.13 | 15.22 | 0.61 | 0.61 | 45.70 |
| 7 | affineBf16 (epilogue separate) | 118.05 | 2,532,365,108.80 | 299,295 | 16.44 | 86.14 | 86.14 | 16.75 | 28.74 | 12.35 | 0.50 | 0.50 | 12.35 |

Top stall reasons (warp cycles stalled per issued instruction, raw page):

- mma_raw attention (bf16 path): math_pipe_throttle 7.78, wait 2.15, selected 1.00, barrier 0.75, not_selected 0.25
- mma_int8kv_fused attention: math_pipe_throttle 2.33, wait 2.30, selected 1.00, long_scoreboard 0.86, short_scoreboard 0.74
- dequantKV (int8 unfused): long_scoreboard 24.72, wait 1.60, short_scoreboard 1.20, selected 1.00, not_selected 0.92
- mma_int8kv_unfused attention (bf16 path): math_pipe_throttle 7.78, wait 2.15, selected 1.00, barrier 0.75, not_selected 0.25
- mma_fp8kv_fused attention: math_pipe_throttle 2.12, wait 1.71, selected 1.00, long_scoreboard 0.81, short_scoreboard 0.56
- mma_epilogue_fused attention: math_pipe_throttle 7.75, wait 2.13, selected 1.00, barrier 0.75, not_selected 0.26
- mma_epilogue_separate attention (bf16 path): math_pipe_throttle 7.78, wait 2.15, selected 1.00, barrier 0.75, not_selected 0.25
- affineBf16 (epilogue separate): long_scoreboard 59.11, wait 2.87, short_scoreboard 1.43, drain 1.25, selected 1.00

Top pipe utilisations (% of peak sustained active, raw page):

- mma_raw attention (bf16 path): sm__pipe_tensor_op_hmma_cycles_active_v2 46.0, sm__pipe_tensor_cycles_active_v2 46.0, sm__inst_executed_pipe_tensor_op_hmma_v2 23.0, sm__inst_executed_pipe_lsu 17.6, sm__pipe_alu_cycles_active 9.6
- mma_int8kv_fused attention: sm__pipe_tensor_op_hmma_cycles_active_v2 34.4, sm__pipe_tensor_cycles_active_v2 34.4, sm__inst_executed_pipe_xu 22.0, sm__inst_executed_pipe_lsu 21.7, sm__pipe_alu_cycles_active 17.8
- dequantKV (int8 unfused): sm__inst_executed_pipe_uniform 23.7, sm__inst_executed_pipe_lsu 21.0, sm__pipe_alu_cycles_active 17.1, sm__inst_executed_pipe_alu 17.1, sm__inst_executed_pipe_adu 10.5
- mma_int8kv_unfused attention (bf16 path): sm__pipe_tensor_op_hmma_cycles_active_v2 46.0, sm__pipe_tensor_cycles_active_v2 46.0, sm__inst_executed_pipe_tensor_op_hmma_v2 23.0, sm__inst_executed_pipe_lsu 17.5, sm__pipe_alu_cycles_active 9.6
- mma_fp8kv_fused attention: sm__pipe_tensor_op_hmma_cycles_active_v2 34.4, sm__pipe_tensor_cycles_active_v2 34.4, sm__pipe_alu_cycles_active 25.3, sm__inst_executed_pipe_alu 25.3, sm__inst_executed_pipe_lsu 21.7
- mma_epilogue_fused attention: sm__pipe_tensor_op_hmma_cycles_active_v2 46.0, sm__pipe_tensor_cycles_active_v2 46.0, sm__inst_executed_pipe_tensor_op_hmma_v2 23.0, sm__inst_executed_pipe_lsu 17.5, sm__pipe_alu_cycles_active 9.6
- mma_epilogue_separate attention (bf16 path): sm__pipe_tensor_op_hmma_cycles_active_v2 46.0, sm__pipe_tensor_cycles_active_v2 46.0, sm__inst_executed_pipe_tensor_op_hmma_v2 23.0, sm__inst_executed_pipe_lsu 17.6, sm__pipe_alu_cycles_active 9.6
- affineBf16 (epilogue separate): sm__inst_executed_pipe_lsu 16.8, sm__pipe_alu_cycles_active 8.4, sm__inst_executed_pipe_alu 8.4, sm__pipe_fmaheavy_cycles_active 7.0, sm__inst_executed_pipe_fmaheavy 5.6
