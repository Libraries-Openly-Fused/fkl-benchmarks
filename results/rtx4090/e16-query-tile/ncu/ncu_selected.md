| metric | raw_bq64_kv32_w4 | int8_bq64_w4 | int8_bq128_w4 | int8nodq_bq64_w4 (TIMING ONLY) |
|---|---:|---:|---:|---:|
| gpu__time_duration.sum [ns] | 1,853,728 | 2,407,392 | 2,123,456 | 2,203,200 |
| smsp__inst_executed.sum [inst] | 434,649,109 | 764,481,084 | 521,842,236 | 599,883,536 |
| smsp__thread_inst_executed_per_inst_executed.ratio [] | 30.19 | 30.97 | 30.49 | 30.69 |
| smsp__inst_issued.sum [inst] | 434,682,391 | 764,510,780 | 521,867,836 | 599,913,232 |
| smsp__issue_active.avg.pct_of_peak_sustained_active [%] | 18.15 | 24.51 | 19.12 | 20.97 |
| sm__warps_active.avg.pct_of_peak_sustained_active [%] | 16.63 | 16.58 | 16.60 | 16.42 |
| launch__registers_per_thread [register/thread] | 171 | 174 | 255 | 174 |
| launch__shared_mem_per_block_dynamic [byte/block] | 24,576 | 49,152 | 49,152 | 49,152 |
| launch__occupancy_limit_registers [block] | 2 | 2 | 2 | 2 |
| launch__occupancy_limit_shared_mem [block] | 2 | 2 | 2 | 2 |
| l1tex__data_pipe_lsu_wavefronts_mem_shared.sum [] | 150,255,799 | 222,557,647 | 118,445,194 | 226,503,292 |
| l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum [] | 134,748,388 | 151,708,739 | 76,575,449 | 151,853,772 |
| l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum [] | 0 | 34,687,044 | 16,798,448 | 36,544,854 |
| l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum [] | 0 | 4,687,916 | 1,957,984 | 8,630,470 |
| dram__bytes_read.sum [byte] | 201,369,472 | 137,028,864 | 136,530,176 | 135,587,328 |
| smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio [inst] | 0.54 | 0.42 | 0.54 | 0.67 |
| smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio [inst] | 0.20 | 0.84 | 0.86 | 0.13 |
| smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio [inst] | 0.46 | 0.75 | 0.90 | 0.85 |
| smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio [inst] | 5.73 | 2.34 | 4.31 | 4.18 |
| smsp__average_warps_issue_stalled_wait_per_issue_active.ratio [inst] | 2.51 | 2.30 | 2.41 | 2.08 |
| sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active [%] | 6.31 | 22.04 | 15.35 | 5.29 |
| sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active [] | n/a | n/a | n/a | n/a |
