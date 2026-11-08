## Findings

**Verdict: binary composition, not drift.** FKL's higher host time in the window e4-hand-cvgs belongs to its binary (2dac1c8). Window B's binary (836a98e), run in the same window, alternating, still has window B's host times.

1. **New against old, in this window** (medians of the per-batch changes, batches 50 to 300):
   - fkl_fused: host_enqueue_us +4.9 % (-0.6 to +10.3), host_launch_us +5.0 % (+0.0 to +9.8);
   - fkl_fused_half_pixel: +4.6 % and +5.3 %.
   - This is the size of the change that the window e4-hand-cvgs showed against window B (+5.0 % and +5.4 %).
   - gpu_mean_ms follows with +2.0 % for both versions, because the GPU waits for FKL's host work inside the interval.
2. **No drift.**
   - The old binary today against window B (2026-11-07): host_enqueue_us -1.4 %, host_launch_us -1.2 % (fkl_fused; half pixel -1.1 % and -1.5 %).
   - The new binary today against the window e4-hand-cvgs: -1.3 % and -1.7 % (half pixel -1.1 % and -1.1 %).
   - Each binary keeps its own host time from one window to the next.
3. **The device code is the same in both binaries** (SASS identical, the step 0 of e4-hand-cvgs). So is the source of the FKL call: the launch lambdas are unchanged since 836a98e.
   - What differs is the compiled host code around the call. At batch 300, `fklb::timeIt` of a launch is a function of its own in 836a98e (3,126 bytes) and is inlined in 2dac1c8, and the `measure<...>` functions grow from 5,646 to 6,317 bytes.
   - Part of that growth is #34's own change to `measure()`, which now returns the checked output. The translation unit also holds cvGPUSpeedup and 90 more kernels. Which of the two changes FKL's host path is not separated here; either way it is the binary, not the machine or the day.
4. **The GPU-time drift columns** (old against window B +3.0 %, new against e4-hand-cvgs +3.1 %) are this window's lower SM clock, not the binaries.
   - With only the two FKL versions, a run is mostly tiny kernels with gaps, and the GPU ran at 2,550 MHz in five of the six runs; the full sweeps ran at 2,685 to 2,700 MHz.
   - Both binaries ran under the same conditions here, so the new-against-old comparison is not affected.
5. **For the window e4-hand-cvgs:** its fkl_fused and fkl_fused_half_pixel calls, and so its hand / FKL ratios, carry this host-code effect. With 836a98e's host code, FKL's call takes about 5 % less host time and about 2 % less time in the interval. hand_batch was compiled in the same binary as the FKL versions it is compared with.
