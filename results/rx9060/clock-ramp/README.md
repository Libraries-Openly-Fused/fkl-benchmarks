# The graphics clock of the RX 9060 and the first timed variant: the evidence

The first run of the RPP comparison (warm-up 0.5 s) read the default (ASYMMETRIC) fused chain of pipeline (a) 24 to 27 percent slower at batches 200 and 300 at FKL c8fcbe8 than an earlier run
of the same binary at 38c4f32 (0.1146 / 0.1577 ms against 0.0924 / 0.1241 ms), with the HALF_PIXEL variant of the same binary unchanged. This folder is how the cause was found. It is not
FKL: the GPU is still ramping its clock when the first variant of a point is timed (see `../README.md` for the rule that follows). Dates: 2026-10-31; machine and conditions as in `../README.md`.
*Correction (review of fkl-benchmarks #6): an earlier version of this paragraph said "22 to 27 percent". The two figures quoted give +24.0 percent at batch 200 (0.1146 / 0.0924) and +27.1 percent at batch 300 (0.1577 / 0.1241);
the 22 percent does not follow from these two figures (22.5 percent is the ratio of the total times of section 5, 0.1549 / 0.1265 ms), and the four times behind them are in the history of the branch, not in a file of this folder.*

## 1. Five pins of FKL, three rounds, alternating (`five-pins-three-rounds/`)

The same benchmark source (fkl-benchmarks `e7f28e2`), the same flags, five builds that differ only in `external/fkl`: `38` = 38c4f32 (the pin of the first run; #409 is in it), `b44725e` = the
merge of #411, `d776713` = #414, `bbc5a7b` = #416, `c8` = c8fcbe8 (#417, the final pin). `r<round>_<pin>/` has the timing CSVs of pipelines (a) and (b) of that run; the pins alternate in every round;
no CI job ran during any of the 15 runs. `summary.txt`: median ms of `fkl_fused`, `fkl_fused_halfpixel`, `fkl_perop_batch` and `rpp_fused_api` at the batches 100, 200 and 300 in each round.
`fkl_fused` (default resize), mean of the three rounds: batch 100: 0.0623 to 0.0624 ms at all pins; batch 200: 0.1094, 0.1107, 0.1109, 0.1114, 0.1097; batch 300: 0.1508, 0.1510, 0.1499, 0.1506, 0.1505.
The build of 38c4f32 made today is as slow as the one of c8fcbe8, and no step between them changes anything: no FKL commit is responsible. The HALF_PIXEL variant and RPP are flat at all pins to 1.5 percent; the per-op batch is not (below).
*Correction: an earlier version said that these variants were flat at all pins, and `../README.md` said that the times are the same at the five pins within 1 percent. Recomputed from the CSVs of `five-pins-three-rounds/`
(`five-pins-spread.py`, output `five-pins-spread.txt`; spread = (max - min) / median of the five pin means, each the mean of three rounds): `fkl_fused` differs between the pins by 0.2 to 2.5 percent over the 30 batches (6 of the
30 above 1 percent; 0.16, 1.74 and 0.74 percent at the batches 100, 200 and 300; the final pin against the first: -0.14, +0.27 and -0.14 percent), `fkl_fused_halfpixel` by up to 1.5 percent, `rpp_fused_api` by up to 1.1 percent and
`fkl_perop_batch` by up to 8.3 percent (batch 180; 16 of its 30 batches above 1 percent). Over the 120 cells of pipeline (a), 25 are above 1 percent, 6 above 2 and 1 above 5. There is no order by pin, so the conclusion above stands.*

## 2. The device code is the same (`device-code-identical-at-five-pins.txt`)

The gfx1200 code objects of `benchmark_rpp_preprocessing_hip` (238 functions), `benchmark_horizontal_fusion_hip` (246) and `benchmark_nv12_pipeline_hip` (46) are identical function by function at the five pins
(instruction text hashed with the `hip_cuid` names normalised). The chain kernel of batch 300 (ASYMMETRIC: 201 instructions, HALF_PIXEL: 211) has 21 VGPRs, 22 SGPRs, no scratch and no spills in both, a
kernarg segment of 352 + 48 * batch bytes (14,752 B at 300) that is read in place with 9 scalar loads. The host side of a launch costs about 2 microseconds at batch 300 (the `burst launch` column of `standalone-harness/hostsplit-*.txt`: 0.0020 to 0.0023 ms, 20 launches back to back without event records), measured at two pins, 38c4f32 and c8fcbe8.
*Correction: an earlier version said "4 microseconds ... at every pin". The `host(launch)` column (0.0041 to 0.0043 ms at batch 300) is the launch plus the two `hipEventRecord` calls that the harness puts around it (about 2 microseconds; 3.4 to 3.5 us
against 1.5 to 1.6 us at batch 100, 3.8 to 3.9 against 1.7 to 1.9 at batch 200), and only two pins have a file.*

## 3. The old build against the new one (`old-build-vs-new-build/`)

fkl-benchmarks `4947870` (the source that produced the first run) built with 38c4f32, against `e7f28e2` with 38c4f32, alternating, two rounds; `clocks.txt`: the graphics clock and the memory clock
(Hz) and the GPU busy percentage about every 115 ms during the run (8.7 Hz: 840 samples in 96.5 s; an earlier version said 10 Hz). Both builds are as slow as in section 1 (batch 300: 0.1500 to 0.1579 ms) with identical device code, so the build and the source are not the cause.
In the busy samples (GPU busy at least 90 percent) of every one of the four runs the graphics clock had both states: 10th percentile 1.87 to 2.02 GHz, median 3.06 to 3.07 GHz, 90th percentile 3.21 to 3.22 GHz, 19 to 20 percent of the samples below 2.3 GHz;
the memory clock was 1124 MHz in 683 to 692 of the 765 to 772 busy samples (89 percent) and 96, 456, 772 or 875 MHz in the others.
*Correction: an earlier version said "1.87 to 2.02 GHz or 3.1 GHz depending on the run (the memory clock was 1.124 GHz always)".*

## 4. The clock during the benchmark (`clock-trace-0.5s-warmup/`)

The first RPP benchmark (warm-up 0.5 s, 38c4f32) under `scripts/gpu_clock_trace.py` (about 176 Hz: 17,000 samples in 96.6 s): `lines.txt` (every output line with its arrival time), `clk.txt` (time, MHz, GPU busy %),
`report-fkl_fused-windows.txt`: the clock in the 0.52 s that end at each `fkl_fused` / `fkl_fused_halfpixel` timing line (the warm-up and the timed runs of the variant). The lines alternate fkl_fused,
fkl_fused_halfpixel for the batches 10, 20, ..., 300, then the same for the crops of pipeline (b). During the warm-up and the timed runs of `fkl_fused` at the batches 150 to 300, the clock is 2.0 GHz
(2017 to 2029 MHz in the last 50 ms; 2659 MHz at batch 100); during `fkl_fused_halfpixel`, right after, 2.8 to 3.1 GHz; during the next variants 3.0 to 3.2 GHz. The GPU is idle (31 MHz) while the host computes the
scalar references between two batches, and the 0.5 s of warm-up are shorter than the ramp.

## 5. Standalone harness (`standalone-harness/`)

`hostsplit.cpp` includes `rpp/benchmark_rpp_preprocessing.h`, builds the fused chain of pipeline (a) for batch 100, 200, 300 and times it with the host launch (from before the first `hipEventRecord` to after the second, so `executeOperations` and the two event records; the column `burst launch` is the launch alone, 20 back to back; the comment at the top of `hostsplit.cpp` still says that `host` is the time until `executeOperations` returns, what the code times is what is written here, and its comment on the sampler thread says about 200 Hz, a rate that was not measured), the whole
run with the stream synchronisation (what the benchmark times) and HIP events, sampling the graphics clock from a thread. `hostsplit-38c4f32.txt` and `hostsplit-c8fcbe8.txt`: (i) back to back, the GPU
busy since the start: batch 300 ASYMMETRIC 0.1277 ms (device 0.1062), HALF_PIXEL 0.1310 ms (device 0.1095) at 3.0 GHz; (ii) after 3 s without GPU work: a warm-up of 0.5 s gives ASYMMETRIC 0.1549 ms (device
0.1328) at 2.0 GHz, and a warm-up of 2 s or 5 s gives 0.1265 ms (device 0.1050) at 3.0 GHz; the trace of the warm-up per 0.25 s: 2001, 2023, 2099, 2595, 2976, 2998, 3019, 3044 MHz, i.e. the boost arrives after
0.9 to 1.0 s of continuous work. The results are the same at both pins.

## 6. Other results of the rule

- E4b (`../e4b-nv12-steady-clock/` against `../e4b-nv12/`): the convert kernel at one crop 0.2602 ms with a warm-up of 0.5 s (in the overshoot of the clock, 3.3 GHz and more) and 0.2808 ms at the steady clock.
- Latency hiding (`../fkl-self/`, FKL-only benchmarks): without keeping the GPU busy before each point the series is up to 52 percent slower at 301 operations.
