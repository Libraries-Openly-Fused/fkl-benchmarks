# The graphics clock of the RX 9060 and the first timed variant: the evidence

The first run of the RPP comparison (warm-up 0.5 s) read the default (ASYMMETRIC) fused chain of pipeline (a) 22 to 27 percent slower at batches 200 and 300 at FKL c8fcbe8 than an earlier run
of the same binary at 38c4f32 (0.1146 / 0.1577 ms against 0.0924 / 0.1241 ms), with the HALF_PIXEL variant of the same binary unchanged. This folder is how the cause was found. It is not
FKL: the GPU is still ramping its clock when the first variant of a point is timed (see `../README.md` for the rule that follows). Dates: 2026-10-31; machine and conditions as in `../README.md`.

## 1. Five pins of FKL, three rounds, alternating (`five-pins-three-rounds/`)

The same benchmark source (fkl-benchmarks `e7f28e2`), the same flags, five builds that differ only in `external/fkl`: `38` = 38c4f32 (the pin of the first run; #409 is in it), `b44725e` = the
merge of #411, `d776713` = #414, `bbc5a7b` = #416, `c8` = c8fcbe8 (#417, the final pin). `r<round>_<pin>/` has the timing CSVs of pipelines (a) and (b) of that run; the pins alternate in every round;
no CI job ran during any of the 15 runs. `summary.txt`: median ms of `fkl_fused`, `fkl_fused_halfpixel`, `fkl_perop_batch` and `rpp_fused_api` at the batches 100, 200 and 300 in each round.
`fkl_fused` (default resize), mean of the three rounds: batch 100: 0.0623 to 0.0624 ms at all pins; batch 200: 0.1094, 0.1107, 0.1109, 0.1114, 0.1097; batch 300: 0.1508, 0.1510, 0.1499, 0.1506, 0.1505.
The build of 38c4f32 made today is as slow as the one of c8fcbe8, and no step between them changes anything: no FKL commit is responsible. The HALF_PIXEL variant, the per-op batch and RPP are flat at all pins.

## 2. The device code is the same (`device-code-identical-at-five-pins.txt`)

The gfx1200 code objects of `benchmark_rpp_preprocessing_hip` (238 functions), `benchmark_horizontal_fusion_hip` (246) and `benchmark_nv12_pipeline_hip` (46) are identical function by function at the five pins
(instruction text hashed with the `hip_cuid` names normalised). The chain kernel of batch 300 (ASYMMETRIC: 201 instructions, HALF_PIXEL: 211) has 21 VGPRs, 22 SGPRs, no scratch and no spills in both, a
kernarg segment of 352 + 48 * batch bytes (14,752 B at 300) that is read in place with 9 scalar loads. The host side of a launch costs 4 microseconds at batch 300 (`standalone-harness/`), at every pin.

## 3. The old build against the new one (`old-build-vs-new-build/`)

fkl-benchmarks `4947870` (the source that produced the first run) built with 38c4f32, against `e7f28e2` with 38c4f32, alternating, two rounds; `clocks.txt`: the graphics clock and the memory clock
(Hz) and the GPU busy percentage at 10 Hz during the run. Both builds are as slow as in section 1 (batch 300: 0.1500 to 0.1579 ms) with identical device code, so the build and the source are not the cause.
The graphics clock sampled during the runs was 1.87 to 2.02 GHz or 3.1 GHz depending on the run (the memory clock was 1.124 GHz always).

## 4. The clock during the benchmark (`clock-trace-0.5s-warmup/`)

The first RPP benchmark (warm-up 0.5 s, 38c4f32) under `scripts/gpu_clock_trace.py` (200 Hz): `lines.txt` (every output line with its arrival time), `clk.txt` (time, MHz, GPU busy %),
`report-fkl_fused-windows.txt`: the clock in the 0.52 s that end at each `fkl_fused` / `fkl_fused_halfpixel` timing line (the warm-up and the timed runs of the variant). The lines alternate fkl_fused,
fkl_fused_halfpixel for the batches 10, 20, ..., 300, then the same for the crops of pipeline (b). During the warm-up and the timed runs of `fkl_fused` at the batches 150 to 300, the clock is 2.0 GHz
(2017 to 2029 MHz in the last 50 ms; 2659 MHz at batch 100); during `fkl_fused_halfpixel`, right after, 2.8 to 3.1 GHz; during the next variants 3.0 to 3.2 GHz. The GPU is idle (31 MHz) while the host computes the
scalar references between two batches, and the 0.5 s of warm-up are shorter than the ramp.

## 5. Standalone harness (`standalone-harness/`)

`hostsplit.cpp` includes `rpp/benchmark_rpp_preprocessing.h`, builds the fused chain of pipeline (a) for batch 100, 200, 300 and times it with the host launch (until `executeOperations` returns), the whole
run with the stream synchronisation (what the benchmark times) and HIP events, sampling the graphics clock from a thread. `hostsplit-38c4f32.txt` and `hostsplit-c8fcbe8.txt`: (i) back to back, the GPU
busy since the start: batch 300 ASYMMETRIC 0.1277 ms (device 0.1062), HALF_PIXEL 0.1310 ms (device 0.1095) at 3.0 GHz; (ii) after 3 s without GPU work: a warm-up of 0.5 s gives ASYMMETRIC 0.1549 ms (device
0.1328) at 2.0 GHz, and a warm-up of 2 s or 5 s gives 0.1265 ms (device 0.1050) at 3.0 GHz; the trace of the warm-up per 0.25 s: 2001, 2023, 2099, 2595, 2976, 2998, 3019, 3044 MHz, i.e. the boost arrives after
0.9 to 1.0 s of continuous work. The results are the same at both pins.

## 6. Other results of the rule

- E4b (`../e4b-nv12-steady-clock/` against `../e4b-nv12/`): the convert kernel at one crop 0.2602 ms with a warm-up of 0.5 s (in the overshoot of the clock, 3.3 GHz and more) and 0.2808 ms at the steady clock.
- Latency hiding (`../fkl-self/`, FKL-only benchmarks): without keeping the GPU busy before each point the series is up to 52 percent slower at 301 operations.
