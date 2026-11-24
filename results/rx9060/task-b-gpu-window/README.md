# Task B on the RX 9060: the GPU window of FKL #433, 2026-11-24, 20:18:11-20:21:24Z

FKL #433 (branch `LTS-C++17-batched-filter-dpps`, head 4b11bc6) batches LinearFilterDPP, MedianFilterDPP and the four Quad DPPs, with each image at its own size, and adds vector types. This folder holds the one GPU window on the RX 9060 that its verification needs. The coordinator session granted it for de19bd3 (window.sh sha256 `452af663…`), on the condition that it start after #433's CI job on this runner.

## Results
**Everything passed. The outputs are bit-identical before and after. The scalar median's times changed by -3.4 % to -0.2 %.**
- **The run:**
  - The window ran as one detached call; this session made no other tool call until it ended.
  - #433's ROCm job (`build (10.1)`) had completed with success before the start, and the foreground check at 20:17:54Z found no `Runner.Worker`.
  - All 29 steps exited 0 with their PASS or "checks passed" line, and no step printed a FAIL line.
  - The first step waited about 60 s for the CI's load to fall below 0.8.
- **The FKL filter tests (`summary/tests.csv`):** all 9 at the PR head and all 6 existing ones at 24478f0 pass on CPU, CPU_MULTITHREAD and the GPU.
  - The uchar Quad tests compare with `==` to their host reference: MorphQuad 11, MedianQuad 11, BoxFilterQuad 12 PASS lines, before and after. Their outputs are therefore the same before and after.
- **E18 and E19, before against after (`summary/fklb.csv`):** equal in every output.
  - E18: the 6 digests (FNV-1a 64), the FKL rows `fkl_box_halfpixel` and `fkl_half_once` among them.
  - E19 and the E19 crop sweep at N = 30: all 5 dumps, `fkl_loop_tensor.f32` (LinearFilterDPP) and `fkl_batch_tensor.f32` among them.
  - The dumps are not committed (27 MB per run); their sha256 are in `window/checks-*/*/dumps-sha256.txt`.
- **The harness outputs (`summary/outputs.csv`):** `filters_outputs.bin` has the same sha256, `4632a677…`, in all 8 runs, A and B. The GPU outputs of the 14 existing float instantiations are therefore bit-identical before and after: test_linear_filter_dpp 5, test_median_filter_dpp 4 and test_convolution_fast_dpp 5. They are within each test's tolerance of the host reference in every run.
- **The scalar median's speed (`summary/compare.csv`):** B (after) against A (before), the median of three quiet runs' means:

| case, interval | old (ms) | new (ms) | change | spread old / new | exceeds the larger spread? | > 3 %? | GPU- or host-bound (old / new) |
|---|--:|--:|--:|--:|---|---|---|
| median3 whole_call | 0.0140 | 0.0140 | -0.3 % | 4.5 / 2.0 % | no | no | host / host |
| median3 kernel_only | 0.0166 | 0.0165 | -1.1 % | 0.2 / 0.2 % | yes | no | host / host |
| median5 whole_call | 0.0390 | 0.0382 | -2.0 % | 1.0 / 1.4 % | yes | no | host / host |
| median5 kernel_only | 0.0419 | 0.0418 | -0.2 % | 0.1 / 0.1 % | yes | no | host / host |
| median7x3-fused whole_call | 0.0307 | 0.0301 | -2.1 % | 0.3 / 0.5 % | yes | no | host / host |
| median7x3-fused kernel_only | 0.0341 | 0.0334 | -1.8 % | 0.2 / 0.8 % | yes | no | host / host |
| minimum-selection whole_call | 0.0119 | 0.0118 | -0.7 % | 21.2 / 22.0 % | no | no | host / host |
| minimum-selection kernel_only | 0.0147 | 0.0142 | -3.4 % | 0.5 / 5.9 % | no | FLAG | host / host |

  - No cause is claimed for any change.
  - The bound column uses the crop sweep's criterion: GPU-bound when kernel_only and whole_call differ by less than 3 %. Here kernel_only is 7.5 % to 23.4 % above whole_call in every case. These images are 31 x 15 to 37 x 19 pixels, 4 to 9 blocks of 16 x 8, as in the test.
  - The clock (`summary/clock.csv`): in every run's timing no hwmon sample was below 3.1 GHz. The median sclk was 3,391 MHz in unused-A (the first harness run) and 3,481-3,482 MHz in the others. The median power was 32-36 W; the maximum temperatures were 43 °C edge, 46 °C junction and 40 °C memory.
  - The unused runs (`summary/unused.csv`): unused-A's median3 whole_call is +20.8 % above A's quiet median. Its sclk after the warm-up was 3,284 MHz, against 3,480-3,487 MHz in the other runs.
- **Reproduction:** the summary was made by `analysis/task_b_report.py` from the whole window folder (`tmp/taskb/window-run` on this machine) and reproduces from it byte for byte. The committed `window/` leaves out the E19 dumps and `filters_outputs.bin`, but keeps their sha256.

**Decision:** the coordinator session's task B brief of 2026-11-24. It implements Oscar's answers to Q1-Q16, his "Accept" of the median's NaN difference, and his request for the scalar median's speed before and after. The coordinator asked for one window script covering the final HIP tests and the speed.

## What the window runs (`window.sh`)
One call. It stops at the first failed step, and checks that the runner is idle (no `Runner.Worker`, load below 0.8) before every run. Every binary it runs is listed with its sha256 in `manifest.sha256`, and the guards check all of them before anything starts.
1. **`tests-after`:** the 9 filter tests of FKL's `tests/image_processing` at the PR head, the 6 existing ones and the 3 new ones.
   - These are HIP builds, which run the CPU, CPU_MULTITHREAD and GPU cases.
   - Each must exit 0, and its last line must end in PASS.
2. **`tests-before`:** the 6 existing filter tests built at 24478f0, with the same rule.
   - The uchar Quad tests (MorphQuad, MedianQuad, BoxFilterQuad) compare every output with `==` to their host reference. With step 1, they show bit-identical outputs before and after.
3. **`checks-before` and `checks-after`:** the checks only of E18 box (`E18_NO_TIMING=1`), E19 (`E19_NO_TIMING=1`) and the E19 crop sweep at N = 30 (`E19_CROPS=30`).
   - They are built against 24478f0 and against the PR head; E19 also gets `E19_DUMP`.
   - For the PR head, only their LinearFilterDPP details are ported, in a scratch copy: `fklb-after-details-port.diff`.
   - The comparison afterwards: E18's output digests and E19's dumps, before against after.
4. **The harness** (`harness/filters_before_after.hip`, built by `harness/build.sh`): A against 24478f0, B against the PR head.
   - **The outputs:** every run first checks and dumps the GPU outputs of the existing float filter instantiations: test_linear_filter_dpp (5), test_median_filter_dpp (4) and test_convolution_fast_dpp (5). These tests check with a tolerance, so their outputs are compared bit for bit, before against after.
   - **The speed:** it then times the four scalar median cases with E19's protocol:
     - a 2 s clock warm-up spin;
     - whole_call: 100 calls after one warm-up call;
     - kernel_only: behind a 5 ms spin;
     - sclk read at every boundary.
   - **The run order:** unused-A, unused-B (not used), then quiet-run1..3, alternating A and B.
- **The sampler:** hwmon sclk, mclk, power and the three temperatures every 100 ms (`gpu_clocks.csv`).

## The binaries (`binaries.txt`, `manifest.sha256`)
They were built on 2026-11-24 on this machine, with no GPU use.

## After the window
`analysis/task_b_report.py <window> <out>` writes:
- the tests' table;
- E18 and E19's before/after equality;
- the harness outputs' equality;
- the median times as old / new / change / spread / exceeds / > 3 %, with the clock, the power, the temperatures and the unused runs.

No cause is claimed for any change.
