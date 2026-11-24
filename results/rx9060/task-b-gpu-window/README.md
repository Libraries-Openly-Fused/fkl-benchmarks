# Task B on the RX 9060: the GPU window of FKL #433 (not run yet)

FKL #433 (branch `LTS-C++17-batched-filter-dpps`, head 4b11bc6) batches LinearFilterDPP, MedianFilterDPP and the four Quad DPPs, with each image at its own size, and adds vector types. This folder holds the one GPU window on the RX 9060 that its verification needs. Until the coordinator session grants it, nothing here has run.

**Decision:** the coordinator session's task B brief of 2026-11-24. It implements Oscar's answers to Q1-Q16, his "Accept" of the median's NaN difference, and his request for the scalar median's speed before and after. The coordinator asked for one window script covering the final HIP tests and the speed.

## What the window runs (`window.sh`)
One foreground call. It stops at the first failed step, and checks that the runner is idle (no `Runner.Worker`, load below 0.8) before every run. Every binary it runs is listed with its sha256 in `manifest.sha256`, and the guards check all of them before anything starts.
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
