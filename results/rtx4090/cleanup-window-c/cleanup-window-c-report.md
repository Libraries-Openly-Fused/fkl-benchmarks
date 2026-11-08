# Cleanup window C, 2026-11-08: E1 and E2, published, reference now and cleaned

## Findings

1. **Order of the passes.** The cleaned programs time every variant's normal pass and then its kernel-only pass (100 launches, each behind a one-thread GPU spin of 5 ms: about 0.5 s of a nearly idle GPU), so the next variant (E1) or pass (E2: warm, kernel-only, flushed) starts after that pass instead of after the previous variant's normal pass. In the reference and the published runs, E1a's variants timed second and third at 5,002 and 10,002 operations (hand_imm, fkl) are 3-5 % slower than the first (hand_params); in the cleaned build the three agree within 0.8 %. The paper's E1a vs_ref at 5,002 and 10,002 operations moves from 1.026 and 1.018 (published) to 0.994 and 0.989 (cleaned; kernel-only 0.998 and 0.998). E2's flushed pass at k = 5,002 is 3.0 % faster (it now follows the kernel-only pass). Probable cause: the GPU's clock or power state after sustained load; not isolated (no clock log in window C).
2. **E1c drift.** The published set (e1-preproc-v3) is 3-6 % slower than reference now on every version; reference now matches e1-preproc-v2 of the same day within 2.3 %. The ratios move only at batch 10 to 100, where the interval is 3.5-8 us and 4-17 % of it is dispatch (kernel-only / normal 0.83-0.96); there the kernel-only ratios against hand_padded are 0.975-1.004.
3. **Kernel-only against normal.** At 10,002 and 19,902 operations (E1a) kernel-only is 4 % below normal: 40-90 us, far more than a dispatch, and a 10 ms spin lowers it by only 0.3-0.5 % more. There the kernel-only time is the kernel after a nearly idle GPU, not only the kernel without its dispatch. E2's FKL warm interval holds no visible dispatch (kernel-only / normal 1.01-1.02 at k <= 500). The Python driver's warm intervals of Triton and Inductor are 37-71 % dispatch at k <= 100 (kernel-only / normal 0.29-0.63).
4. **E2 without the dispatch.** Kernel-only, FKL takes 0.0205 ms at k = 2 and 0.0653 ms at k = 500, Triton 0.0180 and 0.0592, Inductor (default) 0.0180 and 0.0594: FKL's kernel is 10-38 % slower than theirs at k <= 500. That is the order of the published L2-flushed times; the warm times compare calls with their dispatch.
5. **The 10 ms spin against the 5 ms spin.** Within 1.9 % for the 72 kernel-only numbers above 10 us. Of the 22 numbers at or below 10 us, three differ by more: E1b batch 1 hand_generic +16.5 % and iterated +10.5 %, and E1c batch 100 hand_generic +4.5 % (one run each).
6. **Agreement.** All bit-exact flags are 1 on all three sides. The E2 FKL output hashes are identical in all six runs and equal to the outputs in e2-work/ (2026-10-31, the published runs). The Python accuracy, kernel and SASS columns equal the published ones.
7. **ncu.** The pass ran with ncu's default (demangled) kernel names; the published pass and scripts/e1_table.py use mangled names. Matched by launch order: the same kernels, registers, launch configurations and theoretical occupancy; achieved occupancy within 0.6 (E1a), 5.5 (E1b) and 5.0 (E1c) points. The cleaned E1 tables need the pass again with --print-kernel-base mangled.

## Changes above the spread and above the 3 % (GPU) or 5 % (host) threshold

33 numbers of 312 compared:

- **E1a, cleanup**, 2 numbers: 5002 hand_imm (GPU) -4.8 % (reference now 0.5521, cleaned 0.5254; spreads 0.6 %, 1.6 %); 5002 fkl (GPU) -3.8 % (reference now 0.548, cleaned 0.5275; spreads 0.7 %, 0.9 %). Cause: order of the passes: in the cleaned build every variant's normal pass follows the previous variant's kernel-only pass (100 one-thread spins of 5 ms, about 0.5 s), in the reference the previous variant's normal pass; at 5,002 and 10,002 operations the reference's hand_imm and fkl (timed second and third) are 3-5 % slower than hand_params (timed first), as in the published runs, and the cleaned build's three variants agree within 0.8 %; probable GPU clock or power state, not isolated
- **E1b, cleanup**, 1 number: 1 iterated (host) +10.6 % (reference now 0.002867, cleaned 0.003172; spreads 0.7 %, 0.7 %). Cause: what precedes the pass: in the cleaned build fkl's kernel-only pass runs between fkl's and iterated's timing; the launch code inside the interval is the same in both builds (one executeOperations per crop); the extra run with the 10 ms spin gave 0.002819 ms, as the reference (0.002867); a 3 us host-bound number; not isolated
- **E1c, drift**, 16 numbers: 10 hand (GPU) -3.2 % (published 0.003591, reference now 0.003477; spreads 0.5 %, 2.2 %); 10 hand_generic (GPU) -5.7 % (published 0.003824, reference now 0.003607; spreads 2.1 %, 2.2 %); 50 hand (GPU) -4.2 % (published 0.005824, reference now 0.005581; spreads 1.4 %, 0.2 %); 50 hand_generic (GPU) -6.0 % (published 0.005873, reference now 0.005519; spreads 1.7 %, 2.1 %); 50 hand_padded (GPU) -5.6 % (published 0.006031, reference now 0.005692; spreads 0.7 %, 1.2 %); 50 fkl (GPU) -5.7 % (published 0.006075, reference now 0.00573; spreads 0.8 %, 0.5 %); 100 hand (GPU) -3.3 % (published 0.008086, reference now 0.00782; spreads 2.1 %, 0.9 %); 100 fkl (GPU) -3.6 % (published 0.008733, reference now 0.00842; spreads 0.9 %, 0.8 %); 200 hand (GPU) -5.9 % (published 0.01303, reference now 0.01226; spreads 0.6 %, 0.7 %); 200 hand_generic (GPU) -5.6 % (published 0.01338, reference now 0.01263; spreads 0.5 %, 0.3 %); 200 hand_padded (GPU) -5.1 % (published 0.0139, reference now 0.01319; spreads 0.4 %, 0.3 %); 200 fkl (GPU) -5.0 % (published 0.01404, reference now 0.01334; spreads 0.3 %, 0.2 %); 300 hand (GPU) -5.4 % (published 0.01761, reference now 0.01667; spreads 0.1 %, 0.2 %); 300 hand_generic (GPU) -5.1 % (published 0.01814, reference now 0.01722; spreads 0.4 %, 0.5 %); 300 hand_padded (GPU) -4.9 % (published 0.01904, reference now 0.01811; spreads 0.7 %, 0.4 %); 300 fkl (GPU) -5.1 % (published 0.0198, reference now 0.01879; spreads 0.9 %, 0.6 %). Cause: drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated
- **E1c, cleanup**, 1 number: 10 fkl (GPU) +4.7 % (reference now 0.003544, cleaned 0.003711; spreads 0.7 %, 2.1 %). Cause: the dispatch part of a 3.7 us interval: fkl's kernel-only time is 3.08 us, as hand's (3.08 us), so the change (0.17 us) is outside the kernel; what precedes the pass differs (hand_padded's kernel-only pass); the extra 10 ms-spin run gave 3.49 us; not isolated
- **E2 FKL program, drift**, 1 number: 10 warm (GPU) -3.8 % (published 0.02123, reference now 0.02042; spreads 3.8 %, 3.8 %). Cause: at the edge: the change (-3.8 %) equals the spreads of the published and the reference runs (3.8 % each); reference and cleaned agree (0.02042, 0.02037 ms)
- **E2 FKL program, cleanup**, 1 number: 5002 L2-flushed (GPU) -3.0 % (reference now 0.5398, cleaned 0.5234; spreads 0.5 %, 0.0 %). Cause: order of the passes: in the cleaned build the flushed pass follows the kernel-only pass of the same k (100 spins of 5 ms), in the reference the warm pass; the cleaned flushed time equals its kernel-only time (0.5234 against 0.5217 ms); probable GPU clock or power state, as E1a; not isolated
- **E2 Python driver, change**, 7 numbers: 2 inductor_graphs warm (host) +5.7 % (published 0.05757, cleaned 0.06087; spreads 2.0 %, 2.1 %); 2 inductor_graphs host_launch_us (host) +6.2 % (published 39.54, cleaned 42; spreads 2.3 %, 2.1 %); 10 inductor_wide host_launch_us (host) +9.8 % (published 25.01, cleaned 27.47; spreads 0.4 %, 1.8 %); 50 inductor_wide host_launch_us (host) +6.7 % (published 24.9, cleaned 26.57; spreads 3.4 %, 5.3 %); 100 triton_hand host_launch_us (host) +7.3 % (published 9.145, cleaned 9.816; spreads 2.3 %, 5.0 %); 100 inductor_wide host_launch_us (host) +5.0 % (published 24.6, cleaned 25.83; spreads 3.8 %, 2.9 %); 100 inductor_graphs host_launch_us (host) +6.9 % (published 40.02, cleaned 42.78; spreads 4.1 %, 3.2 %). Cause: drift: the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them in the same process); window C pinned the driver to cores 0-3 and 16-19 (four cores and their SMT siblings), the published runs' pinning is not recorded; not isolated
- **E2 Python driver, change**, 4 numbers: 100 eager warm (GPU) -3.2 % (published 2.921, cleaned 2.827; spreads 0.4 %, 0.5 %); 100 eager L2-flushed (GPU) -3.0 % (published 2.875, cleaned 2.788; spreads 0.3 %, 0.0 %); 500 eager warm (GPU) -3.7 % (published 14.3, cleaned 13.76; spreads 0.5 %, 0.0 %); 500 eager L2-flushed (GPU) -3.7 % (published 14.27, cleaned 13.74; spreads 0.9 %, 0.0 %). Cause: drift: the code of these numbers is unchanged since the published runs (as above); eager enqueues one PyTorch kernel per operation; not isolated

## The paper's E1 ratios that move by more than 0.02 (published against cleaned)

- E1a 5002, FKL over hand_params: published 1.026, reference now 1.037, cleaned 0.994
- E1a 10002, FKL over hand_params: published 1.018, reference now 1.045, cleaned 0.989
- E1b 1, FKL over hand: published 0.962, reference now 0.980, cleaned 0.999
- E1c 10, FKL over hand_padded: published 0.932, reference now 0.974, cleaned 1.040
- E1c 10, FKL over hand: published 1.015, reference now 1.018, cleaned 1.062
- E1c 50, FKL over hand_padded: published 1.005, reference now 1.010, cleaned 1.031
- E1c 100, FKL over hand_padded: published 1.020, reference now 1.017, cleaned 0.997
- E1c 100, FKL over hand: published 1.081, reference now 1.079, cleaned 1.053

## Kernel-only: the 10 ms spin against the 5 ms spin, beyond the 5 ms spread and above 3 %

- E1b: kernel-only (cleaned build; a GPU spin before the start event), 5 ms spin (three runs) and 10 ms spin (one run, E1_SPIN_US=10000) / 1 hand_generic: 10 ms spin +16.5 % against the 5 ms spin (spread 4.2 %)
- E1c: kernel-only (cleaned build; a GPU spin before the start event), 5 ms spin (three runs) and 10 ms spin (one run, E1_SPIN_US=10000) / 100 hand_generic: 10 ms spin +4.5 % against the 5 ms spin (spread 0.4 %)

Bound: host when the kernel-only time of the cleaned build is below 80 % of its normal time (more than a fifth of the interval is the dispatch of the
launch), GPU otherwise; the L2-flushed passes are GPU (the flush keeps the GPU busy while the launch is enqueued); host_launch_us is host.

### E1a (e1_vf_chain), GPU time per launch (ms; CUDA events around one launch, mean of 100; median of the three runs)

| ops, version | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| 2 hand_params | GPU | 0.02056 (4.3 %) | 0.02062 (4.3 %) | 0.02052 (0.2 %) | +0.3 % | within | -0.5 % | within |  |
| 2 hand_imm | GPU | 0.02046 (4.1 %) | 0.02045 (4.9 %) | 0.0204 (0.6 %) | -0.1 % | within | -0.2 % | within |  |
| 2 fkl | GPU | 0.02023 (4.2 %) | 0.02017 (5.1 %) | 0.02027 (0.5 %) | -0.3 % | within | +0.5 % | within |  |
| 102 hand_params | GPU | 0.02678 (4.6 %) | 0.02672 (4.9 %) | 0.02674 (0.2 %) | -0.2 % | within | +0.1 % | within |  |
| 102 hand_imm | GPU | 0.02679 (4.5 %) | 0.02655 (5.1 %) | 0.02654 (0.6 %) | -0.9 % | within | -0.0 % | within |  |
| 102 fkl | GPU | 0.02573 (4.7 %) | 0.02566 (5.1 %) | 0.02564 (0.2 %) | -0.3 % | within | -0.1 % | within |  |
| 1002 hand_params | GPU | 0.117 (4.5 %) | 0.1165 (4.9 %) | 0.1165 (0.1 %) | -0.4 % | within | -0.0 % | within |  |
| 1002 hand_imm | GPU | 0.1168 (4.4 %) | 0.1163 (4.8 %) | 0.1163 (0.1 %) | -0.4 % | within | +0.0 % | within |  |
| 1002 fkl | GPU | 0.1174 (3.5 %) | 0.1163 (4.8 %) | 0.115 (0.1 %) | -0.9 % | within | -1.1 % | within |  |
| 5002 hand_params | GPU | 0.534 (2.9 %) | 0.5314 (4.4 %) | 0.5297 (0.5 %) | -0.5 % | within | -0.3 % | within |  |
| 5002 hand_imm | GPU | 0.5524 (1.1 %) | 0.5521 (0.6 %) | 0.5254 (1.6 %) | -0.1 % | within | -4.8 % | exceeds FLAG | order of the passes: in the cleaned build every variant's normal pass follows the previous variant's kernel-only pass (100 one-thread spins of 5 ms, about 0.5 s), in the reference the previous variant's normal pass; at 5,002 and 10,002 operations the reference's hand_imm and fkl (timed second and third) are 3-5 % slower than hand_params (timed first), as in the published runs, and the cleaned build's three variants agree within 0.8 %; probable GPU clock or power state, not isolated |
| 5002 fkl | GPU | 0.5511 (0.8 %) | 0.548 (0.7 %) | 0.5275 (0.9 %) | -0.5 % | within | -3.8 % | exceeds FLAG | order of the passes: in the cleaned build every variant's normal pass follows the previous variant's kernel-only pass (100 one-thread spins of 5 ms, about 0.5 s), in the reference the previous variant's normal pass; at 5,002 and 10,002 operations the reference's hand_imm and fkl (timed second and third) are 3-5 % slower than hand_params (timed first), as in the published runs, and the cleaned build's three variants agree within 0.8 %; probable GPU clock or power state, not isolated |
| 10002 hand_params | GPU | 1.073 (1.1 %) | 1.047 (3.9 %) | 1.078 (1.8 %) | -2.5 % | within | +3.0 % | within |  |
| 10002 hand_imm | GPU | 1.097 (0.3 %) | 1.095 (0.2 %) | 1.073 (1.4 %) | -0.1 % | within | -2.0 % | exceeds |  |
| 10002 fkl | GPU | 1.093 (0.2 %) | 1.093 (0.0 %) | 1.078 (3.8 %) | +0.0 % | within | -1.4 % | within |  |
| 19902 hand_params | GPU | 2.206 (0.8 %) | 2.201 (1.6 %) | 2.195 (0.8 %) | -0.2 % | within | -0.3 % | within |  |
| 19902 hand_imm | GPU | 2.226 (0.4 %) | 2.223 (0.2 %) | 2.211 (0.4 %) | -0.1 % | within | -0.5 % | exceeds |  |
| 19902 fkl | GPU | 2.225 (0.5 %) | 2.212 (0.7 %) | 2.194 (1.5 %) | -0.6 % | within | -0.8 % | within |  |

### E1a: kernel-only (cleaned build; a GPU spin before the start event), 5 ms spin (three runs) and 10 ms spin (one run, E1_SPIN_US=10000)

| ops, version | normal (spread) | kernel-only, 5 ms spin (spread) | kernel-only, 10 ms spin (one run) | 10 ms vs 5 ms | | kernel-only / normal | normal in the 10 ms run |
|---|---|---|---|---|---|---|---|
| 2 hand_params | 0.02052 (0.2 %) | 0.0205 (0.1 %) | 0.02049 | -0.1 % | within the 5 ms spread | 0.999 | 0.02049 |
| 2 hand_imm | 0.0204 (0.6 %) | 0.02049 (0.0 %) | 0.0205 | +0.1 % | exceeds the 5 ms spread | 1 | 0.02036 |
| 2 fkl | 0.02027 (0.5 %) | 0.02048 (0.0 %) | 0.02048 | +0.0 % | within the 5 ms spread | 1.01 | 0.02041 |
| 102 hand_params | 0.02674 (0.2 %) | 0.02736 (0.5 %) | 0.02731 | -0.2 % | within the 5 ms spread | 1.02 | 0.02668 |
| 102 hand_imm | 0.02654 (0.6 %) | 0.0269 (0.3 %) | 0.02729 | +1.5 % | exceeds the 5 ms spread | 1.01 | 0.02646 |
| 102 fkl | 0.02564 (0.2 %) | 0.02631 (0.4 %) | 0.02662 | +1.2 % | exceeds the 5 ms spread | 1.03 | 0.0259 |
| 1002 hand_params | 0.1165 (0.1 %) | 0.1173 (0.2 %) | 0.117 | -0.2 % | exceeds the 5 ms spread | 1.01 | 0.1163 |
| 1002 hand_imm | 0.1163 (0.1 %) | 0.1169 (0.1 %) | 0.117 | +0.1 % | exceeds the 5 ms spread | 1 | 0.1162 |
| 1002 fkl | 0.115 (0.1 %) | 0.1158 (0.0 %) | 0.1159 | +0.1 % | exceeds the 5 ms spread | 1.01 | 0.1152 |
| 5002 hand_params | 0.5297 (0.5 %) | 0.5236 (0.2 %) | 0.522 | -0.3 % | exceeds the 5 ms spread | 0.988 | 0.5284 |
| 5002 hand_imm | 0.5254 (1.6 %) | 0.5237 (0.3 %) | 0.522 | -0.3 % | exceeds the 5 ms spread | 0.997 | 0.5219 |
| 5002 fkl | 0.5275 (0.9 %) | 0.5225 (0.4 %) | 0.5212 | -0.2 % | within the 5 ms spread | 0.99 | 0.5249 |
| 10002 hand_params | 1.078 (1.8 %) | 1.032 (0.4 %) | 1.029 | -0.3 % | within the 5 ms spread | 0.958 | 1.043 |
| 10002 hand_imm | 1.073 (1.4 %) | 1.033 (0.1 %) | 1.029 | -0.4 % | exceeds the 5 ms spread | 0.962 | 1.072 |
| 10002 fkl | 1.078 (3.8 %) | 1.031 (0.2 %) | 1.028 | -0.4 % | exceeds the 5 ms spread | 0.956 | 1.081 |
| 19902 hand_params | 2.195 (0.8 %) | 2.108 (0.4 %) | 2.099 | -0.4 % | exceeds the 5 ms spread | 0.96 | 2.213 |
| 19902 hand_imm | 2.211 (0.4 %) | 2.111 (0.3 %) | 2.102 | -0.5 % | exceeds the 5 ms spread | 0.955 | 2.2 |
| 19902 fkl | 2.194 (1.5 %) | 2.106 (0.2 %) | 2.098 | -0.4 % | exceeds the 5 ms spread | 0.96 | 2.172 |

### E1a: preparation of one launch (cleaned build; host time, outside the GPU interval; median of the three runs)

| ops, version | prep mean ms (spread) | prep min ms (smallest of the three) | total mean ms (GPU + prep) | prep / GPU |
|---|---|---|---|---|
| 2 hand_params | 3.75e-05 (2.9 %) | 2e-05 | 0.02056 | 0.00183 |
| 2 hand_imm | 3.839e-05 (2.1 %) | 2e-05 | 0.02044 | 0.00188 |
| 2 fkl | 3.79e-05 (1.6 %) | 2e-05 | 0.02031 | 0.00187 |
| 102 hand_params | 3.78e-05 (6.1 %) | 2e-05 | 0.02677 | 0.00141 |
| 102 hand_imm | 3.83e-05 (1.3 %) | 2e-05 | 0.02658 | 0.00144 |
| 102 fkl | 3.84e-05 (1.3 %) | 2e-05 | 0.02568 | 0.0015 |
| 1002 hand_params | 3.75e-05 (2.1 %) | 2e-05 | 0.1165 | 0.000322 |
| 1002 hand_imm | 3.78e-05 (1.9 %) | 2e-05 | 0.1164 | 0.000325 |
| 1002 fkl | 3.85e-05 (1.0 %) | 2e-05 | 0.115 | 0.000335 |
| 5002 hand_params | 3.77e-05 (0.8 %) | 2e-05 | 0.5297 | 7.12e-05 |
| 5002 hand_imm | 3.83e-05 (1.3 %) | 2e-05 | 0.5255 | 7.29e-05 |
| 5002 fkl | 3.88e-05 (1.5 %) | 2e-05 | 0.5275 | 7.36e-05 |
| 10002 hand_params | 3.81e-05 (3.1 %) | 2e-05 | 1.078 | 3.53e-05 |
| 10002 hand_imm | 3.91e-05 (3.1 %) | 2e-05 | 1.073 | 3.64e-05 |
| 10002 fkl | 3.86e-05 (1.3 %) | 2e-05 | 1.078 | 3.58e-05 |
| 19902 hand_params | 3.8e-05 (2.6 %) | 2e-05 | 2.195 | 1.73e-05 |
| 19902 hand_imm | 3.81e-05 (1.0 %) | 2e-05 | 2.211 | 1.72e-05 |
| 19902 fkl | 3.869e-05 (1.3 %) | 2e-05 | 2.194 | 1.76e-05 |

### E1b (e1_hf_batch), GPU time per launch (ms; CUDA events around one launch, mean of 100; median of the three runs)

| batch, version | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| 1 hand | host | 0.002979 (48.1 %) | 0.002987 (4.8 %) | 0.002933 (4.4 %) | +0.3 % | within | -1.8 % | within |  |
| 1 hand_generic | host | 0.002918 (11.2 %) | 0.002958 (1.2 %) | 0.002933 (2.7 %) | +1.4 % | within | -0.8 % | within |  |
| 1 fkl | host | 0.002941 (7.6 %) | 0.002985 (5.4 %) | 0.00293 (2.7 %) | +1.5 % | within | -1.8 % | within |  |
| 1 iterated | host | 0.002858 (5.2 %) | 0.002867 (0.7 %) | 0.003172 (0.7 %) | +0.3 % | within | +10.6 % | exceeds FLAG | what precedes the pass: in the cleaned build fkl's kernel-only pass runs between fkl's and iterated's timing; the launch code inside the interval is the same in both builds (one executeOperations per crop); the extra run with the 10 ms spin gave 0.002819 ms, as the reference (0.002867); a 3 us host-bound number; not isolated |
| 51 hand | GPU | 0.004188 (5.5 %) | 0.004108 (1.9 %) | 0.004194 (3.3 %) | -1.9 % | within | +2.1 % | within |  |
| 51 hand_generic | GPU | 0.004194 (6.1 %) | 0.004208 (1.7 %) | 0.004194 (1.1 %) | +0.3 % | within | -0.3 % | within |  |
| 51 fkl | GPU | 0.004204 (3.1 %) | 0.004228 (3.1 %) | 0.004252 (2.5 %) | +0.6 % | within | +0.6 % | within |  |
| 51 iterated | GPU | 0.08632 (0.1 %) | 0.08645 (0.7 %) | 0.08654 (1.9 %) | +0.1 % | within | +0.1 % | within |  |
| 201 hand | GPU | 0.007333 (3.9 %) | 0.007286 (0.0 %) | 0.00734 (1.1 %) | -0.6 % | within | +0.7 % | within |  |
| 201 hand_generic | GPU | 0.007943 (4.0 %) | 0.007883 (0.2 %) | 0.007928 (1.3 %) | -0.8 % | within | +0.6 % | within |  |
| 201 fkl | GPU | 0.007845 (4.5 %) | 0.007778 (1.4 %) | 0.007772 (1.8 %) | -0.9 % | within | -0.1 % | within |  |
| 201 iterated | GPU | 0.3318 (0.5 %) | 0.3294 (0.8 %) | 0.3274 (0.3 %) | -0.7 % | within | -0.6 % | within |  |
| 601 hand | GPU | 0.01577 (4.7 %) | 0.01556 (0.5 %) | 0.01562 (0.6 %) | -1.4 % | within | +0.4 % | within |  |
| 601 hand_generic | GPU | 0.01785 (4.9 %) | 0.01775 (1.4 %) | 0.01774 (0.5 %) | -0.6 % | within | -0.1 % | within |  |
| 601 fkl | GPU | 0.01744 (4.9 %) | 0.0174 (0.3 %) | 0.01737 (0.3 %) | -0.3 % | within | -0.2 % | within |  |
| 601 iterated | GPU | 0.9918 (0.3 %) | 0.9786 (2.1 %) | 0.9711 (1.0 %) | -1.3 % | within | -0.8 % | within |  |
| 1191 hand | GPU | 0.02818 (0.2 %) | 0.02818 (0.4 %) | 0.02812 (0.1 %) | -0.0 % | within | -0.2 % | within |  |
| 1191 hand_generic | GPU | 0.03296 (0.3 %) | 0.03271 (0.4 %) | 0.03276 (0.2 %) | -0.7 % | exceeds | +0.1 % | within |  |
| 1191 fkl | GPU | 0.03202 (0.5 %) | 0.03179 (0.5 %) | 0.03209 (0.5 %) | -0.7 % | exceeds | +0.9 % | exceeds |  |
| 1191 iterated | GPU | 1.954 (0.5 %) | 1.947 (1.3 %) | 1.929 (1.4 %) | -0.4 % | within | -0.9 % | within |  |

### E1b: kernel-only (cleaned build; a GPU spin before the start event), 5 ms spin (three runs) and 10 ms spin (one run, E1_SPIN_US=10000)

| batch, version | normal (spread) | kernel-only, 5 ms spin (spread) | kernel-only, 10 ms spin (one run) | 10 ms vs 5 ms | | kernel-only / normal | normal in the 10 ms run |
|---|---|---|---|---|---|---|---|
| 1 hand | 0.002933 (4.4 %) | 0.002255 (2.0 %) | 0.002269 | +0.7 % | within the 5 ms spread | 0.769 | 0.002962 |
| 1 hand_generic | 0.002933 (2.7 %) | 0.002256 (4.2 %) | 0.002629 | +16.5 % | exceeds the 5 ms spread | 0.769 | 0.002969 |
| 1 fkl | 0.00293 (2.7 %) | 0.002243 (1.8 %) | 0.002253 | +0.5 % | within the 5 ms spread | 0.765 | 0.003164 |
| 1 iterated | 0.003172 (0.7 %) | 0.002503 (24.0 %) | 0.002767 | +10.5 % | within the 5 ms spread | 0.789 | 0.002819 |
| 51 hand | 0.004194 (3.3 %) | 0.004086 (0.0 %) | 0.004086 | +0.0 % | within the 5 ms spread | 0.974 | 0.0041 |
| 51 hand_generic | 0.004194 (1.1 %) | 0.004096 (0.0 %) | 0.004096 | +0.0 % | within the 5 ms spread | 0.977 | 0.004189 |
| 51 fkl | 0.004252 (2.5 %) | 0.004096 (0.0 %) | 0.004097 | +0.0 % | exceeds the 5 ms spread | 0.963 | 0.004267 |
| 51 iterated | 0.08654 (1.9 %) | 0.08183 (0.3 %) | 0.0818 | -0.0 % | within the 5 ms spread | 0.946 | 0.08766 |
| 201 hand | 0.00734 (1.1 %) | 0.007169 (0.0 %) | 0.007168 | -0.0 % | within the 5 ms spread | 0.977 | 0.007304 |
| 201 hand_generic | 0.007928 (1.3 %) | 0.007219 (0.4 %) | 0.007238 | +0.3 % | within the 5 ms spread | 0.911 | 0.007791 |
| 201 fkl | 0.007772 (1.8 %) | 0.007186 (0.3 %) | 0.0072 | +0.2 % | within the 5 ms spread | 0.924 | 0.0077 |
| 201 iterated | 0.3274 (0.3 %) | 0.3196 (0.2 %) | 0.3191 | -0.2 % | within the 5 ms spread | 0.976 | 0.3318 |
| 601 hand | 0.01562 (0.6 %) | 0.01537 (0.1 %) | 0.01537 | -0.0 % | within the 5 ms spread | 0.984 | 0.01563 |
| 601 hand_generic | 0.01774 (0.5 %) | 0.01712 (1.7 %) | 0.01732 | +1.2 % | within the 5 ms spread | 0.965 | 0.01773 |
| 601 fkl | 0.01737 (0.3 %) | 0.01639 (0.0 %) | 0.01639 | -0.0 % | within the 5 ms spread | 0.943 | 0.01733 |
| 601 iterated | 0.9711 (1.0 %) | 0.9543 (0.0 %) | 0.9535 | -0.1 % | exceeds the 5 ms spread | 0.983 | 0.9854 |
| 1191 hand | 0.02812 (0.1 %) | 0.02789 (0.3 %) | 0.0279 | +0.0 % | within the 5 ms spread | 0.992 | 0.02818 |
| 1191 hand_generic | 0.03276 (0.2 %) | 0.03098 (0.3 %) | 0.03096 | -0.1 % | within the 5 ms spread | 0.946 | 0.03288 |
| 1191 fkl | 0.03209 (0.5 %) | 0.0297 (0.0 %) | 0.0297 | -0.0 % | exceeds the 5 ms spread | 0.926 | 0.0321 |
| 1191 iterated | 1.929 (1.4 %) | 1.882 (0.4 %) | 1.881 | -0.1 % | within the 5 ms spread | 0.976 | 1.944 |

### E1b: preparation of one launch (cleaned build; host time, outside the GPU interval; median of the three runs)

| batch, version | prep mean ms (spread) | prep min ms (smallest of the three) | total mean ms (GPU + prep) | prep / GPU |
|---|---|---|---|---|
| 1 hand | 3.81e-05 (1.3 %) | 2e-05 | 0.002972 | 0.013 |
| 1 hand_generic | 3.79e-05 (1.8 %) | 2e-05 | 0.002971 | 0.0129 |
| 1 fkl | 7.54e-05 (2.1 %) | 6e-05 | 0.003004 | 0.0257 |
| 1 iterated | 7.23e-05 (2.5 %) | 6e-05 | 0.003243 | 0.0228 |
| 51 hand | 4.759e-05 (2.9 %) | 3e-05 | 0.004242 | 0.0113 |
| 51 hand_generic | 6.63e-05 (2.6 %) | 5e-05 | 0.004261 | 0.0158 |
| 51 fkl | 0.0006545 (0.5 %) | 0.00063 | 0.004905 | 0.154 |
| 51 iterated | 0.0006401 (1.4 %) | 0.0006 | 0.08718 | 0.0074 |
| 201 hand | 8.18e-05 (3.9 %) | 7e-05 | 0.007425 | 0.0111 |
| 201 hand_generic | 0.0001569 (0.8 %) | 0.00014 | 0.008084 | 0.0198 |
| 201 fkl | 0.002255 (3.8 %) | 0.00219 | 0.01003 | 0.29 |
| 201 iterated | 0.002246 (2.1 %) | 0.00216 | 0.3296 | 0.00686 |
| 601 hand | 0.0001671 (0.6 %) | 0.00015 | 0.01579 | 0.0107 |
| 601 hand_generic | 0.0003879 (1.2 %) | 0.00037 | 0.01813 | 0.0219 |
| 601 fkl | 0.007659 (3.8 %) | 0.00742 | 0.02503 | 0.441 |
| 601 iterated | 0.007681 (2.9 %) | 0.00746 | 0.9788 | 0.00791 |
| 1191 hand | 0.0002883 (1.2 %) | 0.00027 | 0.02841 | 0.0103 |
| 1191 hand_generic | 0.0007298 (0.5 %) | 0.00071 | 0.03349 | 0.0223 |
| 1191 fkl | 0.01565 (2.3 %) | 0.01524 | 0.0477 | 0.488 |
| 1191 iterated | 0.01557 (1.5 %) | 0.0152 | 1.944 | 0.00807 |

### E1c (e1_preproc), GPU time per launch (ms; CUDA events around one launch, mean of 100; median of the three runs)

| batch, version | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| 10 hand | GPU | 0.003591 (0.5 %) | 0.003477 (2.2 %) | 0.003496 (1.3 %) | -3.2 % | exceeds FLAG | +0.6 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 10 hand_generic | GPU | 0.003824 (2.1 %) | 0.003607 (2.2 %) | 0.00361 (1.3 %) | -5.7 % | exceeds FLAG | +0.1 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 10 hand_padded | GPU | 0.003869 (9.6 %) | 0.003638 (0.4 %) | 0.003576 (0.4 %) | -6.0 % | within FLAG | -1.7 % | exceeds |  |
| 10 fkl | GPU | 0.003644 (2.8 %) | 0.003544 (0.7 %) | 0.003711 (2.1 %) | -2.7 % | within | +4.7 % | exceeds FLAG | the dispatch part of a 3.7 us interval: fkl's kernel-only time is 3.08 us, as hand's (3.08 us), so the change (0.17 us) is outside the kernel; what precedes the pass differs (hand_padded's kernel-only pass); the extra 10 ms-spin run gave 3.49 us; not isolated |
| 50 hand | GPU | 0.005824 (1.4 %) | 0.005581 (0.2 %) | 0.005583 (0.3 %) | -4.2 % | exceeds FLAG | +0.0 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 50 hand_generic | GPU | 0.005873 (1.7 %) | 0.005519 (2.1 %) | 0.005499 (1.2 %) | -6.0 % | exceeds FLAG | -0.4 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 50 hand_padded | GPU | 0.006031 (0.7 %) | 0.005692 (1.2 %) | 0.005626 (0.5 %) | -5.6 % | exceeds FLAG | -1.2 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 50 fkl | GPU | 0.006075 (0.8 %) | 0.00573 (0.5 %) | 0.005821 (6.7 %) | -5.7 % | exceeds FLAG | +1.6 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 100 hand | GPU | 0.008086 (2.1 %) | 0.00782 (0.9 %) | 0.007871 (1.2 %) | -3.3 % | exceeds FLAG | +0.6 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 100 hand_generic | GPU | 0.008189 (0.4 %) | 0.007947 (0.8 %) | 0.008034 (0.6 %) | -3.0 % | exceeds | +1.1 % | exceeds |  |
| 100 hand_padded | GPU | 0.008518 (0.7 %) | 0.008273 (0.5 %) | 0.008281 (1.3 %) | -2.9 % | exceeds | +0.1 % | within |  |
| 100 fkl | GPU | 0.008733 (0.9 %) | 0.00842 (0.8 %) | 0.008256 (0.5 %) | -3.6 % | exceeds FLAG | -2.0 % | exceeds | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 200 hand | GPU | 0.01303 (0.6 %) | 0.01226 (0.7 %) | 0.01221 (0.3 %) | -5.9 % | exceeds FLAG | -0.4 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 200 hand_generic | GPU | 0.01338 (0.5 %) | 0.01263 (0.3 %) | 0.01255 (0.5 %) | -5.6 % | exceeds FLAG | -0.6 % | exceeds | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 200 hand_padded | GPU | 0.0139 (0.4 %) | 0.01319 (0.3 %) | 0.01315 (0.3 %) | -5.1 % | exceeds FLAG | -0.3 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 200 fkl | GPU | 0.01404 (0.3 %) | 0.01334 (0.2 %) | 0.01329 (0.1 %) | -5.0 % | exceeds FLAG | -0.4 % | exceeds | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 300 hand | GPU | 0.01761 (0.1 %) | 0.01667 (0.2 %) | 0.01663 (0.2 %) | -5.4 % | exceeds FLAG | -0.2 % | exceeds | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 300 hand_generic | GPU | 0.01814 (0.4 %) | 0.01722 (0.5 %) | 0.01719 (0.8 %) | -5.1 % | exceeds FLAG | -0.2 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 300 hand_padded | GPU | 0.01904 (0.7 %) | 0.01811 (0.4 %) | 0.0181 (0.4 %) | -4.9 % | exceeds FLAG | -0.1 % | within | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |
| 300 fkl | GPU | 0.0198 (0.9 %) | 0.01879 (0.6 %) | 0.01849 (0.3 %) | -5.1 % | exceeds FLAG | -1.6 % | exceeds | drift of the published set: e1-preproc-v3 (2026-10-30 12:54) is 3-6 % slower than reference now on every version, and as much against e1-preproc-v2 (10:44 the same day; reference now agrees with v2 within 2.3 % on the versions they share); v3's conditions are not recorded (no load or GPU-state file); the ratios between the versions are unchanged (ratio table); not isolated |

### E1c: kernel-only (cleaned build; a GPU spin before the start event), 5 ms spin (three runs) and 10 ms spin (one run, E1_SPIN_US=10000)

| batch, version | normal (spread) | kernel-only, 5 ms spin (spread) | kernel-only, 10 ms spin (one run) | 10 ms vs 5 ms | | kernel-only / normal | normal in the 10 ms run |
|---|---|---|---|---|---|---|---|
| 10 hand | 0.003496 (1.3 %) | 0.003076 (0.2 %) | 0.003078 | +0.1 % | within the 5 ms spread | 0.88 | 0.003404 |
| 10 hand_generic | 0.00361 (1.3 %) | 0.003132 (0.7 %) | 0.003217 | +2.7 % | exceeds the 5 ms spread | 0.867 | 0.003617 |
| 10 hand_padded | 0.003576 (0.4 %) | 0.003092 (0.4 %) | 0.003098 | +0.2 % | within the 5 ms spread | 0.865 | 0.003891 |
| 10 fkl | 0.003711 (2.1 %) | 0.003079 (0.1 %) | 0.003118 | +1.3 % | exceeds the 5 ms spread | 0.83 | 0.003492 |
| 50 hand | 0.005583 (0.3 %) | 0.005136 (0.2 %) | 0.005129 | -0.1 % | within the 5 ms spread | 0.92 | 0.005565 |
| 50 hand_generic | 0.005499 (1.2 %) | 0.005137 (0.2 %) | 0.005186 | +1.0 % | exceeds the 5 ms spread | 0.934 | 0.005505 |
| 50 hand_padded | 0.005626 (0.5 %) | 0.005144 (0.4 %) | 0.00514 | -0.1 % | within the 5 ms spread | 0.914 | 0.005951 |
| 50 fkl | 0.005821 (6.7 %) | 0.005164 (0.2 %) | 0.005158 | -0.1 % | within the 5 ms spread | 0.887 | 0.005787 |
| 100 hand | 0.007871 (1.2 %) | 0.007374 (0.2 %) | 0.007366 | -0.1 % | within the 5 ms spread | 0.937 | 0.00779 |
| 100 hand_generic | 0.008034 (0.6 %) | 0.007518 (0.4 %) | 0.007858 | +4.5 % | exceeds the 5 ms spread | 0.936 | 0.007875 |
| 100 hand_padded | 0.008281 (1.3 %) | 0.00791 (1.0 %) | 0.00793 | +0.3 % | within the 5 ms spread | 0.955 | 0.008612 |
| 100 fkl | 0.008256 (0.5 %) | 0.007916 (0.9 %) | 0.007902 | -0.2 % | within the 5 ms spread | 0.959 | 0.008331 |
| 200 hand | 0.01221 (0.3 %) | 0.01229 (0.0 %) | 0.01229 | -0.0 % | within the 5 ms spread | 1.01 | 0.01218 |
| 200 hand_generic | 0.01255 (0.5 %) | 0.01231 (0.1 %) | 0.0123 | -0.1 % | exceeds the 5 ms spread | 0.981 | 0.01258 |
| 200 hand_padded | 0.01315 (0.3 %) | 0.01281 (0.5 %) | 0.01286 | +0.4 % | within the 5 ms spread | 0.974 | 0.01317 |
| 200 fkl | 0.01329 (0.1 %) | 0.01249 (0.1 %) | 0.01252 | +0.2 % | exceeds the 5 ms spread | 0.94 | 0.01337 |
| 300 hand | 0.01663 (0.2 %) | 0.01671 (0.2 %) | 0.01673 | +0.1 % | within the 5 ms spread | 1.01 | 0.01666 |
| 300 hand_generic | 0.01719 (0.8 %) | 0.01741 (0.1 %) | 0.01741 | +0.0 % | within the 5 ms spread | 1.01 | 0.01725 |
| 300 hand_padded | 0.0181 (0.4 %) | 0.01749 (0.1 %) | 0.01747 | -0.1 % | exceeds the 5 ms spread | 0.966 | 0.01805 |
| 300 fkl | 0.01849 (0.3 %) | 0.01746 (0.2 %) | 0.01746 | -0.0 % | within the 5 ms spread | 0.945 | 0.01848 |

### E1c: preparation of one launch (cleaned build; host time, outside the GPU interval; median of the three runs)

| batch, version | prep mean ms (spread) | prep min ms (smallest of the three) | total mean ms (GPU + prep) | prep / GPU |
|---|---|---|---|---|
| 10 hand | 3.82e-05 (2.9 %) | 3e-05 | 0.003534 | 0.0109 |
| 10 hand_generic | 4.09e-05 (1.7 %) | 3e-05 | 0.003652 | 0.0113 |
| 10 hand_padded | 4.09e-05 (3.9 %) | 3e-05 | 0.003618 | 0.0114 |
| 10 fkl | 0.000349 (1.7 %) | 0.00032 | 0.00406 | 0.094 |
| 50 hand | 4.69e-05 (1.1 %) | 3e-05 | 0.005631 | 0.0084 |
| 50 hand_generic | 6.71e-05 (3.0 %) | 5e-05 | 0.005565 | 0.0122 |
| 50 hand_padded | 6.59e-05 (8.6 %) | 5e-05 | 0.005692 | 0.0117 |
| 50 fkl | 0.001396 (0.6 %) | 0.00135 | 0.007215 | 0.24 |
| 100 hand | 5.75e-05 (2.8 %) | 4e-05 | 0.007929 | 0.00731 |
| 100 hand_generic | 9.52e-05 (2.0 %) | 8e-05 | 0.008129 | 0.0118 |
| 100 hand_padded | 9.4e-05 (1.1 %) | 8e-05 | 0.008375 | 0.0114 |
| 100 fkl | 0.002642 (1.7 %) | 0.0026 | 0.0109 | 0.32 |
| 200 hand | 8.12e-05 (1.6 %) | 7e-05 | 0.01229 | 0.00665 |
| 200 hand_generic | 0.0001593 (6.0 %) | 0.00014 | 0.01271 | 0.0127 |
| 200 hand_padded | 0.0001599 (2.1 %) | 0.00014 | 0.01331 | 0.0122 |
| 200 fkl | 0.005215 (1.0 %) | 0.0051 | 0.01851 | 0.392 |
| 300 hand | 0.0001025 (35.2 %) | 9e-05 | 0.01675 | 0.00617 |
| 300 hand_generic | 0.000211 (19.7 %) | 0.0002 | 0.0174 | 0.0123 |
| 300 hand_padded | 0.000215 (2.9 %) | 0.0002 | 0.01831 | 0.0119 |
| 300 fkl | 0.007919 (2.1 %) | 0.00772 | 0.02644 | 0.428 |

### E1, the ratios of the paper's compact table (scripts/e1_compact.py: FKL's time over the reference's, per run, median of the three runs; smallest and largest of the three in brackets)

vs_ref: E1a hand_params, E1b hand_generic, E1c hand_padded; vs_best: E1a hand_imm, E1b and E1c hand. Kernel-only: the cleaned build's kernel-only times.

| exp, point | vs_ref published | reference now | cleaned | cleaned kernel-only | vs_best published | reference now | cleaned | cleaned kernel-only |
|---|---|---|---|---|---|---|---|---|
| E1a 2 | 0.985 (0.984-0.986) | 0.983 (0.979-0.990) | 0.988 (0.985-0.988) | 0.999 (0.998-1.000) | 0.989 (0.989-0.991) | 0.989 (0.987-0.991) | 0.990 (0.989-0.996) | 1.000 (1.000-1.000) |
| E1a 102 | 0.961 (0.960-0.962) | 0.961 (0.960-0.963) | 0.959 (0.957-0.961) | 0.962 (0.961-0.962) | 0.968 (0.960-0.969) | 0.968 (0.967-0.968) | 0.966 (0.965-0.969) | 0.981 (0.978-0.982) |
| E1a 1002 | 0.996 (0.987-1.004) | 0.988 (0.987-0.998) | 0.987 (0.986-0.988) | 0.987 (0.987-0.989) | 0.997 (0.989-1.005) | 0.989 (0.988-1.000) | 0.988 (0.988-0.989) | 0.991 (0.991-0.991) |
| E1a 5002 | 1.026 (1.012-1.039) | 1.037 (1.000-1.044) | 0.994 (0.992-0.998) | 0.998 (0.997-0.999) | 0.993 (0.988-1.000) | 0.998 (0.991-0.998) | 1.006 (0.984-1.006) | 0.997 (0.996-1.001) |
| E1a 10002 | 1.018 (1.008-1.019) | 1.045 (1.016-1.056) | 0.989 (0.963-1.007) | 0.998 (0.996-0.999) | 0.997 (0.995-0.997) | 0.998 (0.998-1.000) | 0.994 (0.971-1.005) | 0.999 (0.998-1.000) |
| E1a 19902 | 1.012 (1.006-1.012) | 1.008 (1.003-1.020) | 0.995 (0.993-1.005) | 0.998 (0.997-0.999) | 1.000 (0.999-1.000) | 0.996 (0.993-0.999) | 0.996 (0.983-0.997) | 0.998 (0.997-0.999) |
| E1b 1 | 0.982 (0.973-1.032) | 1.018 (0.961-1.020) | 1.000 (0.973-1.026) | 0.980 (0.971-1.008) | 0.962 (0.702-0.991) | 0.980 (0.972-0.999) | 0.999 (0.942-1.011) | 0.999 (0.974-1.007) |
| E1b 51 | 1.002 (0.979-1.008) | 1.005 (0.988-1.036) | 1.006 (1.001-1.029) | 1.000 (1.000-1.000) | 1.002 (0.995-1.021) | 1.035 (1.029-1.047) | 1.014 (1.013-1.021) | 1.003 (1.003-1.003) |
| E1b 201 | 0.990 (0.988-0.995) | 0.988 (0.978-0.991) | 0.974 (0.963-0.986) | 0.996 (0.990-0.996) | 1.075 (1.058-1.080) | 1.068 (1.058-1.074) | 1.059 (1.039-1.070) | 1.002 (1.000-1.002) |
| E1b 601 | 0.980 (0.975-0.983) | 0.979 (0.968-0.984) | 0.979 (0.977-0.983) | 0.958 (0.944-0.960) | 1.109 (1.106-1.111) | 1.117 (1.113-1.121) | 1.113 (1.110-1.114) | 1.066 (1.066-1.067) |
| E1b 1191 | 0.970 (0.967-0.973) | 0.971 (0.968-0.973) | 0.980 (0.977-0.984) | 0.959 (0.956-0.959) | 1.135 (1.131-1.137) | 1.129 (1.124-1.132) | 1.141 (1.140-1.145) | 1.065 (1.062-1.065) |
| E1c 10 | 0.932 (0.889-0.958) | 0.974 (0.971-0.980) | 1.040 (1.028-1.051) | 0.996 (0.994-0.996) | 1.015 (1.008-1.032) | 1.018 (1.015-1.031) | 1.062 (1.058-1.066) | 1.000 (1.000-1.002) |
| E1c 50 | 1.005 (1.005-1.008) | 1.010 (1.000-1.014) | 1.031 (1.027-1.099) | 1.004 (0.999-1.005) | 1.035 (1.034-1.047) | 1.028 (1.024-1.030) | 1.043 (1.037-1.104) | 1.004 (1.004-1.007) |
| E1c 100 | 1.020 (1.019-1.026) | 1.017 (1.014-1.024) | 0.997 (0.983-1.000) | 0.999 (0.997-1.001) | 1.081 (1.064-1.095) | 1.079 (1.075-1.082) | 1.053 (1.048-1.061) | 1.073 (1.069-1.079) |
| E1c 200 | 1.010 (1.007-1.012) | 1.010 (1.008-1.012) | 1.010 (1.010-1.013) | 0.975 (0.972-0.978) | 1.076 (1.075-1.084) | 1.087 (1.086-1.094) | 1.089 (1.085-1.089) | 1.016 (1.016-1.017) |
| E1c 300 | 1.036 (1.035-1.046) | 1.037 (1.035-1.039) | 1.023 (1.020-1.024) | 0.999 (0.999-1.000) | 1.124 (1.120-1.129) | 1.129 (1.122-1.129) | 1.112 (1.110-1.114) | 1.046 (1.045-1.047) |

### E2, the FKL program (e2_fkl_chain), GPU time per launch (ms; mean of 100; published: runs 2, 3, 4)

| k, pass | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |
|---|---|---|---|---|---|---|---|---|---|
| 2 warm | GPU | 0.02115 (5.6 %) | 0.02069 (3.7 %) | 0.02028 (0.4 %) | -2.2 % | within | -2.0 % | within |  |
| 2 L2-flushed | GPU | 0.02656 (5.9 %) | 0.02592 (3.6 %) | 0.02586 (0.3 %) | -2.4 % | within | -0.2 % | within |  |
| 10 warm | GPU | 0.02123 (3.8 %) | 0.02042 (3.8 %) | 0.02037 (2.0 %) | -3.8 % | exceeds FLAG | -0.2 % | within | at the edge: the change (-3.8 %) equals the spreads of the published and the reference runs (3.8 % each); reference and cleaned agree (0.02042, 0.02037 ms) |
| 10 L2-flushed | GPU | 0.02691 (4.3 %) | 0.02626 (3.4 %) | 0.02618 (0.4 %) | -2.4 % | within | -0.3 % | within |  |
| 50 warm | GPU | 0.02285 (4.2 %) | 0.02276 (4.5 %) | 0.02271 (0.3 %) | -0.4 % | within | -0.2 % | within |  |
| 50 L2-flushed | GPU | 0.02798 (3.6 %) | 0.02776 (0.3 %) | 0.02774 (0.2 %) | -0.8 % | within | -0.1 % | within |  |
| 100 warm | GPU | 0.02545 (4.9 %) | 0.02544 (0.1 %) | 0.02548 (2.0 %) | -0.1 % | within | +0.2 % | within |  |
| 100 L2-flushed | GPU | 0.03032 (1.7 %) | 0.02983 (0.6 %) | 0.02992 (0.2 %) | -1.6 % | within | +0.3 % | within |  |
| 500 warm | GPU | 0.06429 (0.3 %) | 0.06425 (0.1 %) | 0.06423 (0.1 %) | -0.1 % | within | -0.0 % | within |  |
| 500 L2-flushed | GPU | 0.06903 (0.6 %) | 0.06897 (0.1 %) | 0.06903 (0.2 %) | -0.1 % | within | +0.1 % | within |  |
| 5002 warm | GPU | 0.5265 (0.7 %) | 0.5289 (1.7 %) | 0.5289 (0.9 %) | +0.4 % | within | -0.0 % | within |  |
| 5002 L2-flushed | GPU | 0.5445 (2.5 %) | 0.5398 (0.5 %) | 0.5234 (0.0 %) | -0.9 % | within | -3.0 % | exceeds FLAG | order of the passes: in the cleaned build the flushed pass follows the kernel-only pass of the same k (100 spins of 5 ms), in the reference the warm pass; the cleaned flushed time equals its kernel-only time (0.5234 against 0.5217 ms); probable GPU clock or power state, as E1a; not isolated |

### E2, the FKL program: kernel-only (cleaned build), 5 ms spin (three runs) and 10 ms spin (one run)

| k | normal (spread) | kernel-only, 5 ms spin (spread) | kernel-only, 10 ms spin (one run) | 10 ms vs 5 ms | | kernel-only / normal | normal in the 10 ms run |
|---|---|---|---|---|---|---|---|
| 2 fkl | 0.02028 (0.4 %) | 0.02048 (0.0 %) | 0.02048 | +0.0 % | within the 5 ms spread | 1.01 | 0.02151 |
| 10 fkl | 0.02037 (2.0 %) | 0.02048 (0.0 %) | 0.02048 | -0.0 % | exceeds the 5 ms spread | 1.01 | 0.02031 |
| 50 fkl | 0.02271 (0.3 %) | 0.02286 (0.2 %) | 0.02286 | -0.0 % | within the 5 ms spread | 1.01 | 0.02264 |
| 100 fkl | 0.02548 (2.0 %) | 0.02571 (0.1 %) | 0.02569 | -0.1 % | within the 5 ms spread | 1.01 | 0.02547 |
| 500 fkl | 0.06423 (0.1 %) | 0.06525 (0.2 %) | 0.06526 | +0.0 % | within the 5 ms spread | 1.02 | 0.0642 |
| 5002 fkl | 0.5289 (0.9 %) | 0.5217 (0.1 %) | 0.5207 | -0.2 % | exceeds the 5 ms spread | 0.986 | 0.5281 |

### E2, the FKL program: preparation (cleaned build; host time of building the operations of one launch, outside the GPU interval)

| k | warm pass prep mean ms (spread) | flushed pass prep mean ms (spread) |
|---|---|---|
| 2 | 3.92e-05 (5.1 %) | 3.88e-05 (3.4 %) |
| 10 | 3.89e-05 (5.4 %) | 3.86e-05 (2.6 %) |
| 50 | 3.83e-05 (1.8 %) | 3.88e-05 (1.3 %) |
| 100 | 3.92e-05 (3.8 %) | 3.87e-05 (1.6 %) |
| 500 | 3.9e-05 (3.1 %) | 3.87e-05 (2.1 %) |
| 5002 | 4.02e-05 (3.0 %) | 3.87e-05 (2.1 %) |

### E2, the Python driver (triton/e2_chain.py; cleaned script only, no reference run: the change is drift and cleanup together; published: runs 2, 3, 4)

| k, version, column | bound | published (spread) | cleaned (spread) | change cleaned vs pub | | cause |
|---|---|---|---|---|---|---|
| 2 triton_hand warm | host | 0.0287 (4.9 %) | 0.02921 (4.1 %) | +1.8 % | within |  |
| 2 triton_hand L2-flushed | GPU | 0.01935 (1.6 %) | 0.01944 (0.5 %) | +0.5 % | within |  |
| 2 triton_hand host_launch_us | host | 9.161 (7.1 %) | 9.458 (6.4 %) | +3.2 % | within |  |
| 2 inductor_default warm | host | 0.04541 (1.1 %) | 0.04692 (2.2 %) | +3.3 % | exceeds |  |
| 2 inductor_default L2-flushed | GPU | 0.01901 (0.6 %) | 0.01926 (1.0 %) | +1.3 % | exceeds |  |
| 2 inductor_default host_launch_us | host | 25.26 (4.8 %) | 26.57 (6.6 %) | +5.2 % | within FLAG |  |
| 2 inductor_wide warm | host | 0.04552 (2.7 %) | 0.04748 (0.5 %) | +4.3 % | exceeds |  |
| 2 inductor_wide L2-flushed | GPU | 0.01925 (1.6 %) | 0.01914 (1.8 %) | -0.6 % | within |  |
| 2 inductor_wide host_launch_us | host | 25.59 (4.2 %) | 26.38 (4.5 %) | +3.1 % | within |  |
| 2 inductor_graphs warm | host | 0.05757 (2.0 %) | 0.06087 (2.1 %) | +5.7 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them in the same process); window C pinned the driver to cores 0-3 and 16-19 (four cores and their SMT siblings), the published runs' pinning is not recorded; not isolated |
| 2 inductor_graphs L2-flushed | GPU | 0.01868 (0.2 %) | 0.01907 (1.3 %) | +2.1 % | exceeds |  |
| 2 inductor_graphs host_launch_us | host | 39.54 (2.3 %) | 42 (2.1 %) | +6.2 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them in the same process); window C pinned the driver to cores 0-3 and 16-19 (four cores and their SMT siblings), the published runs' pinning is not recorded; not isolated |
| 2 eager warm | GPU | 0.1331 (0.3 %) | 0.1297 (0.5 %) | -2.6 % | exceeds |  |
| 2 eager L2-flushed | GPU | 0.1104 (0.2 %) | 0.1085 (0.5 %) | -1.8 % | exceeds |  |
| 2 eager host_launch_us | host | 20.72 (0.2 %) | 20.29 (1.0 %) | -2.1 % | exceeds |  |
| 2 eager_graphs warm | GPU | 0.1192 (0.7 %) | 0.1197 (0.2 %) | +0.4 % | within |  |
| 2 eager_graphs L2-flushed | GPU | 0.1049 (0.3 %) | 0.1042 (0.5 %) | -0.7 % | exceeds |  |
| 2 eager_graphs host_launch_us | host | 2.286 (4.2 %) | 2.243 (0.7 %) | -1.9 % | within |  |
| 10 triton_hand warm | host | 0.02889 (8.5 %) | 0.02943 (1.5 %) | +1.9 % | within |  |
| 10 triton_hand L2-flushed | GPU | 0.01952 (1.5 %) | 0.01949 (0.3 %) | -0.2 % | within |  |
| 10 triton_hand host_launch_us | host | 9.092 (61.9 %) | 9.618 (2.9 %) | +5.8 % | within FLAG |  |
| 10 inductor_default warm | host | 0.04467 (2.2 %) | 0.04617 (2.6 %) | +3.4 % | exceeds |  |
| 10 inductor_default L2-flushed | GPU | 0.0192 (1.2 %) | 0.01935 (1.8 %) | +0.8 % | within |  |
| 10 inductor_default host_launch_us | host | 24.87 (5.6 %) | 25.63 (6.2 %) | +3.1 % | within |  |
| 10 inductor_wide warm | host | 0.04515 (0.2 %) | 0.04684 (0.1 %) | +3.7 % | exceeds |  |
| 10 inductor_wide L2-flushed | GPU | 0.01931 (2.1 %) | 0.01947 (1.6 %) | +0.8 % | within |  |
| 10 inductor_wide host_launch_us | host | 25.01 (0.4 %) | 27.47 (1.8 %) | +9.8 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them in the same process); window C pinned the driver to cores 0-3 and 16-19 (four cores and their SMT siblings), the published runs' pinning is not recorded; not isolated |
| 10 inductor_graphs warm | host | 0.05768 (1.0 %) | 0.05914 (2.0 %) | +2.5 % | exceeds |  |
| 10 inductor_graphs L2-flushed | GPU | 0.01895 (4.3 %) | 0.019 (0.7 %) | +0.3 % | within |  |
| 10 inductor_graphs host_launch_us | host | 40.24 (1.7 %) | 41.66 (4.7 %) | +3.5 % | within |  |
| 10 eager warm | GPU | 0.3621 (0.3 %) | 0.3522 (0.3 %) | -2.7 % | exceeds |  |
| 10 eager L2-flushed | GPU | 0.3388 (1.1 %) | 0.3298 (0.1 %) | -2.6 % | exceeds |  |
| 10 eager host_launch_us | host | 56.8 (1.9 %) | 55.93 (1.5 %) | -1.5 % | within |  |
| 10 eager_graphs warm | GPU | 0.3223 (0.9 %) | 0.3196 (0.2 %) | -0.8 % | within |  |
| 10 eager_graphs L2-flushed | GPU | 0.3162 (0.2 %) | 0.3134 (0.7 %) | -0.9 % | exceeds |  |
| 10 eager_graphs host_launch_us | host | 2.34 (4.1 %) | 2.262 (1.6 %) | -3.3 % | within |  |
| 50 triton_hand warm | host | 0.02903 (2.6 %) | 0.02956 (2.1 %) | +1.8 % | within |  |
| 50 triton_hand L2-flushed | GPU | 0.01981 (2.5 %) | 0.01965 (0.3 %) | -0.8 % | within |  |
| 50 triton_hand host_launch_us | host | 9.234 (1.8 %) | 9.527 (2.7 %) | +3.2 % | exceeds |  |
| 50 inductor_default warm | host | 0.0455 (1.3 %) | 0.04627 (4.2 %) | +1.7 % | within |  |
| 50 inductor_default L2-flushed | GPU | 0.01966 (0.9 %) | 0.01979 (0.9 %) | +0.7 % | within |  |
| 50 inductor_default host_launch_us | host | 24.95 (2.5 %) | 25.51 (5.4 %) | +2.2 % | within |  |
| 50 inductor_wide warm | host | 0.04591 (0.7 %) | 0.04699 (2.6 %) | +2.4 % | within |  |
| 50 inductor_wide L2-flushed | GPU | 0.01965 (0.8 %) | 0.01977 (1.0 %) | +0.7 % | within |  |
| 50 inductor_wide host_launch_us | host | 24.9 (3.4 %) | 26.57 (5.3 %) | +6.7 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them in the same process); window C pinned the driver to cores 0-3 and 16-19 (four cores and their SMT siblings), the published runs' pinning is not recorded; not isolated |
| 50 inductor_graphs warm | host | 0.05798 (3.2 %) | 0.05939 (2.3 %) | +2.4 % | within |  |
| 50 inductor_graphs L2-flushed | GPU | 0.01936 (5.3 %) | 0.01929 (1.7 %) | -0.3 % | within |  |
| 50 inductor_graphs host_launch_us | host | 39.38 (1.3 %) | 41.44 (5.4 %) | +5.2 % | within FLAG |  |
| 50 eager warm | GPU | 1.502 (0.5 %) | 1.461 (0.4 %) | -2.7 % | exceeds |  |
| 50 eager L2-flushed | GPU | 1.461 (0.0 %) | 1.42 (0.1 %) | -2.8 % | exceeds |  |
| 50 eager host_launch_us | host | 1,201 (2.0 %) | 1,154 (1.0 %) | -4.0 % | exceeds |  |
| 50 eager_graphs warm | GPU | 1.384 (0.2 %) | 1.37 (0.4 %) | -1.0 % | exceeds |  |
| 50 eager_graphs L2-flushed | GPU | 1.351 (0.0 %) | 1.338 (0.1 %) | -0.9 % | exceeds |  |
| 50 eager_graphs host_launch_us | host | 2.339 (2.6 %) | 2.336 (4.0 %) | -0.1 % | within |  |
| 100 triton_hand warm | host | 0.02942 (2.1 %) | 0.02982 (2.8 %) | +1.3 % | within |  |
| 100 triton_hand L2-flushed | GPU | 0.02007 (0.6 %) | 0.0199 (0.9 %) | -0.8 % | within |  |
| 100 triton_hand host_launch_us | host | 9.145 (2.3 %) | 9.816 (5.0 %) | +7.3 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them in the same process); window C pinned the driver to cores 0-3 and 16-19 (four cores and their SMT siblings), the published runs' pinning is not recorded; not isolated |
| 100 inductor_default warm | host | 0.04697 (10.1 %) | 0.04814 (24.6 %) | +2.5 % | within |  |
| 100 inductor_default L2-flushed | GPU | 0.02304 (0.4 %) | 0.02299 (1.8 %) | -0.2 % | within |  |
| 100 inductor_default host_launch_us | host | 23.89 (4.3 %) | 25.44 (55.4 %) | +6.5 % | within FLAG |  |
| 100 inductor_wide warm | host | 0.04685 (1.0 %) | 0.04803 (1.2 %) | +2.5 % | exceeds |  |
| 100 inductor_wide L2-flushed | GPU | 0.02278 (2.5 %) | 0.02314 (0.2 %) | +1.6 % | within |  |
| 100 inductor_wide host_launch_us | host | 24.6 (3.8 %) | 25.83 (2.9 %) | +5.0 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them in the same process); window C pinned the driver to cores 0-3 and 16-19 (four cores and their SMT siblings), the published runs' pinning is not recorded; not isolated |
| 100 inductor_graphs warm | host | 0.0591 (1.4 %) | 0.06308 (35.0 %) | +6.7 % | within FLAG |  |
| 100 inductor_graphs L2-flushed | GPU | 0.02268 (1.4 %) | 0.02279 (0.4 %) | +0.5 % | within |  |
| 100 inductor_graphs host_launch_us | host | 40.02 (4.1 %) | 42.78 (3.2 %) | +6.9 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (the cleaned script adds only the kernel-only pass, after them in the same process); window C pinned the driver to cores 0-3 and 16-19 (four cores and their SMT siblings), the published runs' pinning is not recorded; not isolated |
| 100 eager warm | GPU | 2.921 (0.4 %) | 2.827 (0.5 %) | -3.2 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (as above); eager enqueues one PyTorch kernel per operation; not isolated |
| 100 eager L2-flushed | GPU | 2.875 (0.3 %) | 2.788 (0.0 %) | -3.0 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (as above); eager enqueues one PyTorch kernel per operation; not isolated |
| 100 eager host_launch_us | host | 2,603 (0.3 %) | 2,521 (0.0 %) | -3.2 % | exceeds |  |
| 100 eager_graphs warm | GPU | 2.694 (0.8 %) | 2.666 (0.4 %) | -1.1 % | exceeds |  |
| 100 eager_graphs L2-flushed | GPU | 2.668 (0.5 %) | 2.624 (0.1 %) | -1.7 % | exceeds |  |
| 100 eager_graphs host_launch_us | host | 2.355 (3.6 %) | 2.415 (5.6 %) | +2.6 % | within |  |
| 500 triton_hand warm | GPU | 0.07298 (3.3 %) | 0.07177 (0.3 %) | -1.7 % | within |  |
| 500 triton_hand L2-flushed | GPU | 0.06223 (0.5 %) | 0.06212 (0.2 %) | -0.2 % | within |  |
| 500 triton_hand host_launch_us | host | 9.162 (2.8 %) | 9.17 (5.2 %) | +0.1 % | within |  |
| 500 inductor_default warm | host | 0.08832 (1.5 %) | 0.0898 (0.9 %) | +1.7 % | exceeds |  |
| 500 inductor_default L2-flushed | GPU | 0.06534 (0.8 %) | 0.06523 (4.8 %) | -0.2 % | within |  |
| 500 inductor_default host_launch_us | host | 24.08 (1.3 %) | 25.19 (6.1 %) | +4.6 % | within |  |
| 500 inductor_wide warm | host | 0.08889 (2.1 %) | 0.09004 (0.2 %) | +1.3 % | within |  |
| 500 inductor_wide L2-flushed | GPU | 0.06555 (0.6 %) | 0.06566 (3.9 %) | +0.2 % | within |  |
| 500 inductor_wide host_launch_us | host | 25.62 (7.0 %) | 25.1 (2.6 %) | -2.0 % | within |  |
| 500 inductor_graphs warm | host | 0.1027 (8.5 %) | 0.1021 (2.3 %) | -0.5 % | within |  |
| 500 inductor_graphs L2-flushed | GPU | 0.06535 (5.3 %) | 0.06391 (4.5 %) | -2.2 % | within |  |
| 500 inductor_graphs host_launch_us | host | 39.48 (24.5 %) | 40.31 (6.7 %) | +2.1 % | within |  |
| 500 eager warm | GPU | 14.3 (0.5 %) | 13.76 (0.0 %) | -3.7 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (as above); eager enqueues one PyTorch kernel per operation; not isolated |
| 500 eager L2-flushed | GPU | 14.27 (0.9 %) | 13.74 (0.0 %) | -3.7 % | exceeds FLAG | drift: the code of these numbers is unchanged since the published runs (as above); eager enqueues one PyTorch kernel per operation; not isolated |
| 500 eager host_launch_us | host | 14,004 (0.9 %) | 13,466 (0.0 %) | -3.8 % | exceeds |  |
| 500 eager_graphs warm | GPU | 13.16 (0.6 %) | 12.95 (0.0 %) | -1.6 % | exceeds |  |
| 500 eager_graphs L2-flushed | GPU | 13.11 (0.6 %) | 12.92 (0.0 %) | -1.4 % | exceeds |  |
| 500 eager_graphs host_launch_us | host | 2.678 (18.7 %) | 2.595 (1.1 %) | -3.1 % | within |  |

### E2, the Python driver: kernel-only (torch.cuda._sleep before the start event), 5 ms spin (three runs) and 10 ms spin (one run)

| k, version | normal (spread) | kernel-only, 5 ms spin (spread) | kernel-only, 10 ms spin (one run) | 10 ms vs 5 ms | | kernel-only / normal | normal in the 10 ms run |
|---|---|---|---|---|---|---|---|
| 2 triton_hand | 0.02921 (4.1 %) | 0.01799 (2.1 %) | 0.01786 | -0.7 % | within the 5 ms spread | 0.616 | 0.02934 |
| 2 inductor_default | 0.04692 (2.2 %) | 0.01796 (1.5 %) | 0.01762 | -1.9 % | exceeds the 5 ms spread | 0.383 | 0.04682 |
| 2 inductor_wide | 0.04748 (0.5 %) | 0.01773 (1.2 %) | 0.01755 | -1.0 % | within the 5 ms spread | 0.373 | 0.04645 |
| 2 inductor_graphs | 0.06087 (2.1 %) | 0.01766 (1.6 %) | 0.01765 | -0.1 % | within the 5 ms spread | 0.29 | 0.06113 |
| 2 eager | 0.1297 (0.5 %) | 0.1199 (0.4 %) | 0.118 | -1.5 % | exceeds the 5 ms spread | 0.925 | 0.1299 |
| 2 eager_graphs | 0.1197 (0.2 %) | 0.1164 (0.2 %) | 0.1152 | -1.0 % | exceeds the 5 ms spread | 0.972 | 0.12 |
| 10 triton_hand | 0.02943 (1.5 %) | 0.01777 (0.5 %) | 0.01786 | +0.5 % | exceeds the 5 ms spread | 0.604 | 0.02896 |
| 10 inductor_default | 0.04617 (2.6 %) | 0.01789 (1.0 %) | 0.01796 | +0.4 % | within the 5 ms spread | 0.387 | 0.04683 |
| 10 inductor_wide | 0.04684 (0.1 %) | 0.01794 (1.4 %) | 0.01769 | -1.4 % | within the 5 ms spread | 0.383 | 0.04755 |
| 10 inductor_graphs | 0.05914 (2.0 %) | 0.0176 (1.1 %) | 0.01778 | +1.1 % | exceeds the 5 ms spread | 0.298 | 0.05902 |
| 10 eager | 0.3522 (0.3 %) | 0.3394 (0.3 %) | 0.3378 | -0.5 % | exceeds the 5 ms spread | 0.964 | 0.3532 |
| 10 eager_graphs | 0.3196 (0.2 %) | 0.315 (0.4 %) | 0.3145 | -0.2 % | within the 5 ms spread | 0.986 | 0.3198 |
| 50 triton_hand | 0.02956 (2.1 %) | 0.0184 (1.2 %) | 0.01818 | -1.2 % | exceeds the 5 ms spread | 0.623 | 0.02932 |
| 50 inductor_default | 0.04627 (4.2 %) | 0.01824 (2.1 %) | 0.01829 | +0.3 % | within the 5 ms spread | 0.394 | 0.04694 |
| 50 inductor_wide | 0.04699 (2.6 %) | 0.01822 (2.4 %) | 0.01808 | -0.7 % | within the 5 ms spread | 0.388 | 0.04641 |
| 50 inductor_graphs | 0.05939 (2.3 %) | 0.01818 (2.3 %) | 0.01793 | -1.4 % | within the 5 ms spread | 0.306 | 0.0603 |
| 50 eager | 1.461 (0.4 %) | 1.434 (0.0 %) | 1.433 | -0.1 % | exceeds the 5 ms spread | 0.981 | 1.46 |
| 50 eager_graphs | 1.37 (0.4 %) | 1.354 (0.1 %) | 1.352 | -0.1 % | exceeds the 5 ms spread | 0.988 | 1.37 |
| 100 triton_hand | 0.02982 (2.8 %) | 0.01867 (1.3 %) | 0.01865 | -0.1 % | within the 5 ms spread | 0.626 | 0.02997 |
| 100 inductor_default | 0.04814 (24.6 %) | 0.01959 (1.8 %) | 0.0197 | +0.6 % | within the 5 ms spread | 0.407 | 0.04894 |
| 100 inductor_wide | 0.04803 (1.2 %) | 0.01978 (1.5 %) | 0.0196 | -0.9 % | within the 5 ms spread | 0.412 | 0.04802 |
| 100 inductor_graphs | 0.06308 (35.0 %) | 0.01951 (0.9 %) | 0.01951 | +0.0 % | within the 5 ms spread | 0.309 | 0.06109 |
| 100 eager | 2.827 (0.5 %) | 2.803 (0.0 %) | 2.802 | -0.0 % | exceeds the 5 ms spread | 0.991 | 2.832 |
| 100 eager_graphs | 2.666 (0.4 %) | 2.642 (0.1 %) | 2.639 | -0.1 % | exceeds the 5 ms spread | 0.991 | 2.653 |
| 500 triton_hand | 0.07177 (0.3 %) | 0.0592 (1.3 %) | 0.05881 | -0.7 % | within the 5 ms spread | 0.825 | 0.07169 |
| 500 inductor_default | 0.0898 (0.9 %) | 0.05939 (0.0 %) | 0.05954 | +0.3 % | exceeds the 5 ms spread | 0.661 | 0.09027 |
| 500 inductor_wide | 0.09004 (0.2 %) | 0.05962 (0.7 %) | 0.05964 | +0.0 % | within the 5 ms spread | 0.662 | 0.09031 |
| 500 inductor_graphs | 0.1021 (2.3 %) | 0.05926 (1.3 %) | 0.05939 | +0.2 % | within the 5 ms spread | 0.58 | 0.1029 |
| 500 eager | 13.76 (0.0 %) | 13.75 (0.0 %) | 13.75 | +0.0 % | within the 5 ms spread | 0.999 | 13.77 |
| 500 eager_graphs | 12.95 (0.0 %) | 12.94 (0.0 %) | 12.94 | -0.0 % | within the 5 ms spread | 0.999 | 12.96 |

## Agreement

- E1a bit-exact flags, cleaned: 48 values, all 1: True
- E1a bit-exact flags, published: 36 values, all 1: True
- E1a bit-exact flags, reference: 36 values, all 1: True
- E1b bit-exact flags, cleaned: 60 values, all 1: True
- E1b bit-exact flags, published: 45 values, all 1: True
- E1b bit-exact flags, reference: 45 values, all 1: True
- E1c bit-exact flags, cleaned: 60 values, all 1: True
- E1c bit-exact flags, published: 45 values, all 1: True
- E1c bit-exact flags, reference: 45 values, all 1: True
- E2 FKL outputs (6 files per run): the SHA-256 of every run equal to reference run 1's: True
- E2 Python, accuracy, kernel and SASS columns (kernels, max_ulp_vs_exact, bitexact_fraction_vs_triton_hand, bitexact_fraction_vs_fkl, max_ulp_vs_fkl, sass_ffma, sass_fmul, sass_fadd, status), every cleaned run and the 10 ms run against the published runs: all equal

## E1 ncu pass of the cleaned build

- E1a: 18 kernels (published 18); the same kernels in the same launch order (function name and, for the hand-written ones, the first template argument): True; equal launch values: {'launch__registers_per_thread': True, 'launch__block_size': True, 'launch__grid_size': True, 'launch__shared_mem_per_block_dynamic': True}; largest difference in points of percent: sm__warps_active.avg.pct_of_peak_sustained_active 0.61, sm__maximum_warps_per_active_cycle_pct 0.00
- E1b: 15 kernels (published 15); the same kernels in the same launch order (function name and, for the hand-written ones, the first template argument): True; equal launch values: {'launch__registers_per_thread': True, 'launch__block_size': True, 'launch__grid_size': True, 'launch__shared_mem_per_block_dynamic': True}; largest difference in points of percent: sm__warps_active.avg.pct_of_peak_sustained_active 5.51, sm__maximum_warps_per_active_cycle_pct 0.00
- E1c: 20 kernels (published 20); the same kernels in the same launch order (function name and, for the hand-written ones, the first template argument): True; equal launch values: {'launch__registers_per_thread': True, 'launch__block_size': True, 'launch__grid_size': True, 'launch__shared_mem_per_block_dynamic': True}; largest difference in points of percent: sm__warps_active.avg.pct_of_peak_sustained_active 4.95, sm__maximum_warps_per_active_cycle_pct 0.00
- The pass ran with ncu's default kernel names (demangled); the published pass has the mangled names, which scripts/e1_table.py needs to join the ncu
  rows with the SASS of the binary. The rows above are matched by launch order (profile mode launches every kernel once, in a fixed order). The tables
  (e1_table.py, e1_compact.py) of the cleaned build need the pass again with --print-kernel-base mangled.

