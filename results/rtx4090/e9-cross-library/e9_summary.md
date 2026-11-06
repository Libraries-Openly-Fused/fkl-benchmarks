# E9 summary (the four facts, in the order of the paper)

## (a) Launches

| variant | kernel nodes per call, runs 0 / 1 / 2 |
|---|---|
| M1 | 1 / 1 / 1 |
| M2 | 1 / 1 / 1 |
| R1 | 1 / 1 / 1 |
| R2 | 1 / 1 / 1 |

nsys cross-check (one call of each of M1, M2, R1, R2 and of the two listings inside the profiler range): total kernel launches in the range: 6

```
1 void fk::launchTransformDPP_Kernel<(fk::ParArch)1, (fk::TF)0, (bool)1, fk::TransformDPPDetails_<void, (bool)0, fk::ReadInstantiableOperation<fk::BatchRead<(fk
1 void fk::launchTransformDPP_Kernel<(fk::ParArch)1, (fk::TF)0, (bool)1, fk::TransformDPPDetails_<void, (bool)0, fk::ReadInstantiableOperation<fk::BatchRead<(fk
2 void fk::launchTransformDPP_Kernel<(fk::ParArch)1, (fk::TF)0, (bool)1, fk::TransformDPPDetails_<void, (bool)0, fk::ReadInstantiableOperation<fk::BatchRead<(fk
1 void fk::launchTransformDPP_Kernel<(fk::ParArch)1, (fk::TF)0, (bool)1, fk::TransformDPPDetails_<void, (bool)0, fk::ReadInstantiableOperation<fk::BatchRead<(fk
1 void fk::launchTransformDPP_Kernel<(fk::ParArch)1, (fk::TF)0, (bool)1, fk::TransformDPPDetails_<void, (bool)0, fk::ReadInstantiableOperation<fk::BatchRead<(fk
```

## (b) Bit identity (function `compare()` of npp/e9_cross_library.cu: memcmp of the 4-byte bit pattern of each output float + NaN count of both outputs; output pre-filled with the NaN pattern)

| comparison | runs x seeds | all identical | elements differing (min..max over rows) | max abs difference (max) | NaN count (max of the two outputs) |
|---|---:|:---:|---|---:|---:|
| M1listing_vs_M1 | 9 | YES | 0..0 of 1228800 | 0 | 0 |
| M1shortlisting_vs_M1 | 9 | YES | 0..0 of 1228800 | 0 | 0 |
| M1_vs_R1 | 9 | YES | 0..0 of 1228800 | 0 | 0 |
| M2_vs_R2 | 9 | YES | 0..0 of 1228800 | 0 | 0 |
| R1_vs_R2 | 9 | NO | 1227858..1227896 of 1228800 | 1.20363 | 0 |
| M1_vs_M2 | 9 | NO | 1227858..1227896 of 1228800 | 1.20363 | 0 |

## (c) Time per call (ms), mixed against the single-library kernel it shares its resize with; THE run is run 0; order of the variants rotated per run

### GPU time per call (CUDA graph of 100 calls), GPU-bound, flag 3 %

| variant | run 0 | run 1 | run 2 | median | spread over runs % | within-run block range (max/min - 1, max over runs) % |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.00595 | 0.00590 | 0.00594 | 0.00594 | 0.8 | 2.4 |
| R1 | 0.00572 | 0.00574 | 0.00579 | 0.00574 | 1.3 | 1.5 |
| M2 | 0.00596 | 0.00595 | 0.00590 | 0.00595 | 1.0 | 2.1 |
| R2 | 0.00604 | 0.00606 | 0.00610 | 0.00606 | 0.9 | 1.8 |

| pair (mixed / single-library) | ratio run 0 | run 1 | run 2 | median | spread over runs (max - min, % points) | exceeds the flag (median) | runs beyond the flag |
|---|---:|---:|---:|---:|---:|:---:|---:|
| M1 / R1 | 1.0398 | 1.0269 | 1.0255 | 1.0269 | 1.4 | no | 1 of 3 |
| M2 / R2 | 0.9858 | 0.9822 | 0.9673 | 0.9822 | 1.8 | no | 1 of 3 |

### launch loop (host and GPU), host-bound (the launch costs about as much as the kernel), flag 5 %

| variant | run 0 | run 1 | run 2 | median | spread over runs % | within-run block range (max/min - 1, max over runs) % |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.00638 | 0.00639 | 0.00640 | 0.00639 | 0.4 | 1.0 |
| R1 | 0.00624 | 0.00624 | 0.00625 | 0.00624 | 0.1 | 0.2 |
| M2 | 0.00644 | 0.00642 | 0.00646 | 0.00644 | 0.5 | 0.8 |
| R2 | 0.00665 | 0.00665 | 0.00666 | 0.00665 | 0.1 | 0.3 |

| pair (mixed / single-library) | ratio run 0 | run 1 | run 2 | median | spread over runs (max - min, % points) | exceeds the flag (median) | runs beyond the flag |
|---|---:|---:|---:|---:|---:|:---:|---:|
| M1 / R1 | 1.0217 | 1.0245 | 1.0247 | 1.0245 | 0.3 | no | 0 of 3 |
| M2 / R2 | 0.9686 | 0.9661 | 0.9701 | 0.9686 | 0.4 | no | 0 of 3 |

## (d) Registers and SASS lines of the final binary (cuobjdump -arch sm_89), before the window / before #45: M1 34 / 552, R1 32 / 544, M2 33 / 544, R2 32 / 560; after #45 (compile time): M2 552, R2 568

| kernel | registers | SASS lines | stack | shared | local |
|---|---:|---:|---:|---:|---:|
| M1 (= long listing) | 34 | 552 | 0 | 0 | 0 |
| M1 short listing | 34 | 552 | 0 | 0 | 0 |
| M2 | 33 | 552 | 0 | 0 | 0 |
| R1 | 32 | 544 | 0 | 0 | 0 |
| R2 | 32 | 568 | 0 | 0 | 0 |
