# Does the NPP stream context of FastNPP's tests change anything? (RTX 4090 8.9, CUDA 13.4.3 NPP)

`initNppStreamContext` of FastNPP's `tests/testsNppCommon.h` (since commit 022ce18, 2025-04-12) stores the minor compute capability in the "major" field and the major in the "minor" one, so NPP is told the 4090
(8.9) is 9.8. It is used by the NPP benchmarks of the paper (`e0-npp`, `e0-cpu-time-npp`) and by FastNPP's new resize test. This folder compares, on the same machine in one window (3 alternating pairs), the two
E0 NPP benchmarks as they are ("swapped") with the same sources built with a corrected copy of the header ("fixed"; CMake generates it from the pinned one, `npp/CMakeLists.txt`, targets `*_ctxfix_cu`), and
the NPP calls of the E4 program with the context built both ways in one process.

**Result: no effect on this GPU.**
- `e0-npp` (`test_npp_batchresize_x_split3D.csv`, with CPU overhead): NPP time fixed / swapped, per batch, median of 3 runs: minimum 0.993, median 1.002, maximum 1.012; without CPU overhead (`..._No_CPU_Overhead.csv`): 0.964, 0.994, 1.015; `e0-cpu-time-npp`: 0.986, 0.997, 1.013. The run-to-run spread of the swapped version alone reaches 2.8 %, 3.8 % and 4.9 % (the differences are inside it). FastNPP's own times: fixed / swapped median 1.005, 0.996, 1.004.
- The agreement CSVs of `e0-npp` (outputs against the references) are identical between the two versions (32 lines each).
- E4 program, NPP chain of the batched pre-processing (E4a, 30 batches of 10 ... 300 crops, both measurement orders): the outputs are **bit-identical** with the two contexts in all 30 cases and the time ratio npp_ctx_as_tests / npp has median 1.000 (0.981 to 1.012 correct context first, 0.991 to 1.046 the other order).
- **Whole frames (E4c), `e4c-npp-correct-first` and `e4c-npp-tests-first`** (the first attempt, in `e4-npp-*`, is NOT valid for these three rows: NPP's batched resize needs more than one image and returned an error for a batch of 1, so those outputs were wrong; the program now uses the single-image `nppiResize_32f_C3R` for a frame and the pair was re-run in a later window): the NPP output is bit-identical with the two contexts at 720x480 -> 768x512, 1920x1080 -> 2048x1152 and 3840x2160 -> 4096x2304 in both orders, the times are equal to within 0.06 % (0.0365 / 0.0365 ms, 0.1546 / 0.1547 ms, 1.8475 / 1.8464 ms), and each output matches NPP's own (half-pixel) scalar reference to 5.35e-6.

So the swapped context does not change the NPP baseline of E0, E4a or E4c on the 4090. (Whether another GPU or another NPP version reacts to the reported compute capability was not tested.)
