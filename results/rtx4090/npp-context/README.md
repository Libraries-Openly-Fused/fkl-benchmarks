# Does the NPP stream context of FastNPP's tests change anything? (RTX 4090 8.9, CUDA 13.4.3 NPP)

`initNppStreamContext` of FastNPP's `tests/testsNppCommon.h` (since commit 022ce18, 2025-04-12) stores the minor compute capability in the "major" field and the major in the "minor" one, so NPP is told the 4090
(8.9) is 9.8. It is used by the NPP benchmarks of the paper (`e0-npp`, `e0-cpu-time-npp`) and by FastNPP's new resize test. This folder compares, on the same machine in one window (3 alternating pairs), the two
E0 NPP benchmarks as they are ("swapped") with the same sources built with a corrected copy of the header ("fixed"; CMake generates it from the pinned one, `npp/CMakeLists.txt`, targets `*_ctxfix_cu`), and
the NPP calls of the E4 program with the context built both ways in one process.

**Result: no effect on this GPU.**
- `e0-npp` (`test_npp_batchresize_x_split3D.csv`, with CPU overhead): NPP time fixed / swapped, per batch, median of 3 runs: minimum 0.993, median 1.002, maximum 1.012; without CPU overhead (`..._No_CPU_Overhead.csv`): 0.964, 0.994, 1.015; `e0-cpu-time-npp`: 0.986, 0.997, 1.013. The run-to-run spread of the swapped version alone reaches 2.8 %, 3.8 % and 4.9 % (the differences are inside it). FastNPP's own times: fixed / swapped median 1.005, 0.996, 1.004.
- The agreement CSVs of `e0-npp` (outputs against the references) are identical between the two versions (32 lines each).
- E4 program, NPP chain of the batched pre-processing (E4a, 30 batches of 10 ... 300 crops, both measurement orders): the outputs are **bit-identical** with the two contexts in all 30 cases and the time ratio npp_ctx_as_tests / npp has median 1.000 (0.981 to 1.012 correct context first, 0.991 to 1.046 the other order).
- **Not valid:** the E4c rows (single whole frames) of `e4-npp-*`: the batched NPP resize needs more than one image and returned an error for a batch of 1, so those outputs are wrong ("an NPP call failed", exit code 255). The program now uses the single-image resize for a frame; the whole-frame pair is re-run in a later window (`e4c-npp-*`).

So the swapped context does not change the NPP baseline of E0 or E4a on the 4090. (Whether another GPU or another NPP version reacts to the reported compute capability was not tested.)
