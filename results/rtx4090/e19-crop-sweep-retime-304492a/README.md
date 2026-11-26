# E19 crop sweep timed again at the new pin, FKL 304492a, against FKL b10fc66 (RTX 4090, RT-1)

**What.** The RTX 4090 half of the E19 crop sweep, `results/rtx4090/e19-crop-sweep/` (old), timed again after the pin move of
fkl-benchmarks to FKL #435:
- #435 was merged as `304492af4d777165bd556fdd6113439635724d02`, the tree of its last head d1cff06;
- the pin move is #66, merged as af96aef.

The window runs two builds of the same program, `opencv-cuda/stencil/e19_crop_sweep.cu`, alternately (`binaries.sha256`):
- **A:** fkl-benchmarks 976e42b with FKL b10fc66, the pin before the move;
- **B:** fkl-benchmarks main with FKL 304492a, the pin after it.

Decision: the coordinator's task RT-1 of 2026-11-26, Oscar's order with Q3. Oscar's rule of the same day applies: "Coordinate with fkl-reduce,
fkl-reduce has priority". So the window starts only in a quiet slot agreed with that session, and any run it disturbs is redone.

**State.** The builds and the kernel resources are done (compile only). The window waits for the coordinator's GRANTED.

## The two binaries (compile only, no GPU visible)

Both are `e19_crop_sweep_cu` of a `scripts/check_fkl_tree.sh` build: a link farm of the checkout, with the FKLB_* options, OpenCV and nvcc of
the reference build `build/local-rtx4090`.
- **The same flags as the record.** That reference build is the preset `local-rtx4090` with which the record's binary was built. The CUDA
  flags of the link-farm builds are those of the reference build (`-O3 -DNDEBUG -std=c++17`, sm_89 with PTX, `-ftemplate-depth=1000`,
  `-lineinfo`), with CMAKE_BUILD_TYPE Release, nvcc 13.4.92 and OpenCV 4.14.0.
- **A** = `scratch/pin-435/base-b10fc66/bench-build/bin/e19_crop_sweep_cu`, sha256 `9cb5a18f...`.
  - This is the base build of #68's identity check: 976e42b with FKL b10fc66, cvGPUSpeedup 6d26c65 and FastNPP a1b5e24. Its sha256 is
    that record's (`../pin-move-check-b10fc66-48fbf3f/sass-identity/binary-sha256-base-b10fc66.txt`).
  - Its fkl_batch kernel and the host code of its timed calls equal those of the record's binary (246b1c8, FKL 24478f0), per #60's records.
    #60 compared exactly that source and FKL with main's, and found the timed calls of `fkl_batch`, `opencv_cuda` and `npp` unchanged
    (`../pin-move-check-b10fc66-48fbf3f/`, section 3.6).
- **B** = `scratch/pin-435/main-304492a/bench-build/bin/e19_crop_sweep_cu`, sha256 `acf58fc7...` (`compile-B/`, 2026-11-26 19:02:09 to
  19:03:33, nice 19, `-j8`, `scripts-used/check_fkl_tree_rt.sh`).
  - It was built from main 73a9650. That is 658a896 plus #67, which changes only `results/rx9060/`, so the program's source and headers are
    658a896's. FKL 304492a, cvGPUSpeedup 6d26c65 and FastNPP a1b5e24.
  - The build script is `scripts/check_fkl_tree.sh` with two changes: the checkout comes from `CHECK_ROOT`, and only six targets are built.
    These are this program, and `e16_attention_cu`, `e16_query_tile_cu`, `e18_box_cu`, `e18_box_split_cu` and `e18_stencil_cu` for the
    later re-times RT-2 and RT-4.
  - Its folder name has the length of A's, so the two builds' source paths have the same length.
- **What the binaries carry.** The link farm has no `.git`, so both binaries name the repository commit "unknown". They carry their FKL
  commit, and no `-dirty` commit.

## The kernels of the program, A against B (`resources/`)

From `cuobjdump -res-usage` and `-sass` (13.4.3, sm_89), paired with `scripts-used/pr435_sass_identity.py` and FKL #435's rename rule
(`resources/sass-e19-A-vs-B.txt`, `fkl-batch-resources.txt`, `res-usage-{A,B}.txt`). There are 18 kernels in each build:

| kernel | N | registers A / B | stack A / B | LDL, STL A / B | constant bank 0 A / B | instructions A / B | SASS |
|---|---|---|---|---|---|---|---|
| fkl_batch (MorphologyDPP) | 1 | 35 / 40 | 40 / 0 B | 2, 5 / 0, 0 | 472 / 472 B | 568 / 544 | different |
| fkl_batch (MorphologyDPP) | 10 to 150 (15 kernels) | 40 / 40 | 0 / 0 B | 0, 0 / 0, 0 | 688 to 4,048 B (both) | 544 / 544 | different |
| fkl_loop (LinearFilterDPP) | (one kernel) | 39 / 39 | 0 / 0 B | 0, 0 / 0, 0 | 456 / 456 B | 696 / 704 | different |
| gpuSpin | | 8 / 8 | 0 / 0 B | 0, 0 / 0, 0 | 360 / 360 B | 24 / 24 | identical |

- At every N the shared memory of fkl_batch's kernel is 1,232 B in both builds, and its constant bank 0 grows by 240 B per 10 crops.
- FKL #435 passes the details of the filter DPPs as a `__grid_constant__` kernel parameter (`../fkl-pr435-check/`); the SASS of these kernels
  differs with it. No cause of a time difference is claimed from this.

## The window (`window-script.sh`)

(To be run on the coordinator's GRANTED.)

**The protocol is the record's.** Each sweep is 16 processes, one per N (`E19_CROPS` = 1, 10, ..., 150), in ascending order. Each process is
a complete run of the program:
- first the checks;
- then whole_call and kernel_only of every row, in the order fkl_loop, fkl_batch, opencv_cuda, npp.

fkl_loop runs first only to keep E19's order. It is in no table.

**A and B alternate, as in #63:**
- `warmup-A`, then `warmup-B` (not used);
- then `quiet-run1-A`, `quiet-run1-B`, `quiet-run2-A`, `quiet-run2-B`, `quiet-run3-A`, `quiet-run3-B`.

That is 128 runs. There is no nsys sweep.

**Before every run**, the script waits for an idle machine: no build, compile job or measurement, a 1-minute load below 0.5, and no other
program on the RTX 4090 (up to 300 s).

**After every run**, it checks again for a build, a compile job or another program on the RTX 4090. If one appeared during the run, the window
stops, and that run is the affected one, to be redone.

**The runs:** each is pinned to cores 0-3,16-19, under a 300 s timeout, and must exit 0, pass its checks and write complete CSVs. nvidia-smi
reads the GPU every 100 ms.

**The report** (`scripts-used/rt1_report.py`) gives, per N, row and interval:
- the medians and spreads of old, A and B;
- B against A, and A against old: the change, whether it exceeds the larger spread or the ranges do not overlap, the bound, and the flag
  (above 3 % for a GPU-bound number, 5 % for a host-bound one).

No cause is named unless shown.

## Files

| path | content |
|---|---|
| `binaries.sha256` | A and B, the two binaries the window runs |
| `window-script.sh` | the window's script |
| `compile-B/` | B's build: the report, target lists, configure log, build time, console, and the sha256 of its six binaries |
| `resources/` | the kernels of A and B: `cuobjdump -res-usage`, the SASS comparison and its CSV, and the per-N table |
| `scripts-used/` | `check_fkl_tree_rt.sh`, `pr435_sass_identity.py`, `fkl_batch_resources.py` and `rt1_report.py` |
