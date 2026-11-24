# E19 crop sweep, phase 1 (e19-crop-sweep-phase1), 2026-11-24: the checks on the RX 9060 at 16 numbers of crops, NOT timed

This is the checks window of `rpp/e19_crop_sweep.h`: E19 (`rpp/e19_batched_box.h`, unchanged) at N = 1, 10, 20, ..., 150 crops, one N per process (`E19_CROPS`). The program's header gives the crop table. **Nothing here is timed for use.** The one timing run, under rocprofv3, serves only the spin coverage.
- **The program:** c8eb575, FKL 24478f0 (the submodule pin), RPP 3.2.0 (ROCm 10.1.0), amdclang++ 24.0.0, gfx1200.
- **Decision:** Oscar's order of 2026-11-24, relayed by the coordinator session the same day (E19 crop sweep on the RX 9060). The coordinator granted this window the same day.
- **`step0/`, the build:** a clean worktree and a fresh build folder; 0 errors and 0 warnings; no `-dirty`. sha256 `f9fcb2a43e96be2249734491e6fecf8e3baadc3484d0b8ccba895559a2192ee8`.
- **The window** (`window-checks.sh`, run as committed; console in `window/window-console.txt`) ran from 07:19:30 to 07:20:00Z.
  - A foreground check at 07:19:24Z found no `Runner.Worker` and a load of 0.15.
  - Every guard passed, and the runner was idle before each of the 33 runs.
  - Every run exited 0 with its success line, and no CSV has `INCOMPLETE` or `-nan`.
  - hwmon (`window/gpu_clocks.csv`, every 100 ms) read at most 3,451 MHz sclk, 83 W, 41 °C edge and 47 °C junction.

## The checks (`window/check/nNNN/`, `E19_NO_TIMING=1`, `E19_DUMP`)
`analysis/checks_summary.py` gives the required rows of every N in `analysis/checks-summary.csv`.
- **All 80 pass:** 5 required rows at each of the 16 N, 0 failing. Every comparison covers all N x 65,536 values. Every buffer was filled with 0xFF bytes before the call.

| row | check | at every N |
|---|---|---|
| `rpp` | grey: the rule found by comparison, `t = 0.587f G; t = fma(0.299f, R, t); t = fma(0.114f, B, t)`, ties to even | 0 differ |
| `rpp` | box against round(sum / 9) of its own grey, REPLICATE | 0 differ |
| `rpp` | float against (v / 255 - 0.45) / 0.225 in double | max 1.19e-7 to 1.35e-7, all finite |
| `fkl_loop` | against FKL's pipeline in double, the same grey order | 0 beyond 1e-5, max 6.1e-7 to 7.1e-7, all finite |
| `fkl_batch` | the same | 0 beyond 1e-5, max 3.56e-7, all finite |

- **N = 30 is E19 bit for bit:**
  - Its checks CSV equals E19 phase 1's (`results/rx9060/e19-batched-box-phase1/check/`) in every field of all 23 rows, apart from the new `crops` column.
  - Its five dump files have the sha256 sums of E19 phase 1's dump (`analysis/dump-sha256.txt`).
- **Every N's output is a prefix of N = 150's** (`analysis/dump_prefix.sh`, `dump-prefix.txt`). For every N, all five dump files equal the first N planes of N = 150's.
- **The dumps** (about 1.1 GB) are kept outside the repository; `analysis/dump-sha256.txt` gives their sizes and sha256 sums.

### fkl_batch against rpp (`analysis/between.py`, `between.csv`, from the dumps)
| N | values | differing | max abs difference | in grey levels (x 57.375) |
|--:|--:|--:|--:|--:|
| 1 | 65,536 | 65,536 | 0.00774652 | 0.444 |
| 10 | 655,360 | 655,360 | 0.00774652 | 0.444 |
| 20 | 1,310,720 | 1,310,719 | 0.00774652 | 0.444 |
| 30 | 1,966,080 | 1,966,079 | 0.00774652 | 0.444 |
| 40 | 2,621,440 | 2,621,438 | 0.00774652 | 0.444 |
| 50 | 3,276,800 | 3,276,798 | 0.00774652 | 0.444 |
| 60 | 3,932,160 | 3,932,158 | 0.00774652 | 0.444 |
| 70 | 4,587,520 | 4,587,518 | 0.00774652 | 0.444 |
| 80 | 5,242,880 | 5,242,878 | 0.00774652 | 0.444 |
| 90 | 5,898,240 | 5,898,238 | 0.00774652 | 0.444 |
| 100 | 6,553,600 | 6,553,597 | 0.00774652 | 0.444 |
| 110 | 7,208,960 | 7,208,956 | 0.00774652 | 0.444 |
| 120 | 7,864,320 | 7,864,316 | 0.00774652 | 0.444 |
| 130 | 8,519,680 | 8,519,676 | 0.00774652 | 0.444 |
| 140 | 9,175,040 | 9,175,034 | 0.00774652 | 0.444 |
| 150 | 9,830,400 | 9,830,393 | 0.00774652 | 0.444 |

- **The 4/9-level gap is RPP's 8-bit box.** RPP's box equals round(sum / 9) exactly (its box check above), and FKL keeps sum / 9 in float. Their difference is at most 4/9 of a grey level, reached where sum mod 9 is 4 or 5. 4/9 = 0.4444, and the measured largest difference is 0.444456 levels.
- **Nothing is non-finite.** No tensor value is non-finite at any N.
- **fkl_loop against fkl_batch** is in `between.csv` too (max 7.7e-7), as in E19's reported row. `fkl_loop` is out of the comparison (Oscar, 2026-11-20).

## Kernels per call (`window/rocprofv3/nNNN/`, `E19_PROFILE=1`; `analysis/kernels_per_n.py`, `kernels-per-n.tsv`)
In every profile run, every row's three profiled calls launch the same kernels. `calls.txt` comes from `results/rx9060/e18-box-phase1/rocprofv3/rocprof_calls.py`. The grids are in work-items.

| row | kernels per call at N | grids |
|---|---|---|
| `fkl_batch` | **1** at every N | `fk::morphologyDPPKernel` 256 x 256 x **N** (one grid plane per crop), 16 x 8 x 1 |
| `rpp` | **N + 3**: N `__amd_rocclr_copyBufferRectAligned` (the N `hipMemcpy2DAsync`), then RPP's 3 kernels | copies: ceil(N/2) x 48 x 256 x 1 and floor(N/2) x 192 x 256 x 1, by count. E19 phase 1 traced 48 to the even columns and 192 to the odd ones, and every crop here has its E19 crop's address mod 64. RPP's 3 kernels: 32 x 256 x N, 48 x 304 x N and 32 x 256 x N |
| `fkl_loop` | **N** x `fk::linearFilterDPPKernel` | 256 x 256 x 1 each |

- **No call allocates, frees, copies outside its own N copies, or sets memory.** At every N, fkl_batch and fkl_loop make no hipMemcpy* call. rpp makes exactly N hipMemcpy2DAsync per call, and the memory-copy trace records none of them as an SDMA copy.

## The spin coverage at N = 150 (`window/timing-rocprofv3/n150/`, one complete run under rocprofv3, times not used)
`calls.txt` comes from `rocprof_calls.py --spin`. For rpp, `rpp-spin-with-copies.txt` comes from `results/rx9060/e19-batched-box/analysis/rpp_spin_with_copies.py`, which counts its 150 copies as part of the call; `--spin` leaves them out.

| row | iterations covered | kernels per iteration | smallest margin | median margin | spin end -> first kernel, max |
|---|--:|--:|--:|--:|--:|
| `fkl_loop` | 100 of 100 | 150 | 3,709.6 us | 4,123.7 us | 26.1 us |
| `fkl_batch` | 100 of 100 | 1 | 5,002.7 us | 5,026.9 us | 29.2 us |
| `rpp` (with its copies) | 100 of 100 | 153 | 4,125.5 us | 4,223.3 us | 26.0 us |

- **The 5 ms spin covers the host's enqueue of every kernel_only call in every row at N = 150.** The smallest margin is fkl_loop's 3.71 ms; among the compared rows it is rpp's 4.13 ms.
- **The large rpp gap is an artefact of the default script.** Without its copies (`--spin`), rpp's "spin end -> first kernel" reads 1,122.6 us, because the first kernel counted there is the one after the 150 copies.
- **The traces.** The kernel and HIP API traces are gzipped; `trace-csv-sha256.txt` gives the sha256 of the CSVs.

## Not from the GPU
- **`isa/kernarg-sizes.tsv`:** fkl_batch's kernarg segment (code-object notes) is 352 + 24 N bytes:
  - 376 at N = 1;
  - 1,072 at N = 30, equal to E19's binary;
  - 3,952 at N = 150.

  fkl_loop's kernel is 368 B at every N.
- **`analysis/crop_table.py`, `crop_table.txt`, the crop table:**
  - The largest right edge is 3640 and the largest bottom edge 2120.
  - The crops are disjoint up to N = 60. Above 60 they overlap: at N = 150 all 150 overlap another, with at most 3 crops on one pixel.
  - Every crop's source address mod 16 equals that of the E19 crop it repeats.
  - The horizontal shifts are 960 B (SX = 320) and 576 B (SX = 192), so the address is also kept mod 64 but not mod 128. The vertical shifts are multiples of the 11,520-byte frame row.
