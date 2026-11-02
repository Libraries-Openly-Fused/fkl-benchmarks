# E3 controlled compile repeat at the final pin (external/fkl at c8fcbe8), 2026-11-02

One window (2026-11-02, 13:00:00 to 13:58:42 by the machine's clock; the requested set was done at 13:50:24), one compile at a time, nothing else of mine running and the other agents asked to run nothing on the machine (compile only: no GPU program ran). It gives the paper one consistent set of compile times, memory, phase times and cubin sizes at the pin it uses; nothing in this folder comes from another compile. The values the paper quoted before the repeat are in `paper-values-before.csv` and compared in `e3_repeat_vs_paper.md`.

## Conditions

- Library: `external/fkl` of fkl-benchmarks at c8fcbe8 (`Beta-0.1.14-LTS-202-gc8fcbe8`, no local changes; the tip of LTS-C++17 with #417 merged); cvGPUSpeedup 6d26c65 and the OpenCV 4.14 headers for the StaticLoop translation units (from `build/merged`; the cubin sizes and the SASS instruction counts equal those of the committed StaticLoop runs at all 14 lengths).
- Toolchain: CUDA 13.4.3, nvcc 13.4.92, g++ 15.2.0; Ryzen 9 5950X, Ubuntu 26.04.1, kernel 7.0.0-38; scaling governor `performance`, boost on (`environment-before.txt`).
- Every compile: ONE core (core 8, the second CCD), `nvcc --threads 1 -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_89,code=[compute_89,sm_89] -ftemplate-depth=1000 -lineinfo --resource-usage`; wall, user and system time and the peak resident memory from GNU time (`/usr/bin/time`, %M); the phase times from nvcc's own `--time` file (cudafe++ = front end, cicc, ptxas, `gcc` = the host compiler: preprocessing, compiling the host code and the final link, `other` = fatbinary, nvlink, ...); the size of the cubin, the SASS instruction count and the .text bytes from `cuobjdump`. The programs are not run.
- The python parents of the sweeps, `cuobjdump` and the watcher ran on cores 4-7,20-23 (the first CCD), the load logger on core 5; the compile CCD had the compile and nothing else of mine. `load.log` (every 20 s: the load average, the busy share of cores 8 and 24, of the other cores of that CCD and of the other CCD, and every process that used more than 3 % of a core) shows what else ran; `environment-before.txt` and `environment-after.txt` are `ps` snapshots with the load, the governor and the memory.
- The CSV headers say `-dirty` for the fkl-benchmarks repository: the only tracked file that differs from the commit named there is `results/rtx4090/e0-run.log` (the complete log of the quiet windows, whose committed copy is older); the scripts are the committed ones (`environment-before.txt` lists the differing files).
- Units: the columns `*_mb` are MiB (GNU time reports KiB and the scripts divide by 1024). The paper's "GB" for the memory at N = 4,000 is MiB / 1000 (13,433 MiB -> "13.4 GB"); exact readings: GiB = MiB / 1024, GB (10^9 bytes) = MiB x 1.048576 / 1000. Cubin sizes are in bytes (MB = 10^6 bytes in the text).

## What was compiled, in this order (`window-script.sh` is the script that ran; `window-console.txt` its complete output)

0. warm-up: one compile of the distinct chain at N = 1 (`e3_repeat_warmup_A.csv`; fills the file cache; kept, not used).
1. REQUESTED, `e3_repeat_staticloop.csv`: the StaticLoop translation units of the vertical-fusion benchmark (cvGPUSpeedup mul_add1 chain, `kernel<k>.cu` with 100 (k - 1) + 2 operations) at k = 1, 101, 200, i.e. 2, 10,002 and 19,902 operations, three compiles each.
2. REQUESTED, `e3_repeat_A.csv`: the distinct-operations chain (variant A of `fkl-self/compiletime/gen_chain.py`: one distinct parameterised operation per fused operation, float) at N = 1, 1,000, 2,000, 3,000 and 4,000, three compiles each.
3. EXTRA, compiled after the requested set so that it stands alone, only so that the figure has the point density of the old one: `e3_repeat_extra_A.csv` (N = 100, 500, 1,500) and `e3_repeat_extra_staticloop.csv` (the other 11 lengths of the old figure: k = 2, 3, 6, 11, 21, 41, 61, 81, 126, 151, 176), three compiles each.

## Files

- `e3_repeat_*.csv`: one row per compile, as written by `scripts/e3_sweep.py` and `scripts/e3_staticloop.py` (columns: wall, user, system time, peak memory, the five phase times, registers, spills, SASS instructions, .text bytes, cubin bytes); `nvcc-time/`: the raw `nvcc --time` file of every compile (66 files for 67 compiles: the warm-up's file has the name of the requested N = 1, rep 0, and was overwritten by it; its phase times are in `e3_repeat_warmup_A.csv`).
- `e3_repeat_summary.csv` / `.md` (`scripts/e3_repeat_table.py`): per length the values of the compiles in the order they ran, median, minimum, maximum and the spread (max - min) / median, the phase medians and their shares of the wall time, the cubin size. `e3_table_chains.csv` and `e3_table_staticloop.csv` (same script): the layout that `scripts/plot_compiletime.py` reads, one value per length (the median of its compiles).
- `e3_repeat_vs_paper.csv` / `.md` (`scripts/e3_repeat_compare.py`): the values the paper quoted before (`paper-values-before.csv`) against the repeat, in the standing form.
- `../../../../figures/compiletime_single_r2_c8fcbe8.{pdf,png,txt}`: the compile-time figure drawn from this set (same size and style as `compiletime_single_r2.*`, which is unchanged): 8 distinct-operation lengths and 14 StaticLoop lengths, the median of three compiles each, no time-out marker; the dotted line at N = 3,848 comes from the committed sweeps of variant B (`../e3_limit_B_3847.csv`, `../e3_limit_B_3848.csv`), and the sidecar says so.
- `window-script.sh`, `window-console.txt`, `load.log`, `environment-before.txt`, `environment-after.txt`, `tu-sha256.txt` (sha256 of every translation unit that was compiled).


## Results of the requested lengths (three compiles each; the other lengths are in `e3_repeat_summary.md`)

| length | wall time of the three compiles (s) | median (s) | spread | peak memory (MiB) | cicc (s) | ptxas (s) | host g++ (s) | cubin (bytes) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| distinct N = 1 | 2.567 / 2.569 / 2.579 | 2.57 | 0.47 % | 296 / 296 / 296 | 0.67 | 0.01 | 1.05 | 18,080 |
| distinct N = 1,000 | 22.213 / 22.310 / 22.395 | 22.31 | 0.82 % | 1,463 / 1,462 / 1,465 | 12.58 | 0.44 | 8.34 | 13,266,080 |
| distinct N = 2,000 | 95.183 / 95.177 / 94.760 | 95.18 | 0.44 % | 4,060 / 4,060 / 4,060 | 65.50 | 1.66 | 26.65 | 50,521,760 |
| distinct N = 3,000 | 262.359 / 262.770 / 262.314 | 262.36 | 0.17 % | 8,048 / 8,048 / 8,048 | 193.86 | 5.00 | 61.61 | 111,770,016 |
| distinct N = 4,000 | 571.738 / 573.148 / 574.402 | 573.15 | 0.46 % | 13,429 / 13,433 / 13,432 | 425.08 | 11.39 | 134.30 | 197,023,392 |
| StaticLoop 2 operations | 3.738 / 3.747 / 3.732 | 3.74 | 0.40 % | 431 / 431 / 431 | 1.06 | 0.01 | 1.37 | 30,048 |
| StaticLoop 10,002 operations | 10.109 / 10.090 / 10.034 | 10.09 | 0.74 % | 437 / 437 / 437 | 6.39 | 0.98 | 1.39 | 687,072 |
| StaticLoop 19,902 operations | 31.450 / 31.552 / 31.532 | 31.53 | 0.32 % | 1,072 / 1,073 / 1,075 | 25.15 | 3.63 | 1.40 | 1,342,176 |

The run-to-run spread, (max - min) / median of the three compiles, is 0.17 to 0.82 % in wall time at the requested lengths (0.08 to 1.30 % over all 22 lengths), and 0.00 to 1.15 % in the peak memory; registers, spills, SASS instructions, .text bytes and cubin bytes are identical in the three compiles of every length. The code is the same as in the earlier runs: for example registers 17, no spills, 14,144 SASS instructions and 197,023,392 cubin bytes at N = 4,000.

## The values the paper quoted before, against the repeat (`e3_repeat_vs_paper.md` has the same comparison with the spread and the 5 % flag per quantity; the paper's values are in `paper-values-before.csv`)

| quantity | the paper before | where it comes from (committed file, 2026-10-30) | other compile jobs on the machine (load audit) | new: median of 3, spread | change |
|---|---|---|---|---|---|
| compile time, N = 1 | 2.7 s | `compile-times-1-100-500-2000/with_fix.csv` 2.688 to 2.715 s (5 compiles), `without_fix.csv` 2.700 to 2.713 s | one: the two sweeps ran side by side on cores 8 and 9 (that folder's README) | 2.569 s, 0.47 % | -4.9 % |
| compile time, N = 1,000 | 25 s | `e3_getarg_pr_A.csv` 25.62 / 24.88 / 24.53 s (12:07, core 13) | four other sweeps (cores 8, 14, 15 and one unpinned) | 22.310 s, 0.82 % | -10.8 % |
| cicc at N = 1,000 of the wall time | 13.9 s of 24.7 s | not traced to one compile (the three compiles of `e3_getarg_pr_A.csv`: cicc 14.48 / 13.98 / 13.73 s) | (as above) | 12.58 s of 22.31 s | -9.5 % / -9.7 % |
| compile time, N = 2,000 | 100 s | `compile-times-1-100-500-2000/with_fix.csv` 99.09 to 100.37 s (5 compiles) | one: the sweep on core 8 ran at the same time | 95.177 s, 0.44 % | -4.8 % |
| compile time, N = 3,000 | 330 s | `e3_getarg_pr_A.csv` 332.1 s, one compile (12:13 to 12:18, core 13) | four other sweeps (cores 8, 14, 15 and one unpinned) | 262.359 s, 0.17 % | -20.5 % |
| compile time, N = 4,000 | 600 to 700 s | `cubin-sizes/e3_cubin_A.csv` 595.2 s (17:50 to 18:00, core 14) and `e3_patched_A.csv` 696.6 s (12:07 to 12:19, core 14) | about one for 22 % of the time (a direct nvcc on core 13, 131 s); four other sweeps | 573.148 s, 0.46 % | -4.5 % from the lower end of the range |
| peak compiler memory, N = 4,000 | 13.4 GB | `cubin-sizes/e3_cubin_A.csv` 13,433 MiB (the paper's GB is MiB / 1000) | does not depend on the load (13,429 to 13,613 MiB in all six compiles at N = 4,000, three old and three new) | 13,429 / 13,433 / 13,432 MiB = 13.12 GiB, 0.03 % | unchanged; the unit is GiB: 13.1 GiB |
| phases, N = 4,000 | cicc 440 of 595 s; host compiler 24 %; ptxas 12 s | `cubin-sizes/e3_cubin_A.csv`: cicc 439.5 s, host 140.8 s (23.7 %), ptxas 12.1 s | (as above) | cicc 425.1 s of 573.1 s (74.2 %), host 134.3 s (23.4 %), ptxas 11.39 s | -3.4 % / -2.6 % / -5.1 % |
| cubin size | 13.3 MB at N = 1,000, 197 MB at N = 4,000 | `cubin-sizes/e3_cubin_A.csv` 13,266,080 and 197,023,392 bytes | not load dependent | 13,266,080 and 197,023,392 bytes, identical in the three compiles | 0.0 % (the same bytes) |
| StaticLoop compile time | 4 to 35 s | `e3_staticloop_mul_add1.csv` 4.04 s (2 operations) to 35.13 s (19,902; one compile) (11:00 to 11:07, core 10) | one: the sweep on core 8 ran during all of it (408 s of 408 s) | 3.74 to 31.53 s, 0.74 % at most | -7.4 % at 2 operations, -9.8 % at 10,002, -10.2 % at 19,902 |
| StaticLoop memory | about 1 GB | the same file: 432 MiB (2 operations), 438 MiB (10,002), 1,071 MiB (19,902) | does not depend on the load | 431, 437 and 1,073 MiB (1.05 GiB at 19,902) | unchanged; only the largest translation unit is near 1 GiB |
| StaticLoop cubin | 1.3 MB | the same file: 1,342,176 bytes at 19,902 operations | not load dependent | 1,342,176 bytes, identical at all 14 lengths | 0.0 % (the same bytes) |
| StaticLoop front end and host compiler | 1.4 s and 1.5 s (constant) | the same file: cudafe++ 1.367 to 1.388 s, host g++ 1.467 to 1.515 s over the 14 lengths | one (as above) | cudafe++ 1.287 to 1.294 s, host g++ 1.367 to 1.402 s over the 14 lengths | -7.8 % / -7.8 % |

Every difference larger than the spread of the repeat that is not rounding of the paper's value is in the same direction: the old compile was slower. The cubin sizes and the memory are the same.

## The machine during the window (`load.log`, 177 samples every 20 s)

Busy share of cores 8 and 24 (the compile core and its SMT sibling): median 50 %, maximum 50 % (one compile thread; the sibling idle). Busy share of the other 14 threads of that CCD: mean 0.0 %, maximum 1.0 %. Busy share of the other CCD (the first, with the python parents, `cuobjdump` and the logger): mean 0.1 %, maximum 10 % (the start of a python sweep). Processes that used more than 3 % of a core in a sample, besides the compile itself (`cicc` 96 samples, at most 100 %, `cc1plus` 23 samples, at most 100 %): `claude` 5 sample(s), at most 6 % of a core, `python` 1 sample(s), at most 141 % of a core, `ptyxis` 1 sample(s), at most 3 % of a core (the `claude` entries are the agents' command-line processes, whose affinity mask is the compile CCD; the `python` entry is the sweep's own start-up on the first CCD). No other compile, benchmark or profiler process appears in the log (it lists every process above 3 % of a core); the guard at the start found nothing running, and the other agents had been asked to run nothing on the machine.

## Why the old compiles were slower: the other compile jobs of that day

The old values were not measured on an idle machine: other compile sweeps were running (the load audit `../../e0-load-audit/load_events.csv` is my record of them; there was no load logger yet on 2026-10-30), all of them but one unpinned sweep on the same CCD (cores 8 to 15 and their SMT siblings), whose cores share the L3 cache, and all cores share the memory controller. The compile is host-bound (one thread of nvcc, cicc, ptxas and g++ on one core). I did not test which shared resource it is; the old compile was slower than the idle one by

| compiles of the same code | other compile jobs | old compile (s) | new, idle (s) | the old one was slower by |
|---|---|---:|---:|---:|
| distinct N = 1 | 1 | 2.71 | 2.57 | +5.4 % |
| distinct N = 2,000 | 1 | 100.00 | 95.18 | +5.1 % |
| StaticLoop 2 operations | 1 | 4.04 | 3.74 | +7.9 % |
| StaticLoop 10,002 operations | 1 | 11.18 | 10.09 | +10.8 % |
| StaticLoop 19,902 operations | 1 | 35.13 | 31.53 | +11.4 % |
| distinct N = 4,000 (`cubin-sizes`) | about 1, for 22 % of the time | 595.20 | 573.15 | +3.8 % |
| distinct N = 1,000 (`e3_getarg_pr_A.csv`) | 4 | 24.88 | 22.31 | +11.5 % |
| distinct N = 3,000 (`e3_getarg_pr_A.csv`) | 4 | 332.10 | 262.36 | +26.6 % |
| distinct N = 4,000 (`e3_patched_A.csv`) | 4 | 696.55 | 573.15 | +21.5 % |

that is, about +5 to +11 % with one other compile job on the CCD, about +11 to +27 % with four, and the old numbers differ from each other by that amount (N = 4,000: 595 s with about one other job for a fifth of the time, 697 s with four). The idle repeat has a spread of 0.2 to 0.9 %, so these differences are far outside the noise of the repeat. The memory and the code do not depend on the load.

## How the derived files were made (after the window, in this folder's parent repository)

```
python scripts/e3_repeat_table.py results/rtx4090/e3-compile/repeat-c8fcbe8
python scripts/e3_repeat_compare.py results/rtx4090/e3-compile/repeat-c8fcbe8      # needs paper-values-before.csv
python scripts/plot_compiletime.py --csv results/rtx4090/e3-compile/repeat-c8fcbe8 --trees c8fcbe8 --out figures/compiletime_single_r2_c8fcbe8.pdf --provenance <the four e3_repeat*_A / staticloop CSVs> \
       results/rtx4090/e3-compile/e3_limit_B_3847.csv results/rtx4090/e3-compile/e3_limit_B_3848.csv --note "<the conditions>" --note "<the pin>"      # the sidecar has the full text
```

## Not part of this repeat

- The 6,000-operation compile (two time-outs of the earlier sweeps: 3,600 s without the get_arg change and 5,400 s with it, `../e3_compile.csv` and `../e3_patched_A.csv`; no memory was recorded for them) was not compiled; the figure has no time-out marker. The local exponents of the idle set between consecutive lengths (time 1.55 to 2.72, memory 1.08 to 1.78, rising with N) extrapolate to roughly 1,700 to 1,950 s and 27 to 30 GiB at N = 6,000 (a constant exponent of the last interval gives 1,724 s and 27.0 GiB, a quadratic or cubic fit of the logarithms 1,887 to 1,949 s and 27.6 to 29.9 GiB), so a time-out of 5,400 s is not what this set predicts; nothing was run to find out.
- Variant B (float3 parameters), the older trees and the kernel run times are not repeated here; the dotted line at N = 3,848 in the figure is the first failing N of variant B (ptxas refuses the kernel parameter space, 0x8508 B against the limit 0x7ffc B) from the committed sweeps.
