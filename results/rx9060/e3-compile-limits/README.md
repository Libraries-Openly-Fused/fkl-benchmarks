# E3 on the AMD side: where a fused chain stops compiling (amdclang++ 24.0.0 for gfx1200, clang++ 23.1.5, g++ 13.4.0), FKL c8fcbe8

The chain of N distinct operations in ONE fused kernel of `fkl-self/compiletime/gen_chain.py` (variant A: float, `Mul`, `Add`, `Sub`, `Div`, `Max`, `Min` cycled, each operation with its own
parameter, one translation unit per N; the source of the RTX 4090 sweep in `../../rtx4090/e3-compile/`) compiled with the three compilers of this machine: **amdclang++** (ROCm 10.1, `-x hip
--offload-arch=gfx1200`), **clang++ 23** and **g++ 13** (the CPU back-end of the same source, `-DE3_CPU`). Flags as in `scripts/e3_sweep.py`: `-O3 -DNDEBUG -std=c++17 -ftemplate-depth=1000`
(CPU: also `-march=native`). FKL `c8fcbe8` (the final pin of LTS-C++17, with #417), fkl-benchmarks at the commit of this folder. Ryzen 7 5800X3D, 60 GiB, governor `powersave`; **one compile at a time**
on an idle machine (the script waits while a CI job runs), not pinned (the compiles are single threaded: user + system time is 97 to 100 percent of the wall time in the CSV), peak resident memory from GNU time (`peak_rss_gb`, the largest
process) and of the whole process tree sampled at 4 Hz (`peak_tree_rss_gb`; the two agree within 5 percent), cap 40 GB or 1 h per compile: **no compile was cut** (the largest, amdclang++ at N = 4,000 with the resource usage remark, took
271.9 s and 35.9 GB). The programs are executed only in section 4.

Files: `e3_limits.csv` (one row per compile: 96 rows), `e3_check.csv` (section 4), `logs/` (the first 400 lines of the output of every compile that did not succeed, gzip; an empty one is a compiler that was
killed without saying a word), `run_series.sh` (the commands, by section), `../../../scripts/e3_clang_limits.py` (the tool: generates the translation unit, compiles it under the monitor, classifies the outcome).

## Result

| | amdclang++ 24.0.0 (HIP, gfx1200) | clang++ 23.1.5 (CPU back-end) | g++ 13.4.0 (CPU back-end) |
|---|---|---|---|
| **Default flags** (stack limit 8 MiB): largest N that always compiles | 1,366 | 1,515 | 4,000 (not tried above) |
| first N that always fails | 1,368 | 1,517 | none |
| what happens | the compiler **crashes** (segmentation fault in code generation, crash dump, exit status 254) | the compiler is **killed by signal 11** (SIGSEGV), without a word in 19 of 21 runs | |
| **Stack limit raised** (16 MiB): largest N that compiles | 2,046 | 2,046 | |
| first N that fails | 2,047: `fatal error: instantiating fold expression with 2049 arguments exceeded expression nesting limit of 2048` at `iop_fuser.h:150:17`, after 1.0 s | the same, after 0.5 s | none |
| `-fbracket-depth` that lifts it | N + 2 (N = 2,100: 2,100 and 2,101 fail with the same error naming that limit, 2,102 compiles) | N + 2 (the same) | has no such limit |
| N = 4,000 (`-fbracket-depth=4002`, stack limit 32 MiB) | compiles: 226.1 s, 25.1 GB (271.9 s, 35.9 GB with the resource usage remark) | compiles: 105.9 s, 25.1 GB | compiles (defaults): 114.6 s, 6.5 GB |

The kernels that compile are all alike: 54 SGPRs, 9 VGPRs, no scratch, no spills, occupancy 16 at N = 1,100, 2,046, 2,100 and 4,000 (section 3), the one with N = 4,000 operations (kernel argument block of 16,312 B) launches on the RX 9060 and
gives the bit-identical result of the CPU back-end (section 4). Up to N = 4,000, the two limits that stop a chain are both in the compiler: the stack of the compiler process and the nesting limit of a fold expression.

## 1. The stack of the compiler (8 MiB by default): N = 1,368 with amdclang++, 1,517 with clang++

`bisect --no-aslr` (address space randomisation off, so that the stack overflows at the same N every time) between 1,100 (compiles) and 2,046 (crashes): 12 compiles for amdclang++, 12 for clang++; the largest N that compiles is
1,367 (amdclang++) and 1,516 (clang++), the first that does not 1,368 and 1,517. With randomisation on (the normal case) the boundary moves by about one operation from run to run, and the compile fails a little earlier than without it; five compiles at each N around it:

| N | 1,366 | 1,367 | 1,368 | 1,369 | | N | 1,515 | 1,516 | 1,517 | 1,518 |
|---|--:|--:|--:|--:|---|---|--:|--:|--:|--:|
| amdclang++ compiles | 5 of 5 | 1 of 5 | 0 of 5 | 0 of 5 | | clang++ compiles | 5 of 5 | 3 of 5 | 0 of 5 | 0 of 5 |

It is the stack and not memory: the compile that crashes has used 3.2 to 3.9 GB of 60; 8 MiB divided by the first failing N is 6.0 KiB (amdclang++) and 5.4 KiB (clang++) of stack per operation of the chain; N = 2,046 needs between 10 and 12 MiB with both
compilers (`stack-needed-2046` in the CSV, randomisation off: 10 MiB crash / 12 MiB ok / 14 MiB ok with amdclang++, crash / ok / ok with clang++); 16 MiB compiles it,
and 32 MiB compiles N = 2,100 and 4,000 (`ulimit -s 32768` before the compiler, or `ulimit -s unlimited`).

What the user sees. amdclang++: `PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace and dumped files.`, a stack dump whose third item is
`3. fused_kernel/core/execution_model/data_parallel_patterns.h:78:34: Generating code for declaration 'fk::TransformDPPBase<fk::TF::DISABLED, fk::TransformDPPDetails_<void, false, fk::ReadInstantiableOperation<...>, ...>>::operate'`
(the function whose code was being generated; its template arguments list every operation of the chain, so that this one line is 67 kB long at N = 1,368), 256 frames in which `EmitCallExpr`, `EmitCall`, `EmitCallArgs`, `EmitCallArg`, `EmitAnyExprToTemp`, `EmitAnyExpr` and `EmitAggExpr` repeat about 24 times each
(the code generator recursing through the N nested calls of the fold `(InputFoldType<>::build(thread, i_data) | ... | iOpInstances)` at `data_parallel_patterns.h:79`), then
`clang++: error: unable to execute command: Segmentation fault (core dumped)` and `clang++: error: clang frontend command failed with exit code -2`; it also writes a crash reproducer of about 12 MB to the temporary directory.
clang++ 23: nothing at all in 19 of the 21 crashes; the other two (both at N = 1,516, the last N that sometimes compiles) printed before dying `data_parallel_patterns.h:79:44: warning: stack nearly exhausted; compilation time may suffer, and crashes due to stack overflow are likely [-Wstack-exhausted]`.

## 2. The nesting limit of the fold expression: N = 2,047

With the stack raised, the first N that fails is **2,047** with both clang based compilers, in the front end, after 1.0 s (amdclang++) and 0.5 s (clang++):

`fused_kernel/core/execution_model/operation_model/iop_fuser.h:150:17: fatal error: instantiating fold expression with 2049 arguments exceeded expression nesting limit of 2048`

The fold is the comma fold of `Fuser::idxFirstNonBack<IOps...>()` (`((index++, ...), ...)` over the operations), instantiated with the N + 2 operations of the chain (the read, the N operations and the write): 2,049 at N = 2,047. The limit is
`-fbracket-depth` (default 2048): at N = 2,100 it fails with the limit written as 2,100 and 2,101 and compiles with `-fbracket-depth=2102` (`bracket-depth` in the CSV), with both compilers; the flag has to follow the length of the chain (N + 2).
g++ 13 has no such limit: N = 2,047 and 2,100 compile with the defaults, as does 4,000.

## 3. What a chain costs to compile

| N | amdclang++ (wall, peak RSS) | clang++ | g++ |
|--:|---|---|---|
| 1,100 | 27.5 s, 3.0 GB (plain: 24.0 s, 2.2 GB) | 11.8 s, 2.2 GB | 10.3 s, 0.8 GB |
| 2,046 | 81.6 s, 9.7 GB (stack 16 MiB; plain: 70.1 s, 6.8 GB) | 33.3 s, 6.9 GB (stack 16 MiB) | 28.5 s, 2.1 GB |
| 2,100 | 82.1 s, 10.2 GB (`-fbracket-depth=2102`, stack 32 MiB) | 33.3 s, 7.2 GB (the same) | 30.0 s, 2.2 GB |
| 4,000 | 271.9 s, 35.9 GB (`-fbracket-depth=4002`, stack 32 MiB; plain: 226.1 s, 25.1 GB) | 105.9 s, 25.1 GB (the same) | 114.6 s, 6.5 GB |

From 2,046 to 4,000 (1.95 times the operations) the time grows as N^1.7 (amdclang++, plain), N^1.7 (clang++) and N^2.1 (g++), and the memory as N^1.9, N^1.9 and N^1.7.
At the same N, amdclang++ needs 2.9 to 3.9 times the memory of g++ and 2.0 to 2.5 times the time (it compiles the source for the device and for the host),
clang++ 2.8 to 3.9 times the memory and 0.9 to 1.2 times the time.

**The resource usage remark of amdclang++ has a price.** The amdclang++ rows above ask for the resource usage of the kernel (`-Rpass-analysis=kernel-resource-usage`); without it (`hip-no-remarks` in the CSV, the compile that a user
does) the same compiles take 24.0 s and 2.17 GB (N = 1,100), 70.1 s and 6.85 GB (N = 2,046) and 226.1 s and 25.13 GB (N = 4,000): the remark adds 15 to 20 percent of the time and 38 to 43 percent of the memory. The exponents and the ratios to g++ above are those of the plain compiles of amdclang++; the plain compile is the one to quote as the cost of a chain.

The fused kernel (`launchTransformDPP_Kernel`, resource usage remark of the compiler) is the same at every N that compiled with amdclang++: 54 SGPRs, 9 VGPRs, scratch 0 bytes per lane, occupancy 16 waves per SIMD, no spills of either kind
(N = 1,100, 2,046, 2,100 and 4,000; there is no register cliff as in the RTX 4090 sweep before #417). The object file grows from 0.3 MB (N = 1,100) to 0.8 MB (N = 4,000).

## 4. Do the kernels launch, and do they compute the same?

The kernel argument block holds the parameter of every operation, 4 B each, plus a fixed part of 312 B (712 B at N = 100, 4,712 B at N = 1,100, 16,312 B at N = 4,000), so it is larger than 4 KiB from N = 1,000. The check builds the program of
the same translation unit with `-DE3_CHECK` (64 x 64 floats of a deterministic pattern through the chain, written to a file) for the GPU with amdclang++ and for the CPU back-end with g++, runs both and compares (`scripts/e3_check_compare.py`).
The chain of the check is variant A with `Mul` and `Add` in place of `Max` and `Min` (those clamp every value after the first operations and the output would not depend on most of the chain) and parameters `1 + 0.0001 ((i mod 13) + 1)`
for `Mul`/`Div` and `1 + 0.001 ((i mod 13) + 1)` for `Add`/`Sub` (the parameters of variant A make the product of the `Mul` and `Div` overflow from N = 2,900); `-ffp-contract=off` on both sides.

| N | kernel argument block (B) | build, amdclang++ (wall) | GPU program exit | CPU program exit | values | bit-identical | max relative difference | range of the output |
|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 100 | 712 | 2.5 s | 0 | 0 | 4096 | 4096 | 0 | 16.1915, 63.9735 |
| 1,100 | 4,712 | 24.3 s | 0 | 0 | 4096 | 4096 | 0 | 197.518, 251.233 |
| 2,046 | 8,496 | 70.1 s | 0 | 0 | 4096 | 4096 | 0 | 387.679, 447.68 |
| 3,000 | 12,312 | 131.9 s | 0 | 0 | 4096 | 4096 | 0 | 602.913, 670.008 |
| 4,000 | 16,312 | 227.1 s | 0 | 0 | 4096 | 4096 | 0 | 853.754, 929.051 |

All sizes launch on the RX 9060 (gfx1200, ROCm 10.1) and give exactly the output of the CPU back-end, the kernel with 16,312 B of arguments included: no limit of the kernel argument block was reached up to N = 4,000
(a larger N was not tried: N = 4,000 already takes 25 to 36 GB of the 40 GB cap). The N = 1,100 and larger programs run the whole chain once on 4,096 elements; this is a correctness check of the launch, not a timing.

Which number proves the equality: the text that `scripts/e3_check_compare.py` prints, which is the column `compare` of `e3_check.csv` and the columns `values` and `bit-identical` above. It counts the values whose 32 bits are equal (4,096 of 4,096 in every row), counts the non-finite values of each
file (0 in both files in every row: no NaN and no infinity) and prints the largest differences (0). The column `compare_exit` of the five committed rows was written by the first version of the script, whose exit status was 0 whenever the two files had the same number of non-finite values, that is for any
two outputs without a NaN or an infinity, equal or not, so a 0 there does not by itself mean that the outputs are equal; the script now exits 0 only when the two files are byte for byte identical, and run again at N = 100 and N = 1,100 with the settings of the table it prints the same text and exits 0. The claim
is limited to what was run: five lengths of the check chain, one deterministic input pattern, one output of 4,096 values per length.

## What this does not cover

nvcc (RTX 4090, `../../rtx4090/e3-compile/`: N = 4,000 in 675 s and 13.6 GB, the predicted limit of the parameter space for this variant at N = 8,177); variants B and L; the execution time of these kernels; N above 4,000 (the limit of the kernel argument block of gfx1200 was not reached).

## Reproduce

`./run_series.sh [section...]` in this folder (about 1 hour of compiling for all nine sections; each compile is one `scripts/e3_clang_limits.py run|bisect|check` call and appends to the CSVs). The console log of the first run was lost to a bug of the first version of the driver
(`tee /dev/stderr` truncated it); the CSV was not affected, and the driver here is that version with the bug removed and sections 6 to 8, which were run afterwards with the same commands, added.
