# E11: what the compilers say about three wrong chains, and a prototype of a compile time check of the chain

Three deliberately wrong chains (`fkl-self/diagnostics/e11_*.cu`; a float3 into an `Add<float>`, a complete Read in the middle, a batch of 50 read and 60 written), written as a user writes them,
`fk::executeOperations<fk::TransformDPP<>>(stream, read, ops..., write)`, compiled with four compilers and `scripts/e11_diagnostics.py`: g++ 13.4.0 and clang++ 23.1.5 (the CPU back-end), amdclang++ 24.0.0 for
gfx1200 (HIP), and nvcc 13.4.92 (CUDA, `-arch sm_89`, host g++ 13.4.0). **The nvcc column was compiled on this machine** (the pip wheel of nvcc that the compile checks of the CUDA targets use, compile only);
the RTX 4090 machine repeats it with the same script. Machine: Ryzen 7 5800X3D. `e11_diagnostics.csv` has one row per (FKL tree, compiler, mistake), `logs/` the full output of the wrong chain of every cell. The correct
version of every chain (no `-DE11_WRONG`) compiles in all 36 cells. Compile only: the three programs are never run (the batch of 50 and 60 has a fourth program that is run, last point of section 1). The wall times (median of 5) are 0.26 to 2.71 s and the same with and without the
limit of one error: all the errors are in the front end, so they say nothing here and are in the CSV only.

## 1. The library at FKL c8fcbe8 (the pin until 2026-11-07)

| mistake | g++ 13.4 | clang++ 23.1.5 | amdclang++ 24.0.0 (gfx1200) | nvcc 13.4.92 |
|---|---|---|---|---|
| float3 into an `Add<float>` | 44 lines, 1 error | 58 lines, 1 error | 46 lines, 1 error | 24 lines, 1 error |
| a complete Read in the middle | 88 lines, 3 errors | 52 lines, 1 error | 46 lines, 1 error | 27 lines, 1 error |
| batch of 50 read, 60 written | **compiles**, no error | **compiles**, no error | **compiles**, no error | **compiles**, no error (only the CUDA 13 deprecation warnings that the correct chain gets too) |

- **None of the three mistakes meets a `static_assert`.** The compiler stops in the middle of the library, `instantiable_operations.h:142` (the call `Operation::exec(input.input, opData)`) or `data_parallel_patterns.h:79` (the fold
  `(InputFoldType<>::build(thread, i_data) | ... | iOpInstances)`), after an instantiation backtrace of 10 to 15 levels that prints the whole list of operations of the chain.
- float3 into `Add<float>`: the error line is "no matching function for call to `fk::Add<float>::exec(const fk::Float3&, ...)`" with g++ (and "no instance of overloaded function `fk::Add<I, P, O, fk::BinaryType>::exec`" with nvcc), which names the next operation and
  the type it got; with clang++ and amdclang++ it is "no matching function for call to 'exec'", which names neither (the candidates are in the notes). **No error line names the previous operation.**
- The complete Read in the middle: "no match for `operator|` (operand types are `fk::InputFoldType<float>` and `const fk::ReadInstantiableOperation<fk::PerThreadRead<...>>`)" (g++, 88 lines because two knock-on errors
  "invalid use of incomplete type `fk::VectorTraits<...>`" follow), "invalid operands to binary expression" (clang++, amdclang++), "no operator "|" matches these operands" (nvcc): it names the Read that does not belong, and not the rule.
- **The batch of 50 read and 60 written compiles with all four compilers**, without a word, and nothing is reported when it runs (exit code 0). **The chain processes the 50 images of the read and leaves the last 10 outputs (50 to 59) untouched, with the CPU back-end
  (g++ 13, one thread), with the multithreaded CPU back-end (g++ 13) and with the HIP back-end (amdclang++ 24.0.0, RX 9060, gfx1200). The CUDA back-end was not run** (this machine has no NVIDIA GPU): `e11_batch_mismatch_run.cu` compiles with nvcc 13.4.92,
  and on a machine that has one `BACKENDS="cpu cpu_mt cuda" NVCC=nvcc batch-run/run.sh` (from the root of this repository) builds and runs it. The program (FKL c8fcbe8, `fkl-self/diagnostics/e11_batch_mismatch_run.cu`) reads 50 images of 8 x 4 floats filled with 1, the 60 output images are filled with -1 and the chain multiplies by 2; it classifies each
  output image by all of its 32 pixels: 50 images are all 2, 10 are all -1 (the first is output 50), none is mixed, in three runs of each back-end (`batch-run/output.txt`). The opposite, 60 read and 50 written, would index past the 50 outputs (not run: undefined behaviour).
- The library does have `static_assert`s for two of these mistakes, in `Fuser::fuse` (`iop_fuser.h:28`, "Fusing two batch operations of different BATCH size is not allowed.", and `iop_fuser.h:99`, "Complete Read Operations as continuations are not
  allowed. It has to be an IncompleteReadBackOperation."), but only the composition of operations with `operator&` reaches them, and `executeOperations(stream, ...)` does not use it. Checked for the Read in the middle written as
  `read & mul & read2 & add & write`: g++ stops at `iop_fuser.h:99` with that message. (`BatchWrite` also has no `BATCH` member, so `isBatchOperation` is false for it and the batch size check of the Fuser could not see it.)

## 2. The prototype

A check of the chain in the executors of `TransformDPP` (the four of them: CPU, CPU_MULTITHREAD, CUDA and HIP), before anything of the chain is instantiated. **It is a prototype: a local branch of FKL (`LTS-C++17-chain-check`, two commits on c8fcbe8), not
pushed, not a pull request; a change to the library is for the maintainers to decide.** The two patches are here (`git apply` in a checkout of c8fcbe8):
`chain-check-types-only.patch` (commit 2d10e5c: the output of each operation converts to the input of the next one) and `chain-check-prototype.patch` (d8d00bf: also only the first operation can be a complete Read, and the batch operations of the chain have the BATCH of the first).

The mechanism (`include/fused_kernel/core/execution_model/operation_model/chain_check.h`, 132 lines, and 37 in `executors.h`): `chainIsValid<IOps...>` is a plain `bool` built from `std::is_convertible_v<OutputType of IOps[i], InputType of IOps[i + 1]>` for the pairs
that have both types (a Read has no input, a Write no output), from `isAnyCompleteReadType` of the IOps after the first, and from the BATCH of the batch operations; the four `executeOperations_helper` of `Executor<TransformDPP<...>>` run
`if constexpr (chainIsValid<IOps...>)` the old body, and otherwise instantiate `reportChain<IOps...>`, one `static_assert` per wrong pair, whose instantiation context names the two operations ("In instantiation of `ChainLinkReport<Mul<Float3>, Add<float>>`").
The `if constexpr` is what keeps the output to one error: a first version with the `static_assert` in `TransformDPP::build_details` left the raw error as a second one with clang. The message of the type check is "FKL: the OutputType of the previous Op is not convertible to the InputType of the next Op";
the other two are the words of the Fuser's. Results, same cells:

| mistake | g++ 13.4 | clang++ 23.1.5 | amdclang++ 24.0.0 (gfx1200) | nvcc 13.4.92 |
|---|---|---|---|---|
| float3 into an `Add<float>` | 18 lines, 1 error (static_assert) | 40 lines, 1 error (static_assert) | 40 lines, 1 error (static_assert) | 16 lines, 1 error (static_assert) |
| a complete Read in the middle | 16 lines, 1 error (static_assert) | 34 lines, 1 error (static_assert) | 34 lines, 1 error (static_assert) | 14 lines, 1 error (static_assert) |
| batch of 50 read, 60 written | 16 lines, 1 error (static_assert) | 34 lines, 1 error (static_assert) | 34 lines, 1 error (static_assert) | 14 lines, 1 error (static_assert) |

With only the first commit (the type check), the float3 into `Add<float>` gives 17 (g++), 37 (clang++), 37 (amdclang++) and 15 (nvcc) lines, one error each, and the other two chains give what they give in section 1 (the lines differ by at most two).

The first error line, before and after:
- float3 into `Add<float>`: g++ "`instantiable_operations.h:142:72: error: no matching function for call to 'fk::Add<float>::exec(const fk::Float3&, const fk::OperationData<fk::Add<float>, void>&)'`" and clang++/amdclang++ "`error: no matching function for call to 'exec'`", after:
  "`chain_check.h:61:57: error: static assertion failed: FKL: the OutputType of the previous Op is not convertible to the InputType of the next Op`" (g++; the line above it is "In instantiation of `struct fk::ChainLinkReport<fk::BinaryInstantiableOperation<fk::Mul<fk::Float3> >, fk::BinaryInstantiableOperation<fk::Add<float> > >`",
  nvcc has the same in "detected during: instantiation of class `fk::ChainLinkReport<PrevIOp, NextIOp> [with PrevIOp=...Mul<float3, float3, float3>..., NextIOp=...Add<float, float, float, fk::BinaryType>...]`"), and "`error: static assertion failed due to requirement 'fk::ChainLinkTypes<fk::BinaryInstantiableOperation<fk::Mul<...>>, fk::BinaryInstantiableOperation<fk::Add<...>>, void>::value': FKL: the OutputType of the previous Op is not convertible to the InputType of the next Op`" (clang++, amdclang++: both operations are in the error line).
- Read in the middle: "`FKL: Complete Read Operations as continuations are not allowed. It has to be an IncompleteReadBackOperation.`" (clang based: "due to requirement `!isAnyCompleteReadType<fk::ReadInstantiableOperation<fk::PerThreadRead<...>>>`", which names the Read).
- Batch: "`FKL: Fusing two batch operations of different BATCH size is not allowed.`" (clang based: "due to requirement `chainBatchSizeMatches<fk::ReadInstantiableOperation<fk::BatchRead<..., 50, ...>>, fk::WriteInstantiableOperation<fk::BatchWrite<60, ...>>>`", which names both with their sizes).

## 3. What it costs and what it breaks

Existing code, with the prototype (d8d00bf; `verification/`): the FKL test suite built for HIP (gfx1200), CPU and CPU_MULTITHREAD (84 targets) and run: **84 of 84 pass**, as at c8fcbe8; all **46 CUDA targets of FKL compile** with nvcc 13.4.92 (sm_89, host g++ 13.4.0, compile only: there is no NVIDIA GPU here to run them);
`tests/examples/compiler_explorer_example.h` builds and passes (`_cpp`, `_hip`) at c8fcbe8 and with the prototype; of the 23 targets of fkl-benchmarks (`_cu` with the same nvcc, `_hip`, `_cpp`, OpenCV CPU) **22 build** (five of them, `benchmark_nv12_pipeline_cu`, `_hip`, `_cpp`, `e5_preprocessing_b_cpp` and `nan_saturate_cu`, were removed later), and the 23rd, `e6_saturate_exhaustive_cu` (also removed later, with the numerics code: `git show 5911330:fkl-self/numerics/saturate_exhaustive.cu`), needs the OpenCV CUDA header
`opencv2/core/cuda/saturate_cast.hpp`, which this machine's OpenCV does not have (a missing dependency, nothing in FKL). Nothing that compiled stops compiling. No warning comes from the new header.

Compile time of the valid chains of E3 (variant A, N distinct operations; `scripts/e3_clang_limits.py` of fkl-benchmarks pull request #10, one compile at a time, median of 3, alternating the two trees, stack limit 16 MiB; amdclang++ without the resource usage remark):

| compiler | N | unchanged | prototype | difference | peak memory, unchanged | prototype |
|---|--:|--:|--:|--:|--:|--:|
| g++ 13 | 1,000 | 8.8 s | 8.8 s | +0.0 % | 0.54 GiB | 0.54 GiB |
| g++ 13 | 2,046 | 28.5 s | 28.8 s | +1.1 % | 2.15 GiB | 2.15 GiB |
| clang++ 23 | 1,000 | 9.8 s | 10.0 s | +2.0 % | 1.83 GiB | 1.83 GiB |
| clang++ 23 | 2,046 | 32.0 s | 31.8 s | -0.6 % | 6.86 GiB | 6.87 GiB |
| amdclang++ | 1,000 | 20.5 s | 20.8 s | +1.5 % | 1.83 GiB | 1.84 GiB |
| amdclang++ | 2,046 | 69.8 s | 70.3 s | +0.7 % | 6.85 GiB | 6.87 GiB |

The limits of the chains do not move (`compile-time-overhead/e3_limits.csv`):

| compiler | N | unchanged | prototype |
|---|--:|---|---|
| g++ 13 | 2,047 | compiles (28.8 s, 2.1 GiB) | compiles (28.8 s, 2.1 GiB) |
| clang++ 23 | 2,047 | `iop_fuser.h:150:17: fatal error: instantiating fold expression with 2049 arguments exceeded expression nesting limit of 2048` | `iop_fuser.h:150:17: fatal error: instantiating fold expression with 2049 arguments exceeded expression nesting limit of 2048` |
| amdclang++ | 2,047 | `iop_fuser.h:150:17: fatal error: instantiating fold expression with 2049 arguments exceeded expression nesting limit of 2048` | `iop_fuser.h:150:17: fatal error: instantiating fold expression with 2049 arguments exceeded expression nesting limit of 2048` |
| g++ 13 | 4,000 | compiles (114.4 s, 6.5 GiB) | compiles (115.4 s, 6.5 GiB) |

## Reproduce

`python scripts/e11_diagnostics.py --compilers g++-13,clang++-23,/opt/rocm/core-10.1/bin/amdclang++ --label <tree> --include <FKL include dir> --out results/rx9060/e11-diagnostics` and, for nvcc,
`--compilers nvcc --ccbin /usr/bin/g++-13 --note "..."`; the include directories were a checkout of c8fcbe8 and the two prototype commits (`git apply` the patches of this folder to c8fcbe8 for the same trees).
`python scripts/e11_diagnostics.py --reanalyse results/rx9060/e11-diagnostics` recomputes the analysis columns of the CSV (lines, errors, notes, static_assert, first_error, the three `names_*` columns and
`points_to_user_line`) from `logs/` without compiling anything; the other columns (date, versions, exit status, wall times) stay as they are.

## Correction of the analysis columns (after the review of this pull request)

The first version of `scripts/e11_diagnostics.py` read the output of nvcc as if it were that of g++ or clang: it did not match nvcc's numbered notes (`note #3326-D:`), it took no instantiation context for nvcc
(`detected during:` and then `instantiation of ...`, innermost first), it did not recognise nvcc's locations (`at line 33 of e11_type_mismatch.cu`), and it cut the `static_assert` and `first_error` cells at 300 characters.
The analysis columns of `e11_diagnostics.csv` were recomputed from `logs/` with the corrected script (`--reanalyse`); 29 cells changed, in 15 of the 36 rows:
- `notes` of nvcc: 0 -> 2, 3 and 3 (CSV lines 11, 12 and 24).
- `points_to_user_line` of nvcc: no -> yes (lines 11, 12, 23, 24, 35, 36 and 37): each of the seven nvcc logs with an error ends its `detected during` list at a line of the translation unit.
- `names_both_in_headline` of nvcc: no -> yes (lines 23, 35 and 37): the innermost context of nvcc, `instantiation of class "fk::ChainLinkReport<PrevIOp, NextIOp> [with PrevIOp=...Mul..., NextIOp=...Add...]"` (the batch
  chain: `ChainBatchReport`), names the pair, as the instantiation context of g++ does.
- `static_assert` and `first_error` of 8 rows of clang++ and amdclang++ (lines 17, 20 and 29 to 34) are no longer cut: 325 to 430 characters instead of 300.
`lines`, `errors`, `names_previous_op_in_error_line`, `names_next_op_in_error_line` and every column of the g++ and clang based rows other than those two text cells are the same in all 36 rows, and the tables of this README
(lines and errors) do not change. The earlier cells are in `e11_diagnostics.csv` at c860e14 (`git show c860e14:results/rx9060/e11-diagnostics/e11_diagnostics.csv`).

The nvcc rows that the RTX 4090 machine adds must be made with this version of the script: for nvcc's own diagnostics (its `note #...-D` notes, its `detected during` list and its `at line N of ...` locations)
the first version gives `notes` 0 (only the notes of the host compiler were counted), `points_to_user_line` no and `names_both_in_headline` no, whatever the log says.
`names_both_in_headline` says something about the compiler only where the context line is the pair of operations (the instantiation context of g++ and of nvcc, the requirement text of clang). For clang++ and amdclang++ on the
complete Read in the middle (the unchanged library, and the prototype of the type check only, which does not check the Read) the first note prints the template arguments of the whole chain, every operation in it, so the
cell is `yes` without the compiler having named the offending pair there; the rule that a class context of nvcc keeps its `[with ...]` works because the innermost class of the nvcc contexts is the class of the pair. For what
a compiler tells the user use the two error-line columns. `notes` counts different things per compiler (nvcc: only the `note #...-D` candidate notes, not the `detected during` list; clang: also every `in instantiation of ...
requested here`; the 25 of the nvcc rows of the batch chain are notes of the host g++) and is not for comparing compilers; `--reanalyse` does not recompute `outcome`, which comes from the exit status.
