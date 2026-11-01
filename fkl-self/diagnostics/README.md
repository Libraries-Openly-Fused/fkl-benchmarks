# E11: what the compiler says about a wrong chain

Three deliberately wrong chains of operations, one translation unit each. They are **compile only** (never run, not part of the CMake build): each file holds the correct chain,
which must compile with every compiler, and, with `-DE11_WRONG`, the wrong one.

| file | the mistake |
|---|---|
| `e11_type_mismatch.cu` | `Mul<float3>` (float3 to float3) followed by `Add<float>`: the Add receives a float3 |
| `e11_read_in_middle.cu` | Read, Mul, **Read**, Add, Write: a complete Read in the middle of the chain |
| `e11_batch_mismatch.cu` | a batch of 50 images read (`PerThreadRead` of `std::array<Ptr2D, 50>`) and a batch of 60 written |

The chains are written as a user writes them, `fk::executeOperations<fk::TransformDPP<>>(stream, read, ops..., write)`; `fk::Stream` and `fk::TransformDPP<>` take the parallel
architecture of the compiler, so the same source is compiled by nvcc (CUDA), amdclang++ `-x hip` (HIP) and g++/clang++ (the CPU back-end).

`scripts/e11_diagnostics.py` compiles them and tabulates what the compiler says (`python scripts/e11_diagnostics.py --compilers g++-13,clang++-23,/opt/rocm/core-10.1/bin/amdclang++
--label unchanged-c8fcbe8 --out results/rx9060/e11-diagnostics`; for nvcc `--compilers nvcc`): the exit status, the number of lines, errors and notes, whether a `static_assert` fired and its
text, the first error line, whether that line names the two operations involved, whether the diagnostic points to the user's source line, and the time of the compile (the median of
5; with the limit of one error, `-fmax-errors=1` for g++ and `-ferror-limit=1` for the clang based compilers, for the time to the first error). `--include` points to another FKL tree.

Results: `results/rx9060/e11-diagnostics/` (g++ 13, clang++ 23, amdclang++ for gfx1200, and nvcc compiled on the same machine) and `results/rtx4090/e11-diagnostics/`
(nvcc on the RTX 4090 machine, when it exists).
