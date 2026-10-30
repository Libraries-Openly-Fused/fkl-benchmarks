# E3: compile time, resources and the limits of one fused kernel

Everything here is **compile only**: nothing in this folder or in `scripts/e3_*.py` executes a program on the GPU (the sweeps run while the E0
experiments are measuring; see the top-level README). The programs that the generator writes do contain a `main()` that runs the chain; they are
executed later, inside a paused window, only for the kernel-time rows.

## Files

- `gen_chain.py <A|B|L> <N> <out.cu> [old|new]`: one translation unit with N operations in ONE fused kernel, each parameterised operation with its own
  parameter value.
  - A: float, Mul/Add/Sub/Div/Max/Min cycled. B: float3 with `VectorReorder` and `Cast<float3, int3>`/`Cast<int3, float3>` (type changes).
  - L: Mul/Add on float written in the API of both the old FKL trees (`.cuh` headers, 1f33224) and the newer ones (`style` old/new), for the
    fold-versus-recursion comparison.
- `../../scripts/e3_sweep.py`: compiles the generated TUs (nvcc 13.4, `-O3 -DNDEBUG -std=c++17 -arch sm_89 -ftemplate-depth=1000 -lineinfo --threads 1`,
  pinned to one core) and the same source as plain C++ with g++ (the CPU back-end). Records wall time, peak RSS, the nvcc phases (`--time`), ptxas
  registers/spills/stack/parameter block, SASS instructions, `.text` and cubin bytes; stops a variant at its first failure and records the first error.
  `--include <fkl include dir> --tree <label>` compiles against another FKL tree (git archive of a commit).
- `../../scripts/e3_staticloop.py`: the same measurements for cvGPUSpeedup's `mul_add1` translation units (2 to 19,902 operations, StaticLoop), with the
  compile line of the benchmark.
- `repro/param_cliff.py`: minimal reproducers without FKL (see below). `repro/get_arg_patch.diff`: the one-function change that removes the cliff.

## The cliff at 1,024 operations (nvcc 13.4.92, sm_89)

A kernel generated for N distinct operations (variant A) compiles to 17 registers and no spills up to **N = 1,023**, and to 255 registers with
thousands of local-memory instructions from **N = 1,024**, with the kernel parameter block of the same size on both sides (4,504 B). The kernel is
`launchTransformDPP_Kernel(const __grid_constant__ TDPPDetails, const __grid_constant__ IOps... operations)`: every operation is its own
`__grid_constant__` kernel parameter.

PTX of the chain kernel: N = 1,020: `ld.param` only. N = 1,023: the generic address of every parameter is taken (`cvta.param`, 1,025 of them) and kept in
registers. N = 1,024: those addresses are stored into `.local .align 8 .b8 __local_depot0[8200]` (`st.local.u64` x 1,025) and reloaded; N = 1,030: depot
of 8,248 B. SASS: no `STL`, `LDL` or `LDC` up to N = 1,023; from N = 1,024 about half of the instructions are address arithmetic and local memory.

Cause: `fk::get_arg<I>(args...)` (`core/utils/parameter_pack_utils.h`, used by `ppFirst`, `ppLast` and `get_arg<0>(iOps...)` in
`data_parallel_patterns.h`) is `TupleUtil::get<I>(forward_as_tuple(args...))`: each call builds a `Tuple<Args&&...>`, one pointer per argument, and
extracts one element. The optimiser removes it for small packs; for a pack of this size it keeps it as a local array of pointers.

Reproducers (`repro/param_cliff.py`, results in `results/rtx4090/e3-compile/repro/`):

- A struct parameter of 4,000 / 4,096 / 4,200 / 8,192 B in four forms (by value; `const __grid_constant__`; `const __grid_constant__` read through a
  `const&` passed to a `__device__` function; the same bytes in two `__grid_constant__` parameters), and K = 1,000 to 1,500 parameters of one float or one
  4-byte struct: **all clean** (8 registers, no spills, plain `ld.param`). Neither the size of the parameter block nor the number of parameters is
  a problem for nvcc 13.4 by itself.
- K `__grid_constant__` floats read only through the library's `fk::get_arg<0>` and `fk::get_arg<last>`: clean up to K = 1,024, `cvta.param` for
  every parameter from K = 1,025, 255 registers and 8,300 B of spills at K = 1,500 (18,572 B at K = 3,000). The local array appears at a pack of
  1,025 pointers in the chain; in this smaller reproducer it appears somewhere between K = 1,030 and K = 1,500.

`repro/get_arg_patch.diff` replaces `get_arg` by an implementation without the tuple (skip the first I arguments with I leading empty
converting-constructor parameters). With it: the reproducer is clean for every K up to 3,000, and the chains of variant A at N = 1,024, 1,030 and 1,500
compile to 17 registers and no spills. The patch was tested on a copy of the include tree; it is **not** applied to the library (a change to the
library is for the maintainers to decide).

## Old trees (fold versus recursion)

`git archive` of 1f33224 (the tree of the old Figure 26), 07b2cb3^ (last tree before the fold expansion) and 07b2cb3 (the fold): the same Mul/Add
chains (variant L) fail at N = 248, 245 and 245 with `error: cannot deduce the return type of function "fk::get<Index, ...>"` at
`parameter_pack_utils.h(34)`: a recursive `auto` function that exceeds the front end's nested-instantiation depth (about 256 frames; not
`-ftemplate-depth`). The current tree compiles 3,000 operations of the same chains.

## Running

```
source scripts/env-cuda13.4.sh
python scripts/e3_sweep.py --variants A,B --ns 4,14,54,100,200,300,500,750,1000,1500,2000,3000,4000 --core 8 --host --out results/rtx4090/e3-compile
python scripts/e3_staticloop.py --build build/<preset> --core 10 --out results/rtx4090/e3-compile
python fkl-self/compiletime/repro/param_cliff.py --out results/rtx4090/e3-compile/repro --fkl-include external/fkl/include
```
Use cores that no running experiment uses and keep the SMT sibling of the compile core idle. The nvcc phase times of one compile are in the `--time`
file (it is appended to, not overwritten: delete it before each compile).
