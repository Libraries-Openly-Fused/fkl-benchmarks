# Host code of the E19 programs at FKL 24478f0 and at b10fc66

**The question.** The paper's E19 factors are whole-call times. So does the pin move change anything inside the timed interval of the
fkl_batch row, not only its kernel?

**What is compared.** The x86-64 host code of `e19_batched_box_hip` and `e19_crop_sweep_hip`, built at a8f9daa with FKL 24478f0 (A) and at
2160c70 with FKL b10fc66 (B), from the builds of `README.md`. It is compared function by function, and each row's timed call is followed
through the functions it calls. Compile products only: nothing was run.

## Answer: nothing that the fkl_batch row's timed call executes differs

In both programs, and in both comparison modes, no function reached from the fkl_batch row's timed call differs between A and B. The same
holds for the rpp row and for the timing harness. The functions that differ are listed below. In both programs they are:
- fkl_loop's own code;
- the host stub of fkl_loop's kernel, which is never called;
- in the stricter mode only, the function that writes the FKL commit into the CSV header before the timing.

## What runs inside a timed call

- **whole_call:** `fklb::timeIt<std::function<void ()>>` records the start event, calls `row.call()`, and records the stop event. The call
  is indirect, through the `std::function`, to the row's `_M_invoke`.
- **kernel_only:** `fklb::timePreparedLaunch` runs the `gpuSpin` launch as `beforeLaunch`, then `row.call()`.
  - In `e19_batched_box_hip` it is inlined into `launch()`.
  - In `e19_crop_sweep_hip` it is 16 functions, one per number of crops.
- **The rows' calls.** Each row's call is a lambda of `launch()` (of `e19amd::runCrops<N>()` in the sweep). The lambdas were identified by what
  they call:

| row | lambda | what it calls |
|---|---|---|
| fkl_loop | `{lambda()#1}` | `fk::executeLinearFilter<...>` |
| fkl_batch | `{lambda()#3}` | `fk::executeMorphology<MorphologyDPPDetails<float, 16, 8, 7, 7>, ReadInstantiableOperation<BatchRead<..., 30, ...>>, ...>`, which launches through `hipLaunchKernel` |
| rpp | `{lambda()#5}` | `rppt_color_to_greyscale`, `rppt_box_filter` and `rppt_crop_mirror_normalize` |

## Method (`step0/host_compare.py`)

- **Disassembly:** `objdump -d -C --no-show-raw-insn -j .text` (GNU objdump 2.46), one function per symbol. The other code sections are
  `.init` and `.fini` (the startup stubs) and `.plt`, which has the same size in A and B (0x780 bytes).
- **Normalisation, the requested form (`offsets`):**
  - the addresses at the start of the lines are removed;
  - a branch target inside the same function becomes a local label;
  - a call or jump to another function becomes its name, with `+OFF` if it lands inside it;
  - RIP-relative displacements become `X(%rip)`, and objdump's symbol offsets become `+OFF`;
  - the per-translation-unit names `__hip_gpubin_handle_<hash>` and `__hip_cuid_<hash>` lose their hash;
  - the padding after a function's last instruction is not compared.
- **The stricter form (`rodata`, `--rodata-bytes 16`).** In the offsets form, a load of a different constant from `.rodata` would not show.
  So this form also compares what each instruction takes from `.rodata`, and drops the nearest symbol's name, which moves with the layout:
  - a data object with a size, by its bytes;
  - a jump table, by its entries as offsets in the function;
  - a C string, with the build folder of A read as B's (the `__FILE__` paths);
  - a load, by the bytes of its operand size.
- **Pairing:** by the demangled name. One name, the `enable_if` overload of `fk::Ptr<ND::_2D, Uchar3>::crop`, is not demangled by objdump
  2.46, and pairs by its mangled name.
- **Reach:** the functions reached from a row's lambda by direct calls and jumps. Calls through the PLT and indirect calls are listed, not
  followed. For each function that differs, the script lists who refers to it: in `.text` (calls, jumps, address loads), and in data (an
  `R_X86_64_RELATIVE` relocation with its address).

## The functions that differ

| program | functions in `.text` (A = B) | offsets: identical / different | rodata: identical / different |
|---|---|---|---|
| `e19_batched_box_hip` | 112 | 109 / 3 | 108 / 4 |
| `e19_crop_sweep_hip` | 621 | 603 / 18 | 602 / 19 |

The names below are shortened; the full names are in `e19-host-code/*/<program>.summary.txt`.

1. **`bool fk::executeLinearFilter<fk::LinearFilterDPPDetails<float, 16, 8, 7, 7>, fk::Tuple<...PerThreadRead<_2D, Uchar3>, RGB2Gray...,
   ConstantFilterRead<float>>, Mul<float...>, Add<float...>, ...Div, Sub, Div, PerThreadWrite<_3D, float>...>`**, one function in each
   program: fkl_loop's linear filter call. It has new checks (#433's `valid()` without the sizes, and the output check).
2. **fkl_loop's lambda:**
   - in `e19_batched_box_hip`, `std::_Function_handler<void (), launch()::{lambda()#1}>::_M_invoke`;
   - in `e19_crop_sweep_hip`, `e19amd::runCrops<N>()::{lambda()#1}::operator()() const` for the 16 N.

   It has 136 instructions in A and 134 in B (207 and 205 for N = 1): the details lose `CROP_W` and `CROP_H`.
3. **`void fk::__device_stub__linearFilterDPPKernel<..., LinearFilterDPPDetails<float, 16, 8, 7, 7>, ...>`**, the host stub of fkl_loop's
   kernel. The details argument is 16 bytes instead of 24, so it is passed in registers instead of on the stack.
   - No function calls it: `executeLinearFilter` launches through `hipLaunchKernel` itself.
   - Its only reference is the kernel handle in `.data.rel.ro`.
4. **`fklb::writeSystemInfoHeader(...)`**, in the rodata form only: the FKL commit string, 24478f0... in A and b10fc66... in B. It runs when
   the CSV file is opened, before the clock warm-up and the timing.

Everything else is identical in both forms. This includes:
- `launch()` and the 16 `runCrops<N>()`, which hold the timing loop, and in `e19_batched_box_hip` the inlined `timePreparedLaunch`;
- `fklb::timeIt` and the 16 `timePreparedLaunch`;
- the lambdas of the fkl_batch and rpp rows;
- every `executeMorphology` instantiation;
- `crop2D` and its checks.

## What each timed call reaches

The reach is the same in both forms, and the lists of A and B are equal (`e19-host-code/*/<program>.reach-<row>-<A|B>.txt`):

| program | row | functions reached | own / shared | changed | indirect calls and jumps |
|---|---|---|---|---|---|
| `e19_batched_box_hip` | fkl_batch | 19 | 3 / 16 | **0** | 1 (a jump table in `freePtr`) |
| | rpp | 7 | 1 / 6 | **0** | 0 |
| | harness (`timeIt`) | 5 | 1 / 4 | **0** | 2 |
| | fkl_loop | 21 | 5 / 16 | 2 (items 1 and 2) | 2 |
| `e19_crop_sweep_hip` | fkl_batch | 79 | 63 / 16 | **0** | 1 (the same) |
| | rpp | 38 | 32 / 6 | **0** | 0 |
| | harness (`timeIt`, 16 `timePreparedLaunch`) | 21 | 17 / 4 | **0** | 34 |
| | fkl_loop | 52 | 36 / 16 | 17 (items 1 and 2) | 2 |

**The fkl_batch row's own functions:**
- in `e19_batched_box_hip`: its `_M_invoke`, `executeMorphology<..., BatchRead<..., 30, ...>, ...>` and `std::array<Ptr2D<Uchar3>, 30>::~array`;
- in the sweep, per N: its `_M_invoke` and `operator()`, `executeMorphology<...>` and the array's destructor (15 of the 16).

**The 16 shared functions** are `fk::Ptr<ND::_2D, Uchar3>::crop` (both overloads) and the `fk::ptr_crop` checks, `freePtr`, `fk::gpuAssert`,
the `std::string` helpers, `std::to_string` and `__clang_call_terminate`.

**PLT calls on the fkl_batch path:** `hipLaunchKernel`, `__hipPushCallConfiguration`, `__hipPopCallConfiguration`, `hipGetLastError`,
`hipGetErrorString`, `hipFree` and `hipHostFree`, and the C++ runtime (exceptions, strings).

**Can the comparison isolate the fkl_batch row?** Yes.
- Its own code is in functions of its own: no function of the fkl_batch path is also fkl_loop's changed code.
- The functions it shares with the other rows and the harness are all identical, so sharing does not stand in the way of the conclusion. They
  are the crop checks, `freePtr`, the string helpers, `timeIt`, and `timePreparedLaunch` (inside `launch()` in `e19_batched_box_hip`).

**Indirect calls.** A changed function could be reached through an indirect call only by its address. In both programs:
- `executeLinearFilter` is only called directly, by fkl_loop's lambda.
- fkl_loop's lambda is reached only through fkl_loop's own `std::function`: `launch()` (and each `runCrops<N>()`) takes its address to build it.
- The stub's only reference is the kernel handle.

The indirect calls inside a timed call are the `std::function` call, which goes to the row's own lambda, and the jump table in `freePtr`.
The entries of that jump table are equal in the rodata form.

## Not covered

- **The shared libraries.** The HIP runtime, RPP and the C and C++ runtimes are not part of the executables. A and B use the same installed
  files.
- **The GPU side.** It is `devcode-compare.txt`: the fkl_batch row's `MorphologyDPP` kernel is identical, and fkl_loop's
  `linearFilterDPPKernel` differs.
- **Run times.** Nothing was run, so this does not measure time; it shows that the host code executed inside the fkl_batch and rpp timed
  calls is the same in both builds.

## Files

- `e19-host-code/offsets/` and `e19-host-code/rodata/`, one set per program and form:
  - `<program>.summary.txt`: the counts, the functions that differ, the reach of each row and who refers to each changed function;
  - `<program>.functions.tsv`: every function with its number of instructions and the hashes in A and B;
  - `<program>.diffs.txt`: the normalised difference of each function that differs;
  - `<program>.reach-<row>-<A|B>.txt`: the functions, PLT targets and indirect calls that each row reaches.
- `step0/host_compare.py`: the script.
- `step0/run_checks.sh`: the commands, with the row and harness patterns.
