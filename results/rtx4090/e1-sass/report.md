# E1-SASS: hand-written kernels against FKL, vertical-fusion chain

binary `build/rtx4090-e1/bin/e1_vf_chain_cu`, arch sm_89, registers renamed by first appearance.

```
nvcc: /home/oscar/tfkl-bench/toolchains/cuda-13.4.3/bin/nvcc  (Build cuda_13.4.r13.4/compiler.38855100_0)
cuobjdump: /home/oscar/tfkl-bench/toolchains/cuda-13.4.3/bin/cuobjdump  (Build cuda_13.4.r13.4/compiler.38855100_0)
GPU, driver: NVIDIA GeForce RTX 5050, 610.43.02 | NVIDIA GeForce RTX 4090, 610.43.02
```

## 2 operations (1 Mul+Add pairs per element)

| kernel | instructions | FFMA | FMUL | FADD | LDG | STG | LDL | STL | const-bank operands | IMAD | ISETP | BRA etc. | barriers | regs | stack | local | cmem[0] B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hand_params | 40 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 18 | 9 | 2 | 3 | 0 | 12 | 0 | 0 | 400 |
| hand_imm | 40 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 13 | 9 | 2 | 3 | 0 | 12 | 0 | 0 | 384 |
| hand_vec | 56 | 4 | 4 | 0 | 1 | 1 | 0 | 0 | 23 | 10 | 2 | 3 | 0 | 14 | 0 | 0 | 400 |
| fkl | 40 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 15 | 5 | 2 | 3 | 0 | 12 | 0 | 0 | 424 |
| fkl_tf | 48 | 4 | 4 | 0 | 1 | 1 | 0 | 0 | 21 | 6 | 2 | 3 | 0 | 14 | 0 | 0 | 432 |

- `hand_params` vs `fkl`, the 5 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **identical except the `.CONSTANT` modifier of the loads** (the hand kernel reads its input through `const __restrict__`, the read-only data path; FKL emits a plain load). Other instructions (thread index, bounds, address arithmetic): 35 in `hand_params`, 35 in `fkl`.
- `hand_params` vs `fkl`: differ: 40 vs 40 lines, 44 changed lines in the diff (`ops2/hand_params-vs-fkl.diff`)
- `hand_vec` vs `fkl_tf`, the 14 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **identical except the `.CONSTANT` modifier of the loads** (the hand kernel reads its input through `const __restrict__`, the read-only data path; FKL emits a plain load). Other instructions (thread index, bounds, address arithmetic): 42 in `hand_vec`, 34 in `fkl_tf`.
- `hand_vec` vs `fkl_tf`: differ: 56 vs 48 lines, 74 changed lines in the diff (`ops2/hand_vec-vs-fkl_tf.diff`)

## 102 operations (51 Mul+Add pairs per element)

| kernel | instructions | FFMA | FMUL | FADD | LDG | STG | LDL | STL | const-bank operands | IMAD | ISETP | BRA etc. | barriers | regs | stack | local | cmem[0] B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hand_params | 96 | 51 | 1 | 0 | 1 | 1 | 0 | 0 | 68 | 9 | 2 | 3 | 0 | 10 | 0 | 0 | 400 |
| hand_imm | 96 | 51 | 0 | 0 | 1 | 1 | 0 | 0 | 13 | 9 | 2 | 3 | 0 | 10 | 0 | 0 | 384 |
| hand_vec | 256 | 204 | 4 | 0 | 1 | 1 | 0 | 0 | 223 | 10 | 2 | 3 | 0 | 14 | 0 | 0 | 400 |
| fkl | 88 | 51 | 1 | 0 | 1 | 1 | 0 | 0 | 66 | 5 | 2 | 3 | 0 | 10 | 0 | 0 | 432 |
| fkl_tf | 256 | 204 | 4 | 0 | 1 | 1 | 0 | 0 | 222 | 6 | 2 | 3 | 0 | 14 | 0 | 0 | 440 |
| fkl_tf_tail | 336 | 255 | 5 | 0 | 2 | 2 | 0 | 0 | 285 | 9 | 5 | 7 | 0 | 14 | 0 | 0 | 440 |

- `hand_params` vs `fkl`, the 55 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **identical except the `.CONSTANT` modifier of the loads** (the hand kernel reads its input through `const __restrict__`, the read-only data path; FKL emits a plain load). Other instructions (thread index, bounds, address arithmetic): 41 in `hand_params`, 33 in `fkl`.
- `hand_params` vs `fkl`: differ: 96 vs 88 lines, 144 changed lines in the diff (`ops102/hand_params-vs-fkl.diff`)
- `hand_vec` vs `fkl_tf`, the 214 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **the same instructions** (ignoring the `.CONSTANT` modifier of the loads) **in a different order** (instruction scheduling of the conversions). Other instructions (thread index, bounds, address arithmetic): 42 in `hand_vec`, 42 in `fkl_tf`.
- `hand_vec` vs `fkl_tf`: differ: 256 vs 256 lines, 500 changed lines in the diff (`ops102/hand_vec-vs-fkl_tf.diff`)

## 1002 operations (501 Mul+Add pairs per element)

| kernel | instructions | FFMA | FMUL | FADD | LDG | STG | LDL | STL | const-bank operands | IMAD | ISETP | BRA etc. | barriers | regs | stack | local | cmem[0] B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hand_params | 544 | 501 | 1 | 0 | 1 | 1 | 0 | 0 | 518 | 9 | 2 | 3 | 0 | 10 | 0 | 0 | 400 |
| hand_imm | 544 | 501 | 0 | 0 | 1 | 1 | 0 | 0 | 13 | 9 | 2 | 3 | 0 | 10 | 0 | 0 | 384 |
| hand_vec | 2056 | 2004 | 4 | 0 | 1 | 1 | 0 | 0 | 2023 | 10 | 2 | 3 | 0 | 14 | 0 | 0 | 400 |
| fkl | 536 | 501 | 1 | 0 | 1 | 1 | 0 | 0 | 516 | 5 | 2 | 3 | 0 | 10 | 0 | 0 | 432 |
| fkl_tf | 2056 | 2004 | 4 | 0 | 1 | 1 | 0 | 0 | 2022 | 6 | 2 | 3 | 0 | 14 | 0 | 0 | 440 |
| fkl_tf_tail | 2584 | 2505 | 5 | 0 | 2 | 2 | 0 | 0 | 2535 | 9 | 5 | 8 | 0 | 14 | 0 | 0 | 440 |

- `hand_params` vs `fkl`, the 505 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **identical except the `.CONSTANT` modifier of the loads** (the hand kernel reads its input through `const __restrict__`, the read-only data path; FKL emits a plain load). Other instructions (thread index, bounds, address arithmetic): 39 in `hand_params`, 31 in `fkl`.
- `hand_params` vs `fkl`: differ: 544 vs 536 lines, 1060 changed lines in the diff (`ops1002/hand_params-vs-fkl.diff`)
- `hand_vec` vs `fkl_tf`, the 2014 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **the same instructions** (ignoring the `.CONSTANT` modifier of the loads) **in a different order** (instruction scheduling of the conversions). Other instructions (thread index, bounds, address arithmetic): 42 in `hand_vec`, 42 in `fkl_tf`.
- `hand_vec` vs `fkl_tf`: differ: 2056 vs 2056 lines, 4080 changed lines in the diff (`ops1002/hand_vec-vs-fkl_tf.diff`)

## 5002 operations (2501 Mul+Add pairs per element)

| kernel | instructions | FFMA | FMUL | FADD | LDG | STG | LDL | STL | const-bank operands | IMAD | ISETP | BRA etc. | barriers | regs | stack | local | cmem[0] B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hand_params | 2544 | 2501 | 1 | 0 | 1 | 1 | 0 | 0 | 2518 | 9 | 2 | 3 | 0 | 10 | 0 | 0 | 400 |
| hand_imm | 2544 | 2501 | 0 | 0 | 1 | 1 | 0 | 0 | 13 | 9 | 2 | 3 | 0 | 10 | 0 | 0 | 384 |
| hand_vec | 10056 | 10004 | 4 | 0 | 1 | 1 | 0 | 0 | 10023 | 10 | 2 | 3 | 0 | 18 | 0 | 0 | 400 |
| fkl | 2536 | 2501 | 1 | 0 | 1 | 1 | 0 | 0 | 2516 | 5 | 2 | 3 | 0 | 10 | 0 | 0 | 432 |
| fkl_tf | 10056 | 10004 | 4 | 0 | 1 | 1 | 0 | 0 | 10022 | 6 | 2 | 3 | 0 | 14 | 0 | 0 | 440 |
| fkl_tf_tail | 12584 | 12505 | 5 | 0 | 2 | 2 | 0 | 0 | 12535 | 9 | 5 | 8 | 0 | 14 | 0 | 0 | 440 |

- `hand_params` vs `fkl`, the 2505 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **identical except the `.CONSTANT` modifier of the loads** (the hand kernel reads its input through `const __restrict__`, the read-only data path; FKL emits a plain load). Other instructions (thread index, bounds, address arithmetic): 39 in `hand_params`, 31 in `fkl`.
- `hand_params` vs `fkl`: differ: 2544 vs 2536 lines, 5044 changed lines in the diff
- `hand_vec` vs `fkl_tf`, the 10014 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **the same instructions** (ignoring the `.CONSTANT` modifier of the loads) **in a different order** (instruction scheduling of the conversions). Other instructions (thread index, bounds, address arithmetic): 42 in `hand_vec`, 42 in `fkl_tf`.
- `hand_vec` vs `fkl_tf`: differ: 10056 vs 10056 lines, 20082 changed lines in the diff

## 10002 operations (5001 Mul+Add pairs per element)

| kernel | instructions | FFMA | FMUL | FADD | LDG | STG | LDL | STL | const-bank operands | IMAD | ISETP | BRA etc. | barriers | regs | stack | local | cmem[0] B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hand_params | 5040 | 5001 | 1 | 0 | 1 | 1 | 0 | 0 | 5018 | 9 | 2 | 3 | 0 | 12 | 0 | 0 | 400 |
| hand_imm | 5040 | 5001 | 0 | 0 | 1 | 1 | 0 | 0 | 13 | 9 | 2 | 3 | 0 | 12 | 0 | 0 | 384 |
| hand_vec | 80 | 20 | 4 | 0 | 1 | 1 | 0 | 0 | 39 | 9 | 3 | 4 | 0 | 16 | 0 | 0 | 400 |
| fkl | 5040 | 5001 | 1 | 0 | 1 | 1 | 0 | 0 | 5016 | 5 | 2 | 3 | 0 | 10 | 0 | 0 | 432 |
| fkl_tf | 20056 | 20004 | 4 | 0 | 1 | 1 | 0 | 0 | 20022 | 6 | 2 | 3 | 0 | 14 | 0 | 0 | 440 |
| fkl_tf_tail | 25088 | 25005 | 5 | 0 | 2 | 2 | 0 | 0 | 25035 | 9 | 5 | 8 | 0 | 14 | 0 | 0 | 440 |

- `hand_params` vs `fkl`, the 5005 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **identical except the `.CONSTANT` modifier of the loads** (the hand kernel reads its input through `const __restrict__`, the read-only data path; FKL emits a plain load). Other instructions (thread index, bounds, address arithmetic): 35 in `hand_params`, 35 in `fkl`.
- `hand_params` vs `fkl`: differ: 5040 vs 5040 lines, 10042 changed lines in the diff
- `hand_vec` vs `fkl_tf`, the 30 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **DIFFERENT**: only in `hand_vec`: {}; only in `fkl_tf`: {'FFMA R, R, R, c[]': 19984} (`hand_vec` has 4 control instructions against 3: nvcc did not fully unroll the hand-written loop despite `#pragma unroll`, it compiled a loop with a partially unrolled body, while FKL's kernel is straight-line code). Other instructions (thread index, bounds, address arithmetic): 50 in `hand_vec`, 42 in `fkl_tf`.
- `hand_vec` vs `fkl_tf`: differ: 80 vs 20056 lines, 20100 changed lines in the diff

## 19902 operations (9951 Mul+Add pairs per element)

| kernel | instructions | FFMA | FMUL | FADD | LDG | STG | LDL | STL | const-bank operands | IMAD | ISETP | BRA etc. | barriers | regs | stack | local | cmem[0] B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hand_params | 9992 | 9951 | 1 | 0 | 1 | 1 | 0 | 0 | 9968 | 9 | 2 | 3 | 0 | 12 | 0 | 0 | 400 |
| hand_imm | 9992 | 9951 | 0 | 0 | 1 | 1 | 0 | 0 | 13 | 9 | 2 | 3 | 0 | 12 | 0 | 0 | 384 |
| hand_vec | 88 | 28 | 4 | 0 | 1 | 1 | 0 | 0 | 47 | 9 | 3 | 4 | 0 | 16 | 0 | 0 | 400 |
| fkl | 9992 | 9951 | 1 | 0 | 1 | 1 | 0 | 0 | 9966 | 5 | 2 | 3 | 0 | 10 | 0 | 0 | 432 |
| fkl_tf | 39856 | 39804 | 4 | 0 | 1 | 1 | 0 | 0 | 39822 | 6 | 2 | 3 | 0 | 14 | 0 | 0 | 440 |
| fkl_tf_tail | 49840 | 49755 | 5 | 0 | 2 | 2 | 0 | 0 | 49785 | 9 | 5 | 8 | 0 | 14 | 0 | 0 | 440 |

- `hand_params` vs `fkl`, the 9955 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **identical except the `.CONSTANT` modifier of the loads** (the hand kernel reads its input through `const __restrict__`, the read-only data path; FKL emits a plain load). Other instructions (thread index, bounds, address arithmetic): 37 in `hand_params`, 37 in `fkl`.
- `hand_params` vs `fkl`: differ: 9992 vs 9992 lines, 19942 changed lines in the diff
- `hand_vec` vs `fkl_tf`, the 38 arithmetic and memory instructions (FFMA, FMUL, FADD, LDG, STG, I2FP...; register numbers and constant-bank offsets ignored): **DIFFERENT**: only in `hand_vec`: {}; only in `fkl_tf`: {'FFMA R, R, R, c[]': 39776} (`hand_vec` has 4 control instructions against 3: nvcc did not fully unroll the hand-written loop despite `#pragma unroll`, it compiled a loop with a partially unrolled body, while FKL's kernel is straight-line code). Other instructions (thread index, bounds, address arithmetic): 50 in `hand_vec`, 42 in `fkl_tf`.
- `hand_vec` vs `fkl_tf`: differ: 88 vs 39856 lines, 39908 changed lines in the diff

