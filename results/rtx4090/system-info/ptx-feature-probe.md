ptxas: Cuda compilation tools, release 13.4, V13.4.92 Build cuda_13.4.r13.4/compiler.38855100_0  (/home/oscar/tfkl-bench/toolchains/cuda-13.4.3/bin/ptxas), PTX ISA 9.4. ACCEPT = the assembler builds it for that target.

| feature | sm_89 | sm_120 | sm_120a | sm_100a | sm_90a |
|---|---|---|---|---|---|
| mma.sync m16n8k16 bf16 (HMMA) | ACCEPT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| mma.sync m16n8k32 e4m3 fp8 (QMMA) | ACCEPT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| mma.sync block-scaled mxf4 e2m1 (fp4) | REJECT | REJECT | ACCEPT | REJECT | REJECT |
| ldmatrix (LDSM) | ACCEPT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| cp.async 16 B (LDGSTS) | ACCEPT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| redux.sync | ACCEPT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| griddepcontrol (programmatic dependent launch) | REJECT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| cluster special registers + barrier.cluster | REJECT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| mapa (distributed shared memory address) | REJECT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| cp.async.bulk (non-tensor bulk copy) | REJECT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| cp.async.bulk.tensor 2d (TMA) | REJECT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| setmaxnreg | REJECT | REJECT | ACCEPT | ACCEPT | ACCEPT |
| elect.sync | REJECT | ACCEPT | ACCEPT | ACCEPT | ACCEPT |
| wgmma.mma_async m64n8k16 bf16 (warpgroup MMA) | REJECT | REJECT | REJECT | REJECT | ACCEPT |
| tcgen05.alloc (Tensor Memory) | REJECT | REJECT | REJECT | ACCEPT | REJECT |

First assembler error of each rejected probe:

- mma.sync block-scaled mxf4 e2m1 (fp4) on sm_89: `line 9; error   : Instruction 'mma with block scale' not supported on .target 'sm_89'`
- mma.sync block-scaled mxf4 e2m1 (fp4) on sm_120: `line 9; error   : Instruction 'mma with block scale' not supported on .target 'sm_120'`
- mma.sync block-scaled mxf4 e2m1 (fp4) on sm_100a: `line 9; error   : Instruction 'mma with block scale' not supported on .target 'sm_100a'`
- mma.sync block-scaled mxf4 e2m1 (fp4) on sm_90a: `line 9; error   : Instruction 'mma with block scale' not supported on .target 'sm_90a'`
- griddepcontrol (programmatic dependent launch) on sm_89: `line 8; error   : Modifier '.launch_dependents' requires .target sm_90 or higher`
- cluster special registers + barrier.cluster on sm_89: `line 9; error   : Feature '%cluster_ctaid' requires .target sm_90 or higher`
- mapa (distributed shared memory address) on sm_89: `line 10; error   : Feature 'mapa' requires .target sm_90 or higher`
- cp.async.bulk (non-tensor bulk copy) on sm_89: `line 10; error   : Feature 'cp.async.bulk' requires .target sm_90 or higher`
- cp.async.bulk.tensor 2d (TMA) on sm_89: `line 10; error   : Feature 'cp.async.bulk.tensor' requires .target sm_90 or higher`
- setmaxnreg on sm_89: `line 8; error   : Instruction 'setmaxnreg.inc' not supported on .target 'sm_89'`
- setmaxnreg on sm_120: `line 8; error   : Instruction 'setmaxnreg.inc' not supported on .target 'sm_120'`
- elect.sync on sm_89: `line 9; error   : Feature 'elect' requires .target sm_90 or higher`
- wgmma.mma_async m64n8k16 bf16 (warpgroup MMA) on sm_89: `line 10; error   : Instruction 'wgmma.fence' not supported on .target 'sm_89'`
- wgmma.mma_async m64n8k16 bf16 (warpgroup MMA) on sm_120: `line 10; error   : Instruction 'wgmma.fence' not supported on .target 'sm_120'`
- wgmma.mma_async m64n8k16 bf16 (warpgroup MMA) on sm_120a: `line 10; error   : Instruction 'wgmma.fence' not supported on .target 'sm_120a'`
- wgmma.mma_async m64n8k16 bf16 (warpgroup MMA) on sm_100a: `line 10; error   : Instruction 'wgmma.fence' not supported on .target 'sm_100a'`
- tcgen05.alloc (Tensor Memory) on sm_89: `line 10; error   : Instruction 'tcgen05.alloc' not supported on .target 'sm_89'`
- tcgen05.alloc (Tensor Memory) on sm_120: `line 10; error   : Instruction 'tcgen05.alloc' not supported on .target 'sm_120'`
- tcgen05.alloc (Tensor Memory) on sm_120a: `line 10; error   : Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'`
- tcgen05.alloc (Tensor Memory) on sm_90a: `line 10; error   : Instruction 'tcgen05.alloc' not supported on .target 'sm_90a'`
