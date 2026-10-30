# E1: hand-written fused CUDA kernels (scalar and vectorised) as the upper bound for FKL, with the SASS of both for the equivalence
# proof (scripts/e1_sass.py). CUDA only. For the SASS proof build with -DFKLB_LINEINFO=OFF (preset rtx4090-e1): the comparison needs
# the same flags on both sides, and the plan says no -lineinfo.

# Operation counts of the vertical-fusion chain (the x axis of the paper: 2, 102 ... 19902). Passed in a generated header: nvcc
# splits a comma separated -D value into several options.
set(E1_VF_OPS "2, 102, 1002, 5002, 10002, 19902" CACHE STRING "E1a: number of operations of each point, comma separated, even numbers")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/e1_vf_ops.h.in" "${CMAKE_CURRENT_BINARY_DIR}/e1_vf_ops.h" @ONLY)

fklb_add_benchmark(e1_vf_chain
    BACKENDS cu LIBS fkl
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/e1_vf_chain.cu"
    INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")

# E1b: the horizontal-fusion batch; batch sizes as in the paper's figure (1 .. 1191)
set(E1_HF_BATCHES "1, 51, 201, 601, 1191" CACHE STRING "E1b: batch sizes, comma separated")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/e1_hf_batches.h.in" "${CMAKE_CURRENT_BINARY_DIR}/e1_hf_batches.h" @ONLY)

fklb_add_benchmark(e1_hf_batch
    BACKENDS cu LIBS fkl
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/e1_hf_batch.cu"
    INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")

# E1c: the batched pre-processing pipeline of E4a (crop, resize, reorder, mul, sub, div, planar split); batch sizes within the paper's 10 .. 300
set(E1_PP_BATCHES "10, 50, 100, 200, 300" CACHE STRING "E1c: batch sizes, comma separated")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/e1_pp_batches.h.in" "${CMAKE_CURRENT_BINARY_DIR}/e1_pp_batches.h" @ONLY)

fklb_add_benchmark(e1_preproc
    BACKENDS cu LIBS fkl
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/e1_preproc.cu"
    INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")
