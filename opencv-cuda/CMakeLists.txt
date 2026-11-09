# cvGPUSpeedup against OpenCV-CUDA. CUDA only: the baselines are OpenCV-CUDA calls.
# Every benchmark is built for the cu back-end only (BACKENDS cu); hip and cpp never see this code.

fklb_add_benchmark(benchmark_CPU_OpenCV_vs_cvGS
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_CPU_OpenCV_vs_cvGS.cu")

fklb_add_benchmark(benchmark_CPUandGPU_cvGS_vs_fk
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_CPUandGPU_cvGS_vs_fk.cu")

fklb_add_benchmark(benchmark_image_resolution_MAD_loop
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_image_resolution_MAD_loop.cu")

add_subdirectory(verticalfusion)

# E4a, batched DNN preprocessing: copied from cvGPUSpeedup tests/batchresize/ (ENABLE_BENCHMARK mode), with its
# batched-OpenCV baseline (test_batchresize_x_split3D_OCVBatch.csv)
fklb_add_benchmark(benchmark_batched_preprocessing
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/batched-preprocessing/test_batchresize_x_split3D.cu")

# E0: how far can the CUDA-Graphs baseline of the VF + HF sweep go? (capture + instantiate time of its graph)
fklb_add_benchmark(graph_instantiate_time
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/graphs/graph_instantiate_time.cu")

# E0 baseline fairness check of the CUDA-Graphs series of the VF + HF sweep: the same OpenCV-CUDA work as plain launches, as the experiment's child-node graph, as chained child nodes, and as a flat graph
# from one stream capture (results/rtx4090/e0-graph-structure)
fklb_add_benchmark(graph_structure_probe
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/graphs/graph_structure_probe.cu")

# E0: the time of the fused kernel (cvGPUSpeedup) of the VF + HF sweep in each loop of the experiment and on its own (results/rtx4090/e0-graph-structure/vf-fused-time)
fklb_add_benchmark(vf_fused_time_probe
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/graphs/vf_fused_time_probe.cu")

# E0 data-type experiment: copied from cvGPUSpeedup tests/batchread/ (benchmark mode, 26 type combinations)
fklb_add_benchmark(benchmark_data_type
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/data-type/test_batchread_x_write3D.cu")

# E6: which pixel-sampling convention does each bilinear resize follow (FKL, cv::cuda::resize, NPP)?
fklb_add_benchmark(resize_convention
    BACKENDS cu LIBS cvgs opencv npp
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/resize-convention/resize_convention.cu")

# Diagnostic of the data-size experiment: the OpenCV-CUDA side alone, by element count (N mod 4 = 0 against N mod 4 = 2). Run only in a paused window.
fklb_add_benchmark(opencv_parity_probe
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/parity/opencv_parity_probe.cu")

# E4a (batches of crops) and E4c (whole frames): the pre-processing chain of a DNN, FKL against OpenCV-CUDA and NPP; on the batches also FKL against the hand-written
# kernels of E1c and cvGPUSpeedup (hence cvgs) against OpenCV-CUDA. It needs common/resize_reference.h (fkl-benchmarks PR #2), so the target exists only when that
# header is present.
if(EXISTS "${CMAKE_SOURCE_DIR}/common/resize_reference.h")
    set(E4A_BATCHES "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300" CACHE STRING "E4a batch sizes (compile time)")
    string(REPLACE ";" "," E4A_BATCHES_LIST "${E4A_BATCHES}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/batched-preprocessing/e4_batches.h.in" "${CMAKE_CURRENT_BINARY_DIR}/e4_batches.h" @ONLY)
    fklb_add_benchmark(e4_preprocessing
        BACKENDS cu LIBS fkl cvgs opencv npp
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/batched-preprocessing/e4_preprocessing.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")
else()
    message(STATUS "fkl-benchmarks: e4_preprocessing skipped (common/resize_reference.h of fkl-benchmarks PR #2 is not present)")
endif()

# E18, the stencil after a resize: FKL's LinearFilterDPP in one kernel against OpenCV-CUDA and NPP. NPP's 2D filter (nppiFilterBorder) is in the
# filtering library, nppif, which the other NPP programs do not use, so only this target links it.
fklb_add_benchmark(e18_stencil
    BACKENDS cu LIBS fkl opencv npp
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/stencil/e18_stencil.cu")
if(TARGET e18_stencil_cu)
    target_link_libraries(e18_stencil_cu PRIVATE CUDA::nppif)
endif()

# The data-size experiment with every size a multiple of 4 (increment 282272 instead of 282270) and only the one-channel pairs: OpenCV-CUDA's one-channel kernels are
# vectorised (cv::transformSmart<4>) only when the element count is a multiple of 4, which the sizes of the paper's experiment alternate (results/rtx4090/e0-data-size-parity).
fklb_add_benchmark(benchmark_image_resolution_MAD_loop_mul4
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_image_resolution_MAD_loop.cu"
    DEFINITIONS FKLB_DATASIZE_INCREMENT=282272 FKLB_DATASIZE_ONE_CHANNEL_ONLY)
