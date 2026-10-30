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

# E0 data-type experiment: copied from cvGPUSpeedup tests/batchread/ (benchmark mode, 26 type combinations)
fklb_add_benchmark(benchmark_data_type
    BACKENDS cu LIBS cvgs opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/data-type/test_batchread_x_write3D.cu")

# E6: which pixel-sampling convention does each bilinear resize follow (FKL, cv::cuda::resize, NPP)?
fklb_add_benchmark(resize_convention
    BACKENDS cu LIBS cvgs opencv npp
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/resize-convention/resize_convention.cu")
