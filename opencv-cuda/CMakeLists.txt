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
