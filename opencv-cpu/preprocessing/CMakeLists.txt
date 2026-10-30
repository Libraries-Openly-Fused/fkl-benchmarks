# E5: FKL CPU back-end (ParArch::CPU and ParArch::CPU_MULTITHREAD) against OpenCV on the CPU.
# bench_preproc_a.cpp: pipeline (a), batched DNN pre-processing (the chain of cvGPUSpeedup test_batchresize_x_split3D).
# Needs the thread-pool submodule of external/fkl (git submodule update --init --recursive) and OpenCV core and imgproc.
# Batch sizes: E5_BATCHES (compile time, comma separated); run-time settings are environment variables, see the source.
if(FKLB_HAVE_CPU_MULTITHREAD)
    fklb_add_benchmark(e5_preprocessing_a
        BACKENDS cpp
        LIBS fkl_mt opencv_cpu
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/bench_preproc_a.cpp"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
        DEFINITIONS "E5_BATCHES=10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300")
    # Pipeline (b): the NV12 video pre-processing, the FKL code is fkl-self/pipelines/nv12_pipeline.h
    fklb_add_benchmark(e5_preprocessing_b
        BACKENDS cpp
        LIBS fkl_mt opencv_cpu
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/bench_preproc_b.cpp"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    message(STATUS "fkl-benchmarks: e5_preprocessing_a and _b skipped (external/fkl/thread-pool is not checked out)")
endif()
