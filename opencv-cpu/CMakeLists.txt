# FKL CPU back-end against OpenCV on the CPU (single- and multi-threaded, vs cv::resize/cvtColor... on the same pipelines).
# Each benchmark: fklb_add_benchmark(<name> BACKENDS cpp LIBS fkl opencv_cpu SOURCES|LAUNCHER_HEADER ...).
# Put the benchmarks of a subject in a subfolder (opencv-cpu/preprocessing/) with its own CMakeLists.txt.
foreach(SUBDIR preprocessing)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt")
        add_subdirectory(${SUBDIR})
    endif()
endforeach()
