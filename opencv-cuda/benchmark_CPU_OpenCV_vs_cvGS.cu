/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// CPU time of OpenCV-CUDA against cvGPUSpeedup: the host time each side spends to enqueue the work of one frame (std::chrono::steady_clock
// around the calls, no synchronisation inside), 100 iterations per batch size after a warm-up sweep (opencv-cuda/testsCommon.cuh). BATCH
// crops of 60 x 120 pixels, crop i at (i, i) of a 3840 x 2160 CV_8UC3 frame, are converted to float, resized to 64 x 128, multiplied, swapped
// to BGR, subtracted and divided, and split into planes.
// Everything a side does for a frame is timed, building its batch included: the crop locations may change on every image, so OpenCV makes the
// ROI of every crop in the call, and cvGPUSpeedup makes its array of ROIs in the call and builds its operations on it; the ROIs are released
// before the interval ends. Each side reads its own frame (cv::cuda::GpuMat, filled with the same value); cvGPUSpeedup writes a continuous
// tensor (cv::cuda::createContinuous).

#include <opencv-cuda/testsCommon.cuh>
#include <cvGPUSpeedup.cuh>
#include <opencv2/cudaimgproc.hpp>

#include <common/main.h>
#include <common/sequence.h>

#ifdef ENABLE_BENCHMARK
constexpr char VARIABLE_DIMENSION[]{ "Batch size" };

constexpr size_t NUM_EXPERIMENTS = 16;

constexpr size_t FIRST_VALUE = 2;
constexpr size_t INCREMENT = 10;
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = fklb::arrayIndexSequence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

template <int CV_TYPE_I, int CV_TYPE_O, int BATCH>
bool test_cpu_batchresize_x_split3D(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, cv::cuda::Stream& cv_stream) {
    bool passed = true;

    struct Parameters {
        cv::Scalar init;
        cv::Scalar alpha;
        cv::Scalar val_sub;
        cv::Scalar val_div;
    };

    double alpha = 0.3;

    std::vector<Parameters> params = {
        {{2u}, {alpha}, {1.f}, {3.2f}},
        {{2u, 37u}, {alpha, alpha}, {1.f, 4.f}, {3.2f, 0.6f}},
        {{5u, 5u, 5u}, {alpha, alpha, alpha}, {1.f, 4.f, 3.2f}, {3.2f, 0.6f, 11.8f}},
        {{2u, 37u, 128u, 20u}, {alpha, alpha, alpha, alpha}, {1.f, 4.f, 3.2f, 0.5f}, {3.2f, 0.6f, 11.8f, 33.f}} };

    cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_O) - 1).init;
    cv::Scalar val_alpha = params.at(CV_MAT_CN(CV_TYPE_O) - 1).alpha;
    cv::Scalar val_sub = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_sub;
    cv::Scalar val_div = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_div;

    constexpr int CROP_WIDTH = 60;
    constexpr int CROP_HEIGHT = 120;

    // each side reads its own frame, filled with the same value
    cv::cuda::GpuMat d_input((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X, CV_TYPE_I, val_init);
    cv::cuda::GpuMat d_input_cvGS((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X, CV_TYPE_I, val_init);
    std::array<cv::Rect2d, BATCH> crops_2d;
    for (int crop_i = 0; crop_i < BATCH; crop_i++) {
        crops_2d[crop_i] =
            cv::Rect2d(cv::Point2d(crop_i, crop_i), cv::Point2d(crop_i + CROP_WIDTH, crop_i + CROP_HEIGHT));
    }

    cv::Size up(64, 128);
    cv::cuda::GpuMat d_up(up, CV_TYPE_O);
    cv::cuda::GpuMat d_temp(up, CV_TYPE_O);
    cv::cuda::GpuMat d_temp2(up, CV_TYPE_O);

    std::array<std::vector<cv::cuda::GpuMat>, BATCH> d_output_cv;
    std::array<std::vector<cv::Mat>, BATCH> h_cvResults;
    std::array<std::vector<cv::Mat>, BATCH> h_cvGSResults;

    // cvGPUSpeedup's output: one continuous tensor, one row per crop holding its planes one after the other
    cv::cuda::GpuMat d_tensor_output;
    cv::cuda::createContinuous(BATCH, up.width * up.height * CV_MAT_CN(CV_TYPE_O), CV_MAT_DEPTH(CV_TYPE_O), d_tensor_output);

    cv::Mat diff(up, CV_MAT_DEPTH(CV_TYPE_O));
    cv::Mat h_tensor_output(BATCH, up.width * up.height * CV_MAT_CN(CV_TYPE_O), CV_MAT_DEPTH(CV_TYPE_O));

    cv::cuda::GpuMat crop_32F(cv::Size(CROP_WIDTH, CROP_HEIGHT), CV_32FC3);
    for (int crop_i = 0; crop_i < BATCH; crop_i++) {
        for (int i = 0; i < CV_MAT_CN(CV_TYPE_I); i++) {
            d_output_cv.at(crop_i).emplace_back(up, CV_MAT_DEPTH(CV_TYPE_O));
            h_cvResults.at(crop_i).emplace_back(up, CV_MAT_DEPTH(CV_TYPE_O));
        }
    }

    constexpr bool correctDept = CV_MAT_DEPTH(CV_TYPE_O) == CV_32F;

    BenchmarkTemp<ITERS> cpuBenchmark =
        initCPUBenchmark<CV_TYPE_I, CV_TYPE_O, ITERS, BATCH, FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>(__func__, VARIABLE_DIMENSION);
    for (int i = 0; i < ITERS; i++) {
        // OpenCV, timed. The crop locations may change on every image, so the ROI of every crop is made here, in the call (and released at
        // the end of its iteration).
        startOCV_CPU(cpuBenchmark);
        for (int crop_i = 0; crop_i < BATCH; crop_i++) {
            const cv::cuda::GpuMat crop = d_input(crops_2d[crop_i]);
            crop.convertTo(crop_32F, CV_TYPE_O, 1, cv_stream);
            cv::cuda::resize(crop_32F, d_up, up, 0., 0., cv::INTER_LINEAR, cv_stream);
            cv::cuda::multiply(d_up, val_alpha, d_temp, 1.0, -1, cv_stream);
            if constexpr (CV_MAT_CN(CV_TYPE_I) == 3 && correctDept) {
                cv::cuda::cvtColor(d_temp, d_temp, cv::COLOR_RGB2BGR, 0, cv_stream);
            } else if constexpr (CV_MAT_CN(CV_TYPE_I) == 4 && correctDept) {
                cv::cuda::cvtColor(d_temp, d_temp, cv::COLOR_RGBA2BGRA, 0, cv_stream);
            }
            cv::cuda::subtract(d_temp, val_sub, d_temp2, cv::noArray(), -1, cv_stream);
            cv::cuda::divide(d_temp2, val_div, d_temp, 1.0, -1, cv_stream);
            cv::cuda::split(d_temp, d_output_cv[crop_i], cv_stream);
        }
        stopOCV_startcvGS_CPU(cpuBenchmark, i);
        // cvGPUSpeedup, timed. The crop locations may change on every image, so its array of ROIs is made here, in the call, and its
        // operations are built on it; the ROIs are released before the interval ends.
        {
            std::array<cv::cuda::GpuMat, BATCH> crops;
            for (int crop_i = 0; crop_i < BATCH; crop_i++) {
                crops[crop_i] = d_input_cvGS(crops_2d[crop_i]);
            }
            cvGS::executeOperations(cv_stream, cvGS::resize<CV_TYPE_I, cv::INTER_LINEAR, BATCH>(crops, up, BATCH),
                cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_TYPE_O>(), cvGS::multiply<CV_TYPE_O>(val_alpha),
                cvGS::subtract<CV_TYPE_O>(val_sub), cvGS::divide<CV_TYPE_O>(val_div),
                cvGS::split<CV_TYPE_O>(d_tensor_output, up));
        }
        stopcvGS_CPU(cpuBenchmark, i);
        if (warmup) break;
    }
    processExecution<CV_TYPE_I, CV_TYPE_O, BATCH, ITERS, batchValues.size(), batchValues>(cpuBenchmark.resF, __func__, cpuBenchmark.OCVelapsedTime, cpuBenchmark.cvGSelapsedTime, VARIABLE_DIMENSION);
    d_tensor_output.download(h_tensor_output, cv_stream);

    // Verify results
    for (int crop_i = 0; crop_i < BATCH; crop_i++) {
        for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
            d_output_cv[crop_i].at(i).download(h_cvResults[crop_i].at(i), cv_stream);
        }
    }

    cv_stream.waitForCompletion();

    for (int crop_i = 0; crop_i < BATCH; crop_i++) {
        cv::Mat row = h_tensor_output.row(crop_i);
        for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
            int planeStart = i * up.width * up.height;
            int planeEnd = ((i + 1) * up.width * up.height) - 1;
            cv::Mat plane = row.colRange(planeStart, planeEnd);
            h_cvGSResults[crop_i].push_back(cv::Mat(up.height, up.width, plane.type(), plane.data));
        }
    }

    for (int crop_i = 0; crop_i < BATCH; crop_i++) {
        for (int i = 0; i < CV_MAT_CN(CV_TYPE_O); i++) {
            cv::Mat cvRes = h_cvResults[crop_i].at(i);
            cv::Mat cvGSRes = h_cvGSResults[crop_i].at(i);
            diff = cv::abs(cvRes - cvGSRes);
            bool passedThisTime = checkResults<CV_MAT_DEPTH(CV_TYPE_O)>(diff.cols, diff.rows, diff);
            passed &= passedThisTime;
        }
    }

    if (!passed) {
        std::stringstream ss;
        ss << "test_batchresize_x_split3D<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
        std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
    }

    return passed;
}

template <int CV_TYPE_I, int CV_TYPE_O, size_t... Is>
bool test_cpu_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...> seq,
    cv::cuda::Stream cv_stream) {
    bool passed = true;
    int dummy[] = { (passed &= test_cpu_batchresize_x_split3D<CV_TYPE_I, CV_TYPE_O, batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y,
                                                                                                cv_stream),
                    0)... };
    return passed;
}

#endif // ENABLE_BENCHMARK

int launch() {
#ifdef ENABLE_BENCHMARK
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::unordered_map<std::string, bool> results;
    results["test_cpu_batchresize_x_split3D"] = true;
    std::make_index_sequence<batchValues.size()> iSeq{};

#define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT)                                                                              \
  results["test_cpu_batchresize_x_split3D"] &=                                                                             \
      test_cpu_batchresize_x_split3D<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, cv_stream);

    try {
        // Warming up for the benchmarks
        warmup = true;
        LAUNCH_TESTS(CV_8UC3, CV_32FC3)
        warmup = false;

        benchmarkNotes = "Host time to enqueue the work of one frame (steady_clock, no synchronisation inside). Each side reads its own frame. "
                         "Everything a side does for a frame is timed, building its batch included, because the crop locations may change "
                         "on every image: OpenCV makes the ROI of every crop in the call; cvGPUSpeedup makes its array of ROIs in the call "
                         "and builds its operations on it.";
        LAUNCH_TESTS(CV_8UC3, CV_32FC3)
    } catch (...) {
        // a failed run must not leave files that look complete
        for (auto&& [name, file] : currentFile) {
            if (file.is_open()) {
                file << "# INCOMPLETE: the program stopped with an error; do not use this file" << std::endl;
            }
        }
        throw;
    }

#undef LAUNCH_TESTS

    stopCPUBenchmark();

    int returnValue = 0;
    for (const auto& [key, passed] : results) {
        if (passed) {
            std::cout << key << " passed!!" << std::endl;
        } else {
            std::cout << key << " failed!!" << std::endl;
            returnValue = -1;
        }
    }

    return returnValue;
#else
    return 0;
#endif // ENABLE_BENCHMARK
}
